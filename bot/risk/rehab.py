"""Phase 4g automatic rehab tier system.

Encodes *"when you're in a drawdown, trade smaller; when
you're in a deep drawdown, trade way smaller; earn your way back"* as a
computable tier (NORMAL / REHAB / DEEP_REHAB) that scales the risk caps
the ``RiskEngine`` applies to new entries. Persistence is a parallel of
``logs/halt.flag``: a single JSON file at ``logs/rehab.flag`` that
survives process restarts so the tier sticks across sessions.

Two tier triggers feed the decision (stricter wins):
  * **consecutive red days** — trailing count of negative-PnL sessions
    in the journal.
  * **cumulative drawdown** — worst peak-to-trough cumulative PnL over
    the configured lookback window, expressed as a multiple of
    ``risk.max_daily_loss_usd`` (the natural unit for the rule).

Hysteresis is provided by ``recovery_drawdown_recovered_fraction``: to
*downgrade* a saved tier we require the operator to have recovered that
fraction of the drawdown recorded at entry. Upgrades (to a stricter
tier) always apply immediately — a worsening drawdown should never be
gated by a prior entry threshold.

This module does not mutate any config object. ``apply_to_caps`` returns
a ``RehabAdjustedCaps`` snapshot that the RiskEngine consults per-entry,
so the base ``RiskConfig`` is preserved for post-hoc analysis and the
``suggest-caps`` advisory CLI.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

from bot.config import RehabConfig, RiskConfig, Settings, get_settings

if TYPE_CHECKING:
    from bot.persistence.journal import Journal, TradeRecord

_log = structlog.get_logger("bot.risk.rehab")

_DEFAULT_REHAB_FLAG_PATH = Path("logs") / "rehab.flag"
_STALE_FLAG_AGE_DAYS = 30


class RehabTier(StrEnum):
    """Tier ladder; ordering is lexical-by-severity for ``max()`` comparisons.

    The integer rank (``.rank``) backs strict-vs-weak comparisons so the
    engine can express "accept upgrades immediately, gate downgrades" in
    one line.
    """

    NORMAL = "NORMAL"
    REHAB = "REHAB"
    DEEP_REHAB = "DEEP_REHAB"

    @property
    def rank(self) -> int:
        """0=NORMAL, 1=REHAB, 2=DEEP_REHAB — used for strictness comparisons."""
        return {"NORMAL": 0, "REHAB": 1, "DEEP_REHAB": 2}[self.value]


@dataclass(frozen=True)
class RehabState:
    """In-memory snapshot of the active tier + entry context.

    ``drawdown_at_entry_usd`` is a *negative* number (cumulative PnL at
    the moment the tier was entered); recovery checks subtract the
    current drawdown from it to compute ``recovered_usd``. A freshly
    reset state (tier NORMAL) has no entry context, so the drawdown
    and consecutive-red fields are 0 on NORMAL rows — don't read them
    unless ``tier != NORMAL``.
    """

    tier: RehabTier
    trigger_reason: str
    entered_at: datetime
    drawdown_at_entry_usd: float
    consecutive_red_days_at_entry: int


@dataclass(frozen=True)
class RehabRecord:
    """JSON-serializable mirror of ``RehabState``; written to ``logs/rehab.flag``."""

    tier: RehabTier
    trigger_reason: str
    entered_at: datetime
    drawdown_at_entry_usd: float
    consecutive_red_days_at_entry: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (enum → str, datetime → ISO 8601)."""
        return {
            "tier": self.tier.value,
            "trigger_reason": self.trigger_reason,
            "entered_at": self.entered_at.isoformat(),
            "drawdown_at_entry_usd": self.drawdown_at_entry_usd,
            "consecutive_red_days_at_entry": self.consecutive_red_days_at_entry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RehabRecord:
        """Rehydrate from a JSON dict; raises ``ValueError`` on bad shape."""
        return cls(
            tier=RehabTier(str(data["tier"])),
            trigger_reason=str(data["trigger_reason"]),
            entered_at=datetime.fromisoformat(str(data["entered_at"])),
            drawdown_at_entry_usd=float(data["drawdown_at_entry_usd"]),
            consecutive_red_days_at_entry=int(data["consecutive_red_days_at_entry"]),
        )

    def to_state(self) -> RehabState:
        """Convert back to the in-memory state shape."""
        return RehabState(
            tier=self.tier,
            trigger_reason=self.trigger_reason,
            entered_at=self.entered_at,
            drawdown_at_entry_usd=self.drawdown_at_entry_usd,
            consecutive_red_days_at_entry=self.consecutive_red_days_at_entry,
        )


@dataclass(frozen=True)
class RehabAdjustedCaps:
    """Effective risk caps after the current tier's multipliers are applied.

    ``base_*`` fields carry the un-scaled config values so rejection logs
    can show both the base and the rehab-adjusted limit — the operator
    shouldn't have to reconstruct "why is my trade count gating at 3
    today?" from ambient knowledge of the tier.
    """

    tier: RehabTier
    trigger_reason: str | None
    max_loss_per_trade_usd: float
    max_daily_loss_usd: float
    max_trades_per_day: int
    base_max_loss_per_trade_usd: float
    base_max_daily_loss_usd: float
    base_max_trades_per_day: int


@dataclass(frozen=True)
class RehabTransition:
    """Returned by ``check_transitions`` when the tier changes between ticks.

    ``reason`` is a short kebab-case tag describing the transition cause
    (``consecutive_red_days``, ``cumulative_drawdown``, ``recovery``).
    """

    old_tier: RehabTier
    new_tier: RehabTier
    reason: str
    drawdown_usd: float
    consecutive_red_days: int


@dataclass(frozen=True)
class RehabStats:
    """Raw aggregates from the journal — drives tier computation + CLI display.

    ``daily_pnl`` is ordered oldest-first so ``[-1]`` is "yesterday" (or
    today, if the session is mid-flight). Empty journals produce all-zero
    stats, which naturally yields ``NORMAL``.
    """

    consecutive_red_days: int
    cumulative_drawdown_usd: float
    lookback_days: int
    daily_pnl: list[tuple[date_cls, float]] = field(default_factory=list)


# ---------- Pure helpers ---------- #


def aggregate_daily_pnl(
    trades: Iterable[TradeRecord],
    timezone: str = "America/New_York",
) -> dict[date_cls, float]:
    """Sum closed-trade PnL per NY-local session date; skips still-open trades.

    Only rows with a non-null ``pnl`` contribute — open positions have
    ``pnl=None`` and are ignored entirely (we don't speculate about their
    eventual exit). The session date is derived from ``closed_at`` in
    local time so a trade closed at 15:55 ET on Friday counts as Friday
    even if UTC has rolled over.
    """
    tz = ZoneInfo(timezone)
    totals: dict[date_cls, float] = {}
    for row in trades:
        if row.pnl is None or row.closed_at is None:
            continue
        day = row.closed_at.astimezone(tz).date()
        totals[day] = totals.get(day, 0.0) + float(row.pnl)
    return totals


def consecutive_red_days(daily_pnl: Sequence[tuple[date_cls, float]]) -> int:
    """Count trailing negative-PnL sessions; stops at the first non-negative day.

    ``daily_pnl`` is oldest-first. A zero-PnL day ends the streak (it's
    not a *red* day in the usage). Empty input yields 0.
    """
    streak = 0
    for _, pnl in reversed(daily_pnl):
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


def cumulative_drawdown_usd(daily_pnl: Sequence[tuple[date_cls, float]]) -> float:
    """Worst peak-to-trough cumulative PnL in the window; returns ≤ 0.

    Standard drawdown computation: walk the equity curve, track the
    running peak, record the largest (peak - current) delta. The return
    is *negative-or-zero* (negative = underwater by that many dollars)
    so downstream comparisons against rehab multiples line up with the
    sign conventions in ``risk.daily_loss_hit``.
    """
    if not daily_pnl:
        return 0.0
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for _, pnl in daily_pnl:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = equity - peak
        if drawdown < worst:
            worst = drawdown
    return worst


def compute_stats_from_journal_entries(
    daily_totals: dict[date_cls, float],
    today: date_cls,
    lookback_days: int,
) -> RehabStats:
    """Bucket the per-day totals into a ``lookback_days`` window ending at ``today``.

    ``today`` is *exclusive* of the current session — an in-progress day
    shouldn't be compared against the red-day thresholds since the book
    isn't closed yet. The earliest day included is
    ``today - lookback_days``; missing days contribute 0 to the equity
    walk (the drawdown computation treats them as no-ops, which matches
    weekend/holiday behaviour).
    """
    if lookback_days < 1:
        return RehabStats(
            consecutive_red_days=0,
            cumulative_drawdown_usd=0.0,
            lookback_days=lookback_days,
        )
    window_start = today - timedelta(days=lookback_days)
    ordered: list[tuple[date_cls, float]] = sorted(
        (day, pnl) for day, pnl in daily_totals.items() if window_start <= day < today
    )
    return RehabStats(
        consecutive_red_days=consecutive_red_days(ordered),
        cumulative_drawdown_usd=cumulative_drawdown_usd(ordered),
        lookback_days=lookback_days,
        daily_pnl=ordered,
    )


def classify_tier(
    stats: RehabStats,
    rehab_cfg: RehabConfig,
    max_daily_loss_usd: float,
) -> tuple[RehabTier, str]:
    """Return the strictest tier triggered by ``stats``, plus the kebab reason.

    Check DEEP_REHAB first (stricter wins), then REHAB, then NORMAL.
    Drawdown thresholds are positive-dollar multiples of
    ``max_daily_loss_usd`` — they're compared against
    ``abs(cumulative_drawdown_usd)`` so the sign doesn't confuse readers.
    ``trigger_reason`` is ``"baseline"`` for the no-trigger NORMAL case.
    """
    drawdown = abs(stats.cumulative_drawdown_usd)
    reds = stats.consecutive_red_days

    deep_dd_threshold = rehab_cfg.deep_rehab_drawdown_multiple_of_daily_loss * max_daily_loss_usd
    if reds >= rehab_cfg.deep_rehab_consecutive_red_days:
        return RehabTier.DEEP_REHAB, "consecutive_red_days"
    if drawdown >= deep_dd_threshold:
        return RehabTier.DEEP_REHAB, "cumulative_drawdown"

    shallow_dd_threshold = rehab_cfg.rehab_drawdown_multiple_of_daily_loss * max_daily_loss_usd
    if reds >= rehab_cfg.rehab_consecutive_red_days:
        return RehabTier.REHAB, "consecutive_red_days"
    if drawdown >= shallow_dd_threshold:
        return RehabTier.REHAB, "cumulative_drawdown"

    return RehabTier.NORMAL, "baseline"


def _apply_multipliers(
    tier: RehabTier,
    risk_cfg: RiskConfig,
    rehab_cfg: RehabConfig,
    trigger_reason: str | None,
) -> RehabAdjustedCaps:
    """Scale the three cap fields by the tier's multiplier; preserve base values."""
    base_trade_loss = risk_cfg.max_loss_per_trade_usd
    base_daily_loss = risk_cfg.max_daily_loss_usd
    base_trades = risk_cfg.max_trades_per_day

    if tier is RehabTier.DEEP_REHAB:
        loss_mult = rehab_cfg.deep_rehab_max_loss_multiplier
        daily_mult = rehab_cfg.deep_rehab_max_daily_loss_multiplier
        trade_cap = rehab_cfg.deep_rehab_max_trades_per_day
    elif tier is RehabTier.REHAB:
        loss_mult = rehab_cfg.rehab_max_loss_multiplier
        daily_mult = rehab_cfg.rehab_max_daily_loss_multiplier
        trade_cap = rehab_cfg.rehab_max_trades_per_day
    else:
        loss_mult = 1.0
        daily_mult = 1.0
        trade_cap = base_trades

    return RehabAdjustedCaps(
        tier=tier,
        trigger_reason=trigger_reason,
        max_loss_per_trade_usd=base_trade_loss * loss_mult,
        max_daily_loss_usd=base_daily_loss * daily_mult,
        max_trades_per_day=min(base_trades, trade_cap),
        base_max_loss_per_trade_usd=base_trade_loss,
        base_max_daily_loss_usd=base_daily_loss,
        base_max_trades_per_day=base_trades,
    )


# ---------- Flag-file I/O ---------- #


def read_rehab_flag(path: Path) -> RehabRecord | None:
    """Return the persisted rehab record, or None on missing/corrupt file."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return RehabRecord.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        _log.warning("rehab_flag.read_failed", path=str(path), error=str(exc))
        return None


def write_rehab_flag(path: Path, record: RehabRecord) -> None:
    """Persist a rehab record; always overwrites (unlike halt.flag)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    _log.info(
        "rehab_flag.written",
        path=str(path),
        tier=record.tier.value,
        reason=record.trigger_reason,
        drawdown=round(record.drawdown_at_entry_usd, 2),
    )


def delete_rehab_flag(path: Path) -> bool:
    """Delete the flag file if it exists; return True iff a file was removed."""
    if not path.exists():
        return False
    path.unlink()
    _log.info("rehab_flag.deleted", path=str(path))
    return True


# ---------- Engine ---------- #


class RehabEngine:
    """Tier tracker + cap-override facade.

    Holds the current ``RehabState`` in memory, persists changes to
    ``logs/rehab.flag``, and exposes a cheap sync ``apply_to_caps`` for
    the RiskEngine's hot path. Journal I/O only happens in the async
    ``recompute`` / ``check_transitions`` methods, which the orchestrator
    drives at session start and periodically thereafter.

    Not safe against fork; intended for a single event loop. The engine
    is intentionally stateless across restarts *except* for the flag
    file — on construction, ``load_state`` should be called to rehydrate.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        journal: Journal | None = None,
        flag_path: Path | None = None,
    ) -> None:
        """Wire config + journal; call ``load_state`` separately to rehydrate."""
        self._settings = settings or get_settings()
        self._journal = journal
        self._flag_path = flag_path or _DEFAULT_REHAB_FLAG_PATH
        self._state: RehabState = RehabState(
            tier=RehabTier.NORMAL,
            trigger_reason="baseline",
            entered_at=datetime.now(UTC),
            drawdown_at_entry_usd=0.0,
            consecutive_red_days_at_entry=0,
        )
        self._simulation_override: list[tuple[date_cls, float]] | None = None

    def set_simulation_override(self, daily_pnl: list[tuple[date_cls, float]] | None) -> None:
        """Inject a synthetic daily-PnL series instead of reading the journal.

        Used by the ``--simulate-red-days`` CLI hook (paper-only). When
        non-``None``, ``compute_stats`` uses this list verbatim and skips
        journal I/O; pass ``None`` to clear the override. Nothing about
        this persists to disk — all cleanup happens by dropping the
        reference on CLI exit.
        """
        self._simulation_override = daily_pnl

    @property
    def state(self) -> RehabState:
        """Expose the current in-memory state (read-only)."""
        return self._state

    @property
    def flag_path(self) -> Path:
        """Path to the persisted rehab flag (``logs/rehab.flag`` by default)."""
        return self._flag_path

    @property
    def enabled(self) -> bool:
        """Shortcut to ``settings.risk.rehab.enabled``."""
        return self._settings.risk.rehab.enabled

    def load_state(self) -> RehabState:
        """Read the flag file (if any) and adopt it into memory; clean stale flags.

        A flag older than ``_STALE_FLAG_AGE_DAYS`` is silently deleted —
        the operator has evidently stopped trading long enough that any
        prior slump is stale context. Missing/corrupt files yield the
        default NORMAL state.
        """
        record = read_rehab_flag(self._flag_path)
        if record is None:
            return self._state
        age = datetime.now(UTC) - record.entered_at.astimezone(UTC)
        if age > timedelta(days=_STALE_FLAG_AGE_DAYS):
            delete_rehab_flag(self._flag_path)
            _log.info(
                "rehab_flag.stale_cleaned",
                entered_at=record.entered_at.isoformat(),
                age_days=age.days,
            )
            return self._state
        self._state = record.to_state()
        _log.info(
            "rehab.state_loaded",
            tier=self._state.tier.value,
            trigger_reason=self._state.trigger_reason,
            drawdown_at_entry=round(self._state.drawdown_at_entry_usd, 2),
        )
        return self._state

    def save_state(self, state: RehabState) -> None:
        """Persist ``state`` to the flag file and update the in-memory copy.

        NORMAL states still write a flag — that way the operator (and
        the ``rehab-status`` CLI) can see "we were in REHAB yesterday,
        recovered today" by inspecting the file's ``entered_at``.
        """
        self._state = state
        record = RehabRecord(
            tier=state.tier,
            trigger_reason=state.trigger_reason,
            entered_at=state.entered_at,
            drawdown_at_entry_usd=state.drawdown_at_entry_usd,
            consecutive_red_days_at_entry=state.consecutive_red_days_at_entry,
        )
        write_rehab_flag(self._flag_path, record)

    def apply_to_caps(self, risk_cfg: RiskConfig | None = None) -> RehabAdjustedCaps:
        """Return scaled caps for the current tier — cheap, no I/O.

        When ``rehab.enabled`` is false, returns caps identical to the
        base config with ``tier=NORMAL`` and ``trigger_reason=None`` so
        log consumers can distinguish "rehab off" from "rehab on,
        nothing triggered".
        """
        cfg = risk_cfg or self._settings.risk
        if not self.enabled:
            return RehabAdjustedCaps(
                tier=RehabTier.NORMAL,
                trigger_reason=None,
                max_loss_per_trade_usd=cfg.max_loss_per_trade_usd,
                max_daily_loss_usd=cfg.max_daily_loss_usd,
                max_trades_per_day=cfg.max_trades_per_day,
                base_max_loss_per_trade_usd=cfg.max_loss_per_trade_usd,
                base_max_daily_loss_usd=cfg.max_daily_loss_usd,
                base_max_trades_per_day=cfg.max_trades_per_day,
            )
        return _apply_multipliers(
            self._state.tier,
            cfg,
            self._settings.risk.rehab,
            self._state.trigger_reason if self._state.tier is not RehabTier.NORMAL else None,
        )

    async def compute_stats(self, today: date_cls | None = None) -> RehabStats:
        """Aggregate the journal over the lookback window and return raw stats.

        With no journal wired (tests, synthetic runs) returns all-zero
        stats — the caller then naturally classifies as NORMAL, which is
        the safe default. If a simulation override is registered via
        ``set_simulation_override``, it bypasses the journal entirely.
        """
        lookback = self._settings.risk.rehab.rehab_lookback_days
        today = today or self._today()
        if self._simulation_override is not None:
            daily_totals = {day: pnl for day, pnl in self._simulation_override}
            return compute_stats_from_journal_entries(daily_totals, today, lookback)
        if self._journal is None:
            return RehabStats(
                consecutive_red_days=0,
                cumulative_drawdown_usd=0.0,
                lookback_days=lookback,
            )
        daily_totals = await self._collect_daily_totals(today, lookback)
        return compute_stats_from_journal_entries(daily_totals, today, lookback)

    async def recompute(self, today: date_cls | None = None) -> RehabTier:
        """Run the tier classifier on fresh journal stats; return the current tier.

        No state mutation — use ``check_transitions`` when you want the
        engine to persist + notify on changes. This method exists so the
        ``rehab-status`` CLI can show "what would the engine decide
        right now?" without side-effects.
        """
        stats = await self.compute_stats(today)
        tier, _reason = classify_tier(
            stats,
            self._settings.risk.rehab,
            self._settings.risk.max_daily_loss_usd,
        )
        return tier

    async def check_transitions(self, today: date_cls | None = None) -> RehabTransition | None:
        """Recompute and persist; return a transition if the tier changed.

        Hysteresis: a *downgrade* (saved is stricter than computed) only
        applies once the operator has recovered
        ``recovery_drawdown_recovered_fraction`` of the drawdown
        recorded at tier entry. Upgrades apply immediately. Returns
        ``None`` when the tier is unchanged.

        When ``rehab.enabled`` is false this is a no-op that returns
        ``None`` and leaves the state untouched.
        """
        if not self.enabled:
            return None
        today = today or self._today()
        stats = await self.compute_stats(today)
        computed_tier, computed_reason = classify_tier(
            stats,
            self._settings.risk.rehab,
            self._settings.risk.max_daily_loss_usd,
        )

        prior = self._state
        if computed_tier.rank > prior.tier.rank:
            new_state = self._build_entry_state(computed_tier, computed_reason, stats)
            self.save_state(new_state)
            _log.warning(
                "rehab.tier_upgraded",
                old=prior.tier.value,
                new=computed_tier.value,
                reason=computed_reason,
                drawdown=round(stats.cumulative_drawdown_usd, 2),
                consecutive_red_days=stats.consecutive_red_days,
            )
            return RehabTransition(
                old_tier=prior.tier,
                new_tier=computed_tier,
                reason=computed_reason,
                drawdown_usd=stats.cumulative_drawdown_usd,
                consecutive_red_days=stats.consecutive_red_days,
            )

        if computed_tier.rank < prior.tier.rank:
            if self._recovery_met(prior, stats):
                new_state = self._build_entry_state(computed_tier, "recovery", stats)
                self.save_state(new_state)
                _log.info(
                    "rehab.tier_downgraded",
                    old=prior.tier.value,
                    new=computed_tier.value,
                    reason="recovery",
                    drawdown=round(stats.cumulative_drawdown_usd, 2),
                    drawdown_at_entry=round(prior.drawdown_at_entry_usd, 2),
                )
                return RehabTransition(
                    old_tier=prior.tier,
                    new_tier=computed_tier,
                    reason="recovery",
                    drawdown_usd=stats.cumulative_drawdown_usd,
                    consecutive_red_days=stats.consecutive_red_days,
                )
            _log.info(
                "rehab.downgrade_held",
                current=prior.tier.value,
                computed=computed_tier.value,
                drawdown=round(stats.cumulative_drawdown_usd, 2),
                drawdown_at_entry=round(prior.drawdown_at_entry_usd, 2),
                recovery_fraction=self._settings.risk.rehab.recovery_drawdown_recovered_fraction,
            )
        return None

    def _recovery_met(self, prior: RehabState, stats: RehabStats) -> bool:
        """True when the operator has earned back the configured recovery fraction.

        Compares |drawdown_at_entry| vs |current_drawdown| — if the delta
        is at least ``recovery_fraction * |drawdown_at_entry|`` the
        operator has recovered enough. An entry drawdown of 0 (shouldn't
        happen on REHAB entry, but guard anyway) short-circuits to True
        so we never get stuck in a tier that was entered on a
        consecutive-red trigger alone.
        """
        entry_dd = abs(prior.drawdown_at_entry_usd)
        if entry_dd <= 0.0:
            return True
        current_dd = abs(stats.cumulative_drawdown_usd)
        recovered = entry_dd - current_dd
        threshold = entry_dd * self._settings.risk.rehab.recovery_drawdown_recovered_fraction
        return recovered >= threshold

    def _build_entry_state(
        self,
        tier: RehabTier,
        reason: str,
        stats: RehabStats,
    ) -> RehabState:
        """Freeze the current stats into a new ``RehabState`` for persistence."""
        return RehabState(
            tier=tier,
            trigger_reason=reason,
            entered_at=datetime.now(UTC),
            drawdown_at_entry_usd=stats.cumulative_drawdown_usd,
            consecutive_red_days_at_entry=stats.consecutive_red_days,
        )

    async def _collect_daily_totals(
        self, today: date_cls, lookback_days: int
    ) -> dict[date_cls, float]:
        """Walk the journal day-by-day for the window; sum PnL per session date.

        Uses ``Journal.trades_for_session`` once per day so the journal's
        existing timezone handling applies. Lookback is inclusive of
        ``today - lookback_days`` through ``today - 1`` (today is
        excluded — the session isn't closed).
        """
        if self._journal is None:
            return {}
        timezone = self._settings.session.timezone
        totals: dict[date_cls, float] = {}
        for offset in range(1, lookback_days + 1):
            day = today - timedelta(days=offset)
            trades = await self._journal.trades_for_session(day, timezone)
            pnl_today = sum(
                float(row.pnl)
                for row in trades
                if row.pnl is not None and row.closed_at is not None
            )
            if pnl_today != 0.0 or trades:
                totals[day] = pnl_today
        return totals

    def _today(self) -> date_cls:
        """NY-local calendar date (matches the trading session boundary)."""
        return datetime.now(ZoneInfo(self._settings.session.timezone)).date()


__all__ = [
    "RehabAdjustedCaps",
    "RehabEngine",
    "RehabRecord",
    "RehabState",
    "RehabStats",
    "RehabTier",
    "RehabTransition",
    "aggregate_daily_pnl",
    "classify_tier",
    "compute_stats_from_journal_entries",
    "consecutive_red_days",
    "cumulative_drawdown_usd",
    "delete_rehab_flag",
    "read_rehab_flag",
    "write_rehab_flag",
]
