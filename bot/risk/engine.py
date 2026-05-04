"""Phase 4b risk engine: sizing, halt triggers, PDT advisory, halt-flag persistence.

Every entry from ``Executor.handle_signal`` funnels through
``RiskEngine.check_entry`` — a non-optional sequence of gates that either
approves the entry with a share size or rejects it with a kebab-case reason.
Gates run in this order (stop at first failure):

1. **halted** — in-process halt flag from a prior ``on_fill_closed`` that
   tripped daily-loss, daily-profit, or give-back.
2. **max_trades_per_day_exceeded** — counter resets on ``reset_for_new_session``.
3. **max_concurrent_positions_exceeded** — global across all symbols.
4. **insufficient_share_count** — ``compute_shares`` returned < 1
   (stop too wide for the risk budget, or the position-value cap is binding
   on a microprice).
5. **margin_awareness_exceeded** — position value > ``AvailableFunds * 0.95``
   (intraday maintenance-margin buffer). Under the post-PDT-abolition FINRA
   regime, this is the real gate; PDT is advisory-only.
6. **insufficient_buying_power** — final sanity check against ``BuyingPower``.

PDT (``DayTradesRemaining``) is emitted as a ``pdt.advisory`` log event but
**never blocks**. FINRA Rule 4210 amendment SR-FINRA-2025-017 (SEC approved
April 14, 2026) eliminated the Pattern Day Trader designation; brokers still
report the field during the 45-day-to-18-month transition, but we don't
duplicate broker-side enforcement.

**Halt persistence (``logs/halt.flag``)** is the single source of truth across
process restarts. ``RiskEngine.halted`` is the in-process cache; the flag
file is the durable record. ``apply_halt_flag_if_current`` reconciles the two
on startup. Stale flags (date != today) auto-clean; same-date overwrites are
refused to preserve the original halt reason.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import date as date_cls
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import structlog

from bot.config import Settings, get_settings

if TYPE_CHECKING:
    from bot.execution.position_state import Position, PositionStore, SymbolHistory
    from bot.risk.rehab import RehabEngine
    from bot.strategies.base import Signal

_log = structlog.get_logger("bot.risk")

_DEFAULT_HALT_FLAG_PATH = Path("logs") / "halt.flag"
_MARGIN_BUFFER = 0.95


# ---------- Decision types ---------- #


@dataclass(frozen=True)
class Approved:
    """Entry approved with a sized share count."""

    shares: int


@dataclass(frozen=True)
class Rejected:
    """Entry rejected with a short kebab-case reason + structured detail."""

    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


RiskDecision = Approved | Rejected


@dataclass(frozen=True)
class ReEntryAllowed:
    """Phase 4d re-entry gate approved; carries the size multiplier for ``entries_count``.

    ``multiplier`` scales ``max_loss_per_trade_usd`` before ``compute_shares``
    so the "cautious on later pullbacks" shows up as a smaller share size,
    not a different stop. ``entries_count`` is 0-indexed (0 = first entry of
    the day).
    """

    multiplier: float
    entries_count: int


ReEntryDecision = ReEntryAllowed | Rejected


# ---------- State ---------- #


@dataclass
class RiskState:
    """Mutable in-process risk tally; reset on ``reset_for_new_session``.

    ``session_date`` is the NY-local date the state was initialized for;
    the scheduler uses this to decide whether a fresh session is required.
    ``max_pnl_today_usd`` tracks the day's peak to drive the give-back rule —
    once peak clears ``giveback_trigger_usd``, a subsequent drop of
    ``giveback_pct`` of that peak will halt.

    Phase 9.6: ``trades_today`` increments only on confirmed fill (see
    ``RiskEngine.on_first_fill``); placement / approval no longer increments.
    Day 8 RPGL exhausted the daily trade budget with three TWS-rejected
    LMTs that never filled, blocking later legitimate signals on other
    symbols. ``broker_rejection_count`` tracks consecutive broker auto-cancels
    per symbol; ``blocked_symbols`` flags symbols that have hit the
    ``_BROKER_REJECTION_THRESHOLD`` and should be locked out for the
    remainder of the session.
    """

    session_date: date_cls
    trades_today: int = 0
    realized_pnl_usd: float = 0.0
    max_pnl_today_usd: float = 0.0
    halted: bool = False
    halt_reason: str | None = None
    halt_at: datetime | None = None
    broker_rejection_count: dict[str, int] = field(default_factory=dict)
    blocked_symbols: set[str] = field(default_factory=set)


# Phase 9.6 — N consecutive broker auto-cancels lock out the symbol for the
# session. Two is conservative: a single rejection could be transient (rate
# limit, momentary halt), but two in a row indicates a structural issue with
# the symbol (eligibility, routing, contract type) that won't resolve until
# the next session. TODO: make configurable in config.yaml if operator wants
# different thresholds.
_BROKER_REJECTION_THRESHOLD = 2


@dataclass(frozen=True)
class HaltRecord:
    """JSON-serializable halt state written to ``logs/halt.flag``."""

    date: date_cls
    reason: str
    triggered_at: datetime
    pnl_at_halt: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (dates + datetimes → ISO 8601)."""
        return {
            "date": self.date.isoformat(),
            "reason": self.reason,
            "triggered_at": self.triggered_at.isoformat(),
            "pnl_at_halt": self.pnl_at_halt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HaltRecord:
        """Rehydrate from a JSON-decoded dict; raises ``ValueError`` on bad shape."""
        return cls(
            date=date_cls.fromisoformat(str(data["date"])),
            reason=str(data["reason"]),
            triggered_at=datetime.fromisoformat(str(data["triggered_at"])),
            pnl_at_halt=float(data["pnl_at_halt"]),
        )


# ---------- Pure functions ---------- #


def compute_shares(
    signal: Signal,
    max_loss_per_trade_usd: float,
    max_position_value_usd: float,
    recent_bar_volume: int | None = None,
    max_pct_of_bar_volume: float = 2.0,
) -> int:
    """Size a trade by the per-trade $ max loss, capped by total position value.

    ``shares_by_risk`` = floor(max_loss / (entry - stop)) — the the methodology rule.
    ``shares_by_value`` = floor(max_position_value / entry) — penny-stock guard.
    ``shares_by_liquidity`` (optional) = floor(recent_bar_volume * pct/100) —
    the "your order ≪ 2-5% of 1-min volume" rule. Skipped when
    ``recent_bar_volume is None`` so synthetic tests opt out cleanly.
    Return the binding minimum; 0 on invalid inputs (non-positive risk or price).

    When the liquidity cap is the binding constraint we emit
    ``risk.shares_capped_by_liquidity`` with all three candidate values so
    tuning the percentage has visibility in the logs.
    """
    risk_per_share = signal.entry - signal.stop
    if risk_per_share <= 0 or signal.entry <= 0:
        return 0
    shares_by_risk = math.floor(max_loss_per_trade_usd / risk_per_share)
    shares_by_value = math.floor(max_position_value_usd / signal.entry)
    if recent_bar_volume is not None and recent_bar_volume >= 0:
        shares_by_liquidity = math.floor(recent_bar_volume * (max_pct_of_bar_volume / 100.0))
        final = min(shares_by_risk, shares_by_value, shares_by_liquidity)
        if shares_by_liquidity < min(shares_by_risk, shares_by_value):
            _log.info(
                "risk.shares_capped_by_liquidity",
                symbol=signal.symbol,
                shares_by_risk=shares_by_risk,
                shares_by_value=shares_by_value,
                shares_by_liquidity=shares_by_liquidity,
                recent_bar_volume=recent_bar_volume,
                max_pct_of_bar_volume=max_pct_of_bar_volume,
            )
        return int(max(final, 0))
    return int(max(min(shares_by_risk, shares_by_value), 0))


def daily_loss_hit(realized_pnl_usd: float, max_daily_loss_usd: float) -> bool:
    """True when realized PnL is at or worse than the negative daily-loss cap."""
    return realized_pnl_usd <= -abs(max_daily_loss_usd)


def profit_goal_hit(realized_pnl_usd: float, daily_profit_goal_usd: float) -> bool:
    """True when realized PnL meets or exceeds the goal (the methodology: walk away)."""
    return realized_pnl_usd >= daily_profit_goal_usd


def giveback_hit(
    realized_pnl_usd: float,
    peak_pnl_usd: float,
    giveback_trigger_usd: float,
    giveback_pct: float,
) -> bool:
    """True when peak cleared the trigger and current has bled back the percentage.

    Only arms once peak >= trigger; below that floor we treat the day as
    undecided and never halt on a pullback.
    """
    if peak_pnl_usd < giveback_trigger_usd:
        return False
    threshold = peak_pnl_usd * (1.0 - giveback_pct / 100.0)
    return realized_pnl_usd <= threshold


# ---------- Halt-flag I/O ---------- #


def read_halt_flag(path: Path) -> HaltRecord | None:
    """Return the persisted halt record, or None on missing/corrupt file."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return HaltRecord.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        _log.warning("halt_flag.read_failed", path=str(path), error=str(exc))
        return None


def write_halt_flag(path: Path, record: HaltRecord) -> bool:
    """Persist a halt record; refuse to overwrite an existing same-date flag.

    Returns True on write, False on same-date duplicate. Cross-date rewrites
    are allowed because ``reset_for_new_session`` rotates the date.
    """
    existing = read_halt_flag(path)
    if existing is not None and existing.date == record.date:
        _log.info(
            "halt_flag.duplicate_refused",
            path=str(path),
            existing_reason=existing.reason,
            new_reason=record.reason,
        )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    _log.warning(
        "halt_flag.written",
        path=str(path),
        reason=record.reason,
        pnl_at_halt=round(record.pnl_at_halt, 2),
        triggered_at=record.triggered_at.isoformat(),
    )
    return True


def delete_halt_flag(path: Path) -> bool:
    """Delete the flag file if it exists; return True iff a file was removed."""
    if not path.exists():
        return False
    path.unlink()
    _log.info("halt_flag.deleted", path=str(path))
    return True


# ---------- Engine ---------- #


class RiskEngine:
    """Process-wide risk gates + halt state.

    One instance per session. Not safe against fork; intended for a single
    event loop. All mutating entrypoints are serialized by an internal
    ``asyncio.Lock`` so fill-close callbacks (which race with the signal
    loop) can't corrupt the counters.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        halt_flag_path: Path | None = None,
        rehab_engine: RehabEngine | None = None,
    ) -> None:
        """Initialize counters for today's NY-local session; caller applies the halt flag.

        ``rehab_engine`` is optional — when supplied, its ``apply_to_caps``
        result overrides the base ``max_loss_per_trade_usd``,
        ``max_daily_loss_usd``, and ``max_trades_per_day`` on each entry.
        Legacy tests and adoption paths pass ``None`` and run against the
        base config unchanged (equivalent to ``rehab.enabled: false``).
        """
        self._settings = settings or get_settings()
        self._halt_flag_path = halt_flag_path or _DEFAULT_HALT_FLAG_PATH
        self._rehab = rehab_engine
        self._lock = asyncio.Lock()
        self._state = RiskState(session_date=self._today())

    @property
    def state(self) -> RiskState:
        """Expose the mutable state for read-only inspection (CLI + tests)."""
        return self._state

    @property
    def halt_flag_path(self) -> Path:
        """Path to the persisted halt flag (``logs/halt.flag`` by default)."""
        return self._halt_flag_path

    def is_halted(self) -> bool:
        """In-process halt cache — cheap to check every tick of the loop."""
        return self._state.halted

    async def check_entry(
        self,
        signal: Signal,
        position_store: PositionStore,
        account_summary: dict[str, str],
    ) -> RiskDecision:
        """Run every gate in order; return the first ``Rejected`` or a final ``Approved``.

        Increments ``trades_today`` on approval. Every rejection is logged
        with ``risk.entry_rejected`` + structured context so the CLI + log
        audit can reconstruct "why did my signal not fire?".
        """
        async with self._lock:
            return self._check_entry_locked(signal, position_store, account_summary)

    def _check_entry_locked(
        self,
        signal: Signal,
        position_store: PositionStore,
        account_summary: dict[str, str],
    ) -> RiskDecision:
        """Unlocked gate evaluation; caller holds ``self._lock``.

        If a ``RehabEngine`` was supplied, its ``apply_to_caps`` result
        is consulted once per entry and overrides the three base risk
        caps (per-trade loss, daily loss, max trades) throughout the
        gate chain. Rejection logs include both the effective and the
        base cap so the operator can see when a rehab tier — not the
        raw config — is gating their trade.
        """
        caps = self._rehab.apply_to_caps(self._settings.risk) if self._rehab else None
        effective_max_trades = (
            caps.max_trades_per_day if caps is not None else self._settings.risk.max_trades_per_day
        )
        effective_max_loss = (
            caps.max_loss_per_trade_usd
            if caps is not None
            else self._settings.risk.max_loss_per_trade_usd
        )
        rehab_detail: dict[str, Any] = {}
        if caps is not None and caps.trigger_reason is not None:
            rehab_detail = {
                "rehab_tier": caps.tier.value,
                "rehab_trigger": caps.trigger_reason,
                "base_max_trades_per_day": caps.base_max_trades_per_day,
                "base_max_loss_per_trade_usd": caps.base_max_loss_per_trade_usd,
            }

        if self._state.halted:
            return self._reject(
                "halted",
                symbol=signal.symbol,
                halt_reason=self._state.halt_reason,
            )

        # Phase 9.6 — symbol locked out after repeated broker auto-cancels.
        # Avoid the cascade where a structural broker reject (e.g. SCM
        # eligibility on RPGL 2026-04-29) kept eating signal budget despite
        # zero fills. Block fires before any other gate so we don't even
        # account it against the per-trade or daily caps.
        if signal.symbol in self._state.blocked_symbols:
            return self._reject(
                "symbol_blocked_broker_rejections",
                symbol=signal.symbol,
                rejection_count=self._state.broker_rejection_count.get(signal.symbol, 0),
                threshold=_BROKER_REJECTION_THRESHOLD,
            )

        if self._state.trades_today >= effective_max_trades:
            return self._reject(
                "max_trades_per_day_exceeded",
                symbol=signal.symbol,
                count=self._state.trades_today,
                limit=effective_max_trades,
                **rehab_detail,
            )

        # Phase 4d re-entry gate. Returns the 0-indexed entry number's size
        # multiplier on approval (1.0 for the first entry), or a Rejected
        # decision with one of the re-entry reasons. Runs *before* the
        # concurrent-positions gate so a superseded same-symbol signal (which
        # the Executor catches with has_active before calling us) doesn't
        # waste cycles on the rest of the chain.
        history = position_store.symbol_history(signal.symbol)
        reentry = self._check_reentry_locked(signal, history)
        if isinstance(reentry, Rejected):
            return reentry
        adjusted_max_loss = effective_max_loss * reentry.multiplier

        active_count = len(position_store.list_active())
        if active_count >= self._settings.risk.max_concurrent_positions:
            return self._reject(
                "max_concurrent_positions_exceeded",
                symbol=signal.symbol,
                active=active_count,
                limit=self._settings.risk.max_concurrent_positions,
            )

        shares = compute_shares(
            signal,
            adjusted_max_loss,
            self._settings.risk.max_position_value_usd,
            recent_bar_volume=signal.recent_bar_volume,
            max_pct_of_bar_volume=self._settings.risk.max_pct_of_bar_volume,
        )
        if shares < 1:
            return self._reject(
                "insufficient_share_count",
                symbol=signal.symbol,
                entry=signal.entry,
                stop=signal.stop,
                max_loss_budget=adjusted_max_loss,
                max_position_value=self._settings.risk.max_position_value_usd,
                reentry_multiplier=reentry.multiplier,
                entries_count=reentry.entries_count,
            )

        # Phase 4c strategy-quality gate: refuse setups where the stop is wider
        # than the published threshold — 50¢ default. Runs after sizing
        # (so we have the real computed risk_per_share) but before margin so
        # a wide stop never burns buying-power checks.
        stop_width = signal.entry - signal.stop
        if stop_width > self._settings.risk.max_stop_width_usd:
            return self._reject(
                "stop_too_wide",
                symbol=signal.symbol,
                stop_width_usd=round(stop_width, 4),
                max=self._settings.risk.max_stop_width_usd,
                entry=signal.entry,
                stop=signal.stop,
                # Phase 7.1: strategy-side stop-calc diagnostics re-emitted
                # here so post-session review can answer: was the stop wide
                # because of volatility, a wick, or a bad reference pick?
                pullback_low=signal.pullback_low,
                pullback_lookback_bars=signal.pullback_lookback_bars,
                bars_available_for_lookback=signal.bars_available_for_lookback,
                vwap_at_entry=signal.vwap_at_entry,
            )

        available_funds = _parse_summary_float(account_summary, "AvailableFunds")
        position_value = shares * signal.entry
        margin_cap = available_funds * _MARGIN_BUFFER
        if available_funds > 0 and position_value > margin_cap:
            return self._reject(
                "margin_awareness_exceeded",
                symbol=signal.symbol,
                position_value=round(position_value, 2),
                available_funds=round(available_funds, 2),
                margin_cap=round(margin_cap, 2),
            )

        buying_power = _parse_summary_float(account_summary, "BuyingPower")
        if buying_power > 0 and position_value > buying_power:
            return self._reject(
                "insufficient_buying_power",
                symbol=signal.symbol,
                position_value=round(position_value, 2),
                buying_power=round(buying_power, 2),
            )

        _emit_pdt_advisory(account_summary)

        # Phase 9.6: ``trades_today`` increments at fill, not approval. The
        # cap-check above is still the authoritative gate, but the counter
        # only advances when ``on_first_fill`` is called from the executor's
        # parent-fill handler. This prevents broker auto-cancels (Day 8 RPGL)
        # from consuming the daily trade budget without ever opening a real
        # position.
        _log.info(
            "risk.entry_approved",
            symbol=signal.symbol,
            strategy=signal.strategy,
            shares=shares,
            trades_today=self._state.trades_today,
            max_trades=effective_max_trades,
            position_value=round(position_value, 2),
            entries_count=reentry.entries_count,
            reentry_multiplier=reentry.multiplier,
            **rehab_detail,
        )
        return Approved(shares=shares)

    async def check_reentry(
        self,
        signal: Signal,
        symbol_history: SymbolHistory,
    ) -> ReEntryDecision:
        """Public async facade over the locked re-entry gate — used by tests + CLI dry-run.

        Callers inside ``check_entry`` go through ``_check_reentry_locked``
        directly (they already hold ``self._lock``). External callers
        (``tests/test_reentry.py``) want the serialized path.
        """
        async with self._lock:
            return self._check_reentry_locked(signal, symbol_history)

    def _check_reentry_locked(
        self,
        signal: Signal,
        history: SymbolHistory,
    ) -> ReEntryDecision:
        """Evaluate the Phase 4d re-entry gates in order; first failure wins.

        Gate order (rationale in PLAN / PHASE_4D_PROMPT):

        1. ``re_entry_disabled`` — master switch off AND this isn't the first
           entry of the session. First entries always pass regardless of the
           switch so disabling re-entries doesn't accidentally block the
           strategy loop entirely.
        2. ``auto_flattened_terminal`` — the 15:55 scheduler already closed a
           position on this symbol; the session is ending and another bracket
           would risk overnight hold if the close runs long.
        3. ``max_reentries_reached`` — hard cap on entries-per-symbol.
        4. ``prior_exit_unprofitable`` — the "don't revenge-trade" rule.
        5. ``reentry_cooldown_active`` — throttle same-symbol signals so the
           bot can't chain three within one bar.
        6. multiplier computation — returns ``ReEntryAllowed`` with the
           ``size_multipliers[entries_count]`` scalar.

        Every rejection carries ``entries_count``, ``last_exit_type``, and the
        cooldown-remaining delta so the log + CLI can explain the block.
        """
        cfg = self._settings.risk.re_entry
        entries = history.entries_count
        last_type = history.last_exit_type
        cooldown_remaining = _cooldown_remaining_seconds(history, cfg.cooldown_seconds)

        base_detail: dict[str, Any] = {
            "symbol": signal.symbol,
            "entries_count": entries,
            "last_exit_type": last_type,
            "cooldown_remaining_s": cooldown_remaining,
        }

        if not cfg.enabled and entries > 0:
            return self._reject("re_entry_disabled", **base_detail)

        if last_type == "auto_flatten":
            return self._reject(
                "auto_flattened_terminal",
                reason_detail="session_ending",
                **base_detail,
            )

        if entries >= cfg.max_entries_per_symbol:
            return self._reject(
                "max_reentries_reached",
                limit=cfg.max_entries_per_symbol,
                **base_detail,
            )

        if (
            cfg.require_profitable_prior_exit
            and entries > 0
            and history.last_exit_pnl is not None
            and history.last_exit_pnl <= 0
        ):
            return self._reject(
                "prior_exit_unprofitable",
                last_exit_pnl=round(history.last_exit_pnl, 2),
                **base_detail,
            )

        if cooldown_remaining > 0:
            return self._reject(
                "reentry_cooldown_active",
                cooldown_total_s=cfg.cooldown_seconds,
                **base_detail,
            )

        # Bounds-check defensively; ReEntryConfig's validator guarantees this
        # but returning a fail-safe 1.0 is better than IndexError on a malformed
        # runtime mutation.
        if entries < len(cfg.size_multipliers):
            multiplier = float(cfg.size_multipliers[entries])
        else:
            multiplier = 1.0
            _log.warning(
                "risk.reentry_multiplier_fallback",
                symbol=signal.symbol,
                entries_count=entries,
                multipliers_len=len(cfg.size_multipliers),
            )
        return ReEntryAllowed(multiplier=multiplier, entries_count=entries)

    def _reject(self, reason: str, **detail: Any) -> Rejected:
        """Log ``risk.entry_rejected`` with the gate reason and return a Rejected decision."""
        _log.warning("risk.entry_rejected", reason=reason, **detail)
        return Rejected(reason=reason, detail=dict(detail))

    async def on_first_fill(self, symbol: str) -> None:
        """Phase 9.6 — called once per opened position when the parent first fills.

        Increments ``trades_today`` (the daily-trade-cap counter; previously
        incremented at approval time, which over-counted broker-rejected
        placements that never produced a real position) and resets the
        per-symbol broker-rejection counter on the assumption that a
        successful fill clears whatever transient condition produced the
        prior rejections, if any.

        Idempotent at the executor level: ``mark_filled`` runs once per
        position via ``filledEvent`` on the parent leg. Subsequent partial
        fills on the same position go through different status pipelines
        and do not re-enter ``_handle_parent_fill``.
        """
        async with self._lock:
            self._state.trades_today += 1
            previous_rejection_count = self._state.broker_rejection_count.pop(symbol, 0)
            self._state.blocked_symbols.discard(symbol)
            _log.info(
                "risk.first_fill",
                symbol=symbol,
                trades_today=self._state.trades_today,
                cleared_broker_rejections=previous_rejection_count,
            )

    async def on_broker_rejection(self, symbol: str, *, error_code: int | None = None) -> bool:
        """Phase 9.6 — record a broker auto-cancel for ``symbol``; return True if just blocked.

        Called by the executor when a parent order finished without our
        explicit cancel (i.e. ``trade.isDone()`` at expire time without
        the bot having issued ``cancelOrder``). Increments the per-symbol
        rejection counter; on the ``_BROKER_REJECTION_THRESHOLD``-th
        consecutive rejection, locks the symbol out for the rest of the
        session via ``blocked_symbols``. Returns whether this call was
        the one that just crossed the threshold so the caller can emit
        the watchlist-drop event without re-emitting on subsequent
        rejections after the lock-out.
        """
        async with self._lock:
            count = self._state.broker_rejection_count.get(symbol, 0) + 1
            self._state.broker_rejection_count[symbol] = count
            already_blocked = symbol in self._state.blocked_symbols
            just_blocked = False
            if count >= _BROKER_REJECTION_THRESHOLD and not already_blocked:
                self._state.blocked_symbols.add(symbol)
                just_blocked = True
            _log.warning(
                "executor.broker_rejection_detected",
                symbol=symbol,
                error_code=error_code,
                rejection_count=count,
                threshold=_BROKER_REJECTION_THRESHOLD,
                blocked=symbol in self._state.blocked_symbols,
            )
            return just_blocked

    def is_symbol_blocked(self, symbol: str) -> bool:
        """Phase 9.6 — true when ``symbol`` has hit the broker-rejection lockout."""
        return symbol in self._state.blocked_symbols

    async def on_fill_closed(self, position: Position, pnl: float) -> None:
        """Update realized PnL + peak PnL; trip halts if a threshold fires.

        The halt state is written to both in-process (``self._state.halted``)
        and on-disk (``logs/halt.flag``) so a restart sees the halt. Halt
        order: daily-loss > daily-profit > give-back (only one fires per
        fill; most-severe-first). The daily-loss cap honours the active
        rehab tier — in REHAB, the operator hits "daily loss limit" at
        half the base dollar amount, matching the scale-down rule.
        """
        async with self._lock:
            self._state.realized_pnl_usd += pnl
            if self._state.realized_pnl_usd > self._state.max_pnl_today_usd:
                self._state.max_pnl_today_usd = self._state.realized_pnl_usd

            cur = self._state.realized_pnl_usd
            peak = self._state.max_pnl_today_usd
            rehab_caps = self._rehab.apply_to_caps(self._settings.risk) if self._rehab else None
            effective_daily_loss = (
                rehab_caps.max_daily_loss_usd
                if rehab_caps is not None
                else self._settings.risk.max_daily_loss_usd
            )
            halt_reason = _classify_halt(cur, peak, self._settings.risk, effective_daily_loss)

            _log.info(
                "risk.on_fill_closed",
                symbol=position.symbol,
                pnl=round(pnl, 2),
                realized_pnl=round(cur, 2),
                peak_pnl=round(peak, 2),
                halt_triggered=halt_reason,
            )

            if halt_reason is None or self._state.halted:
                return

            now_local = self._now_local()
            self._state.halted = True
            self._state.halt_reason = halt_reason
            self._state.halt_at = now_local
            record = HaltRecord(
                date=now_local.date(),
                reason=halt_reason,
                triggered_at=now_local,
                pnl_at_halt=cur,
            )
            write_halt_flag(self._halt_flag_path, record)
            _log.warning(
                "risk.halt_triggered",
                reason=halt_reason,
                pnl=round(cur, 2),
                peak=round(peak, 2),
            )

    async def reset_for_new_session(self) -> None:
        """Zero the counters + clear in-process halt state. Does NOT delete the flag.

        Intentional: halt persistence is user-driven friction. The operator
        must explicitly ``reset-halt`` (or delete the file) to resume after
        a halt. A fresh session date also auto-cleans stale flags via
        ``apply_halt_flag_if_current``, preserving same-day friction.
        """
        async with self._lock:
            self._state = RiskState(session_date=self._today())
            _log.info("risk.session_reset", session_date=self._state.session_date.isoformat())

    async def load_halt_flag(self) -> HaltRecord | None:
        """Read and return the current halt flag (no state mutation)."""
        return read_halt_flag(self._halt_flag_path)

    async def apply_halt_flag_if_current(self) -> HaltRecord | None:
        """On startup: adopt a same-day halt flag into state; stale flags auto-clean.

        Returns the adopted record (if any). The caller typically treats a
        non-None return as "refuse to trade this session" — the CLI does
        exactly that on ``trade``/``watch``.
        """
        today = self._today()
        record = read_halt_flag(self._halt_flag_path)
        if record is None:
            return None
        if record.date != today:
            delete_halt_flag(self._halt_flag_path)
            _log.info(
                "halt_flag.stale_cleaned",
                stale_date=record.date.isoformat(),
                today=today.isoformat(),
            )
            return None
        self._state.halted = True
        self._state.halt_reason = record.reason
        self._state.halt_at = record.triggered_at
        _log.warning(
            "halt_flag.adopted_on_startup",
            reason=record.reason,
            pnl_at_halt=round(record.pnl_at_halt, 2),
        )
        return record

    def _today(self) -> date_cls:
        """NY-local calendar date (matches the trading session boundary)."""
        return self._now_local().date()

    def _now_local(self) -> datetime:
        """Current time in the configured session timezone (tz-aware)."""
        return datetime.now(ZoneInfo(self._settings.session.timezone))


# ---------- Helpers ---------- #


def _classify_halt(
    realized_pnl_usd: float,
    peak_pnl_usd: float,
    risk_cfg: Any,
    effective_daily_loss_usd: float | None = None,
) -> str | None:
    """Return the halt reason kebab-case name, or None if no threshold fires.

    ``effective_daily_loss_usd`` overrides ``risk_cfg.max_daily_loss_usd``
    so the rehab tier can shrink the loss floor without mutating config.
    Legacy callers pass ``None`` and fall back to the base config value.
    """
    daily_loss_cap = (
        effective_daily_loss_usd
        if effective_daily_loss_usd is not None
        else risk_cfg.max_daily_loss_usd
    )
    if daily_loss_hit(realized_pnl_usd, daily_loss_cap):
        return "daily_loss_limit"
    if profit_goal_hit(realized_pnl_usd, risk_cfg.daily_profit_goal_usd):
        return "daily_profit_goal"
    if giveback_hit(
        realized_pnl_usd,
        peak_pnl_usd,
        risk_cfg.giveback_trigger_usd,
        risk_cfg.giveback_pct,
    ):
        return "giveback_limit"
    return None


def _cooldown_remaining_seconds(history: SymbolHistory, cooldown_seconds: int) -> int:
    """Seconds left before the re-entry cooldown expires; 0 if already clear.

    ``last_exit_time`` is stored tz-aware UTC (``executor._now_utc``), so the
    comparison is always against ``datetime.now(UTC)``. Returns an int so
    structlog renders it cleanly in the rejection log.
    """
    if history.last_exit_time is None:
        return 0
    now = datetime.now(UTC)
    elapsed = (now - history.last_exit_time).total_seconds()
    remaining = cooldown_seconds - elapsed
    if remaining <= 0:
        return 0
    return int(math.ceil(remaining))


def _parse_summary_float(account_summary: dict[str, str], key: str) -> float:
    """Safe float parse from an IBKR account-summary dict; 0.0 on missing/bad value."""
    raw = account_summary.get(key)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _emit_pdt_advisory(account_summary: dict[str, str]) -> None:
    """Log ``pdt.advisory`` for the current ``DayTradesRemaining`` value; never blocks.

    FINRA abolished the PDT designation in the April 2026 rule amendment;
    during the transition IBKR still reports the field. We surface it as a
    dashboard signal (operator awareness) but do not duplicate the broker's
    hard gate.
    """
    raw = account_summary.get("DayTradesRemaining")
    if raw is None:
        return
    try:
        remaining = int(raw)
    except (ValueError, TypeError):
        return
    if remaining == -1:
        _log.info("pdt.advisory", day_trades_remaining=-1, note="unlimited_account")
    elif remaining <= 1:
        _log.warning(
            "pdt.advisory",
            day_trades_remaining=remaining,
            hint="approaching_limit_broker_may_reject",
        )
    else:
        _log.info("pdt.advisory", day_trades_remaining=remaining)


__all__ = [
    "Approved",
    "HaltRecord",
    "ReEntryAllowed",
    "ReEntryDecision",
    "Rejected",
    "RiskDecision",
    "RiskEngine",
    "RiskState",
    "compute_shares",
    "daily_loss_hit",
    "delete_halt_flag",
    "giveback_hit",
    "profit_goal_hit",
    "read_halt_flag",
    "write_halt_flag",
]
