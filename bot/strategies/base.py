"""Strategy ABC + shared ``Signal`` dataclass.

Strategies are pure-ish evaluators: given a recent bar DataFrame they emit
zero or one ``Signal`` per call. No I/O, no network, no IBKR references. That
keeps backtests and unit tests deterministic — the orchestrator is responsible
for actually placing orders (in Phase 4+).

This module also defines ``RejectedCandidate`` and the rejection-logging
convention (``signal.rejected`` structlog events). Strategies call
``self._reject(...)`` on every early-return path so the ``backtest`` CLI can
surface "almost-fired" setups alongside the signal list. Window-outside
rejections are intentionally silent — see ``WINDOW_REJECTIONS_SILENT``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import structlog

from bot.indicators import premarket_high

_log = structlog.get_logger("bot.strategies")

REJECTION_EVENT = "signal.rejected"

# Phase 8.4 — one penny below the premarket high when the cap binds. Works
# for stocks ≥ $1 (penny tick); slightly coarse on sub-$1 names but
# acceptable since fills typically come in just below resistance regardless.
_PMH_CAP_TICK = 0.01


def _apply_stop_distance_floor(
    *,
    entry: float,
    structural_stop: float,
    floor_min_abs: float,
    floor_min_pct: float,
    symbol: str,
    strategy: str,
    bar_time: datetime | pd.Timestamp,
) -> float:
    """Phase 10.2 — return the stop floored to ``max(min_abs, entry × min_pct)`` below entry.

    The strategy-emitted structural stop (lowest 10-RTH-bar low for
    momentum, ``min(VWAP, 3-bar pullback low)`` for gap-and-go) can
    occasionally land 1-2 cents below entry on tight consolidation
    breakouts. Microstructure noise tags those stops immediately;
    Phase 10.2 enforces a minimum distance below entry as a floor.

    Returns the wider of the two stops (lower price; "min" of the two
    price levels on a long). When the floor binds — i.e. structural
    stop sits closer to entry than the floor distance — emits
    ``entry.stop_distance_floor_applied`` with which branch won
    (``min_abs`` or ``min_pct``). When the floor does not bind, returns
    the structural stop unchanged and emits no event (the floor is a
    guard, not a signal).

    Caller must ensure ``structural_stop < entry`` (i.e. the existing
    ``nonpositive_risk`` early-return has already screened out broken
    setups). The floor is for "valid but too tight" stops, not for
    rescuing degenerate ones — ``risk_per_share`` after flooring is at
    least ``floor_min_abs`` (or zero if both floors are zero).
    """
    abs_floor = floor_min_abs
    pct_floor = entry * floor_min_pct
    floor_distance = max(abs_floor, pct_floor)
    if floor_distance <= 0.0:
        return structural_stop
    floored_stop = entry - floor_distance
    if floored_stop >= structural_stop:
        # Structural stop is already at or below the floored value
        # (i.e. risk is already at least the floor distance) — no bind.
        return structural_stop
    which = "min_abs" if abs_floor >= pct_floor else "min_pct"
    ts_iso = bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
    _log.info(
        "entry.stop_distance_floor_applied",
        symbol=symbol,
        strategy=strategy,
        bar_time=ts_iso,
        entry_price=round(entry, 4),
        structural_stop=round(structural_stop, 4),
        floor_distance=round(floor_distance, 4),
        floored_stop=round(floored_stop, 4),
        which_floor_won=which,
    )
    return floored_stop


def _apply_premarket_high_cap(
    *,
    entry: float,
    default_scale_out: float,
    bars: pd.DataFrame,
    enabled: bool,
) -> tuple[float, str | None, float]:
    """Phase 8.4 — return ``(scale_out, cap_reason, capped_target)``.

    When ``enabled`` is True and a premarket high above ``entry`` exists
    in ``bars``, returns ``min(default_scale_out, premarket_high − tick)``.
    Otherwise returns ``default_scale_out`` unchanged.

    ``cap_reason`` is ``"premarket_high"`` only when the cap actually
    binds — i.e. PMH exists, PMH > entry, AND PMH − tick < default
    scale-out. ``None`` in every other case (cap disabled, no premarket
    bars, gap already broke PMH, or PMH so far above that 2R is the
    binding ceiling). ``capped_target`` is the cap's value (PMH − tick)
    or 0.0 when no cap candidate exists; useful for log observability
    even when the cap didn't bind.
    """
    if not enabled:
        return default_scale_out, None, 0.0
    pmh = premarket_high(bars)
    if pmh is None or pmh <= entry:
        return default_scale_out, None, 0.0
    capped_target = pmh - _PMH_CAP_TICK
    if capped_target >= default_scale_out:
        # PMH so far above that 2R is the binding ceiling — no cap.
        return default_scale_out, None, capped_target
    return capped_target, "premarket_high", capped_target


# Every pre-09:30 and post-11:30 bar is "outside window". Logging each would
# drown the sidecar in thousands of identical rows and obscure the real
# rejections, so strategies short-circuit silently on window mismatch. This
# is the *only* silent rejection stage.
WINDOW_REJECTIONS_SILENT = True


@dataclass(frozen=True)
class Signal:
    """A single long-side entry proposal from a strategy.

    ``entry``/``stop`` are absolute prices. Phase 4e split the old ``target``
    into two explicit fields and Phase 4i made the runner ceiling optional:
    ``scale_out_price`` is the first-half take profit (the 2:1 anchor,
    computed from ``execution.scale_out_multiple``), and
    ``runner_target_price`` is the executor-chosen bracket ceiling — set
    only when ``execution.runner_target_enabled`` is true, else ``None``.
    The methodology doesn't place hard profit ceilings on the runner, so the
    default is no-runner-LMT and the trailing logic drives the exit. R:R
    (``risk_reward`` property) is measured against ``scale_out_price`` for
    observability — surfaced in ``signal.emitted`` and ``signal_bus.published``
    events so an operator can grep it across sessions.
    """

    symbol: str
    strategy: str
    entry: float
    stop: float
    scale_out_price: float
    runner_target_price: float | None
    timestamp: datetime
    reasons: list[str] = field(default_factory=list)
    # Phase 4c: latest-bar volume so the risk engine can apply the # "your order ≪ 2–5% of 1-min volume" liquidity cap. Opt-out via None
    # (synthetic tests don't have to bother); live strategies fill it.
    recent_bar_volume: int | None = None
    # Phase 7.1: stop-calculation diagnostics for post-session review. The
    # risk engine re-emits these in the stop_too_wide rejection event so an
    # operator can tell whether a too-wide stop was a volatility artefact,
    # a premarket wick leaking through (fixed in 7.1 but worth watching),
    # or a strategy-logic error. Optional so synthetic tests can skip them.
    pullback_low: float | None = None
    pullback_lookback_bars: int | None = None
    bars_available_for_lookback: int | None = None
    vwap_at_entry: float | None = None
    # Phase 12.5: a "current market" proxy for the LMT buffer ceiling
    # calculation. When set, the executor's percentage cap is anchored
    # on ``min(entry, market_anchor_price)`` rather than on ``entry``
    # alone -- preventing IBKR Error 202 cancellations on breakout-fade
    # bars where the breakout close sits well above current market.
    # Strategies populate this with the prior bar close (the most
    # recent fully-quoted price before the candidate breakout bar).
    # ``None`` falls back to the legacy entry-only ceiling.
    market_anchor_price: float | None = None
    # Standing-order override: when set, the executor uses this order type
    # regardless of ``execution.entry_order_type`` in config. Gap-and-go
    # sets this to ``"STP_LMT"`` for first-bar standing orders placed before
    # price breaks the trigger — the resting buy-stop sits on IBKR's servers
    # until the trigger is hit or the strategy window closes. ``None`` falls
    # back to the configured ``entry_order_type``.
    preferred_order_type: str | None = None

    @property
    def risk_per_share(self) -> float:
        """Positive dollar amount risked per share from entry to stop."""
        return max(self.entry - self.stop, 0.0)

    @property
    def reward_per_share(self) -> float:
        """Positive dollar amount gained per share from entry to scale-out.

        R:R is anchored on the scale-out (+1R for strategy-emitted signals by
        default, but whatever the strategy chose to emit). The runner ceiling
        is a separate ergonomic and deliberately does not affect R:R.
        """
        return max(self.scale_out_price - self.entry, 0.0)

    @property
    def risk_reward(self) -> float:
        """Reward-to-risk ratio. Returns 0.0 if risk is zero (invalid signal)."""
        if self.risk_per_share <= 0:
            return 0.0
        return self.reward_per_share / self.risk_per_share


@dataclass(frozen=True)
class RejectedCandidate:
    """Observational record of a bar that almost-fired but was filtered out.

    Emitted as a ``signal.rejected`` structlog event by strategies and the R:R
    gate; harvested by ``bot.backtest`` via ``structlog.testing.capture_logs``.
    Purely informational — never alters live trading behavior.
    """

    symbol: str
    strategy: str
    bar_time: datetime
    stage: str  # "setup" | "entry_trigger" | "stop_calculation"
    reason: str
    context: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract strategy contract.

    Concrete strategies implement ``evaluate`` and declare a ``name`` class
    attribute. ``evaluate`` should never mutate ``bars`` and never perform I/O.

    Phase 8.1: the ``rr_min`` floor + ``passes_rr`` gate were removed.
    Strategies build ``scale_out_price = entry + scale_out_multiple × risk``
    by construction, so R:R is pinned to ``scale_out_multiple`` (default 2.0)
    at emission time. The old post-emission ``passes_rr`` check compared
    that constructed R:R to the same 2.0 floor — a tautology that only
    rejected signals via float-division drift (``1.9999999999999998``).
    Strategies that adopt a dynamic scale-out target later should inline
    their own R:R validation at the point the target is chosen.
    """

    name: str = "strategy"

    # Phase 12.4 — per-strategy admission flag, defaulted at the ABC so generic
    # dispatcher code can read ``getattr(strategy, "catalyst_required", True)``
    # without per-strategy hasattr juggling. Concrete strategies override
    # via their constructor; the safe default is True (strict admission).
    catalyst_required: bool = True

    def __init__(self, scale_out_multiple: float = 2.0) -> None:
        """Store the scale-out anchor multiple.

        ``scale_out_multiple`` is the R-multiple at which the strategy emits
        its first-half take-profit anchor (``entry + N × initial_risk``).
        Defaults to 2.0 — the 2:1 R:R rule — and is overridden from config
        by the orchestrator.
        """
        self.scale_out_multiple = scale_out_multiple

    @abstractmethod
    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """Return a Signal if the setup fires on the latest bar, else None."""

    def _reject(
        self,
        symbol: str,
        bar_time: datetime | pd.Timestamp,
        stage: str,
        reason: str,
        **context: Any,
    ) -> Signal | None:
        """Emit a ``signal.rejected`` log event and return ``None``.

        Designed for use as ``return self._reject(...)`` at every non-window
        early-return site inside ``evaluate()``. The return type is
        ``Signal | None`` (always None) so callers can inline the return
        statement without mypy flagging the void-call pattern. The
        ``bar_time`` is serialised to ISO-8601 before logging so capture
        consumers can rehydrate it with ``datetime.fromisoformat``.
        """
        ts_iso = bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time)
        _log.info(
            REJECTION_EVENT,
            symbol=symbol,
            strategy=self.name,
            bar_time=ts_iso,
            stage=stage,
            reason=reason,
            **context,
        )
        return None
