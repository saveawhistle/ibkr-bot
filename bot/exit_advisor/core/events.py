"""Event taxonomy for the exit-advisor harness.

Layer 1 implements three classes: time, pnl, order_state. Each event type
maps to a config flag in ``bot.config.ExitEventsConfig``; if the flag is
False, the harness must not emit that event type at all. Disabled deferred
classes (price_levels, moving_averages, etc.) are not represented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from bot.exit_advisor.decision.policy import ExitDecision


@dataclass(frozen=True)
class Event:
    """Common base for all events the harness emits during replay."""

    timestamp: datetime
    symbol: str


# --- time class ---


@dataclass(frozen=True)
class TimeOfDayMilestone(Event):
    """Wall-clock minutes after RTH open (09:30 ET) crossed a configured milestone."""

    minutes_after_open: int


@dataclass(frozen=True)
class TimeInTradeMilestone(Event):
    """Minutes since position entry crossed a configured milestone."""

    minutes_in_trade: int


# --- pnl class ---


@dataclass(frozen=True)
class RMultipleReached(Event):
    """An R-multiple threshold was crossed in the named direction."""

    r_multiple: float
    direction: Literal["up", "down"]


@dataclass(frozen=True)
class DrawdownFromPeak(Event):
    """Open P&L pulled back by ``drawdown_pct`` from the running peak R."""

    drawdown_pct: float
    peak_r_multiple: float
    current_r_multiple: float


@dataclass(frozen=True)
class MaxFavorableExcursionUpdate(Event):
    """The trade made a new high-water R since entry."""

    new_peak_r_multiple: float
    previous_peak_r_multiple: float


# --- order_state class ---


@dataclass(frozen=True)
class PositionProtected(Event):
    """Fires once per trade after entry fill AND initial bracket children
    are confirmed working. Until this fires, advisor recommendations are
    suppressed (see :class:`harness.TradeReplayHarness`).
    """

    entry_price: float
    initial_stop: float
    initial_scale_out: float
    position_size: int


@dataclass(frozen=True)
class PartialFillEvent(Event):
    order_id: int
    filled_quantity: int
    remaining_quantity: int
    fill_price: float
    side: Literal["buy", "sell"]


@dataclass(frozen=True)
class OrderRejectionEvent(Event):
    order_id: int
    error_code: int | None
    reason: str


# --- price_levels class (layer 2) ---


@dataclass(frozen=True)
class LevelTouched(Event):
    """A configured price level was reached for the first time this
    session (or the first time since price meaningfully retreated)."""

    level_name: Literal[
        "hod",
        "lod",
        "prior_day_high",
        "prior_day_low",
        "prior_day_close",
        "gap_fill",
    ]
    level_price: float
    current_price: float
    direction: Literal["from_below", "from_above"]


@dataclass(frozen=True)
class LevelReclaimed(Event):
    """A bar closes back through a level it had previously broken."""

    level_name: Literal[
        "hod",
        "lod",
        "prior_day_high",
        "prior_day_low",
        "prior_day_close",
        "gap_fill",
    ]
    level_price: float
    direction: Literal["above_to_below", "below_to_above"]


@dataclass(frozen=True)
class LevelDataUnavailable(Event):
    """One-shot warning emitted when a configured level cannot be computed
    because the upstream data is missing (e.g. prior-day session log
    absent). Detectors that depend on the level emit this once and then
    suppress subsequent events for that level."""

    level_name: str
    reason: str


# --- moving_averages class (layer 2) ---


@dataclass(frozen=True)
class MovingAverageCross(Event):
    ma_name: Literal["vwap", "ema_9"]
    ma_value: float
    direction: Literal["price_above_to_below", "price_below_to_above"]
    bar_close: float


# --- volume class (layer 2) ---


@dataclass(frozen=True)
class VolumeSpike(Event):
    bar_volume: int
    rolling_average: float
    ratio: float
    threshold: float


@dataclass(frozen=True)
class VolumeDryUp(Event):
    bar_volume: int
    rolling_average: float
    ratio: float
    threshold: float


@dataclass(frozen=True)
class RVolMilestone(Event):
    """Session-cumulative volume crossed a configured RVOL milestone.

    RVOL = today's session-cumulative volume to current bar / prior-N-day
    average cumulative volume to same time-of-day.

    ``prior_days_used`` reports how many days of cache data backed the
    average — may be less than the configured lookback if some days
    are missing (delisting, .unavailable markers).
    """

    rvol: float
    milestone: float
    cumulative_volume_today: int = 0
    prior_n_day_average_at_time: float = 0.0
    prior_days_used: int = 0


@dataclass(frozen=True)
class RVolDataUnavailable(Event):
    """One-shot warning when prior-day session logs aren't available so
    RVOL milestones cannot be computed."""

    reason: str


# --- bar_shape class (layer 2) ---


@dataclass(frozen=True)
class BarShapeDetected(Event):
    shape: Literal[
        "doji",
        "hammer",
        "shooting_star",
        "engulfing",
        "inside_bar",
        "outside_bar",
    ]
    bar_open: float
    bar_high: float
    bar_low: float
    bar_close: float


@dataclass(frozen=True)
class WickEvent(Event):
    """A bar's upper or lower wick exceeded the configured ratio of the
    bar's total range."""

    wick_side: Literal["upper", "lower"]
    wick_size: float
    body_size: float
    total_range: float
    wick_ratio: float


@dataclass(frozen=True)
class ConsecutiveBars(Event):
    """N consecutive same-direction bars. Fires every bar past the
    threshold, with updated count."""

    direction: Literal["green", "red"]
    count: int


# --- L2 class (layer L2-A) ---


@dataclass(frozen=True)
class BidPulled(Event):
    """A visible bid level was deleted without an offsetting print —
    the bidder withdrew (vs. being consumed by a market sell)."""

    price: float
    size_pulled: int
    position_in_book: int


@dataclass(frozen=True)
class OfferPulled(Event):
    """A visible offer was deleted without an offsetting print."""

    price: float
    size_pulled: int
    position_in_book: int


@dataclass(frozen=True)
class AbsorptionDetected(Event):
    """A price level is being repeatedly hit but not breaking — iceberg
    behavior, demand or supply larger than the visible size."""

    price: float
    side: Literal["bid", "ask"]
    cumulative_size_consumed: int
    visible_size_at_level: int
    refresh_count: int


@dataclass(frozen=True)
class SpreadEvent(Event):
    """Sharp widening or tightening of the bid-ask spread vs. its
    rolling-window average."""

    spread_now: float
    rolling_average_spread: float
    direction: Literal["widening", "tightening"]
    ratio: float


@dataclass(frozen=True)
class ImbalanceEvent(Event):
    """Top-K depth on one side outweighs the other by a configured ratio."""

    bid_total_size: int
    ask_total_size: int
    favored_side: Literal["bid", "ask"]
    ratio: float
    levels_summed: int


@dataclass(frozen=True)
class PrintCluster(Event):
    """N+ same-side prints within a rolling time window — aggressive
    buying or selling pressure."""

    side: Literal["buy", "sell"]
    print_count: int
    total_volume: int
    window_seconds: float


@dataclass(frozen=True)
class LargePrint(Event):
    """A single print exceeded the rolling-average size by a
    configured multiple."""

    price: float
    size: int
    rolling_average_size: float
    ratio: float
    aggressor_side: Literal["buy", "sell", "unknown"]


# --- gate chain (layer 3) ---


@dataclass(frozen=True)
class GateRejection(Event):
    """A single gate vetoed a policy decision. Fires once per veto.
    Forensic detail (numeric thresholds, observed values) lives in
    ``rejection_detail`` so post-replay analysis can calibrate gate
    thresholds against real decision streams."""

    gate_name: str
    rejection_reason: str
    rejection_detail: dict[str, Any] = field(default_factory=dict)
    original_decision: ExitDecision | None = None


@dataclass(frozen=True)
class GateChainResult(Event):
    """Full chain result for one policy decision. Fires once per
    decision the chain processed, regardless of accept/reject outcome.
    ``gate_results`` carries the per-gate breakdown so a calibration
    pass can answer "which gates *would* have rejected, in what order"
    without re-running the replay."""

    original_decision: ExitDecision | None
    final_decision: ExitDecision | None  # None ⇒ rejected, harness treats as hold
    gate_results: list[tuple[str, bool, str]] = field(default_factory=list)


# --- harness internal ---


@dataclass(frozen=True)
class ReplayTerminalTick(Event):
    """Harness-internal terminal tick emitted at the recorded exit timestamp.

    Not part of the public event taxonomy and not config-gated. Exists so
    replay-anchored policies (e.g. ActualPolicy) get a chance to fire when
    the recorded exit doesn't align with a bar boundary or a configured
    milestone — without it, a 29-second trade like ZENA's 2026-04-30
    closeout could not surface its decision through the policy.
    """


# --- production hook (Phase 11) ---


@dataclass(frozen=True)
class BarFinalizedEvent(Event):
    """A 1-min bar finalized for a tracked open position.

    Emitted from ``TradeManager.on_bar_update`` (production hook path,
    Phase 11) after the bot's existing bar-close logic runs. Carries the
    just-closed bar's OHLCV so an advisor can reason without re-pulling
    bars. Distinct from the spike harness's replay-driven event flow:
    this event represents live data arriving on the production hook.

    ``extra`` is an opt-in escape hatch so the bot can attach derived
    state (running R-multiple, indicator snapshot) without enlarging
    the typed surface. Advisors may ignore it.
    """

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
