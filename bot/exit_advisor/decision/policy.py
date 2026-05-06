"""ExitPolicy protocol + reference implementations.

A policy receives events and optionally returns an :class:`ExitDecision`.
The harness only invokes ``on_event`` after the position is confirmed
protected (entry filled + initial bracket children working). This is the
sacred-ground precondition: the advisor never operates on an unprotected
position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Protocol

from bot.exit_advisor.core.events import Event

if TYPE_CHECKING:
    from bot.exit_advisor.replay.replay_source import TradeReplayData


@dataclass
class TradeState:
    """Live state the policy sees on each invocation.

    ``is_protected`` is False until the harness emits ``PositionProtected``.
    The harness gates ``policy.on_event`` on this flag, so policies need
    not check it themselves — but the field is exposed for visibility.
    """

    symbol: str
    entry_price: float
    entry_timestamp: datetime
    current_position_size: int
    initial_position_size: int
    initial_stop: float
    initial_scale_out: float
    current_stop: float
    realized_pnl: float
    is_protected: bool
    peak_price: float
    current_price: float


@dataclass(frozen=True)
class ExitDecision:
    """A policy's recommendation for the current event.

    ``fill_price`` is an oracle-only override used by replay-anchored
    policies (e.g. :class:`ActualPolicy`) that already know the recorded
    exit price. Production policies leave it ``None`` and let the harness
    pick the appropriate level (bar close on full exits, stop level on
    stop fills). It is intentionally absent from the spec but required to
    make ActualPolicy reproduce the recorded P&L when bar granularity
    cannot recover the precise exit price.
    """

    action: Literal["hold", "exit_full", "exit_partial", "tighten_stop"]
    partial_pct: float = 0.0
    new_stop_price: float | None = None
    confidence: float = 1.0
    reason: str = ""
    fill_price: float | None = None


class ExitPolicy(Protocol):
    def on_event(
        self, trade_state: TradeState, event: Event
    ) -> ExitDecision | None:  # pragma: no cover - protocol
        ...


class ActualPolicy:
    """Replays the bot's recorded trade outcome.

    Used as the harness's correctness oracle: running the harness with
    ActualPolicy on a closed trade should reproduce the recorded P&L
    within $0.01. Any larger delta means the harness's bookkeeping has
    drifted from the bot's.
    """

    def __init__(self, replay_data: TradeReplayData) -> None:
        self.replay_data = replay_data
        self._exit_emitted = False

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if self._exit_emitted:
            return None
        if event.timestamp >= self.replay_data.recorded_exit_timestamp:
            self._exit_emitted = True
            return ExitDecision(
                action="exit_full",
                confidence=1.0,
                reason="ActualPolicy: matches recorded exit",
                fill_price=self.replay_data.recorded_exit_price,
            )
        return None


class OracleExitPolicy:
    """Theoretical upper bound — exits at the highest-close bar in the
    trade window. Requires foresight (peeks at the full bar sequence at
    construction), so it's not achievable in real trading; it exists as
    the analytical ceiling that other policies are measured against.

    Tie-break on identical maxes: earliest timestamp wins. Don't reward
    holding longer at the same peak.
    """

    name = "oracle_exit"

    def __init__(self, replay_data: TradeReplayData) -> None:
        self._optimal_exit_timestamp = self._find_optimal_exit(replay_data)
        self._optimal_close = self._find_optimal_close(replay_data)
        self._exit_emitted = False

    @staticmethod
    def _find_optimal_exit(replay_data: TradeReplayData) -> datetime | None:
        """Return the bar-close timestamp of the highest-close bar in
        the trade window. None if the trade window has no bars."""
        if not replay_data.bars:
            return None
        best_idx = 0
        best_close = replay_data.bars[0].close
        for i, bar in enumerate(replay_data.bars[1:], start=1):
            if bar.close > best_close:
                best_close = bar.close
                best_idx = i
        # Bar-close timestamp = bar_time + 1 minute, matching the harness's
        # event-emission convention.
        bar = replay_data.bars[best_idx]
        return bar.timestamp + timedelta(minutes=1)

    @staticmethod
    def _find_optimal_close(replay_data: TradeReplayData) -> float | None:
        if not replay_data.bars:
            return None
        return max(b.close for b in replay_data.bars)

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if self._exit_emitted or not trade_state.is_protected:
            return None
        if self._optimal_exit_timestamp is None:
            return None
        if event.timestamp >= self._optimal_exit_timestamp:
            self._exit_emitted = True
            return ExitDecision(
                action="exit_full",
                confidence=1.0,
                reason="oracle_exit_at_optimal_bar",
                fill_price=self._optimal_close,
            )
        return None


class MechanicalTrailPolicy:
    """Parameterized trailing stop. Recommends ``tighten_stop`` decisions
    as the running peak rises; the harness's stop-fill simulation
    handles the actual exit when a bar's low pierces the trail.

    Provide exactly one of ``trail_abs`` (dollar offset) or ``trail_pct``
    (fractional offset). The policy is idempotent across events: it
    only proposes when the computed trail would actually tighten the
    current stop. Self-throttling — once accepted, ``state.current_stop``
    rises and the next event's proposal must beat it.
    """

    name = "mechanical_trail"

    def __init__(
        self,
        trail_abs: float | None = None,
        trail_pct: float | None = None,
    ) -> None:
        if (trail_abs is None) == (trail_pct is None):
            raise ValueError("Provide exactly one of trail_abs or trail_pct")
        if trail_abs is not None and trail_abs <= 0:
            raise ValueError("trail_abs must be positive")
        if trail_pct is not None and not 0 < trail_pct < 1:
            raise ValueError("trail_pct must be in (0, 1)")
        self.trail_abs = trail_abs
        self.trail_pct = trail_pct

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if not trade_state.is_protected:
            return None
        peak = trade_state.peak_price
        if self.trail_abs is not None:
            proposed_stop = peak - self.trail_abs
        else:
            assert self.trail_pct is not None
            proposed_stop = peak * (1 - self.trail_pct)
        if proposed_stop <= trade_state.current_stop:
            return None
        param_str = (
            f"abs_{self.trail_abs}" if self.trail_abs is not None else f"pct_{self.trail_pct}"
        )
        return ExitDecision(
            action="tighten_stop",
            new_stop_price=proposed_stop,
            confidence=1.0,
            reason=f"trail_ratchet_{param_str}",
        )


class FixedRTakeProfit:
    """Exits fully when the trade reaches a target R-multiple. Tests
    the hypothesis that simple fixed-R rules might outperform more
    complex exit logic on certain trade types — addresses failure
    mode 1 (breakouts that don't reach 2:1) when ``target_r < 2``.
    """

    name = "fixed_r_take_profit"

    def __init__(self, target_r: float) -> None:
        if target_r <= 0:
            raise ValueError("target_r must be positive")
        self.target_r = target_r
        self._exit_emitted = False

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if self._exit_emitted or not trade_state.is_protected:
            return None
        risk_per_share = trade_state.entry_price - trade_state.initial_stop
        if risk_per_share <= 0:
            return None
        current_r = (trade_state.current_price - trade_state.entry_price) / risk_per_share
        if current_r >= self.target_r:
            self._exit_emitted = True
            return ExitDecision(
                action="exit_full",
                confidence=1.0,
                reason=f"fixed_r_target_{self.target_r}_reached",
            )
        return None


class StallExitPolicy:
    """Exits the trade if it hasn't reached ``target_r`` within
    ``max_minutes``. Becomes inert once the target IS reached — at
    that point other policies / trail logic / runner management
    take over. Directly addresses failure mode 1: breakouts that
    don't reach 2:1 within a reasonable timeframe.
    """

    name = "stall_exit"

    def __init__(self, target_r: float, max_minutes: int) -> None:
        if target_r <= 0:
            raise ValueError("target_r must be positive")
        if max_minutes <= 0:
            raise ValueError("max_minutes must be positive")
        self.target_r = target_r
        self.max_minutes = max_minutes
        self._exit_emitted = False
        self._target_reached = False

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if self._exit_emitted or not trade_state.is_protected:
            return None

        risk_per_share = trade_state.entry_price - trade_state.initial_stop
        if risk_per_share <= 0:
            return None

        current_r = (trade_state.current_price - trade_state.entry_price) / risk_per_share
        if current_r >= self.target_r:
            self._target_reached = True
        if self._target_reached:
            return None  # Inert after target reached — let other policies manage.

        time_in_trade = event.timestamp - trade_state.entry_timestamp
        if time_in_trade >= timedelta(minutes=self.max_minutes):
            self._exit_emitted = True
            return ExitDecision(
                action="exit_full",
                confidence=1.0,
                reason=(f"stall_exit_target_{self.target_r}_not_reached_in_{self.max_minutes}m"),
            )
        return None


class MaxHoldTimePolicy:
    """Built-in policy: force-exit when the trade reaches the configured
    maximum hold time. Always added to the harness's policy list when
    layer-3 gates are enabled; runs alongside whatever advisor policy
    the operator passed in.

    Priority resolution in the harness: when both this and the advisor
    policy return a decision on the same event, the force-exit from
    here wins. This is the deliberate ordering — the bot's
    risk-of-time-decay is a hard constraint, not a recommendation.

    The reason string is prefixed ``max_hold_time_reached_`` so the
    matching ``MaxHoldTimeGuardrail`` can identify the force-exit and
    let it through unconditionally.
    """

    name = "max_hold_time_policy"

    def __init__(self, max_hold_minutes: int) -> None:
        self.max_hold_minutes = max_hold_minutes
        self._fired = False

    def on_event(self, trade_state: TradeState, event: Event) -> ExitDecision | None:
        if self._fired or not trade_state.is_protected:
            return None
        time_in_trade = event.timestamp - trade_state.entry_timestamp
        if time_in_trade >= timedelta(minutes=self.max_hold_minutes):
            self._fired = True
            return ExitDecision(
                action="exit_full",
                confidence=1.0,
                reason=f"max_hold_time_reached_{self.max_hold_minutes}_minutes",
            )
        return None
