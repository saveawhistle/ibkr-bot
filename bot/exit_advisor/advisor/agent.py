"""Live LLM-powered exit advisor.

Implements :class:`bot.exit_advisor.core.types.ExitAdvisorHook` against
the Phase 11 hook surface. Combines:

* Per-position state tracking (entry timestamp, peak price, current
  price, scale-out hit) — the hook's :class:`PositionLike` doesn't
  carry these directly, so the agent maintains a small shadow
  (:class:`_LiveTradeState`) updated from the events it receives.
* Event buffering with significance triggering (:class:`EventBuffer`).
* LLM calls via :class:`AnthropicLLMClient` with cost cap enforcement
  (:class:`CostTracker`).
* Three mechanical baselines running in shadow
  (:class:`ShadowBaselines`).
* Three-way logging: actual LLM call, shadow baseline recommendations,
  failures.
* Self-disable when the LLM failure rate exceeds the configured
  threshold (after a minimum number of calls so the rate is
  meaningful).

The advisor is sync because the hook wrapper already runs each
notify in a worker thread; adding asyncio here would gain nothing.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from bot.exit_advisor.advisor.buffer import EventBuffer
from bot.exit_advisor.advisor.cost_tracker import CostTracker
from bot.exit_advisor.advisor.llm_client import AnthropicLLMClient
from bot.exit_advisor.advisor.prompts import (
    EXIT_ADVISOR_SYSTEM_PROMPT,
    EXIT_RECOMMENDATION_TOOL_SCHEMA,
)
from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines
from bot.exit_advisor.core.events import (
    BarFinalizedEvent,
    DrawdownFromPeak,
    Event,
    PositionProtected,
    RMultipleReached,
)
from bot.exit_advisor.core.types import (
    AdvisorResponse,
    PositionLike,
)
from bot.exit_advisor.decision.policy import TradeState

_log = structlog.get_logger("bot.exit_advisor.advisor.agent")


@dataclass
class _LiveTradeState:
    """Per-position tracking state the agent maintains in lieu of plumbing.

    The hook's :class:`PositionLike` carries the immutable fields
    (avg_price, stop_price, scale_out_price, scaled_out). The agent
    needs additional context to render a useful prompt and to feed the
    shadow baselines:

    * ``entry_timestamp`` — captured on ``on_position_protected`` (the
      ``PositionProtected`` event's wall-clock timestamp).
    * ``initial_stop`` / ``initial_scale_out`` — captured on
      ``on_position_protected`` so we can compute R-multiples even after
      the bot adjusts the working stop (post-scale-out trail, etc.).
    * ``peak_price`` — running maximum across received bar events.
    * ``current_price`` — last observed bar close (or last bar event
      price).
    * ``last_event_timestamp`` — wall-clock of the most recent event,
      used for time-in-trade derivations.
    * ``initial_position_size`` / ``current_position_size`` — captured
      on protection; current is reduced by partial fills.
    """

    symbol: str
    entry_price: float
    initial_stop: float
    initial_scale_out: float
    initial_position_size: int
    entry_timestamp: datetime

    peak_price: float = 0.0
    current_price: float = 0.0
    last_event_timestamp: datetime | None = None
    current_position_size: int = 0
    realized_pnl: float = 0.0
    scale_out_was_hit: bool = False

    def __post_init__(self) -> None:
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.current_position_size == 0:
            self.current_position_size = self.initial_position_size

    def update_from_event(self, event: Event, position: PositionLike) -> None:
        """Refresh tracking state from one event + the live position record."""
        if self.last_event_timestamp is None and (
            (event.timestamp.tzinfo is None) != (self.entry_timestamp.tzinfo is None)
        ):
            # First event: align entry_timestamp's tz-awareness to the event stream
            # so subsequent (event_ts - entry_ts) subtractions don't trip on a mix.
            # The live bot uses UTC; the test harness sometimes uses naive — both work.
            self.entry_timestamp = event.timestamp
        self.last_event_timestamp = event.timestamp
        if isinstance(event, BarFinalizedEvent):
            # Treat the bar's high as the running peak and close as current.
            if event.high > self.peak_price:
                self.peak_price = event.high
            self.current_price = event.close
        # Position fields are read-only for the agent, but we mirror
        # scale-out + remaining size so the prompt always reflects
        # current bot state regardless of which event class fired.
        self.scale_out_was_hit = bool(getattr(position, "scaled_out", False))
        self.current_position_size = int(getattr(position, "shares", self.current_position_size))

    def to_trade_state(
        self, current_stop: float, current_price_override: float | None = None
    ) -> TradeState:
        """Project this shadow into a :class:`TradeState` for the baselines."""
        cur_price = (
            current_price_override if current_price_override is not None else self.current_price
        )
        return TradeState(
            symbol=self.symbol,
            entry_price=self.entry_price,
            entry_timestamp=self.entry_timestamp,
            current_position_size=self.current_position_size,
            initial_position_size=self.initial_position_size,
            initial_stop=self.initial_stop,
            initial_scale_out=self.initial_scale_out,
            current_stop=current_stop,
            realized_pnl=self.realized_pnl,
            is_protected=True,
            peak_price=self.peak_price,
            current_price=cur_price,
        )

    def r_multiple(self, price: float) -> float:
        """R-multiple of ``price`` relative to entry, where 1R = entry - initial_stop."""
        risk_per_share = self.entry_price - self.initial_stop
        if risk_per_share <= 0.0:
            return 0.0
        return (price - self.entry_price) / risk_per_share


@dataclass
class _CallCounters:
    """LLM call accounting for the self-disable check."""

    total_calls: int = 0
    failed_calls: int = 0
    held_calls: int = 0
    actionable_calls: int = 0


@dataclass
class _PerTradeContext:
    """Bundle of per-position state. Lives from on_position_protected to on_position_closed."""

    trade_state: _LiveTradeState
    buffer: EventBuffer
    counters: _CallCounters = field(default_factory=_CallCounters)


class ExitAdvisor:
    """The live LLM-powered exit advisor.

    Implements the :class:`ExitAdvisorHook` protocol. The hook
    registry's invocation wrappers run each call inside a worker
    thread with its own timeout, so this class is entirely
    synchronous internally.

    Failure handling: every LLM call is wrapped by the client and
    returns an :class:`LLMCallResult` rather than raising. The agent
    counts failures and self-disables once the rate exceeds the
    configured threshold (after a minimum number of calls so the
    rate is meaningful). Once self-disabled, every subsequent
    ``on_event`` returns a deterministic skipped response for the
    rest of the session.
    """

    SELF_DISABLE_FAILURE_RATE = 0.5
    SELF_DISABLE_MIN_CALLS = 5

    def __init__(
        self,
        llm_client: AnthropicLLMClient,
        cost_tracker: CostTracker,
        event_buffer_factory: Any,
        shadow_baselines: ShadowBaselines,
        hook_acts: bool,
        self_disable_failure_rate: float = SELF_DISABLE_FAILURE_RATE,
        self_disable_min_calls: int = SELF_DISABLE_MIN_CALLS,
        notify_callback: Any = None,
    ) -> None:
        self._llm_client = llm_client
        self._cost_tracker = cost_tracker
        self._event_buffer_factory = event_buffer_factory
        self._shadow_baselines = shadow_baselines
        self._hook_acts = hook_acts
        self._self_disable_failure_rate = self_disable_failure_rate
        self._self_disable_min_calls = self_disable_min_calls
        self._notify_callback = notify_callback
        self._self_disabled = False
        self._contexts: dict[str, _PerTradeContext] = {}

    # ---------------- ExitAdvisorHook protocol ---------------- #

    def on_position_protected(self, position: PositionLike) -> None:
        """Initialise per-position state. Called once after entry + protection are confirmed."""
        entry_ts = datetime.now(UTC)
        ctx = _PerTradeContext(
            trade_state=_LiveTradeState(
                symbol=position.symbol,
                entry_price=position.avg_price,
                initial_stop=position.stop_price,
                initial_scale_out=position.scale_out_price,
                initial_position_size=int(position.shares),
                entry_timestamp=entry_ts,
            ),
            buffer=self._event_buffer_factory(),
        )
        self._contexts[position.symbol] = ctx
        self._shadow_baselines.reset_for_new_trade()
        _log.info(
            "advisor.position_protected",
            symbol=position.symbol,
            entry_price=position.avg_price,
            initial_stop=position.stop_price,
            initial_scale_out=position.scale_out_price,
            position_size=int(position.shares),
        )

    def on_event(self, position: PositionLike, event: Event) -> AdvisorResponse:
        """Main advisor entrypoint. Returns a typed AdvisorResponse; never raises.

        Flow:
          1. If self-disabled or hard-cost-capped → deterministic skipped.
          2. Update per-position tracking state.
          3. Run shadow baselines (always logged regardless of trigger).
          4. Buffer event; consult buffer for trigger decision.
          5. If not triggered → skipped response.
          6. If triggered → call LLM; on success return actionable/held;
             on failure return skipped + bump counters.
        """
        ctx = self._contexts.get(position.symbol)
        if ctx is None:
            # First event for this symbol arrived before on_position_protected.
            # Construct context lazily so we don't drop the event entirely.
            self.on_position_protected(position)
            ctx = self._contexts[position.symbol]

        if self._self_disabled:
            _log.info(
                "advisor.skipped",
                symbol=position.symbol,
                event_type=type(event).__name__,
                skip_reason="self_disabled",
            )
            return AdvisorResponse(
                recommendation=None,
                evaluation_performed=False,
                reasoning="advisor self-disabled for session due to failure rate",
            )

        if self._cost_tracker.is_hard_capped():
            _log.info(
                "advisor.skipped",
                symbol=position.symbol,
                event_type=type(event).__name__,
                skip_reason="cost_cap_reached",
            )
            return AdvisorResponse(
                recommendation=None,
                evaluation_performed=False,
                reasoning="cost_cap_reached",
            )

        ctx.trade_state.update_from_event(event, position)

        # Run shadow baselines on every event — log results unconditionally.
        baseline_results = self._shadow_baselines.consume_event(
            event,
            ctx.trade_state.to_trade_state(current_stop=position.stop_price),
        )
        for name, baseline_decision in baseline_results.items():
            self._log_baseline(position.symbol, name, event, baseline_decision)

        buffer_decision = ctx.buffer.consume(event, event.timestamp)
        if not buffer_decision.trigger:
            _log.info(
                "advisor.skipped",
                symbol=position.symbol,
                event_type=type(event).__name__,
                skip_reason=buffer_decision.skip_reason or "non_significant",
                pending_count=ctx.buffer.pending_count(),
            )
            return AdvisorResponse(
                recommendation=None,
                evaluation_performed=False,
                reasoning=buffer_decision.skip_reason or "non_significant",
            )

        # Triggered → call LLM.
        user_message = _render_user_message(
            ctx.trade_state,
            position,
            triggering_event=event,
            buffered_events=buffer_decision.buffered_events,
        )
        result = self._llm_client.call(
            EXIT_ADVISOR_SYSTEM_PROMPT,
            user_message,
            EXIT_RECOMMENDATION_TOOL_SCHEMA,
        )
        ctx.counters.total_calls += 1
        self._cost_tracker.record_cost(result.cost_usd)

        if not result.success:
            ctx.counters.failed_calls += 1
            _log.warning(
                "advisor.failure",
                symbol=position.symbol,
                event_type=type(event).__name__,
                failure_reason=result.failure_reason,
                duration_seconds=round(result.duration_seconds, 3),
                cost_usd=round(result.cost_usd, 6),
            )
            self._check_self_disable()
            return AdvisorResponse(
                recommendation=None,
                evaluation_performed=False,
                reasoning=f"llm_call_failed: {result.failure_reason}",
            )

        recommendation = result.recommendation
        assert recommendation is not None  # success path narrows for mypy
        if recommendation.action == "hold":
            ctx.counters.held_calls += 1
            _log.info(
                "advisor.call",
                symbol=position.symbol,
                event_type=type(event).__name__,
                buffered_event_count=len(buffer_decision.buffered_events),
                decision="hold",
                confidence=recommendation.confidence,
                reasoning=recommendation.reason,
                cost_usd=round(result.cost_usd, 6),
                duration_seconds=round(result.duration_seconds, 3),
            )
            return AdvisorResponse(
                recommendation=None,
                evaluation_performed=True,
                reasoning=recommendation.reason,
            )

        ctx.counters.actionable_calls += 1
        _log.info(
            "advisor.call",
            symbol=position.symbol,
            event_type=type(event).__name__,
            buffered_event_count=len(buffer_decision.buffered_events),
            decision="actionable",
            action=recommendation.action,
            confidence=recommendation.confidence,
            reasoning=recommendation.reason,
            cost_usd=round(result.cost_usd, 6),
            duration_seconds=round(result.duration_seconds, 3),
            hook_acts=self._hook_acts,
        )
        return AdvisorResponse(
            recommendation=recommendation,
            evaluation_performed=True,
            reasoning=recommendation.reason,
        )

    def on_position_closed(self, position: PositionLike, final_pnl: float) -> None:
        """Log a per-trade summary and tear down per-position state."""
        ctx = self._contexts.pop(position.symbol, None)
        if ctx is None:
            _log.info(
                "advisor.position_closed_no_context",
                symbol=position.symbol,
                final_pnl=round(final_pnl, 2),
            )
            return
        _log.info(
            "advisor.position_closed",
            symbol=position.symbol,
            final_pnl=round(final_pnl, 2),
            total_calls=ctx.counters.total_calls,
            actionable_calls=ctx.counters.actionable_calls,
            held_calls=ctx.counters.held_calls,
            failed_calls=ctx.counters.failed_calls,
            session_cost_usd=round(self._cost_tracker.session_cost_usd(), 4),
            self_disabled=self._self_disabled,
        )

    # ---------------- introspection ---------------- #

    def is_self_disabled(self) -> bool:
        """True iff the failure-rate self-disable has tripped this session."""
        return self._self_disabled

    # ---------------- internals ---------------- #

    def _check_self_disable(self) -> None:
        """Trip the session-wide kill switch if failure rate is too high."""
        # Aggregate counts across every active and recently-closed context.
        # We track on a per-trade basis, but the disable threshold is global —
        # one bad LLM session affects every position.
        total = 0
        failed = 0
        for ctx in self._contexts.values():
            total += ctx.counters.total_calls
            failed += ctx.counters.failed_calls
        if total < self._self_disable_min_calls:
            return
        rate = failed / total
        if rate > self._self_disable_failure_rate:
            self._self_disabled = True
            message = (
                f"exit-advisor self-disabled for session: "
                f"failure rate {rate:.0%} > threshold "
                f"{self._self_disable_failure_rate:.0%} "
                f"after {total} calls ({failed} failed)"
            )
            _log.error(
                "advisor.self_disabled",
                failure_rate=round(rate, 3),
                threshold=self._self_disable_failure_rate,
                total_calls=total,
                failed_calls=failed,
            )
            self._notify(message)

    def _notify(self, message: str) -> None:
        """Best-effort operator notification. Never raise."""
        if self._notify_callback is None:
            return
        with contextlib.suppress(Exception):
            self._notify_callback(message)

    @staticmethod
    def _log_baseline(
        symbol: str,
        baseline_name: str,
        event: Event,
        decision: Any,
    ) -> None:
        """Emit one log line per baseline per event so a session JSONL has full shadow detail."""
        if decision is None:
            _log.info(
                "shadow_baseline.recommendation",
                source=f"mechanical_{baseline_name}",
                symbol=symbol,
                event_type=type(event).__name__,
                recommendation=None,
            )
            return
        _log.info(
            "shadow_baseline.recommendation",
            source=f"mechanical_{baseline_name}",
            symbol=symbol,
            event_type=type(event).__name__,
            action=decision.action,
            partial_pct=decision.partial_pct,
            new_stop_price=decision.new_stop_price,
            confidence=decision.confidence,
            reason=decision.reason,
        )


def _render_user_message(
    state: _LiveTradeState,
    position: PositionLike,
    *,
    triggering_event: Event,
    buffered_events: list[Event],
) -> str:
    """Build the user-message payload for the LLM call as a JSON-rendered string.

    The system prompt declares the schema; this function fills it. Time
    fields are rendered as ISO-8601 UTC. R-multiples are computed from
    captured initial values so post-scale-out moves don't reset them.
    """
    last_event_ts = state.last_event_timestamp or state.entry_timestamp
    time_in_trade = max(0.0, (last_event_ts - state.entry_timestamp).total_seconds())
    payload: dict[str, Any] = {
        "trade_state": {
            "symbol": state.symbol,
            "entry_price": round(state.entry_price, 4),
            "entry_timestamp": state.entry_timestamp.isoformat(),
            "current_price": round(state.current_price, 4),
            "current_timestamp": last_event_ts.isoformat(),
            "position_size": state.current_position_size,
            "initial_stop": round(state.initial_stop, 4),
            "current_stop": round(float(position.stop_price), 4),
            "scale_out_price": round(state.initial_scale_out, 4),
            "peak_price": round(state.peak_price, 4),
            "peak_r_multiple": round(state.r_multiple(state.peak_price), 3),
            "current_r_multiple": round(state.r_multiple(state.current_price), 3),
            "drawdown_from_peak_r": round(
                state.r_multiple(state.peak_price) - state.r_multiple(state.current_price), 3
            ),
            "time_in_trade_seconds": round(time_in_trade, 1),
            "scale_out_was_hit": state.scale_out_was_hit,
        },
        "triggering_event": _serialize_event(triggering_event),
        "buffered_events": [
            _serialize_event(ev) for ev in buffered_events if ev is not triggering_event
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _serialize_event(event: Event) -> dict[str, Any]:
    """Compact dict view of an event suitable for the LLM prompt."""
    base: dict[str, Any] = {
        "type": type(event).__name__,
        "timestamp": event.timestamp.isoformat(),
        "symbol": event.symbol,
    }
    # Pull the dataclass-specific fields without depending on dataclasses.asdict
    # (asdict deep-copies dict fields like BarFinalizedEvent.extra; we want shallow).
    for slot in event.__dataclass_fields__:
        if slot in ("timestamp", "symbol"):
            continue
        value = getattr(event, slot)
        if isinstance(value, datetime):
            base[slot] = value.isoformat()
        else:
            base[slot] = value
    # Trim BarFinalizedEvent.extra to keep the prompt size predictable.
    if isinstance(event, BarFinalizedEvent):
        extra = base.get("extra")
        if isinstance(extra, dict) and len(extra) > 8:
            base["extra"] = dict(list(extra.items())[:8])
    return base


# Re-exports referenced by tests so they don't have to import from core/events directly.
__all__ = [
    "BarFinalizedEvent",
    "DrawdownFromPeak",
    "ExitAdvisor",
    "PositionProtected",
    "RMultipleReached",
    "_LiveTradeState",
]
