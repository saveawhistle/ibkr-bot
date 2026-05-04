"""MaxHoldTimePolicy tests — fires at the threshold, returns None
before, has the right reason prefix for the gate-chain whitelist."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.exit_advisor.core.events import TimeOfDayMilestone
from bot.exit_advisor.decision.policy import ExitDecision, MaxHoldTimePolicy, TradeState


def _state(entry_minutes_ago: int = 0, is_protected: bool = True) -> TradeState:
    entry_ts = datetime(2026, 4, 30, 13, 30, tzinfo=UTC)
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=entry_ts,
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,
        initial_scale_out=2.20,
        current_stop=1.90,
        realized_pnl=0.0,
        is_protected=is_protected,
        peak_price=2.00,
        current_price=2.00,
    )


def test_returns_none_before_threshold() -> None:
    policy = MaxHoldTimePolicy(max_hold_minutes=60)
    state = _state()
    event_ts = state.entry_timestamp + timedelta(minutes=30)  # halfway
    event = TimeOfDayMilestone(timestamp=event_ts, symbol="X", minutes_after_open=30)
    assert policy.on_event(state, event) is None


def test_fires_force_exit_at_threshold() -> None:
    policy = MaxHoldTimePolicy(max_hold_minutes=60)
    state = _state()
    event_ts = state.entry_timestamp + timedelta(minutes=60)
    event = TimeOfDayMilestone(timestamp=event_ts, symbol="X", minutes_after_open=60)
    decision = policy.on_event(state, event)
    assert decision is not None
    assert decision.action == "exit_full"
    assert decision.confidence == 1.0


def test_reason_prefix_matches_guardrail_whitelist() -> None:
    """Reason string starts with ``max_hold_time_reached_`` so the
    matching guardrail can identify the force-exit and let it through."""
    policy = MaxHoldTimePolicy(max_hold_minutes=45)
    state = _state()
    event_ts = state.entry_timestamp + timedelta(minutes=50)
    event = TimeOfDayMilestone(timestamp=event_ts, symbol="X", minutes_after_open=50)
    decision = policy.on_event(state, event)
    assert decision is not None
    assert decision.reason.startswith("max_hold_time_reached_")
    assert "45" in decision.reason


def test_fires_only_once() -> None:
    """Subsequent events past the threshold don't re-fire — once-per-trade."""
    policy = MaxHoldTimePolicy(max_hold_minutes=60)
    state = _state()
    e1 = TimeOfDayMilestone(
        timestamp=state.entry_timestamp + timedelta(minutes=60), symbol="X", minutes_after_open=60
    )
    e2 = TimeOfDayMilestone(
        timestamp=state.entry_timestamp + timedelta(minutes=61), symbol="X", minutes_after_open=61
    )
    assert policy.on_event(state, e1) is not None
    assert policy.on_event(state, e2) is None


def test_does_not_fire_when_not_protected() -> None:
    """Sacred-ground rule: no decisions on unprotected positions, even
    a force-exit-on-time."""
    policy = MaxHoldTimePolicy(max_hold_minutes=60)
    state = _state(is_protected=False)
    event_ts = state.entry_timestamp + timedelta(minutes=120)
    event = TimeOfDayMilestone(timestamp=event_ts, symbol="X", minutes_after_open=120)
    assert policy.on_event(state, event) is None


def test_force_exit_priority_over_advisor_in_harness() -> None:
    """When both the advisor policy and MaxHoldTimePolicy return a
    decision on the same event, force-exit must win. This test
    exercises the harness's priority resolution directly."""
    from bot.exit_advisor.replay.harness import TradeReplayHarness

    advisor_decision = ExitDecision(action="tighten_stop", new_stop_price=2.05)
    forced_decision = ExitDecision(
        action="exit_full",
        reason="max_hold_time_reached_60_minutes",
    )
    resolved = TradeReplayHarness._resolve_priority([advisor_decision, forced_decision])
    assert resolved is forced_decision

    # Order swap shouldn't matter — force-exit is identified by reason prefix.
    resolved = TradeReplayHarness._resolve_priority([forced_decision, advisor_decision])
    assert resolved is forced_decision
