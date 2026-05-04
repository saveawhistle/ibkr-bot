"""StallExitPolicy tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.exit_advisor.core.events import TimeOfDayMilestone
from bot.exit_advisor.decision.policy import StallExitPolicy, TradeState


def _ts(minute: int) -> datetime:
    return datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _state(current_price: float = 2.00) -> TradeState:
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=_ts(0),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,
        initial_scale_out=2.20,
        current_stop=1.90,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=current_price,
        current_price=current_price,
    )


def _evt(minute: int) -> TimeOfDayMilestone:
    return TimeOfDayMilestone(timestamp=_ts(minute), symbol="X", minutes_after_open=minute)


def test_constructor_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError, match="target_r"):
        StallExitPolicy(target_r=0, max_minutes=5)


def test_constructor_rejects_non_positive_minutes() -> None:
    with pytest.raises(ValueError, match="max_minutes"):
        StallExitPolicy(target_r=1.0, max_minutes=0)


def test_fires_when_target_unreached_after_window() -> None:
    """Trade hasn't moved past +0.3R; max_minutes=5 elapsed → stall fires."""
    policy = StallExitPolicy(target_r=1.0, max_minutes=5)
    state = _state(current_price=2.03)  # 0.3R
    early = policy.on_event(state, _evt(3))
    assert early is None  # not yet 5 min
    late = policy.on_event(state, _evt(5))
    assert late is not None
    assert late.action == "exit_full"
    assert "stall_exit" in late.reason


def test_inert_after_target_reached() -> None:
    """Once price hits target_r, the policy goes inert — even after
    max_minutes elapses, no stall exit. Other policies / trail logic
    take over the runner."""
    policy = StallExitPolicy(target_r=1.0, max_minutes=5)
    # Bar at minute 2: hits 1.5R, target reached, policy goes inert
    state_high = _state(current_price=2.15)
    assert policy.on_event(state_high, _evt(2)) is None
    # Bar at minute 6: pulled back to 0.5R — target was previously reached,
    # so stall stays inert.
    state_pullback = _state(current_price=2.05)
    assert policy.on_event(state_pullback, _evt(6)) is None


def test_does_not_fire_when_unprotected() -> None:
    policy = StallExitPolicy(target_r=1.0, max_minutes=5)
    state = _state(current_price=2.03)
    state.is_protected = False
    assert policy.on_event(state, _evt(10)) is None


def test_fires_only_once() -> None:
    policy = StallExitPolicy(target_r=1.0, max_minutes=5)
    state = _state(current_price=2.03)
    first = policy.on_event(state, _evt(5))
    assert first is not None
    second = policy.on_event(state, _evt(6))
    assert second is None  # latch held


def test_handles_degenerate_risk() -> None:
    policy = StallExitPolicy(target_r=1.0, max_minutes=5)
    state = _state(current_price=2.50)
    state.initial_stop = 2.00  # risk = 0
    assert policy.on_event(state, _evt(10)) is None
