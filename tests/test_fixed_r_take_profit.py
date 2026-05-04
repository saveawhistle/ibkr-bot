"""FixedRTakeProfit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.exit_advisor.core.events import TimeOfDayMilestone
from bot.exit_advisor.decision.policy import FixedRTakeProfit, TradeState


def _ts(minute: int) -> datetime:
    return datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _state(current_price: float = 2.00) -> TradeState:
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=_ts(0),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,  # risk = 0.10
        initial_scale_out=2.20,
        current_stop=1.90,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=current_price,
        current_price=current_price,
    )


def _evt() -> TimeOfDayMilestone:
    return TimeOfDayMilestone(timestamp=_ts(1), symbol="X", minutes_after_open=1)


def test_constructor_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        FixedRTakeProfit(target_r=0)


def test_constructor_rejects_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        FixedRTakeProfit(target_r=-1.0)


def test_fires_at_target_r() -> None:
    policy = FixedRTakeProfit(target_r=1.0)
    # Price 2.10 = +1R exactly.
    decision = policy.on_event(_state(current_price=2.10), _evt())
    assert decision is not None
    assert decision.action == "exit_full"
    assert "fixed_r_target_1.0_reached" in decision.reason


def test_does_not_fire_below_target() -> None:
    policy = FixedRTakeProfit(target_r=1.0)
    # Price 2.05 = +0.5R, below target.
    assert policy.on_event(_state(current_price=2.05), _evt()) is None


def test_fires_only_once() -> None:
    policy = FixedRTakeProfit(target_r=1.0)
    state = _state(current_price=2.15)  # +1.5R
    first = policy.on_event(state, _evt())
    assert first is not None
    assert policy.on_event(state, _evt()) is None  # latch held


def test_does_not_fire_when_unprotected() -> None:
    policy = FixedRTakeProfit(target_r=1.0)
    state = _state(current_price=2.20)
    state.is_protected = False
    assert policy.on_event(state, _evt()) is None


def test_handles_degenerate_risk() -> None:
    """If entry <= initial_stop (impossible in practice but defensive),
    the policy refuses to compute and returns None."""
    policy = FixedRTakeProfit(target_r=1.0)
    state = _state(current_price=2.50)
    state.initial_stop = 2.00  # risk = 0
    assert policy.on_event(state, _evt()) is None
