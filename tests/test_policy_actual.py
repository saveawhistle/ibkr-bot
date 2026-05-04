"""ActualPolicy unit tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from bot.exit_advisor.core.events import (
    PositionProtected,
    ReplayTerminalTick,
    TimeOfDayMilestone,
)
from bot.exit_advisor.decision.policy import ActualPolicy, TradeState
from bot.exit_advisor.replay.replay_source import TradeReplayData


def _make_replay_data(exit_ts: datetime, exit_price: float = 2.16) -> TradeReplayData:
    return TradeReplayData(
        symbol="TST",
        trade_date=date(2026, 4, 30),
        bars=[],
        entry_event={},
        bracket_event={},
        order_events=[],
        exit_event={},
        recorded_pnl=-1.0,
        recorded_exit_price=exit_price,
        recorded_exit_timestamp=exit_ts,
    )


def _state() -> TradeState:
    return TradeState(
        symbol="TST",
        entry_price=2.18,
        entry_timestamp=datetime(2026, 4, 30, 15, 2, 57, tzinfo=UTC),
        current_position_size=119,
        initial_position_size=119,
        initial_stop=2.16,
        initial_scale_out=2.21,
        current_stop=2.16,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=2.18,
        current_price=2.18,
    )


def test_actual_policy_emits_exit_at_recorded_timestamp() -> None:
    exit_ts = datetime(2026, 4, 30, 15, 3, 26, tzinfo=UTC)
    rd = _make_replay_data(exit_ts)
    policy = ActualPolicy(rd)

    early = TimeOfDayMilestone(
        timestamp=exit_ts - timedelta(seconds=1),
        symbol="TST",
        minutes_after_open=5,
    )
    assert policy.on_event(_state(), early) is None

    on_time = ReplayTerminalTick(timestamp=exit_ts, symbol="TST")
    decision = policy.on_event(_state(), on_time)
    assert decision is not None
    assert decision.action == "exit_full"
    assert decision.fill_price == 2.16


def test_actual_policy_emits_only_once() -> None:
    exit_ts = datetime(2026, 4, 30, 15, 3, 26, tzinfo=UTC)
    rd = _make_replay_data(exit_ts)
    policy = ActualPolicy(rd)

    first = ReplayTerminalTick(timestamp=exit_ts, symbol="TST")
    second = ReplayTerminalTick(timestamp=exit_ts + timedelta(seconds=1), symbol="TST")

    assert policy.on_event(_state(), first) is not None
    # Subsequent events past the exit timestamp must not produce another decision.
    assert policy.on_event(_state(), second) is None


def test_actual_policy_returns_none_for_pre_exit_events() -> None:
    exit_ts = datetime(2026, 4, 30, 15, 3, 26, tzinfo=UTC)
    rd = _make_replay_data(exit_ts)
    policy = ActualPolicy(rd)

    pre_event = PositionProtected(
        timestamp=exit_ts - timedelta(minutes=5),
        symbol="TST",
        entry_price=2.18,
        initial_stop=2.16,
        initial_scale_out=2.21,
        position_size=119,
    )
    assert policy.on_event(_state(), pre_event) is None
