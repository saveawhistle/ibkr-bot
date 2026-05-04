"""OracleExitPolicy tests — the foresight-cheating ceiling baseline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from bot.exit_advisor.core.events import TimeOfDayMilestone
from bot.exit_advisor.decision.policy import OracleExitPolicy, TradeState
from bot.exit_advisor.replay.replay_source import Bar, TradeReplayData


def _ts(minute: int) -> datetime:
    return datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _bar(minute: int, close: float) -> Bar:
    return Bar(
        timestamp=_ts(minute),
        open=close - 0.01,
        high=close + 0.01,
        low=close - 0.02,
        close=close,
        volume=1000,
    )


def _replay_data(bars: list[Bar]) -> TradeReplayData:
    entry_ts = _ts(0)
    return TradeReplayData(
        symbol="X",
        trade_date=date(2026, 4, 30),
        bars=bars,
        entry_event={"timestamp": entry_ts.isoformat().replace("+00:00", "Z")},
        bracket_event={"stop_price": 1.95, "shares": 100, "entry_price": 2.00},
        exit_event={},
        recorded_pnl=0.0,
        recorded_exit_price=2.00,
        recorded_exit_timestamp=entry_ts + timedelta(minutes=10),
    )


def _state() -> TradeState:
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=_ts(0),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.95,
        initial_scale_out=2.20,
        current_stop=1.95,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=2.00,
        current_price=2.00,
    )


def test_oracle_picks_highest_close_bar() -> None:
    """Three bars with closes [2.05, 2.20, 2.10]. Oracle should target
    bar 1 (the 2.20 close). Optimal exit timestamp = bar 1's close
    (= bar.timestamp + 1min)."""
    bars = [_bar(0, 2.05), _bar(1, 2.20), _bar(2, 2.10)]
    rd = _replay_data(bars)
    oracle = OracleExitPolicy(rd)
    assert oracle._optimal_exit_timestamp == _ts(1) + timedelta(minutes=1)
    assert oracle._optimal_close == 2.20


def test_oracle_tie_breaks_on_earliest_timestamp() -> None:
    """Two bars share max close (2.15). Earliest wins so the policy
    doesn't reward 'hold longer at the same peak'."""
    bars = [_bar(0, 2.10), _bar(1, 2.15), _bar(2, 2.15), _bar(3, 2.05)]
    rd = _replay_data(bars)
    oracle = OracleExitPolicy(rd)
    assert oracle._optimal_exit_timestamp == _ts(1) + timedelta(minutes=1)


def test_oracle_returns_none_before_protection() -> None:
    bars = [_bar(0, 2.20)]
    rd = _replay_data(bars)
    oracle = OracleExitPolicy(rd)
    state = _state()
    state.is_protected = False
    event = TimeOfDayMilestone(timestamp=_ts(5), symbol="X", minutes_after_open=5)
    assert oracle.on_event(state, event) is None


def test_oracle_emits_exit_full_exactly_once() -> None:
    bars = [_bar(0, 2.20)]
    rd = _replay_data(bars)
    oracle = OracleExitPolicy(rd)
    state = _state()
    e1 = TimeOfDayMilestone(timestamp=_ts(2), symbol="X", minutes_after_open=2)
    e2 = TimeOfDayMilestone(timestamp=_ts(3), symbol="X", minutes_after_open=3)
    decision = oracle.on_event(state, e1)
    assert decision is not None
    assert decision.action == "exit_full"
    assert decision.reason == "oracle_exit_at_optimal_bar"
    assert decision.fill_price == 2.20
    # Subsequent events past the optimal point: no further decisions.
    assert oracle.on_event(state, e2) is None


def test_oracle_returns_none_before_optimal_timestamp() -> None:
    bars = [_bar(5, 2.20)]  # optimal is bar at minute 5, fires at minute 6
    rd = _replay_data(bars)
    oracle = OracleExitPolicy(rd)
    state = _state()
    early = TimeOfDayMilestone(timestamp=_ts(2), symbol="X", minutes_after_open=2)
    assert oracle.on_event(state, early) is None


def test_oracle_handles_empty_trade_window() -> None:
    """No bars in the trade window → no optimal point → policy never fires."""
    rd = _replay_data([])
    oracle = OracleExitPolicy(rd)
    assert oracle._optimal_exit_timestamp is None
    state = _state()
    event = TimeOfDayMilestone(timestamp=_ts(5), symbol="X", minutes_after_open=5)
    assert oracle.on_event(state, event) is None


def test_oracle_constructor_does_not_mutate_replay_data() -> None:
    """Sanity: instantiating an Oracle shouldn't modify the input."""
    bars = [_bar(0, 2.10), _bar(1, 2.20)]
    rd = _replay_data(bars)
    bars_before = list(rd.bars)
    OracleExitPolicy(rd)
    assert rd.bars == bars_before


# Avoid an unused-import-only pytest; if this file is collected with no
# tests by mistake, fail loudly.
def test_module_collected() -> None:
    assert pytest.__version__
