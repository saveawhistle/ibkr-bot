"""Aggregation tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from bot.exit_advisor.analysis.aggregation import (
    TradeOutcome,
    aggregate_results,
    aggregate_subset_by_trades,
)
from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef


def _ref(trade_id: str) -> ClosedTradeRef:
    return ClosedTradeRef(
        symbol="X",
        trade_date=date(2026, 4, 30),
        trade_id=trade_id,
        entry_timestamp=datetime(2026, 4, 30, 13, 30, tzinfo=UTC),
        exit_timestamp=datetime(2026, 4, 30, 14, 0, tzinfo=UTC),
        session_log_path=Path("logs/x.jsonl"),
    )


def _outcome(
    trade_id: str,
    policy_name: str,
    params: dict,  # type: ignore[type-arg]
    gates_enabled: bool,
    final_pnl: float,
    fired: bool = True,
    actual_pnl: float = 0.0,
    oracle_pnl: float = 0.0,
) -> TradeOutcome:
    return TradeOutcome(
        trade_ref=_ref(trade_id),
        policy_name=policy_name,
        policy_params=params,
        gates_enabled=gates_enabled,
        final_pnl=final_pnl,
        fired=fired,
        decisions_proposed=1 if fired else 0,
        decisions_accepted=1 if fired else 0,
        decisions_rejected=0,
        open_at_end_of_replay=not fired,
        actual_pnl=actual_pnl,
        oracle_pnl=oracle_pnl,
    )


def test_groups_by_policy_params_and_gates() -> None:
    outcomes = [
        _outcome("t1", "FixedRTakeProfit", {"target_r": 1.0}, True, 5.0),
        _outcome("t2", "FixedRTakeProfit", {"target_r": 1.0}, True, 7.0),
        _outcome("t1", "FixedRTakeProfit", {"target_r": 1.5}, True, 3.0),
        _outcome("t1", "FixedRTakeProfit", {"target_r": 1.0}, False, 6.0),
    ]
    aggs = aggregate_results(outcomes)
    # Three distinct groups:
    assert len(aggs) == 3
    # Verify the (target_r=1.0, gates_enabled=True) group rolled up correctly.
    found = [a for a in aggs.values() if a.policy_params == {"target_r": 1.0} and a.gates_enabled]
    assert len(found) == 1
    agg = found[0]
    assert agg.trades_total == 2
    assert agg.trades_fired == 2
    assert agg.total_pnl == 12.0
    assert agg.mean_pnl_when_fired == 6.0


def test_mean_excludes_non_firing_trades() -> None:
    """Mark-to-market trades contribute to total_pnl but NOT to
    mean_pnl_when_fired — that statistic describes the policy's
    behavior when it actually engaged."""
    outcomes = [
        _outcome("t1", "FixedR", {"r": 1.0}, True, 5.0, fired=True),
        _outcome("t2", "FixedR", {"r": 1.0}, True, -2.0, fired=False),  # MTM
    ]
    aggs = aggregate_results(outcomes)
    agg = next(iter(aggs.values()))
    assert agg.trades_total == 2
    assert agg.trades_fired == 1
    assert agg.trades_never_fired == 1
    assert agg.total_pnl == 3.0  # both trades contribute
    assert agg.mean_pnl_when_fired == 5.0  # only fired trades contribute


def test_delta_vs_actual_and_oracle_computed() -> None:
    outcomes = [
        _outcome("t1", "FixedR", {"r": 1.0}, True, 5.0, fired=True, actual_pnl=3.0, oracle_pnl=8.0),
        _outcome(
            "t2", "FixedR", {"r": 1.0}, True, 7.0, fired=True, actual_pnl=2.0, oracle_pnl=10.0
        ),
    ]
    aggs = aggregate_results(outcomes)
    agg = next(iter(aggs.values()))
    assert agg.total_pnl == 12.0
    assert agg.total_actual_pnl == 5.0
    assert agg.total_oracle_pnl == 18.0
    assert agg.delta_vs_actual == 7.0
    assert agg.delta_vs_oracle == -6.0


def test_subset_aggregation_filters_by_trade_ids() -> None:
    outcomes = [
        _outcome("t1", "FixedR", {"r": 1.0}, True, 5.0),
        _outcome("t2", "FixedR", {"r": 1.0}, True, 7.0),
        _outcome("t3", "FixedR", {"r": 1.0}, True, 100.0),
    ]
    aggs = aggregate_subset_by_trades(outcomes, {"t1", "t2"})
    agg = next(iter(aggs.values()))
    assert agg.trades_total == 2
    assert agg.total_pnl == 12.0
    # t3 was excluded by the subset filter.
    assert "t3" not in agg.contributing_trades


def test_zero_fired_no_division_by_zero() -> None:
    """If a policy never fired across the dataset, mean/median/stdev
    stay at 0.0 (their default) without raising."""
    outcomes = [
        _outcome("t1", "FixedR", {"r": 1.0}, True, -2.0, fired=False),
        _outcome("t2", "FixedR", {"r": 1.0}, True, -1.0, fired=False),
    ]
    aggs = aggregate_results(outcomes)
    agg = next(iter(aggs.values()))
    assert agg.trades_fired == 0
    assert agg.mean_pnl_when_fired == 0.0
    assert agg.stdev_pnl_when_fired == 0.0


def test_single_fired_trade_stdev_is_zero() -> None:
    outcomes = [
        _outcome("t1", "FixedR", {"r": 1.0}, True, 5.0, fired=True),
        _outcome("t2", "FixedR", {"r": 1.0}, True, -2.0, fired=False),  # MTM
    ]
    aggs = aggregate_results(outcomes)
    agg = next(iter(aggs.values()))
    assert agg.trades_fired == 1
    assert agg.stdev_pnl_when_fired == 0.0
