"""Pure-functional aggregation over per-trade comparison JSONL output.

Groups by ``(policy_name, frozenset(params), gates_enabled)`` and
computes summary metrics for each group. The inputs are
:class:`TradeOutcome` rows, one per (trade × policy × params × gate-mode);
outputs are :class:`PolicyAggregateMetrics` keyed by a stable string.

Mark-to-market trades (policy never fired, position open at end of
replay) contribute to ``total_pnl`` but are excluded from
``mean_pnl_when_fired`` / median / stdev — those statistics describe
the policy's behavior when it actually engaged, not its passive
mark-to-market default.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef


@dataclass(frozen=True)
class TradeOutcome:
    """One row in the aggregation pipeline. Carries the trade's
    reference, the policy/params/gate identity, and the realized
    metrics needed to roll up."""

    trade_ref: ClosedTradeRef
    policy_name: str
    policy_params: dict[str, Any]
    gates_enabled: bool
    final_pnl: float
    fired: bool
    decisions_proposed: int
    decisions_accepted: int
    decisions_rejected: int
    open_at_end_of_replay: bool
    actual_pnl: float
    oracle_pnl: float


@dataclass
class PolicyAggregateMetrics:
    """Aggregated stats for one (policy, params, gates_enabled) group."""

    policy_name: str
    policy_params: dict[str, Any]
    gates_enabled: bool

    trades_total: int = 0
    trades_fired: int = 0
    trades_never_fired: int = 0

    mean_pnl_when_fired: float = 0.0
    median_pnl_when_fired: float = 0.0
    stdev_pnl_when_fired: float = 0.0

    total_pnl: float = 0.0
    total_actual_pnl: float = 0.0
    total_oracle_pnl: float = 0.0
    delta_vs_actual: float = 0.0
    delta_vs_oracle: float = 0.0

    contributing_trades: list[str] = field(default_factory=list)


def _params_key(params: dict[str, Any]) -> str:
    """Stable string for grouping. JSON with sorted keys works for any
    JSON-serializable param set; sufficient for our integer/float
    sweep values."""
    import json

    return json.dumps(params, sort_keys=True, default=str)


def _group_key(name: str, params: dict[str, Any], gates_enabled: bool) -> str:
    return f"{name}|{_params_key(params)}|gates={gates_enabled}"


def aggregate_results(
    per_trade_results: list[TradeOutcome],
) -> dict[str, PolicyAggregateMetrics]:
    """Group outcomes by (policy_name, params, gates_enabled). Returns
    a dict keyed by a stable group string. Insertion order is
    determined by the first occurrence of each group in the input.
    """
    groups: dict[str, list[TradeOutcome]] = {}
    for outcome in per_trade_results:
        key = _group_key(outcome.policy_name, outcome.policy_params, outcome.gates_enabled)
        groups.setdefault(key, []).append(outcome)

    out: dict[str, PolicyAggregateMetrics] = {}
    for key, items in groups.items():
        first = items[0]
        agg = PolicyAggregateMetrics(
            policy_name=first.policy_name,
            policy_params=dict(first.policy_params),
            gates_enabled=first.gates_enabled,
        )
        agg.trades_total = len(items)
        fired_pnls: list[float] = [o.final_pnl for o in items if o.fired]
        agg.trades_fired = len(fired_pnls)
        agg.trades_never_fired = agg.trades_total - agg.trades_fired

        if fired_pnls:
            agg.mean_pnl_when_fired = statistics.fmean(fired_pnls)
            agg.median_pnl_when_fired = statistics.median(fired_pnls)
            agg.stdev_pnl_when_fired = (
                statistics.pstdev(fired_pnls) if len(fired_pnls) > 1 else 0.0
            )

        agg.total_pnl = sum(o.final_pnl for o in items)
        agg.total_actual_pnl = sum(o.actual_pnl for o in items)
        agg.total_oracle_pnl = sum(o.oracle_pnl for o in items)
        agg.delta_vs_actual = agg.total_pnl - agg.total_actual_pnl
        agg.delta_vs_oracle = agg.total_pnl - agg.total_oracle_pnl
        agg.contributing_trades = [o.trade_ref.trade_id for o in items]

        out[key] = agg

    return out


def aggregate_subset_by_trades(
    per_trade_results: list[TradeOutcome],
    trade_ids: set[str],
) -> dict[str, PolicyAggregateMetrics]:
    """Aggregate a subset of trades (e.g. all MODE_1 trades) — same
    structure, different denominator. Used for the per-mode rankings."""
    subset = [o for o in per_trade_results if o.trade_ref.trade_id in trade_ids]
    return aggregate_results(subset)
