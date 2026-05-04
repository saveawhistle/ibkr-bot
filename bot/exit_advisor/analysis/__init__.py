"""Multi-trade aggregation + outcome classification.

Cross-trade analytics over a corpus of replayed trades — used by
the comparison runner to bin policies into buckets (mean PnL,
realized R distribution, failure-mode incidence).
"""

from bot.exit_advisor.analysis.aggregation import (
    PolicyAggregateMetrics,
    TradeOutcome,
    aggregate_results,
    aggregate_subset_by_trades,
)
from bot.exit_advisor.analysis.failure_modes import (
    DEGENERATE_MIN_BAR_COUNT,
    DEGENERATE_MIN_DURATION_MINUTES,
    MODE_1_ACTUAL_R_THRESHOLD,
    MODE_1_PEAK_R_THRESHOLD,
    MODE_2_MIN_GAP_R,
    STOP_OUT_PROXIMITY_PCT,
    SUCCESSFUL_RUNNER_MAX_GAP_R,
    FailureMode,
    TradeClassification,
    classify_trade,
)

__all__ = [
    # aggregation
    "PolicyAggregateMetrics",
    "TradeOutcome",
    "aggregate_results",
    "aggregate_subset_by_trades",
    # failure_modes
    "FailureMode",
    "TradeClassification",
    "classify_trade",
    "DEGENERATE_MIN_BAR_COUNT",
    "DEGENERATE_MIN_DURATION_MINUTES",
    "MODE_1_ACTUAL_R_THRESHOLD",
    "MODE_1_PEAK_R_THRESHOLD",
    "MODE_2_MIN_GAP_R",
    "STOP_OUT_PROXIMITY_PCT",
    "SUCCESSFUL_RUNNER_MAX_GAP_R",
]
