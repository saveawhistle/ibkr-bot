"""Failure-mode classifier for closed trades.

Classifies each closed trade into one of six buckets, applied in order.
First match wins — the order matters, and the rules are documented in
:func:`classify_trade` so future tuning is grounded.

Six modes:

- ``DEGENERATE``: too short to evaluate (sub-5-minute or sub-5-bar trades).
  These tell us nothing about exit policy quality; they reflect entry-side
  issues or stop-out luck.
- ``STOP_OUT``: trade exited at (or very near) the initial protective stop.
  Mechanical exit policies have nothing to add — the bracket stop did its
  job and there's no improvable exit decision to be made.
- ``MODE_2_RUNNER_EXHAUSTION``: scale-out fired, peak was substantially
  higher than recorded exit. The bot took partial profit but the runner
  half gave back gains before the trail caught it.
- ``MODE_1_FLAGGING_BREAKOUT``: trade reached at least +1R intraday but
  recorded P&L came out below +0.5R. The bot was in profit and gave it
  back without taking the win.
- ``SUCCESSFUL_RUNNER``: scale-out fired and runner managed well — oracle
  ceiling barely beats actual outcome.
- ``UNCLASSIFIED``: didn't match the above. The reasoning string explains
  why each rule failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef


class FailureMode(Enum):
    MODE_1_FLAGGING_BREAKOUT = "mode_1_flagging_breakout"
    MODE_2_RUNNER_EXHAUSTION = "mode_2_runner_exhaustion"
    DEGENERATE = "degenerate"
    STOP_OUT = "stop_out"
    SUCCESSFUL_RUNNER = "successful_runner"
    UNCLASSIFIED = "unclassified"


# Threshold constants — surfaced here (not hidden in the function body)
# so the operator can find and tune them without re-reading the logic.
DEGENERATE_MIN_DURATION_MINUTES = 5.0
DEGENERATE_MIN_BAR_COUNT = 5
STOP_OUT_PROXIMITY_PCT = 0.05  # exit within 5% of initial stop
MODE_2_MIN_GAP_R = 0.5  # oracle - actual >= 0.5R in dollar terms
MODE_1_PEAK_R_THRESHOLD = 1.0  # peak reached >= 1R intraday
MODE_1_ACTUAL_R_THRESHOLD = 0.5  # but actual outcome was < 0.5R
SUCCESSFUL_RUNNER_MAX_GAP_R = 0.25  # oracle - actual < 0.25R


@dataclass(frozen=True)
class TradeClassification:
    """One trade's bucket + a sentence explaining the bucketing."""

    trade_ref: ClosedTradeRef
    mode: FailureMode
    reasoning: str


def classify_trade(
    trade_ref: ClosedTradeRef,
    actual_pnl: float,
    oracle_pnl: float,
    initial_stop: float,
    entry_price: float,
    peak_price_during_trade: float,
    actual_exit_price: float,
    trade_duration_minutes: float,
    bar_count_post_protection: int,
    scale_out_was_hit: bool,
    position_size: int,
) -> TradeClassification:
    """Classify ``trade_ref``. Pure function — same inputs always
    produce the same output. Rules apply in declared order; first
    match wins.

    ``position_size`` is taken from the recorded fill so dollar-denominated
    R thresholds (mode 1, mode 2, successful runner) compare apples to
    apples across trades of different sizes.
    """
    # Rule 1: DEGENERATE
    if trade_duration_minutes < DEGENERATE_MIN_DURATION_MINUTES:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.DEGENERATE,
            reasoning=(
                f"trade duration {trade_duration_minutes:.2f}min < "
                f"{DEGENERATE_MIN_DURATION_MINUTES}min threshold"
            ),
        )
    if bar_count_post_protection < DEGENERATE_MIN_BAR_COUNT:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.DEGENERATE,
            reasoning=(
                f"post-protection bar count {bar_count_post_protection} < "
                f"{DEGENERATE_MIN_BAR_COUNT} threshold"
            ),
        )

    # Rule 2: STOP_OUT (proximity-based; absolute zero division guarded).
    if initial_stop > 0:
        proximity = abs(actual_exit_price - initial_stop) / initial_stop
        if proximity < STOP_OUT_PROXIMITY_PCT:
            return TradeClassification(
                trade_ref=trade_ref,
                mode=FailureMode.STOP_OUT,
                reasoning=(
                    f"exit {actual_exit_price:.4f} within {proximity:.2%} of "
                    f"initial stop {initial_stop:.4f} (< {STOP_OUT_PROXIMITY_PCT:.0%} threshold)"
                ),
            )

    risk_per_share = entry_price - initial_stop
    if risk_per_share <= 0:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.UNCLASSIFIED,
            reasoning=(
                "degenerate risk: entry <= initial_stop — cannot compute "
                "R-multiples for mode-1/mode-2 rules"
            ),
        )

    # Dollar gap between oracle and actual P&L, expressed in R-units.
    risk_dollars = risk_per_share * position_size
    pnl_gap_in_r = (oracle_pnl - actual_pnl) / risk_dollars if risk_dollars > 0 else 0.0

    # Rule 3: MODE_2_RUNNER_EXHAUSTION
    if scale_out_was_hit and pnl_gap_in_r >= MODE_2_MIN_GAP_R:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.MODE_2_RUNNER_EXHAUSTION,
            reasoning=(
                f"scale-out hit; oracle exceeds actual by {pnl_gap_in_r:.2f}R "
                f"(>= {MODE_2_MIN_GAP_R}R threshold)"
            ),
        )

    # Rule 4: MODE_1_FLAGGING_BREAKOUT
    peak_r = (peak_price_during_trade - entry_price) / risk_per_share
    actual_r = actual_pnl / risk_dollars if risk_dollars > 0 else 0.0
    if peak_r >= MODE_1_PEAK_R_THRESHOLD and actual_r < MODE_1_ACTUAL_R_THRESHOLD:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.MODE_1_FLAGGING_BREAKOUT,
            reasoning=(
                f"peak reached {peak_r:.2f}R (>= {MODE_1_PEAK_R_THRESHOLD}R) "
                f"but actual outcome {actual_r:.2f}R (< {MODE_1_ACTUAL_R_THRESHOLD}R)"
            ),
        )

    # Rule 5: SUCCESSFUL_RUNNER
    if scale_out_was_hit and pnl_gap_in_r < SUCCESSFUL_RUNNER_MAX_GAP_R:
        return TradeClassification(
            trade_ref=trade_ref,
            mode=FailureMode.SUCCESSFUL_RUNNER,
            reasoning=(
                f"scale-out hit; oracle barely exceeds actual by "
                f"{pnl_gap_in_r:.2f}R (< {SUCCESSFUL_RUNNER_MAX_GAP_R}R threshold)"
            ),
        )

    # Rule 6: UNCLASSIFIED — explain why nothing matched.
    return TradeClassification(
        trade_ref=trade_ref,
        mode=FailureMode.UNCLASSIFIED,
        reasoning=(
            f"no rule matched: duration={trade_duration_minutes:.1f}min, "
            f"bars={bar_count_post_protection}, "
            f"stop_proximity={(abs(actual_exit_price - initial_stop) / initial_stop):.2%}, "
            f"scale_out_hit={scale_out_was_hit}, peak_r={peak_r:.2f}, "
            f"actual_r={actual_r:.2f}, pnl_gap_r={pnl_gap_in_r:.2f}"
        ),
    )
