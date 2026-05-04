"""Failure-mode classifier tests — one positive case per mode plus
boundary tests around each threshold. Order-sensitivity (DEGENERATE
beats STOP_OUT etc.) tested explicitly."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from bot.exit_advisor.analysis.failure_modes import (
    DEGENERATE_MIN_BAR_COUNT,
    DEGENERATE_MIN_DURATION_MINUTES,
    STOP_OUT_PROXIMITY_PCT,
    FailureMode,
    TradeClassification,
    classify_trade,
)
from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef


def _ref() -> ClosedTradeRef:
    return ClosedTradeRef(
        symbol="X",
        trade_date=date(2026, 4, 30),
        trade_id="t1",
        entry_timestamp=datetime(2026, 4, 30, 13, 30, tzinfo=UTC),
        exit_timestamp=datetime(2026, 4, 30, 14, 0, tzinfo=UTC),
        session_log_path=Path("logs/session_2026-04-30.jsonl"),
    )


def _classify(**kwargs: object) -> TradeClassification:
    """Helper with sensible defaults — tests override only what they need.
    Defaults represent a 'plain' winning trade that doesn't match any
    rule (becomes UNCLASSIFIED). Returns the TradeClassification."""
    base: dict[str, object] = dict(
        trade_ref=_ref(),
        actual_pnl=10.0,
        oracle_pnl=12.0,
        initial_stop=1.90,
        entry_price=2.00,
        peak_price_during_trade=2.05,
        actual_exit_price=2.05,
        trade_duration_minutes=30.0,
        bar_count_post_protection=20,
        scale_out_was_hit=False,
        position_size=100,
    )
    base.update(kwargs)
    return classify_trade(**base)  # type: ignore[arg-type]


# --- DEGENERATE ---


def test_degenerate_when_duration_below_threshold() -> None:
    c = _classify(trade_duration_minutes=DEGENERATE_MIN_DURATION_MINUTES - 0.1)
    assert c.mode == FailureMode.DEGENERATE
    assert "duration" in c.reasoning


def test_degenerate_when_bar_count_below_threshold() -> None:
    c = _classify(
        trade_duration_minutes=10.0,
        bar_count_post_protection=DEGENERATE_MIN_BAR_COUNT - 1,
    )
    assert c.mode == FailureMode.DEGENERATE
    assert "bar count" in c.reasoning


def test_not_degenerate_at_threshold_exactly() -> None:
    """Boundary: exactly at the threshold is NOT degenerate."""
    c = _classify(
        trade_duration_minutes=DEGENERATE_MIN_DURATION_MINUTES,
        bar_count_post_protection=DEGENERATE_MIN_BAR_COUNT,
        actual_exit_price=2.05,
    )
    assert c.mode != FailureMode.DEGENERATE


def test_degenerate_beats_stop_out() -> None:
    """A trade that's BOTH degenerate AND a stop-out is classified
    DEGENERATE because that rule runs first."""
    c = _classify(
        trade_duration_minutes=2.0,  # degenerate
        actual_exit_price=1.91,  # within 5% of stop 1.90 — also stop-out
        initial_stop=1.90,
    )
    assert c.mode == FailureMode.DEGENERATE


# --- STOP_OUT ---


def test_stop_out_within_proximity() -> None:
    c = _classify(initial_stop=1.90, actual_exit_price=1.92)  # ~1% proximity
    assert c.mode == FailureMode.STOP_OUT


def test_stop_out_boundary_just_outside() -> None:
    """Just-above the proximity threshold → not STOP_OUT."""
    c = _classify(
        initial_stop=1.00,
        actual_exit_price=1.0 + 1.0 * STOP_OUT_PROXIMITY_PCT * 1.5,  # well above proximity
        trade_duration_minutes=20.0,
        bar_count_post_protection=10,
    )
    assert c.mode != FailureMode.STOP_OUT


# --- MODE_2_RUNNER_EXHAUSTION ---


def test_mode_2_runner_exhaustion() -> None:
    """Scale-out hit, oracle exceeds actual by >= 0.5R in dollar terms.
    With entry=2, stop=1.90, position=100 → risk_dollars = $10. So
    oracle - actual >= $5 should trigger MODE_2."""
    c = _classify(
        actual_pnl=5.0,
        oracle_pnl=15.0,  # gap = $10 = 1.0R
        scale_out_was_hit=True,
        peak_price_during_trade=2.20,
    )
    assert c.mode == FailureMode.MODE_2_RUNNER_EXHAUSTION


def test_mode_2_requires_scale_out() -> None:
    """Same gap but no scale-out → falls through to MODE_1 if eligible."""
    c = _classify(
        actual_pnl=2.0,  # 0.2R, below MODE_1_ACTUAL_R_THRESHOLD
        oracle_pnl=12.0,
        scale_out_was_hit=False,
        peak_price_during_trade=2.15,  # 1.5R peak — triggers MODE_1
    )
    assert c.mode == FailureMode.MODE_1_FLAGGING_BREAKOUT


def test_mode_2_below_gap_threshold() -> None:
    """Scale-out hit but oracle gap < 0.5R → SUCCESSFUL_RUNNER instead."""
    c = _classify(
        actual_pnl=10.0,
        oracle_pnl=11.0,  # gap = $1 = 0.1R, below 0.5R
        scale_out_was_hit=True,
    )
    assert c.mode == FailureMode.SUCCESSFUL_RUNNER


# --- MODE_1_FLAGGING_BREAKOUT ---


def test_mode_1_flagging_breakout() -> None:
    """Peak reached >= 1R, actual < 0.5R."""
    c = _classify(
        peak_price_during_trade=2.15,  # 1.5R
        actual_pnl=2.0,  # 0.2R
        scale_out_was_hit=False,
    )
    assert c.mode == FailureMode.MODE_1_FLAGGING_BREAKOUT


def test_mode_1_peak_below_threshold() -> None:
    """Peak < 1R → not MODE_1."""
    c = _classify(
        peak_price_during_trade=2.05,  # 0.5R
        actual_pnl=2.0,
    )
    assert c.mode != FailureMode.MODE_1_FLAGGING_BREAKOUT


def test_mode_1_actual_above_threshold() -> None:
    """Peak >= 1R but actual >= 0.5R → not MODE_1 (took the win)."""
    c = _classify(
        peak_price_during_trade=2.15,  # 1.5R peak
        actual_pnl=8.0,  # 0.8R
    )
    assert c.mode != FailureMode.MODE_1_FLAGGING_BREAKOUT


# --- SUCCESSFUL_RUNNER ---


def test_successful_runner() -> None:
    c = _classify(
        actual_pnl=12.0,
        oracle_pnl=13.0,  # gap = 0.1R, below 0.25R
        scale_out_was_hit=True,
    )
    assert c.mode == FailureMode.SUCCESSFUL_RUNNER


def test_successful_runner_requires_scale_out() -> None:
    c = _classify(
        actual_pnl=12.0,
        oracle_pnl=13.0,
        scale_out_was_hit=False,
    )
    assert c.mode != FailureMode.SUCCESSFUL_RUNNER


# --- UNCLASSIFIED ---


def test_unclassified_when_no_rule_matches() -> None:
    """A modest winner with no scale-out, no MODE_1 conditions, no
    stop-out — falls through to UNCLASSIFIED."""
    c = _classify(
        actual_pnl=3.0,
        oracle_pnl=4.0,
        peak_price_during_trade=2.04,
        actual_exit_price=2.03,  # not near stop
        scale_out_was_hit=False,
    )
    assert c.mode == FailureMode.UNCLASSIFIED


def test_unclassified_when_degenerate_risk() -> None:
    """entry <= initial_stop is degenerate risk → UNCLASSIFIED with
    a specific reasoning string, regardless of other inputs."""
    c = _classify(
        entry_price=2.00,
        initial_stop=2.00,  # zero risk
        actual_exit_price=2.10,  # not near stop
        trade_duration_minutes=20.0,
        bar_count_post_protection=10,
    )
    assert c.mode == FailureMode.UNCLASSIFIED
    assert "degenerate risk" in c.reasoning


# --- Order sensitivity ---


def test_classifier_is_pure() -> None:
    """Same inputs → same output, every time."""
    c1 = _classify()
    c2 = _classify()
    assert c1.mode == c2.mode
    assert c1.reasoning == c2.reasoning
