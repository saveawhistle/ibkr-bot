"""Soft-gate tests — parameterized, calibratable bounds. Each gate gets
positive + negative cases plus the once-per-session and contextual
behaviors that matter for calibration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.exit_advisor.decision.gates import (
    ConfidenceThresholdGate,
    DrawdownAccelerationGate,
    GateContext,
    MinRMultipleForPartialGate,
    MinRMultipleForStopTightenGate,
    RecencyThrottleGate,
)
from bot.exit_advisor.decision.policy import ExitDecision, TradeState


def _state(**overrides: object) -> TradeState:
    base = dict(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=datetime(2026, 4, 30, 13, 30, tzinfo=UTC),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,  # risk_per_share = 0.10
        initial_scale_out=2.20,
        current_stop=1.90,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=2.00,
        current_price=2.00,
    )
    base.update(overrides)
    return TradeState(**base)  # type: ignore[arg-type]


def _ctx(ts: datetime | None = None) -> GateContext:
    return GateContext(current_timestamp=ts or datetime(2026, 4, 30, 14, 0, tzinfo=UTC))


# --- ConfidenceThresholdGate ---


def test_confidence_below_threshold_rejected() -> None:
    g = ConfidenceThresholdGate(threshold=0.7)
    decision = ExitDecision(action="exit_full", confidence=0.5)
    assert not g.evaluate(_state(), decision, _ctx()).accepted


def test_confidence_at_threshold_accepted() -> None:
    g = ConfidenceThresholdGate(threshold=0.7)
    decision = ExitDecision(action="exit_full", confidence=0.7)
    assert g.evaluate(_state(), decision, _ctx()).accepted


def test_confidence_threshold_ignores_hold() -> None:
    """``hold`` decisions skip the confidence check — they're no-ops,
    confidence is meaningless for them."""
    g = ConfidenceThresholdGate(threshold=0.99)
    decision = ExitDecision(action="hold", confidence=0.0)
    assert g.evaluate(_state(), decision, _ctx()).accepted


def test_confidence_drawdown_reduced_threshold_used_when_active() -> None:
    """When the drawdown gate has set the shared flag, the confidence
    gate falls back to the (lower) drawdown threshold so the advisor
    can act decisively under deteriorating conditions."""
    g = ConfidenceThresholdGate(threshold=0.7, drawdown_reduced_threshold=0.4)
    ctx = _ctx()
    ctx.drawdown_acceleration_active = True
    decision = ExitDecision(action="exit_full", confidence=0.5)
    assert g.evaluate(_state(), decision, ctx).accepted


def test_confidence_drawdown_reduced_threshold_inactive_uses_main() -> None:
    g = ConfidenceThresholdGate(threshold=0.7, drawdown_reduced_threshold=0.4)
    ctx = _ctx()
    ctx.drawdown_acceleration_active = False
    decision = ExitDecision(action="exit_full", confidence=0.5)
    assert not g.evaluate(_state(), decision, ctx).accepted


# --- RecencyThrottleGate ---


def test_recency_throttle_vetoes_reversal_within_window() -> None:
    g = RecencyThrottleGate(throttle_seconds=30)
    last_ts = datetime(2026, 4, 30, 14, 0, tzinfo=UTC)
    last = ExitDecision(action="tighten_stop", new_stop_price=1.95)
    ctx = _ctx(ts=last_ts + timedelta(seconds=10))
    ctx.recent_decisions = [(last_ts, last)]
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    result = g.evaluate(_state(), decision, ctx)
    assert not result.accepted
    assert result.rejection_reason == "recency_throttle_active"


def test_recency_throttle_accepts_reversal_outside_window() -> None:
    g = RecencyThrottleGate(throttle_seconds=30)
    last_ts = datetime(2026, 4, 30, 14, 0, tzinfo=UTC)
    last = ExitDecision(action="tighten_stop", new_stop_price=1.95)
    ctx = _ctx(ts=last_ts + timedelta(seconds=60))  # outside throttle
    ctx.recent_decisions = [(last_ts, last)]
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    assert g.evaluate(_state(), decision, ctx).accepted


def test_recency_throttle_accepts_same_action_repeat() -> None:
    """Same-action decisions are not reversals — repeated tighten_stops
    on consecutive bars are normal trail-up behavior, not flapping."""
    g = RecencyThrottleGate(throttle_seconds=30)
    last_ts = datetime(2026, 4, 30, 14, 0, tzinfo=UTC)
    last = ExitDecision(action="tighten_stop", new_stop_price=1.95)
    ctx = _ctx(ts=last_ts + timedelta(seconds=5))
    ctx.recent_decisions = [(last_ts, last)]
    decision = ExitDecision(action="tighten_stop", new_stop_price=1.97)
    assert g.evaluate(_state(), decision, ctx).accepted


def test_recency_throttle_accepts_first_decision() -> None:
    g = RecencyThrottleGate(throttle_seconds=30)
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    assert g.evaluate(_state(), decision, _ctx()).accepted


# --- MinRMultipleForPartialGate ---


def test_min_r_partial_below_threshold_rejected() -> None:
    g = MinRMultipleForPartialGate(min_r=1.0)
    state = _state(current_price=2.05)  # +0.05 / 0.10 risk = 0.5R
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    result = g.evaluate(state, decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "below_min_r_for_partial"


def test_min_r_partial_at_threshold_accepted() -> None:
    g = MinRMultipleForPartialGate(min_r=1.0)
    state = _state(current_price=2.10)  # +0.10 / 0.10 = 1.0R
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    assert g.evaluate(state, decision, _ctx()).accepted


def test_min_r_partial_ignores_other_actions() -> None:
    g = MinRMultipleForPartialGate(min_r=10.0)
    state = _state(current_price=2.05)
    for action in ("hold", "exit_full", "tighten_stop"):
        decision = ExitDecision(action=action, new_stop_price=1.95)
        assert g.evaluate(state, decision, _ctx()).accepted


def test_min_r_partial_invalid_risk_rejected() -> None:
    """Degenerate state (entry <= initial_stop) is itself a problem;
    the gate refuses to evaluate rather than dividing by zero."""
    g = MinRMultipleForPartialGate(min_r=1.0)
    state = _state(initial_stop=2.00)  # risk_per_share = 0
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    result = g.evaluate(state, decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "invalid_risk_per_share"


# --- MinRMultipleForStopTightenGate ---


def test_min_r_stop_tighten_below_threshold_rejected() -> None:
    g = MinRMultipleForStopTightenGate(min_r=0.5)
    state = _state(current_price=2.02)  # 0.2R
    decision = ExitDecision(action="tighten_stop", new_stop_price=1.95)
    assert not g.evaluate(state, decision, _ctx()).accepted


def test_min_r_stop_tighten_at_threshold_accepted() -> None:
    """0.5R exactly. ``2.05 - 2.00`` lands at 0.5R but FP makes the
    computed ratio 0.4999...; use 2.06 to clear the threshold without
    relying on float exactness."""
    g = MinRMultipleForStopTightenGate(min_r=0.5)
    state = _state(current_price=2.06)  # 0.6R, comfortably above 0.5
    decision = ExitDecision(action="tighten_stop", new_stop_price=1.95)
    assert g.evaluate(state, decision, _ctx()).accepted


# --- DrawdownAccelerationGate ---


def test_drawdown_gate_sets_flag_when_drawdown_exceeds_threshold() -> None:
    """Peak at +2R, current back to +1R → 50% drawdown from peak →
    flag set. Confidence gate downstream can then use the reduced threshold."""
    g = DrawdownAccelerationGate(drawdown_pct=0.5)
    state = _state(peak_price=2.20, current_price=2.10)  # peak +2R, current +1R, drawdown 50%
    ctx = _ctx()
    g.evaluate(state, ExitDecision(action="hold"), ctx)
    assert ctx.drawdown_acceleration_active is True


def test_drawdown_gate_clears_flag_when_below_threshold() -> None:
    g = DrawdownAccelerationGate(drawdown_pct=0.5)
    state = _state(peak_price=2.20, current_price=2.18)  # tiny drawdown
    ctx = _ctx()
    ctx.drawdown_acceleration_active = True  # left over from a prior bar
    g.evaluate(state, ExitDecision(action="hold"), ctx)
    assert ctx.drawdown_acceleration_active is False


def test_drawdown_gate_handles_no_peak_above_entry() -> None:
    """If peak never went above entry, peak_r <= 0 — flag must be False
    rather than dividing by zero."""
    g = DrawdownAccelerationGate(drawdown_pct=0.5)
    state = _state(peak_price=1.95, current_price=1.92)
    ctx = _ctx()
    ctx.drawdown_acceleration_active = True
    g.evaluate(state, ExitDecision(action="hold"), ctx)
    assert ctx.drawdown_acceleration_active is False


def test_drawdown_gate_always_accepts() -> None:
    """The gate signals via ``GateContext``, never via veto. A clean
    bypass design — the confidence gate downstream is what reads the flag."""
    g = DrawdownAccelerationGate(drawdown_pct=0.5)
    state = _state(peak_price=2.20, current_price=2.10)
    result = g.evaluate(state, ExitDecision(action="exit_full"), _ctx())
    assert result.accepted
