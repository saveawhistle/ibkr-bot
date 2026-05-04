"""Gate-chain composition tests — order matters, veto wins but all
gates evaluate, accept-all clean path works."""

from __future__ import annotations

from datetime import UTC, datetime

from bot.config import ExitGatesConfig
from bot.exit_advisor.decision.gates import (
    DrawdownAccelerationGate,
    Gate,
    GateContext,
    GateResult,
    apply_gate_chain,
    build_default_gate_chain,
)
from bot.exit_advisor.decision.policy import ExitDecision, TradeState


def _state() -> TradeState:
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=datetime(2026, 4, 30, 13, 30, tzinfo=UTC),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,
        initial_scale_out=2.20,
        current_stop=1.90,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=2.10,
        current_price=2.05,
    )


def _ctx() -> GateContext:
    return GateContext(current_timestamp=datetime(2026, 4, 30, 14, 0, tzinfo=UTC))


def test_default_chain_drawdown_before_confidence() -> None:
    """The drawdown gate must run before the confidence gate so the
    shared ``drawdown_acceleration_active`` flag is set in time. If
    this ordering breaks, drawdown-aware confidence relaxation silently
    stops working."""
    chain = build_default_gate_chain(ExitGatesConfig())
    names = [g.name for g in chain]
    assert names.index("drawdown_acceleration") < names.index("confidence_threshold")


def test_default_chain_hard_guardrails_first() -> None:
    """Hard guardrails appear before any soft gate. Cheaper to evaluate
    and fail fast on safety violations before tunable thresholds run."""
    chain = build_default_gate_chain(ExitGatesConfig())
    names = [g.name for g in chain]
    hard = ("protected_position", "no_reentry", "stop_protection", "naked_position", "max_hold_time")
    soft = ("drawdown_acceleration", "confidence_threshold", "recency_throttle")
    last_hard = max(names.index(h) for h in hard)
    first_soft = min(names.index(s) for s in soft)
    assert last_hard < first_soft


def test_apply_chain_first_rejection_makes_final_none() -> None:
    chain = build_default_gate_chain(ExitGatesConfig())
    state = _state()
    state.is_protected = False  # ProtectedPositionGuardrail will veto
    decision = ExitDecision(action="exit_full", confidence=0.9)
    final, results = apply_gate_chain(chain, state, decision, _ctx())
    assert final is None
    rejected = [name for name, r in results if not r.accepted]
    assert "protected_position" in rejected


def test_apply_chain_evaluates_all_gates_after_rejection() -> None:
    """Forensic completeness — even after a veto, downstream gates still
    evaluate. Calibration analysis needs to know which gates would
    *also* have rejected for a given decision."""
    cfg = ExitGatesConfig(
        confidence_threshold=0.99,
        drawdown_reduced_confidence_threshold=0.99,  # keep reduced threshold strict too
    )
    chain = build_default_gate_chain(cfg)
    state = _state()
    state.is_protected = False  # First gate vetoes
    decision = ExitDecision(action="exit_full", confidence=0.5)  # Confidence also vetoes
    _, results = apply_gate_chain(chain, state, decision, _ctx())
    names_evaluated = [name for name, _ in results]
    # Every gate in the chain produced a result entry.
    assert len(names_evaluated) == len(chain)
    rejected_names = {name for name, r in results if not r.accepted}
    assert "protected_position" in rejected_names
    assert "confidence_threshold" in rejected_names


def test_apply_chain_clean_path_accepts() -> None:
    chain = build_default_gate_chain(ExitGatesConfig())
    state = _state()
    decision = ExitDecision(action="exit_full", confidence=0.9)
    final, results = apply_gate_chain(chain, state, decision, _ctx())
    assert final is decision
    assert all(r.accepted for _, r in results)


def test_apply_chain_drawdown_relaxes_confidence() -> None:
    """End-to-end: a deteriorating-position decision below the normal
    confidence threshold but above the drawdown-reduced threshold
    should pass when drawdown acceleration is detected."""
    cfg = ExitGatesConfig(
        confidence_threshold=0.7, drawdown_reduced_confidence_threshold=0.4
    )
    chain = build_default_gate_chain(cfg)
    # Peak at +2R (2.20), current at +1R (2.10) → 50% drawdown.
    state = _state()
    state.peak_price = 2.20
    state.current_price = 2.10
    decision = ExitDecision(action="exit_full", confidence=0.5)
    final, _ = apply_gate_chain(chain, state, decision, _ctx())
    assert final is decision  # accepted via reduced threshold


class _ExplodingGate:
    """Test fixture: a gate that raises. ``apply_gate_chain`` must
    capture the failure as a synthetic rejection rather than crashing
    the whole replay."""

    name = "exploding_gate"

    def evaluate(
        self, trade_state: TradeState, decision: ExitDecision, gate_context: GateContext
    ) -> GateResult:
        raise RuntimeError("simulated gate misbehavior")


def test_apply_chain_captures_gate_errors() -> None:
    chain: list[Gate] = [_ExplodingGate()]
    final, results = apply_gate_chain(chain, _state(), ExitDecision(action="hold"), _ctx())
    assert final is None
    assert results[0][0] == "exploding_gate"
    assert results[0][1].rejection_reason == "gate_evaluation_error"


def test_drawdown_gate_independent_of_decision_order() -> None:
    """Calling the drawdown gate twice with different state shouldn't
    leak between calls — the flag must reflect the state at each call,
    not accumulate across them."""
    chain: list[Gate] = [DrawdownAccelerationGate(drawdown_pct=0.5)]
    ctx = _ctx()
    state_high_peak = _state()
    state_high_peak.peak_price = 2.20
    state_high_peak.current_price = 2.10
    apply_gate_chain(chain, state_high_peak, ExitDecision(action="hold"), ctx)
    assert ctx.drawdown_acceleration_active is True

    # Now a state with no drawdown — flag must clear.
    state_no_dd = _state()
    state_no_dd.peak_price = 2.05
    state_no_dd.current_price = 2.05
    apply_gate_chain(chain, state_no_dd, ExitDecision(action="hold"), ctx)
    assert ctx.drawdown_acceleration_active is False


def test_disabled_gates_chain_passes_through() -> None:
    """``apply_gate_chain([], ...)`` is a no-op — empty chain accepts."""
    decision = ExitDecision(action="exit_full", confidence=0.0)
    final, results = apply_gate_chain([], _state(), decision, _ctx())
    assert final is decision
    assert results == []
