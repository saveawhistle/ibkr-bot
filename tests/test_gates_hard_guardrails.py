"""Hard-guardrail tests — non-negotiable safety rules. One veto case
and one accept case per guardrail."""

from __future__ import annotations

from datetime import UTC, datetime

from bot.exit_advisor.decision.gates import (
    GateContext,
    MaxHoldTimeGuardrail,
    NakedPositionGuardrail,
    NoReentryGuardrail,
    ProtectedPositionGuardrail,
    StopProtectionGuardrail,
)
from bot.exit_advisor.decision.policy import ExitDecision, TradeState


def _ctx() -> GateContext:
    return GateContext(current_timestamp=datetime(2026, 4, 30, 13, 30, tzinfo=UTC))


def _state(**overrides: object) -> TradeState:
    base = dict(
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
        peak_price=2.05,
        current_price=2.05,
    )
    base.update(overrides)
    return TradeState(**base)  # type: ignore[arg-type]


# --- StopProtectionGuardrail ---


def test_stop_protection_vetoes_widening() -> None:
    """Widening the stop = moving it AWAY from price (lower for a long).
    Hard veto: the protective level must only ever tighten."""
    g = StopProtectionGuardrail()
    state = _state(current_stop=1.95)
    decision = ExitDecision(action="tighten_stop", new_stop_price=1.90)
    result = g.evaluate(state, decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "stop_widening_attempted"


def test_stop_protection_accepts_tightening() -> None:
    g = StopProtectionGuardrail()
    state = _state(current_stop=1.95)
    decision = ExitDecision(action="tighten_stop", new_stop_price=1.98)
    assert g.evaluate(state, decision, _ctx()).accepted


def test_stop_protection_vetoes_missing_new_price() -> None:
    """``tighten_stop`` without a new_stop_price is malformed — veto so
    the harness doesn't silently apply a no-op."""
    g = StopProtectionGuardrail()
    decision = ExitDecision(action="tighten_stop")
    result = g.evaluate(_state(), decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "tighten_stop_missing_new_price"


def test_stop_protection_ignores_non_stop_actions() -> None:
    """The gate only constrains tighten_stop; other actions pass through."""
    g = StopProtectionGuardrail()
    for action in ("hold", "exit_full", "exit_partial"):
        decision = ExitDecision(action=action)
        assert g.evaluate(_state(), decision, _ctx()).accepted


# --- NoReentryGuardrail ---


def test_no_reentry_vetoes_action_on_flat_position() -> None:
    g = NoReentryGuardrail()
    state = _state(current_position_size=0)
    decision = ExitDecision(action="exit_full")
    result = g.evaluate(state, decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "no_action_on_flat_position"


def test_no_reentry_accepts_action_on_active_position() -> None:
    g = NoReentryGuardrail()
    state = _state(current_position_size=50)
    assert g.evaluate(state, ExitDecision(action="exit_full"), _ctx()).accepted


# --- ProtectedPositionGuardrail ---


def test_protected_position_vetoes_unprotected() -> None:
    """Defense-in-depth — even though policy invocation is gated on
    is_protected at the harness, the gate layer re-checks. A future
    code path that bypasses the policy gate gets caught here."""
    g = ProtectedPositionGuardrail()
    state = _state(is_protected=False)
    result = g.evaluate(state, ExitDecision(action="exit_full"), _ctx())
    assert not result.accepted
    assert result.rejection_reason == "position_not_yet_protected"


def test_protected_position_accepts_protected() -> None:
    g = ProtectedPositionGuardrail()
    assert g.evaluate(_state(is_protected=True), ExitDecision(action="exit_full"), _ctx()).accepted


# --- NakedPositionGuardrail ---


def test_naked_position_vetoes_full_exit_partial() -> None:
    """A partial exit with pct=1.0 leaves zero shares — that's a full
    exit dressed as a partial. Veto so the operator uses the right action."""
    g = NakedPositionGuardrail()
    decision = ExitDecision(action="exit_partial", partial_pct=1.0)
    result = g.evaluate(_state(current_position_size=100), decision, _ctx())
    assert not result.accepted
    assert result.rejection_reason == "partial_would_leave_no_position"


def test_naked_position_accepts_real_partial() -> None:
    g = NakedPositionGuardrail()
    decision = ExitDecision(action="exit_partial", partial_pct=0.5)
    assert g.evaluate(_state(current_position_size=100), decision, _ctx()).accepted


def test_naked_position_vetoes_partial_with_tiny_remainder() -> None:
    """100 shares × 99.5% out = 0.5 shares left = below 1, still a veto."""
    g = NakedPositionGuardrail()
    decision = ExitDecision(action="exit_partial", partial_pct=0.995)
    result = g.evaluate(_state(current_position_size=100), decision, _ctx())
    assert not result.accepted


# --- MaxHoldTimeGuardrail ---


def test_max_hold_time_guardrail_always_accepts() -> None:
    """The guardrail itself is a no-veto whitelisting layer — the
    *policy* (MaxHoldTimePolicy) produces the force-exit; the guardrail's
    role is to stand alongside the soft gates and remind future readers
    that the time-based force-exit is non-negotiable."""
    g = MaxHoldTimeGuardrail(max_hold_minutes=60)
    decision = ExitDecision(action="exit_full", reason="max_hold_time_reached_60_minutes")
    assert g.evaluate(_state(), decision, _ctx()).accepted

    # Also accepts unrelated decisions — it's not a veto gate.
    assert g.evaluate(_state(), ExitDecision(action="hold"), _ctx()).accepted
