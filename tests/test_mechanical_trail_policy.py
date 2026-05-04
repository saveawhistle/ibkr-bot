"""MechanicalTrailPolicy tests — ratchet behavior + gate interaction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.exit_advisor.core.events import TimeOfDayMilestone
from bot.exit_advisor.decision.policy import MechanicalTrailPolicy, TradeState


def _ts(minute: int) -> datetime:
    return datetime(2026, 4, 30, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)


def _state(peak: float, current_stop: float, current_price: float | None = None) -> TradeState:
    return TradeState(
        symbol="X",
        entry_price=2.00,
        entry_timestamp=_ts(0),
        current_position_size=100,
        initial_position_size=100,
        initial_stop=1.90,
        initial_scale_out=2.20,
        current_stop=current_stop,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=peak,
        current_price=current_price if current_price is not None else peak,
    )


def _evt(minute: int = 1) -> TimeOfDayMilestone:
    return TimeOfDayMilestone(timestamp=_ts(minute), symbol="X", minutes_after_open=minute)


def test_constructor_rejects_both_args() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MechanicalTrailPolicy(trail_abs=0.10, trail_pct=0.02)


def test_constructor_rejects_neither_arg() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MechanicalTrailPolicy()


def test_constructor_rejects_negative_abs() -> None:
    with pytest.raises(ValueError, match="positive"):
        MechanicalTrailPolicy(trail_abs=-0.10)


def test_constructor_rejects_pct_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        MechanicalTrailPolicy(trail_pct=1.5)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        MechanicalTrailPolicy(trail_pct=0.0)


def test_trail_abs_proposes_when_peak_rises() -> None:
    policy = MechanicalTrailPolicy(trail_abs=0.05)
    # Peak 2.10, current_stop 1.90, proposed = 2.05. Tightens.
    decision = policy.on_event(_state(peak=2.10, current_stop=1.90), _evt())
    assert decision is not None
    assert decision.action == "tighten_stop"
    assert decision.new_stop_price == pytest.approx(2.05)


def test_trail_abs_no_op_when_proposed_below_current_stop() -> None:
    """Initial bracket stop tighter than the trail level → no tighten."""
    policy = MechanicalTrailPolicy(trail_abs=0.10)
    # Peak 2.10, current_stop 2.05, proposed = 2.00. Wouldn't tighten.
    decision = policy.on_event(_state(peak=2.10, current_stop=2.05), _evt())
    assert decision is None


def test_trail_pct_ratchets_on_higher_peak() -> None:
    policy = MechanicalTrailPolicy(trail_pct=0.02)
    # Peak 2.20 × 0.98 = 2.156. Current stop 2.10 → tightens to 2.156.
    decision = policy.on_event(_state(peak=2.20, current_stop=2.10), _evt())
    assert decision is not None
    assert decision.new_stop_price == pytest.approx(2.156)


def test_trail_self_throttles_after_acceptance() -> None:
    """If a previous proposal was accepted, current_stop has moved up.
    A subsequent on_event with the SAME peak should produce no new
    proposal — the trail level hasn't moved."""
    policy = MechanicalTrailPolicy(trail_abs=0.05)
    state = _state(peak=2.10, current_stop=1.90)
    first = policy.on_event(state, _evt())
    assert first is not None
    # Simulate harness applying the tighten:
    state.current_stop = first.new_stop_price  # type: ignore[assignment]
    # Same peak, no new proposal:
    assert policy.on_event(state, _evt(2)) is None


def test_trail_does_not_run_unprotected() -> None:
    policy = MechanicalTrailPolicy(trail_abs=0.05)
    state = _state(peak=2.10, current_stop=1.90)
    state.is_protected = False
    assert policy.on_event(state, _evt()) is None


def test_trail_reason_string_carries_param() -> None:
    """Reason embeds the parameter for forensic readability — useful
    when scanning a comparison report's per-policy detail."""
    policy_abs = MechanicalTrailPolicy(trail_abs=0.10)
    policy_pct = MechanicalTrailPolicy(trail_pct=0.04)
    d_abs = policy_abs.on_event(_state(peak=2.20, current_stop=1.90), _evt())
    d_pct = policy_pct.on_event(_state(peak=2.20, current_stop=1.90), _evt())
    assert d_abs is not None and "abs_0.1" in d_abs.reason
    assert d_pct is not None and "pct_0.04" in d_pct.reason


def test_trail_interacts_with_gate_chain_min_r_for_stop_tighten() -> None:
    """A tight trail at low R-multiple should have its tighten_stop
    decisions vetoed by ``min_r_for_stop_tighten`` (default 0.5R).
    Verifies the trail policy and the gate compose correctly."""
    from bot.config import ExitGatesConfig
    from bot.exit_advisor.decision.gates import (
        GateContext,
        MinRMultipleForStopTightenGate,
    )

    policy = MechanicalTrailPolicy(trail_abs=0.05)
    # Peak 2.04 = 0.4R from entry 2.00 / risk 0.10 — below 0.5R threshold.
    state = _state(peak=2.04, current_stop=1.90, current_price=2.04)
    decision = policy.on_event(state, _evt())
    assert decision is not None
    gate = MinRMultipleForStopTightenGate(min_r=0.5)
    result = gate.evaluate(state, decision, GateContext(current_timestamp=_ts(1)))
    assert not result.accepted
    assert result.rejection_reason == "below_min_r_for_stop_tighten"
    cfg = ExitGatesConfig(min_r_for_stop_tighten=0.5)
    assert cfg.min_r_for_stop_tighten == 0.5  # sanity: matches default
