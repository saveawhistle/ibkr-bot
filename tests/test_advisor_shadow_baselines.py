"""Unit tests for ShadowBaselines."""

from __future__ import annotations

from datetime import datetime, timedelta

from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines
from bot.exit_advisor.core.events import BarFinalizedEvent
from bot.exit_advisor.decision.policy import (
    FixedRTakeProfit,
    MechanicalTrailPolicy,
    StallExitPolicy,
    TradeState,
)

_T0 = datetime(2026, 5, 4, 14, 30, 0)


def _state(
    *, current_price: float, peak_price: float | None = None, entry_offset_min: int = 0
) -> TradeState:
    return TradeState(
        symbol="ABC",
        entry_price=1.0,
        entry_timestamp=_T0,
        current_position_size=100,
        initial_position_size=100,
        initial_stop=0.9,
        initial_scale_out=1.2,
        current_stop=0.9,
        realized_pnl=0.0,
        is_protected=True,
        peak_price=peak_price if peak_price is not None else current_price,
        current_price=current_price,
    )


def test_three_baselines_present() -> None:
    baselines = ShadowBaselines()
    names = baselines.baseline_names()
    assert set(names) == {"stall_1r_5min", "trail", "fixed_r_1_5"}


def test_underlying_policies_are_correct_types() -> None:
    baselines = ShadowBaselines()
    # The dict isn't public, but consume_event returns one decision per baseline.
    event = BarFinalizedEvent(
        timestamp=_T0,
        symbol="ABC",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1000,
    )
    state = _state(current_price=1.0)
    results = baselines.consume_event(event, state)
    # All three baseline names should appear in results.
    assert set(results.keys()) == {"stall_1r_5min", "trail", "fixed_r_1_5"}


def test_baselines_produce_independent_recommendations() -> None:
    baselines = ShadowBaselines()
    # Price reached 1.5R — should trigger fixed_r_1_5 but neither stall nor trail.
    event = BarFinalizedEvent(
        timestamp=_T0 + timedelta(seconds=60),
        symbol="ABC",
        open=1.0,
        high=1.20,
        low=1.0,
        close=1.20,
        volume=2000,
    )
    state = _state(current_price=1.20, peak_price=1.20)
    results = baselines.consume_event(event, state)
    assert results["fixed_r_1_5"] is not None
    assert results["fixed_r_1_5"].action == "exit_full"
    # trail proposes a tighten if peak rose enough; with 0.10 trail and peak 1.20,
    # proposed = 1.10 which is > 0.9 current_stop, so it should fire too.
    assert results["trail"] is not None
    assert results["trail"].action == "tighten_stop"
    # stall_1r_5min: only 1 minute in trade, target reached → inert.
    assert results["stall_1r_5min"] is None


def test_stall_baseline_fires_when_target_not_reached_in_time() -> None:
    baselines = ShadowBaselines()
    event = BarFinalizedEvent(
        timestamp=_T0 + timedelta(minutes=6),
        symbol="ABC",
        open=1.0,
        high=1.02,
        low=1.0,
        close=1.02,
        volume=1000,
    )
    state = _state(current_price=1.02)
    results = baselines.consume_event(event, state)
    assert results["stall_1r_5min"] is not None
    assert results["stall_1r_5min"].action == "exit_full"


def test_reset_for_new_trade_reinstantiates_baselines() -> None:
    baselines = ShadowBaselines()
    # Trip fixed_r_1_5 once.
    event = BarFinalizedEvent(
        timestamp=_T0,
        symbol="ABC",
        open=1.0,
        high=1.20,
        low=1.0,
        close=1.20,
        volume=2000,
    )
    state = _state(current_price=1.20)
    first = baselines.consume_event(event, state)
    assert first["fixed_r_1_5"] is not None

    # Without reset, fixed_r_1_5 would stay latched (_exit_emitted=True).
    second = baselines.consume_event(event, state)
    assert second["fixed_r_1_5"] is None  # still latched

    baselines.reset_for_new_trade()
    third = baselines.consume_event(event, state)
    assert third["fixed_r_1_5"] is not None  # fresh baseline fires again


def test_baseline_exception_does_not_propagate() -> None:
    class _ExplodingPolicy:
        def on_event(self, _trade_state: TradeState, _event: BarFinalizedEvent) -> None:
            raise RuntimeError("boom")

    baselines = ShadowBaselines()
    # Inject an exploding policy alongside the legitimate ones.
    baselines._baselines["exploding"] = _ExplodingPolicy()  # type: ignore[assignment]
    event = BarFinalizedEvent(
        timestamp=_T0,
        symbol="ABC",
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1000,
    )
    state = _state(current_price=1.0)
    results = baselines.consume_event(event, state)
    assert results["exploding"] is None
    # Other baselines still ran.
    assert "stall_1r_5min" in results


# Smoke: ensure the underlying classes haven't moved.
_ = (StallExitPolicy, MechanicalTrailPolicy, FixedRTakeProfit)
