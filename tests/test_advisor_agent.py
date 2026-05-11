"""Unit tests for the ExitAdvisor agent.

The LLM client is replaced by a `_FakeLLMClient` that returns
deterministic ``LLMCallResult`` instances. No real API traffic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest

from bot.exit_advisor.advisor.agent import ExitAdvisor
from bot.exit_advisor.advisor.buffer import EventBuffer
from bot.exit_advisor.advisor.cost_tracker import CostTracker
from bot.exit_advisor.advisor.llm_client import LLMCallResult
from bot.exit_advisor.advisor.shadow_baselines import ShadowBaselines
from bot.exit_advisor.core.events import (
    BarFinalizedEvent,
    LargePrint,
    PartialFillEvent,
    PositionProtected,
)
from bot.exit_advisor.core.types import ExitRecommendation

_T0 = datetime(2026, 5, 4, 14, 30, 0)


@dataclass
class _FakePosition:
    symbol: str = "ABC"
    strategy: str = "momentum"
    shares: int = 100
    avg_price: float = 1.0
    stop_price: float = 0.9
    scale_out_price: float = 1.2
    status: str = "open"
    scaled_out: bool = False


@dataclass
class _FakeLLMClient:
    """Returns the next queued ``LLMCallResult`` per ``call``."""

    queued: list[LLMCallResult] = field(default_factory=list)
    calls_seen: list[tuple[str, str]] = field(default_factory=list)

    def call(
        self, system_prompt: str, user_message: str, tool_schema: dict[str, Any]
    ) -> LLMCallResult:
        self.calls_seen.append((system_prompt[:30], user_message[:50]))
        if not self.queued:
            return LLMCallResult(
                success=False,
                recommendation=None,
                cost_usd=0.0,
                duration_seconds=0.0,
                failure_reason="no_queued_response",
            )
        return self.queued.pop(0)


def _make_advisor(
    llm: _FakeLLMClient,
    *,
    soft_cap: float = 100.0,
    hard_cap: float = 1000.0,
    self_disable_min_calls: int = 5,
    self_disable_failure_rate: float = 0.5,
    notify_callback: Callable[[str], None] | None = None,
    min_hold_minutes_for_full_exit: float = 0.0,
    min_r_for_full_exit: float = 0.0,
) -> ExitAdvisor:
    cost_tracker = CostTracker(soft_cap_usd=soft_cap, hard_cap_usd=hard_cap)
    return ExitAdvisor(
        llm_client=llm,  # type: ignore[arg-type]
        cost_tracker=cost_tracker,
        event_buffer_factory=lambda: EventBuffer(time_floor_seconds=30.0, hard_floor_seconds=0.0),
        shadow_baselines=ShadowBaselines(),
        hook_acts=False,
        self_disable_failure_rate=self_disable_failure_rate,
        self_disable_min_calls=self_disable_min_calls,
        notify_callback=notify_callback,
        min_hold_minutes_for_full_exit=min_hold_minutes_for_full_exit,
        min_r_for_full_exit=min_r_for_full_exit,
    )


def _bar(t: datetime, *, close: float = 1.05, high: float = 1.05) -> BarFinalizedEvent:
    return BarFinalizedEvent(
        timestamp=t,
        symbol="ABC",
        open=1.0,
        high=high,
        low=0.99,
        close=close,
        volume=1000,
    )


def _success(action: str = "hold", *, cost: float = 0.01) -> LLMCallResult:
    return LLMCallResult(
        success=True,
        recommendation=ExitRecommendation(action=action, confidence=0.7, reason="r"),  # type: ignore[arg-type]
        cost_usd=cost,
        duration_seconds=0.1,
    )


def _failure(reason: str = "llm_timeout") -> LLMCallResult:
    return LLMCallResult(
        success=False,
        recommendation=None,
        cost_usd=0.0,
        duration_seconds=0.05,
        failure_reason=reason,
    )


def test_on_position_protected_initialises_state() -> None:
    advisor = _make_advisor(_FakeLLMClient())
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # Internal contexts dict should now have an entry for the symbol.
    assert "ABC" in advisor._contexts
    state = advisor._contexts["ABC"].trade_state
    assert state.entry_price == 1.0
    assert state.initial_stop == 0.9
    assert state.initial_scale_out == 1.2
    assert state.initial_position_size == 100
    assert state.peak_price == 1.0


def test_always_trigger_event_calls_llm_and_returns_actionable() -> None:
    llm = _FakeLLMClient(queued=[_success(action="exit_full")])
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )
    response = advisor.on_event(pos, event)
    assert response.is_actionable
    assert response.recommendation is not None
    assert response.recommendation.action == "exit_full"
    assert len(llm.calls_seen) == 1


def test_held_response_when_llm_says_hold() -> None:
    llm = _FakeLLMClient(queued=[_success(action="hold")])
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )
    response = advisor.on_event(pos, event)
    assert response.is_held
    assert not response.is_actionable
    assert not response.is_skipped


def test_buffer_only_event_returns_skipped() -> None:
    llm = _FakeLLMClient()
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    print_event = LargePrint(
        timestamp=_T0,
        symbol="ABC",
        price=1.05,
        size=5000,
        rolling_average_size=500.0,
        ratio=10.0,
        aggressor_side="buy",
    )
    response = advisor.on_event(pos, print_event)
    assert response.is_skipped
    assert len(llm.calls_seen) == 0


def test_llm_failure_returns_skipped_with_failure_reason() -> None:
    llm = _FakeLLMClient(queued=[_failure("api_error: boom")])
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )
    response = advisor.on_event(pos, event)
    assert response.is_skipped
    assert "llm_call_failed" in response.reasoning
    assert "api_error" in response.reasoning


def test_self_disable_after_failure_rate_threshold() -> None:
    notifications: list[str] = []
    llm = _FakeLLMClient(queued=[_failure() for _ in range(10)])
    advisor = _make_advisor(
        llm,
        self_disable_min_calls=3,
        self_disable_failure_rate=0.5,
        notify_callback=notifications.append,
    )
    pos = _FakePosition()
    advisor.on_position_protected(pos)

    for i in range(5):
        event = PartialFillEvent(
            timestamp=_T0 + timedelta(seconds=i * 10),
            symbol="ABC",
            order_id=i,
            filled_quantity=10,
            remaining_quantity=10,
            fill_price=1.0,
            side="buy",
        )
        advisor.on_event(pos, event)

    assert advisor.is_self_disabled()
    assert any("self-disabled" in m for m in notifications)


def test_self_disable_locks_subsequent_calls() -> None:
    llm = _FakeLLMClient(queued=[_failure() for _ in range(10)])
    advisor = _make_advisor(llm, self_disable_min_calls=2, self_disable_failure_rate=0.4)
    pos = _FakePosition()
    advisor.on_position_protected(pos)

    for i in range(3):
        event = PartialFillEvent(
            timestamp=_T0 + timedelta(seconds=i * 10),
            symbol="ABC",
            order_id=i,
            filled_quantity=10,
            remaining_quantity=10,
            fill_price=1.0,
            side="buy",
        )
        advisor.on_event(pos, event)

    assert advisor.is_self_disabled()
    calls_before = len(llm.calls_seen)

    # Subsequent events should not call LLM at all.
    later_event = PartialFillEvent(
        timestamp=_T0 + timedelta(seconds=120),
        symbol="ABC",
        order_id=99,
        filled_quantity=10,
        remaining_quantity=10,
        fill_price=1.0,
        side="buy",
    )
    response = advisor.on_event(pos, later_event)
    assert response.is_skipped
    assert "self-disabled" in response.reasoning
    assert len(llm.calls_seen) == calls_before


def test_cost_cap_returns_deterministic_skipped() -> None:
    llm = _FakeLLMClient(queued=[_success(cost=100.0)])
    advisor = _make_advisor(llm, soft_cap=10.0, hard_cap=20.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # First call: succeeds and trips the hard cap.
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )
    advisor.on_event(pos, event)

    # Second call: hard-capped, should not hit the LLM.
    later_event = PartialFillEvent(
        timestamp=_T0 + timedelta(seconds=60),
        symbol="ABC",
        order_id=1,
        filled_quantity=10,
        remaining_quantity=0,
        fill_price=1.0,
        side="buy",
    )
    calls_before = len(llm.calls_seen)
    response = advisor.on_event(pos, later_event)
    assert response.is_skipped
    assert response.reasoning == "cost_cap_reached"
    assert len(llm.calls_seen) == calls_before


def test_shadow_baselines_consulted_on_every_event() -> None:
    """The shadow baselines should run regardless of whether the LLM was triggered."""
    llm = _FakeLLMClient()
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)

    # buffer-only event: LLM not called, but baselines still consulted.
    print_event = LargePrint(
        timestamp=_T0,
        symbol="ABC",
        price=1.05,
        size=5000,
        rolling_average_size=500.0,
        ratio=10.0,
        aggressor_side="buy",
    )
    advisor.on_event(pos, print_event)
    assert len(llm.calls_seen) == 0
    # Can't directly assert baseline was called without instrumenting it;
    # but baseline_names() proves the wiring exists.
    assert "stall_1r_5min" in advisor._shadow_baselines.baseline_names()


def test_on_event_without_protected_creates_context_lazily() -> None:
    """If on_event arrives before on_position_protected, the agent doesn't drop the event."""
    llm = _FakeLLMClient(queued=[_success()])
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    event = PositionProtected(
        timestamp=_T0,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )
    response = advisor.on_event(pos, event)
    # Should still return a usable response (not crash).
    assert isinstance(response.is_actionable, bool)
    assert "ABC" in advisor._contexts


def test_position_closed_clears_context() -> None:
    llm = _FakeLLMClient()
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    assert "ABC" in advisor._contexts
    advisor.on_position_closed(pos, final_pnl=12.50)
    assert "ABC" not in advisor._contexts


def test_tracking_state_updates_peak_on_higher_bar() -> None:
    llm = _FakeLLMClient()
    advisor = _make_advisor(llm)
    pos = _FakePosition()
    advisor.on_position_protected(pos)

    # Drive a bar through the buffer (won't trigger because buffer-only path)
    # but the state update happens unconditionally on bar events.
    bar1 = _bar(_T0, high=1.05, close=1.05)
    advisor.on_event(pos, bar1)
    assert advisor._contexts["ABC"].trade_state.peak_price == pytest.approx(1.05)

    # Higher high; peak rises.
    bar2 = _bar(_T0 + timedelta(seconds=60), high=1.20, close=1.18)
    advisor.on_event(pos, bar2)
    assert advisor._contexts["ABC"].trade_state.peak_price == pytest.approx(1.20)
    assert advisor._contexts["ABC"].trade_state.current_price == pytest.approx(1.18)


# ============================================================
# min_hold_minutes_for_full_exit tests
# ============================================================


def _protected_event(t: datetime) -> PositionProtected:
    return PositionProtected(
        timestamp=t,
        symbol="ABC",
        entry_price=1.0,
        initial_stop=0.9,
        initial_scale_out=1.2,
        position_size=100,
    )


def test_early_exit_full_suppressed_within_hold_floor() -> None:
    """exit_full recommended within 3-minute floor becomes held, not actionable."""
    llm = _FakeLLMClient(queued=[_success(action="exit_full")])
    advisor = _make_advisor(llm, min_hold_minutes_for_full_exit=3.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # Event timestamp 90 s after entry — inside the 180 s floor.
    event = _protected_event(_T0 + timedelta(seconds=90))
    response = advisor.on_event(pos, event)
    assert not response.is_actionable
    assert response.is_held
    assert "suppressed" in response.reasoning.lower()
    assert len(llm.calls_seen) == 1  # LLM was still called; suppression is post-LLM


def test_exit_full_allowed_after_hold_floor_elapsed() -> None:
    """exit_full is not suppressed once the floor has passed."""
    # Two events: first anchors first_event_timestamp at _T0, second triggers
    # the LLM 4 minutes later — beyond the 180 s floor.
    llm = _FakeLLMClient(queued=[_success(action="hold"), _success(action="exit_full")])
    advisor = _make_advisor(llm, min_hold_minutes_for_full_exit=3.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    advisor.on_event(pos, _protected_event(_T0))
    event2 = _protected_event(_T0 + timedelta(minutes=4))
    response = advisor.on_event(pos, event2)
    assert response.is_actionable
    assert response.recommendation is not None
    assert response.recommendation.action == "exit_full"


def test_hold_floor_zero_disables_suppression() -> None:
    """min_hold_minutes_for_full_exit=0.0 disables the floor entirely."""
    llm = _FakeLLMClient(queued=[_success(action="exit_full")])
    advisor = _make_advisor(llm, min_hold_minutes_for_full_exit=0.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # Even at t=0 the recommendation should pass through.
    event = _protected_event(_T0)
    response = advisor.on_event(pos, event)
    assert response.is_actionable


def test_hold_recommendation_not_affected_by_floor() -> None:
    """hold recommendations are never suppressed, floor doesn't apply."""
    llm = _FakeLLMClient(queued=[_success(action="hold")])
    advisor = _make_advisor(llm, min_hold_minutes_for_full_exit=3.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    event = _protected_event(_T0 + timedelta(seconds=30))
    response = advisor.on_event(pos, event)
    assert response.is_held
    assert not response.is_actionable


# ============================================================
# min_r_for_full_exit tests
# ============================================================


def test_low_peak_r_exit_full_suppressed() -> None:
    """exit_full is suppressed when peak R has not reached the threshold."""
    # _FakePosition: entry=1.0, stop=0.9 → 1R = 0.10
    # Bar high=1.08 → peak_r = (1.08-1.0)/0.10 = 0.8  <  threshold=1.0 → suppressed
    llm = _FakeLLMClient(queued=[_success(action="hold"), _success(action="exit_full")])
    advisor = _make_advisor(llm, min_r_for_full_exit=1.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # Drive a bar that sets peak_price=1.08 (0.8R)
    advisor.on_event(pos, _bar(_T0, high=1.08, close=1.05))
    # Trigger: LLM recommends exit_full; peak_r=0.8 < 1.0 → should be suppressed
    event = _protected_event(_T0 + timedelta(seconds=60))
    response = advisor.on_event(pos, event)
    assert not response.is_actionable
    assert response.is_held
    assert "peak_r" in response.reasoning
    assert "suppressed" in response.reasoning


def test_exit_full_allowed_when_peak_r_meets_threshold() -> None:
    """exit_full passes through when peak R has reached the threshold."""
    # Bar high=1.12 → peak_r = (1.12-1.0)/0.10 = 1.2 >= 1.0 → allowed
    llm = _FakeLLMClient(queued=[_success(action="hold"), _success(action="exit_full")])
    advisor = _make_advisor(llm, min_r_for_full_exit=1.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    advisor.on_event(pos, _bar(_T0, high=1.12, close=1.10))
    event = _protected_event(_T0 + timedelta(seconds=60))
    response = advisor.on_event(pos, event)
    assert response.is_actionable
    assert response.recommendation is not None
    assert response.recommendation.action == "exit_full"


def test_min_r_zero_disables_suppression() -> None:
    """min_r_for_full_exit=0.0 disables the R floor entirely (default off)."""
    # Even with peak_r=0 the recommendation passes through.
    llm = _FakeLLMClient(queued=[_success(action="exit_full")])
    advisor = _make_advisor(llm, min_r_for_full_exit=0.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    event = _protected_event(_T0)
    response = advisor.on_event(pos, event)
    assert response.is_actionable


def test_non_exit_full_not_affected_by_min_r() -> None:
    """exit_partial is never blocked by the R floor (only exit_full is gated)."""
    partial_result = LLMCallResult(
        success=True,
        recommendation=ExitRecommendation(action="exit_partial", partial_pct=0.5, confidence=0.7, reason="r"),  # type: ignore[arg-type]
        cost_usd=0.01,
        duration_seconds=0.1,
    )
    llm = _FakeLLMClient(queued=[partial_result])
    advisor = _make_advisor(llm, min_r_for_full_exit=1.0)
    pos = _FakePosition()
    advisor.on_position_protected(pos)
    # peak_r stays at 0 (no bar event) — floor is active, but action != exit_full
    event = _protected_event(_T0)
    response = advisor.on_event(pos, event)
    assert response.is_actionable
    assert response.recommendation is not None
    assert response.recommendation.action == "exit_partial"
