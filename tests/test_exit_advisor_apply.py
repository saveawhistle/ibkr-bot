"""Phase 11 — RecommendationApplier dispatch + safety semantics.

Verifies that:

* ``hold`` is a no-op (returns False).
* ``exit_full`` routes through ``execute_advisor_exit`` and returns its
  submission status.
* ``exit_partial`` and ``tighten_stop`` are rejected in Phase 11 with
  a structured warning (no executor primitive yet).
* An invalid action string is rejected loudly.
"""

from __future__ import annotations

from typing import Any

import pytest
from structlog.testing import capture_logs

from bot.exit_advisor.core.types import ExitRecommendation
from bot.exit_advisor.hook.apply import RecommendationApplier


class _FakeTradeManager:
    """Records calls to ``execute_advisor_exit`` so the applier's routing is observable."""

    def __init__(self, *, return_value: bool = True) -> None:
        self.return_value = return_value
        self.calls: list[tuple[Any, float, str]] = []

    async def execute_advisor_exit(
        self, position: Any, *, exit_price: float, reason: str
    ) -> bool:
        self.calls.append((position, exit_price, reason))
        return self.return_value


class _StubPosition:
    """Minimal duck-typed position for routing tests."""

    symbol = "ZENA"
    strategy = "momentum"
    shares = 100
    avg_price = 2.20
    stop_price = 2.15
    scale_out_price = 2.30
    status = "open"
    scaled_out = False


@pytest.mark.asyncio
async def test_apply_hold_is_noop_returns_false() -> None:
    """``hold`` doesn't call the trade manager and returns False."""
    tm = _FakeTradeManager()
    applier = RecommendationApplier(tm)
    rec = ExitRecommendation(action="hold", reason="all green")
    result = await applier.apply(rec, _StubPosition(), exit_price=2.27)
    assert result is False
    assert tm.calls == []


@pytest.mark.asyncio
async def test_apply_exit_full_routes_to_trade_manager() -> None:
    """``exit_full`` calls execute_advisor_exit with the exit_price + reason."""
    tm = _FakeTradeManager(return_value=True)
    applier = RecommendationApplier(tm)
    rec = ExitRecommendation(
        action="exit_full", reason="9ema break", source="advisor_v1"
    )
    pos = _StubPosition()
    result = await applier.apply(rec, pos, exit_price=2.18)
    assert result is True
    assert len(tm.calls) == 1
    called_pos, called_price, called_reason = tm.calls[0]
    assert called_pos is pos
    assert called_price == 2.18
    assert called_reason == "9ema break"


@pytest.mark.asyncio
async def test_apply_exit_full_propagates_submission_failure() -> None:
    """If execute_advisor_exit returns False (e.g. position already closed),
    the applier reports the same."""
    tm = _FakeTradeManager(return_value=False)
    applier = RecommendationApplier(tm)
    rec = ExitRecommendation(action="exit_full", reason="x")
    result = await applier.apply(rec, _StubPosition(), exit_price=2.18)
    assert result is False


@pytest.mark.asyncio
async def test_apply_exit_partial_rejected_with_warning() -> None:
    """``exit_partial`` has no executor primitive in Phase 11 → warning + False."""
    tm = _FakeTradeManager()
    applier = RecommendationApplier(tm)
    rec = ExitRecommendation(action="exit_partial", partial_pct=0.5, reason="x")
    with capture_logs() as captured:
        result = await applier.apply(rec, _StubPosition(), exit_price=2.27)
    assert result is False
    assert tm.calls == []  # no order submitted
    rejections = [
        e for e in captured if e["event"] == "exit_advisor.recommendation_rejected_unsupported"
    ]
    assert len(rejections) == 1
    assert rejections[0]["action"] == "exit_partial"
    assert rejections[0]["partial_pct"] == 0.5


@pytest.mark.asyncio
async def test_apply_tighten_stop_rejected_with_warning() -> None:
    """``tighten_stop`` has no executor primitive in Phase 11 → warning + False."""
    tm = _FakeTradeManager()
    applier = RecommendationApplier(tm)
    rec = ExitRecommendation(action="tighten_stop", new_stop_price=2.20, reason="x")
    with capture_logs() as captured:
        result = await applier.apply(rec, _StubPosition(), exit_price=2.27)
    assert result is False
    assert tm.calls == []
    rejections = [
        e for e in captured if e["event"] == "exit_advisor.recommendation_rejected_unsupported"
    ]
    assert len(rejections) == 1
    assert rejections[0]["action"] == "tighten_stop"
    assert rejections[0]["new_stop_price"] == 2.20


@pytest.mark.asyncio
async def test_apply_unknown_action_rejected_loudly() -> None:
    """A bad action via Any-erasure is rejected with an ERROR log."""
    tm = _FakeTradeManager()
    applier = RecommendationApplier(tm)
    # Bypass the dataclass validator — simulating a malicious / buggy advisor
    # that produced an invalid action through Any-typed reflection.
    rec = ExitRecommendation(action="hold")
    object.__setattr__(rec, "action", "do_a_barrel_roll")
    with capture_logs() as captured:
        result = await applier.apply(rec, _StubPosition(), exit_price=2.27)
    assert result is False
    rejections = [
        e
        for e in captured
        if e["event"] == "exit_advisor.recommendation_rejected_unknown_action"
    ]
    assert len(rejections) == 1
