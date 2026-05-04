"""Phase 11 — type validation for the exit-advisor surface.

Covers ExitRecommendation, AdvisorResponse, BarFinalizedEvent, and the
runtime-checkable ExitAdvisorHook protocol. Type-only tests, no I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from bot.exit_advisor.core.types import (
    AdvisorResponse,
    BarFinalizedEvent,
    Event,
    ExitAdvisorHook,
    ExitRecommendation,
)

# ---------------------------------------------------------------------------
# ExitRecommendation
# ---------------------------------------------------------------------------


def test_exit_recommendation_hold_minimal_constructs() -> None:
    """``hold`` is the canonical minimal recommendation: no extras required."""
    rec = ExitRecommendation(action="hold")
    assert rec.action == "hold"
    assert rec.partial_pct == 0.0
    assert rec.new_stop_price is None
    assert rec.confidence == 1.0


def test_exit_recommendation_exit_full_constructs() -> None:
    """``exit_full`` doesn't require partial_pct or new_stop_price."""
    rec = ExitRecommendation(action="exit_full", reason="9 ema break", source="advisor_v1")
    assert rec.action == "exit_full"
    assert rec.reason == "9 ema break"
    assert rec.source == "advisor_v1"


def test_exit_recommendation_exit_partial_requires_partial_pct() -> None:
    """``exit_partial`` without partial_pct raises (zero/missing is ambiguous)."""
    with pytest.raises(ValueError, match="partial_pct"):
        ExitRecommendation(action="exit_partial")
    with pytest.raises(ValueError, match="partial_pct"):
        ExitRecommendation(action="exit_partial", partial_pct=0.0)
    with pytest.raises(ValueError, match="partial_pct"):
        ExitRecommendation(action="exit_partial", partial_pct=1.5)


def test_exit_recommendation_exit_partial_valid_range() -> None:
    """``exit_partial`` with valid partial_pct constructs."""
    rec = ExitRecommendation(action="exit_partial", partial_pct=0.5)
    assert rec.partial_pct == 0.5
    rec_full = ExitRecommendation(action="exit_partial", partial_pct=1.0)
    assert rec_full.partial_pct == 1.0


def test_exit_recommendation_partial_pct_only_with_exit_partial() -> None:
    """A non-zero partial_pct on a non-partial action is rejected (incoherent)."""
    with pytest.raises(ValueError, match="partial_pct must be 0.0"):
        ExitRecommendation(action="hold", partial_pct=0.5)
    with pytest.raises(ValueError, match="partial_pct must be 0.0"):
        ExitRecommendation(action="exit_full", partial_pct=0.3)


def test_exit_recommendation_tighten_stop_requires_new_stop_price() -> None:
    """``tighten_stop`` without a positive new_stop_price raises."""
    with pytest.raises(ValueError, match="new_stop_price"):
        ExitRecommendation(action="tighten_stop")
    with pytest.raises(ValueError, match="new_stop_price"):
        ExitRecommendation(action="tighten_stop", new_stop_price=0.0)
    with pytest.raises(ValueError, match="new_stop_price"):
        ExitRecommendation(action="tighten_stop", new_stop_price=-1.0)


def test_exit_recommendation_tighten_stop_valid_constructs() -> None:
    """``tighten_stop`` with positive new_stop_price constructs."""
    rec = ExitRecommendation(action="tighten_stop", new_stop_price=2.45)
    assert rec.new_stop_price == 2.45


def test_exit_recommendation_new_stop_only_with_tighten_stop() -> None:
    """A new_stop_price on a non-tighten action is rejected (incoherent)."""
    with pytest.raises(ValueError, match="new_stop_price must be None"):
        ExitRecommendation(action="hold", new_stop_price=2.0)
    with pytest.raises(ValueError, match="new_stop_price must be None"):
        ExitRecommendation(action="exit_full", new_stop_price=2.0)


def test_exit_recommendation_confidence_bounds() -> None:
    """Confidence must lie in [0.0, 1.0]; out-of-range raises."""
    ExitRecommendation(action="hold", confidence=0.0)
    ExitRecommendation(action="hold", confidence=0.5)
    ExitRecommendation(action="hold", confidence=1.0)
    with pytest.raises(ValueError, match="confidence"):
        ExitRecommendation(action="hold", confidence=-0.1)
    with pytest.raises(ValueError, match="confidence"):
        ExitRecommendation(action="hold", confidence=1.1)


def test_exit_recommendation_is_frozen() -> None:
    """Frozen dataclass — mutation must raise FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    rec = ExitRecommendation(action="hold")
    with pytest.raises(FrozenInstanceError):
        rec.action = "exit_full"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AdvisorResponse three-state semantics
# ---------------------------------------------------------------------------


def test_advisor_response_default_is_skipped() -> None:
    """Empty AdvisorResponse() means skipped — no recommendation, no eval."""
    resp = AdvisorResponse()
    assert resp.is_skipped
    assert not resp.is_held
    assert not resp.is_actionable


def test_advisor_response_held() -> None:
    """held: evaluation_performed=True with no recommendation."""
    resp = AdvisorResponse(evaluation_performed=True, reasoning="all green, hold runner")
    assert resp.is_held
    assert not resp.is_skipped
    assert not resp.is_actionable


def test_advisor_response_actionable() -> None:
    """actionable: recommendation set + evaluation_performed=True."""
    rec = ExitRecommendation(action="exit_full", reason="9ema break")
    resp = AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
    assert resp.is_actionable
    assert not resp.is_held
    assert not resp.is_skipped
    assert resp.recommendation is rec


def test_advisor_response_recommendation_without_evaluation_rejected() -> None:
    """A recommendation without evaluation is incoherent — reject at construction."""
    rec = ExitRecommendation(action="hold")
    with pytest.raises(ValueError, match="evaluation_performed=True"):
        AdvisorResponse(recommendation=rec, evaluation_performed=False)


# ---------------------------------------------------------------------------
# BarFinalizedEvent
# ---------------------------------------------------------------------------


def test_bar_finalized_event_constructs() -> None:
    """BarFinalizedEvent carries OHLCV plus the inherited symbol/timestamp."""
    ts = datetime(2026, 4, 30, 14, 31, tzinfo=UTC)
    ev = BarFinalizedEvent(
        timestamp=ts,
        symbol="ZENA",
        open=2.20,
        high=2.31,
        low=2.18,
        close=2.27,
        volume=125_000.0,
    )
    assert ev.symbol == "ZENA"
    assert ev.timestamp == ts
    assert ev.close == 2.27
    assert ev.extra == {}
    assert isinstance(ev, Event)


def test_bar_finalized_event_extra_opt_in() -> None:
    """``extra`` is the escape hatch for derived state without enlarging the surface."""
    ev = BarFinalizedEvent(
        timestamp=datetime.now(UTC),
        symbol="X",
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        volume=1000.0,
        extra={"ema9": 1.02, "running_r": 0.5},
    )
    assert ev.extra["ema9"] == 1.02


# ---------------------------------------------------------------------------
# ExitAdvisorHook protocol — runtime_checkable smoke
# ---------------------------------------------------------------------------


class _NoOpAdvisor:
    """Minimal advisor — satisfies the protocol without doing anything."""

    def on_position_protected(self, position: Any) -> None:
        return None

    def on_event(self, position: Any, event: Event) -> AdvisorResponse:
        return AdvisorResponse()

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        return None


def test_no_op_advisor_satisfies_protocol() -> None:
    """A minimal class with the three methods is a structural ExitAdvisorHook."""
    advisor = _NoOpAdvisor()
    assert isinstance(advisor, ExitAdvisorHook)


class _IncompleteAdvisor:
    """Missing on_event — should NOT satisfy the protocol."""

    def on_position_protected(self, position: Any) -> None:
        return None

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        return None


def test_incomplete_advisor_fails_protocol_check() -> None:
    """Missing methods on a candidate fail the runtime-checkable isinstance check."""
    assert not isinstance(_IncompleteAdvisor(), ExitAdvisorHook)
