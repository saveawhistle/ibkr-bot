"""Unit tests for CostTracker."""

from __future__ import annotations

import pytest

from bot.exit_advisor.advisor.cost_tracker import CostTracker


def test_constructor_validates_caps() -> None:
    with pytest.raises(ValueError):
        CostTracker(soft_cap_usd=0.0, hard_cap_usd=1.0)
    with pytest.raises(ValueError):
        CostTracker(soft_cap_usd=1.0, hard_cap_usd=0.0)
    with pytest.raises(ValueError, match="strictly less"):
        CostTracker(soft_cap_usd=10.0, hard_cap_usd=10.0)


def test_session_cost_starts_zero() -> None:
    tracker = CostTracker()
    assert tracker.session_cost_usd() == 0.0
    assert not tracker.is_hard_capped()
    assert not tracker.soft_warning_fired()


def test_record_cost_accumulates() -> None:
    tracker = CostTracker(soft_cap_usd=10.0, hard_cap_usd=50.0)
    tracker.record_cost(1.5)
    tracker.record_cost(2.5)
    assert tracker.session_cost_usd() == pytest.approx(4.0)


def test_record_cost_rejects_negative() -> None:
    tracker = CostTracker()
    with pytest.raises(ValueError):
        tracker.record_cost(-0.01)


def test_soft_warning_fires_once_on_first_crossing() -> None:
    seen: list[str] = []
    tracker = CostTracker(soft_cap_usd=1.0, hard_cap_usd=10.0, notify_callback=seen.append)
    tracker.record_cost(0.5)
    assert not tracker.soft_warning_fired()
    assert seen == []

    tracker.record_cost(0.5)  # crosses soft cap (1.0)
    assert tracker.soft_warning_fired()
    assert len(seen) == 1
    assert "soft warning" in seen[0]

    tracker.record_cost(2.0)
    assert len(seen) == 1, "soft warning must not refire after first crossing"


def test_hard_cap_activates_and_latches() -> None:
    seen: list[str] = []
    tracker = CostTracker(soft_cap_usd=1.0, hard_cap_usd=2.0, notify_callback=seen.append)
    tracker.record_cost(1.5)
    assert not tracker.is_hard_capped()

    tracker.record_cost(0.6)  # crosses hard cap
    assert tracker.is_hard_capped()
    hard_cap_msgs = [m for m in seen if "HARD CAP" in m]
    assert len(hard_cap_msgs) == 1

    tracker.record_cost(100.0)  # ignored once latched
    assert tracker.session_cost_usd() == pytest.approx(2.1)
    hard_cap_msgs = [m for m in seen if "HARD CAP" in m]
    assert len(hard_cap_msgs) == 1


def test_notifier_failure_does_not_propagate() -> None:
    def _broken(_msg: str) -> None:
        raise RuntimeError("notifier down")

    tracker = CostTracker(soft_cap_usd=1.0, hard_cap_usd=2.0, notify_callback=_broken)
    # Should not raise even though the callback explodes.
    tracker.record_cost(1.5)
    tracker.record_cost(0.6)
    assert tracker.is_hard_capped()
