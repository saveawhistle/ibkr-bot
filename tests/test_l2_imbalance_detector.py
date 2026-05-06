"""ImbalanceDetector tests."""

from __future__ import annotations

from datetime import UTC, datetime

from bot.exit_advisor.core.events import ImbalanceEvent
from bot.exit_advisor.detectors.l2.imbalance import ImbalanceDetector
from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2BookUpdate


def _ts() -> datetime:
    return datetime(2026, 5, 5, 13, 30, 0, tzinfo=UTC)


def _build_book(bid_sizes: list[int], ask_sizes: list[int]) -> BookStateTracker:
    """Build a synthetic book with the given top-K sizes per side."""
    tracker = BookStateTracker()
    for i, size in enumerate(bid_sizes):
        tracker.consume(L2BookUpdate(_ts(), "X", "bid", "insert", i, 10.00 - i * 0.01, size))
    for i, size in enumerate(ask_sizes):
        tracker.consume(L2BookUpdate(_ts(), "X", "ask", "insert", i, 10.05 + i * 0.01, size))
    return tracker


def test_imbalance_fires_when_ratio_exceeds_threshold() -> None:
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=5)
    tracker = _build_book(
        bid_sizes=[1000, 1000, 1000, 1000, 1000],  # total 5000
        ask_sizes=[200, 200, 200, 200, 200],  # total 1000 → ratio 5x
    )
    # Trigger an update so the detector evaluates.
    trigger = L2BookUpdate(_ts(), "X", "bid", "update", 0, 10.00, 1000)
    tracker.consume(trigger)
    events = detector.consume(trigger, tracker.get_state())
    fires = [e for e in events if isinstance(e, ImbalanceEvent)]
    assert len(fires) == 1
    assert fires[0].favored_side == "bid"
    assert fires[0].ratio == 5.0


def test_imbalance_correctly_identifies_ask_favored() -> None:
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=5)
    tracker = _build_book(
        bid_sizes=[100, 100, 100, 100, 100],  # 500
        ask_sizes=[1000, 1000, 1000, 1000, 1000],  # 5000 → ratio 10x
    )
    trigger = L2BookUpdate(_ts(), "X", "ask", "update", 0, 10.05, 1000)
    tracker.consume(trigger)
    events = detector.consume(trigger, tracker.get_state())
    assert isinstance(events[0], ImbalanceEvent) and events[0].favored_side == "ask"


def test_imbalance_below_threshold_does_not_fire() -> None:
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=5)
    tracker = _build_book(
        bid_sizes=[1000, 1000, 1000, 1000, 1000],  # 5000
        ask_sizes=[2500, 2500, 2500, 2500, 2500],  # 12500 → ratio 2.5x, below 3.0
    )
    trigger = L2BookUpdate(_ts(), "X", "ask", "update", 0, 10.05, 2500)
    tracker.consume(trigger)
    events = detector.consume(trigger, tracker.get_state())
    assert not any(isinstance(e, ImbalanceEvent) for e in events)


def test_levels_to_sum_respected() -> None:
    """levels_to_sum=2 means only the top 2 levels per side count.
    A book with massive deeper levels won't trigger if the top-2 are balanced."""
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=2)
    tracker = _build_book(
        bid_sizes=[100, 100, 1000, 1000, 1000],
        ask_sizes=[100, 100, 1000, 1000, 1000],
    )
    trigger = L2BookUpdate(_ts(), "X", "bid", "update", 0, 10.00, 100)
    tracker.consume(trigger)
    events = detector.consume(trigger, tracker.get_state())
    assert not any(isinstance(e, ImbalanceEvent) for e in events)


def test_once_per_direction_crossing() -> None:
    """Sustained imbalance fires once. Until ratio drops below
    threshold (or favored side flips), no re-fire."""
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=5)
    tracker = _build_book(
        bid_sizes=[1000, 1000, 1000, 1000, 1000],
        ask_sizes=[200, 200, 200, 200, 200],
    )
    trigger1 = L2BookUpdate(_ts(), "X", "bid", "update", 0, 10.00, 1000)
    tracker.consume(trigger1)
    out = detector.consume(trigger1, tracker.get_state())
    trigger2 = L2BookUpdate(_ts(), "X", "bid", "update", 1, 9.99, 1000)
    tracker.consume(trigger2)
    out.extend(detector.consume(trigger2, tracker.get_state()))
    fires = [e for e in out if isinstance(e, ImbalanceEvent)]
    assert len(fires) == 1


def test_one_sided_book_does_not_fire() -> None:
    """A book missing one side entirely → no imbalance can be computed."""
    detector = ImbalanceDetector(symbol="X", threshold_ratio=3.0, levels_to_sum=5)
    tracker = _build_book(bid_sizes=[1000, 1000], ask_sizes=[])
    trigger = L2BookUpdate(_ts(), "X", "bid", "update", 0, 10.00, 1000)
    tracker.consume(trigger)
    events = detector.consume(trigger, tracker.get_state())
    assert events == []
