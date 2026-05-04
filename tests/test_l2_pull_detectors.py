"""BidPulled / OfferPulled detector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bot.exit_advisor.core.events import BidPulled, OfferPulled
from bot.exit_advisor.detectors.l2.pulls import (
    BidPulledDetector,
    OfferPulledDetector,
)
from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


def _ts(ms: int) -> datetime:
    """Microsecond-aware timestamps so the ``lookback_ms`` window is
    testable at sub-second granularity."""
    return datetime(2026, 5, 5, 13, 30, 0, tzinfo=UTC) + timedelta(milliseconds=ms)


def test_bid_pulled_when_no_offsetting_print() -> None:
    """Insert + delete with no print at the price → pulled."""
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100)
    tracker.consume(insert)
    detector.consume(insert, tracker.get_state())

    delete = L2BookUpdate(_ts(50), "X", "bid", "delete", 0, 10.00, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert len(events) == 1
    assert isinstance(events[0], BidPulled)
    assert events[0].price == 10.00
    assert events[0].size_pulled == 100


def test_bid_consumed_does_not_fire() -> None:
    """Delete with offsetting print(s) → consumed → no event."""
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100)
    tracker.consume(insert)
    detector.consume(insert, tracker.get_state())

    # 100 shares hit the bid via two prints.
    tracker.consume(L2Print(_ts(20), "X", 10.00, 60, "sell"))
    tracker.consume(L2Print(_ts(40), "X", 10.00, 40, "sell"))

    delete = L2BookUpdate(_ts(50), "X", "bid", "delete", 0, 10.00, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert events == []


def test_partial_consumption_treated_as_pull() -> None:
    """Documented choice: <80% offsetting prints → pull. The withdrawal
    of the remainder is the more interesting signal."""
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100)
    tracker.consume(insert)
    detector.consume(insert, tracker.get_state())

    # 30 shares hit the bid; 70 still resting when delete arrives.
    tracker.consume(L2Print(_ts(20), "X", 10.00, 30, "sell"))

    delete = L2BookUpdate(_ts(50), "X", "bid", "delete", 0, 10.00, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert len(events) == 1
    assert isinstance(events[0], BidPulled)


def test_lookback_window_respected() -> None:
    """Print outside the lookback window doesn't count as offsetting —
    the level is classified as pulled."""
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100)
    tracker.consume(insert)
    detector.consume(insert, tracker.get_state())

    # Print 200ms before delete, well outside the 100ms lookback.
    tracker.consume(L2Print(_ts(50), "X", 10.00, 100, "sell"))

    delete = L2BookUpdate(_ts(300), "X", "bid", "delete", 0, 10.00, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert len(events) == 1
    assert isinstance(events[0], BidPulled)


def test_offer_pulled_symmetric() -> None:
    detector = OfferPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.05, 100)
    tracker.consume(insert)
    detector.consume(insert, tracker.get_state())

    delete = L2BookUpdate(_ts(50), "X", "ask", "delete", 0, 10.05, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert len(events) == 1
    assert isinstance(events[0], OfferPulled)


def test_bid_detector_ignores_ask_side_events() -> None:
    """Each side has its own detector — they shouldn't fire on the
    opposite side's events."""
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()

    insert = L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.05, 100)
    tracker.consume(insert)
    delete = L2BookUpdate(_ts(50), "X", "ask", "delete", 0, 10.05, 0)
    tracker.consume(delete)
    events = detector.consume(delete, tracker.get_state())

    assert events == []


def test_prints_alone_do_not_fire_pull_detector() -> None:
    detector = BidPulledDetector(symbol="X", lookback_ms=100)
    tracker = BookStateTracker()
    tracker.consume(L2Print(_ts(0), "X", 10.00, 100, "sell"))
    events = detector.consume(L2Print(_ts(0), "X", 10.00, 100, "sell"), tracker.get_state())
    assert events == []
