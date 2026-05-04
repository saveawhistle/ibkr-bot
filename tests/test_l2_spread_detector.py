"""SpreadEventDetector tests.

Note: BookStateTracker keys levels by price (matching IBKR's
``reqMktDepth`` semantics, where a top-of-book change comes through
as ``delete`` of the old price + ``insert`` of the new). To simulate
top-of-book moving in tests, we explicitly delete the old level
before inserting the new one — never just "update" to a different
price, which would leave the old level in place.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bot.exit_advisor.core.events import SpreadEvent
from bot.exit_advisor.detectors.l2.spread import SpreadEventDetector
from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2BookUpdate


def _ts(ms: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, 0, tzinfo=UTC) + timedelta(milliseconds=ms)


def _move_top_bid(
    tracker: Any, detector: Any, old_price: float, new_price: float, ts: Any
) -> list[Any]:
    """Simulate IBKR's "top-of-book moved" sequence: delete the old
    price, then insert the new. Returns whatever events the detector
    emitted while processing the pair."""
    out: list = []  # type: ignore[type-arg]
    delete = L2BookUpdate(ts, "X", "bid", "delete", 0, old_price, 0)
    tracker.consume(delete)
    out.extend(detector.consume(delete, tracker.get_state()))
    insert = L2BookUpdate(ts, "X", "bid", "insert", 0, new_price, 100)
    tracker.consume(insert)
    out.extend(detector.consume(insert, tracker.get_state()))
    return out


def test_widening_fires_above_threshold() -> None:
    detector = SpreadEventDetector(
        symbol="X",
        widening_ratio=2.0,
        tightening_ratio=0.5,
        rolling_window_events=5,
    )
    tracker = BookStateTracker()
    tracker.consume(L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.10, 100))
    out: list = []  # type: ignore[type-arg]
    # 5 warmup updates with bid=10.05 → spread=0.05.
    for i in range(5):
        evt = L2BookUpdate(_ts(i + 1), "X", "bid", "update", 0, 10.05, 100)
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    # Move top bid from 10.05 to 9.90 → spread widens to 0.20 = 4x average.
    out.extend(_move_top_bid(tracker, detector, 10.05, 9.90, _ts(10)))
    widenings = [e for e in out if isinstance(e, SpreadEvent) and e.direction == "widening"]
    assert len(widenings) == 1


def test_tightening_fires_below_threshold() -> None:
    detector = SpreadEventDetector(
        symbol="X",
        widening_ratio=2.0,
        tightening_ratio=0.5,
        rolling_window_events=5,
    )
    tracker = BookStateTracker()
    tracker.consume(L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.10, 100))
    out: list = []  # type: ignore[type-arg]
    # 5 warmup updates with bid=9.90 → spread=0.20 each.
    for i in range(5):
        evt = L2BookUpdate(_ts(i + 1), "X", "bid", "update", 0, 9.90, 100)
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    # Move top bid from 9.90 to 10.05 → spread tightens to 0.05 = 0.25x avg.
    out.extend(_move_top_bid(tracker, detector, 9.90, 10.05, _ts(10)))
    tightenings = [e for e in out if isinstance(e, SpreadEvent) and e.direction == "tightening"]
    assert len(tightenings) == 1


def test_no_event_during_warmup() -> None:
    """The first ``rolling_window_events`` are warmup — no event fires."""
    detector = SpreadEventDetector(
        symbol="X",
        widening_ratio=2.0,
        tightening_ratio=0.5,
        rolling_window_events=20,
    )
    tracker = BookStateTracker()
    tracker.consume(L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.10, 100))
    out: list = []  # type: ignore[type-arg]
    for i in range(5):
        evt = L2BookUpdate(_ts(i + 1), "X", "bid", "update", 0, 9.90, 100)
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    assert out == []


def test_once_per_direction_crossing() -> None:
    """Sustained widening fires once; the latch holds until the spread
    returns to the normal range."""
    detector = SpreadEventDetector(
        symbol="X",
        widening_ratio=2.0,
        tightening_ratio=0.5,
        rolling_window_events=5,
    )
    tracker = BookStateTracker()
    tracker.consume(L2BookUpdate(_ts(0), "X", "ask", "insert", 0, 10.10, 100))
    out: list = []  # type: ignore[type-arg]
    for i in range(5):
        evt = L2BookUpdate(_ts(i + 1), "X", "bid", "update", 0, 10.05, 100)
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    # Move bid down to 9.90 and stay wide for 3 more events.
    out.extend(_move_top_bid(tracker, detector, 10.05, 9.90, _ts(10)))
    for i in range(2):
        evt = L2BookUpdate(_ts(11 + i), "X", "bid", "update", 0, 9.90, 100)
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    widenings = [e for e in out if isinstance(e, SpreadEvent) and e.direction == "widening"]
    assert len(widenings) == 1
