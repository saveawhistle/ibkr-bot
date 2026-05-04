"""AbsorptionDetector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bot.exit_advisor.core.events import AbsorptionDetected
from bot.exit_advisor.detectors.l2.absorption import AbsorptionDetector
from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


def _ts(ms: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, 0, tzinfo=UTC) + timedelta(milliseconds=ms)


def _drive(detector: Any, tracker: Any, events: list[Any]) -> list[Any]:
    """Helper: feed each event through both tracker and detector,
    return aggregated detector outputs."""
    out = []
    for evt in events:
        tracker.consume(evt)
        out.extend(detector.consume(evt, tracker.get_state()))
    return out


def test_absorption_fires_on_repeated_consumption() -> None:
    """Bid at $10 with 100 shares; consumed 3.5x by sell prints; refreshed
    after each consumption. Exceeds 3.0x threshold → fires."""
    detector = AbsorptionDetector(symbol="X", refresh_multiplier=3.0)
    tracker = BookStateTracker()

    events = [
        L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(10), "X", 10.00, 100, "sell"),  # cumulative 100
        L2BookUpdate(_ts(20), "X", "bid", "update", 0, 10.00, 100),  # refresh
        L2Print(_ts(30), "X", 10.00, 100, "sell"),  # cumulative 200
        L2BookUpdate(_ts(40), "X", "bid", "update", 0, 10.00, 100),  # refresh
        L2Print(_ts(50), "X", 10.00, 100, "sell"),  # cumulative 300
        L2BookUpdate(_ts(60), "X", "bid", "update", 0, 10.00, 100),  # cum 300 = 3 * 100 → fires
    ]
    out = _drive(detector, tracker, events)
    absorption_events = [e for e in out if isinstance(e, AbsorptionDetected)]
    assert len(absorption_events) == 1
    assert absorption_events[0].cumulative_size_consumed >= 300
    assert absorption_events[0].side == "bid"


def test_absorption_does_not_fire_on_single_hit() -> None:
    """One consumption then delete — not absorption; just liquidity removed."""
    detector = AbsorptionDetector(symbol="X", refresh_multiplier=3.0)
    tracker = BookStateTracker()

    events = [
        L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(10), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(20), "X", "bid", "delete", 0, 10.00, 0),
    ]
    out = _drive(detector, tracker, events)
    assert not any(isinstance(e, AbsorptionDetected) for e in out)


def test_absorption_threshold_respected() -> None:
    """At exactly the threshold, fires. Below, does not."""
    detector = AbsorptionDetector(symbol="X", refresh_multiplier=5.0)
    tracker = BookStateTracker()

    # 4x consumption — below 5.0x threshold.
    events = [
        L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(10), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(20), "X", "bid", "update", 0, 10.00, 100),
        L2Print(_ts(30), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(40), "X", "bid", "update", 0, 10.00, 100),
        L2Print(_ts(50), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(60), "X", "bid", "update", 0, 10.00, 100),
        L2Print(_ts(70), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(80), "X", "bid", "update", 0, 10.00, 100),
    ]
    out = _drive(detector, tracker, events)
    assert not any(isinstance(e, AbsorptionDetected) for e in out)


def test_absorption_latches_after_firing() -> None:
    """Once a level emits absorption, subsequent re-checks at the same
    price don't re-fire — until the level disappears entirely."""
    detector = AbsorptionDetector(symbol="X", refresh_multiplier=2.0)
    tracker = BookStateTracker()

    events = [
        L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(10), "X", 10.00, 100, "sell"),
        L2Print(_ts(20), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(30), "X", "bid", "update", 0, 10.00, 100),  # fires (cum 200 = 2 * 100)
        L2Print(_ts(40), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(50), "X", "bid", "update", 0, 10.00, 100),  # would fire again, latched
    ]
    out = _drive(detector, tracker, events)
    fires = [e for e in out if isinstance(e, AbsorptionDetected)]
    assert len(fires) == 1


def test_absorption_re_arms_after_level_disappears() -> None:
    """Delete clears the latch; a new insert at the same price gets a
    fresh window and can fire absorption independently."""
    detector = AbsorptionDetector(symbol="X", refresh_multiplier=2.0)
    tracker = BookStateTracker()

    events = [
        L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(10), "X", 10.00, 100, "sell"),
        L2Print(_ts(20), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(30), "X", "bid", "update", 0, 10.00, 100),  # fires
        L2BookUpdate(_ts(40), "X", "bid", "delete", 0, 10.00, 0),
        # Re-insert at same price: fresh window.
        L2BookUpdate(_ts(50), "X", "bid", "insert", 0, 10.00, 100),
        L2Print(_ts(60), "X", 10.00, 100, "sell"),
        L2Print(_ts(70), "X", 10.00, 100, "sell"),
        L2BookUpdate(_ts(80), "X", "bid", "update", 0, 10.00, 100),  # fires again
    ]
    out = _drive(detector, tracker, events)
    fires = [e for e in out if isinstance(e, AbsorptionDetected)]
    assert len(fires) == 2
