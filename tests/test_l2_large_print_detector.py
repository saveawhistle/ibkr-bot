"""LargePrintDetector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bot.exit_advisor.core.events import LargePrint
from bot.exit_advisor.detectors.l2.large_print import LargePrintDetector
from bot.exit_advisor.market.book_state import BookStateTracker
from bot.exit_advisor.market.l2_events import L2Print


def _ts(s: float) -> datetime:
    return datetime(2026, 5, 5, 13, 30, 0, tzinfo=UTC) + timedelta(seconds=s)


def _drive(detector: Any, prints: list[Any]) -> list[Any]:
    tracker = BookStateTracker()
    out = []
    for p in prints:
        tracker.consume(p)
        out.extend(detector.consume(p, tracker.get_state()))
    return out


def test_no_emission_during_warmup() -> None:
    """Warmup is 10 prints. Below that, no large-print signal even
    if the size is huge."""
    detector = LargePrintDetector(symbol="X", size_multiplier=5.0, rolling_window_prints=50)
    prints = [L2Print(_ts(i), "X", 10.00, 100, "buy") for i in range(5)]
    prints.append(L2Print(_ts(5), "X", 10.00, 10000, "buy"))
    out = _drive(detector, prints)
    assert not any(isinstance(e, LargePrint) for e in out)


def test_large_print_fires_above_threshold() -> None:
    detector = LargePrintDetector(symbol="X", size_multiplier=5.0, rolling_window_prints=50)
    # 15 prints of size 100 → average 100 once warmup clears.
    prints = [L2Print(_ts(i), "X", 10.00, 100, "buy") for i in range(15)]
    # 16th print at 600 (= 6x average) — fires.
    prints.append(L2Print(_ts(15), "X", 10.00, 600, "buy"))
    out = _drive(detector, prints)
    fires = [e for e in out if isinstance(e, LargePrint)]
    assert len(fires) == 1
    assert fires[0].size == 600
    assert fires[0].ratio == 6.0


def test_below_threshold_does_not_fire() -> None:
    detector = LargePrintDetector(symbol="X", size_multiplier=5.0, rolling_window_prints=50)
    prints = [L2Print(_ts(i), "X", 10.00, 100, "buy") for i in range(15)]
    prints.append(L2Print(_ts(15), "X", 10.00, 400, "buy"))  # 4x average
    out = _drive(detector, prints)
    assert not any(isinstance(e, LargePrint) for e in out)


def test_aggressor_side_propagated() -> None:
    detector = LargePrintDetector(symbol="X", size_multiplier=5.0, rolling_window_prints=50)
    prints = [L2Print(_ts(i), "X", 10.00, 100, "buy") for i in range(15)]
    prints.append(L2Print(_ts(15), "X", 10.00, 1000, "sell"))
    out = _drive(detector, prints)
    fires = [e for e in out if isinstance(e, LargePrint)]
    assert fires[0].aggressor_side == "sell"


def test_rolling_window_truncates() -> None:
    """Old prints age out — the rolling average reflects only the
    most recent ``rolling_window_prints`` entries."""
    detector = LargePrintDetector(symbol="X", size_multiplier=5.0, rolling_window_prints=10)
    # 10 prints of size 100 (fills window) then 5 prints of size 10 (
    # pushes the 100s out).
    prints = [L2Print(_ts(i), "X", 10.00, 100, "buy") for i in range(10)]
    prints.extend(L2Print(_ts(10 + i), "X", 10.00, 10, "buy") for i in range(9))
    # Now average should be ~14 (computed from window). A print at 100
    # should fire (ratio ~7x against the new ~14 average).
    prints.append(L2Print(_ts(20), "X", 10.00, 100, "buy"))
    out = _drive(detector, prints)
    fires = [e for e in out if isinstance(e, LargePrint)]
    assert len(fires) >= 1
    assert fires[-1].size == 100


def test_book_updates_ignored() -> None:
    """Book updates don't drive the print detector."""
    from bot.exit_advisor.market.l2_events import L2BookUpdate

    detector = LargePrintDetector(symbol="X", size_multiplier=5.0)
    tracker = BookStateTracker()
    update = L2BookUpdate(_ts(0), "X", "bid", "insert", 0, 10.00, 100)
    tracker.consume(update)
    out = detector.consume(update, tracker.get_state())
    assert out == []
