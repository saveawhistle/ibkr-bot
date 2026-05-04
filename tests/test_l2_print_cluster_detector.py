"""PrintClusterDetector tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from bot.exit_advisor.core.events import PrintCluster
from bot.exit_advisor.detectors.l2.print_cluster import PrintClusterDetector
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


def test_cluster_fires_on_n_same_side_prints() -> None:
    detector = PrintClusterDetector(symbol="X", window_seconds=10.0, min_prints=5)
    prints = [L2Print(_ts(i), "X", 10.05, 100, "buy") for i in range(5)]
    out = _drive(detector, prints)
    fires = [e for e in out if isinstance(e, PrintCluster)]
    assert len(fires) == 1
    assert fires[0].side == "buy"
    assert fires[0].print_count == 5
    assert fires[0].total_volume == 500


def test_cluster_does_not_fire_on_mixed_sides() -> None:
    detector = PrintClusterDetector(symbol="X", window_seconds=10.0, min_prints=5)
    prints = [
        L2Print(_ts(0), "X", 10.05, 100, "buy"),
        L2Print(_ts(1), "X", 10.00, 100, "sell"),
        L2Print(_ts(2), "X", 10.05, 100, "buy"),
        L2Print(_ts(3), "X", 10.00, 100, "sell"),
        L2Print(_ts(4), "X", 10.05, 100, "buy"),
    ]
    out = _drive(detector, prints)
    assert not any(isinstance(e, PrintCluster) for e in out)


def test_unknown_aggressor_excluded() -> None:
    """Mid-spread prints (aggressor=unknown) don't count toward
    cluster classification."""
    detector = PrintClusterDetector(symbol="X", window_seconds=10.0, min_prints=5)
    prints = [
        L2Print(_ts(0), "X", 10.05, 100, "buy"),
        L2Print(_ts(1), "X", 10.02, 100, "unknown"),
        L2Print(_ts(2), "X", 10.05, 100, "buy"),
        L2Print(_ts(3), "X", 10.02, 100, "unknown"),
        L2Print(_ts(4), "X", 10.05, 100, "buy"),
    ]
    out = _drive(detector, prints)
    # Only 3 buy prints — below min_prints=5.
    assert not any(isinstance(e, PrintCluster) for e in out)


def test_window_slides() -> None:
    """Prints aging out of the window stop counting."""
    detector = PrintClusterDetector(symbol="X", window_seconds=2.0, min_prints=5)
    # First 4 prints at t=0,1,2,3 — within 2s window pairwise but not cumulatively.
    prints = [
        L2Print(_ts(0.0), "X", 10.05, 100, "buy"),
        L2Print(_ts(0.5), "X", 10.05, 100, "buy"),
        L2Print(_ts(1.0), "X", 10.05, 100, "buy"),
        L2Print(_ts(1.5), "X", 10.05, 100, "buy"),
        # 5th print arrives at t=10 — first 4 have aged out.
        L2Print(_ts(10.0), "X", 10.05, 100, "buy"),
    ]
    out = _drive(detector, prints)
    assert not any(isinstance(e, PrintCluster) for e in out)


def test_buy_and_sell_clusters_independent() -> None:
    """A buy-side cluster and a sell-side cluster on the same window
    can both fire."""
    detector = PrintClusterDetector(symbol="X", window_seconds=10.0, min_prints=3)
    prints = [
        L2Print(_ts(0), "X", 10.05, 100, "buy"),
        L2Print(_ts(1), "X", 10.05, 100, "buy"),
        L2Print(_ts(2), "X", 10.05, 100, "buy"),  # buy cluster fires
        L2Print(_ts(3), "X", 10.00, 100, "sell"),
        L2Print(_ts(4), "X", 10.00, 100, "sell"),
        L2Print(_ts(5), "X", 10.00, 100, "sell"),  # sell cluster fires
    ]
    out = _drive(detector, prints)
    fires = [e for e in out if isinstance(e, PrintCluster)]
    sides = {f.side for f in fires}
    assert sides == {"buy", "sell"}
    assert len(fires) == 2
