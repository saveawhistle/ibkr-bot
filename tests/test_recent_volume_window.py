"""Unit tests for ``bot.strategies.volume.RecentVolumeWindow``."""

from __future__ import annotations

import pytest

from bot.strategies.volume import RecentVolumeWindow


class _FakeBar:
    """Minimal duck-typed bar with ``.volume``."""

    def __init__(self, volume: float) -> None:
        self.volume = volume


def test_empty_window_average_is_none() -> None:
    w = RecentVolumeWindow(window_bars=5)
    assert w.average_volume() is None
    assert w.bars_seen == 0
    assert w.is_populated is False


def test_average_is_none_until_window_full() -> None:
    """Conservative spec: window must be fully populated before average is meaningful."""
    w = RecentVolumeWindow(window_bars=5)
    for v in [100, 200, 300, 400]:  # only 4 of 5
        w.add_volume(v)
    assert w.bars_seen == 4
    assert w.is_populated is False
    assert w.average_volume() is None
    w.add_volume(500)  # now 5 of 5
    assert w.is_populated is True
    assert w.average_volume() == pytest.approx(300.0)


def test_average_is_correct_mean_when_populated() -> None:
    w = RecentVolumeWindow(window_bars=4)
    w.extend_from_volumes([10, 20, 30, 40])
    assert w.average_volume() == pytest.approx(25.0)


def test_window_evicts_oldest_at_capacity() -> None:
    """Beyond ``window_bars`` adds, the oldest volume falls out (FIFO)."""
    w = RecentVolumeWindow(window_bars=3)
    w.extend_from_volumes([1, 2, 3])  # avg 2.0
    assert w.average_volume() == pytest.approx(2.0)
    w.add_volume(7)  # evicts 1; window now [2, 3, 7] → avg 4.0
    assert w.average_volume() == pytest.approx(4.0)
    w.add_volume(10)  # evicts 2; window now [3, 7, 10] → avg ~6.67
    assert w.average_volume() == pytest.approx(6.6667, abs=1e-3)


def test_relative_volume_returns_ratio_when_populated() -> None:
    """relative_volume = candidate / average (NOT mutating the window)."""
    w = RecentVolumeWindow(window_bars=4)
    w.extend_from_volumes([100, 100, 100, 100])  # avg 100
    assert w.relative_volume(250) == pytest.approx(2.5)
    # Window is unchanged after relative_volume call.
    assert w.bars_seen == 4
    assert w.average_volume() == pytest.approx(100.0)


def test_relative_volume_returns_none_when_window_not_populated() -> None:
    w = RecentVolumeWindow(window_bars=5)
    w.extend_from_volumes([100, 100])  # only 2 of 5
    assert w.relative_volume(500) is None


def test_relative_volume_returns_none_on_zero_average() -> None:
    """20 consecutive zero-volume bars → no baseline; returns None not divide-by-zero."""
    w = RecentVolumeWindow(window_bars=3)
    w.extend_from_volumes([0, 0, 0])
    assert w.average_volume() == pytest.approx(0.0)
    assert w.relative_volume(100) is None


def test_add_bar_uses_bar_volume_attribute() -> None:
    w = RecentVolumeWindow(window_bars=3)
    for vol in [50, 100, 150]:
        w.add_bar(_FakeBar(volume=vol))
    assert w.average_volume() == pytest.approx(100.0)
    assert w.relative_volume(300) == pytest.approx(3.0)


def test_negative_volume_clamped_to_zero() -> None:
    """Defensive: a negative bar volume corrupts the average -- treat as zero."""
    w = RecentVolumeWindow(window_bars=3)
    w.extend_from_volumes([100, -50, 200])
    # The -50 was clamped to 0; window is [100, 0, 200] → avg 100.
    assert w.average_volume() == pytest.approx(100.0)


def test_constructor_rejects_zero_or_negative_window() -> None:
    with pytest.raises(ValueError, match="window_bars must be >= 1"):
        RecentVolumeWindow(window_bars=0)
    with pytest.raises(ValueError, match="window_bars must be >= 1"):
        RecentVolumeWindow(window_bars=-5)


def test_window_bars_property_reflects_constructor() -> None:
    assert RecentVolumeWindow(window_bars=20).window_bars == 20


def test_extend_from_volumes_handles_empty_iterable() -> None:
    w = RecentVolumeWindow(window_bars=5)
    w.extend_from_volumes([])
    assert w.bars_seen == 0
    assert w.average_volume() is None


def test_window_size_one_works() -> None:
    """Edge case: a window of 1 means each bar is compared to the prior bar."""
    w = RecentVolumeWindow(window_bars=1)
    w.add_volume(100)
    assert w.is_populated is True
    assert w.average_volume() == pytest.approx(100.0)
    assert w.relative_volume(250) == pytest.approx(2.5)
