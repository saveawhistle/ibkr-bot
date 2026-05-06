"""Tests for ``bot.brokerage.bar_aggregator.RollingMinuteAggregator``.

The aggregator is the load-bearing piece of Phase 10.4: it decides
when a 1-min candle is "final" and what its OHLCV values are. Bugs
here propagate directly into strategy evaluation. Coverage targets:

* Twelfth-bar trigger fires on ``time.second == 55`` and produces an
  OHLCV that matches a hand-computed ground truth (12 known 5-sec bars).
* New-minute (gap) trigger fires when the :55 bar is dropped.
* Idempotency: a minute is finalized at most once even if a stray
  in-minute bar arrives after the trigger fired.
* Out-of-order older-minute bars are dropped + logged.
* In-progress candle snapshot is correct mid-minute (Phase 7.4 needs
  a trailing in-progress row in ``BarStream.bars``).
* VWAP arithmetic matches ``Σ(wap × volume) / Σ(volume)`` and
  degrades gracefully when total volume is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from structlog.testing import capture_logs

from bot.brokerage.bar_aggregator import MinuteCandle, RollingMinuteAggregator

# ---------------------------------------------------------------------------
# Test fixtures — minimal RealTimeBar stand-in
# ---------------------------------------------------------------------------


@dataclass
class _FakeRealTimeBar:
    """Duck-typed stand-in for ``ib_async.RealTimeBar``."""

    time: datetime
    open_: float
    high: float
    low: float
    close: float
    volume: float
    wap: float


def _bar(
    minute: int,
    second: int,
    *,
    open_: float = 100.0,
    high: float = 100.5,
    low: float = 99.5,
    close: float = 100.0,
    volume: float = 1000.0,
    wap: float | None = None,
) -> _FakeRealTimeBar:
    """Build a test bar at 09:{minute}:{second} UTC.

    ``wap`` defaults to ``close`` when not specified; tests that exercise
    the volume-weighted-average can pass a distinct value.
    """
    return _FakeRealTimeBar(
        time=datetime(2026, 4, 30, 9, minute, second, tzinfo=UTC),
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        wap=wap if wap is not None else close,
    )


def _make_aggregator() -> tuple[RollingMinuteAggregator, list[tuple[MinuteCandle, str]]]:
    """Build an aggregator that records every ``on_minute_final`` payload."""
    received: list[tuple[MinuteCandle, str]] = []

    def _cb(candle: MinuteCandle, finalized_at: datetime, trigger: str) -> None:
        received.append((candle, trigger))

    agg = RollingMinuteAggregator(symbol="TEST", on_minute_final=_cb)
    return agg, received


# ---------------------------------------------------------------------------
# Twelfth-bar trigger — the typical happy path
# ---------------------------------------------------------------------------


def test_twelfth_bar_finalizes_minute_with_correct_ohlcv() -> None:
    """12 5-sec bars → one finalized 1-min candle with hand-computed OHLCV.

    Constructs 12 bars with monotonically rising prices so high/low are
    obvious from the input. Volume sums to a known total. VWAP equals
    the volume-weighted average of per-bar ``wap``.
    """
    agg, received = _make_aggregator()

    # 12 bars covering 09:31:00, :05, :10, ..., :55
    seconds = list(range(0, 60, 5))
    assert seconds[-1] == 55 and len(seconds) == 12  # sanity
    for i, sec in enumerate(seconds):
        agg.on_5sec_bar(
            _bar(
                minute=31,
                second=sec,
                open_=100.0 + i,
                high=100.5 + i,
                low=99.5 + i,
                close=100.0 + i,
                volume=100.0 * (i + 1),  # rising volume so VWAP is non-trivial
                wap=100.0 + i,  # wap equals close per bar
            )
        )

    assert len(received) == 1, "exactly one finalization per minute"
    candle, trigger = received[0]
    assert trigger == "twelfth_bar"
    assert candle.minute_start == datetime(2026, 4, 30, 9, 31, 0, tzinfo=UTC)
    # Open from first bar (i=0): 100.0
    assert candle.open == pytest.approx(100.0)
    # Close from last bar (i=11): 100.0 + 11 = 111.0
    assert candle.close == pytest.approx(111.0)
    # High = max(100.5..111.5) = 111.5
    assert candle.high == pytest.approx(111.5)
    # Low = min(99.5..110.5) = 99.5
    assert candle.low == pytest.approx(99.5)
    # Volume = 100 + 200 + ... + 1200 = 100 × (12 × 13 / 2) = 7800
    assert candle.volume == pytest.approx(7800.0)
    # VWAP = Σ(wap × vol) / Σ(vol) = Σ((100+i) × 100×(i+1)) / 7800
    expected_vwap = sum((100.0 + i) * 100.0 * (i + 1) for i in range(12)) / 7800.0
    assert candle.vwap == pytest.approx(expected_vwap)
    assert candle.bars_contributing == 12


def test_partial_minute_finalizes_when_twelfth_bar_arrives() -> None:
    """Subscribe mid-minute (only 5 of 12 bars seen) — :55 still triggers finalize.

    Models the bootstrap case where MarketData subscribes part-way
    through a minute. The aggregator should still finalize on the :55
    bar, with the candle reflecting only the bars it observed.
    """
    agg, received = _make_aggregator()

    for sec in (35, 40, 45, 50, 55):
        agg.on_5sec_bar(_bar(minute=31, second=sec, close=100.0 + sec / 5.0, volume=100.0))

    assert len(received) == 1
    candle, trigger = received[0]
    assert trigger == "twelfth_bar"
    assert candle.bars_contributing == 5
    assert candle.volume == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# New-minute trigger — gap fallback
# ---------------------------------------------------------------------------


def test_new_minute_trigger_fires_when_55_bar_dropped() -> None:
    """:55 bar missing → next-minute :00 bar finalizes the prior minute via gap path."""
    agg, received = _make_aggregator()

    # 11 bars of minute 31 (everything except :55).
    for sec in range(0, 55, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec, close=100.0, volume=100.0))
    assert received == [], "no finalization until next-minute bar arrives"

    # First bar of minute 32 fires the gap path.
    agg.on_5sec_bar(_bar(minute=32, second=0, close=200.0, volume=100.0))
    assert len(received) == 1
    candle, trigger = received[0]
    assert trigger == "new_minute_bar"
    assert candle.minute_start == datetime(2026, 4, 30, 9, 31, 0, tzinfo=UTC)
    assert candle.bars_contributing == 11  # the 11 bars we did receive


def test_gap_finalization_emits_gap_detected_event() -> None:
    """The gap path logs ``bar_aggregator.gap_detected`` for forensic review."""
    agg, _ = _make_aggregator()

    for sec in range(0, 55, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec))

    with capture_logs() as captured:
        agg.on_5sec_bar(_bar(minute=32, second=0))

    events = [e["event"] for e in captured]
    assert "bar_aggregator.gap_detected" in events


def test_twelfth_bar_emits_no_gap_event_on_clean_close() -> None:
    """The clean :55 close path must NOT log ``bar_aggregator.gap_detected``."""
    agg, _ = _make_aggregator()

    with capture_logs() as captured:
        for sec in range(0, 60, 5):
            agg.on_5sec_bar(_bar(minute=31, second=sec))

    events = [e["event"] for e in captured]
    assert "bar_aggregator.minute_finalized" in events
    assert "bar_aggregator.gap_detected" not in events


# ---------------------------------------------------------------------------
# Idempotency + out-of-order
# ---------------------------------------------------------------------------


def test_minute_finalized_at_most_once() -> None:
    """A late same-minute bar arriving after the :55 trigger is dropped."""
    agg, received = _make_aggregator()

    for sec in range(0, 60, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec))
    assert len(received) == 1

    # Stray bar from minute 31 arrives after we already finalized.
    agg.on_5sec_bar(_bar(minute=31, second=58, close=999.0, volume=999.0))
    assert len(received) == 1, "no second finalization for minute 31"


def test_out_of_order_older_minute_bar_dropped_with_event() -> None:
    """Receiving a bar from an earlier minute after we've moved on logs discontinuity."""
    agg, received = _make_aggregator()

    # Settle into minute 32 by finalizing 31 cleanly + starting 32.
    for sec in range(0, 60, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec))
    agg.on_5sec_bar(_bar(minute=32, second=0))
    assert len(received) == 1  # 31 finalized

    # Now an out-of-order bar from minute 30 arrives.
    with capture_logs() as captured:
        agg.on_5sec_bar(_bar(minute=30, second=10, close=50.0))
    assert len(received) == 1, "out-of-order bar must not finalize"
    events = [e["event"] for e in captured]
    assert "bar_aggregator.discontinuity" in events


def test_double_55_bar_idempotent() -> None:
    """A duplicate :55 bar (e.g., ib_async re-emission) does not double-fire."""
    agg, received = _make_aggregator()

    for sec in range(0, 60, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec))
    assert len(received) == 1

    # Stray duplicate :55 bar.
    agg.on_5sec_bar(_bar(minute=31, second=55, close=200.0))
    assert len(received) == 1


# ---------------------------------------------------------------------------
# In-progress candle snapshot
# ---------------------------------------------------------------------------


def test_in_progress_candle_returns_none_before_first_bar() -> None:
    """No bars in → no in-progress candle."""
    agg, _ = _make_aggregator()
    assert agg.in_progress_candle is None


def test_in_progress_candle_reflects_partial_minute() -> None:
    """Mid-minute snapshot has accumulated OHLCV through the latest bar.

    Open is locked to the first bar of the minute; high/low/close/volume
    update as bars arrive. ``bars_contributing`` matches the number of
    5-sec bars seen so far.
    """
    agg, _ = _make_aggregator()

    agg.on_5sec_bar(
        _bar(minute=31, second=0, open_=10.0, high=10.5, low=9.5, close=10.0, volume=100.0)
    )
    agg.on_5sec_bar(
        _bar(minute=31, second=5, open_=10.0, high=11.0, low=10.0, close=10.8, volume=200.0)
    )
    agg.on_5sec_bar(
        _bar(minute=31, second=10, open_=10.8, high=10.9, low=10.5, close=10.7, volume=150.0)
    )

    snap = agg.in_progress_candle
    assert snap is not None
    assert snap.minute_start == datetime(2026, 4, 30, 9, 31, 0, tzinfo=UTC)
    assert snap.open == pytest.approx(10.0)  # first bar's open
    assert snap.high == pytest.approx(11.0)
    assert snap.low == pytest.approx(9.5)
    assert snap.close == pytest.approx(10.7)  # latest bar's close
    assert snap.volume == pytest.approx(450.0)
    assert snap.bars_contributing == 3


def test_in_progress_candle_clears_after_finalization() -> None:
    """After :55 fires, the slot is cleared until the next minute's first bar."""
    agg, _ = _make_aggregator()

    for sec in range(0, 60, 5):
        agg.on_5sec_bar(_bar(minute=31, second=sec))

    # No new bar yet for minute 32 → in-progress is None.
    assert agg.in_progress_candle is None

    agg.on_5sec_bar(_bar(minute=32, second=0))
    snap = agg.in_progress_candle
    assert snap is not None
    assert snap.minute_start == datetime(2026, 4, 30, 9, 32, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# VWAP arithmetic
# ---------------------------------------------------------------------------


def test_vwap_matches_volume_weighted_formula() -> None:
    """VWAP = Σ(wap × volume) / Σ(volume) with handcomputed inputs."""
    agg, received = _make_aggregator()

    # Two bars of minute 31 with explicit waps.
    agg.on_5sec_bar(_bar(minute=31, second=0, close=10.0, volume=100.0, wap=10.10))
    agg.on_5sec_bar(_bar(minute=31, second=5, close=11.0, volume=300.0, wap=10.50))
    # Skip to :55 to force finalization.
    agg.on_5sec_bar(_bar(minute=31, second=55, close=12.0, volume=200.0, wap=11.75))

    candle, _ = received[0]
    expected = (10.10 * 100.0 + 10.50 * 300.0 + 11.75 * 200.0) / 600.0
    assert candle.vwap == pytest.approx(expected)


def test_vwap_falls_back_to_close_when_zero_volume() -> None:
    """A zero-volume minute returns ``close`` as VWAP rather than dividing by zero."""
    agg, received = _make_aggregator()

    for sec in range(0, 60, 5):
        agg.on_5sec_bar(
            _bar(
                minute=31,
                second=sec,
                close=10.0,
                volume=0.0,  # nobody traded this minute
                wap=10.0,
            )
        )
    candle, _ = received[0]
    assert candle.vwap == pytest.approx(10.0)
    assert candle.volume == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multi-minute integration — strict alternation
# ---------------------------------------------------------------------------


def test_multi_minute_each_emits_once() -> None:
    """Three minutes of clean bars produce three finalizations in order."""
    agg, received = _make_aggregator()

    for minute in (31, 32, 33):
        for sec in range(0, 60, 5):
            agg.on_5sec_bar(_bar(minute=minute, second=sec, close=100.0 + minute))

    assert len(received) == 3
    assert [c.minute_start.minute for c, _ in received] == [31, 32, 33]
    assert all(trigger == "twelfth_bar" for _, trigger in received)


def test_logs_latency_ms_in_finalized_event() -> None:
    """``bar_aggregator.minute_finalized`` carries a ``latency_ms`` field for forensics."""
    agg, _ = _make_aggregator()

    with capture_logs() as captured:
        for sec in range(0, 60, 5):
            agg.on_5sec_bar(_bar(minute=31, second=sec))

    finalized = next(e for e in captured if e["event"] == "bar_aggregator.minute_finalized")
    assert "latency_ms" in finalized
    # Synthetic fake-bar minute is in the past relative to wall-clock,
    # so latency_ms is always positive (often very large in tests).
    assert isinstance(finalized["latency_ms"], (int, float))


# ---------------------------------------------------------------------------
# Naive-tz tolerance
# ---------------------------------------------------------------------------


def test_accepts_tz_naive_bar_time_and_normalises_to_utc() -> None:
    """ib_async sometimes hands tz-naive datetimes; aggregator coerces to UTC."""
    agg, received = _make_aggregator()

    naive = _FakeRealTimeBar(
        time=datetime(2026, 4, 30, 9, 31, 55),  # no tzinfo
        open_=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        volume=100.0,
        wap=10.0,
    )
    agg.on_5sec_bar(naive)
    assert len(received) == 1
    candle, _ = received[0]
    assert candle.minute_start.tzinfo is not None
    assert candle.minute_start == datetime(2026, 4, 30, 9, 31, 0, tzinfo=UTC)
