"""Tests for ``bot.signal_bus.SignalBus`` — FIFO ordering + streaming iteration + co-signal dedup."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from structlog.testing import capture_logs

from bot.signal_bus import SignalBus
from bot.strategies.base import Signal


def _signal(symbol: str, entry: float = 10.0, strategy: str = "test") -> Signal:
    """Build a minimal valid Signal for bus round-trip tests."""
    return Signal(
        symbol=symbol,
        strategy=strategy,
        entry=entry,
        stop=entry - 1.0,
        scale_out_price=entry + 2.0,
        runner_target_price=entry + 2.0,
        timestamp=datetime(2026, 4, 16, 9, 31, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_put_then_stream_yields_in_fifo_order() -> None:
    """Two signals pushed in order come back in the same order."""
    bus = SignalBus()
    await bus.put(_signal("AAA", 10))
    await bus.put(_signal("BBB", 11))

    received: list[str] = []

    async def consume() -> None:
        async for s in bus.stream():
            received.append(s.symbol)
            if len(received) == 2:
                return

    await asyncio.wait_for(consume(), timeout=1.0)
    assert received == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_qsize_reflects_buffered_signals() -> None:
    """``qsize`` is a plain passthrough and must track puts accurately."""
    bus = SignalBus()
    assert bus.qsize() == 0
    await bus.put(_signal("X"))
    assert bus.qsize() == 1


@pytest.mark.asyncio
async def test_put_batch_co_signal_prefers_gap_and_go() -> None:
    """Both strategies firing on the same bar → only Gap-and-Go reaches the queue."""
    bus = SignalBus()
    gap = _signal("AAA", strategy="gap_and_go")
    mom = _signal("AAA", strategy="momentum")

    with capture_logs() as captured:
        await bus.put_batch([mom, gap])

    assert bus.qsize() == 1
    received = await asyncio.wait_for(bus._queue.get(), timeout=1.0)  # noqa: SLF001 - test inspection
    assert received.strategy == "gap_and_go"
    superseded = [e for e in captured if e.get("event") == "signal.superseded_same_bar"]
    assert len(superseded) == 1
    assert superseded[0]["strategy"] == "momentum"
    assert superseded[0]["winning_strategy"] == "gap_and_go"


@pytest.mark.asyncio
async def test_put_batch_momentum_only_passes_through() -> None:
    """Only Momentum fires — it publishes unchanged and emits no supersede event."""
    bus = SignalBus()
    mom = _signal("BBB", strategy="momentum")

    with capture_logs() as captured:
        await bus.put_batch([mom])

    assert bus.qsize() == 1
    assert [e for e in captured if e.get("event") == "signal.superseded_same_bar"] == []


@pytest.mark.asyncio
async def test_put_batch_on_different_bars_both_reach_queue() -> None:
    """Phase 4d: two Gap-and-Go signals on the SAME symbol but DIFFERENT bars both publish.

    Dedup is strictly per-bar (``signal.superseded_same_bar``). Re-entries
    happen on later bars, so they must pass through the bus untouched —
    the re-entry gate lives downstream in RiskEngine, not here.
    """
    bus = SignalBus()
    first = Signal(
        symbol="AAA",
        strategy="gap_and_go",
        entry=10.0,
        stop=9.0,
        scale_out_price=13.0,
        runner_target_price=13.0,
        timestamp=datetime(2026, 4, 16, 9, 32, tzinfo=UTC),
    )
    second = Signal(
        symbol="AAA",
        strategy="gap_and_go",
        entry=10.5,
        stop=9.5,
        scale_out_price=13.5,
        runner_target_price=13.5,
        timestamp=datetime(2026, 4, 16, 9, 35, tzinfo=UTC),
    )

    with capture_logs() as captured:
        await bus.put_batch([first])
        await bus.put_batch([second])

    assert bus.qsize() == 2
    assert [e for e in captured if e.get("event") == "signal.superseded_same_bar"] == []


@pytest.mark.asyncio
async def test_put_batch_duplicate_gap_and_go_keeps_first() -> None:
    """Two Gap-and-Go signals on the same bar (pathological but possible) → keep the first."""
    bus = SignalBus()
    first = _signal("CCC", entry=10.0, strategy="gap_and_go")
    second = _signal("CCC", entry=10.5, strategy="gap_and_go")

    with capture_logs() as captured:
        await bus.put_batch([first, second])

    assert bus.qsize() == 1
    received = await asyncio.wait_for(bus._queue.get(), timeout=1.0)  # noqa: SLF001 - test inspection
    assert received.entry == 10.0
    superseded = [e for e in captured if e.get("event") == "signal.superseded_same_bar"]
    assert len(superseded) == 1
    assert superseded[0]["strategy"] == "gap_and_go"
