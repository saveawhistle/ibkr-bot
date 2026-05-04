"""Tests for ``bot.brokerage.market_data.MarketData`` — subscription lifecycle (Phase 5.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from ib_async.objects import TickAttribLast, TickByTickAllLast

from bot.brokerage.ibkr_client import SubscriptionRegistry
from bot.brokerage.market_data import MarketData, _missing_bars_between


class _FakeEvent:
    """Duck-typed ib_async event supporting ``+= handler`` and manual emit."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeEvent:
        self.handlers.append(handler)
        return self

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for handler in self.handlers:
            handler(*args, **kwargs)


class _FakeBarList(list[Any]):
    """Empty list-shaped BarDataList with ``reqId`` + ``updateEvent``.

    Must be a real list so pandas / ``_bars_to_frame`` hits the ``not bars``
    empty-frame branch instead of iterating a MagicMock.
    """

    def __init__(self, req_id: int = 0) -> None:
        super().__init__()
        self.reqId = req_id  # noqa: N815 - mirror ib_async
        self.updateEvent = _FakeEvent()


def _fake_bar_list(req_id: int = 0) -> _FakeBarList:
    """Build a duck-typed BarDataList that supports ``+=`` on its update event."""
    return _FakeBarList(req_id=req_id)


def _mock_ibkr(trading_class: str = "NMS", primary_exchange: str = "NASDAQ") -> MagicMock:
    """IBKRClient mock with a real SubscriptionRegistry and stubbed IB surface.

    ``trading_class`` and ``primary_exchange`` are populated on the qualified
    contract so Phase 9.4 gap-detection events can pivot on them.
    """
    ibkr = MagicMock(name="IBKRClient")
    ibkr.ib = MagicMock(name="IB")
    ibkr.ib.cancelHistoricalData = MagicMock()
    ibkr.qualify_stock = AsyncMock(
        side_effect=lambda symbol: SimpleNamespace(
            symbol=symbol,
            conId=123,
            tradingClass=trading_class,
            primaryExchange=primary_exchange,
        )
    )
    ibkr.subscriptions = SubscriptionRegistry()
    return ibkr


def _ibkr_bar(
    bar_time_utc: datetime,
    *,
    open_: float = 10.0,
    high: float = 10.1,
    low: float = 9.9,
    close: float = 10.05,
    volume: float = 100.0,
    average: float = 10.0,
) -> SimpleNamespace:
    """Phase 9.4 — IBKR BarData-shaped object for synthetic stream tests.

    Mirrors the field set ``_bars_to_frame`` reads. ``date`` matches
    ``formatDate=2`` (tz-naive UTC) so the conversion path runs identically
    to live data.
    """
    return SimpleNamespace(
        date=bar_time_utc.replace(tzinfo=None),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        average=average,
    )


def _emit_new_bar(bar_list: _FakeBarList, just_closed: SimpleNamespace) -> None:
    """Append a finalized bar (+ in-progress placeholder) and fire ``has_new_bar=True``.

    Mirrors IBKR semantics: when a minute rolls, the freshly-finalized bar
    sits at ``bars[-2]`` and a new in-progress bar is at ``bars[-1]``.
    The in-progress placeholder is popped after the emit so the next call
    can append its own pair without leaving stale rows behind.
    """
    in_progress = SimpleNamespace(
        date=just_closed.date + pd.Timedelta(minutes=1),
        open=just_closed.close,
        high=just_closed.close,
        low=just_closed.close,
        close=just_closed.close,
        volume=0.0,
        average=just_closed.close,
    )
    bar_list.append(just_closed)
    bar_list.append(in_progress)
    bar_list.updateEvent.emit(bar_list, True)
    del bar_list[-1]


def _emit_new_bar_silent(bar_list: _FakeBarList, just_closed: SimpleNamespace) -> None:
    """SBLX-scenario helper: append a finalized bar but emit ``has_new_bar=False``.

    Phase 9.4: replicates the Day 7 (2026-04-28) failure mode where a bar
    landed in ``bar_list`` but ib_async dispatched the update without the
    ``has_new_bar=True`` flag set. The diff-driven detector must still
    catch it.
    """
    in_progress = SimpleNamespace(
        date=just_closed.date + pd.Timedelta(minutes=1),
        open=just_closed.close,
        high=just_closed.close,
        low=just_closed.close,
        close=just_closed.close,
        volume=0.0,
        average=just_closed.close,
    )
    bar_list.append(just_closed)
    bar_list.append(in_progress)
    bar_list.updateEvent.emit(bar_list, False)
    del bar_list[-1]


@pytest.mark.asyncio
async def test_subscribe_bars_registers_subscription() -> None:
    """After ``subscribe_bars`` the registry contains one ``historical`` entry for the symbol."""
    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=101)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("AAPL")

    active = await ibkr.subscriptions.list_active()
    assert len(active) == 1
    sub = active[0]
    assert sub.kind == "historical"
    assert sub.symbol == "AAPL"
    assert sub.req_id == 101
    assert sub.ref is bar_list


@pytest.mark.asyncio
async def test_unsubscribe_unregisters_and_cancels() -> None:
    """``unsubscribe`` must call cancelHistoricalData on TWS and pop the registry entry."""
    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=202)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("MSFT")
    await md.unsubscribe("MSFT")

    ibkr.ib.cancelHistoricalData.assert_called_once_with(bar_list)
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_subscribe_bars_fires_on_new_bar_callback() -> None:
    """Phase 7.3 + 9.4: callback fires when a new finalized bar appears.

    Phase 9.4 reworked detection to diff ``bar_list`` against a per-symbol
    cursor (instead of trusting ``has_new_bar``). The callback is now
    triggered by the diff result. This test verifies:

    * Updates with no new finalized bars do NOT fire the callback.
    * An update that adds a new finalized bar DOES fire the callback.
    """
    import asyncio

    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=303)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    invocations: list[int] = []

    async def _cb() -> None:
        invocations.append(1)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("AAPL", on_new_bar=_cb)

    # The subscribe call wires a single handler onto updateEvent.
    assert len(bar_list.updateEvent.handlers) == 1

    # Empty-bar-list update (no new finalized bars) must NOT fire the callback.
    bar_list.updateEvent.emit(bar_list, True)
    await asyncio.sleep(0)
    assert invocations == []

    # Add a finalized bar to the list and emit — callback must fire.
    _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
    await asyncio.sleep(0)
    assert invocations == [1]


@pytest.mark.asyncio
async def test_subscribe_bars_without_callback_does_not_crash_on_new_bar() -> None:
    """Phase 7.3: ``on_new_bar=None`` — must be a no-op even when a new bar arrives, not a crash."""
    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=404)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("TSLA")  # no on_new_bar

    # Must not raise — even when a finalized bar appears.
    _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))


class _FakeTicker:
    """Minimal Ticker shape for tick-by-tick subscription tests.

    Has ``reqId`` (ref_req_id reads this) and a ``_FakeEvent``-compatible
    ``updateEvent`` the test can manually emit. ``tickByTicks`` is the
    batched tick list ib_async populates before firing updateEvent.
    """

    def __init__(self, req_id: int = 0) -> None:
        self.reqId = req_id  # noqa: N815 - mirror ib_async
        self.updateEvent = _FakeEvent()
        self.tickByTicks: list[Any] = []
        self.contract: Any = None


def _trade_tick(price: float, size: float = 100.0, conditions: str = "") -> TickByTickAllLast:
    """Build a TickByTickAllLast populated enough for handler code paths."""
    return TickByTickAllLast(
        tickType=1,
        time=datetime.now(UTC),
        price=price,
        size=size,
        tickAttribLast=TickAttribLast(pastLimit=False, unreported=False),
        exchange="ARCA",
        specialConditions=conditions,
    )


@pytest.mark.asyncio
async def test_subscribe_ticks_fires_callback_on_regular_prints() -> None:
    """Phase 7.5: non-Form-T ticks are delivered to the async callback."""
    ibkr = _mock_ibkr()
    ticker = _FakeTicker(req_id=501)
    ibkr.ib.reqTickByTickData = MagicMock(return_value=ticker)

    invocations: list[float] = []

    async def _on_tick(tick: TickByTickAllLast) -> None:
        invocations.append(tick.price)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_ticks("AAPL", on_tick=_on_tick)

    # Simulate a batch of two trade ticks arriving.
    ticker.tickByTicks = [_trade_tick(272.30), _trade_tick(272.35)]
    ticker.updateEvent.emit(ticker)

    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)  # drain both scheduled tasks

    assert invocations == [pytest.approx(272.30), pytest.approx(272.35)]


@pytest.mark.asyncio
async def test_subscribe_ticks_skips_form_t_extended_hours_prints() -> None:
    """Phase 7.5: ``specialConditions`` containing 'FT' (extended hours) is filtered out."""
    ibkr = _mock_ibkr()
    ticker = _FakeTicker(req_id=502)
    ibkr.ib.reqTickByTickData = MagicMock(return_value=ticker)

    invocations: list[float] = []

    async def _on_tick(tick: TickByTickAllLast) -> None:
        invocations.append(tick.price)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_ticks("TSLA", on_tick=_on_tick)

    ticker.tickByTicks = [
        _trade_tick(100.0, conditions="FT"),  # after-hours print — skip
        _trade_tick(101.0, conditions=""),  # regular print — deliver
    ]
    ticker.updateEvent.emit(ticker)

    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert invocations == [pytest.approx(101.0)]


@pytest.mark.asyncio
async def test_unsubscribe_ticks_cancels_and_unregisters() -> None:
    """Phase 7.5: ``unsubscribe_ticks`` calls cancelTickByTickData + drops registry entry."""
    ibkr = _mock_ibkr()
    ticker = _FakeTicker(req_id=503)
    ibkr.ib.reqTickByTickData = MagicMock(return_value=ticker)
    ibkr.ib.cancelTickByTickData = MagicMock()

    md = MarketData(ibkr=ibkr)

    async def _noop(_tick: Any) -> None:
        return

    await md.subscribe_ticks("MSFT", on_tick=_noop)
    assert len(ibkr.subscriptions) == 1

    await md.unsubscribe_ticks("MSFT")
    ibkr.ib.cancelTickByTickData.assert_called_once()
    args, _ = ibkr.ib.cancelTickByTickData.call_args
    assert args[1] == "Last"  # default tick type
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_subscribe_ticks_idempotent() -> None:
    """Phase 7.5: re-subscribing the same symbol returns the existing TickStream."""
    ibkr = _mock_ibkr()
    ticker = _FakeTicker(req_id=504)
    ibkr.ib.reqTickByTickData = MagicMock(return_value=ticker)

    md = MarketData(ibkr=ibkr)

    async def _noop(_tick: Any) -> None:
        return

    stream_a = await md.subscribe_ticks("AAPL", on_tick=_noop)
    stream_b = await md.subscribe_ticks("AAPL", on_tick=_noop)
    assert stream_a is stream_b
    # Only one IBKR call.
    assert ibkr.ib.reqTickByTickData.call_count == 1


@pytest.mark.asyncio
async def test_close_unsubscribes_all() -> None:
    """``close()`` sweeps every active subscription, leaving the registry empty."""
    ibkr = _mock_ibkr()
    counter = {"n": 0}

    async def fake_req(*_args: object, **_kwargs: object) -> MagicMock:
        counter["n"] += 1
        return _fake_bar_list(req_id=counter["n"])

    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=fake_req)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("AAA")
    await md.subscribe_bars("BBB")
    assert len(ibkr.subscriptions) == 2

    await md.close()
    assert ibkr.ib.cancelHistoricalData.call_count == 2
    assert len(ibkr.subscriptions) == 0


# ---------- Phase 9.4: bar-receipt logging + gap detection ---------- #


def test_missing_bars_between_returns_zero_for_first_bar() -> None:
    """First bar of the session has no prior to compare → 0 missing."""
    new_ts = pd.Timestamp("2026-04-28 09:30", tz="America/New_York")
    assert _missing_bars_between(None, new_ts) == 0


def test_missing_bars_between_returns_zero_for_consecutive() -> None:
    """09:30 → 09:31 is the expected one-minute step → 0 missing."""
    prev = pd.Timestamp("2026-04-28 09:30", tz="America/New_York")
    new = pd.Timestamp("2026-04-28 09:31", tz="America/New_York")
    assert _missing_bars_between(prev, new) == 0


def test_missing_bars_between_counts_single_skip() -> None:
    """09:31 → 09:33 skips 09:32 → 1 missing bar (the SBLX 9:32 case)."""
    prev = pd.Timestamp("2026-04-28 09:31", tz="America/New_York")
    new = pd.Timestamp("2026-04-28 09:33", tz="America/New_York")
    assert _missing_bars_between(prev, new) == 1


def test_missing_bars_between_counts_multiple_skips() -> None:
    """09:30 → 09:35 skips 09:31, 09:32, 09:33, 09:34 → 4 missing bars."""
    prev = pd.Timestamp("2026-04-28 09:30", tz="America/New_York")
    new = pd.Timestamp("2026-04-28 09:35", tz="America/New_York")
    assert _missing_bars_between(prev, new) == 4


@pytest.mark.asyncio
async def test_bar_received_emits_log_with_ohlc_and_volume() -> None:
    """Phase 9.4: a new finalized bar emits ``market_data.bar_received`` once."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="NMS")
    bar_list = _fake_bar_list(req_id=901)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("RMAX")

    bar_utc = datetime(2026, 4, 28, 13, 30, tzinfo=UTC)  # 09:30 ET
    with capture_logs() as captured:
        _emit_new_bar(bar_list, _ibkr_bar(bar_utc, open_=10.0, close=10.05, volume=12_000.0))

    received = [e for e in captured if e.get("event") == "market_data.bar_received"]
    assert len(received) == 1
    evt = received[0]
    assert evt["symbol"] == "RMAX"
    assert evt["source"] == "stream"
    assert evt["open"] == pytest.approx(10.0)
    assert evt["close"] == pytest.approx(10.05)
    assert evt["volume"] == pytest.approx(12_000.0)
    # Bar time is the just-closed bar in NY-local form.
    assert evt["bar_time"].endswith("-04:00")  # April → EDT
    assert "09:30" in evt["bar_time"]


@pytest.mark.asyncio
async def test_consecutive_bars_emit_no_gap_warning() -> None:
    """Three consecutive minutes (09:30, 09:31, 09:32) must not flag a gap."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=902)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("RMAX")

    with capture_logs() as captured:
        for offset_min in (0, 1, 2):
            ts = datetime(2026, 4, 28, 13, 30 + offset_min, tzinfo=UTC)
            _emit_new_bar(bar_list, _ibkr_bar(ts))

    gaps = [e for e in captured if e.get("event") == "market_data.bar_gap_detected"]
    assert gaps == []


@pytest.mark.asyncio
async def test_single_missing_bar_emits_gap_warning_with_count_one() -> None:
    """Bars 09:30, 09:31, 09:33 → gap warning fires with missing_bars=1.

    Replays the SBLX 2026-04-28 09:32 / 09:35 pattern. Trading-class +
    primary-exchange make it possible to pivot the warning rate by
    ticker classification post-session.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="SCM", primary_exchange="NASDAQ")
    bar_list = _fake_bar_list(req_id=903)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("SBLX")

    with capture_logs() as captured:
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 31, tzinfo=UTC)))
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 33, tzinfo=UTC)))

    gaps = [e for e in captured if e.get("event") == "market_data.bar_gap_detected"]
    assert len(gaps) == 1
    evt = gaps[0]
    assert evt["symbol"] == "SBLX"
    assert evt["missing_bars"] == 1
    assert evt["trading_class"] == "SCM"
    assert evt["primary_exchange"] == "NASDAQ"
    assert "09:31" in evt["previous_bar_time"]
    assert "09:33" in evt["new_bar_time"]


@pytest.mark.asyncio
async def test_multiple_missing_bars_emits_correct_count() -> None:
    """Bars 09:30 then 09:35 → gap warning with missing_bars=4."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="SCM")
    bar_list = _fake_bar_list(req_id=904)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("SBLX")

    with capture_logs() as captured:
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 35, tzinfo=UTC)))

    gaps = [e for e in captured if e.get("event") == "market_data.bar_gap_detected"]
    assert len(gaps) == 1
    assert gaps[0]["missing_bars"] == 4


@pytest.mark.asyncio
async def test_first_bar_emits_no_gap_warning() -> None:
    """The first ``has_new_bar`` after subscription has no prior to compare."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=905)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("RMAX")

    with capture_logs() as captured:
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))

    gaps = [e for e in captured if e.get("event") == "market_data.bar_gap_detected"]
    assert gaps == []
    received = [e for e in captured if e.get("event") == "market_data.bar_received"]
    assert len(received) == 1


@pytest.mark.asyncio
async def test_session_gap_summary_emitted_on_close() -> None:
    """``close()`` emits ``market_data.session_gap_summary`` with aggregate counters.

    Drives one symbol with a single 1-bar gap and one symbol with a clean
    sequence so the per-symbol and per-trading-class breakdowns are
    distinguishable from totals.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="SCM")
    counter = {"n": 0}

    def _next_bar_list() -> _FakeBarList:
        counter["n"] += 1
        return _fake_bar_list(req_id=counter["n"])

    bar_list_a = _next_bar_list()
    bar_list_b = _next_bar_list()
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=[bar_list_a, bar_list_b])

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("SBLX")
    await md.subscribe_bars("WLDS")

    # SBLX: 09:30, 09:32 → 1 missing bar.
    _emit_new_bar(bar_list_a, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
    _emit_new_bar(bar_list_a, _ibkr_bar(datetime(2026, 4, 28, 13, 32, tzinfo=UTC)))
    # WLDS: 09:30, 09:31, 09:32 → no gap.
    for offset_min in (0, 1, 2):
        ts = datetime(2026, 4, 28, 13, 30 + offset_min, tzinfo=UTC)
        _emit_new_bar(bar_list_b, _ibkr_bar(ts))

    with capture_logs() as captured:
        await md.close()

    summaries = [e for e in captured if e.get("event") == "market_data.session_gap_summary"]
    assert len(summaries) == 1
    s = summaries[0]
    assert s["total_bars_received"] == 5  # 2 from SBLX + 3 from WLDS
    assert s["total_gaps_detected"] == 1
    assert s["gaps_by_symbol"] == {"SBLX": 1}
    assert s["gaps_by_trading_class"] == {"SCM": 1}
    assert s["longest_gap_minutes"] == 1


@pytest.mark.asyncio
async def test_bar_received_fires_when_has_new_bar_is_false() -> None:
    """SBLX 2026-04-28 scenario: a finalized bar lands in ``bar_list`` but
    ``has_new_bar=False``. The diff-driven detector must still log it.

    Day 7 evidence: a one-shot ``reqHistoricalDataAsync`` re-fetch of SBLX
    returned the 09:32 / 09:35 bars that the bot's live stream skipped. So
    the bars were in IBKR's data — the bug is in trusting the
    ``has_new_bar`` flag. This test pins the new diff-based behaviour.
    """
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="SCM")
    bar_list = _fake_bar_list(req_id=910)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    invocations: list[int] = []

    async def _cb() -> None:
        invocations.append(1)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("SBLX", on_new_bar=_cb)

    with capture_logs() as captured:
        # Establish baseline cursor at 09:30 with a normal has_new_bar=True emit.
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
        # Now the SBLX-class bug: 09:31 finalizes but ib_async dispatches
        # the update with has_new_bar=False.
        _emit_new_bar_silent(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 31, tzinfo=UTC)))

    received = [e for e in captured if e.get("event") == "market_data.bar_received"]
    received_times = [e["bar_time"] for e in received]
    # Both bars must be logged — the silent one cannot be missed.
    assert len(received) == 2
    assert any("09:30" in t for t in received_times)
    assert any("09:31" in t for t in received_times)
    # Callback must have fired for both — the orchestrator depends on it
    # to evaluate strategies on the just-closed bar.
    await asyncio.sleep(0)
    assert invocations == [1, 1]


@pytest.mark.asyncio
async def test_bar_received_catches_silent_gap_with_warning() -> None:
    """SBLX combination: a 1-minute gap appears across a silent emit.

    09:30 fires normally → 09:32 lands silently (has_new_bar=False) with
    09:31 missing. The gap detector must still flag the missing 09:31
    and log the bar_received for 09:32.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ibkr = _mock_ibkr(trading_class="SCM")
    bar_list = _fake_bar_list(req_id=911)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("SBLX")

    with capture_logs() as captured:
        _emit_new_bar(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 30, tzinfo=UTC)))
        _emit_new_bar_silent(bar_list, _ibkr_bar(datetime(2026, 4, 28, 13, 32, tzinfo=UTC)))

    gaps = [e for e in captured if e.get("event") == "market_data.bar_gap_detected"]
    assert len(gaps) == 1
    assert gaps[0]["missing_bars"] == 1
    received = [e for e in captured if e.get("event") == "market_data.bar_received"]
    assert len(received) == 2
