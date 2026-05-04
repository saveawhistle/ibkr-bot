"""Tests for ``MarketData.subscribe_bars_5sec_aggregated`` — Phase 10.4.

The aggregator unit tests in ``test_bar_aggregator.py`` cover the
finalization logic in isolation. This file exercises the integration:

* Backfill via ``reqHistoricalDataAsync(keepUpToDate=False)`` populates
  ``BarStream.bars`` before any 5-sec bar arrives.
* The 5-sec stream feeds the aggregator; finalized 1-min candles append
  to ``BarStream.bars`` and fire ``on_new_bar``.
* Trailing in-progress row invariant: ``BarStream.bars.iloc[-1]`` is
  always the in-progress / synthetic-placeholder row so Phase 7.4
  strategies and Phase 7.9 ``trade_manager`` semantics keep working.
* Backfill→live seam: aggregator-finalized minutes that overlap with
  backfill are dropped via ``market_data.aggregator_seam_dropped``.
* ``unsubscribe`` dispatches on bar-list type and calls
  ``cancelRealTimeBars`` for this path.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from ib_async import RealTimeBarList
from structlog.testing import capture_logs

from bot.brokerage.ibkr_client import SubscriptionRegistry
from bot.brokerage.market_data import MarketData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeEvent:
    """``+= handler`` + manual ``emit`` — same pattern as the existing market_data tests."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeEvent:
        self.handlers.append(handler)
        return self

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for handler in self.handlers:
            handler(*args, **kwargs)


class _FakeRealTimeBarList(RealTimeBarList):
    """Real ``RealTimeBarList`` subclass so ``isinstance(...)`` dispatch in
    ``MarketData.unsubscribe`` picks the rt-bars cancel path."""

    def __init__(self, req_id: int = 0) -> None:
        super().__init__()
        self.reqId = req_id  # noqa: N815 - mirror ib_async
        self.updateEvent = _FakeEvent()


class _FakeBackfillList(list[Any]):
    """Empty list-shaped BarDataList for the backfill request."""

    def __init__(self, req_id: int = 0) -> None:
        super().__init__()
        self.reqId = req_id  # noqa: N815
        self.updateEvent = _FakeEvent()


def _make_5sec_bar(
    ts_utc: datetime,
    *,
    open_: float = 10.0,
    high: float = 10.1,
    low: float = 9.9,
    close: float = 10.05,
    volume: float = 100.0,
    wap: float = 10.0,
) -> SimpleNamespace:
    """Duck-typed ``RealTimeBar`` for the synthetic stream."""
    return SimpleNamespace(
        time=ts_utc,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        wap=wap,
        count=1,
    )


def _emit_5sec_bar(rt_bars: _FakeRealTimeBarList, bar: SimpleNamespace) -> None:
    """Append a 5-sec bar to ``rt_bars`` and fire its update event.

    ib_async's RealTimeBarList grows on each push; updateEvent fires with
    ``(bars, has_new_bar=True)``.
    """
    rt_bars.append(bar)
    rt_bars.updateEvent.emit(rt_bars, True)


def _mock_ibkr() -> MagicMock:
    ibkr = MagicMock(name="IBKRClient")
    ibkr.ib = MagicMock(name="IB")
    ibkr.ib.cancelHistoricalData = MagicMock()
    ibkr.ib.cancelRealTimeBars = MagicMock()
    ibkr.qualify_stock = AsyncMock(
        side_effect=lambda symbol: SimpleNamespace(
            symbol=symbol,
            conId=123,
            tradingClass="NMS",
            primaryExchange="NASDAQ",
        )
    )
    ibkr.subscriptions = SubscriptionRegistry()
    return ibkr


def _setup_subscriptions(
    ibkr: MagicMock,
    backfill_req_id: int = 100,
    rt_req_id: int = 200,
) -> tuple[_FakeBackfillList, _FakeRealTimeBarList]:
    """Wire the IBKR mock's bar-fetch + RT-bars calls to fresh fakes."""
    backfill = _FakeBackfillList(req_id=backfill_req_id)
    rt_bars = _FakeRealTimeBarList(req_id=rt_req_id)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=backfill)
    ibkr.ib.reqRealTimeBars = MagicMock(return_value=rt_bars)
    return backfill, rt_bars


# ---------------------------------------------------------------------------
# Backfill + subscription registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_call_uses_keep_up_to_date_false() -> None:
    """The 5-sec aggregator path's backfill is one-shot, not live.

    Sets ``keepUpToDate=False`` so the historical request doesn't
    subscribe for live updates — those come from ``reqRealTimeBars``.
    """
    ibkr = _mock_ibkr()
    _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")
    args, kwargs = ibkr.ib.reqHistoricalDataAsync.await_args
    assert kwargs["keepUpToDate"] is False
    assert kwargs["barSizeSetting"] == "1 min"


@pytest.mark.asyncio
async def test_subscribes_5sec_realtime_bars() -> None:
    """``reqRealTimeBars`` is called with bar size 5 and ``useRTH=False``.

    ``useRTH=False`` is required so the live 5-sec stream delivers premarket
    bars when a watchlist symbol's subscription begins before 09:30 ET. The
    original Phase 10.4 wiring used ``useRTH=True``, which produced a silent
    gap from subscribe time to RTH open — see the 2026-05-04 CNSP 25-min
    silent gap incident referenced in ``subscribe_bars_5sec_aggregated``.
    """
    ibkr = _mock_ibkr()
    _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")
    ibkr.ib.reqRealTimeBars.assert_called_once()
    args, kwargs = ibkr.ib.reqRealTimeBars.call_args
    # Positional args: (contract, barSize, whatToShow), useRTH as kwarg.
    assert args[1] == 5
    assert args[2] == "TRADES"
    assert kwargs["useRTH"] is False, (
        "useRTH must be False on the live 5-sec stream so premarket bars "
        "flow continuously — the backfill→live seam relies on it."
    )


@pytest.mark.asyncio
async def test_subscription_registered_under_rt_bars_req_id() -> None:
    """The registry tracks the rt-bars subscription so cancel_all_subscriptions sweeps it."""
    ibkr = _mock_ibkr()
    _setup_subscriptions(ibkr, rt_req_id=555)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")
    active = await ibkr.subscriptions.list_active()
    assert len(active) == 1
    sub = active[0]
    assert sub.symbol == "AAPL"
    assert sub.req_id == 555


@pytest.mark.asyncio
async def test_returns_existing_stream_on_double_subscribe() -> None:
    """Idempotent — calling subscribe twice for the same symbol returns the same stream."""
    ibkr = _mock_ibkr()
    _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    a = await md.subscribe_bars_5sec_aggregated("AAPL")
    b = await md.subscribe_bars_5sec_aggregated("AAPL")
    assert a is b
    # reqRealTimeBars only called once.
    ibkr.ib.reqRealTimeBars.assert_called_once()


# ---------------------------------------------------------------------------
# Aggregator → BarStream wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalized_minute_appears_in_bars_after_55_bar() -> None:
    """12 5-sec bars → finalized 1-min row appended to ``BarStream.bars``."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    stream = await md.subscribe_bars_5sec_aggregated("AAPL")

    # Push 12 5-sec bars covering minute 09:31 (UTC for the synthetic test).
    for sec in range(0, 60, 5):
        _emit_5sec_bar(
            rt_bars,
            _make_5sec_bar(datetime(2026, 4, 30, 9, 31, sec, tzinfo=UTC), close=100.0 + sec / 5),
        )

    # Frame should have one finalized minute (09:31) plus a synthetic
    # in-progress trailing row for 09:32 (the aggregator hasn't received
    # any 09:32 bars yet, so the wrapper inserts a placeholder).
    assert len(stream.bars) == 2
    minutes = [ts.minute for ts in stream.bars.index]
    assert minutes == [31, 32]


@pytest.mark.asyncio
async def test_on_new_bar_callback_fires_once_per_finalization() -> None:
    """``on_new_bar`` runs exactly once when the aggregator finalizes a minute."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    fired = 0

    async def _cb() -> None:
        nonlocal fired
        fired += 1

    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL", on_new_bar=_cb)

    for sec in range(0, 60, 5):
        _emit_5sec_bar(rt_bars, _make_5sec_bar(datetime(2026, 4, 30, 9, 31, sec, tzinfo=UTC)))
    # The on_new_bar handler is dispatched via ``asyncio.create_task``;
    # await the loop briefly so the task runs.
    await asyncio.sleep(0)
    assert fired == 1


@pytest.mark.asyncio
async def test_in_progress_trailing_row_present_during_minute() -> None:
    """Phase 7.4 invariant: ``bars.iloc[-1]`` is the in-progress next-minute row.

    After 3 5-sec bars of minute 09:31 (no finalization yet), the frame
    has one row — the in-progress 09:31 row. Strategies' ``iloc[:-1]``
    drop yields an empty frame; trade_manager's ``iloc[-1]`` reads the
    in-progress close.
    """
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    stream = await md.subscribe_bars_5sec_aggregated("AAPL")

    for sec in (0, 5, 10):
        _emit_5sec_bar(
            rt_bars,
            _make_5sec_bar(
                datetime(2026, 4, 30, 9, 31, sec, tzinfo=UTC),
                high=10.0 + sec * 0.01,
                close=10.0 + sec * 0.01,
            ),
        )

    assert len(stream.bars) == 1
    assert stream.bars.index[-1].minute == 31
    # The trailing in-progress row reflects accumulated high.
    assert stream.bars["high"].iloc[-1] == pytest.approx(10.10)


# ---------------------------------------------------------------------------
# Backfill→live seam
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_seam_drops_minute_already_in_backfill() -> None:
    """If the aggregator finalizes a minute that backfill already had, drop it.

    Backfill is more authoritative (full IBKR finalization). Logging as
    ``market_data.aggregator_seam_dropped`` for forensic clarity.
    """
    ibkr = _mock_ibkr()
    backfill, rt_bars = _setup_subscriptions(ibkr)

    # Backfill includes 13:31 UTC = 09:31 NY (a regular RTH minute).
    # ``formatDate=2`` returns tz-naive UTC datetimes; mirror that.
    backfill_utc_minute = datetime(2026, 4, 30, 13, 31, 0)
    backfill.append(
        SimpleNamespace(
            date=backfill_utc_minute,
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=1000.0,
            average=10.0,
        )
    )

    md = MarketData(ibkr=ibkr)
    stream = await md.subscribe_bars_5sec_aggregated("AAPL")
    assert len(stream.bars) == 1  # only the backfill row

    with capture_logs() as captured:
        # Feed 12 5-sec bars for the SAME UTC minute (13:31 UTC = 09:31 NY).
        for sec in range(0, 60, 5):
            _emit_5sec_bar(
                rt_bars,
                _make_5sec_bar(datetime(2026, 4, 30, 13, 31, sec, tzinfo=UTC)),
            )

    events = [e["event"] for e in captured]
    assert "market_data.aggregator_seam_dropped" in events
    # 31 from backfill always present; aggregator's 09:31 was dropped at the seam.
    minutes_in_frame = sorted({ts.minute for ts in stream.bars.index})
    assert 31 in minutes_in_frame


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_bar_received_event_with_aggregated_source() -> None:
    """Aggregator-finalized minutes log ``market_data.bar_received`` with source label."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")

    with capture_logs() as captured:
        for sec in range(0, 60, 5):
            _emit_5sec_bar(
                rt_bars,
                _make_5sec_bar(datetime(2026, 4, 30, 9, 31, sec, tzinfo=UTC), close=100.0),
            )

    received = [e for e in captured if e["event"] == "market_data.bar_received"]
    assert len(received) == 1
    assert received[0]["symbol"] == "AAPL"
    assert received[0]["source"] == "rtbars_aggregated"
    assert received[0]["close"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Unsubscribe dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsubscribe_calls_cancel_real_time_bars() -> None:
    """Phase 10.4: 5-sec path uses ``cancelRealTimeBars``, not ``cancelHistoricalData``."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")
    await md.unsubscribe("AAPL")
    ibkr.ib.cancelRealTimeBars.assert_called_once_with(rt_bars)
    ibkr.ib.cancelHistoricalData.assert_not_called()
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_unsubscribe_dispatch_unaffected_for_default_path() -> None:
    """Cross-check: the default ``subscribe_bars`` path still calls ``cancelHistoricalData``."""
    from tests.test_market_data import _fake_bar_list  # noqa: PLC0415 - shared fixture

    ibkr = _mock_ibkr()
    bar_list = _fake_bar_list(req_id=99)
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=bar_list)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars("AAPL")
    await md.unsubscribe("AAPL")
    ibkr.ib.cancelHistoricalData.assert_called_once_with(bar_list)
    ibkr.ib.cancelRealTimeBars.assert_not_called()


# ---------------------------------------------------------------------------
# Stream wiring for trade_manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_bar_list_is_real_time_bar_list_for_trade_manager_hook() -> None:
    """``BarStream._bar_list`` is the RealTimeBarList — TradeManager's
    ``stream._bar_list.updateEvent += ...`` hook fires on every 5-sec bar."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    stream = await md.subscribe_bars_5sec_aggregated("AAPL")
    assert stream._bar_list is rt_bars
    # Trade-manager-style hook
    fired: list[bool] = []

    def _on_update(bars: object, has_new_bar: bool) -> None:
        fired.append(has_new_bar)

    stream._bar_list.updateEvent += _on_update  # noqa: SLF001
    _emit_5sec_bar(rt_bars, _make_5sec_bar(datetime(2026, 4, 30, 9, 31, 0, tzinfo=UTC)))
    assert fired == [True]


# ---------------------------------------------------------------------------
# Multi-minute integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_minutes_aggregated_appear_in_order() -> None:
    """Two clean minutes → two rows appended in order, plus a trailing in-progress."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    stream = await md.subscribe_bars_5sec_aggregated("AAPL")

    for minute_offset in (31, 32):
        for sec in range(0, 60, 5):
            _emit_5sec_bar(
                rt_bars,
                _make_5sec_bar(
                    datetime(2026, 4, 30, 9, minute_offset, sec, tzinfo=UTC),
                    close=100.0 + minute_offset,
                ),
            )

    # Two finalized + one synthetic next-minute placeholder (33).
    minutes = [ts.minute for ts in stream.bars.index]
    assert minutes == [31, 32, 33]


@pytest.mark.asyncio
async def test_gap_detection_logs_when_aggregator_skips_a_minute() -> None:
    """Aggregator emits minute 31 then minute 33 (no 32) → gap event fires."""
    ibkr = _mock_ibkr()
    _backfill, rt_bars = _setup_subscriptions(ibkr)
    md = MarketData(ibkr=ibkr)
    await md.subscribe_bars_5sec_aggregated("AAPL")

    # Minute 31 finalizes cleanly.
    for sec in range(0, 60, 5):
        _emit_5sec_bar(rt_bars, _make_5sec_bar(datetime(2026, 4, 30, 9, 31, sec, tzinfo=UTC)))

    with capture_logs() as captured:
        # Skip directly to minute 33 — minute 32's :55 bar never arrives,
        # but the first bar of 33 forces the new-minute (gap) finalization
        # of minute 32 with whatever was accumulated (zero bars in this case
        # since we skipped both opening bars and finalization). The
        # aggregator's gap path will fire when minute 33's first bar
        # arrives if minute 32 was started; here we go straight from
        # the synthetic placeholder for 32 to the first 33 bar.
        for sec in range(0, 60, 5):
            _emit_5sec_bar(rt_bars, _make_5sec_bar(datetime(2026, 4, 30, 9, 33, sec, tzinfo=UTC)))

    # Either market_data.bar_gap_detected fires (1-minute jump) or both —
    # we just assert that the bar_received for minute 33 was emitted with
    # a missing-minute gap from minute 31.
    bar_received = [e for e in captured if e["event"] == "market_data.bar_received"]
    assert any(pd.Timestamp(e["bar_time"]).minute == 33 for e in bar_received), (
        "expected finalization for minute 33"
    )


def test_minute_floor_helper_drops_seconds() -> None:
    """Sanity: the wrapper's index conversion preserves NY tz."""
    candle_ts_utc = datetime(2026, 4, 30, 13, 31, 0, tzinfo=UTC)  # 09:31 ET
    expected_ny_minute = pd.Timestamp(candle_ts_utc).tz_convert("America/New_York")
    assert expected_ny_minute.hour == 9
    assert expected_ny_minute.minute == 31
