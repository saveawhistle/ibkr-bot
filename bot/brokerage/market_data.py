"""Live + historical 1-min bar access via ``reqHistoricalData(keepUpToDate=True)``.

Why ``reqHistoricalData`` and not ``reqRealTimeBars``: IBKR's real-time bars are
5-second only. Our strategies work on 1-minute bars, so we use
``reqHistoricalData`` with ``keepUpToDate=True``, which delivers a historical
backfill followed by live 1-minute bars through ``barUpdateEvent``. All bar
DataFrames are indexed by ``America/New_York`` timezone-aware timestamps — no
UTC leaks into downstream indicators (VWAP session boundaries care about local
09:30 ET).

Phase 7.5: ``subscribe_ticks`` wraps ``reqTickByTickData`` so trade-management
code paths that care about tick-level price crosses (notably the scale-out
target check) can react in 100-300 ms instead of 1-3 s via the bar stream.

Phase 10.4 adds ``subscribe_bars_5sec_aggregated`` — an alternative live-bar
path that backfills via the same 1-min historical request (one-shot, no
``keepUpToDate``) and drives live updates from ``reqRealTimeBars(5)`` plus
an in-process :class:`bot.brokerage.bar_aggregator.RollingMinuteAggregator`. Day-7
paper trading observed ~5,250 ms median bar-finalization latency on the
historical path (BIYA 2026-04-30). Spike measurement on AAPL across 8
paired minutes (``scripts/measure_aggregator_to_submit.py``) showed the
aggregator path delivers finalized 1-min candles with ~312 ms median and
~735 ms p95 latency. Selected via ``settings.session.bar_source``;
default ``ibkr_1min`` preserves pre-10.4 behaviour pending shadow-mode
parity verification.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from ib_async import BarDataList, RealTimeBarList, Ticker
from ib_async.objects import TickByTickAllLast

from bot.brokerage.bar_aggregator import MinuteCandle, RollingMinuteAggregator
from bot.brokerage.ibkr_client import ActiveSubscription, IBKRClient, ref_req_id

_log = structlog.get_logger("bot.brokerage.market_data")

_NY = ZoneInfo("America/New_York")
_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]
_ONE_MINUTE = pd.Timedelta(minutes=1)


def _missing_bars_between(prev_ts: pd.Timestamp | None, new_ts: pd.Timestamp) -> int:
    """Phase 9.4 — gap-detection helper: count bars missing between two stamps.

    Returns 0 when ``prev_ts`` is ``None`` (first bar of the session — no
    prior to compare), when bars are exactly one minute apart (no gap), or
    when ``new_ts`` is at/before the expected next-minute slot. Otherwise
    returns the integer count of fully-missing 1-minute bars between them.

    Pure function so the gap arithmetic can be unit-tested without the
    IBKR/event-loop fixture noise.
    """
    if prev_ts is None:
        return 0
    expected = prev_ts + _ONE_MINUTE
    if new_ts <= expected:
        return 0
    return int((new_ts - expected).total_seconds() // 60)


@dataclass
class BarStream:
    """Live-updating bar feed for a single symbol.

    ``bars`` is a pandas DataFrame indexed by ``America/New_York`` timestamps
    with columns ``open, high, low, close, volume, vwap``. The DataFrame object
    is replaced (not mutated in-place) on every update so callers can snapshot
    it safely by assignment.

    Phase 7.3: ``on_new_bar`` is an optional async callback fired when IBKR
    rolls the current bar (i.e. a brand-new bar is appended to the list at
    minute close). Wiring this lets the orchestrator evaluate strategies
    event-driven instead of waiting for the next poll tick — the dominant
    latency source in the poll model (up to ``poll_interval`` s of dead
    time between bar arrival and evaluation).

    Phase 9.4: ``trading_class`` and ``primary_exchange`` are captured at
    qualification time so gap-detection events can pivot SCM (Nasdaq Small
    Cap) vs NMS without re-qualifying inside the hot path.

    Phase 10.4: ``_bar_list`` widens to accept either a ``BarDataList``
    (the ``subscribe_bars`` historical-stream path) or a ``RealTimeBarList``
    (the ``subscribe_bars_5sec_aggregated`` path). Both expose
    ``updateEvent`` with the same ``(bars, has_new_bar)`` signature, so
    callers that hook ``stream._bar_list.updateEvent`` (notably
    ``TradeManager.start_tracking``) work uniformly across both sources.
    """

    symbol: str
    bars: pd.DataFrame
    _bar_list: BarDataList | RealTimeBarList
    _req_id: int = 0
    on_new_bar: Callable[[], Coroutine[Any, Any, None]] | None = field(default=None)
    trading_class: str = ""
    primary_exchange: str = ""


@dataclass
class TickStream:
    """Phase 7.5 tick-by-tick feed for a single symbol.

    Wraps ``reqTickByTickData(contract, "Last")``. ``ticker.tickByTicks`` is
    cleared by ib_async at the end of every TCP batch, so the handler reads
    what's there *during* the event and cannot consult it later. Callers
    should process the batch inside the callback. The ``scale_out_fired``
    latch guards against scheduling the scale-out task twice when multiple
    ticks in the same batch cross the threshold.
    """

    symbol: str
    ticker: Ticker
    tick_type: str
    _req_id: int = 0
    scale_out_fired: bool = False


class MarketData:
    """Per-symbol 1-min bar subscriptions backed by ``ib_async``.

    Instances hold a table of active subscriptions keyed by symbol; callers are
    expected to ``unsubscribe`` each symbol they subscribed to before the IBKR
    socket closes (``close()`` sweeps any remaining ones).
    """

    def __init__(self, ibkr: IBKRClient) -> None:
        """Wire a MarketData around an already-connected ``IBKRClient``."""
        self._ibkr = ibkr
        self._streams: dict[str, BarStream] = {}
        self._ticks: dict[str, TickStream] = {}
        # Phase 9.4 gap-tracking state. Day 7 (2026-04-28) SBLX was missing
        # 9:32 and 9:35 bar evaluations even though a one-shot
        # ``reqHistoricalData(whatToShow="TRADES")`` re-fetch returned both
        # bars with non-trivial volume (2,761 and 47,391 shares). The bars
        # ARE in IBKR's TRADES feed — what failed is the ``has_new_bar=True``
        # signal on the live ``keepUpToDate=True`` stream. So gap detection
        # diffs ``bar_list`` against a per-symbol cursor instead of trusting
        # the flag: any finalized bar that appeared since the last update
        # is logged + gap-checked, even when ib_async dispatched
        # ``has_new_bar=False`` (or skipped the event entirely).
        self._last_bar_time: dict[str, pd.Timestamp] = {}
        self._gap_count_by_symbol: dict[str, int] = {}
        self._gap_count_by_trading_class: dict[str, int] = {}
        self._total_bars_received: int = 0
        self._total_gaps_detected: int = 0
        self._longest_gap_minutes: int = 0

    async def subscribe_bars(
        self,
        symbol: str,
        bar_size: str = "1 min",
        on_new_bar: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> BarStream:
        """Start a live 1-min bar subscription for ``symbol`` and return the ``BarStream``.

        Phase 7.3: ``on_new_bar`` (if set) is scheduled as an asyncio task
        each time IBKR rolls the current bar — i.e. a new minute has started
        and the prior bar's values are now frozen. The callback is async so
        it can ``await`` the full evaluation path (strategy → risk →
        executor). Exceptions inside the task are logged via the task's
        done-callback — a faulting strategy on one symbol must not silently
        kill the event pump for the other symbols.
        """
        if symbol in self._streams:
            return self._streams[symbol]

        contract = await self._ibkr.qualify_stock(symbol)
        bar_list = await self._ibkr.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,  # UTC seconds; we tz-convert to NY ourselves
            keepUpToDate=True,
        )
        req_id = ref_req_id(bar_list)
        initial_frame = _bars_to_frame(bar_list)
        stream = BarStream(
            symbol=symbol,
            bars=initial_frame,
            _bar_list=bar_list,
            _req_id=req_id,
            on_new_bar=on_new_bar,
            trading_class=str(getattr(contract, "tradingClass", "") or ""),
            primary_exchange=str(getattr(contract, "primaryExchange", "") or ""),
        )
        self._streams[symbol] = stream
        # Phase 9.4 — seed the diff cursor to the latest finalized bar in the
        # initial backfill so the first day's worth of historical bars don't
        # get re-logged as "received". When the backfill is empty (starting
        # before any bars exist), the cursor stays unset and the first live
        # bar will be logged as new.
        if len(initial_frame) >= 2:
            self._last_bar_time[symbol] = initial_frame.index[-2]
        await self._ibkr.subscriptions.register(
            ActiveSubscription(
                req_id=req_id,
                kind="historical",
                symbol=symbol,
                ref=bar_list,
            )
        )

        def _on_update(bars: BarDataList, has_new_bar: bool) -> None:
            # Rebuild the frame on each update — simple, cheap for 1-min data
            # over a single session (<= ~400 bars), and avoids edge cases where
            # the trailing bar is overwritten rather than appended.
            stream.bars = _bars_to_frame(bars)
            # Phase 9.4 — diff the bar list against the per-symbol cursor
            # rather than trusting ``has_new_bar``. Day 7 (2026-04-28) SBLX
            # showed that ib_async's ``has_new_bar`` flag can be False (or
            # the event coalesced) even when finalized bars are sitting in
            # ``bar_list``. The diff catches those silent appends.
            new_count = self._record_new_bars(stream)
            if new_count == 0:
                return
            _log.debug(
                "market_data.new_bar",
                symbol=symbol,
                bar_count=len(bars),
                new_finalized=new_count,
                has_new_bar=has_new_bar,
            )
            cb = stream.on_new_bar
            if cb is None:
                return
            task = asyncio.create_task(cb(), name=f"on_new_bar:{symbol}")
            task.add_done_callback(_log_new_bar_task_error)

        bar_list.updateEvent += _on_update
        _log.info("market_data.subscribed", symbol=symbol, bar_size=bar_size)
        return stream

    async def subscribe_bars_5sec_aggregated(
        self,
        symbol: str,
        on_new_bar: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> BarStream:
        """Phase 10.4 — 5-sec → 1-min aggregator alternative to ``subscribe_bars``.

        Two-stage subscription:

        1. **One-shot backfill** via
           ``reqHistoricalDataAsync(barSizeSetting="1 min",
           keepUpToDate=False)`` — same shape and pre-RTH coverage as the
           default path's backfill, but without subscribing for live
           updates from this stream.
        2. **Live updates** via ``reqRealTimeBars(contract, 5, "TRADES",
           useRTH=False)`` + a per-symbol :class:`RollingMinuteAggregator`.
           Each finalized 1-min candle the aggregator emits is appended
           to ``BarStream.bars`` and triggers the same ``on_new_bar``
           callback the default path uses.

        ``useRTH=False`` on the live stream is deliberate. With
        ``useRTH=True`` (the original Phase 10.4 wiring) IBKR doesn't
        deliver any 5-sec bars until 09:30 ET, which produces a silent
        gap for any symbol whose watchlist subscription begins during
        premarket: the backfill covers premarket up to subscribe time,
        then nothing until RTH open. Monday 2026-05-04's CNSP added at
        09:05 ET silently went 25 minutes without bars (logged as
        ``market_data.bar_gap_detected`` at the 09:30 boundary, then
        ``strategy.bar_stale`` skipped the open-bar evaluation). With
        ``useRTH=False`` the live stream covers premarket too, so the
        backfill→live seam is contiguous regardless of subscribe
        time. The default ``subscribe_bars`` path's
        ``keepUpToDate=True`` historical stream already behaves this
        way; this brings the aggregator path to parity.

        ``BarStream._bar_list`` is set to the ``RealTimeBarList`` (not
        the historical backfill list) so callers that hook
        ``stream._bar_list.updateEvent`` for in-progress updates
        (notably ``TradeManager``) keep working — both list types
        publish a ``(bars, has_new_bar)`` event with the same signature.

        ``BarStream.bars`` invariant: trailing row is the in-progress
        next-minute candle (Phase 7.4 — strategies do ``bars.iloc[:-1]``
        to drop it, ``trade_manager.on_bar_update`` reads
        ``bars["close"].iloc[-1]`` for the in-progress and
        ``iloc[-2]`` for the just-closed). When the aggregator has
        no in-progress candle yet (the brief window between :55
        finalization and the next minute's first 5-sec bar arriving),
        we synthesise a placeholder trailing row that copies the
        just-finalized candle but advances the index by one minute,
        preserving the iloc[-1]/iloc[-2] semantics.

        Backfill→live seam: when the aggregator's first finalized
        minute matches a minute already present in the backfill (the
        :55 trigger landed on a minute IBKR's historical query also
        included), the duplicate is dropped — the historical
        backfill's value is more authoritative. Logged as
        ``market_data.aggregator_seam_dropped``.
        """
        if symbol in self._streams:
            return self._streams[symbol]

        contract = await self._ibkr.qualify_stock(symbol)

        # Stage 1: one-shot backfill. Same parameters as ``subscribe_bars``
        # except ``keepUpToDate=False`` — this snapshot is static; live
        # updates come from ``reqRealTimeBars`` below.
        backfill_bar_list = await self._ibkr.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        backfill_df = _bars_to_frame(backfill_bar_list)
        backfill_minutes: set[pd.Timestamp] = set(backfill_df.index)

        # State held in closure variables so the callbacks below can
        # mutate the finalized row list and the bar stream without a
        # dataclass wrapper.
        finalized_rows: list[dict[str, Any]] = []

        # Stage 2: live 5-sec real-time bars. ``useRTH=False`` so the
        # stream covers premarket too — see the docstring's note on the
        # 2026-05-04 CNSP 25-minute silent gap that motivated this.
        rt_bars = self._ibkr.ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=False)
        req_id = ref_req_id(rt_bars)

        # Forward-declare so the aggregator's callback can reference the
        # stream after it's constructed.
        stream_ref: dict[str, BarStream | None] = {"value": None}

        def _on_minute_final(
            candle: MinuteCandle,
            finalized_at: datetime,
            trigger: str,
        ) -> None:
            """Aggregator just finalized minute X — append + fire on_new_bar."""
            ny_minute = pd.Timestamp(candle.minute_start).tz_convert(_NY)
            if ny_minute in backfill_minutes:
                _log.info(
                    "market_data.aggregator_seam_dropped",
                    symbol=symbol,
                    minute=ny_minute.isoformat(),
                    trigger=trigger,
                    hint="aggregator finalized a minute already in the historical backfill; dropping.",
                )
                return
            finalized_rows.append(_candle_to_row(candle))
            stream = stream_ref["value"]
            if stream is None:
                return
            self._record_aggregator_minute(stream, candle)
            cb = stream.on_new_bar
            if cb is None:
                return
            task = asyncio.create_task(cb(), name=f"on_new_bar:{symbol}")
            task.add_done_callback(_log_new_bar_task_error)

        aggregator = RollingMinuteAggregator(symbol=symbol, on_minute_final=_on_minute_final)

        # Build the BarStream. ``_bar_list`` is the RealTimeBarList so
        # TradeManager's ``stream._bar_list.updateEvent += ...`` hook
        # continues to fire on every 5-sec update.
        stream = BarStream(
            symbol=symbol,
            bars=backfill_df,
            _bar_list=rt_bars,
            _req_id=req_id,
            on_new_bar=on_new_bar,
            trading_class=str(getattr(contract, "tradingClass", "") or ""),
            primary_exchange=str(getattr(contract, "primaryExchange", "") or ""),
        )
        stream_ref["value"] = stream
        self._streams[symbol] = stream

        # Seed the gap-detection cursor from the backfill so the
        # aggregator's first emitted minute doesn't get spuriously
        # logged as a gap from "no prior bar".
        if len(backfill_df) >= 1:
            self._last_bar_time[symbol] = backfill_df.index[-1]

        await self._ibkr.subscriptions.register(
            ActiveSubscription(
                req_id=req_id,
                kind="historical",  # rt-bars use the same cancel kind for sweep
                symbol=symbol,
                ref=rt_bars,
            )
        )

        def _on_5sec_update(bars: RealTimeBarList, has_new_bar: bool) -> None:
            """Drive the aggregator on each 5-sec bar; rebuild ``stream.bars``."""
            if not bars:
                return
            try:
                aggregator.on_5sec_bar(bars[-1])
            except Exception as exc:  # noqa: BLE001 - aggregator faults must not crash dispatch
                _log.error("market_data.aggregator_failed", symbol=symbol, error=str(exc))
                return
            stream.bars = _build_aggregated_frame(backfill_df, finalized_rows, aggregator)

        rt_bars.updateEvent += _on_5sec_update
        _log.info(
            "market_data.subscribed_5sec_aggregated",
            symbol=symbol,
            backfill_minutes=len(backfill_df),
        )
        return stream

    def _record_aggregator_minute(self, stream: BarStream, candle: MinuteCandle) -> None:
        """Phase 10.4 — emit ``market_data.bar_received`` for a finalized aggregator minute.

        Mirrors the per-bar fields ``_record_new_bars`` writes for the
        historical-bar path so downstream consumers (CLI status, JSONL
        grep tooling, gap-summary aggregator) see the same event shape
        regardless of which source produced the bar. Also runs the
        same gap-detection arithmetic against ``_last_bar_time`` and
        bumps the same per-symbol / per-trading-class counters that
        feed ``market_data.session_gap_summary``.
        """
        ts = pd.Timestamp(candle.minute_start).tz_convert(_NY)
        last_ts = self._last_bar_time.get(stream.symbol)
        missing = _missing_bars_between(last_ts, ts) if last_ts is not None else 0
        _log.info(
            "market_data.bar_received",
            symbol=stream.symbol,
            bar_time=ts.isoformat(),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
            source="rtbars_aggregated",
        )
        self._total_bars_received += 1
        if missing > 0:
            self._total_gaps_detected += 1
            self._gap_count_by_symbol[stream.symbol] = (
                self._gap_count_by_symbol.get(stream.symbol, 0) + 1
            )
            tc_key = stream.trading_class or "unknown"
            self._gap_count_by_trading_class[tc_key] = (
                self._gap_count_by_trading_class.get(tc_key, 0) + 1
            )
            if missing > self._longest_gap_minutes:
                self._longest_gap_minutes = missing
            _log.warning(
                "market_data.bar_gap_detected",
                symbol=stream.symbol,
                previous_bar_time=last_ts.isoformat() if last_ts is not None else None,
                new_bar_time=ts.isoformat(),
                missing_bars=missing,
                trading_class=stream.trading_class,
                primary_exchange=stream.primary_exchange,
            )
        self._last_bar_time[stream.symbol] = ts

    async def historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "1 min",
    ) -> pd.DataFrame:
        """One-shot historical pull — premarket included via ``useRTH=False``."""
        contract = await self._ibkr.qualify_stock(symbol)
        bar_list = await self._ibkr.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        return _bars_to_frame(bar_list)

    async def subscribe_ticks(
        self,
        symbol: str,
        on_tick: Callable[[TickByTickAllLast], Coroutine[Any, Any, None]],
        tick_type: str = "Last",
    ) -> TickStream:
        """Phase 7.5: subscribe to tick-by-tick prints for ``symbol``.

        ``on_tick`` is fired once per non-Form-T trade print during the
        subscription's lifetime. It is scheduled via ``asyncio.create_task``
        so a slow handler does not block ib_async's event dispatch. The
        returned ``TickStream`` exposes ``scale_out_fired`` — callers that
        use this feed for one-shot triggers (scale-out) set that flag in
        their handler to stop scheduling duplicate tasks when a batch of
        ticks all cross the threshold.

        Form T (``specialConditions`` containing ``"FT"``) prints — late,
        irregular, or out-of-hours reports — are filtered out before
        invoking ``on_tick``. For scale-out we want regular RTH prints
        only; a Form T print that happens to cross the target should not
        fire a market sell.
        """
        if symbol in self._ticks:
            return self._ticks[symbol]

        contract = await self._ibkr.qualify_stock(symbol)
        ticker = self._ibkr.ib.reqTickByTickData(
            contract, tick_type, numberOfTicks=0, ignoreSize=False
        )
        req_id = ref_req_id(ticker)
        stream = TickStream(symbol=symbol, ticker=ticker, tick_type=tick_type, _req_id=req_id)
        self._ticks[symbol] = stream
        await self._ibkr.subscriptions.register(
            ActiveSubscription(
                req_id=req_id,
                kind="tick_by_tick",
                symbol=symbol,
                ref=ticker,
            )
        )

        def _on_ticker_update(t: Ticker) -> None:
            # ib_async clears ``tickByTicks`` at the end of each TCP batch,
            # so we must iterate within this event handler.
            for tick in t.tickByTicks:
                if not isinstance(tick, TickByTickAllLast):
                    continue
                if "FT" in (tick.specialConditions or ""):
                    continue
                task = asyncio.create_task(on_tick(tick), name=f"on_tick:{symbol}")
                task.add_done_callback(_log_tick_task_error)

        ticker.updateEvent += _on_ticker_update
        _log.info("market_data.tick_subscribed", symbol=symbol, tick_type=tick_type)
        return stream

    async def unsubscribe_ticks(self, symbol: str) -> None:
        """Cancel the tick-by-tick subscription for ``symbol`` if active."""
        stream = self._ticks.pop(symbol, None)
        if stream is None:
            return
        try:
            contract = await self._ibkr.qualify_stock(symbol)
            self._ibkr.ib.cancelTickByTickData(contract, stream.tick_type)
        except Exception as exc:  # noqa: BLE001 - cancel shapes vary
            _log.warning("market_data.tick_cancel_failed", symbol=symbol, error=str(exc))
        await self._ibkr.subscriptions.unregister(stream._req_id)  # noqa: SLF001 — internal field
        _log.info("market_data.tick_unsubscribed", symbol=symbol)

    async def unsubscribe(self, symbol: str) -> None:
        """Cancel the live subscription for ``symbol`` if one is active.

        Phase 10.4: dispatches on ``stream._bar_list`` type so the right
        IBKR cancel method fires. ``RealTimeBarList`` (from the 5-sec
        aggregator path) needs ``cancelRealTimeBars``; ``BarDataList``
        (from the historical 1-min path) needs ``cancelHistoricalData``.
        """
        stream = self._streams.pop(symbol, None)
        if stream is None:
            return
        try:
            if isinstance(stream._bar_list, RealTimeBarList):
                self._ibkr.ib.cancelRealTimeBars(stream._bar_list)
            else:
                self._ibkr.ib.cancelHistoricalData(stream._bar_list)
        except Exception as exc:  # noqa: BLE001 - ib can raise many shapes on cancel
            _log.warning("market_data.cancel_failed", symbol=symbol, error=str(exc))
        await self._ibkr.subscriptions.unregister(stream._req_id)
        _log.info("market_data.unsubscribed", symbol=symbol)

    def _record_new_bars(self, stream: BarStream) -> int:
        """Phase 9.4 — diff ``stream.bars`` against per-symbol cursor; log new finalized bars.

        Returns the number of newly-finalized bars detected on this update.
        Fires ``market_data.bar_received`` for each, plus
        ``market_data.bar_gap_detected`` when consecutive finalized bars
        are not 1 minute apart.

        The cursor (``self._last_bar_time[stream.symbol]``) advances to the
        latest finalized bar, so subsequent updates only emit deltas. The
        in-progress bar (``stream.bars.iloc[-1]``) is excluded — it can
        change on every tick and isn't a "received" bar in the closed-bar
        sense.

        This replaces the prior ``has_new_bar``-driven design: the SBLX
        2026-04-28 evidence shows that flag is unreliable. A one-shot
        re-fetch of SBLX returned all 11 morning bars (including the
        "missing" 9:32 / 9:35 with non-trivial volume), so the bars were
        in ``bar_list`` — the ``has_new_bar`` event just didn't fire for
        the right transitions. Diffing the list directly bypasses that.
        """
        df = stream.bars
        if len(df) < 2:
            return 0  # only the in-progress bar (or empty) — nothing finalized
        finalized = df.iloc[:-1]
        last_ts = self._last_bar_time.get(stream.symbol)
        # No backfill seed (empty initial fetch) → every finalized bar is
        # a live-stream arrival; otherwise diff against the cursor.
        new_rows = finalized.loc[finalized.index > last_ts] if last_ts is not None else finalized
        if new_rows.empty:
            return 0

        prev_ts: pd.Timestamp | None = last_ts
        for raw_ts, row in new_rows.iterrows():
            ts = cast("pd.Timestamp", raw_ts)
            missing = _missing_bars_between(prev_ts, ts)
            _log.info(
                "market_data.bar_received",
                symbol=stream.symbol,
                bar_time=ts.isoformat(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                source="stream",
            )
            self._total_bars_received += 1
            if missing > 0:
                self._total_gaps_detected += 1
                self._gap_count_by_symbol[stream.symbol] = (
                    self._gap_count_by_symbol.get(stream.symbol, 0) + 1
                )
                tc_key = stream.trading_class or "unknown"
                self._gap_count_by_trading_class[tc_key] = (
                    self._gap_count_by_trading_class.get(tc_key, 0) + 1
                )
                if missing > self._longest_gap_minutes:
                    self._longest_gap_minutes = missing
                _log.warning(
                    "market_data.bar_gap_detected",
                    symbol=stream.symbol,
                    previous_bar_time=prev_ts.isoformat() if prev_ts is not None else None,
                    new_bar_time=ts.isoformat(),
                    missing_bars=missing,
                    trading_class=stream.trading_class,
                    primary_exchange=stream.primary_exchange,
                )
            prev_ts = ts

        self._last_bar_time[stream.symbol] = cast("pd.Timestamp", new_rows.index[-1])
        return len(new_rows)

    async def close(self) -> None:
        """Cancel every active subscription (call before disconnecting the IB socket)."""
        for symbol in list(self._ticks):
            await self.unsubscribe_ticks(symbol)
        for symbol in list(self._streams):
            await self.unsubscribe(symbol)
        # Phase 9.4 — emit a final aggregate so post-session review can scan
        # one event for the gap profile across the trading day.
        _log.info(
            "market_data.session_gap_summary",
            total_bars_received=self._total_bars_received,
            total_gaps_detected=self._total_gaps_detected,
            gaps_by_symbol=dict(self._gap_count_by_symbol),
            gaps_by_trading_class=dict(self._gap_count_by_trading_class),
            longest_gap_minutes=self._longest_gap_minutes,
        )


def _log_tick_task_error(task: asyncio.Task[None]) -> None:
    """Phase 7.5: surface exceptions from event-driven ``on_tick`` tasks.

    Mirrors ``_log_new_bar_task_error``. Tick-handler faults on one
    position's feed must not silently starve the others.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.error("market_data.on_tick_failed", task=task.get_name(), error=str(exc))


def _log_new_bar_task_error(task: asyncio.Task[None]) -> None:
    """Surface exceptions from event-driven ``on_new_bar`` tasks.

    ``asyncio.create_task`` fires forget-and-wait; an unhandled exception
    only logs a warning at task gc time. Wiring a done-callback means
    strategy faults on one symbol's bar event are visible immediately and
    don't silently starve downstream handlers.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.error("market_data.on_new_bar_failed", task=task.get_name(), error=str(exc))


def _candle_to_row(candle: MinuteCandle) -> dict[str, Any]:
    """Phase 10.4 — convert a finalized aggregator candle to a DataFrame row dict.

    Schema matches ``_bars_to_frame``'s row dict so the resulting frames
    are concatable. ``date`` is the candle's ``minute_start`` (tz-aware
    UTC); the wrapping ``_build_aggregated_frame`` converts the index
    to NY tz to match the rest of the system.
    """
    return {
        "date": pd.Timestamp(candle.minute_start),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "vwap": candle.vwap,
    }


def _build_aggregated_frame(
    backfill_df: pd.DataFrame,
    finalized_rows: list[dict[str, Any]],
    aggregator: RollingMinuteAggregator,
) -> pd.DataFrame:
    """Phase 10.4 — combine backfill + aggregator-finalized rows + in-progress trailing.

    The trailing in-progress row is *required* by Phase 7.4 (strategies
    drop ``iloc[-1]``) and Phase 7.9 (trade_manager reads ``iloc[-1]``
    as in-progress and ``iloc[-2]`` as just-closed). When the aggregator
    has emitted a finalized minute but hasn't yet seen the next minute's
    first 5-sec bar, we synthesise a placeholder trailing row keyed on
    ``minute_start + 1 minute`` with values copied from the just-finalized
    candle (volume zeroed). The placeholder is replaced with a real
    in-progress candle on the next 5-sec arrival.
    """
    rows = list(finalized_rows)
    in_progress = aggregator.in_progress_candle
    if in_progress is not None:
        rows.append(_candle_to_row(in_progress))
    elif rows:
        last = rows[-1]
        synthetic_ts = pd.Timestamp(last["date"]) + pd.Timedelta(minutes=1)
        rows.append(
            {
                "date": synthetic_ts,
                "open": last["close"],
                "high": last["close"],
                "low": last["close"],
                "close": last["close"],
                "volume": 0.0,
                "vwap": last["close"],
            }
        )

    if not rows:
        return backfill_df

    addition = pd.DataFrame(rows).set_index("date")
    # Aggregator candle timestamps are tz-aware UTC; convert to NY to
    # match the backfill frame's index.
    addition_idx = cast("pd.DatetimeIndex", addition.index)
    if addition_idx.tz is None:
        addition_idx = addition_idx.tz_localize("UTC")
    addition.index = addition_idx.tz_convert(_NY)
    addition.index.name = "date"

    if backfill_df.empty:
        return addition[_BAR_COLUMNS]
    return pd.concat([backfill_df, addition[_BAR_COLUMNS]])


def _bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    """Convert an ``ib_async`` BarDataList into a tz-aware (NY) DataFrame."""
    if not bars:
        return pd.DataFrame(columns=_BAR_COLUMNS).astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "vwap": float,
            }
        )
    rows = [
        {
            "date": bar.date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.average),
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows).set_index("date")
    # formatDate=2 returns tz-naive UTC datetimes; localize then convert to NY.
    idx = pd.to_datetime(frame.index, utc=True)
    frame.index = idx.tz_convert(_NY)
    frame.index.name = "date"
    return frame[_BAR_COLUMNS]
