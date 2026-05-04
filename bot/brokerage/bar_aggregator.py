"""Phase 10.4 — 5-sec → 1-min OHLCV aggregator.

Drives the new ``MarketData.subscribe_bars_5sec_aggregated`` path. Receives
5-sec bars from ``IB.reqRealTimeBars(contract, 5, ...)`` and emits
finalized 1-min candles via a callback. The point of the move: today's
``reqHistoricalData(keepUpToDate=True, "1 min")`` path delivers a 1-min
bar ~5,250 ms after its close (BIYA 2026-04-30 reference); the 5-sec
real-time-bars path delivers each 5-sec bar ~300-500 ms after close,
so finalizing on the 12th 5-sec bar drops the wait by an order of
magnitude (measured median: 312 ms post-aggregation, in
``scripts/measure_aggregator_to_submit.py``).

Two finalization triggers, in priority order:

1. **Twelfth-bar trigger** — a 5-sec bar with ``time.second == 55``
   covers ``[xx:55, xx+1:00)`` and is mathematically the last bar of
   minute ``xx``. Receipt of this bar means the minute is closed; fire
   the finalization event immediately. This is the typical path.

2. **New-minute trigger (gap fallback)** — if a 5-sec bar arrives whose
   minute differs from the current accumulating minute *without* the
   :55 bar having appeared, the :55 bar was dropped (Phase 9.4
   precedent: SBLX 2026-04-28 had one-minute bars silently missing in
   live ``keepUpToDate=True`` updates; by symmetry the 5-sec stream is
   not immune). Finalize the prior minute on best-effort information
   and emit a ``bar_aggregator.gap_detected`` event so the operator
   can audit forensic resolution of the missing bar later.

Idempotency: a minute is finalized at most once. The
``_finalized_minutes`` set guards both triggers — if the :55 bar
arrives *and then* a same-minute "duplicate" or out-of-order bar
shows up, no double-emission. Out-of-order or older-minute bars after
finalization log ``bar_aggregator.discontinuity`` and are dropped.

VWAP arithmetic: emitted as ``Σ(p_i × v_i) / Σ(v_i)`` where ``p_i``
is each 5-sec bar's ``wap`` (volume-weighted average price IBKR
already computed for that 5-sec window) and ``v_i`` is the bar's
volume. This matches the canonical 1-min VWAP definition for
TRADES bars; parity against IBKR's own 1-min VWAP within float-
arithmetic tolerance is asserted by the integration spike before
flipping the production default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import structlog

_log = structlog.get_logger("bot.brokerage.bar_aggregator")

# A 1-min OHLCV candle; emitted by the aggregator when a minute is
# finalized. Fields mirror the columns in ``BarStream.bars`` so the
# wrapping ``MarketData`` layer can drop it straight into the DataFrame.
_TWELFTH_BAR_SECOND: Final[int] = 55


@dataclass
class MinuteCandle:
    """One finalized 1-min OHLCV candle.

    ``minute_start`` is the open of the minute (e.g. ``09:31:00`` for the
    candle covering ``[09:31:00, 09:32:00)``). All timestamps are
    tz-aware UTC.
    """

    minute_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    bars_contributing: int


# Callback signature: (finalized_candle, finalized_at_wall_clock, trigger_label)
MinuteFinalCallback = Callable[[MinuteCandle, datetime, str], None]


@dataclass
class _Accumulator:
    """In-progress 1-min candle being accumulated from 5-sec bars."""

    minute_start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    weighted_price_sum: float  # Σ(wap × volume) for VWAP
    bars_contributing: int

    def merge(self, bar: Any) -> None:
        """Fold one more 5-sec bar into the accumulator."""
        bar_high = float(bar.high)
        bar_low = float(bar.low)
        bar_close = float(bar.close)
        bar_volume = float(bar.volume)
        bar_wap = float(getattr(bar, "wap", bar_close) or bar_close)
        if bar_high > self.high:
            self.high = bar_high
        if bar_low < self.low:
            self.low = bar_low
        self.close = bar_close
        self.volume += bar_volume
        self.weighted_price_sum += bar_wap * bar_volume
        self.bars_contributing += 1

    def to_candle(self) -> MinuteCandle:
        """Snapshot the accumulator to an immutable :class:`MinuteCandle`.

        VWAP defaults to ``close`` when total volume is zero (a synthetic
        zero-volume minute), preventing a ZeroDivisionError. Real RTH
        minutes always have non-zero volume on a tradeable name.
        """
        vwap = self.weighted_price_sum / self.volume if self.volume > 0 else self.close
        return MinuteCandle(
            minute_start=self.minute_start,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=vwap,
            bars_contributing=self.bars_contributing,
        )

    @classmethod
    def from_first_bar(cls, bar: Any, minute_start: datetime) -> _Accumulator:
        """Seed an accumulator from the first 5-sec bar of a minute."""
        bar_open = float(bar.open_)
        bar_high = float(bar.high)
        bar_low = float(bar.low)
        bar_close = float(bar.close)
        bar_volume = float(bar.volume)
        bar_wap = float(getattr(bar, "wap", bar_close) or bar_close)
        return cls(
            minute_start=minute_start,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
            volume=bar_volume,
            weighted_price_sum=bar_wap * bar_volume,
            bars_contributing=1,
        )


def _minute_floor(dt: datetime) -> datetime:
    """Truncate ``dt`` to its minute (drop seconds and microseconds)."""
    return dt.replace(second=0, microsecond=0)


def _to_utc(dt: datetime) -> datetime:
    """Coerce a (possibly naive) datetime to tz-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class RollingMinuteAggregator:
    """Aggregate 5-sec ``RealTimeBar`` updates into finalized 1-min candles.

    Stateful, single-symbol. One instance per ``BarStream``.

    Intended use:

    .. code-block:: python

        agg = RollingMinuteAggregator(symbol="AAPL", on_minute_final=cb)
        rt_bars = ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=True)
        rt_bars.updateEvent += lambda bars, has_new: agg.on_5sec_bar(bars[-1])

    The callback fires at most once per (symbol, minute) tuple.
    """

    def __init__(
        self,
        *,
        symbol: str,
        on_minute_final: MinuteFinalCallback,
    ) -> None:
        """Bind the symbol label (forensics only) and the finalization callback."""
        self._symbol = symbol
        self._on_minute_final = on_minute_final
        self._accumulator: _Accumulator | None = None
        self._finalized_minutes: set[datetime] = set()

    @property
    def in_progress_candle(self) -> MinuteCandle | None:
        """Snapshot of the currently-accumulating (un-finalized) minute candle.

        Used by ``MarketData`` to populate the trailing in-progress row of
        ``BarStream.bars`` (Phase 7.4 — strategies skip ``iloc[-1]`` so the
        in-progress row needs to exist for the slice math to land on the
        latest finalized minute).

        Returns ``None`` when no bars have been received yet, or when the
        most recent minute was just finalized and the next minute's first
        bar hasn't arrived.
        """
        if self._accumulator is None:
            return None
        return self._accumulator.to_candle()

    def on_5sec_bar(self, bar: Any) -> None:
        """Fold one 5-sec bar into the aggregator state.

        ``bar`` is duck-typed: any object exposing ``time``, ``open_``,
        ``high``, ``low``, ``close``, ``volume``, ``wap`` works. This is
        ``ib_async.objects.RealTimeBar`` in production; the test fixture
        provides a minimal stand-in.

        Returns nothing; finalization is signalled via the constructor's
        ``on_minute_final`` callback.
        """
        bar_time = _to_utc(bar.time)
        bar_minute = _minute_floor(bar_time)

        # Bootstrap on first ever bar.
        if self._accumulator is None:
            self._accumulator = _Accumulator.from_first_bar(bar, bar_minute)
            self._maybe_finalize_on_twelfth(bar)
            return

        if bar_minute == self._accumulator.minute_start:
            # Same-minute accumulate.
            self._accumulator.merge(bar)
            self._maybe_finalize_on_twelfth(bar)
            return

        if bar_minute < self._accumulator.minute_start:
            # Out-of-order older bar — drop with a forensic event.
            _log.warning(
                "bar_aggregator.discontinuity",
                symbol=self._symbol,
                bar_minute=bar_minute.isoformat(),
                accumulator_minute=self._accumulator.minute_start.isoformat(),
                reason="older_than_accumulator",
            )
            return

        # New minute arrived without a :55 bar firing the twelfth-bar
        # trigger — the :55 bar was dropped. Finalize via the gap path.
        if self._accumulator.minute_start not in self._finalized_minutes:
            self._fire_final(self._accumulator, trigger="new_minute_bar", gap=True)
        self._accumulator = _Accumulator.from_first_bar(bar, bar_minute)
        self._maybe_finalize_on_twelfth(bar)

    def _maybe_finalize_on_twelfth(self, bar: Any) -> None:
        """If ``bar`` is the 12th 5-sec bar of its minute, finalize."""
        bar_time = _to_utc(bar.time)
        if bar_time.second != _TWELFTH_BAR_SECOND:
            return
        if self._accumulator is None:
            return
        if self._accumulator.minute_start in self._finalized_minutes:
            # Idempotent: receiving a "12th" bar for a minute we've
            # already finalized via the gap path is benign — drop.
            return
        self._fire_final(self._accumulator, trigger="twelfth_bar", gap=False)

    def _fire_final(self, acc: _Accumulator, *, trigger: str, gap: bool) -> None:
        """Emit the finalized candle, mark the minute done, and clear the slot."""
        candle = acc.to_candle()
        self._finalized_minutes.add(acc.minute_start)
        finalized_at = datetime.now(UTC)
        _log.info(
            "bar_aggregator.minute_finalized",
            symbol=self._symbol,
            minute_start=candle.minute_start.isoformat(),
            trigger=trigger,
            bars_contributing=candle.bars_contributing,
            volume=candle.volume,
            close=candle.close,
            latency_ms=round(
                (finalized_at - (candle.minute_start + timedelta(minutes=1))).total_seconds()
                * 1000.0,
                1,
            ),
        )
        if gap:
            _log.warning(
                "bar_aggregator.gap_detected",
                symbol=self._symbol,
                minute_start=candle.minute_start.isoformat(),
                bars_contributing=candle.bars_contributing,
                hint="finalized via new_minute_bar trigger; the :55 5-sec bar was dropped.",
            )
        # Clear the slot: the next bar starts a fresh accumulator.
        self._accumulator = None
        self._on_minute_final(candle, finalized_at, trigger)
