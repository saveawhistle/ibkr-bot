"""End-to-end latency comparison: Path A (5-sec → 1-min) vs Path B (tick → 1-min).

Builds a real 1-minute candle from each source feed, then runs a
representative "would-submit-the-order" pipeline (strategy-style risk
calc + LimitOrder/StopOrder construction with the Phase 10.3
``apply_default_tif`` helper). Records per-minute wall-clock from
"minute X boundary passed" to "ready to call ib.placeOrder()" for
both paths. The actual ``placeOrder`` call is *not* made — this is a
simulation, not a real entry.

What each path declares as "minute X final":

  * **Path A** (5-sec ``reqRealTimeBars`` → 1-min aggregator):
    when the bar with ``time.second == 55`` (the 12th bar of minute X)
    arrives, OR when any bar from minute X+1 arrives (gap fallback for
    a dropped 12th bar — same idea as Phase 9.4's diff-driven detection).
  * **Path B** (``reqTickByTickData("Last")`` → 1-min aggregator):
    when the first tick whose ``time`` falls in minute X+1 arrives.
    (No wall-clock grace fallback in this spike — AAPL prints
    continuously so the trigger always fires within hundreds of ms.
    Thin low-float symbols would need a fallback in the production
    aggregator; that's documented in the prior exploration report.)

After "minute final" fires, both paths run identical simulated work
(toy risk calc + bracket order construction with TIF=DAY) and stamp
``submit_ready_at``. The delta between paths at ``submit_ready_at``
is the actionable answer to "how much faster is B than A end-to-end".

Run during RTH:
    uv run python scripts/measure_aggregator_to_submit.py

Defaults: AAPL, 240 second window, clientId=97 (avoiding 17/96/98/99).
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ib_async import IB, LimitOrder, Stock, StopOrder

# Reuse the production helper so the spike's "would-submit" path matches
# what the executor actually does post-Phase-10.3.
from bot.execution.executor import apply_default_tif

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "AAPL"
DURATION_SECONDS = 480  # ~8 minute boundaries
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 97


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


@dataclass
class _Candle:
    """In-progress 1-minute OHLCV candle."""

    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_bar(cls, bar: Any) -> _Candle:
        return cls(
            open=float(bar.open_),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )

    @classmethod
    def from_tick(cls, price: float, size: float) -> _Candle:
        return cls(open=price, high=price, low=price, close=price, volume=size)

    def merge_bar(self, bar: Any) -> None:
        self.high = max(self.high, float(bar.high))
        self.low = min(self.low, float(bar.low))
        self.close = float(bar.close)
        self.volume += float(bar.volume)

    def merge_tick(self, price: float, size: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size


# Callback signature: (minute_start_utc, candle, finalized_at_wall_clock, trigger_label)
MinuteFinalCallback = Callable[[datetime, _Candle, datetime, str], None]


def _minute_floor(dt: datetime) -> datetime:
    """Truncate a datetime to its minute (drop seconds/microseconds)."""
    return dt.replace(second=0, microsecond=0)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class PathAAggregator:
    """Path A — 5-sec ``reqRealTimeBars`` rolled into 1-min candles.

    Two finalization triggers:
      1. **Twelfth-bar trigger**: a bar whose ``time.second == 55`` is
         the last 5-sec bar of its minute. Receipt of this bar means the
         minute is done — fire immediately.
      2. **New-minute trigger** (gap fallback): a bar whose minute
         differs from the current accumulating minute. Phase 9.4 noted
         IBKR sometimes drops bars; if the :55 bar never arrives we'd
         otherwise stall waiting for it. Falling back when the next
         minute's first bar shows up keeps things moving.
    """

    def __init__(self, on_minute_final: MinuteFinalCallback) -> None:
        self._current_minute: datetime | None = None
        self._candle: _Candle | None = None
        self._on_minute_final = on_minute_final
        self._finalized_minutes: set[datetime] = set()

    def on_5sec_bar(self, bar: Any) -> None:
        bar_time = _to_utc(bar.time)
        bar_minute = _minute_floor(bar_time)

        if self._current_minute is None:
            self._current_minute = bar_minute
            self._candle = _Candle.from_bar(bar)
            self._maybe_finalize_on_twelfth(bar)
            return

        if bar_minute == self._current_minute:
            assert self._candle is not None
            self._candle.merge_bar(bar)
            self._maybe_finalize_on_twelfth(bar)
            return

        # New minute arrived without us finalizing the prior — gap path.
        # Fire the prior as final via the new-minute trigger.
        if self._current_minute not in self._finalized_minutes:
            assert self._candle is not None
            self._fire_final(self._current_minute, self._candle, trigger="new_minute_bar")
        # Start accumulating into the new minute.
        self._current_minute = bar_minute
        self._candle = _Candle.from_bar(bar)
        self._maybe_finalize_on_twelfth(bar)

    def _maybe_finalize_on_twelfth(self, bar: Any) -> None:
        """If this bar is the 12th of the minute (time.second == 55), finalize."""
        bar_time = _to_utc(bar.time)
        if bar_time.second != 55:
            return
        if self._current_minute is None or self._candle is None:
            return
        if self._current_minute in self._finalized_minutes:
            return
        self._fire_final(self._current_minute, self._candle, trigger="twelfth_bar")

    def _fire_final(self, minute: datetime, candle: _Candle, *, trigger: str) -> None:
        self._finalized_minutes.add(minute)
        self._on_minute_final(minute, candle, datetime.now(UTC), trigger)


class PathBAggregator:
    """Path B — tick-by-tick rolled into 1-min candles.

    Single finalization trigger: the first tick whose ``time`` is in
    minute X+1 means minute X is done. No wall-clock fallback in this
    spike — AAPL prints continuously and the trigger always fires
    within hundreds of ms. A production aggregator targeting thin
    names should add a wall-clock backstop.
    """

    def __init__(self, on_minute_final: MinuteFinalCallback) -> None:
        self._current_minute: datetime | None = None
        self._candle: _Candle | None = None
        self._on_minute_final = on_minute_final
        self._finalized_minutes: set[datetime] = set()

    def on_tick(self, price: float, size: float, tick_time: datetime) -> None:
        tick_minute = _minute_floor(_to_utc(tick_time))

        if self._current_minute is None:
            self._current_minute = tick_minute
            self._candle = _Candle.from_tick(price, size)
            return

        if tick_minute == self._current_minute:
            assert self._candle is not None
            self._candle.merge_tick(price, size)
            return

        # First tick of a new minute → prior minute is final.
        if self._current_minute not in self._finalized_minutes:
            assert self._candle is not None
            self._finalized_minutes.add(self._current_minute)
            self._on_minute_final(
                self._current_minute,
                self._candle,
                datetime.now(UTC),
                "next_minute_tick",
            )
        # Start the new minute's candle.
        self._current_minute = tick_minute
        self._candle = _Candle.from_tick(price, size)


# ---------------------------------------------------------------------------
# Simulated submission work
# ---------------------------------------------------------------------------


def simulate_submit_work(candle: _Candle, contract: Any) -> datetime:
    """Run the post-finalization in-process work and return wall-clock when "ready".

    Mirrors what the production hot path does between candle-final and
    ``ib.placeOrder``: a toy strategy/risk computation, then bracket
    order construction with the Phase 10.3 ``apply_default_tif`` helper.
    Does *not* actually call ``placeOrder`` — that's a TWS round-trip
    we deliberately exclude so the comparison isolates the aggregator
    latency. In production both paths share the same post-construction
    placement cost, so it cancels out of the delta.
    """
    # Toy "strategy" — risk per share and a sized position.
    entry = candle.close
    stop = candle.low if candle.low < entry else entry - 0.05
    risk = max(entry - stop, 0.01)
    shares = max(int(24.0 / risk), 1)  # toy: $24 per-trade max loss

    # Build the bracket exactly as _place_bracket does (sans placeOrder).
    parent = LimitOrder("BUY", shares, round(entry + 0.10, 2))
    parent.transmit = False
    parent.outsideRth = False
    apply_default_tif(parent)

    stop_order = StopOrder("SELL", shares, round(stop, 2))
    stop_order.transmit = True
    stop_order.outsideRth = False
    apply_default_tif(stop_order)

    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Per-minute observation registry
# ---------------------------------------------------------------------------


@dataclass
class PathObservation:
    """One path's view of a minute: when it finalized and when submit was ready."""

    finalized_at: datetime
    submit_ready_at: datetime
    trigger: str
    candle: _Candle


@dataclass
class MinuteRecord:
    """Paired A/B observations for one minute."""

    minute_start: datetime
    a: PathObservation | None = None
    b: PathObservation | None = None


@dataclass
class Registry:
    """Collect minute records keyed by minute_start; print a side-by-side report."""

    records: dict[datetime, MinuteRecord] = field(default_factory=dict)

    def record(
        self,
        path: str,
        minute_start: datetime,
        candle: _Candle,
        finalized_at: datetime,
        trigger: str,
        contract: Any,
    ) -> None:
        submit_ready_at = simulate_submit_work(candle, contract)
        observation = PathObservation(
            finalized_at=finalized_at,
            submit_ready_at=submit_ready_at,
            trigger=trigger,
            candle=candle,
        )
        rec = self.records.setdefault(minute_start, MinuteRecord(minute_start=minute_start))
        if path == "A":
            rec.a = observation
        else:
            rec.b = observation


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _ms_since(boundary: datetime, t: datetime) -> float:
    """Milliseconds between ``boundary`` and ``t`` (positive when t is after)."""
    return (t - boundary).total_seconds() * 1000.0


def _report(registry: Registry) -> None:
    paired = [r for r in registry.records.values() if r.a is not None and r.b is not None]
    paired.sort(key=lambda r: r.minute_start)

    print()
    print("=" * 92)
    print(
        f"PER-MINUTE COMPARISON — {SYMBOL} — {len(paired)} minutes paired (A and B both observed)"
    )
    print("=" * 92)
    print(
        f"{'minute (UTC)':<22} {'A_final_ms':>10} {'A_submit_ms':>11} "
        f"{'B_final_ms':>10} {'B_submit_ms':>11} {'delta_submit_A-B':>17} {'A trigger':<18}"
    )
    print("-" * 92)
    for r in paired:
        minute_close = r.minute_start + timedelta(minutes=1)
        assert r.a is not None
        assert r.b is not None
        print(
            f"{r.minute_start.strftime('%H:%M:%S'):<22} "
            f"{_ms_since(minute_close, r.a.finalized_at):>10.1f} "
            f"{_ms_since(minute_close, r.a.submit_ready_at):>11.1f} "
            f"{_ms_since(minute_close, r.b.finalized_at):>10.1f} "
            f"{_ms_since(minute_close, r.b.submit_ready_at):>11.1f} "
            f"{(r.a.submit_ready_at - r.b.submit_ready_at).total_seconds() * 1000.0:>13.1f} "
            f"{r.a.trigger:<18}"
        )

    if not paired:
        print("(no paired minutes — try running for longer)")
        return

    a_submit_ms = [
        _ms_since(r.minute_start + timedelta(minutes=1), r.a.submit_ready_at)
        for r in paired
        if r.a is not None
    ]
    b_submit_ms = [
        _ms_since(r.minute_start + timedelta(minutes=1), r.b.submit_ready_at)
        for r in paired
        if r.b is not None
    ]
    deltas_ms = [
        (r.a.submit_ready_at - r.b.submit_ready_at).total_seconds() * 1000.0
        for r in paired
        if r.a is not None and r.b is not None
    ]

    print()
    print("SUMMARY (milliseconds from minute boundary to ready-to-placeOrder)")
    print("-" * 92)
    print(
        f"  Path A submit_ready_ms: "
        f"min={min(a_submit_ms):7.1f}  median={statistics.median(a_submit_ms):7.1f}  "
        f"max={max(a_submit_ms):7.1f}  mean={statistics.mean(a_submit_ms):7.1f}"
    )
    print(
        f"  Path B submit_ready_ms: "
        f"min={min(b_submit_ms):7.1f}  median={statistics.median(b_submit_ms):7.1f}  "
        f"max={max(b_submit_ms):7.1f}  mean={statistics.mean(b_submit_ms):7.1f}"
    )
    print(
        f"  delta(A - B) submit_ready_ms (positive = B is faster): "
        f"min={min(deltas_ms):7.1f}  median={statistics.median(deltas_ms):7.1f}  "
        f"max={max(deltas_ms):7.1f}  mean={statistics.mean(deltas_ms):7.1f}"
    )
    print()
    median_delta = statistics.median(deltas_ms)
    if median_delta > 200:
        print(
            f"VERDICT: Path B beats Path A by ~{median_delta:.0f} ms median — "
            "meaningful win, ship Path B."
        )
    elif median_delta > 50:
        print(
            f"VERDICT: Path B beats Path A by ~{median_delta:.0f} ms median — "
            "small but real; weigh against Path B's higher dev complexity."
        )
    elif median_delta > -50:
        print(
            "VERDICT: Paths are within ~50 ms of each other — "
            "Path A's lower complexity wins on the cost/benefit."
        )
    else:
        print(
            f"VERDICT: Path A actually faster by ~{-median_delta:.0f} ms median — "
            "unexpected; investigate before any commitment."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    registry = Registry()

    print(f"Connecting to TWS at {TWS_HOST}:{TWS_PORT} (clientId={CLIENT_ID})...")
    ib = IB()
    await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(1)
    print(
        f"Connected. Building 1-min candles for {SYMBOL} via Path A (5-sec) "
        f"AND Path B (ticks) for {DURATION_SECONDS} seconds."
    )

    contract = Stock(SYMBOL, "SMART", "USD")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print(f"Failed to qualify {SYMBOL}; aborting.")
        ib.disconnect()
        return
    contract = qualified[0]

    # Path A
    def _path_a_final(
        minute: datetime, candle: _Candle, finalized_at: datetime, trigger: str
    ) -> None:
        registry.record("A", minute, candle, finalized_at, trigger, contract)

    path_a = PathAAggregator(on_minute_final=_path_a_final)
    rt_bars = ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=True)

    def _on_5sec_update(bars: Any, has_new_bar: bool) -> None:
        if not bars:
            return
        path_a.on_5sec_bar(bars[-1])

    rt_bars.updateEvent += _on_5sec_update

    # Path B
    def _path_b_final(
        minute: datetime, candle: _Candle, finalized_at: datetime, trigger: str
    ) -> None:
        registry.record("B", minute, candle, finalized_at, trigger, contract)

    path_b = PathBAggregator(on_minute_final=_path_b_final)
    ticker = ib.reqTickByTickData(contract, "Last", numberOfTicks=0, ignoreSize=False)

    def _on_tick_update(t: Any) -> None:
        for tick in list(t.tickByTicks):
            path_b.on_tick(float(tick.price), float(tick.size), _to_utc(tick.time))

    ticker.updateEvent += _on_tick_update

    # Capture errors so a permission/subscription issue doesn't silently skew results.
    errors: list[str] = []

    def _on_error(req_id: int, code: int, msg: str, contract_obj: object = None) -> None:
        if 2000 <= code < 3000:
            return
        errors.append(f"  reqId={req_id} code={code}: {msg}")

    ib.errorEvent += _on_error

    started = datetime.now()
    last_print = started
    print()
    print("Sampling... (progress every 30s)")
    print()
    while (datetime.now() - started).total_seconds() < DURATION_SECONDS:
        await asyncio.sleep(1.0)
        if (datetime.now() - last_print).total_seconds() >= 30.0:
            elapsed = (datetime.now() - started).total_seconds()
            paired = sum(
                1 for r in registry.records.values() if r.a is not None and r.b is not None
            )
            print(
                f"  [{elapsed:5.1f}s elapsed] minutes_paired={paired} "
                f"a_recorded={sum(1 for r in registry.records.values() if r.a is not None)} "
                f"b_recorded={sum(1 for r in registry.records.values() if r.b is not None)}"
            )
            last_print = datetime.now()

    # Cleanup
    try:
        ib.cancelRealTimeBars(rt_bars)
    except Exception as exc:  # noqa: BLE001
        print(f"  (cancelRealTimeBars warned: {exc})")
    try:
        ib.cancelTickByTickData(contract, "Last")
    except Exception as exc:  # noqa: BLE001
        print(f"  (cancelTickByTickData warned: {exc})")
    ib.disconnect()

    _report(registry)

    if errors:
        print()
        print("CAPTURED IB errorEvent ENTRIES")
        print("-" * 78)
        for e in errors:
            print(e)


if __name__ == "__main__":
    asyncio.run(main())
