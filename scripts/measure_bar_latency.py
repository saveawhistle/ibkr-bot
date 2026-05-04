"""One-off latency measurement spike for the Phase 10.4 exploration.

Question: does ``reqRealTimeBars(5)`` deliver bars with materially lower
arrival latency than ``reqHistoricalData(keepUpToDate=True, "1 min")``,
or do both share the same TWS-side finalization wait that produced the
~5,250 ms delay observed on today's BIYA entry?

Subscribes three parallel feeds on a single liquid symbol (AAPL) for
~90 seconds during RTH and records per-update wall-clock latency
relative to each event's notional bar-close time:

  * ``reqHistoricalData(keepUpToDate=True, "1 min")``  — current production path
  * ``reqRealTimeBars(5, "TRADES", useRTH=True)``      — Path A candidate
  * ``reqTickByTickData("Last")``                       — already-validated
                                                          low-latency path
                                                          (Phase 7.5: 100-300 ms)

Final report prints min / median / p95 / max latency per feed plus a
bucket verdict per the decision tree from the prior conversation:

  * < 500 ms        → Path A viable, ship it
  * 500 ms – 2 s    → Real but reduced win
  * > 2 s           → Same finalization pipeline as historical bars,
                       Path A is dead, fall back to Path B (tick agg)

Run during RTH:
    uv run python scripts/measure_bar_latency.py

Defaults: AAPL, 90 second sample window, clientId=96 (separate from
the bot's clientId=17 so we don't collide if the bot is running). No
production code paths are touched; this is a scratch-style probe like
``scripts/investigate_close_only.py``.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ib_async import IB, Stock

# ---------------------------------------------------------------------------
# Config — tweak inline; this is a scratch script, not a CLI tool.
# ---------------------------------------------------------------------------

SYMBOL = "AAPL"
DURATION_SECONDS = 90
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 96  # avoid collision with main bot (17) and other probes (98, 99)


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------


@dataclass
class LatencySample:
    """One measurement: wall-clock receive time vs notional bar-close time."""

    feed: str  # "1min" | "5sec" | "tick"
    received_at: datetime  # tz-aware UTC
    notional_close_at: datetime  # tz-aware UTC
    description: str  # human-readable per-event detail

    @property
    def latency_seconds(self) -> float:
        return (self.received_at - self.notional_close_at).total_seconds()


@dataclass
class SampleSink:
    """Append-only collector partitioned by feed name."""

    by_feed: dict[str, list[LatencySample]] = field(default_factory=dict)

    def add(self, sample: LatencySample) -> None:
        self.by_feed.setdefault(sample.feed, []).append(sample)


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _to_utc(dt: datetime) -> datetime:
    """ib_async hands us datetimes that are sometimes tz-naive; coerce to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Feed handlers
# ---------------------------------------------------------------------------


def _wire_one_min_handler(bar_list: Any, sink: SampleSink) -> None:
    """Record every 1-min bar update from reqHistoricalData(keepUpToDate=True).

    Each ``updateEvent`` fires on bar-close as well as during in-progress
    updates. We record only the events where ``has_new_bar=True`` *and*
    where the previous trailing bar (now finalized) was not yet recorded.
    """
    seen_close_times: set[datetime] = set()

    def _on_update(bars: Any, has_new_bar: bool) -> None:
        if not has_new_bar or len(bars) < 2:
            return
        # The freshly-finalized bar is at bars[-2]; bars[-1] is the new
        # in-progress bar. The IBKR 1-min bar's ``date`` is the start of
        # the minute; close is start + 60 s.
        finalized = bars[-2]
        bar_start = _to_utc(finalized.date)
        if bar_start in seen_close_times:
            return
        seen_close_times.add(bar_start)
        bar_close = bar_start + timedelta(seconds=60)
        sink.add(
            LatencySample(
                feed="1min",
                received_at=_now_utc(),
                notional_close_at=bar_close,
                description=(
                    f"close={finalized.close:.2f} vol={finalized.volume:.0f} "
                    f"bar_start={bar_start.isoformat()}"
                ),
            )
        )

    bar_list.updateEvent += _on_update


def _wire_five_sec_handler(rt_bars: Any, sink: SampleSink) -> None:
    """Record every 5-sec bar pushed by reqRealTimeBars."""
    seen: set[datetime] = set()

    def _on_update(bars: Any, has_new_bar: bool) -> None:
        # ib_async appends each new RealTimeBar to the list; pull the
        # most recent and dedup on its ``time`` (start of the 5-sec
        # window).
        if not bars:
            return
        latest = bars[-1]
        bar_start = _to_utc(latest.time)
        if bar_start in seen:
            return
        seen.add(bar_start)
        bar_close = bar_start + timedelta(seconds=5)
        sink.add(
            LatencySample(
                feed="5sec",
                received_at=_now_utc(),
                notional_close_at=bar_close,
                description=(
                    f"close={latest.close:.2f} vol={latest.volume:.0f} "
                    f"bar_start={bar_start.isoformat()}"
                ),
            )
        )

    rt_bars.updateEvent += _on_update


def _wire_tick_handler(ticker: Any, sink: SampleSink) -> None:
    """Record every tick-by-tick "Last" print.

    Tick latency uses the print's own timestamp as ``notional_close_at``
    (a tick's "close time" is when the trade printed). We intentionally
    don't dedup; every tick is a fresh measurement.
    """

    def _on_update(t: Any) -> None:
        for tick in list(t.tickByTicks):
            print_time = _to_utc(tick.time)
            sink.add(
                LatencySample(
                    feed="tick",
                    received_at=_now_utc(),
                    notional_close_at=print_time,
                    description=f"price={tick.price:.2f} size={tick.size}",
                )
            )

    ticker.updateEvent += _on_update


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _percentile(values: Iterable[float], p: float) -> float:
    """Plain p-percentile via sort-and-index. Good enough for ~hundreds of samples."""
    sorted_vals = sorted(values)
    if not sorted_vals:
        return float("nan")
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _summarise(name: str, samples: list[LatencySample]) -> str:
    if not samples:
        return f"  {name:6s}  no samples"
    latencies = [s.latency_seconds for s in samples]
    n = len(latencies)
    mn = min(latencies)
    p50 = statistics.median(latencies)
    p95 = _percentile(latencies, 95)
    mx = max(latencies)
    mean = statistics.mean(latencies)
    return (
        f"  {name:6s}  n={n:4d}  "
        f"min={mn:7.3f}  p50={p50:7.3f}  mean={mean:7.3f}  "
        f"p95={p95:7.3f}  max={mx:7.3f}  (seconds)"
    )


def _verdict(samples_5sec: list[LatencySample]) -> str:
    if not samples_5sec:
        return "INDETERMINATE — no 5-sec samples collected (RTH closed? subscription issue?)"
    p50 = statistics.median(s.latency_seconds for s in samples_5sec)
    if p50 < 0.5:
        return f"PATH A VIABLE — p50={p50:.3f}s < 500 ms; ship the 5-sec aggregator."
    if p50 < 2.0:
        return (
            f"REDUCED WIN — p50={p50:.3f}s in [500ms, 2s). Path A still beats today's "
            "5.25s baseline, but the win is smaller than projected."
        )
    return (
        f"PATH A DEAD — p50={p50:.3f}s >= 2s. 5-sec bars share the historical-bar "
        "finalization pipeline. Fall back to Path B (tick aggregation)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    sink = SampleSink()
    ib = IB()

    print(f"Connecting to TWS at {TWS_HOST}:{TWS_PORT} (clientId={CLIENT_ID})...")
    await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(1)  # 1 = live
    print(
        f"Connected. Sampling {SYMBOL} for {DURATION_SECONDS} seconds across "
        "three feeds: 1-min historical, 5-sec real-time, tick-by-tick."
    )

    contract = Stock(SYMBOL, "SMART", "USD")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print(f"Failed to qualify {SYMBOL}; aborting.")
        ib.disconnect()
        return
    contract = qualified[0]

    # 1-min historical with keepUpToDate — the production baseline
    one_min_bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=True,
    )
    _wire_one_min_handler(one_min_bars, sink)

    # 5-sec real-time bars — the candidate
    five_sec_bars = ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=True)
    _wire_five_sec_handler(five_sec_bars, sink)

    # Tick-by-tick — the validated low-latency reference (Phase 7.5)
    ticker = ib.reqTickByTickData(contract, "Last", numberOfTicks=0, ignoreSize=False)
    _wire_tick_handler(ticker, sink)

    # Capture errors so a subscription-permission issue is visible.
    errors: list[str] = []

    def _on_error(req_id: int, code: int, msg: str, contract_obj: object = None) -> None:
        if code in (162, 200, 354, 10089, 10167, 10168, 10197):
            errors.append(f"  reqId={req_id} code={code}: {msg}")
        elif 2000 <= code < 3000:  # informational status
            return
        else:
            errors.append(f"  reqId={req_id} code={code}: {msg}")

    ib.errorEvent += _on_error

    # Periodic progress so a long sample window doesn't look hung.
    started_at = datetime.now()
    print()
    print("Sampling... (events stream below as they arrive)")
    print()

    last_print = started_at
    while (datetime.now() - started_at).total_seconds() < DURATION_SECONDS:
        await asyncio.sleep(1.0)
        # Print a progress line every 15 s so the operator can sanity-check
        # the feed is alive. Includes a per-feed running count.
        if (datetime.now() - last_print).total_seconds() >= 15.0:
            counts = {feed: len(samples) for feed, samples in sink.by_feed.items()}
            elapsed = (datetime.now() - started_at).total_seconds()
            print(f"  [{elapsed:5.1f}s elapsed] counts: {counts}")
            last_print = datetime.now()

    # Cancel subscriptions cleanly.
    try:
        ib.cancelHistoricalData(one_min_bars)
    except Exception as exc:  # noqa: BLE001 - probe; best-effort cancel
        print(f"  (cancelHistoricalData warned: {exc})")
    try:
        ib.cancelRealTimeBars(five_sec_bars)
    except Exception as exc:  # noqa: BLE001
        print(f"  (cancelRealTimeBars warned: {exc})")
    try:
        ib.cancelTickByTickData(contract, "Last")
    except Exception as exc:  # noqa: BLE001
        print(f"  (cancelTickByTickData warned: {exc})")

    ib.disconnect()

    # ---- Report ----
    print()
    print("=" * 78)
    print(f"LATENCY SUMMARY — {SYMBOL}, {DURATION_SECONDS}s window")
    print("=" * 78)
    print()
    print(_summarise("1min", sink.by_feed.get("1min", [])))
    print(_summarise("5sec", sink.by_feed.get("5sec", [])))
    print(_summarise("tick", sink.by_feed.get("tick", [])))
    print()
    print("Per-bar 5-sec samples (oldest first):")
    five_sec = sink.by_feed.get("5sec", [])
    for s in five_sec[:30]:  # first 30 to keep the dump readable
        print(
            f"  bar_close={s.notional_close_at.isoformat()}  "
            f"latency={s.latency_seconds:7.3f}s  {s.description}"
        )
    if len(five_sec) > 30:
        print(f"  ... ({len(five_sec) - 30} more 5-sec samples elided)")
    print()
    print("VERDICT")
    print("-" * 78)
    print(_verdict(five_sec))

    if errors:
        print()
        print("CAPTURED IB errorEvent ENTRIES")
        print("-" * 78)
        for e in errors:
            print(e)


if __name__ == "__main__":
    asyncio.run(main())
