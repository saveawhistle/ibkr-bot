"""Standalone fetcher for historical bar cache (1-min, RTH).

Run after a trade closes (or in batch over the full closed-trade history)
to populate the cache that ``replay_source.load_trade_replay_data`` reads.

Usage:
    python scripts/fetch_historical_bars.py \\
        --symbol ZENA --date 2026-04-30
    python scripts/fetch_historical_bars.py --all-trades

Idempotency: a (symbol, date) is skipped when its session cache file
already exists OR the ``.unavailable`` placeholder is present. Atomic
file writes (tmp → rename) keep partial fetches from corrupting the cache.

Pacing: IBKR throttles historical requests aggressively. Default 10s
between requests is conservative but safe; tune via ``--pacing-seconds``.
A two-day fetch (same-day + prior-day) per trade waits ``pacing_seconds``
between the two requests; cross-trade fetches do the same.

Holidays: a small hardcoded calendar (US market holidays for 2025-2026)
covers the dates we care about. Walk back beyond a holiday to find the
prior trading day. When the calendar is exhausted, IBKR will return no
data and we write the ``.unavailable`` marker.

This script does NOT touch any production bot state. It only writes
files into ``data/historical_bars/`` (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot.exit_advisor.core.timeutil import rth_close_utc, rth_open_utc
from bot.exit_advisor.replay.replay_source import DEFAULT_CACHE_DIR, _iter_structured_events

if TYPE_CHECKING:
    from bot.brokerage.ibkr_client import IBKRClient

# Share the writer's CACHE_DIR with the reader's DEFAULT_CACHE_DIR so they
# can never drift apart again (prior spike-merge reorg let them diverge —
# the fetcher wrote to ``cache/historical_bars/`` while the reader looked
# at ``data/historical_bars/``).
CACHE_DIR = DEFAULT_CACHE_DIR
SESSION_LOGS_DIR = Path("logs")

# US market holidays for the dates we care about. Extend as needed; an
# unrecognised holiday just means IBKR will return no data and the script
# writes the .unavailable marker — graceful degradation, not a crash.
US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    [
        # 2025
        date(2025, 1, 1),  # New Year's
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    ]
)


def prior_trading_day(d: date) -> date:
    """Walk back from ``d`` until we hit a weekday that isn't a holiday."""
    candidate = d - timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in US_MARKET_HOLIDAYS:
        candidate -= timedelta(days=1)
    return candidate


@dataclass
class FetchTarget:
    symbol: str
    trading_date: date


@dataclass
class FetchSummary:
    attempted: int = 0
    succeeded: int = 0
    skipped_existing: int = 0
    marked_unavailable: int = 0
    failed: int = 0


def _session_cache_path(symbol: str, trading_date: date) -> Path:
    return CACHE_DIR / f"{symbol}_{trading_date.isoformat()}.jsonl"


def _unavailable_marker(symbol: str, trading_date: date) -> Path:
    return CACHE_DIR / f"{symbol}_{trading_date.isoformat()}.unavailable"


def _is_already_fetched(symbol: str, trading_date: date) -> bool:
    return (
        _session_cache_path(symbol, trading_date).exists()
        or _unavailable_marker(symbol, trading_date).exists()
    )


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write ``rows`` as JSONL via tmp file + rename — partial fetches
    can't leave a half-written .jsonl that the loader would then read."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")))
            fh.write("\n")
    tmp.replace(path)


def _atomic_write_marker(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)


def _bar_to_cache_row(bar: Any, symbol: str, trading_date: date) -> dict[str, Any]:
    """Map an ib_async historical BarData object to our cache row shape."""
    ts = bar.date
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        ts_iso = ts.astimezone(UTC).isoformat().replace("+00:00", "Z")
    else:
        ts_iso = str(ts)
    return {
        "timestamp": ts_iso,
        "symbol": symbol,
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": int(bar.volume),
        "source": "reqHistoricalData",
        "bar_size": "1 min",
        "trading_date": trading_date.isoformat(),
    }


async def _fetch_one(
    client: IBKRClient,
    symbol: str,
    trading_date: date,
    log: logging.Logger,
) -> tuple[str, int]:
    """Fetch one (symbol, trading_date). Returns (status, n_bars).

    status ∈ {"hit", "unavailable", "error"}.
    """
    if _is_already_fetched(symbol, trading_date):
        return "hit", 0
    if trading_date.weekday() >= 5 or trading_date in US_MARKET_HOLIDAYS:
        log.info("non-trading date, skipping: %s %s", symbol, trading_date)
        return "hit", 0

    end_dt = rth_close_utc(trading_date)
    duration_sec = int((end_dt - rth_open_utc(trading_date)).total_seconds())
    # ib_async expects endDateTime as a UTC string with " UTC" suffix when tz-aware.
    end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

    # Both qualify_stock AND reqHistoricalDataAsync go inside the try block.
    # Originally only the historical fetch was wrapped, but a symbol that no
    # longer qualifies on IBKR (delisted, ticker changed) raises ValueError
    # from qualify_stock and would crash the whole batch — surfaced when
    # SBLX failed to qualify during the layer 4.5 pipeline run on 2026-05-02.
    # Treat qualify failures the same as "no data" so the operator gets a
    # ``.unavailable`` marker and the batch continues.
    try:
        contract = await client.qualify_stock(symbol)
    except Exception as exc:  # noqa: BLE001 — we want every failure logged
        _atomic_write_marker(
            _unavailable_marker(symbol, trading_date),
            {
                "symbol": symbol,
                "trading_date": trading_date.isoformat(),
                "fetched_at": datetime.now(UTC).isoformat(),
                "reason": f"qualify_stock failed: {exc}",
            },
        )
        log.info(
            "could not qualify %s for %s — marker written (%s)",
            symbol,
            trading_date,
            exc,
        )
        return "unavailable", 0

    try:
        bars = await client._ib.reqHistoricalDataAsync(  # noqa: SLF001 — script-level access OK
            contract,
            endDateTime=end_str,
            durationStr=f"{duration_sec} S",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,  # UTC seconds-since-epoch (ib_async exposes datetime objects)
        )
    except Exception as exc:  # noqa: BLE001 — we want every failure logged
        log.error("fetch failed for %s %s: %s", symbol, trading_date, exc)
        return "error", 0

    if not bars:
        _atomic_write_marker(
            _unavailable_marker(symbol, trading_date),
            {
                "symbol": symbol,
                "trading_date": trading_date.isoformat(),
                "fetched_at": datetime.now(UTC).isoformat(),
                "reason": "IBKR returned no bars",
            },
        )
        log.info("no data for %s %s — marker written", symbol, trading_date)
        return "unavailable", 0

    rows = [_bar_to_cache_row(b, symbol, trading_date) for b in bars]
    _atomic_write_jsonl(_session_cache_path(symbol, trading_date), rows)
    log.info("wrote %d bars for %s %s", len(rows), symbol, trading_date)
    return "hit", len(rows)


DEFAULT_MANIFEST = Path("reports/exit_advisor/closed_trades_manifest.jsonl")


def _enumerate_closed_trades(logs_dir: Path) -> list[FetchTarget]:
    """Walk every session log under ``logs_dir`` and yield one FetchTarget
    per closed trade discovered. Fallback for ``--all-trades`` when no
    manifest has been produced; in normal operation the manifest is
    canonical (run ``discover_closed_trades`` first)."""
    targets: list[FetchTarget] = []
    for log_path in sorted(logs_dir.glob("session_*.jsonl")):
        date_str = log_path.stem.removeprefix("session_")
        try:
            trading_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        for evt in _iter_structured_events(log_path):
            if evt.get("event") == "position.closed" and evt.get("symbol"):
                targets.append(FetchTarget(symbol=evt["symbol"], trading_date=trading_date))
    # Dedupe (a trade that re-enters the same symbol shows up twice).
    seen: set[tuple[str, date]] = set()
    deduped: list[FetchTarget] = []
    for t in targets:
        key = (t.symbol, t.trading_date)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


def _enumerate_from_manifest(manifest_path: Path) -> list[FetchTarget]:
    """Read the closed-trades manifest produced by
    ``bot.exit_advisor.scripts.discover_closed_trades`` and project
    to dedupe'd (symbol, date) FetchTargets."""
    from bot.exit_advisor.replay.trade_discovery import read_manifest

    refs = read_manifest(manifest_path)
    seen: set[tuple[str, date]] = set()
    targets: list[FetchTarget] = []
    for ref in refs:
        key = (ref.symbol, ref.trade_date)
        if key in seen:
            continue
        seen.add(key)
        targets.append(FetchTarget(symbol=ref.symbol, trading_date=ref.trade_date))
    return targets


def trading_dates_to_fetch(trade_date: date, prior_days: int) -> list[date]:
    """Return ``[trade_date, prior_day_1, prior_day_2, ..., prior_day_N]``,
    walking back through trading days (skipping weekends + holidays via
    ``prior_trading_day``). Stable sort so the fetch loop is
    deterministic + the trade-date file is fetched first."""
    out = [trade_date]
    cursor = trade_date
    for _ in range(prior_days):
        cursor = prior_trading_day(cursor)
        out.append(cursor)
    return out


async def _run(targets: list[FetchTarget], pacing_seconds: float, prior_days: int) -> FetchSummary:
    from bot.brokerage.ibkr_client import IBKRClient

    log = logging.getLogger("fetch_historical_bars")
    summary = FetchSummary()

    client = IBKRClient()
    await client.connect()
    try:
        first = True
        for target in targets:
            for trading_date in trading_dates_to_fetch(target.trading_date, prior_days):
                summary.attempted += 1
                if _is_already_fetched(target.symbol, trading_date):
                    summary.skipped_existing += 1
                    continue
                if not first:
                    await asyncio.sleep(pacing_seconds)
                first = False
                status, _ = await _fetch_one(client, target.symbol, trading_date, log)
                if status == "hit":
                    summary.succeeded += 1
                elif status == "unavailable":
                    summary.marked_unavailable += 1
                else:
                    summary.failed += 1
    finally:
        await client.disconnect()

    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--symbol", help="Single-symbol mode")
    parser.add_argument("--date", help="Trading date (YYYY-MM-DD) for --symbol mode")
    parser.add_argument(
        "--all-trades", action="store_true", help="Batch mode: every closed trade in logs/"
    )
    parser.add_argument("--pacing-seconds", type=float, default=10.0)
    parser.add_argument(
        "--prior-days",
        type=int,
        default=1,
        help=(
            "How many prior trading days to fetch alongside the trade date. "
            "Layer 2.5 default was 1 (just the prior day). Layer L2-A bumped "
            "to support up to 10 for the RVOL milestone detector. The fetcher "
            "skips already-cached files, so increasing this is safe."
        ),
    )
    parser.add_argument("--logs-dir", default=str(SESSION_LOGS_DIR))
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Manifest used by --all-trades. Falls back to scanning logs/ if missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.all_trades and not (args.symbol and args.date):
        print("Specify either --all-trades or both --symbol and --date.", file=sys.stderr)
        return 2

    if args.all_trades:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            targets = _enumerate_from_manifest(manifest_path)
            print(f"Loaded {len(targets)} trade(s) from manifest {manifest_path}")
        else:
            targets = _enumerate_closed_trades(Path(args.logs_dir))
            print(
                f"No manifest at {manifest_path}; fell back to scanning "
                f"{args.logs_dir} (found {len(targets)})"
            )
    else:
        try:
            trading_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"Bad --date: {args.date}", file=sys.stderr)
            return 2
        targets = [FetchTarget(symbol=args.symbol, trading_date=trading_date)]

    if not targets:
        print("No targets found.", file=sys.stderr)
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(_run(targets, args.pacing_seconds, args.prior_days))
    print(
        f"attempts={summary.attempted} ok={summary.succeeded} "
        f"skipped={summary.skipped_existing} unavailable={summary.marked_unavailable} "
        f"failed={summary.failed}"
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
