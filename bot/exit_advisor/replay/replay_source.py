"""Reconstruct a closed trade's bar/order/exit timeline from a JSONL session log.

Layer 1 reads only the structured events (those with an ``event`` field).
The raw ib_async ``orderStatus: Trade(...)`` repr lines that also appear in
the session log are ignored — they are debug dumps, not parseable JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from bot.exit_advisor.core.timeutil import rth_open_utc

# Cache lives at the repo-top ``data/historical_bars/`` (consolidated from
# the spike's old ``spike/exit_advisor/cache/historical_bars/`` location
# during the Phase 11 spike-merge). ``data/`` is gitignored; operators
# populate the directory via ``scripts/fetch_historical_bars.py``.
# ``__file__`` is ``bot/exit_advisor/replay_source.py``; three ``.parent``
# steps reach the repo root.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical_bars"


@dataclass(frozen=True)
class Bar:
    """Minimal 1-minute bar shape used by the harness.

    Defined locally rather than imported from ``bot.brokerage.market_data`` to keep
    the spike package decoupled from production code. If/when graduation
    happens, the production Bar replaces this.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TradeReplayData:
    """All inputs the harness needs to replay a closed trade end-to-end."""

    symbol: str
    trade_date: date
    bars: list[Bar]
    entry_event: dict[str, Any]
    bracket_event: dict[str, Any]
    order_events: list[dict[str, Any]] = field(default_factory=list)
    exit_event: dict[str, Any] = field(default_factory=dict)
    recorded_pnl: float = 0.0
    recorded_exit_price: float = 0.0
    recorded_exit_timestamp: datetime = field(default_factory=lambda: datetime.min)
    fill_event: dict[str, Any] = field(default_factory=dict)
    """``position.filled`` event for the entry parent — carries actual fill price."""

    protection_anchored_event: dict[str, Any] = field(default_factory=dict)
    """``executor.protection_fill_anchored`` — children working, position protected."""

    pre_trade_bars: list[Bar] = field(default_factory=list)
    """RTH bars for the same symbol from session open through the bar
    immediately before the entry bar. Sourced from a merge of (a) the
    bot's live JSONL session log (post-subscription bars) and (b) the
    historical bar cache (pre-subscription bars filled retrospectively).
    See ``pre_trade_bar_sources`` for per-bar attribution.
    """

    pre_trade_bar_sources: dict[datetime, str] = field(default_factory=dict)
    """Per-bar origin: ``"session_log"`` for bars the bot received live,
    ``"historical_cache"`` for bars filled in retrospectively. Lets
    forensic analysis distinguish between live-observed and
    retrospectively-fetched data when a result looks suspect."""

    prior_day_bars: list[Bar] = field(default_factory=list)
    """RTH bars for the same symbol on the prior trading day. Sourced
    from the historical bar cache when populated, else empty.
    Detectors that depend on prior-day data must degrade gracefully."""

    prior_day_cache_state: Literal["hit", "marked_unavailable", "not_populated"] = "not_populated"
    """Three operational states for the prior-day cache (see
    :class:`cache_loader.HistoricalBarCache`). The harness can emit
    different warnings based on which state applied."""

    prior_day_session_high: float | None = None
    prior_day_session_low: float | None = None
    prior_day_session_close: float | None = None


def _parse_iso(ts: str) -> datetime:
    """Parse ISO-8601 timestamps with optional ``Z`` UTC suffix."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _iter_structured_events(path: Path) -> list[dict[str, Any]]:
    """Yield every JSON-parseable event line from the session log.

    Session logs interleave structured JSON events (everything emitted via
    structlog) with raw ib_async repr lines and HTTP debug output. Skip
    anything that isn't valid JSON with an ``event`` key.
    """
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "event" in obj:
                out.append(obj)
    return out


def session_log_path(trade_date: date, logs_dir: Path | str = "logs") -> Path:
    return Path(logs_dir) / f"session_{trade_date.isoformat()}.jsonl"


def _bar_from_event(e: dict[str, Any]) -> Bar:
    return Bar(
        timestamp=_parse_iso(e["bar_time"]),
        open=float(e["open"]),
        high=float(e["high"]),
        low=float(e["low"]),
        close=float(e["close"]),
        volume=int(e["volume"]),
    )


def _load_symbol_bars(path: Path, symbol: str, *, rth_only: bool = True) -> list[Bar]:
    """Load every ``market_data.bar_received`` entry for ``symbol`` from
    a session log, filtered to RTH if requested.

    Returns bars sorted by ``bar_time``. RTH filter compares the bar's
    NY-local date+time to 09:30/16:00 ET; bars outside that window
    (premarket, after-hours) are dropped when ``rth_only=True``.
    """
    if not path.exists():
        return []
    out: list[Bar] = []
    for raw in _iter_structured_events(path):
        if raw.get("event") != "market_data.bar_received" or raw.get("symbol") != symbol:
            continue
        bar = _bar_from_event(raw)
        if rth_only:
            ny_date = bar.timestamp.date()
            session_start = rth_open_utc(ny_date)
            session_end = session_start + timedelta(hours=6, minutes=30)
            bar_utc = bar.timestamp.astimezone(session_start.tzinfo)
            if bar_utc < session_start or bar_utc >= session_end:
                continue
        out.append(bar)
    out.sort(key=lambda b: b.timestamp)
    return out


def _prior_trading_day(d: date) -> date:
    """Calendar prior day, skipping weekends. No holiday calendar — the
    spike degrades gracefully (no prior-day file → empty bars) so a
    holiday landing on the calendar prior day is correctly handled by
    the missing-file fallback."""
    candidate = d - timedelta(days=1)
    while candidate.weekday() >= 5:  # Saturday=5, Sunday=6
        candidate -= timedelta(days=1)
    return candidate


def load_trade_replay_data(
    symbol: str,
    trade_date: date,
    logs_dir: Path | str = "logs",
    cache_dir: Path | str | None = None,
) -> TradeReplayData:
    """Reconstruct one closed trade's lifecycle from a session log.

    Strategy: scan the entire day's events, filter to ``symbol``, then
    locate the first ``executor.*_bracket_placed`` and matching
    ``position.closed``. Bars between those two timestamps form the
    replay window.

    Raises ValueError if the trade cannot be located (no bracket, no
    close, or mismatched parent_order_id between the two).
    """
    path = session_log_path(trade_date, logs_dir)
    if not path.exists():
        raise FileNotFoundError(f"Session log not found: {path}")
    events = _iter_structured_events(path)

    sym_events = [e for e in events if e.get("symbol") == symbol]

    bracket_event: dict[str, Any] | None = None
    for e in sym_events:
        if e.get("event") in ("executor.lmt_bracket_placed", "executor.mkt_bracket_placed"):
            bracket_event = e
            break
    if bracket_event is None:
        raise ValueError(f"No bracket placement event found for {symbol} on {trade_date}")

    parent_id = bracket_event.get("parent_order_id")
    bracket_ts = _parse_iso(bracket_event["timestamp"])

    entry_event: dict[str, Any] | None = None
    fill_event: dict[str, Any] | None = None
    protection_anchored: dict[str, Any] | None = None
    exit_event: dict[str, Any] | None = None
    order_events: list[dict[str, Any]] = []

    for e in sym_events:
        ts_raw = e.get("timestamp")
        if not ts_raw:
            continue
        ts = _parse_iso(ts_raw)
        if ts < bracket_ts:
            continue
        ev_name = e.get("event")
        if (
            ev_name == "position.opened"
            and entry_event is None
            and (e.get("parent_order_id") == parent_id or parent_id is None)
        ):
            entry_event = e
        elif ev_name == "position.filled" and fill_event is None:
            fill_event = e
        elif ev_name == "executor.protection_fill_anchored" and protection_anchored is None:
            protection_anchored = e
        elif ev_name == "position.closed":
            exit_event = e
            break
        elif ev_name in ("order.placed", "executor.entry_expired"):
            order_events.append(e)

    if entry_event is None:
        raise ValueError(f"No position.opened event found for {symbol} on {trade_date}")
    if exit_event is None:
        raise ValueError(f"No position.closed event found for {symbol} on {trade_date}")

    exit_ts = _parse_iso(exit_event["timestamp"])

    # All RTH bars for the symbol on the trade day (chronological).
    all_day_bars = _load_symbol_bars(path, symbol, rth_only=True)

    # Trade-window bars: bar emission timestamp between bracket placement
    # and recorded exit. Layer 1 used the bar's emission timestamp here,
    # not bar_time — preserved for backwards compatibility, since the bar
    # at bar_time = entry's start-of-minute closes after bracket and
    # should belong to the trade window.
    trade_bars: list[Bar] = []
    pre_trade_bars_session: list[Bar] = []
    bracket_bar_minute = bracket_ts.replace(second=0, microsecond=0)
    for bar in all_day_bars:
        bar_close = (bar.timestamp + timedelta(minutes=1)).astimezone(exit_ts.tzinfo)
        if bar_close <= bracket_bar_minute:
            pre_trade_bars_session.append(bar)
        elif bar_close <= exit_ts:
            # Trade-window bars are those whose CLOSE happened on or before
            # the recorded exit. A bar that started before the exit but
            # closed after isn't in the window — its close-time data wasn't
            # available to the bot when the position was already closed.
            trade_bars.append(bar)

    # Layer 2.5: merge cache-sourced bars to fill the pre-subscription gap
    # between session open (09:30 ET) and the bot's first live bar.
    cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    pre_trade_bars, pre_trade_bar_sources = _merge_pre_trade_bars(
        symbol=symbol,
        trade_date=trade_date,
        bracket_bar_minute=bracket_bar_minute,
        session_log_bars=pre_trade_bars_session,
        cache_dir=cache_path,
    )

    # Layer 2.5: prior-day data comes exclusively from the cache. The
    # production bot wasn't subscribed to the symbol on prior days (it
    # only adds symbols once the catalyst classifier flags them mid-day),
    # so the prior session log won't have bars for this symbol — that's
    # what the cache was built to fill in.
    prior_bars, prior_state = _load_prior_day_from_cache(
        symbol=symbol, trade_date=trade_date, cache_dir=cache_path
    )
    prior_high: float | None = None
    prior_low: float | None = None
    prior_close: float | None = None
    if prior_bars:
        prior_high = max(b.high for b in prior_bars)
        prior_low = min(b.low for b in prior_bars)
        prior_close = prior_bars[-1].close

    return TradeReplayData(
        symbol=symbol,
        trade_date=trade_date,
        bars=trade_bars,
        entry_event=entry_event,
        bracket_event=bracket_event,
        order_events=order_events,
        exit_event=exit_event,
        recorded_pnl=float(exit_event["pnl"]),
        recorded_exit_price=float(exit_event["exit_price"]),
        recorded_exit_timestamp=exit_ts,
        fill_event=fill_event or {},
        protection_anchored_event=protection_anchored or {},
        pre_trade_bars=pre_trade_bars,
        pre_trade_bar_sources=pre_trade_bar_sources,
        prior_day_bars=prior_bars,
        prior_day_cache_state=prior_state,
        prior_day_session_high=prior_high,
        prior_day_session_low=prior_low,
        prior_day_session_close=prior_close,
    )


def _merge_pre_trade_bars(
    *,
    symbol: str,
    trade_date: date,
    bracket_bar_minute: datetime,
    session_log_bars: list[Bar],
    cache_dir: Path,
) -> tuple[list[Bar], dict[datetime, str]]:
    """Combine session-log pre-trade bars with cache bars covering the
    pre-subscription gap. Live bars (session_log) take precedence over
    cache bars at the same timestamp — the bot's actual feed is the
    source of truth for any bar it received.
    """
    # Late-binding import — cache_loader imports Bar from this module.
    from .cache_loader import HistoricalBarCache

    cache = HistoricalBarCache(cache_dir=cache_dir)
    cache_bars = cache.load_session_bars(symbol, trade_date) or []

    # Only keep cache bars that close before the trade entry's bracket
    # minute — anything from there onward belongs to the live trade-window
    # feed, which is sourced from the session log.
    cache_pre_trade = [
        b
        for b in cache_bars
        if (b.timestamp + timedelta(minutes=1)).astimezone(bracket_bar_minute.tzinfo)
        <= bracket_bar_minute
    ]

    by_ts: dict[datetime, tuple[Bar, str]] = {}
    for bar in cache_pre_trade:
        by_ts[bar.timestamp] = (bar, "historical_cache")
    for bar in session_log_bars:
        by_ts[bar.timestamp] = (bar, "session_log")  # session log wins

    sorted_pairs = sorted(by_ts.items(), key=lambda kv: kv[0])
    merged_bars = [pair[1][0] for pair in sorted_pairs]
    sources = {pair[0]: pair[1][1] for pair in sorted_pairs}
    return merged_bars, sources


def _load_prior_day_from_cache(
    *, symbol: str, trade_date: date, cache_dir: Path
) -> tuple[list[Bar], Literal["hit", "marked_unavailable", "not_populated"]]:
    """Load prior-day RTH bars from cache. Returns the bars and the
    cache state — the harness uses the state to emit a precise warning
    (real gap vs. operator hasn't run the fetch script)."""
    from .cache_loader import HistoricalBarCache

    cache = HistoricalBarCache(cache_dir=cache_dir)
    prior_day = _prior_trading_day(trade_date)

    bars = cache.load_session_bars(symbol, prior_day)
    if bars is None:
        if cache.is_marked_unavailable(symbol, prior_day):
            return [], "marked_unavailable"
        return [], "not_populated"
    return bars, "hit"


def load_prior_n_day_volume_curve(
    *,
    symbol: str,
    trade_date: date,
    n_days: int,
    cache_dir: Path | None = None,
) -> tuple[dict[int, float], int]:
    """Build a minute-of-RTH-day → average cumulative volume curve from
    the cached prior N trading days. Returns ``(curve, days_used)`` —
    ``days_used`` may be less than ``n_days`` if some days are missing
    (delisting, holidays the calendar didn't catch, .unavailable
    markers). The harness uses this for the RVOL milestone detector.

    Walks back through trading days via :func:`_prior_trading_day`,
    skipping any day whose cache file is absent or marked unavailable.
    For each day with bars, computes the per-minute cumulative volume
    curve (minute 0 = RTH open). Averages across all available days.
    """
    from .cache_loader import HistoricalBarCache

    cache = HistoricalBarCache(cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR)
    daily_curves: list[dict[int, int]] = []
    cursor = _prior_trading_day(trade_date)
    days_attempted = 0
    while days_attempted < n_days:
        bars = cache.load_session_bars(symbol, cursor)
        days_attempted += 1
        if bars:
            curve = _build_cumulative_volume_curve(bars, cursor)
            if curve:
                daily_curves.append(curve)
        cursor = _prior_trading_day(cursor)

    if not daily_curves:
        return {}, 0

    averaged: dict[int, float] = {}
    all_minutes: set[int] = set()
    for curve in daily_curves:
        all_minutes.update(curve.keys())
    for minute in all_minutes:
        values = [curve.get(minute, 0) for curve in daily_curves]
        averaged[minute] = sum(values) / len(daily_curves)
    return averaged, len(daily_curves)


def _build_cumulative_volume_curve(bars: list[Bar], trade_date: date) -> dict[int, int]:
    """For one day's bars, return ``minute_offset → cumulative_volume_so_far``."""
    from bot.exit_advisor.core.timeutil import rth_open_utc

    open_ts = rth_open_utc(trade_date)
    cumulative = 0
    out: dict[int, int] = {}
    for bar in sorted(bars, key=lambda b: b.timestamp):
        bar_ts_utc = bar.timestamp.astimezone(open_ts.tzinfo)
        minute = int((bar_ts_utc - open_ts).total_seconds() // 60)
        if minute < 0:
            continue
        cumulative += bar.volume
        out[minute] = cumulative
    return out
