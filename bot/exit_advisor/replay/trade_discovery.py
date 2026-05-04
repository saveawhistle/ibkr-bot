"""Discover closed trades by scanning JSONL session logs.

A closed trade is a ``position.opened`` event followed by a
``position.closed`` (or equivalent terminal) event for the same
symbol within the same session log. Multiple trades on the same
symbol within one session each get their own ``ClosedTradeRef``,
distinguished by entry timestamp.

Filtered out (with a WARNING log line each, but no exception):
- Trades opened but never closed (truncated logs, manual aborts)
- Trades whose recorded strategy field marks them as test artifacts
  (``test``, ``manual_test``) — the production bot's force-entry CLI
  generates these and they don't belong in an analytical dataset
- Trades whose symbol is the operator-test placeholder ``TEST``,
  even if the strategy field happens to be missing or set to a real
  strategy (defensive: layer 1's session logs include legacy TEST
  entries from before the strategy field was reliably populated)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Symbols and strategies that mark synthetic/test trades. Production bot
# data uses ``momentum`` and ``gap_and_go``; anything containing the
# substring "test" is operator-injected for harness validation.
TEST_SYMBOLS: frozenset[str] = frozenset({"TEST", "AAA", "BBB"})
TEST_STRATEGY_TOKENS: tuple[str, ...] = ("test", "manual_test")


@dataclass(frozen=True)
class ClosedTradeRef:
    """One closed trade. ``trade_id`` is the entry-timestamp ISO string,
    which uniquely identifies a trade within the dataset (no two trades
    open at the exact same instant on the same symbol)."""

    symbol: str
    trade_date: date
    trade_id: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    session_log_path: Path


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _is_test_strategy(strategy: str | None) -> bool:
    if strategy is None:
        return False
    s = strategy.lower()
    return any(token in s for token in TEST_STRATEGY_TOKENS)


def _iter_structured_events(path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Yield every JSON-parseable event line (with an ``event`` key)
    from a session log. Mirrors the parser in ``replay_source`` so this
    module can stand alone."""
    out: list[dict] = []  # type: ignore[type-arg]
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


def _trade_date_from_log_path(path: Path) -> date | None:
    """Extract YYYY-MM-DD from filenames like ``session_2026-04-30.jsonl``."""
    stem = path.stem
    if not stem.startswith("session_"):
        return None
    try:
        return date.fromisoformat(stem.removeprefix("session_"))
    except ValueError:
        return None


def discover_closed_trades(session_logs_dir: Path) -> list[ClosedTradeRef]:
    """Scan all JSONL session logs in the directory, return closed
    trades sorted by entry_timestamp.

    Edge cases handled:
    - position.opened with no matching position.closed → WARNING, excluded
    - Multiple trades on same symbol → each returned separately
    - Test trades (synthetic strategy or symbol) → excluded silently
      (these are noise from the bot's test injection harness, not
      real trades worth analysing)
    """
    refs: list[ClosedTradeRef] = []
    for log_path in sorted(session_logs_dir.glob("session_*.jsonl")):
        trade_date = _trade_date_from_log_path(log_path)
        if trade_date is None:
            continue
        refs.extend(_discover_in_log(log_path, trade_date))
    refs.sort(key=lambda r: r.entry_timestamp)
    return refs


def _discover_in_log(log_path: Path, trade_date: date) -> list[ClosedTradeRef]:
    """Walk a single session log and emit one ClosedTradeRef per
    matched (opened, closed) pair on the same symbol.

    Multiple trades on the same symbol are matched in FIFO order:
    the first ``position.opened`` for symbol X pairs with the next
    ``position.closed`` for symbol X, the second opened pairs with
    the second closed, etc. Truncated logs (opened with no close)
    are detected and emit a WARNING.
    """
    events = _iter_structured_events(log_path)
    open_queue: dict[str, list[dict]] = {}  # type: ignore[type-arg]
    out: list[ClosedTradeRef] = []
    for evt in events:
        ev_name = evt.get("event")
        symbol = evt.get("symbol")
        if not symbol or symbol in TEST_SYMBOLS:
            continue
        if ev_name == "position.opened":
            if _is_test_strategy(evt.get("strategy")):
                continue
            open_queue.setdefault(symbol, []).append(evt)
        elif ev_name == "position.closed":
            queue = open_queue.get(symbol)
            if not queue:
                # Closed without an opened — out-of-order log, skip silently.
                continue
            opened = queue.pop(0)
            entry_ts = _parse_iso(opened["timestamp"])
            exit_ts = _parse_iso(evt["timestamp"])
            out.append(
                ClosedTradeRef(
                    symbol=symbol,
                    trade_date=trade_date,
                    trade_id=entry_ts.isoformat(),
                    entry_timestamp=entry_ts,
                    exit_timestamp=exit_ts,
                    session_log_path=log_path,
                )
            )

    # Anything left in open_queue is a trade that opened but never closed.
    for symbol, queue in open_queue.items():
        for opened in queue:
            log.warning(
                "trade opened but never closed in %s: %s @ %s — excluded",
                log_path.name,
                symbol,
                opened.get("timestamp"),
            )
    return out


def write_manifest(refs: list[ClosedTradeRef], manifest_path: Path) -> None:
    """Write a JSONL manifest. One trade per line; the schema mirrors
    ``ClosedTradeRef``'s fields. Re-runnable: overwrites previous manifest.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for ref in refs:
            obj = {
                "symbol": ref.symbol,
                "trade_date": ref.trade_date.isoformat(),
                "trade_id": ref.trade_id,
                "entry_timestamp": ref.entry_timestamp.isoformat(),
                "exit_timestamp": ref.exit_timestamp.isoformat(),
                # POSIX form so the manifest is cross-platform — readers
                # convert via Path() which accepts forward slashes on Windows.
                "session_log_path": ref.session_log_path.as_posix(),
            }
            fh.write(json.dumps(obj) + "\n")


def read_manifest(manifest_path: Path) -> list[ClosedTradeRef]:
    """Inverse of :func:`write_manifest`. Used by the batch-mode
    fetch and comparison scripts."""
    refs: list[ClosedTradeRef] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            refs.append(
                ClosedTradeRef(
                    symbol=obj["symbol"],
                    trade_date=date.fromisoformat(obj["trade_date"]),
                    trade_id=obj["trade_id"],
                    entry_timestamp=_parse_iso(obj["entry_timestamp"]),
                    exit_timestamp=_parse_iso(obj["exit_timestamp"]),
                    session_log_path=Path(obj["session_log_path"]),
                )
            )
    return refs
