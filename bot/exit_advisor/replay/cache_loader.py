"""Read-only loader for the historical bar cache.

Three operational states the harness must distinguish:
1. **Hit** — cache file exists and has bars. ``load_session_bars`` returns them.
2. **Attempted-and-missing** — a ``.unavailable`` placeholder file exists,
   meaning the fetch script ran and IBKR returned no data (delisted symbol,
   non-trading date, etc.). ``load_session_bars`` returns ``None`` and
   ``is_marked_unavailable`` returns ``True``. The harness emits a
   ``BackfillUnavailable`` warning so the operator knows the gap is real.
3. **Never attempted** — no cache file and no placeholder. The fetch script
   hasn't run for this (symbol, date). ``load_session_bars`` returns ``None``
   and ``is_marked_unavailable`` returns ``False``. The harness emits a
   ``CacheNotPopulated`` warning so the operator knows to run the script.

These three states are different operational realities — collapsing them
into a single "no data" path would hide whether a missing prior-day file
means the data really doesn't exist, or just that no one has fetched it yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .replay_source import Bar


class CacheCorruptError(RuntimeError):
    """Cache file exists but is malformed JSONL — better to fail loudly
    than silently return partial data the harness will treat as truth."""


@dataclass(frozen=True)
class HistoricalBarCache:
    cache_dir: Path

    def session_file(self, symbol: str, trading_date: date) -> Path:
        return self.cache_dir / f"{symbol}_{trading_date.isoformat()}.jsonl"

    def unavailable_marker(self, symbol: str, trading_date: date) -> Path:
        return self.cache_dir / f"{symbol}_{trading_date.isoformat()}.unavailable"

    def is_marked_unavailable(self, symbol: str, trading_date: date) -> bool:
        return self.unavailable_marker(symbol, trading_date).exists()

    def is_available(self, symbol: str, trading_date: date) -> bool:
        return self.session_file(symbol, trading_date).exists()

    def load_session_bars(self, symbol: str, trading_date: date) -> list[Bar] | None:
        """Return cached RTH bars for ``(symbol, trading_date)``.

        Returns:
        - ``list[Bar]`` (possibly empty) if the cache file exists.
        - ``None`` if no cache file exists. The caller can disambiguate
          attempted-and-missing vs never-attempted via
          :meth:`is_marked_unavailable`.
        """
        path = self.session_file(symbol, trading_date)
        if not path.exists():
            return None

        bars: list[Bar] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CacheCorruptError(
                        f"{path} line {line_num}: invalid JSON ({exc})"
                    ) from exc
                try:
                    bars.append(_bar_from_cache_entry(obj))
                except (KeyError, TypeError, ValueError) as exc:
                    raise CacheCorruptError(
                        f"{path} line {line_num}: missing or malformed field ({exc})"
                    ) from exc
        bars.sort(key=lambda b: b.timestamp)
        return bars


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _bar_from_cache_entry(obj: dict) -> Bar:  # type: ignore[type-arg]
    return Bar(
        timestamp=_parse_iso(obj["timestamp"]),
        open=float(obj["open"]),
        high=float(obj["high"]),
        low=float(obj["low"]),
        close=float(obj["close"]),
        volume=int(obj["volume"]),
    )
