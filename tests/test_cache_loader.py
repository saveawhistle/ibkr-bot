"""HistoricalBarCache tests — three operational states distinguished
correctly, malformed JSONL surfaces a clear error rather than silently
returning partial data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from bot.exit_advisor.replay.cache_loader import CacheCorruptError, HistoricalBarCache


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_cache_hit_returns_bars_chronologically(tmp_path: Path) -> None:
    cache = HistoricalBarCache(cache_dir=tmp_path)
    rows = [
        {
            "timestamp": "2026-04-30T13:31:00Z",
            "symbol": "TST",
            "open": 1.05,
            "high": 1.10,
            "low": 1.00,
            "close": 1.08,
            "volume": 200,
        },
        {
            "timestamp": "2026-04-30T13:30:00Z",
            "symbol": "TST",
            "open": 1.00,
            "high": 1.05,
            "low": 0.95,
            "close": 1.05,
            "volume": 100,
        },
    ]
    _write_jsonl(cache.session_file("TST", date(2026, 4, 30)), rows)
    bars = cache.load_session_bars("TST", date(2026, 4, 30))
    assert bars is not None
    assert len(bars) == 2
    assert bars[0].timestamp < bars[1].timestamp


def test_cache_miss_no_placeholder_returns_none(tmp_path: Path) -> None:
    cache = HistoricalBarCache(cache_dir=tmp_path)
    assert cache.load_session_bars("TST", date(2026, 4, 30)) is None
    assert not cache.is_marked_unavailable("TST", date(2026, 4, 30))
    assert not cache.is_available("TST", date(2026, 4, 30))


def test_cache_miss_with_placeholder_distinguishable(tmp_path: Path) -> None:
    """Three states must be distinguishable: hit, attempted-and-missing,
    never-attempted. The placeholder is the marker for the middle case."""
    cache = HistoricalBarCache(cache_dir=tmp_path)
    cache.unavailable_marker("TST", date(2026, 4, 30)).write_text("{}", encoding="utf-8")
    assert cache.load_session_bars("TST", date(2026, 4, 30)) is None
    assert cache.is_marked_unavailable("TST", date(2026, 4, 30))


def test_corrupt_jsonl_raises_clear_error(tmp_path: Path) -> None:
    cache = HistoricalBarCache(cache_dir=tmp_path)
    path = cache.session_file("TST", date(2026, 4, 30))
    path.write_text(
        '{"timestamp":"2026-04-30T13:30:00Z","symbol":"TST","open":1,'
        '"high":1.1,"low":0.9,"close":1,"volume":1}\nNOT_JSON\n',
        encoding="utf-8",
    )
    with pytest.raises(CacheCorruptError, match="line 2"):
        cache.load_session_bars("TST", date(2026, 4, 30))


def test_jsonl_missing_field_raises(tmp_path: Path) -> None:
    cache = HistoricalBarCache(cache_dir=tmp_path)
    path = cache.session_file("TST", date(2026, 4, 30))
    # Missing "volume" — better to fail loudly than silently default to zero.
    path.write_text(
        '{"timestamp":"2026-04-30T13:30:00Z","symbol":"TST","open":1,'
        '"high":1.1,"low":0.9,"close":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(CacheCorruptError, match="line 1"):
        cache.load_session_bars("TST", date(2026, 4, 30))


def test_empty_file_returns_empty_list_not_none(tmp_path: Path) -> None:
    """A file that exists but contains no bars is the empty-result hit
    case (e.g. an early-day fetch before any bars existed). It must
    NOT collapse to the never-attempted case."""
    cache = HistoricalBarCache(cache_dir=tmp_path)
    cache.session_file("TST", date(2026, 4, 30)).write_text("", encoding="utf-8")
    bars = cache.load_session_bars("TST", date(2026, 4, 30))
    assert bars == []
