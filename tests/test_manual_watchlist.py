"""Tests for ``bot.scanning.manual_watchlist`` -- load/save/upsert/remove + expiry filter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bot.scanning.manual_watchlist import (
    ManualWatchlistEntry,
    load_active_entries,
    load_entries,
    remove_entry,
    save_entries,
    upsert_entry,
)


def _entry(
    symbol: str = "ATRA",
    *,
    expires_in_hours: float = 6.0,
    note: str | None = "test",
    added_by: str = "cli",
) -> ManualWatchlistEntry:
    now = datetime.now(UTC)
    return ManualWatchlistEntry(
        symbol=symbol,
        expires_at=now + timedelta(hours=expires_in_hours),
        note=note,
        added_at=now,
        added_by=added_by,
    )


def test_load_entries_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_entries(tmp_path / "absent.json") == []


def test_load_entries_malformed_json_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json {{ ", encoding="utf-8")
    assert load_entries(p) == []


def test_load_entries_non_list_root_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "wrong_shape.json"
    p.write_text(json.dumps({"symbol": "ATRA"}), encoding="utf-8")
    assert load_entries(p) == []


def test_load_entries_skips_malformed_individual_entries(tmp_path: Path) -> None:
    """One bad entry doesn't poison the rest of the file."""
    p = tmp_path / "mixed.json"
    p.write_text(
        json.dumps(
            [
                _entry("ATRA").to_dict(),
                {"symbol": "MISSING_FIELDS"},  # missing expires_at/added_at
                _entry("ENVB").to_dict(),
            ]
        ),
        encoding="utf-8",
    )
    symbols = [e.symbol for e in load_entries(p)]
    assert symbols == ["ATRA", "ENVB"]


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "round.json"
    entries = [_entry("ATRA"), _entry("ENVB", note=None)]
    save_entries(entries, p)
    loaded = load_entries(p)
    assert [e.symbol for e in loaded] == ["ATRA", "ENVB"]
    assert loaded[1].note is None


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    p = tmp_path / "deeply" / "nested" / "watchlist.json"
    save_entries([_entry("ATRA")], p)
    assert p.exists()


def test_upsert_replaces_existing_symbol(tmp_path: Path) -> None:
    p = tmp_path / "upsert.json"
    save_entries([_entry("ATRA", note="first")], p)
    second = _entry("ATRA", note="second")
    upsert_entry(second, p)
    loaded = load_entries(p)
    assert len(loaded) == 1
    assert loaded[0].note == "second"


def test_upsert_appends_new_symbol(tmp_path: Path) -> None:
    p = tmp_path / "append.json"
    save_entries([_entry("ATRA")], p)
    upsert_entry(_entry("ENVB"), p)
    symbols = sorted(e.symbol for e in load_entries(p))
    assert symbols == ["ATRA", "ENVB"]


def test_remove_returns_false_when_symbol_absent(tmp_path: Path) -> None:
    p = tmp_path / "rem.json"
    save_entries([_entry("ATRA")], p)
    assert remove_entry("NOPE", p) is False
    assert [e.symbol for e in load_entries(p)] == ["ATRA"]


def test_remove_returns_true_and_persists_when_symbol_present(tmp_path: Path) -> None:
    p = tmp_path / "rem.json"
    save_entries([_entry("ATRA"), _entry("ENVB")], p)
    assert remove_entry("ATRA", p) is True
    assert [e.symbol for e in load_entries(p)] == ["ENVB"]


def test_load_active_entries_filters_expired(tmp_path: Path) -> None:
    p = tmp_path / "active.json"
    now = datetime.now(UTC)
    save_entries(
        [
            _entry("ATRA", expires_in_hours=1),  # active
            _entry("STALE", expires_in_hours=-1),  # expired 1h ago
            _entry("ENVB", expires_in_hours=24),  # active
        ],
        p,
    )
    active = load_active_entries(now=now, path=p)
    symbols = sorted(e.symbol for e in active)
    assert symbols == ["ATRA", "ENVB"]


def test_is_active_strict_less_than_at_boundary() -> None:
    """``now == expires_at`` counts as expired (matches catalyst_overrides semantics)."""
    expires = datetime(2026, 5, 7, 16, 0, tzinfo=UTC)
    entry = ManualWatchlistEntry(
        symbol="ATRA",
        expires_at=expires,
        note=None,
        added_at=expires - timedelta(hours=1),
        added_by="cli",
    )
    assert entry.is_active(expires - timedelta(seconds=1))
    assert not entry.is_active(expires)
    assert not entry.is_active(expires + timedelta(seconds=1))


def test_to_dict_round_trips_through_from_dict() -> None:
    original = _entry("ATRA", note="FDA agreement on tab-cel")
    parsed = ManualWatchlistEntry.from_dict(original.to_dict())
    assert parsed == original
