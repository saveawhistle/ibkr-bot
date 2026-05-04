"""Tests for Phase 6.8 manual catalyst override — CLI + store round-trip.

Scanner-side override application lives in ``tests/test_scanner.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from bot.cli import app
from bot.config import Settings
from bot.scanning.catalyst_overrides import (
    CatalystOverride,
    find_active_override,
    load_overrides,
    upsert_override,
)


def _settings_with_flag(*, allow: bool) -> Settings:
    """Return a Settings with ``testing.allow_catalyst_overrides`` flipped."""
    s = Settings()
    return s.model_copy(
        update={"testing": s.testing.model_copy(update={"allow_catalyst_overrides": allow})}
    )


def test_inject_catalyst_writes_file(tmp_path: Path, monkeypatch: Any) -> None:
    """Valid injection with the flag on writes a well-formed JSON entry."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(
        app,
        [
            "inject-catalyst",
            "AGPU",
            "--category",
            "contract_or_m&a",
            "--duration-hours",
            "2",
            "--note",
            "Axe Compute contract",
        ],
    )
    assert result.exit_code == 0, result.output
    assert store_path.exists(), "injection must create the store file"

    data = json.loads(store_path.read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["symbol"] == "AGPU"
    assert entry["category"] == "contract_or_m&a"
    assert entry["note"] == "Axe Compute contract"
    assert entry["injected_by"] == "cli"
    # Expires ~2 hours from now; allow 5-min slack for test jitter.
    expires = datetime.fromisoformat(entry["expires_at"])
    delta = expires - datetime.now(UTC)
    assert timedelta(hours=1, minutes=55) < delta < timedelta(hours=2, minutes=5)


def test_inject_catalyst_rejects_when_disabled(tmp_path: Path, monkeypatch: Any) -> None:
    """Flag off → CLI exits 1, error printed, no file created."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=False))

    result = CliRunner().invoke(
        app,
        ["inject-catalyst", "AGPU", "--category", "contract_or_m&a"],
    )
    assert result.exit_code == 1
    # This CliRunner version merges stdout + stderr into ``output``; the
    # error was emitted via ``typer.echo(..., err=True)`` — so it lands
    # in ``output`` regardless of stream.
    assert "disabled" in result.output.lower()
    assert "allow_catalyst_overrides" in result.output
    assert not store_path.exists(), "failed injection must not create the store file"


def test_inject_catalyst_replaces_duplicate_symbol(tmp_path: Path, monkeypatch: Any) -> None:
    """Injecting AGPU twice replaces the first entry; the file holds one row with latest values."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    runner = CliRunner()
    first = runner.invoke(
        app,
        ["inject-catalyst", "AGPU", "--category", "contract_or_m&a", "--note", "first"],
    )
    assert first.exit_code == 0, first.output
    second = runner.invoke(
        app,
        ["inject-catalyst", "AGPU", "--category", "clinical", "--note", "second"],
    )
    assert second.exit_code == 0, second.output

    data = json.loads(store_path.read_text())
    assert len(data) == 1, "duplicate symbol must replace, not append"
    assert data[0]["category"] == "clinical"
    assert data[0]["note"] == "second"


def test_invalid_category_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    """An unknown category must be refused with a clear error; no file written."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(
        app,
        ["inject-catalyst", "AGPU", "--category", "bogus_category"],
    )
    assert result.exit_code == 1
    assert "bogus_category" in result.output
    assert not store_path.exists()


# ---------- Store-level unit tests ---------- #


def test_store_upsert_and_find_roundtrip(tmp_path: Path) -> None:
    """Write one override via upsert and read it back via find_active_override."""
    path = tmp_path / "overrides.json"
    now = datetime.now(UTC)
    override = CatalystOverride(
        symbol="AGPU",
        category="contract_or_m&a",
        expires_at=now + timedelta(hours=2),
        note="round-trip",
        injected_at=now,
        injected_by="cli",
    )
    upsert_override(override, path)

    found = find_active_override("AGPU", now=now, path=path)
    assert found is not None
    assert found.category == "contract_or_m&a"
    assert found.note == "round-trip"

    # Different symbol → None.
    assert find_active_override("MISS", now=now, path=path) is None


def test_store_expired_entries_ignored(tmp_path: Path) -> None:
    """Entries whose ``expires_at`` is past the provided ``now`` are skipped by finders."""
    path = tmp_path / "overrides.json"
    base = datetime.now(UTC)
    fresh = CatalystOverride(
        symbol="FRESH",
        category="clinical",
        expires_at=base + timedelta(hours=1),
        note=None,
        injected_at=base,
        injected_by="cli",
    )
    stale = CatalystOverride(
        symbol="STALE",
        category="clinical",
        expires_at=base - timedelta(hours=1),
        note=None,
        injected_at=base - timedelta(hours=2),
        injected_by="cli",
    )
    upsert_override(fresh, path)
    upsert_override(stale, path)

    # Both on disk, only the fresh one is "active".
    assert len(load_overrides(path)) == 2
    assert find_active_override("FRESH", now=base, path=path) is not None
    assert find_active_override("STALE", now=base, path=path) is None


def test_store_load_returns_empty_when_file_absent(tmp_path: Path) -> None:
    """No file → no overrides. Must not raise."""
    path = tmp_path / "nonexistent.json"
    assert load_overrides(path) == []
    assert find_active_override("ANY", now=datetime.now(UTC), path=path) is None


def test_store_load_survives_malformed_file(tmp_path: Path) -> None:
    """A hand-edited file with bad JSON must log a warning and yield []."""
    path = tmp_path / "overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not: valid json", encoding="utf-8")
    # No raise.
    assert load_overrides(path) == []


def test_inject_catalyst_logs_manual_override_injected(tmp_path: Path, monkeypatch: Any) -> None:
    """Successful injection emits ``catalyst.manual_override_injected`` with the key fields."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    # Reach into bot.cli's logger and substitute a capture; simpler than
    # structlog.testing.capture_logs because the CLI uses the module-level
    # ``_log`` bound at import time.
    events: list[dict[str, Any]] = []

    def fake_info(event: str, **fields: Any) -> None:
        events.append({"event": event, **fields})

    with patch("bot.cli._log.info", side_effect=fake_info):
        result = CliRunner().invoke(
            app,
            ["inject-catalyst", "AGPU", "--category", "clinical", "--note", "Phase III readout"],
        )
    assert result.exit_code == 0, result.output
    matches = [e for e in events if e["event"] == "catalyst.manual_override_injected"]
    assert len(matches) == 1
    evt = matches[0]
    assert evt["symbol"] == "AGPU"
    assert evt["category"] == "clinical"
    assert evt["note"] == "Phase III readout"
    assert evt["injected_by"] == "cli"


@pytest.mark.parametrize(
    ("symbol_in", "expected"),
    [("agpu", "AGPU"), ("AgPU", "AGPU"), ("AGPU", "AGPU")],
)
def test_symbol_normalised_to_upper(
    tmp_path: Path, monkeypatch: Any, symbol_in: str, expected: str
) -> None:
    """The CLI upper-cases symbols so scanner lookups match regardless of input case."""
    store_path = tmp_path / "data" / "test_catalyst_overrides.json"
    monkeypatch.setattr("bot.cli._OVERRIDES_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(app, ["inject-catalyst", symbol_in, "--category", "clinical"])
    assert result.exit_code == 0, result.output
    data = json.loads(store_path.read_text())
    assert data[0]["symbol"] == expected
