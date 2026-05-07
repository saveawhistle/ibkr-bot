"""Tests for the ``watch-symbol`` and ``unwatch-symbol`` CLI commands.

Module-level store round-trip tests live in ``tests/test_manual_watchlist.py``;
scanner-side merge integration lives in ``tests/test_scanner.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from bot.cli import app
from bot.config import Settings


def _settings_with_flag(*, allow: bool) -> Settings:
    """Return a Settings with ``testing.allow_catalyst_overrides`` flipped.

    The manual watchlist reuses the catalyst-override gate (same risk
    profile: bypassing a scanner gate that must stay closed in live
    trading). Flipping this also gates the new commands.
    """
    s = Settings()
    return s.model_copy(
        update={"testing": s.testing.model_copy(update={"allow_catalyst_overrides": allow})}
    )


def test_watch_symbol_writes_file(tmp_path: Path, monkeypatch: Any) -> None:
    """Valid call with the flag on writes a well-formed JSON entry."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(
        app,
        [
            "watch-symbol",
            "ATRA",
            "--duration-hours",
            "2",
            "--note",
            "FDA agreement on tab-cel",
        ],
    )
    assert result.exit_code == 0, result.output
    assert store_path.exists()
    data = json.loads(store_path.read_text())
    assert len(data) == 1
    entry = data[0]
    assert entry["symbol"] == "ATRA"
    assert entry["note"] == "FDA agreement on tab-cel"
    assert entry["added_by"] == "cli"
    expires = datetime.fromisoformat(entry["expires_at"])
    delta = expires - datetime.now(UTC)
    assert timedelta(hours=1, minutes=55) < delta < timedelta(hours=2, minutes=5)


def test_watch_symbol_uppercases_input(tmp_path: Path, monkeypatch: Any) -> None:
    """Lowercase input on the CLI must be normalized -- IBKR symbols are uppercase."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(app, ["watch-symbol", "atra"])
    assert result.exit_code == 0, result.output
    data = json.loads(store_path.read_text())
    assert data[0]["symbol"] == "ATRA"


def test_watch_symbol_rejects_when_gate_disabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Flag off → CLI exits 1, error printed, no file created."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=False))

    result = CliRunner().invoke(app, ["watch-symbol", "ATRA"])
    assert result.exit_code == 1
    assert "disabled" in result.output.lower()
    assert "allow_catalyst_overrides" in result.output
    assert not store_path.exists()


def test_watch_symbol_replaces_duplicate(tmp_path: Path, monkeypatch: Any) -> None:
    """Re-adding the same ticker overwrites — operator extends expiry / updates note."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    runner = CliRunner()
    runner.invoke(app, ["watch-symbol", "ATRA", "--note", "first"])
    runner.invoke(app, ["watch-symbol", "ATRA", "--note", "second"])
    data = json.loads(store_path.read_text())
    assert len(data) == 1
    assert data[0]["note"] == "second"


def test_watch_symbol_rejects_negative_duration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Defensive: negative or zero duration must fail the CLI."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(
        app, ["watch-symbol", "ATRA", "--duration-hours", "-1"]
    )
    assert result.exit_code == 1
    assert "duration-hours" in result.output


def test_watch_symbol_rejects_past_expires_at(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An explicit ``--expires-at`` in the past must be rejected."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    result = CliRunner().invoke(
        app, ["watch-symbol", "ATRA", "--expires-at", past]
    )
    assert result.exit_code == 1
    assert "past" in result.output.lower()


def test_unwatch_symbol_removes_entry(tmp_path: Path, monkeypatch: Any) -> None:
    """``unwatch-symbol`` deletes the matching entry by symbol."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    runner = CliRunner()
    runner.invoke(app, ["watch-symbol", "ATRA"])
    runner.invoke(app, ["watch-symbol", "ENVB"])
    runner.invoke(app, ["unwatch-symbol", "atra"])  # case-insensitive
    data = json.loads(store_path.read_text())
    assert [e["symbol"] for e in data] == ["ENVB"]


def test_unwatch_symbol_idempotent_when_absent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Removing an absent symbol exits 0 and prints a friendly message."""
    store_path = tmp_path / "data" / "manual_watchlist.json"
    monkeypatch.setattr("bot.cli._MANUAL_WATCHLIST_DEFAULT_PATH", store_path)
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_with_flag(allow=True))

    result = CliRunner().invoke(app, ["unwatch-symbol", "NONEXIST"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()
