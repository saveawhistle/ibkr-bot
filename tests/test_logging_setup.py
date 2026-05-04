"""Tests for ``bot.logging_setup`` — FileHandler wiring + idempotence."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import structlog

from bot import logging_setup
from bot.config import LoggingSettings, SessionConfig, Settings
from bot.logging_setup import configure_logging, resolve_session_log_path


@pytest.fixture(autouse=True)
def _reset_logging_flag() -> object:
    """Reset the module-level idempotence latch around every test.

    Also restores the default ``structlog`` configuration after each test.
    ``configure_logging`` sets ``cache_logger_on_first_use=True``, which
    pins the processor chain on every ``structlog.get_logger`` call and
    causes later tests' ``capture_logs`` calls to see an empty buffer —
    the cached BoundLogger no longer routes through ``capture_logs``'s
    processor swap. Resetting here isolates the side-effect to this file.
    """
    logging_setup._LOG_CONFIGURED = False
    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in prior_handlers:
        root.addHandler(handler)
    root.setLevel(prior_level)
    logging_setup._LOG_CONFIGURED = False
    structlog.reset_defaults()


def _make_settings(path: Path | None) -> Settings:
    return Settings(
        logging=LoggingSettings(level="INFO", json=True, path=path),
        session=SessionConfig(timezone="America/New_York"),
    )


def test_configure_logging_stdout_only_when_path_is_none() -> None:
    """With ``logging.path=None``, no FileHandler is attached."""
    configure_logging(_make_settings(path=None))
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers == []


def test_configure_logging_attaches_file_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``logging.path`` is set, a FileHandler writes to the expected filename.

    ``configure_logging`` suppresses the FileHandler when running under pytest
    (see ``_running_under_pytest``). To exercise the production path the test
    must clear ``PYTEST_CURRENT_TEST`` before calling.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    configure_logging(_make_settings(path=tmp_path))
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    expected = resolve_session_log_path(_make_settings(path=tmp_path))
    assert expected is not None
    assert Path(file_handlers[0].baseFilename) == expected.resolve()


def test_configure_logging_creates_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``logging.path`` that doesn't exist yet is created on configure (production path)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    target = tmp_path / "deep" / "nested" / "logs"
    assert not target.exists()
    configure_logging(_make_settings(path=target))
    assert target.exists()


def test_configure_logging_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-calling ``configure_logging`` is a no-op; no duplicate handlers."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    settings = _make_settings(path=tmp_path)
    configure_logging(settings)
    first_count = len(logging.getLogger().handlers)
    configure_logging(settings)
    second_count = len(logging.getLogger().handlers)
    assert first_count == second_count


def test_configure_logging_writes_json_lines_to_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emitted log entry lands in the session JSONL file as one JSON object."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    settings = _make_settings(path=tmp_path)
    configure_logging(settings)

    structlog.get_logger("bot.test").info("phase5_1.smoke", symbol="ACME")

    for handler in logging.getLogger().handlers:
        handler.flush()

    log_path = resolve_session_log_path(settings)
    assert log_path is not None
    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "expected at least one log line written to the file"
    parsed = [json.loads(line) for line in content]
    events = [row.get("event") for row in parsed]
    assert "phase5_1.smoke" in events


def test_configure_logging_skips_file_handler_under_pytest(tmp_path: Path) -> None:
    """Under pytest, the FileHandler is suppressed even when ``logging.path`` is set.

    Without this guard, every ``pytest`` run would append hundreds of test-fixture
    events to the operator's production session JSONL — exactly what we observed
    on 2026-05-04 with the ``_RecordingAdvisor`` / ``_RaisingAdvisor`` /
    ``_SlowAdvisor`` / ``_BareNoneAdvisor`` lines drowning real session activity.
    """
    # PYTEST_CURRENT_TEST is set by the pytest harness for the duration of this test;
    # we don't need to set it explicitly. Just sanity-check it's there and proceed.
    import os as _os

    assert _os.environ.get("PYTEST_CURRENT_TEST"), "PYTEST_CURRENT_TEST must be set under pytest"

    configure_logging(_make_settings(path=tmp_path))
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers == [], (
        "FileHandler must not attach under pytest — production session log "
        "should never see test-fixture events. Got: " + repr(file_handlers)
    )
    # Verify the file was not even created.
    expected = resolve_session_log_path(_make_settings(path=tmp_path))
    assert expected is not None
    assert not expected.exists(), (
        f"session log file {expected} was created under pytest — the gate didn't fire"
    )


def test_configure_logging_attaches_file_handler_when_pytest_var_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path (no ``PYTEST_CURRENT_TEST``) wires the FileHandler.

    Symmetric counterpart to ``test_configure_logging_skips_file_handler_under_pytest``
    so both branches of the gate are exercised explicitly.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    configure_logging(_make_settings(path=tmp_path))
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, (
        "FileHandler must attach when not running under pytest. Got: " + repr(file_handlers)
    )


def test_resolve_session_log_path_returns_none_when_path_unset() -> None:
    """``resolve_session_log_path`` with no ``logging.path`` returns None."""
    assert resolve_session_log_path(_make_settings(path=None)) is None


def test_resolve_session_log_path_uses_ny_date(tmp_path: Path) -> None:
    """Filename uses ``session_{YYYY-MM-DD}.jsonl`` in the session timezone."""
    settings = _make_settings(path=tmp_path)
    path = resolve_session_log_path(settings)
    assert path is not None
    assert path.name.startswith("session_")
    assert path.name.endswith(".jsonl")
