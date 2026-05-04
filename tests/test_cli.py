"""Tests for ``bot.cli`` Phase 4b surfaces: ``--live`` CONFIRM + ``reset-halt`` + status."""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as date_cls
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click.exceptions
import pytest
from typer.testing import CliRunner

from bot.cli import _confirm_live_or_exit, _install_shutdown_handler, app
from bot.config import AccountConfig, RiskConfig, Settings
from bot.risk import HaltRecord, write_halt_flag


def _settings() -> Settings:
    """Settings fixture defaulting to paper."""
    base = Settings()
    return base.model_copy(update={"account": AccountConfig(mode="paper"), "risk": RiskConfig()})


def test_confirm_live_requires_literal_confirm_word() -> None:
    """``--live`` prompt must accept only the literal 'CONFIRM'; anything else exits."""
    settings = _settings()
    # Lower-case / substring → must be rejected.
    with (
        patch("bot.cli.typer.prompt", return_value="confirm"),
        pytest.raises(click.exceptions.Exit) as exc_info,
    ):
        _confirm_live_or_exit(settings)
    assert exc_info.value.exit_code == 1


def test_confirm_live_flips_mode_when_confirmed() -> None:
    """Literal ``CONFIRM`` returns Settings with ``account.mode='live'``."""
    settings = _settings()
    with patch("bot.cli.typer.prompt", return_value="CONFIRM"):
        result = _confirm_live_or_exit(settings)
    assert result.account.mode == "live"


def test_reset_halt_clears_flag_with_yes(tmp_path: Path, monkeypatch: Any) -> None:
    """``reset-halt --yes`` deletes the flag and prints the cleared message."""
    flag_path = tmp_path / "halt.flag"
    write_halt_flag(
        flag_path,
        HaltRecord(
            date=date_cls.today(),
            reason="daily_loss_limit",
            triggered_at=datetime.now(UTC),
            pnl_at_halt=-320.0,
        ),
    )
    monkeypatch.setattr("bot.cli._halt_flag_path", lambda: flag_path)
    runner = CliRunner()
    result = runner.invoke(app, ["reset-halt", "--yes"])
    assert result.exit_code == 0, result.output
    assert not flag_path.exists()
    assert "Halt flag cleared" in result.output


def test_reset_halt_noop_when_no_flag(tmp_path: Path, monkeypatch: Any) -> None:
    """No flag on disk → ``reset-halt`` exits 0 with a 'No halt flag present' message."""
    flag_path = tmp_path / "halt.flag"
    monkeypatch.setattr("bot.cli._halt_flag_path", lambda: flag_path)
    runner = CliRunner()
    result = runner.invoke(app, ["reset-halt", "--yes"])
    assert result.exit_code == 0, result.output
    assert "No halt flag present" in result.output


def test_reset_halt_requires_confirmation_without_yes(tmp_path: Path, monkeypatch: Any) -> None:
    """Without ``--yes`` the command prompts; declining exits 1 and keeps the flag."""
    flag_path = tmp_path / "halt.flag"
    write_halt_flag(
        flag_path,
        HaltRecord(
            date=date_cls.today(),
            reason="giveback_limit",
            triggered_at=datetime.now(UTC),
            pnl_at_halt=100.0,
        ),
    )
    monkeypatch.setattr("bot.cli._halt_flag_path", lambda: flag_path)
    runner = CliRunner()
    # Answer 'n' to typer.confirm.
    result = runner.invoke(app, ["reset-halt"], input="n\n")
    assert result.exit_code == 1
    assert flag_path.exists()


def test_rehab_status_on_fresh_journal_reports_normal(tmp_path: Path, monkeypatch: Any) -> None:
    """Empty journal + no persisted flag → tier is NORMAL and exit is clean.

    Chdir so the default ``logs/trades.db`` + ``logs/rehab.flag`` paths
    land in ``tmp_path`` — keeps the test hermetic without monkeypatching
    every filesystem helper in ``bot.risk.rehab``.
    """
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["rehab-status"])
    assert result.exit_code == 0, result.output
    assert "NORMAL" in result.output


@pytest.mark.asyncio
async def test_install_shutdown_handler_sets_event_via_registered_handler() -> None:
    """The installed handler must set the asyncio.Event when invoked.

    Avoids actually raising SIGINT (which on Windows translates to
    KeyboardInterrupt in the test thread and breaks pytest). Instead we
    fetch whichever path got installed — ``loop._signal_handlers`` on
    POSIX or ``signal.getsignal`` on Windows — and invoke the callback
    directly, then tick the event loop so ``call_soon_threadsafe`` lands.
    """
    import asyncio
    import signal as signal_mod
    import sys

    event = asyncio.Event()
    uninstall = _install_shutdown_handler(event)
    try:
        if sys.platform == "win32":
            handler = signal_mod.getsignal(signal_mod.SIGINT)
            assert callable(handler)
            handler(signal_mod.SIGINT, None)
            # call_soon_threadsafe scheduled the set; let the loop pick it up.
            await asyncio.wait_for(event.wait(), timeout=1.0)
        else:
            # On POSIX the asyncio loop owns the handler; we can exercise the
            # same callback by invoking the internal dispatch path.
            loop = asyncio.get_running_loop()
            handlers = getattr(loop, "_signal_handlers", {})
            sig_handle = handlers.get(signal_mod.SIGINT)
            assert sig_handle is not None, "SIGINT handler not registered on loop"
            sig_handle._run()
            await asyncio.wait_for(event.wait(), timeout=1.0)
    finally:
        uninstall()
    assert event.is_set()


# ---------- Phase 6.13: force-entry paper-testing CLI ---------- #


def _settings_for_force_entry(*, allow: bool, mode: str = "paper") -> Settings:
    """Settings with ``testing.allow_force_entry`` + ``account.mode`` overrides."""
    s = Settings()
    return s.model_copy(
        update={
            "account": AccountConfig(mode=mode),  # type: ignore[arg-type]
            "testing": s.testing.model_copy(update={"allow_force_entry": allow}),
        }
    )


def test_force_entry_rejects_when_disabled(monkeypatch: Any) -> None:
    """Flag off → CLI exits 1 with a clear error; no IBKR connection attempted."""
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_for_force_entry(allow=False))
    # Guard against accidental connection: if the CLI gets past the gate,
    # the run-wrapper would try to connect. Swap it out to detect that.
    called = {"ran": False}

    def _fail_if_called(_coro_factory: Any) -> None:
        called["ran"] = True

    monkeypatch.setattr("bot.cli._run_with_connection_handling", _fail_if_called)

    result = CliRunner().invoke(
        app,
        ["force-entry", "AGPU", "--entry", "1.50", "--stop", "1.35"],
    )
    assert result.exit_code == 1
    assert "disabled" in result.output.lower()
    assert "allow_force_entry" in result.output
    assert called["ran"] is False, "safety gate must short-circuit before IBKR connect"


def test_force_entry_rejects_in_live_mode(monkeypatch: Any) -> None:
    """Flag on + live mode → CLI still rejects. Double-gate."""
    monkeypatch.setattr(
        "bot.cli.get_settings",
        lambda: _settings_for_force_entry(allow=True, mode="live"),
    )
    called = {"ran": False}

    def _fail_if_called(_coro_factory: Any) -> None:
        called["ran"] = True

    monkeypatch.setattr("bot.cli._run_with_connection_handling", _fail_if_called)

    result = CliRunner().invoke(
        app,
        ["force-entry", "AGPU", "--entry", "1.50", "--stop", "1.35"],
    )
    assert result.exit_code == 1
    assert "live mode" in result.output.lower()
    assert called["ran"] is False


def test_force_entry_auto_computes_scale_out_at_2r(monkeypatch: Any) -> None:
    """With ``--scale-out`` omitted, the synthesized Signal carries entry + 2R.

    Intercepts the async body via ``_run_with_connection_handling`` so we
    don't need IBKR. The kwargs captured there prove the CLI derived the
    right scale-out value before handing off to the coroutine factory.
    """
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_for_force_entry(allow=True))
    captured: dict[str, Any] = {}

    def _capture(coro_factory: Any) -> None:
        # The factory is ``lambda: _force_entry(**kwargs)``. We can't
        # easily introspect the lambda's bound kwargs, so we instead
        # patch ``_force_entry`` to record them when called.
        pass

    monkeypatch.setattr("bot.cli._run_with_connection_handling", _capture)

    async def _fake_force_entry(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("bot.cli._force_entry", _fake_force_entry)

    # Invoke the factory lambda ourselves so _fake_force_entry runs.
    def _capture_and_run(coro_factory: Any) -> None:
        import asyncio as _asyncio

        _asyncio.run(coro_factory())

    monkeypatch.setattr("bot.cli._run_with_connection_handling", _capture_and_run)

    result = CliRunner().invoke(
        app,
        ["force-entry", "AGPU", "--entry", "1.50", "--stop", "1.35"],
    )
    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "AGPU"
    assert captured["entry"] == pytest.approx(1.50)
    assert captured["stop"] == pytest.approx(1.35)
    # 2R above entry: 1.50 + 2 × (1.50 - 1.35) = 1.80
    assert captured["scale_out"] == pytest.approx(1.80)


def test_force_entry_rejects_stop_at_or_above_entry(monkeypatch: Any) -> None:
    """A long signal requires stop < entry; stop >= entry exits 1 before IBKR."""
    monkeypatch.setattr("bot.cli.get_settings", lambda: _settings_for_force_entry(allow=True))
    called = {"ran": False}

    def _fail_if_called(_coro_factory: Any) -> None:
        called["ran"] = True

    monkeypatch.setattr("bot.cli._run_with_connection_handling", _fail_if_called)

    result = CliRunner().invoke(
        app,
        ["force-entry", "AGPU", "--entry", "1.50", "--stop", "1.50"],
    )
    assert result.exit_code == 1
    assert "below" in result.output.lower() and "stop" in result.output.lower()
    assert called["ran"] is False
