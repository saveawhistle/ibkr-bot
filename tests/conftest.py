"""Shared pytest fixtures for the bot test suite."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to (host, port) succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def paper_tws_available() -> Iterator[bool]:
    """Yield True when a paper TWS/Gateway socket is reachable on the configured host/port."""
    from bot.config import get_settings

    settings = get_settings()
    yield _port_open(settings.ibkr.host, settings.ibkr.port)


@pytest.fixture(autouse=True)
def _disable_recent_rvol_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 12.4: default-disable the breakout-bar RVOL suppression in tests.

    The new ``check_recent_window_rvol`` helper requires 21+ bars to
    populate the default 20-bar window; legacy strategy and orchestrator
    tests build small synthetic frames (10-15 bars) and would otherwise
    fail on ``signal_suppressed_window_not_populated``. Default-disable
    keeps those tests asserting their original invariants.

    Tests that DO want to exercise the recent-rvol gate opt in via the
    ``@pytest.mark.recent_rvol_enabled`` marker. The Phase 12.4 unit/
    integration tests live in ``test_strategy_differentiation.py`` and
    use the marker explicitly.
    """
    if request.node.get_closest_marker("recent_rvol_enabled"):
        return

    def _noop(**_kwargs: Any) -> None:
        return None

    # Patch at the strategy modules (where the symbol is bound after
    # `from bot.strategies.volume import check_recent_window_rvol`).
    monkeypatch.setattr("bot.strategies.gap_and_go.check_recent_window_rvol", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_recent_window_rvol", _noop)


@pytest.fixture(autouse=True)
def _pin_legacy_momentum_window_start(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 12.6: pin the momentum default window_start back to 09:30 for legacy tests.

    Pre-12.6, ``MomentumStrategy`` started evaluating bars at the 09:30
    market open. Phase 12.6 raised that default to 10:00 so momentum
    sequences after gap-and-go's opening window. Many pre-existing
    tests construct ``MomentumStrategy()`` without an explicit
    ``window_start`` and use 09:30+ bar fixtures -- they'd silently
    drop those bars under the new default.

    The autouse fixture rewrites the class default back to 09:30 for
    every test except those marked ``@pytest.mark.momentum_default_window_start``,
    which opt in to the production 10:00 default. The Phase 12.6
    dedicated tests (test_momentum_window_start.py) construct the
    strategy with explicit window_start values so the rewrite is a
    no-op for them.
    """
    if request.node.get_closest_marker("momentum_default_window_start"):
        return
    from datetime import time as _time

    from bot.strategies import momentum as _mom_module

    monkeypatch.setattr(_mom_module, "_DEFAULT_WINDOW_START", _time(9, 30))


@pytest.fixture(autouse=True)
def _disable_entry_quality_gates_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 13: default-disable the entry-quality gates for legacy strategy tests.

    The five new gates (halt detection, impulse strength, consolidation
    tightness, volume contraction, VWAP extension) require structurally
    realistic bar fixtures (>=10 bars with proper impulse + consolidation
    shape, real volume signature, and in-line VWAP). Legacy strategy and
    orchestrator tests build small synthetic frames that wouldn't satisfy
    those constraints; default-disable the gates so those tests continue
    to assert their original invariants.

    Tests that DO want to exercise the gates opt in via the
    ``@pytest.mark.entry_quality_enabled`` marker. The Phase 13 unit and
    integration tests live in ``test_entry_quality.py`` and use the
    marker explicitly.
    """
    if request.node.get_closest_marker("entry_quality_enabled"):
        return

    def _noop(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("bot.strategies.gap_and_go.check_halt_detection", _noop)
    monkeypatch.setattr("bot.strategies.gap_and_go.check_impulse_strength", _noop)
    monkeypatch.setattr("bot.strategies.gap_and_go.check_consolidation_tightness", _noop)
    monkeypatch.setattr("bot.strategies.gap_and_go.check_volume_contraction", _noop)
    monkeypatch.setattr("bot.strategies.gap_and_go.check_vwap_extension", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_halt_detection", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_impulse_strength", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_consolidation_tightness", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_volume_contraction", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_vwap_extension", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_consolidation_vwap_hold", _noop)
    monkeypatch.setattr("bot.strategies.momentum.check_breakout_volume_ratio", _noop)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so ``--strict-markers`` runs stay clean."""
    config.addinivalue_line(
        "markers",
        "recent_rvol_enabled: Phase 12.4 -- exercise the breakout-bar RVOL "
        "suppression gate (default-disabled by conftest autouse fixture).",
    )
    config.addinivalue_line(
        "markers",
        "momentum_default_window_start: Phase 12.6 -- exercise the production "
        "10:00 default for momentum.window_start (autouse fixture pins back "
        "to 09:30 for legacy compat otherwise).",
    )
    config.addinivalue_line(
        "markers",
        "entry_quality_enabled: Phase 13 -- exercise the entry-quality gates "
        "(default-disabled by conftest autouse fixture for legacy tests).",
    )
