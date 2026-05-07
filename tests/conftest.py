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


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so ``--strict-markers`` runs stay clean."""
    config.addinivalue_line(
        "markers",
        "recent_rvol_enabled: Phase 12.4 -- exercise the breakout-bar RVOL "
        "suppression gate (default-disabled by conftest autouse fixture).",
    )
