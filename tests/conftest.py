"""Shared pytest fixtures for the bot test suite."""

from __future__ import annotations

import socket
from collections.abc import Iterator

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
