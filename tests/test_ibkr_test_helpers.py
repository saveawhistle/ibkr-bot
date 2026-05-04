"""Unit tests for ``tests._ibkr_test_helpers.make_test_client_id``.

The helper is small but its contract is load-bearing for the integration
test's stability — once it's in use, regressing the range or the
determinism could re-introduce the Error 326 collision class.
"""

from __future__ import annotations

import os

from _ibkr_test_helpers import make_test_client_id


def test_make_test_client_id_returns_value_in_documented_range() -> None:
    """Default invocation maps the current PID into [100, 999]."""
    cid = make_test_client_id()
    assert 100 <= cid <= 999


def test_make_test_client_id_is_deterministic_for_same_pid() -> None:
    """Same PID input → same output. Otherwise the integration test would be flaky."""
    a = make_test_client_id(pid=12345)
    b = make_test_client_id(pid=12345)
    assert a == b


def test_make_test_client_id_uses_current_pid_when_none() -> None:
    """``pid=None`` uses ``os.getpid()`` — same value as passing the PID explicitly."""
    auto = make_test_client_id()
    explicit = make_test_client_id(pid=os.getpid())
    assert auto == explicit


def test_make_test_client_id_distinguishes_typical_pids() -> None:
    """Two adjacent PIDs (the common case for back-to-back test invocations) map distinctly.

    The collision class we explicitly *don't* protect against is PIDs differing by
    exactly the span (900). Two PIDs differing by 1 must always differ; this is
    the practically-relevant case for back-to-back ``pytest`` runs.
    """
    a = make_test_client_id(pid=10001)
    b = make_test_client_id(pid=10002)
    assert a != b


def test_make_test_client_id_modular_collision_at_span_boundary() -> None:
    """Document the known modular collision: PIDs differing by exactly 900 collide.

    This is acceptable because the operational scenario we're protecting against
    is a fresh test process whose PID differs from a recently-terminated process —
    these are nearly always within a small delta of each other on real systems,
    not 900 apart. Pinning this in a test makes the trade-off explicit so a
    future refactor that widens the span doesn't accidentally also drop this
    deliberate property.
    """
    base = make_test_client_id(pid=42_000)
    collision = make_test_client_id(pid=42_900)
    assert base == collision
