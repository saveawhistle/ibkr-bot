"""Test-only helpers for talking to a real paper TWS / IB Gateway.

Imported by ``test_ping_paper_account`` and any future integration tests
that need to connect to a real broker socket. **Not** imported by
production code under ``bot/`` — production behavior is unchanged.

Why this exists: when a prior bot or test process disconnects from TWS
ungracefully (Ctrl-C in a critical section, network blip, IDE
termination), TWS holds the client slot for ~30-60 seconds. A subsequent
test using the same hard-coded ``client_id`` then fails with
``Error 326, reqId -1: Unable to connect as the client id is already in
use``. The auto-skip predicate in the integration test only checks
socket reachability, so it can't tell the difference between "TWS is
down" (skip) and "TWS is up but my slot is occupied" (test fails).

The fix is to derive the test ``client_id`` from the OS process PID so
each test invocation lands on a distinct slot, eliminating the
collision class entirely. Production keeps its operator-configured
``client_id`` (typically 17 — predictable for log filtering, manual
status checks in TWS, and the executor's "filter foreign orders" logic).
"""

from __future__ import annotations

import os

# Mapping window for PID-derived test client IDs. ``[100, 999]`` keeps
# the value comfortably above the production-configured client_id
# (typically 17) and within IBKR's supported range when TWS is set to
# accept client IDs up to 999.
#
# **TWS configuration prerequisite for integration tests**: TWS
# (Global Config → API → Settings) ships with a "Master API client ID"
# selector that gates which IDs are allowed. The default range varies
# by version but is often capped at 31 or 99. To run integration tests
# you must either widen the range to >= 999 (recommended), or set the
# Master Client ID to 0 (which permits any client ID). Without this
# step ``connectAsync`` will refuse the PID-derived ID even when the
# slot is otherwise free. Production operators using ``client_id: 17``
# in ``config.yaml`` are unaffected.
_PID_CLIENT_ID_LOWER = 100
_PID_CLIENT_ID_UPPER = 999
_PID_CLIENT_ID_SPAN = _PID_CLIENT_ID_UPPER - _PID_CLIENT_ID_LOWER + 1


def make_test_client_id(pid: int | None = None) -> int:
    """Return a PID-derived ``client_id`` in ``[100, 999]`` for integration tests.

    Deterministic within a process: the same PID always maps to the same
    ID. Different PIDs produce different IDs except for the rare
    modular collision (PIDs differing by 900). The collision class we
    actually care about — back-to-back test invocations after an
    ungraceful disconnect — is eliminated because the new test's PID
    differs from the lingering connection's.

    ``pid`` is overridable so unit tests can pin behavior without
    monkeypatching ``os.getpid``.
    """
    pid_value = os.getpid() if pid is None else pid
    return (pid_value % _PID_CLIENT_ID_SPAN) + _PID_CLIENT_ID_LOWER
