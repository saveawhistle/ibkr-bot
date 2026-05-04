"""Risk engine + automatic rehab tier system.

Two related concerns share this package:

* :mod:`bot.risk.engine` — per-trade sizing (``compute_shares``),
  daily/giveback halt triggers, the ``RiskEngine`` orchestration
  class, and the on-disk halt flag for crash-restart persistence.
* :mod:`bot.risk.rehab` — automatic post-loss-streak tier scaling
  (``RehabTier`` + ``RehabPolicy``) that shrinks per-trade and daily
  caps after consecutive red days or drawdown thresholds, and
  unwinds the scaling once the operator earns the configured
  recovery fraction back.

The names below are re-exported here so callers can keep using
``from bot.risk import RiskEngine`` (and friends) — the rename of
``risk.py`` to ``risk/engine.py`` is invisible above this line. This
is the only re-export façade in the package; sibling subpackages
(``execution``, ``brokerage``, etc.) follow the full-path convention
already established by ``bot/strategies/``.
"""

from bot.risk.engine import (
    Approved,
    HaltRecord,
    ReEntryAllowed,
    Rejected,
    RiskEngine,
    RiskState,
    compute_shares,
    daily_loss_hit,
    delete_halt_flag,
    giveback_hit,
    profit_goal_hit,
    read_halt_flag,
    write_halt_flag,
)

__all__ = [
    "Approved",
    "HaltRecord",
    "ReEntryAllowed",
    "Rejected",
    "RiskEngine",
    "RiskState",
    "compute_shares",
    "daily_loss_hit",
    "delete_halt_flag",
    "giveback_hit",
    "profit_goal_hit",
    "read_halt_flag",
    "write_halt_flag",
]
