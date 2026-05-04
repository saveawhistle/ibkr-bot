"""Three mechanical baseline policies running in shadow alongside the LLM advisor.

The baselines wrap the existing replay-harness :class:`ExitPolicy`
implementations from :mod:`bot.exit_advisor.decision.policy`. They
receive every event the agent does and produce recommendations that
are LOGGED but never affect actual exits — the agent's LLM
recommendation is the only thing the bot can act on.

Three baselines, picked to cover failure modes 1 + 2 from the
methodology:

* ``stall_1r_5min`` — exits if the trade hasn't reached 1R within 5
  minutes (catches breakouts that won't reach 2:1).
* ``trail`` — fixed $0.10 trailing stop on the running peak (catches
  give-back on the runner).
* ``fixed_r_1_5`` — exits fully at 1.5R (alternative to mechanical
  trail for noisy trades).

Each baseline is independent — their recommendations don't influence
each other and don't influence the agent. Per-trade state is held in
the baseline instances themselves; ``reset_for_new_trade`` rebuilds
them on every ``on_position_protected``.
"""

from __future__ import annotations

import structlog

from bot.exit_advisor.core.events import Event
from bot.exit_advisor.decision.policy import (
    ExitDecision,
    ExitPolicy,
    FixedRTakeProfit,
    MechanicalTrailPolicy,
    StallExitPolicy,
    TradeState,
)

_log = structlog.get_logger("bot.exit_advisor.advisor.shadow_baselines")


class ShadowBaselines:
    """Runs three mechanical policies in shadow alongside the LLM agent.

    Construct once per advisor; ``reset_for_new_trade`` reinstantiates
    the underlying policies on each new position so per-trade state
    (``_exit_emitted`` flags, etc.) starts fresh. The agent calls
    ``consume_event`` on every event it processes — even ones the
    buffer suppressed from triggering the LLM.
    """

    BASELINE_NAMES = ("stall_1r_5min", "trail", "fixed_r_1_5")

    def __init__(self) -> None:
        self._baselines: dict[str, ExitPolicy] = {}
        self.reset_for_new_trade()

    def reset_for_new_trade(self) -> None:
        """Re-instantiate all three baselines for a new position."""
        self._baselines = {
            "stall_1r_5min": StallExitPolicy(target_r=1.0, max_minutes=5),
            "trail": MechanicalTrailPolicy(trail_abs=0.10),
            "fixed_r_1_5": FixedRTakeProfit(target_r=1.5),
        }

    def consume_event(
        self,
        event: Event,
        trade_state: TradeState,
    ) -> dict[str, ExitDecision | None]:
        """Run all baselines on ``event``. Return a dict of baseline name → decision (or None).

        Each baseline runs in isolation. A baseline raising is caught and
        logged; the offending baseline's slot is set to None so the
        caller can keep going. (Mechanical baselines should never raise
        on well-formed input, but the LLM advisor must not be brought
        down by a baseline bug.)
        """
        results: dict[str, ExitDecision | None] = {}
        for name, policy in self._baselines.items():
            try:
                decision = policy.on_event(trade_state, event)
            except Exception as exc:  # noqa: BLE001 - shadow baselines must not crash the advisor
                _log.error(
                    "shadow_baseline.error",
                    baseline=name,
                    event_type=type(event).__name__,
                    error=str(exc),
                )
                results[name] = None
                continue
            results[name] = decision
        return results

    def baseline_names(self) -> tuple[str, ...]:
        """Return the names of the baselines (for logging headers + tests)."""
        return tuple(self._baselines.keys())
