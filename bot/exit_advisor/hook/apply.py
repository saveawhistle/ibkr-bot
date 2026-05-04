"""Phase 11 — apply an advisor's recommendation through the bot's existing exit paths.

The applier is the seam between the advisor's typed
:class:`ExitRecommendation` and the executor / trade manager's
existing exit primitives. It enforces:

* The advisor's gating is *not* re-run here — the spike's gate
  framework already filtered the advisor's output; the bot only
  passes through to the existing safety infrastructure.
* Action types not supported by the bot's current exit primitives
  are **rejected with a logged warning**, never silently skipped:
  the advisor knowing what got applied vs. dropped is forensically
  important.

What's supported in Phase 11:

* ``hold`` — no-op (returns False).
* ``exit_full`` — routed through ``TradeManager.execute_advisor_exit``,
  which reuses the existing ``_execute_trailing_exit`` plumbing so the
  watchdog, journal, risk engine, and OCA-children cancellation paths
  all behave identically to a normal trailing exit.

What's **not yet** supported (logged + rejected):

* ``exit_partial`` — the executor has no partial-exit-only primitive
  (scale-out is order-driven via the LMT bracket leg, not advisor-
  driven). Rejecting matches the spec's "implementation detail to
  verify" guidance.
* ``tighten_stop`` — no executor primitive for modify-stop yet (the
  spike branch is expected to add one if it needs this action).
"""

from __future__ import annotations

from typing import Protocol

import structlog

from bot.exit_advisor.core.types import ExitRecommendation, PositionLike

_log = structlog.get_logger("bot.exit_advisor")


class _AdvisorExitCapable(Protocol):
    """Structural protocol for the trade-manager surface the applier needs.

    Defined here (not as an ABC) so the applier can be tested with a
    minimal stand-in object and so :class:`bot.execution.trade_manager.TradeManager`
    doesn't have to inherit anything new.
    """

    async def execute_advisor_exit(
        self, position: PositionLike, *, exit_price: float, reason: str
    ) -> bool:
        """Full-position market-close on advisor request. Returns True on submit."""
        ...


class RecommendationApplier:
    """Routes an :class:`ExitRecommendation` to the bot's existing exit paths.

    Construct once at startup with a reference to the trade manager;
    call :meth:`apply` from the hook-call site whenever an actionable
    recommendation arrives AND ``exit_advisor.hook_acts=true``.

    Returns True iff the recommendation was acted on (an order was
    submitted). False covers all rejections + holds + unsupported
    actions; in every False branch, a structured log is emitted so
    the operator can audit advisor output that the bot didn't honour.
    """

    def __init__(self, trade_manager: _AdvisorExitCapable) -> None:
        self._trade_manager = trade_manager

    async def apply(
        self,
        recommendation: ExitRecommendation,
        position: PositionLike,
        *,
        exit_price: float,
    ) -> bool:
        """Dispatch ``recommendation`` to the matching executor primitive.

        ``exit_price`` is the reference price at the moment the
        recommendation was produced (typically the just-closed bar's
        close). The applier passes it through to the executor so
        downstream PnL math has a deterministic anchor.
        """
        action = recommendation.action
        if action == "hold":
            # The hook wrapper already logged event_actionable; this
            # confirms that "hold" actually got applied as a no-op so
            # the audit trail closes cleanly.
            _log.info(
                "exit_advisor.applied_hold",
                symbol=position.symbol,
                reason=recommendation.reason,
                source=recommendation.source,
            )
            return False
        if action == "exit_full":
            submitted = await self._trade_manager.execute_advisor_exit(
                position, exit_price=exit_price, reason=recommendation.reason or "advisor"
            )
            _log.info(
                "exit_advisor.applied_exit_full",
                symbol=position.symbol,
                exit_price=exit_price,
                submitted=submitted,
                reason=recommendation.reason,
                source=recommendation.source,
            )
            return submitted
        if action == "exit_partial":
            _log.warning(
                "exit_advisor.recommendation_rejected_unsupported",
                symbol=position.symbol,
                action=action,
                hint=(
                    "exit_partial has no executor primitive in Phase 11; "
                    "advisor-driven partial exits require a future "
                    "executor capability. Recommendation logged but not acted on."
                ),
                partial_pct=recommendation.partial_pct,
                reason=recommendation.reason,
                source=recommendation.source,
            )
            return False
        if action == "tighten_stop":
            _log.warning(
                "exit_advisor.recommendation_rejected_unsupported",
                symbol=position.symbol,
                action=action,
                hint=(
                    "tighten_stop has no executor primitive in Phase 11; "
                    "the bot has no public modify-stop API yet. "
                    "Recommendation logged but not acted on."
                ),
                new_stop_price=recommendation.new_stop_price,
                reason=recommendation.reason,
                source=recommendation.source,
            )
            return False
        # Defensive: ExitAction is a Literal so this is unreachable
        # under normal type-checking, but a poorly-typed advisor could
        # construct an invalid action via Any-erasure. Reject loudly.
        _log.error(
            "exit_advisor.recommendation_rejected_unknown_action",
            symbol=position.symbol,
            action=action,
            reason=recommendation.reason,
            source=recommendation.source,
        )
        return False
