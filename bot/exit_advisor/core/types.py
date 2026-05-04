"""Phase 11 — public types for the exit advisor hook.

The advisor protocol is intentionally minimal. Three lifecycle methods
and three immutable value types form the entire contract between the
bot and any advisor implementation:

* :class:`ExitRecommendation` — what an advisor wants the bot to do.
* :class:`AdvisorResponse` — the advisor's three-state reply (skipped /
  held / actionable) so forensic logging can distinguish "no opinion
  formed" from "evaluated and decided to hold".
* :class:`ExitAdvisorHook` — the advisor's protocol. Implemented on the
  spike branch (and potentially graduated implementations in future).

Bot-side event taxonomy (:class:`Event`, :class:`BarFinalizedEvent`)
is deliberately small: the bot only emits raw lifecycle / data-flow
events. Detection (turning these into "9 EMA broke", "absorption",
etc.) lives with the advisor — that's why this Phase ships no detector
imports from ``spike/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# Re-export the canonical Event taxonomy so callers can keep importing
# ``from bot.exit_advisor.core.types import Event, BarFinalizedEvent``.
# The actual definitions live in ``events.py`` (a single source of truth
# shared with the spike harness's event flow).
from bot.exit_advisor.core.events import BarFinalizedEvent, Event

__all__ = [
    "AdvisorResponse",
    "BarFinalizedEvent",
    "Event",
    "ExitAction",
    "ExitAdvisorHook",
    "ExitRecommendation",
    "PositionLike",
]

ExitAction = Literal["hold", "exit_full", "exit_partial", "tighten_stop"]


@dataclass(frozen=True)
class ExitRecommendation:
    """An advisor's actionable recommendation for an open position.

    Validation: ``exit_partial`` requires ``partial_pct`` strictly inside
    ``(0.0, 1.0]``; ``tighten_stop`` requires ``new_stop_price`` to be
    set (positive). ``confidence`` is constrained to ``[0.0, 1.0]``.

    Validation runs in ``__post_init__`` because frozen dataclasses don't
    play with pydantic and these checks are cheap. Bad recommendations
    raise ``ValueError`` rather than being silently coerced — an advisor
    bug is better surfaced loudly than acted on incoherently.
    """

    action: ExitAction
    partial_pct: float = 0.0
    new_stop_price: float | None = None
    confidence: float = 1.0
    reason: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ExitRecommendation.confidence must be in [0.0, 1.0]; got {self.confidence}"
            )
        if self.action == "exit_partial":
            if not 0.0 < self.partial_pct <= 1.0:
                raise ValueError(
                    "ExitRecommendation.partial_pct must be in (0.0, 1.0] for action="
                    f"'exit_partial'; got {self.partial_pct}"
                )
        elif self.partial_pct != 0.0:
            raise ValueError(
                "ExitRecommendation.partial_pct must be 0.0 unless action='exit_partial'; "
                f"got {self.partial_pct} for action={self.action!r}"
            )
        if self.action == "tighten_stop":
            if self.new_stop_price is None or self.new_stop_price <= 0.0:
                raise ValueError(
                    "ExitRecommendation.new_stop_price must be > 0.0 for action="
                    f"'tighten_stop'; got {self.new_stop_price}"
                )
        elif self.new_stop_price is not None:
            raise ValueError(
                "ExitRecommendation.new_stop_price must be None unless action='tighten_stop'; "
                f"got {self.new_stop_price} for action={self.action!r}"
            )


@dataclass(frozen=True)
class AdvisorResponse:
    """The advisor's three-state reply to an :meth:`ExitAdvisorHook.on_event`.

    Three legal shapes (the property helpers below name them):

    * **skipped** — ``recommendation=None, evaluation_performed=False``.
      The advisor saw the event but did not reason about it (e.g. event
      type the advisor doesn't care about, or buffered for later).
    * **held** — ``recommendation=None, evaluation_performed=True``. The
      advisor evaluated and decided to hold.
    * **actionable** — ``recommendation`` set, ``evaluation_performed=True``.

    The hook wrapper logs the state explicitly so a session JSONL can be
    grepped for "advisor evaluated and held" vs "advisor skipped" vs
    "advisor recommended action".
    """

    recommendation: ExitRecommendation | None = None
    evaluation_performed: bool = False
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.recommendation is not None and not self.evaluation_performed:
            raise ValueError(
                "AdvisorResponse with a recommendation must have evaluation_performed=True; "
                "an unreasoned recommendation is meaningless."
            )

    @property
    def is_skipped(self) -> bool:
        """True iff the advisor did not reason about the event (no recommendation)."""
        return self.recommendation is None and not self.evaluation_performed

    @property
    def is_held(self) -> bool:
        """True iff the advisor reasoned and decided to hold (no recommendation)."""
        return self.recommendation is None and self.evaluation_performed

    @property
    def is_actionable(self) -> bool:
        """True iff the advisor produced a recommendation."""
        return self.recommendation is not None


class PositionLike(Protocol):
    """Structural protocol matching ``bot.execution.position_state.Position``.

    Used by the hook protocol so advisor implementations can type
    against this rather than importing the concrete dataclass (which
    keeps the spike branch's advisor decoupled from internal bot
    refactors of Position's auxiliary fields).

    Fields are declared as read-only properties so the concrete
    ``Position`` (a frozen dataclass with read-only attributes)
    satisfies the protocol under mypy strict — declaring them as
    plain class attributes would imply read-write semantics that
    frozen dataclasses can't provide.
    """

    @property
    def symbol(self) -> str: ...

    @property
    def strategy(self) -> str: ...

    @property
    def shares(self) -> int: ...

    @property
    def avg_price(self) -> float: ...

    @property
    def stop_price(self) -> float: ...

    @property
    def scale_out_price(self) -> float: ...

    @property
    def status(self) -> str: ...

    @property
    def scaled_out(self) -> bool: ...


@runtime_checkable
class ExitAdvisorHook(Protocol):
    """Interface implemented by exit-advisor implementations.

    Lifecycle:

    1. :meth:`on_position_protected` — called once per position after the
       entry fill is confirmed and protection children (STP / target)
       are working. The advisor can stand up per-position resources.
    2. :meth:`on_event` — called on each event fed during the position's
       open lifetime. Returns an :class:`AdvisorResponse`.
    3. :meth:`on_position_closed` — called when the position transitions
       to terminal ``closed`` (any cause). The advisor can tear down.

    Implementations should be defensive: any exception is caught by the
    hook wrapper and treated as if the call returned ``None`` /
    ``AdvisorResponse(skipped)``. The bot will not crash on advisor
    bugs.
    """

    def on_position_protected(self, position: PositionLike) -> None:
        """Called once after the position is confirmed protected.

        ``position`` reflects the post-protection state (parent filled,
        stop_order_id set). The advisor may capture references but must
        not mutate.
        """
        ...

    def on_event(self, position: PositionLike, event: Event) -> AdvisorResponse:
        """Called whenever an event is emitted for this open position.

        Return an :class:`AdvisorResponse` indicating skipped / held /
        actionable. Returning a bare ``None`` is treated as
        ``AdvisorResponse(evaluation_performed=False)`` (skipped) so
        legacy / minimal advisors degrade gracefully — but new code
        should return the typed response so logging is precise.
        """
        ...

    def on_position_closed(self, position: PositionLike, final_pnl: float) -> None:
        """Called when the position transitions to ``closed`` (any cause).

        ``final_pnl`` is the realized signed dollar PnL (post-scale +
        post-runner totals). The advisor should release per-position
        resources here.
        """
        ...
