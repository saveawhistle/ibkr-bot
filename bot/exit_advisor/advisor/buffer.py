"""Event accumulator + significance triggering for the LLM advisor.

The advisor sees every bar event, every L2 event, every milestone — but
must not call the LLM on every one (latency, cost, signal-to-noise).
The buffer's job is to:

1. Always record the event so the LLM has full context when it does run.
2. Decide whether THIS event should trigger an LLM call right now.
3. On trigger, drain the accumulated buffer to the caller and clear it.

Three event tiers:

* **ALWAYS_TRIGGER** — high-signal classes (PositionProtected,
  PartialFillEvent, OrderRejectionEvent, RMultipleReached,
  DrawdownFromPeak). Trigger on arrival, subject to the hard floor.
* **TIME_FLOOR_TRIGGER** — moderate-signal classes (BarFinalizedEvent,
  MovingAverageCross). Trigger only if at least
  ``time_floor_seconds`` has elapsed since the last LLM call.
* **buffer-only** — everything else (L2 prints, level touches, bar
  shapes, volume spikes). Buffered for context but never trigger
  on their own.

The hard floor (default 10 s) bounds the LLM call rate at one per
10 seconds even on always-trigger events, preventing flooding from a
burst of partial fills or rapid R-multiple crossings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import ClassVar

from bot.exit_advisor.core.events import (
    BarFinalizedEvent,
    DrawdownFromPeak,
    Event,
    MovingAverageCross,
    OrderRejectionEvent,
    PartialFillEvent,
    PositionProtected,
    RMultipleReached,
)


@dataclass(frozen=True)
class BufferDecision:
    """Result of one ``EventBuffer.consume`` call.

    When ``trigger`` is True, ``triggering_event`` is the event that
    crossed the threshold and ``buffered_events`` is the full set of
    events accumulated since the last trigger (including the triggering
    event itself, in chronological order).

    When ``trigger`` is False, ``buffered_events`` is empty (the buffer
    holds them internally for the next trigger) and ``skip_reason`` is
    a short tag for the log: ``non_significant``, ``hard_floor_active``,
    or ``time_floor_active``.
    """

    trigger: bool
    triggering_event: Event | None
    buffered_events: list[Event] = field(default_factory=list)
    skip_reason: str | None = None


class EventBuffer:
    """Per-position event buffer with significance triggering.

    A new instance is constructed per protected position (the agent
    creates one on ``on_position_protected`` and discards on
    ``on_position_closed``). State is intentionally per-trade; cross-
    trade triggering would mix unrelated context.
    """

    ALWAYS_TRIGGER_EVENTS: ClassVar[tuple[type[Event], ...]] = (
        PositionProtected,
        PartialFillEvent,
        OrderRejectionEvent,
        RMultipleReached,
        DrawdownFromPeak,
    )

    TIME_FLOOR_TRIGGER_EVENTS: ClassVar[tuple[type[Event], ...]] = (
        BarFinalizedEvent,
        MovingAverageCross,
    )

    DEFAULT_TIME_FLOOR_SECONDS = 30.0
    DEFAULT_HARD_FLOOR_SECONDS = 10.0

    def __init__(
        self,
        time_floor_seconds: float = DEFAULT_TIME_FLOOR_SECONDS,
        hard_floor_seconds: float = DEFAULT_HARD_FLOOR_SECONDS,
    ) -> None:
        if time_floor_seconds < 0.0:
            raise ValueError(f"time_floor_seconds must be >= 0 (got {time_floor_seconds})")
        if hard_floor_seconds < 0.0:
            raise ValueError(f"hard_floor_seconds must be >= 0 (got {hard_floor_seconds})")
        self._time_floor = timedelta(seconds=time_floor_seconds)
        self._hard_floor = timedelta(seconds=hard_floor_seconds)
        self._pending: list[Event] = []
        self._last_trigger_at: datetime | None = None

    def consume(self, event: Event, timestamp: datetime) -> BufferDecision:
        """Record ``event`` and decide whether it triggers an LLM call now.

        ``timestamp`` is the wall-clock time at which the consumer
        observed the event (typically ``event.timestamp`` for
        bar/replay events, ``datetime.now(UTC)`` for live ticks). The
        agent decides which to use; the buffer just measures elapsed
        time against it.
        """
        self._pending.append(event)

        is_always = isinstance(event, self.ALWAYS_TRIGGER_EVENTS)
        is_time_floor = isinstance(event, self.TIME_FLOOR_TRIGGER_EVENTS)

        if not is_always and not is_time_floor:
            return BufferDecision(
                trigger=False,
                triggering_event=None,
                skip_reason="non_significant",
            )

        if self._last_trigger_at is not None:
            elapsed = timestamp - self._last_trigger_at
            if elapsed < self._hard_floor:
                # Hard floor caps absolute call rate; even always-trigger events buffer here.
                return BufferDecision(
                    trigger=False,
                    triggering_event=None,
                    skip_reason="hard_floor_active",
                )
            if is_time_floor and not is_always and elapsed < self._time_floor:
                return BufferDecision(
                    trigger=False,
                    triggering_event=None,
                    skip_reason="time_floor_active",
                )

        drained = list(self._pending)
        self._pending.clear()
        self._last_trigger_at = timestamp
        return BufferDecision(
            trigger=True,
            triggering_event=event,
            buffered_events=drained,
        )

    def pending_count(self) -> int:
        """Number of events accumulated since the last trigger (forensics + tests)."""
        return len(self._pending)

    def last_trigger_at(self) -> datetime | None:
        """Wall-clock timestamp of the most recent triggered call, or None."""
        return self._last_trigger_at
