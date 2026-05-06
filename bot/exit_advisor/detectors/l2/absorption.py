"""Absorption detector — a price level being repeatedly hit but not breaking.

Iceberg behavior: the visible size at a level keeps refilling because
real demand or supply is larger than what's quoted. Detection rule:

    cumulative_size_consumed_at_level >= refresh_multiplier * visible_size_at_level
    AND the level is currently present in the book

Once a level emits absorption, the latch holds until the level is fully
removed from the book — only then does the cumulative tracker reset and
a future re-insertion of the same price get a clean window.

The cumulative-volume accounting comes from :class:`book_state.BookState`,
which the :class:`book_state.BookStateTracker` populates from L2Print
aggressor-side classification. So the detector's signal is only as
good as the upstream aggressor derivation; "unknown"-side prints don't
contribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bot.exit_advisor.core.events import AbsorptionDetected, Event
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


@dataclass
class AbsorptionDetector:
    symbol: str
    refresh_multiplier: float = 3.0

    # Per-(side, price) refresh count + already-fired latch. The latch
    # stays True until the level disappears entirely; the refresh count
    # increments on every insert/update at the same price.
    _refresh_counts: dict[tuple[Literal["bid", "ask"], float], int] = field(
        default_factory=dict, init=False
    )
    _fired: set[tuple[Literal["bid", "ask"], float]] = field(default_factory=set, init=False)

    def consume(self, event: L2BookUpdate | L2Print, book_state: BookState) -> list[Event]:
        if isinstance(event, L2BookUpdate):
            self._track_refresh(event)
            return self._maybe_emit(event, book_state)
        # Prints don't fire absorption directly, but they update the
        # cumulative counts in BookState; we re-check on the next book
        # update naturally.
        return []

    def _track_refresh(self, evt: L2BookUpdate) -> None:
        key = (evt.side, evt.price)
        if evt.operation == "delete":
            # Reset everything for this level — a fresh insertion at
            # the same price is a new absorption window.
            self._refresh_counts.pop(key, None)
            self._fired.discard(key)
            return
        # insert/update both count as refreshes; the level was either
        # placed or replenished.
        self._refresh_counts[key] = self._refresh_counts.get(key, 0) + 1

    def _maybe_emit(self, evt: L2BookUpdate, book_state: BookState) -> list[Event]:
        if evt.operation == "delete":
            return []
        key = (evt.side, evt.price)
        if key in self._fired:
            return []
        cumulative = book_state.cumulative_volume_at_level.get(key, 0)
        visible = evt.size
        if visible <= 0:
            return []
        if cumulative < self.refresh_multiplier * visible:
            return []
        # Confirm the level is still in the book at query time.
        levels = book_state.bids if evt.side == "bid" else book_state.asks
        if not any(abs(lv.price - evt.price) < 1e-9 for lv in levels):
            return []
        self._fired.add(key)
        return [
            AbsorptionDetected(
                timestamp=evt.timestamp,
                symbol=self.symbol,
                price=evt.price,
                side=evt.side,
                cumulative_size_consumed=cumulative,
                visible_size_at_level=visible,
                refresh_count=self._refresh_counts.get(key, 0),
            )
        ]
