"""Book state tracker — maintains current depth-of-book + recent prints
from the canonical L2 event stream. Detectors query :meth:`get_state`
for context; the tracker itself is independent of detector logic.

Out-of-order or malformed updates are logged at WARNING and the tracker
attempts graceful continuation rather than crashing — IBKR's depth feed
occasionally delivers updates with positions outside the maintained
window (e.g. a position-15 update when only 10 rows are subscribed),
and a robust live runtime can't afford to die on every such glitch.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .l2_events import L2BookUpdate, L2Print

log = logging.getLogger(__name__)


@dataclass
class BookLevel:
    price: float
    size: int
    last_operation: Literal["insert", "update", "delete"]
    last_update_timestamp: datetime
    market_maker: str | None = None


@dataclass
class BookState:
    bids: list[BookLevel]
    """Sorted by price descending; ``bids[0]`` is the top-of-book bid."""

    asks: list[BookLevel]
    """Sorted by price ascending; ``asks[0]`` is the top-of-book ask."""

    recent_prints: deque[L2Print]
    """Rolling window of the most recent prints; bounded by the
    tracker's ``max_print_history``."""

    cumulative_volume_at_level: dict[tuple[Literal["bid", "ask"], float], int]
    """Total size traded at each ``(side, price)`` level since the level
    was first seen. Cleared for a level when the level is fully removed
    from the book (delete with no immediate re-insert)."""

    spread: float | None
    """``asks[0].price - bids[0].price`` when both sides have a top
    level; ``None`` otherwise. Convenience field — recomputed each
    update so detectors don't need to."""


class BookStateTracker:
    """Consumes :class:`L2BookUpdate` and :class:`L2Print` events,
    maintains the current :class:`BookState`. Stateless across symbols —
    instantiate one tracker per symbol.

    The tracker stores its bid/ask sides as price-keyed dicts internally
    for O(1) update/delete; ``get_state`` materializes them into sorted
    lists at query time. This makes per-event updates cheap; query cost
    is proportional to depth (typically 10 rows = trivial).
    """

    def __init__(self, max_print_history: int = 100) -> None:
        self._max_print_history = max_print_history
        # Keyed by price for O(1) update/delete. Position is tracked but
        # is informational; IBKR's position semantics tend to drift on
        # busy books, and detectors that care about top-of-book just
        # want the price-sorted view.
        self._bid_levels: dict[float, BookLevel] = {}
        self._ask_levels: dict[float, BookLevel] = {}
        self._recent_prints: deque[L2Print] = deque(maxlen=max_print_history)
        self._cum_volume: dict[tuple[Literal["bid", "ask"], float], int] = {}

    # --- public ---

    def consume(self, event: L2BookUpdate | L2Print) -> None:
        if isinstance(event, L2BookUpdate):
            self._consume_book_update(event)
        else:
            self._consume_print(event)

    def get_state(self) -> BookState:
        bids = sorted(self._bid_levels.values(), key=lambda lv: -lv.price)
        asks = sorted(self._ask_levels.values(), key=lambda lv: lv.price)
        spread: float | None = None
        if bids and asks:
            spread = asks[0].price - bids[0].price
        return BookState(
            bids=bids,
            asks=asks,
            recent_prints=deque(self._recent_prints, maxlen=self._max_print_history),
            cumulative_volume_at_level=dict(self._cum_volume),
            spread=spread,
        )

    # --- internals ---

    def _consume_book_update(self, evt: L2BookUpdate) -> None:
        levels = self._bid_levels if evt.side == "bid" else self._ask_levels
        if evt.operation == "delete":
            # Look up by price (canonical) — IBKR's position field can
            # be stale by the time the message arrives, but price is
            # authoritative.
            if evt.price in levels:
                del levels[evt.price]
                # Reset the cumulative_volume_at_level latch so a future
                # re-insertion of the same price starts fresh accounting.
                self._cum_volume.pop((evt.side, evt.price), None)
            else:
                log.warning(
                    "delete for non-existent %s level at %.4f (%s); ignoring",
                    evt.side, evt.price, evt.symbol,
                )
            return

        if evt.size <= 0:
            # IBKR sometimes uses size=0 with operation=update as a soft
            # delete. Treat it as a delete for our purposes.
            if evt.price in levels:
                del levels[evt.price]
                self._cum_volume.pop((evt.side, evt.price), None)
            return

        # insert/update — canonicalize on price.
        levels[evt.price] = BookLevel(
            price=evt.price,
            size=evt.size,
            last_operation=evt.operation,
            last_update_timestamp=evt.timestamp,
            market_maker=evt.market_maker,
        )

    def _consume_print(self, evt: L2Print) -> None:
        self._recent_prints.append(evt)
        # Allocate the print volume to the side it took liquidity from.
        # buy aggressor → liquidity taken from ask side; sell → bid side.
        # unknown → neither side gets the cumulative; absorption can't
        # see it, but that's the right thing for an undecided print.
        if evt.aggressor_side == "buy":
            key: tuple[Literal["bid", "ask"], float] = ("ask", evt.price)
            self._cum_volume[key] = self._cum_volume.get(key, 0) + evt.size
        elif evt.aggressor_side == "sell":
            key = ("bid", evt.price)
            self._cum_volume[key] = self._cum_volume.get(key, 0) + evt.size
        # unknown: do not attribute to either side
