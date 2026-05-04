"""Bid/Offer pulled detectors.

A "pulled" delete is a level disappearing without an offsetting print —
the bidder/offerer withdrew. A "consumed" delete is a level disappearing
with prints at that price near the same instant — the level was hit by
trades. The signal is meaningfully different: a pull is an absence of
participation; a consumption is active counter-side aggression.

The classifier uses the recent-prints window plus a configurable
``lookback_ms``. When a delete arrives, we look back ``lookback_ms``
milliseconds in the print stream for prints at-or-near the deleted
price. If the matching prints' total size approximately matches the
deleted level's size, classify as consumed (no event). Otherwise: pulled.

Edge case: partial consumption (a print smaller than the deleted level).
Treated as PULLED — even if some shares hit the bid, the rest were
withdrawn, and the withdrawal is the more interesting signal. Tunable
later if calibration suggests otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from bot.exit_advisor.core.events import BidPulled, Event, OfferPulled
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print

# When matching prints to a deleted level, accept prints within this
# fraction of the deleted price as "at the same level". Accommodates
# IBKR feed sub-cent variation on dark prints.
PRICE_MATCH_TOLERANCE = 0.0005


@dataclass
class _PullsDetectorBase:
    symbol: str
    side: Literal["bid", "ask"]
    lookback_ms: int = 100

    # Track the last seen size at each price so we know what "consumed
    # size" comparison should be against — a delete carries position +
    # price but not the size that was there before it.
    _last_seen_size: dict[float, int] = field(default_factory=dict, init=False)
    _last_seen_position: dict[float, int] = field(default_factory=dict, init=False)

    def consume(
        self, event: L2BookUpdate | L2Print, book_state: BookState
    ) -> list[Event]:
        if isinstance(event, L2BookUpdate):
            return self._on_book_update(event, book_state)
        return []  # Prints don't drive this detector directly.

    def _on_book_update(
        self, evt: L2BookUpdate, book_state: BookState
    ) -> list[Event]:
        if evt.side != self.side:
            return []
        if evt.operation in ("insert", "update"):
            self._last_seen_size[evt.price] = evt.size
            self._last_seen_position[evt.price] = evt.position
            return []
        # delete
        size_before = self._last_seen_size.pop(evt.price, evt.size)
        position = self._last_seen_position.pop(evt.price, evt.position)

        # Look back through recent prints for offsetting trade volume.
        cutoff = evt.timestamp - timedelta(milliseconds=self.lookback_ms)
        offsetting_size = 0
        for prn in book_state.recent_prints:
            if prn.timestamp < cutoff or prn.timestamp > evt.timestamp:
                continue
            if abs(prn.price - evt.price) > PRICE_MATCH_TOLERANCE * max(evt.price, 1.0):
                continue
            offsetting_size += prn.size

        # Consumed if offsetting prints cover at least 80% of the level.
        # Below that, the level was largely pulled — fire the event.
        if offsetting_size >= 0.80 * size_before:
            return []

        return [self._make_event(evt, size_before, position)]

    def _make_event(
        self, evt: L2BookUpdate, size_pulled: int, position: int
    ) -> Event:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class BidPulledDetector(_PullsDetectorBase):
    side: Literal["bid", "ask"] = "bid"

    def _make_event(
        self, evt: L2BookUpdate, size_pulled: int, position: int
    ) -> Event:
        return BidPulled(
            timestamp=evt.timestamp,
            symbol=self.symbol,
            price=evt.price,
            size_pulled=size_pulled,
            position_in_book=position,
        )


@dataclass
class OfferPulledDetector(_PullsDetectorBase):
    side: Literal["bid", "ask"] = "ask"

    def _make_event(
        self, evt: L2BookUpdate, size_pulled: int, position: int
    ) -> Event:
        return OfferPulled(
            timestamp=evt.timestamp,
            symbol=self.symbol,
            price=evt.price,
            size_pulled=size_pulled,
            position_in_book=position,
        )
