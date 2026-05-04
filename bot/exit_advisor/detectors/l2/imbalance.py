"""Imbalance detector — top-K bid vs. ask total size, ratio crossings.

Sums the top ``levels_to_sum`` levels per side and computes
``max / min`` ratio. When the ratio crosses ``threshold_ratio``,
fires ``ImbalanceEvent`` identifying the favored side.

Once-per-direction-crossing semantics: an active "bid-favored"
imbalance must wind down (ratio falls below threshold) before another
"bid-favored" event can fire. The opposite-side fires fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from bot.exit_advisor.core.events import Event, ImbalanceEvent
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


@dataclass
class ImbalanceDetector:
    symbol: str
    threshold_ratio: float = 3.0
    levels_to_sum: int = 5

    _state: Literal["balanced", "bid_favored", "ask_favored"] = field(
        default="balanced", init=False
    )

    def consume(
        self, event: L2BookUpdate | L2Print, book_state: BookState
    ) -> list[Event]:
        if not isinstance(event, L2BookUpdate):
            return []
        if not book_state.bids or not book_state.asks:
            return []
        bid_total = sum(lv.size for lv in book_state.bids[: self.levels_to_sum])
        ask_total = sum(lv.size for lv in book_state.asks[: self.levels_to_sum])
        levels_summed = min(
            self.levels_to_sum, len(book_state.bids), len(book_state.asks)
        )
        if bid_total <= 0 or ask_total <= 0:
            return []
        favored: Literal["bid", "ask"]
        if bid_total >= ask_total:
            favored = "bid"
            ratio = bid_total / max(ask_total, 1)
        else:
            favored = "ask"
            ratio = ask_total / max(bid_total, 1)

        events: list[Event] = []
        new_state: Literal["balanced", "bid_favored", "ask_favored"]
        if ratio >= self.threshold_ratio:
            new_state = "bid_favored" if favored == "bid" else "ask_favored"
            if new_state != self._state:
                events.append(
                    ImbalanceEvent(
                        timestamp=event.timestamp,
                        symbol=self.symbol,
                        bid_total_size=bid_total,
                        ask_total_size=ask_total,
                        favored_side=favored,
                        ratio=ratio,
                        levels_summed=levels_summed,
                    )
                )
            self._state = new_state
        else:
            self._state = "balanced"
        return events
