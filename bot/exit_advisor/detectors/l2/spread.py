"""Spread detector — sharp widening or tightening vs. rolling average.

Maintains an N-event rolling window of spreads (recomputed each book
update that touches top-of-book). Fires ``SpreadEvent`` when the
current spread crosses the configured widening or tightening threshold.

Once-per-direction-crossing: an active widening must wind back to
within the normal range (between tightening and widening ratios)
before another widening can fire. Same for tightening. Without this
latch, every event in a sustained wide-spread regime would fire — too
noisy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from bot.exit_advisor.core.events import Event, SpreadEvent
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


@dataclass
class SpreadEventDetector:
    symbol: str
    widening_ratio: float = 2.0
    tightening_ratio: float = 0.5
    rolling_window_events: int = 20

    _history: deque[float] = field(default_factory=deque, init=False)
    _state: Literal["normal", "wide", "tight"] = field(default="normal", init=False)

    def consume(
        self, event: L2BookUpdate | L2Print, book_state: BookState
    ) -> list[Event]:
        # Only book updates can move top-of-book (prints don't shift
        # quotes). Skip prints — they'd add noise to the rolling window.
        if not isinstance(event, L2BookUpdate):
            return []
        spread = book_state.spread
        if spread is None or spread <= 0:
            return []
        # Update the rolling window AFTER we read it for ratio
        # comparison: compare current spread against the prior window's
        # average so a single big tick doesn't immediately raise the bar.
        prior = list(self._history)
        if len(prior) < self.rolling_window_events:
            self._history.append(spread)
            if len(self._history) > self.rolling_window_events:
                self._history.popleft()
            return []  # warmup — no signal until window is full

        average = sum(prior) / len(prior)
        if average <= 0:
            self._history.append(spread)
            if len(self._history) > self.rolling_window_events:
                self._history.popleft()
            return []

        ratio = spread / average
        events: list[Event] = []
        if ratio >= self.widening_ratio and self._state != "wide":
            events.append(
                SpreadEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol,
                    spread_now=spread,
                    rolling_average_spread=average,
                    direction="widening",
                    ratio=ratio,
                )
            )
            self._state = "wide"
        elif ratio <= self.tightening_ratio and self._state != "tight":
            events.append(
                SpreadEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol,
                    spread_now=spread,
                    rolling_average_spread=average,
                    direction="tightening",
                    ratio=ratio,
                )
            )
            self._state = "tight"
        elif self.tightening_ratio < ratio < self.widening_ratio:
            self._state = "normal"

        self._history.append(spread)
        if len(self._history) > self.rolling_window_events:
            self._history.popleft()
        return events
