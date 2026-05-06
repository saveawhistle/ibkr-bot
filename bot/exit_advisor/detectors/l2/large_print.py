"""Large-print detector — single print exceeds rolling-average size by N×.

Maintains a rolling N-print average (configurable, default 50). When a
print's size is at least ``size_multiplier`` times the average, fires
``LargePrint``.

Warmup: doesn't emit until the rolling window has at least 10 prints.
A 1-print average isn't a meaningful comparison; 10 is a small but
defensible floor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from bot.exit_advisor.core.events import Event, LargePrint
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print

WARMUP_MIN_PRINTS = 10


@dataclass
class LargePrintDetector:
    symbol: str
    size_multiplier: float = 5.0
    rolling_window_prints: int = 50

    _history: deque[int] = field(default_factory=deque, init=False)

    def consume(self, event: L2BookUpdate | L2Print, book_state: BookState) -> list[Event]:
        if not isinstance(event, L2Print):
            return []
        events: list[Event] = []
        if len(self._history) >= WARMUP_MIN_PRINTS:
            average = sum(self._history) / len(self._history)
            if average > 0 and event.size >= self.size_multiplier * average:
                events.append(
                    LargePrint(
                        timestamp=event.timestamp,
                        symbol=self.symbol,
                        price=event.price,
                        size=event.size,
                        rolling_average_size=average,
                        ratio=event.size / average,
                        aggressor_side=event.aggressor_side,
                    )
                )
        self._history.append(event.size)
        if len(self._history) > self.rolling_window_prints:
            self._history.popleft()
        return events
