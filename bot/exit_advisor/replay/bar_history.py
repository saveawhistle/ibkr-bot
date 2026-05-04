"""Rolling bar buffer + session-cumulative state.

Each detector queries this buffer for the data it needs (recent bars,
running VWAP, session high/low). Centralizing the bookkeeping here means
a detector can ask "what's VWAP right now?" without having to recompute
it from raw bars on every event.

Session-cumulative state is reset implicitly: a fresh ``BarHistory`` is
constructed per trade replay, so the cumulative numerators/denominators
restart at each replay's first ``add_bar`` call. There is no explicit
mid-session reset hook — the harness instantiates one buffer per trade.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .replay_source import Bar


@dataclass
class BarHistory:
    max_bars: int = 200

    def __post_init__(self) -> None:
        self._bars: deque[Bar] = deque(maxlen=self.max_bars)
        self._vwap_num: float = 0.0
        self._vwap_den: int = 0
        self._session_high: float = float("-inf")
        self._session_low: float = float("inf")
        self._cum_volume: int = 0

    def add_bar(self, bar: Bar) -> None:
        self._bars.append(bar)
        typical = (bar.high + bar.low + bar.close) / 3.0
        self._vwap_num += typical * bar.volume
        self._vwap_den += bar.volume
        if bar.high > self._session_high:
            self._session_high = bar.high
        if bar.low < self._session_low:
            self._session_low = bar.low
        self._cum_volume += bar.volume

    def recent_bars(self, n: int) -> list[Bar]:
        """Return the most recent ``n`` bars (oldest first). May be shorter
        than ``n`` early in the session."""
        if n <= 0:
            return []
        return list(self._bars)[-n:]

    def session_vwap(self) -> float | None:
        if self._vwap_den == 0:
            return None
        return self._vwap_num / self._vwap_den

    def session_high(self) -> float:
        if not self._bars:
            return 0.0
        return self._session_high

    def session_low(self) -> float:
        if not self._bars:
            return 0.0
        return self._session_low

    def cumulative_volume(self) -> int:
        return self._cum_volume

    def bar_count(self) -> int:
        return len(self._bars)

    def last_bar(self) -> Bar | None:
        if not self._bars:
            return None
        return self._bars[-1]
