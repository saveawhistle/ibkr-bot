"""Print-cluster detector — N+ same-side prints within rolling time window.

Aggressor-side classification: ``derive_aggressor_side`` from
:mod:`l2_events` returns ``"unknown"`` for mid-spread prints; those are
excluded from cluster classification (counted neither as buy nor sell).

Once-per-direction-crossing: an active buy-side cluster must wind down
(drop below ``min_prints`` in window on the buy side) before another
buy cluster fires. Sell side independent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from bot.exit_advisor.core.events import Event, PrintCluster
from bot.exit_advisor.market.book_state import BookState
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


@dataclass
class PrintClusterDetector:
    symbol: str
    window_seconds: float = 10.0
    min_prints: int = 5

    _recent_buys: deque[L2Print] = field(default_factory=deque, init=False)
    _recent_sells: deque[L2Print] = field(default_factory=deque, init=False)
    _buy_cluster_active: bool = field(default=False, init=False)
    _sell_cluster_active: bool = field(default=False, init=False)

    def consume(
        self, event: L2BookUpdate | L2Print, book_state: BookState
    ) -> list[Event]:
        if not isinstance(event, L2Print):
            return []
        cutoff = event.timestamp - timedelta(seconds=self.window_seconds)
        # Append the new print to its side, then trim either side's
        # window. Trim BOTH because the active cluster on the opposite
        # side may need to deactivate as its prints age out.
        if event.aggressor_side == "buy":
            self._recent_buys.append(event)
        elif event.aggressor_side == "sell":
            self._recent_sells.append(event)
        # else: unknown — excluded
        while self._recent_buys and self._recent_buys[0].timestamp < cutoff:
            self._recent_buys.popleft()
        while self._recent_sells and self._recent_sells[0].timestamp < cutoff:
            self._recent_sells.popleft()

        events: list[Event] = []
        events.extend(self._maybe_emit("buy", self._recent_buys, event.timestamp))
        events.extend(self._maybe_emit("sell", self._recent_sells, event.timestamp))
        return events

    def _maybe_emit(
        self,
        side: Literal["buy", "sell"],
        prints: deque[L2Print],
        now: datetime,
    ) -> list[Event]:
        active_attr = (
            "_buy_cluster_active" if side == "buy" else "_sell_cluster_active"
        )
        active = getattr(self, active_attr)
        if len(prints) >= self.min_prints:
            if active:
                return []
            setattr(self, active_attr, True)
            return [
                PrintCluster(
                    timestamp=now,
                    symbol=self.symbol,
                    side=side,
                    print_count=len(prints),
                    total_volume=sum(p.size for p in prints),
                    window_seconds=self.window_seconds,
                )
            ]
        else:
            setattr(self, active_attr, False)
            return []
