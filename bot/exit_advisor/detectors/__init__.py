"""Layer 2 event detectors.

Each detector consumes finalized bars (one at a time, in order) plus the
``BarHistory`` buffer the harness maintains, and yields zero or more
events for that bar. The harness instantiates one detector per enabled
event class and routes each finalized bar through them all.
"""

from __future__ import annotations

from typing import Protocol

from bot.exit_advisor.core.events import Event
from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.replay_source import Bar


class Detector(Protocol):
    def on_bar(self, bar: Bar, history: BarHistory) -> list[Event]:  # pragma: no cover - protocol
        ...
