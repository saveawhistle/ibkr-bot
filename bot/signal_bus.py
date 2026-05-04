"""Thin async pub/sub over ``asyncio.Queue`` — strategies put, orchestrator/notifier stream.

Phase 4a adds ``put_batch`` for per-bar co-signal deduplication: when both
strategies fire on the same ``(symbol, bar_time)``, Gap-and-Go wins over
Momentum. Single-signal bars are uncommon-to-rare but exact; batched fan-in is
the rule, so the orchestrator always goes through ``put_batch``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Final

import structlog

from bot.strategies.base import Signal

_log = structlog.get_logger("bot.signal_bus")

# Strategy name with priority when two strategies fire for the same bar.
# Phase 4a uses a single preferred name; a Phase 5+ expansion could promote
# this to a ranked list if a third strategy lands.
_PREFERRED_STRATEGY: Final[str] = "gap_and_go"


class SignalBus:
    """Bounded async queue of ``Signal`` objects with a streaming consumer API."""

    def __init__(self, maxsize: int = 256) -> None:
        """Construct the underlying queue (bounded to ``maxsize`` to surface runaway producers)."""
        self._queue: asyncio.Queue[Signal] = asyncio.Queue(maxsize=maxsize)

    async def put(self, signal: Signal) -> None:
        """Publish a signal. Blocks if the queue is full — on purpose."""
        await self._queue.put(signal)
        _log.info(
            "signal_bus.published",
            symbol=signal.symbol,
            strategy=signal.strategy,
            entry=signal.entry,
            stop=signal.stop,
            scale_out=signal.scale_out_price,
            runner_target=signal.runner_target_price,
            risk_reward=round(signal.risk_reward, 2),
        )

    async def put_batch(self, signals: list[Signal]) -> None:
        """Publish a per-bar batch, deduping co-signals before they reach the executor.

        When multiple signals share a ``(symbol, bar_time)`` key, keep the one
        with ``strategy == "gap_and_go"`` if present, otherwise the first. Every
        dropped signal emits ``signal.superseded_same_bar`` so the rejection
        ledger records the loser alongside the winner. Empty input is a no-op.
        """
        winners = _dedupe_co_signals(signals)
        for signal in winners:
            await self.put(signal)

    async def stream(self) -> AsyncIterator[Signal]:
        """Async iterator that yields signals in FIFO order until the task is cancelled."""
        while True:
            signal = await self._queue.get()
            try:
                yield signal
            finally:
                self._queue.task_done()

    def qsize(self) -> int:
        """Number of signals currently buffered (test / introspection hook)."""
        return self._queue.qsize()


def _dedupe_co_signals(signals: list[Signal]) -> list[Signal]:
    """Collapse multi-strategy hits on the same ``(symbol, bar_time)`` to one winner.

    The loser (not the winner) gets a ``signal.superseded_same_bar`` log event
    so diagnostic tooling can see which strategy was demoted. Order of
    non-duplicate signals is preserved.
    """
    groups: dict[tuple[str, str], list[Signal]] = {}
    order: list[tuple[str, str]] = []
    for signal in signals:
        key = (signal.symbol, signal.timestamp.isoformat())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(signal)

    winners: list[Signal] = []
    for key in order:
        bucket = groups[key]
        if len(bucket) == 1:
            winners.append(bucket[0])
            continue
        winner = _pick_winner(bucket)
        for loser in bucket:
            if loser is winner:
                continue
            _log.info(
                "signal.superseded_same_bar",
                symbol=loser.symbol,
                strategy=loser.strategy,
                bar_time=loser.timestamp.isoformat(),
                winning_strategy=winner.strategy,
            )
        winners.append(winner)
    return winners


def _pick_winner(bucket: list[Signal]) -> Signal:
    """Prefer ``gap_and_go``; otherwise the first signal in input order."""
    for signal in bucket:
        if signal.strategy == _PREFERRED_STRATEGY:
            return signal
    return bucket[0]
