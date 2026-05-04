"""Tests for ``bot.backtest.Replayer`` — synthetic bars, empty input, and the slicing invariant."""

from __future__ import annotations

from datetime import date, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from bot.backtest import Replayer
from bot.brokerage.ibkr_client import IBKRClient
from bot.brokerage.market_data import MarketData
from bot.strategies.base import Signal, Strategy
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy


def _bars(frames: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build a NY-tz 1-min bar DataFrame from (timestamp, high, low, close, volume) tuples."""
    idx = pd.to_datetime([f[0] for f in frames]).tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "open": [f[3] for f in frames],
            "high": [f[1] for f in frames],
            "low": [f[2] for f in frames],
            "close": [f[3] for f in frames],
            "volume": [f[4] for f in frames],
            "vwap": [f[3] for f in frames],
        },
        index=idx,
    )


def _mock_ibkr() -> MagicMock:
    """Build a mock IBKRClient whose qualify_stock resolves without hitting TWS."""
    ibkr = MagicMock(name="IBKRClient")
    ibkr.qualify_stock = AsyncMock(return_value=MagicMock(symbol="TEST"))
    return ibkr


def _install_bars(replayer: Replayer, frame: pd.DataFrame) -> None:
    """Patch the Replayer's bar-fetch to return a fixed DataFrame (no IBKR round-trip)."""

    async def _fetch(_contract: object, _target_date: date) -> pd.DataFrame:
        return frame

    replayer._fetch_bars = _fetch  # type: ignore[assignment,method-assign]


@pytest.mark.asyncio
async def test_momentum_fires_on_hand_crafted_bull_flag() -> None:
    """A synthetic bull-flag + HOD-break sequence must produce exactly one momentum signal."""
    frame = _bars(
        [
            # impulse
            ("2026-04-16 09:30", 10.3, 10.0, 10.2, 1_000),
            ("2026-04-16 09:31", 10.5, 10.3, 10.45, 1_000),
            ("2026-04-16 09:32", 10.5, 10.3, 10.5, 1_000),
            # flag
            ("2026-04-16 09:33", 10.4, 10.3, 10.4, 1_000),
            ("2026-04-16 09:34", 10.4, 10.25, 10.3, 1_000),
            ("2026-04-16 09:35", 10.35, 10.25, 10.3, 1_000),
            ("2026-04-16 09:36", 10.35, 10.3, 10.32, 1_000),
            ("2026-04-16 09:37", 10.32, 10.3, 10.32, 1_000),
            ("2026-04-16 09:38", 10.4, 10.3, 10.35, 1_000),
            # breakout bar
            ("2026-04-16 09:39", 10.6, 10.35, 10.6, 1_000),
        ]
    )

    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=GapAndGoStrategy(),
        momentum=MomentumStrategy(flag_max_pullback_pct=5.0),
    )
    _install_bars(replayer, frame)

    result = await replayer.replay("TEST", date(2026, 4, 16), strategy_selection="momentum")
    assert len(result.signals) == 1
    hit = result.signals[0]
    assert hit.strategy == "momentum"
    assert hit.timestamp.strftime("%H:%M") == "09:39"


@pytest.mark.asyncio
async def test_flat_bars_produce_no_signals() -> None:
    """A completely flat day (no HOD breaks, no pullbacks) must produce zero signals."""
    frame = _bars([(f"2026-04-16 09:{30 + i:02d}", 10.0, 10.0, 10.0, 100) for i in range(30)])
    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=GapAndGoStrategy(),
        momentum=MomentumStrategy(),
    )
    _install_bars(replayer, frame)

    result = await replayer.replay(
        "FLAT", date(2026, 4, 16), strategy_selection="both", force_catalyst=True
    )
    assert result.signals == []


@pytest.mark.asyncio
async def test_no_future_peeking_slice_invariant() -> None:
    """At iteration N, the strategy must see exactly N+1 bars (0..N inclusive)."""
    frame = _bars(
        [
            (f"2026-04-16 09:{30 + i:02d}", 10.0 + i * 0.1, 10.0, 10.0 + i * 0.1, 100)
            for i in range(5)
        ]
    )

    class _RecordingStrategy(Strategy):
        """Records the length of every ``bars`` DataFrame it is passed."""

        name = "recorder"

        def __init__(self) -> None:
            super().__init__()
            self.seen_lengths: list[int] = []

        def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
            self.seen_lengths.append(len(bars))
            return None

    recorder = _RecordingStrategy()
    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=GapAndGoStrategy(),
        momentum=MomentumStrategy(),
    )
    _install_bars(replayer, frame)
    # Swap momentum for the recorder so we see every call the Replayer makes.
    replayer._momentum = recorder  # type: ignore[assignment]

    await replayer.replay("TEST", date(2026, 4, 16), strategy_selection="momentum")
    assert recorder.seen_lengths == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_gap_and_go_suppressed_without_catalyst_override() -> None:
    """Without ``--catalyst``, Gap-and-Go must not run even when selected."""

    class _SpyStrategy(Strategy):
        name = "gap_and_go"

        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
            self.call_count += 1
            return Signal(
                symbol=symbol,
                strategy=self.name,
                entry=10.0,
                stop=9.0,
                scale_out_price=12.0,
                runner_target_price=13.0,
                timestamp=datetime(2026, 4, 16, 9, 31),
            )

    spy = _SpyStrategy()
    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=cast("GapAndGoStrategy", spy),
        momentum=MomentumStrategy(),
    )
    _install_bars(
        replayer,
        _bars([(f"2026-04-16 09:{30 + i:02d}", 10.0, 10.0, 10.0, 100) for i in range(3)]),
    )
    result = await replayer.replay(
        "TEST", date(2026, 4, 16), strategy_selection="both", force_catalyst=False
    )
    assert spy.call_count == 0
    assert result.signals == []
    assert result.context.catalyst is None
