"""Tests for rejection logging — strategy _reject and window-rejection silence."""

from __future__ import annotations

from datetime import date
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.backtest import Replayer
from bot.brokerage.ibkr_client import IBKRClient
from bot.brokerage.market_data import MarketData
from bot.strategies.base import REJECTION_EVENT
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
async def test_gap_and_go_no_catalyst_emits_single_setup_catalyst_rejection() -> None:
    """Selecting gap_and_go without --catalyst produces exactly one setup/catalyst rejection."""
    # Three flat in-window bars — momentum won't fire, gap_and_go is suppressed.
    frame = _bars([(f"2026-04-16 09:{30 + i:02d}", 10.0, 10.0, 10.0, 100) for i in range(3)])
    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=GapAndGoStrategy(),
        momentum=MomentumStrategy(),
    )
    _install_bars(replayer, frame)

    result = await replayer.replay(
        "TEST", date(2026, 4, 16), strategy_selection="gap_and_go", force_catalyst=False
    )

    catalyst_rejections = [
        r for r in result.rejections if r.strategy == "gap_and_go" and "catalyst" in r.reason
    ]
    assert len(catalyst_rejections) == 1
    assert catalyst_rejections[0].stage == "setup"
    assert catalyst_rejections[0].reason == "missing_catalyst"


def test_window_rejections_do_not_log() -> None:
    """A bar stamped outside the strategy window must short-circuit with zero log events."""
    # Gap-and-Go window is 09:30-10:00; bars at 10:30+ are outside.
    gap_bars = _bars(
        [
            ("2026-04-16 10:28", 10.0, 9.9, 10.0, 1_000),
            ("2026-04-16 10:29", 10.1, 10.0, 10.05, 1_000),
            ("2026-04-16 10:30", 10.2, 10.1, 10.15, 1_000),
        ]
    )
    gap_strategy = GapAndGoStrategy()
    with capture_logs() as captured:
        result = gap_strategy.evaluate("TEST", gap_bars)
    assert result is None
    assert [e for e in captured if e.get("event") == REJECTION_EVENT] == []

    # Momentum window is 09:30-11:30; 11:35+ is outside.
    momentum_bars = _bars(
        [(f"2026-04-16 11:{35 + i:02d}", 10.0, 9.9, 10.0, 1_000) for i in range(10)]
    )
    momentum_strategy = MomentumStrategy()
    with capture_logs() as captured:
        result = momentum_strategy.evaluate("TEST", momentum_bars)
    assert result is None
    assert [e for e in captured if e.get("event") == REJECTION_EVENT] == []


@pytest.mark.asyncio
async def test_replay_produces_rejections_when_signals_empty() -> None:
    """Flat in-window bars produce zero signals but a non-empty rejection list — the diagnostic hook."""
    # 15 flat bars inside the momentum window → never a new HOD after the first one,
    # so momentum rejects every bar at entry_trigger/not_new_hod.
    frame = _bars([(f"2026-04-16 09:{30 + i:02d}", 10.0, 10.0, 10.0, 100) for i in range(15)])
    replayer = Replayer(
        ibkr=cast("IBKRClient", _mock_ibkr()),
        market_data=cast("MarketData", MagicMock()),
        gap_and_go=GapAndGoStrategy(),
        momentum=MomentumStrategy(),
    )
    _install_bars(replayer, frame)

    result = await replayer.replay(
        "FLAT", date(2026, 4, 16), strategy_selection="momentum", force_catalyst=True
    )

    assert result.signals == []
    assert len(result.rejections) > 0
    # All rejections should carry the momentum strategy name and a recognized stage.
    assert all(r.strategy == "momentum" for r in result.rejections)
    assert all(r.stage in {"setup", "entry_trigger", "stop_calculation"} for r in result.rejections)
