"""Tests for ``bot.scanning.float_source.FloatSource`` — chain, fallback, cache, and failure paths."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.scanning.finnhub_client import CompanyProfile
from bot.scanning.float_source import (
    SOURCE_FINNHUB_FALLBACK,
    SOURCE_YFINANCE,
    FloatSource,
)


def _fake_finnhub(share_outstanding_millions: float | None) -> MagicMock:
    """Build a FinnhubClient stub whose company_profile returns a known shareOutstanding."""
    client = MagicMock(name="FinnhubClient")
    if share_outstanding_millions is None:
        client.company_profile = AsyncMock(return_value=None)
    else:
        profile = CompanyProfile(symbol="X", shareOutstanding=share_outstanding_millions)
        client.company_profile = AsyncMock(return_value=profile)
    return client


@pytest.mark.asyncio
async def test_yfinance_happy_path_returns_float_data() -> None:
    """When yfinance returns an int, FloatSource emits a FloatData tagged ``yfinance``."""
    yf_fetcher = MagicMock(return_value=3_200_000)
    source = FloatSource(finnhub=_fake_finnhub(None), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("LOWF")
    assert data is not None
    assert data.symbol == "LOWF"
    assert data.float_shares == 3_200_000
    assert data.source == SOURCE_YFINANCE


@pytest.mark.asyncio
async def test_yfinance_raises_falls_back_to_finnhub() -> None:
    """yfinance raising any Exception should trigger the Finnhub fallback."""

    def yf_boom(symbol: str) -> int | None:
        raise RuntimeError("rate limited")

    finnhub = _fake_finnhub(share_outstanding_millions=12.0)
    source = FloatSource(finnhub=finnhub, yfinance_fetcher=yf_boom)
    data = await source.get_float("BOOMS")
    assert data is not None
    assert data.source == SOURCE_FINNHUB_FALLBACK
    # 12 million shares outstanding → 12_000_000 raw.
    assert data.float_shares == 12_000_000


@pytest.mark.asyncio
async def test_yfinance_none_falls_back_to_finnhub() -> None:
    """yfinance returning None (no floatShares field) should also trigger Finnhub fallback."""
    yf_fetcher = MagicMock(return_value=None)
    source = FloatSource(finnhub=_fake_finnhub(5.5), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("NONE1")
    assert data is not None
    assert data.source == SOURCE_FINNHUB_FALLBACK
    assert data.float_shares == 5_500_000


@pytest.mark.asyncio
async def test_both_sources_fail_returns_none() -> None:
    """yfinance None + Finnhub None → the source should return None, not raise."""
    yf_fetcher = MagicMock(return_value=None)
    source = FloatSource(finnhub=_fake_finnhub(None), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("GHOST")
    assert data is None


@pytest.mark.asyncio
async def test_cache_hit_suppresses_second_fetch() -> None:
    """A second call inside the TTL window must serve from cache — no new fetch."""
    yf_fetcher = MagicMock(return_value=7_777_777)
    source = FloatSource(
        finnhub=_fake_finnhub(None),
        yfinance_fetcher=yf_fetcher,
        cache_ttl=timedelta(hours=24),
    )
    first = await source.get_float("CACHE")
    second = await source.get_float("CACHE")
    assert first == second
    yf_fetcher.assert_called_once()  # second call was a cache hit


@pytest.mark.asyncio
async def test_cache_expires_after_ttl() -> None:
    """Expired cache entries must trigger a re-fetch on the next call."""
    yf_fetcher = MagicMock(return_value=1_000_000)
    source = FloatSource(
        finnhub=_fake_finnhub(None),
        yfinance_fetcher=yf_fetcher,
        cache_ttl=timedelta(seconds=0),  # every entry is stale by the next call
    )
    await source.get_float("STALE")
    await source.get_float("STALE")
    assert yf_fetcher.call_count == 2


@pytest.mark.asyncio
async def test_yfinance_non_numeric_value_falls_back() -> None:
    """A weird yfinance return that can't int() should trigger fallback, not raise."""
    yf_fetcher = MagicMock(return_value="not-a-number")
    source = FloatSource(finnhub=_fake_finnhub(2.0), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("WEIRD")
    assert data is not None
    assert data.source == SOURCE_FINNHUB_FALLBACK
    assert data.float_shares == 2_000_000


@pytest.mark.asyncio
async def test_yfinance_legacy_int_return_keeps_avg_volume_none() -> None:
    """Pre-rvol fetcher contract (bare int) must still parse cleanly with avg_daily_volume=None."""
    yf_fetcher = MagicMock(return_value=4_000_000)
    source = FloatSource(finnhub=_fake_finnhub(None), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("LEGACY")
    assert data is not None
    assert data.float_shares == 4_000_000
    assert data.avg_daily_volume is None


@pytest.mark.asyncio
async def test_yfinance_tuple_return_populates_avg_daily_volume() -> None:
    """Rvol-aware fetcher returns (float, avg_vol) and FloatData carries both."""
    yf_fetcher = MagicMock(return_value=(3_500_000, 850_000))
    source = FloatSource(finnhub=_fake_finnhub(None), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("RVOL1")
    assert data is not None
    assert data.float_shares == 3_500_000
    assert data.avg_daily_volume == 850_000
    assert data.source == SOURCE_YFINANCE


@pytest.mark.asyncio
async def test_yfinance_tuple_with_none_avg_volume_still_qualifies_float() -> None:
    """Float present + avg_volume missing must keep the symbol qualified (rvol filter handles unknown)."""
    yf_fetcher = MagicMock(return_value=(2_100_000, None))
    source = FloatSource(finnhub=_fake_finnhub(None), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("PARTL")
    assert data is not None
    assert data.float_shares == 2_100_000
    assert data.avg_daily_volume is None


@pytest.mark.asyncio
async def test_finnhub_fallback_carries_avg_daily_volume_none() -> None:
    """Finnhub fallback path doesn't expose avg volume; must populate None, not raise."""
    yf_fetcher = MagicMock(return_value=(None, None))
    source = FloatSource(finnhub=_fake_finnhub(8.0), yfinance_fetcher=yf_fetcher)
    data = await source.get_float("FBACK")
    assert data is not None
    assert data.source == SOURCE_FINNHUB_FALLBACK
    assert data.float_shares == 8_000_000
    assert data.avg_daily_volume is None
