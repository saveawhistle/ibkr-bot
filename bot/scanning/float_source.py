"""Chained float-share data source: yfinance primary, Finnhub shareOutstanding fallback.

Finnhub's free-tier ``/stock/profile2`` does not expose true free-float, only
shares outstanding. Using outstanding as a low-float proxy errs in the wrong
direction (it's always ≥ float, so high-float names slip through the filter).
This module tries yfinance's ``Ticker.info['floatShares']`` first, then falls
back to Finnhub outstanding × 1e6 when yfinance punts. Results are cached per
symbol for 24 hours — float is a slowly-changing fundamental, not intraday data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
import yfinance  # type: ignore[import-untyped]

from bot.scanning.finnhub_client import FinnhubClient

_log = structlog.get_logger("bot.scanning.float_source")

SOURCE_YFINANCE = "yfinance"
SOURCE_FINNHUB_FALLBACK = "finnhub_outstanding_fallback"

_DEFAULT_CACHE_TTL = timedelta(hours=24)


@dataclass(frozen=True)
class FloatData:
    """Resolved float information for a single symbol with provenance."""

    symbol: str
    float_shares: int
    source: str
    fetched_at: datetime


class FloatSource:
    """Resolve float shares via yfinance → Finnhub fallback, with a 24h in-memory cache."""

    def __init__(
        self,
        finnhub: FinnhubClient | None = None,
        *,
        cache_ttl: timedelta = _DEFAULT_CACHE_TTL,
        yfinance_fetcher: Any | None = None,
    ) -> None:
        """Construct a FloatSource; optionally inject a yfinance fetcher override for tests."""
        self._finnhub = finnhub
        self._cache: dict[str, FloatData] = {}
        self._cache_ttl = cache_ttl
        # _yfinance_fetcher(symbol) → int | None; defaults to the real yfinance Ticker.info lookup.
        self._yfinance_fetcher = yfinance_fetcher or _default_yfinance_fetch

    async def get_float(self, symbol: str) -> FloatData | None:
        """Return float data for ``symbol`` from cache / yfinance / Finnhub fallback, or None."""
        cached = self._cache.get(symbol)
        if cached is not None and self._is_fresh(cached):
            return cached

        yf_value = await self._try_yfinance(symbol)
        if yf_value is not None:
            data = FloatData(
                symbol=symbol,
                float_shares=yf_value,
                source=SOURCE_YFINANCE,
                fetched_at=datetime.now(UTC),
            )
            self._cache[symbol] = data
            return data

        fallback = await self._try_finnhub_fallback(symbol)
        if fallback is not None:
            data = FloatData(
                symbol=symbol,
                float_shares=fallback,
                source=SOURCE_FINNHUB_FALLBACK,
                fetched_at=datetime.now(UTC),
            )
            self._cache[symbol] = data
            return data

        _log.warning("float_source.unavailable", symbol=symbol)
        return None

    def _is_fresh(self, data: FloatData) -> bool:
        """Return True if ``data`` is still within the configured cache TTL."""
        return datetime.now(UTC) - data.fetched_at < self._cache_ttl

    async def _try_yfinance(self, symbol: str) -> int | None:
        """Attempt yfinance in a worker thread; swallow any exception as a warning."""
        try:
            value = await asyncio.to_thread(self._yfinance_fetcher, symbol)
        except Exception as exc:  # noqa: BLE001 - yfinance throws a grab bag
            _log.warning("float_source.yfinance_failed", symbol=symbol, error=str(exc))
            return None
        if value is None:
            _log.warning("float_source.yfinance_missing_field", symbol=symbol)
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            _log.warning(
                "float_source.yfinance_non_numeric",
                symbol=symbol,
                value=repr(value),
                error=str(exc),
            )
            return None

    async def _try_finnhub_fallback(self, symbol: str) -> int | None:
        """Convert Finnhub's shareOutstanding (millions) into a raw share count as a last resort."""
        if self._finnhub is None:
            return None
        try:
            profile = await self._finnhub.company_profile(symbol)
        except Exception as exc:  # noqa: BLE001 - network/limiter; same degradation policy
            _log.warning("float_source.finnhub_failed", symbol=symbol, error=str(exc))
            return None
        if profile is None or profile.share_outstanding is None:
            return None
        return int(profile.share_outstanding * 1_000_000)


def _default_yfinance_fetch(symbol: str) -> int | None:
    """Blocking yfinance call — executed inside ``asyncio.to_thread`` by ``FloatSource``."""
    ticker = yfinance.Ticker(symbol)
    info = ticker.info
    value = info.get("floatShares") if isinstance(info, dict) else None
    if value is None:
        return None
    return int(value)
