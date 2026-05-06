"""Chained float-share data source: yfinance primary, Finnhub shareOutstanding fallback.

Finnhub's free-tier ``/stock/profile2`` does not expose true free-float, only
shares outstanding. Using outstanding as a low-float proxy errs in the wrong
direction (it's always >= float, so high-float names slip through the filter).
This module tries yfinance's ``Ticker.info['floatShares']`` first, then falls
back to Finnhub outstanding x 1e6 when yfinance punts. Results are cached per
symbol for 24 hours -- float is a slowly-changing fundamental, not intraday data.

The same yfinance ``info`` payload also exposes ``averageVolume10days`` which
the rvol pillar needs for its denominator. Pulling both fields in one fetch
keeps the avg-volume lookup free relative to the existing float fetch; Finnhub
fallback returns ``avg_daily_volume=None`` (the rvol filter then drops the
ticker as ``rvol_unknown`` rather than silently passing it through).
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

# yfinance fetcher contract: (symbol) -> (float_shares | None, avg_daily_volume | None).
# Existing tests built against the pre-rvol contract (returning a bare int |
# None) keep working via :func:`_normalize_yfinance_result` below.
YFinanceMetrics = tuple[int | None, int | None]


@dataclass(frozen=True)
class FloatData:
    """Resolved float + avg-volume data for a single symbol with provenance."""

    symbol: str
    float_shares: int
    source: str
    fetched_at: datetime
    avg_daily_volume: int | None = None


class FloatSource:
    """Resolve float shares (+ avg volume) via yfinance -> Finnhub fallback, with a 24h cache."""

    def __init__(
        self,
        finnhub: FinnhubClient | None = None,
        *,
        cache_ttl: timedelta = _DEFAULT_CACHE_TTL,
        yfinance_fetcher: Any | None = None,
    ) -> None:
        """Construct a FloatSource; optionally inject a yfinance fetcher override for tests.

        ``yfinance_fetcher(symbol)`` may return either a tuple
        ``(float_shares | None, avg_daily_volume | None)`` (the rvol-aware
        contract) or a bare ``int | None`` (the pre-rvol contract). The
        bare-int form is treated as ``(value, None)`` so older tests keep
        working without modification.
        """
        self._finnhub = finnhub
        self._cache: dict[str, FloatData] = {}
        self._cache_ttl = cache_ttl
        self._yfinance_fetcher = yfinance_fetcher or _default_yfinance_fetch

    async def get_float(self, symbol: str) -> FloatData | None:
        """Return float data for ``symbol`` from cache / yfinance / Finnhub fallback, or None."""
        cached = self._cache.get(symbol)
        if cached is not None and self._is_fresh(cached):
            return cached

        yf_float, yf_avg_vol = await self._try_yfinance(symbol)
        if yf_float is not None:
            data = FloatData(
                symbol=symbol,
                float_shares=yf_float,
                source=SOURCE_YFINANCE,
                fetched_at=datetime.now(UTC),
                avg_daily_volume=yf_avg_vol,
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
                avg_daily_volume=None,
            )
            self._cache[symbol] = data
            return data

        _log.warning("float_source.unavailable", symbol=symbol)
        return None

    def _is_fresh(self, data: FloatData) -> bool:
        """Return True if ``data`` is still within the configured cache TTL."""
        return datetime.now(UTC) - data.fetched_at < self._cache_ttl

    async def _try_yfinance(self, symbol: str) -> YFinanceMetrics:
        """Attempt yfinance in a worker thread; swallow any exception as a warning."""
        try:
            raw = await asyncio.to_thread(self._yfinance_fetcher, symbol)
        except Exception as exc:  # noqa: BLE001 - yfinance throws a grab bag
            _log.warning("float_source.yfinance_failed", symbol=symbol, error=str(exc))
            return None, None
        float_value, avg_vol_value = _normalize_yfinance_result(raw)
        if float_value is None:
            _log.warning("float_source.yfinance_missing_field", symbol=symbol)
            return None, _coerce_int(avg_vol_value, symbol, field="avg_daily_volume")
        return (
            _coerce_int(float_value, symbol, field="float_shares"),
            _coerce_int(avg_vol_value, symbol, field="avg_daily_volume"),
        )

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


def _normalize_yfinance_result(raw: Any) -> YFinanceMetrics:
    """Accept either ``int | None`` (legacy) or ``(int|None, int|None)`` (rvol-aware)."""
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw
    return raw, None


def _coerce_int(value: Any, symbol: str, *, field: str) -> int | None:
    """Best-effort int conversion; returns None on failure with a structured warning."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        _log.warning(
            "float_source.yfinance_non_numeric",
            symbol=symbol,
            field=field,
            value=repr(value),
            error=str(exc),
        )
        return None


def _default_yfinance_fetch(symbol: str) -> YFinanceMetrics:
    """Blocking yfinance call -- executed inside ``asyncio.to_thread`` by ``FloatSource``.

    Pulls float and 10-day average daily volume from the same ``info`` payload.
    ``averageVolume10days`` tracks recent volume more responsively than the
    3-month ``averageVolume`` field -- the rvol pillar wants "is today
    abnormally heavy vs the recent baseline", not "vs the quarterly baseline".
    """
    ticker = yfinance.Ticker(symbol)
    info = ticker.info
    if not isinstance(info, dict):
        return None, None
    float_shares = info.get("floatShares")
    avg_volume = info.get("averageVolume10days") or info.get("averageVolume")
    return float_shares, avg_volume
