"""yfinance news fallback for symbols Finnhub's free tier doesn't cover.

Finnhub's free ``/company-news`` endpoint has patchy small-cap biotech
coverage -- ERNA on 2026-05-06 was dropped as ``no_news`` despite a
clinical readout that day because Finnhub simply hadn't indexed the PR.
yfinance's news endpoint pulls from Yahoo Finance's broader news feed
and catches most of these gaps without a paid API key.

This module is a thin, defensive wrapper:

* ``yfinance.Ticker(sym).news`` is blocking, so we wrap it in
  ``asyncio.to_thread`` for the scanner's parallel ``asyncio.gather``.
* yfinance has shipped at least three response shapes across recent
  versions (top-level dict per item, dict-with-``content`` envelope).
  The parser accepts both.
* Any exception falls through to ``[]`` with a structured warning --
  the scanner already treats empty as "no news available" and drops
  the symbol with the standard ``scanner.dropped_no_catalyst`` event.
* The freshness filter (``hours_back``) matches Finnhub's contract so
  downstream code (cache keying, prompt rendering) sees identical
  shapes regardless of source.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
import yfinance  # type: ignore[import-untyped]

from bot.scanning.finnhub_client import NewsItem

_log = structlog.get_logger("bot.scanning.yfinance_news")


async def fetch_yfinance_news(
    symbol: str,
    *,
    hours_back: int = 96,
    fetcher: Any | None = None,
) -> list[NewsItem]:
    """Fetch recent news for ``symbol`` via yfinance, filtered to last ``hours_back``.

    Returns ``[]`` on any failure (network error, parse error, missing
    fields, no results) -- never raises. ``fetcher`` is injectable for
    tests; defaults to the real :func:`_default_yfinance_news_fetch`.
    """
    fn = fetcher or _default_yfinance_news_fetch
    try:
        raw = await asyncio.to_thread(fn, symbol)
    except Exception as exc:  # noqa: BLE001 - yfinance throws a grab bag
        _log.warning("yfinance_news.fetch_failed", symbol=symbol, error=str(exc))
        return []
    if not raw:
        return []
    cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
    items: list[NewsItem] = []
    for entry in raw:
        parsed = _parse_entry(entry, symbol=symbol)
        if parsed is None:
            continue
        if parsed.datetime < cutoff:
            continue
        items.append(parsed)
    return items


def _default_yfinance_news_fetch(symbol: str) -> list[dict[str, Any]]:
    """Blocking yfinance call -- executed inside ``asyncio.to_thread``."""
    ticker = yfinance.Ticker(symbol)
    raw = ticker.news
    return raw if isinstance(raw, list) else []


def _parse_entry(entry: dict[str, Any], *, symbol: str) -> NewsItem | None:
    """Map one yfinance news dict into a :class:`NewsItem`; return None on bad shape.

    yfinance has shipped at least two shapes across recent versions:

    * Flat: ``{"title": ..., "publisher": ..., "link": ..., "providerPublishTime": <unix>, ...}``
    * Wrapped: ``{"id": "...", "content": {"title": ..., "provider": {"displayName": ...},
      "canonicalUrl": {"url": ...}, "pubDate": "<iso>", ...}}``

    The wrapped form lives under ``content``; if it's present we recurse
    into it, otherwise we read the flat keys directly. Missing required
    fields (title, datetime) drop the item silently -- a malformed entry
    is functionally identical to "no news".
    """
    if not isinstance(entry, dict):
        return None
    if "content" in entry and isinstance(entry["content"], dict):
        return _parse_wrapped(entry["content"], symbol=symbol)
    return _parse_flat(entry, symbol=symbol)


def _parse_flat(entry: dict[str, Any], *, symbol: str) -> NewsItem | None:
    """Parse the flat yfinance news shape (``title`` / ``providerPublishTime`` at top level)."""
    headline = entry.get("title")
    publish_time = entry.get("providerPublishTime")
    if not isinstance(headline, str) or not isinstance(publish_time, int | float):
        return None
    try:
        dt = datetime.fromtimestamp(float(publish_time), tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        _log.warning(
            "yfinance_news.timestamp_parse_failed",
            symbol=symbol,
            value=publish_time,
            error=str(exc),
        )
        return None
    return NewsItem(
        headline=headline,
        source=str(entry.get("publisher") or "yfinance"),
        url=str(entry.get("link") or ""),
        datetime=dt,
        summary=str(entry.get("summary") or ""),
        category="yfinance",
    )


def _parse_wrapped(content: dict[str, Any], *, symbol: str) -> NewsItem | None:
    """Parse the wrapped yfinance news shape (``content`` envelope, ISO ``pubDate``)."""
    headline = content.get("title")
    pub_date = content.get("pubDate") or content.get("displayTime")
    if not isinstance(headline, str) or not isinstance(pub_date, str):
        return None
    try:
        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
    except ValueError as exc:
        _log.warning(
            "yfinance_news.timestamp_parse_failed",
            symbol=symbol,
            value=pub_date,
            error=str(exc),
        )
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    provider = content.get("provider")
    publisher = (
        provider.get("displayName") if isinstance(provider, dict) else None
    ) or "yfinance"
    canonical = content.get("canonicalUrl")
    url = canonical.get("url") if isinstance(canonical, dict) else None
    return NewsItem(
        headline=headline,
        source=str(publisher),
        url=str(url or ""),
        datetime=dt,
        summary=str(content.get("summary") or content.get("description") or ""),
        category="yfinance",
    )
