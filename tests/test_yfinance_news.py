"""Tests for ``bot.scanning.yfinance_news`` -- parser shapes + freshness + failure modes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bot.scanning.yfinance_news import fetch_yfinance_news


def _flat(title: str, *, age_hours: float = 1.0, **overrides: Any) -> dict[str, Any]:
    """Build a flat-shape yfinance news dict with a publish time ``age_hours`` ago."""
    publish_ts = (datetime.now(UTC) - timedelta(hours=age_hours)).timestamp()
    base: dict[str, Any] = {
        "title": title,
        "publisher": "Reuters",
        "link": "https://example.com/x",
        "providerPublishTime": publish_ts,
        "summary": "summary text",
    }
    base.update(overrides)
    return base


def _wrapped(title: str, *, age_hours: float = 1.0, **overrides: Any) -> dict[str, Any]:
    """Build a wrapped-shape yfinance news dict (``content`` envelope, ISO ``pubDate``)."""
    pub_dt = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat().replace("+00:00", "Z")
    content: dict[str, Any] = {
        "title": title,
        "pubDate": pub_dt,
        "provider": {"displayName": "Yahoo Finance"},
        "canonicalUrl": {"url": "https://example.com/y"},
        "summary": "wrapped summary",
    }
    content.update(overrides)
    return {"id": "abc123", "content": content}


@pytest.mark.asyncio
async def test_flat_shape_parses_into_news_items() -> None:
    """The pre-2024 yfinance flat-dict shape parses cleanly."""
    raw = [_flat("ERNA reports positive Phase 2 data")]
    items = await fetch_yfinance_news("ERNA", fetcher=lambda _sym: raw)
    assert len(items) == 1
    assert items[0].headline == "ERNA reports positive Phase 2 data"
    assert items[0].source == "Reuters"
    assert items[0].category == "yfinance"


@pytest.mark.asyncio
async def test_wrapped_shape_parses_into_news_items() -> None:
    """The current yfinance ``content``-envelope shape parses cleanly."""
    raw = [_wrapped("ERNA announces clinical readout")]
    items = await fetch_yfinance_news("ERNA", fetcher=lambda _sym: raw)
    assert len(items) == 1
    assert items[0].headline == "ERNA announces clinical readout"
    assert items[0].source == "Yahoo Finance"
    assert items[0].url == "https://example.com/y"


@pytest.mark.asyncio
async def test_mixed_shapes_in_one_response_both_parse() -> None:
    """A response that mixes both shapes (transitional yfinance versions) parses both."""
    raw = [_flat("flat headline"), _wrapped("wrapped headline")]
    items = await fetch_yfinance_news("MIX", fetcher=lambda _sym: raw)
    assert {item.headline for item in items} == {"flat headline", "wrapped headline"}


@pytest.mark.asyncio
async def test_items_older_than_lookback_are_filtered() -> None:
    """Stale items past ``hours_back`` are dropped to match Finnhub's contract."""
    raw = [
        _flat("fresh", age_hours=1),
        _flat("stale", age_hours=200),  # well past 96h default
    ]
    items = await fetch_yfinance_news("AAA", hours_back=96, fetcher=lambda _sym: raw)
    headlines = {item.headline for item in items}
    assert "fresh" in headlines
    assert "stale" not in headlines


@pytest.mark.asyncio
async def test_empty_yfinance_response_returns_empty_list() -> None:
    """No news → empty list (no exception, no spurious item)."""
    items = await fetch_yfinance_news("EMPTY", fetcher=lambda _sym: [])
    assert items == []


@pytest.mark.asyncio
async def test_yfinance_exception_returns_empty_list() -> None:
    """Network/yfinance internal errors degrade to no-news rather than raising."""

    def boom(_sym: str) -> list[dict[str, Any]]:
        raise RuntimeError("yfinance rate limited")

    items = await fetch_yfinance_news("RATE", fetcher=boom)
    assert items == []


@pytest.mark.asyncio
async def test_malformed_entry_dropped_silently() -> None:
    """A dict missing ``title`` or ``providerPublishTime`` is skipped, not crashing."""
    raw = [
        {"publisher": "no title here"},  # missing title
        _flat("good headline"),
        {"title": "no timestamp", "publisher": "x"},  # missing providerPublishTime
    ]
    items = await fetch_yfinance_news("MIXED", fetcher=lambda _sym: raw)
    assert [item.headline for item in items] == ["good headline"]


@pytest.mark.asyncio
async def test_non_list_response_returns_empty() -> None:
    """yfinance occasionally returns ``None`` or odd types; treat as no-news."""
    items = await fetch_yfinance_news("WEIRD", fetcher=lambda _sym: None)  # type: ignore[arg-type,return-value]
    assert items == []
