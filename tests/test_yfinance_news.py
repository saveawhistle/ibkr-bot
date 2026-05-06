"""Tests for ``bot.scanning.yfinance_news`` -- parser shapes + freshness + failure modes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bot.scanning.yfinance_news import fetch_yfinance_news


def _flat(
    title: str,
    *,
    age_hours: float = 1.0,
    related: list[str] | None = ("AAA",),  # type: ignore[assignment]
    **overrides: Any,
) -> dict[str, Any]:
    """Build a flat-shape yfinance Search news dict with a publish time ``age_hours`` ago.

    ``related`` defaults to ``("AAA",)`` so tests using the bare ``_flat``
    helper survive the relatedTickers filter when they fetch for symbol
    ``"AAA"``. Tests targeting the filter itself override this explicitly.
    """
    publish_ts = (datetime.now(UTC) - timedelta(hours=age_hours)).timestamp()
    base: dict[str, Any] = {
        "title": title,
        "publisher": "Reuters",
        "link": "https://example.com/x",
        "providerPublishTime": publish_ts,
        "summary": "summary text",
        "relatedTickers": list(related) if related is not None else None,
    }
    if related is None:
        base.pop("relatedTickers")
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
    """The Search flat-dict shape parses cleanly."""
    raw = [_flat("ERNA reports positive Phase 2 data", related=["ERNA"])]
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
    """A response that mixes both shapes (Search vs Ticker.news) parses both.

    Wrapped-shape entries lack ``relatedTickers`` and are passed through
    by the filter (the filter only enforces strict relatedTickers on the
    flat Search shape where the field is reliable).
    """
    raw = [_flat("flat headline", related=["MIX"]), _wrapped("wrapped headline")]
    items = await fetch_yfinance_news("MIX", fetcher=lambda _sym: raw)
    assert {item.headline for item in items} == {"flat headline", "wrapped headline"}


@pytest.mark.asyncio
async def test_items_older_than_lookback_are_filtered() -> None:
    """Stale items past ``hours_back`` are dropped to match Finnhub's contract."""
    raw = [
        _flat("fresh", age_hours=1, related=["AAA"]),
        _flat("stale", age_hours=200, related=["AAA"]),  # well past 96h default
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
        {"publisher": "no title here", "relatedTickers": ["MIXED"]},  # missing title
        _flat("good headline", related=["MIXED"]),
        # missing providerPublishTime
        {"title": "no timestamp", "publisher": "x", "relatedTickers": ["MIXED"]},
    ]
    items = await fetch_yfinance_news("MIXED", fetcher=lambda _sym: raw)
    assert [item.headline for item in items] == ["good headline"]


@pytest.mark.asyncio
async def test_filter_drops_items_when_symbol_not_in_related_tickers() -> None:
    """Search-tokenization noise (e.g. ``Sanford Burnham`` matching ``ERNA``) is dropped.

    The actual ERNA reproducer: yfinance.Search('ERNA') returns the real
    Ernexa clinical PR (relatedTickers=['ERNA','ERNAW']) AND incidental
    items like 'Sanford Burnham Prebys ... receives $5 million from
    Qualcomm co-founder...' (relatedTickers=['QCOM']) that contain the
    substring "erna" somewhere. Only the former should reach the LLM.
    """
    raw = [
        _flat("ERNA-101 achieves complete tumor elimination", related=["ERNA", "ERNAW"]),
        _flat("Sanford Burnham receives $5M from Qualcomm", related=["QCOM"]),
        _flat("Sail Biomedicines appoints CMO", related=None),  # no relatedTickers at all
    ]
    items = await fetch_yfinance_news("ERNA", fetcher=lambda _sym: raw)
    assert [item.headline for item in items] == [
        "ERNA-101 achieves complete tumor elimination"
    ]


@pytest.mark.asyncio
async def test_filter_keeps_item_when_symbol_is_one_of_many_related() -> None:
    """Items naming the symbol alongside others (e.g. with index/warrant) still pass."""
    raw = [_flat("Ernexa announces 1-for-25 reverse split", related=["ERNA", "^IXIC", "ERNAW"])]
    items = await fetch_yfinance_news("ERNA", fetcher=lambda _sym: raw)
    assert len(items) == 1
    assert items[0].headline == "Ernexa announces 1-for-25 reverse split"


@pytest.mark.asyncio
async def test_non_list_response_returns_empty() -> None:
    """yfinance occasionally returns ``None`` or odd types; treat as no-news."""
    items = await fetch_yfinance_news("WEIRD", fetcher=lambda _sym: None)  # type: ignore[arg-type,return-value]
    assert items == []
