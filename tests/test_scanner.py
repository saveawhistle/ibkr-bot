"""Tests for ``bot.scanning.scanner.IBKRScanner`` with mocked IBKR + mocked Finnhub + mocked FloatSource."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from bot.brokerage.ibkr_client import SubscriptionRegistry
from bot.config import DataSourcesSettings, Settings, UniverseConfig
from bot.scanning.finnhub_client import NewsItem
from bot.scanning.float_source import SOURCE_FINNHUB_FALLBACK, SOURCE_YFINANCE, FloatSource
from bot.scanning.scanner import IBKRScanner


def _fake_scan_row(symbol: str) -> SimpleNamespace:
    """Build a duck-typed stand-in for an ib_async ScanData row."""
    return SimpleNamespace(contractDetails=SimpleNamespace(contract=SimpleNamespace(symbol=symbol)))


def _settings(float_max: int = 20_000_000) -> Settings:
    """Build a Settings instance with a known UniverseConfig for assertions."""
    return Settings(universe=UniverseConfig(float_max=float_max))


def _news(headline: str, summary: str = "") -> NewsItem:
    """Build a NewsItem for scanner test fixtures.

    Uses ``datetime.now(UTC)`` so the classifier's Phase 5.1 freshness
    filter (``news_max_age_hours_for_classify``) doesn't drop the fixture
    as stale when the test is run at some future date.
    """
    return NewsItem(
        headline=headline,
        source="test",
        url="https://example.com/x",
        datetime=datetime.now(UTC),
        summary=summary,
        category="company",
    )


def _mock_ibkr(symbols: list[str]) -> MagicMock:
    """Return a MagicMock IBKRClient whose scanner call yields rows for the given symbols.

    Default ``reqHistoricalDataAsync`` returns an empty list so the Phase 5.3
    enrichment path lands on the ``no_bars_returned`` branch and leaves
    ``price``/``change_pct``/``volume`` as ``None`` — the same pre-5.3
    behaviour existing tests were written against.
    """
    ibkr = MagicMock(name="IBKRClient")
    ibkr.ib = MagicMock(name="IB")
    ibkr.ib.reqScannerDataAsync = AsyncMock(return_value=[_fake_scan_row(s) for s in symbols])
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
    ibkr.ib.cancelScannerSubscription = MagicMock()
    ibkr.ib.cancelHistoricalData = MagicMock()
    # Attach a real SubscriptionRegistry — scanner.py calls async
    # register/unregister on it; a plain MagicMock would return non-awaitables.
    ibkr.subscriptions = SubscriptionRegistry()
    return ibkr


def _fake_daily_bar(close: float, volume: float) -> SimpleNamespace:
    """Build a duck-typed stand-in for an ib_async BarData row (daily bar)."""
    return SimpleNamespace(close=close, volume=volume)


def _mock_finnhub(news: dict[str, list[NewsItem]]) -> MagicMock:
    """Return a MagicMock FinnhubClient whose company_news is backed by the provided map."""
    finnhub = MagicMock(name="FinnhubClient")

    async def get_news(symbol: str, hours_back: int = 24) -> list[NewsItem]:
        return news.get(symbol, [])

    finnhub.company_news = AsyncMock(side_effect=get_news)
    # company_profile isn't used by the scanner in Phase 3 (FloatSource owns that lookup)
    # but keep a stub so accidental calls don't explode.
    finnhub.company_profile = AsyncMock(return_value=None)
    return finnhub


def _float_source(yf_map: dict[str, int | None]) -> FloatSource:
    """Build a FloatSource whose yfinance fetcher is a deterministic dict lookup."""

    def fetch(symbol: str) -> int | None:
        return yf_map.get(symbol)

    return FloatSource(finnhub=None, yfinance_fetcher=fetch)


@pytest.mark.asyncio
async def test_high_float_symbols_are_dropped() -> None:
    """A symbol whose float exceeds ``universe.float_max`` must not appear in the watchlist.

    Phase 6.11: LOWF gets a green-catalyst news item so it survives the
    ``no_catalyst`` drop filter. The test's invariant is float-based
    filtering; catalyst is incidental here.
    """
    settings = _settings(float_max=10_000_000)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["LOWF", "HIGHF"]),
        finnhub=_mock_finnhub(news={"LOWF": [_news("LOWF tops estimates")]}),
        settings=settings,
        float_source=_float_source({"LOWF": 3_000_000, "HIGHF": 50_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    symbols = {h.symbol for h in hits}
    assert "LOWF" in symbols
    assert "HIGHF" not in symbols


@pytest.mark.asyncio
async def test_float_unknown_symbol_dropped() -> None:
    """Phase 6.3: unknown-float symbols are dropped entirely, not flagged and kept.

    Leveraged ETFs like UVIX/MSTU typically return None from both yfinance
    and Finnhub. Previously we kept them on the watchlist with
    ``float_unknown`` tagged — they'd eat a bar subscription and never
    produce a signal. Now we drop them at the filter step and log
    ``scanner.dropped_float_unknown`` for session-review visibility.
    """
    from structlog.testing import capture_logs

    structlog.reset_defaults()
    settings = _settings(float_max=10_000_000)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["UVIX"]),
        finnhub=_mock_finnhub(news={}),
        settings=settings,
        float_source=_float_source({"UVIX": None}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == [], "unknown-float symbol must not appear in scan results"
    drops = [e for e in captured if e.get("event") == "scanner.dropped_float_unknown"]
    assert drops, "expected scanner.dropped_float_unknown event"
    assert drops[0]["symbol"] == "UVIX"
    assert drops[0]["sources_attempted"] == ["yfinance", "finnhub"]


@pytest.mark.asyncio
async def test_float_known_symbols_pass_through() -> None:
    """Sanity regression — symbols with valid within-range float still pass.

    Phase 6.11: OKAY needs a catalyst news item to survive the
    ``no_catalyst`` drop; the test's invariant is float bookkeeping.
    """
    settings = _settings(float_max=20_000_000)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["OKAY"]),
        finnhub=_mock_finnhub(news={"OKAY": [_news("OKAY tops estimates")]}),
        settings=settings,
        float_source=_float_source({"OKAY": 5_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].symbol == "OKAY"
    assert hits[0].float_shares == 5_000_000
    assert "float_unknown" not in hits[0].reasons


@pytest.mark.asyncio
async def test_float_high_dropped_differently() -> None:
    """High-float drop fires ``dropped_high_float``, not ``dropped_float_unknown``."""
    from structlog.testing import capture_logs

    structlog.reset_defaults()
    settings = _settings(float_max=20_000_000)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["BIGF"]),
        finnhub=_mock_finnhub(news={}),
        settings=settings,
        float_source=_float_source({"BIGF": 100_000_000}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    high = [e for e in captured if e.get("event") == "scanner.dropped_high_float"]
    unknown = [e for e in captured if e.get("event") == "scanner.dropped_float_unknown"]
    assert len(high) == 1 and high[0]["symbol"] == "BIGF"
    assert unknown == [], "high-float drop must not emit float_unknown event"


@pytest.mark.asyncio
async def test_black_list_news_overrides_green_list() -> None:
    """A headline with both green and black-list phrases yields no catalyst → dropped.

    Phase 6.11: a no-catalyst hit never makes it into the returned list;
    instead we emit ``scanner.dropped_no_catalyst``. The test now asserts
    both the empty hits list and the drop event.
    """
    from structlog.testing import capture_logs

    structlog.reset_defaults()
    settings = _settings()
    news = {
        "MIXED": [
            _news(
                headline="MIXED wins FDA approval",
                summary="Announced alongside a reverse split to maintain listing.",
            ),
        ]
    }
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["MIXED"]),
        finnhub=_mock_finnhub(news=news),
        settings=settings,
        float_source=_float_source({"MIXED": 5_000_000}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    drops = [e for e in captured if e.get("event") == "scanner.dropped_no_catalyst"]
    assert len(drops) == 1
    assert drops[0]["symbol"] == "MIXED"


@pytest.mark.asyncio
async def test_no_catalyst_symbols_dropped_with_log() -> None:
    """Phase 6.11: symbols whose classifier returns None are dropped from the watchlist.

    the 5-pillar rule treats news as mandatory. Subscribing 1-min
    bars for a no-news symbol burns an IBKR slot that the next 5-min
    rescan's catalyst-bearing candidate could use. This test drives a
    symbol with zero news through the pipeline and asserts:

    1. The returned hits list is empty (not surfaced with
       ``reasons=["no_catalyst"]`` as it was pre-6.11).
    2. ``scanner.dropped_no_catalyst`` fires with the symbol and the
       resolved float metadata for forensic grep.
    """
    from structlog.testing import capture_logs

    structlog.reset_defaults()
    settings = _settings()
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["NONEWS"]),
        finnhub=_mock_finnhub(news={}),
        settings=settings,
        float_source=_float_source({"NONEWS": 3_200_000}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    drops = [e for e in captured if e.get("event") == "scanner.dropped_no_catalyst"]
    assert len(drops) == 1
    assert drops[0]["symbol"] == "NONEWS"
    # Float metadata carried through so post-session review can tell
    # a yfinance-sourced small float from a Finnhub-fallback one.
    assert drops[0]["float_shares"] == 3_200_000
    assert drops[0]["float_source"] == SOURCE_YFINANCE


@pytest.mark.asyncio
async def test_manual_override_survives_no_catalyst_drop() -> None:
    """Phase 6.11 + Phase 6.8: an injected override attaches a catalyst BEFORE the drop.

    A symbol with zero Finnhub news would normally be dropped by Phase
    6.11. But if an active CatalystOverride exists for the symbol and
    the testing gate is on, the override injects a category and the
    drop filter sees ``catalyst != None`` → symbol survives.
    """
    from bot.scanning.catalyst_overrides import CatalystOverride  # noqa: PLC0415

    override = CatalystOverride(
        symbol="NONEWS",
        category="contract_or_m&a",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        note="operator injection",
        injected_at=datetime.now(UTC),
        injected_by="cli",
    )

    def _loader(*, now: datetime) -> dict[str, CatalystOverride]:
        return {"NONEWS": override}

    import bot.scanning.scanner  # noqa: PLC0415

    original = bot.scanning.scanner.load_active_overrides_map
    bot.scanning.scanner.load_active_overrides_map = _loader
    try:
        s = _settings()
        settings = s.model_copy(
            update={"testing": s.testing.model_copy(update={"allow_catalyst_overrides": True})}
        )
        scanner = IBKRScanner(
            ibkr=_mock_ibkr(["NONEWS"]),
            finnhub=_mock_finnhub(news={}),
            settings=settings,
            float_source=_float_source({"NONEWS": 3_200_000}),
        )
        hits = await scanner.scan_top_gappers()
    finally:
        bot.scanning.scanner.load_active_overrides_map = original

    assert len(hits) == 1
    assert hits[0].symbol == "NONEWS"
    assert hits[0].catalyst == "contract_or_m&a"


@pytest.mark.asyncio
async def test_green_list_news_populates_catalyst() -> None:
    """A clean green-list headline should populate the catalyst field and drop ``no_catalyst``."""
    settings = _settings()
    news = {"WINNR": [_news(headline="WINNR tops estimates in Q1")]}
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["WINNR"]),
        finnhub=_mock_finnhub(news=news),
        settings=settings,
        float_source=_float_source({"WINNR": 4_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].catalyst == "earnings_beat"
    assert "no_catalyst" not in hits[0].reasons


@pytest.mark.asyncio
async def test_empty_scan_returns_empty_list() -> None:
    """No IBKR hits → empty watchlist, no float / news calls attempted."""
    scanner = IBKRScanner(
        ibkr=_mock_ibkr([]),
        finnhub=_mock_finnhub(news={}),
        settings=_settings(),
        float_source=_float_source({}),
    )
    hits = await scanner.scan_top_gappers()
    assert hits == []


@pytest.mark.asyncio
async def test_mixed_float_sources_roundtrip() -> None:
    """Mixed results: yfinance hit, Finnhub fallback, and None — all three must round-trip."""
    settings = _settings(float_max=20_000_000)

    # yfinance covers YFSYM; raises for FBSYM (forcing fallback); returns None for GHOST.
    def yf_fetch(symbol: str) -> int | None:
        if symbol == "YFSYM":
            return 4_200_000
        if symbol == "FBSYM":
            raise RuntimeError("yfinance flaky")
        return None  # GHOST

    # Finnhub has shareOutstanding only for FBSYM; None for YFSYM (unused) and GHOST.
    finnhub = MagicMock(name="FinnhubClient")

    async def get_profile(symbol: str) -> object:
        from bot.scanning.finnhub_client import CompanyProfile

        if symbol == "FBSYM":
            return CompanyProfile(symbol="FBSYM", shareOutstanding=9.0)
        return None

    async def get_news(symbol: str, hours_back: int = 24) -> list[NewsItem]:
        # Phase 6.11: both surviving symbols need a green-catalyst item
        # to avoid the ``no_catalyst`` drop; this test's invariant is
        # float bookkeeping, not catalyst semantics.
        if symbol in ("YFSYM", "FBSYM"):
            return [_news(f"{symbol} tops estimates")]
        return []

    finnhub.company_profile = AsyncMock(side_effect=get_profile)
    finnhub.company_news = AsyncMock(side_effect=get_news)

    float_source = FloatSource(finnhub=finnhub, yfinance_fetcher=yf_fetch)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["YFSYM", "FBSYM", "GHOST"]),
        finnhub=finnhub,
        settings=settings,
        float_source=float_source,
    )

    hits = {h.symbol: h for h in await scanner.scan_top_gappers()}
    # Phase 6.3: GHOST (unknown float) is dropped at the filter step; the
    # yfinance-sourced and Finnhub-fallback-sourced rows still pass through.
    assert set(hits) == {"YFSYM", "FBSYM"}

    assert hits["YFSYM"].float_shares == 4_200_000
    assert hits["YFSYM"].float_source == SOURCE_YFINANCE

    assert hits["FBSYM"].float_shares == 9_000_000
    assert hits["FBSYM"].float_source == SOURCE_FINNHUB_FALLBACK


# ---------- Phase 5.1 news window plumbing ---------- #


@pytest.mark.asyncio
async def test_scanner_passes_configured_lookback_to_finnhub() -> None:
    """Scanner forwards ``data_sources.news_lookback_hours`` to ``company_news``."""
    settings = Settings(
        universe=UniverseConfig(float_max=20_000_000),
        data_sources=DataSourcesSettings(
            news_lookback_hours=96,
            news_max_age_hours_for_classify=72,
        ),
    )
    finnhub = _mock_finnhub(news={"ACME": [_news("ACME tops estimates")]})
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=finnhub,
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
    )
    await scanner.scan_top_gappers()
    finnhub.company_news.assert_awaited_once_with("ACME", hours_back=96)


@pytest.mark.asyncio
async def test_scanner_classifier_filters_stale_fetched_news() -> None:
    """Stale news → classifier returns None → Phase 6.11 no-catalyst drop."""
    from structlog.testing import capture_logs

    structlog.reset_defaults()
    stale_when = datetime.now(UTC) - timedelta(hours=80)  # outside 72h window
    stale_item = NewsItem(
        headline="ACME tops estimates",
        source="test",
        url="https://example.com/x",
        datetime=stale_when,
        summary="",
        category="company",
    )
    settings = Settings(
        universe=UniverseConfig(float_max=20_000_000),
        data_sources=DataSourcesSettings(
            news_lookback_hours=96,
            news_max_age_hours_for_classify=72,
        ),
    )
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub(news={"ACME": [stale_item]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    drops = [e for e in captured if e.get("event") == "scanner.dropped_no_catalyst"]
    assert len(drops) == 1
    assert drops[0]["symbol"] == "ACME"


@pytest.mark.asyncio
async def test_scanner_keeps_fresh_news_within_classifier_window() -> None:
    """A green-list item inside ``news_max_age_hours_for_classify`` still classifies."""
    fresh_when = datetime.now(UTC) - timedelta(hours=2)
    fresh_item = NewsItem(
        headline="ACME tops estimates",
        source="test",
        url="https://example.com/x",
        datetime=fresh_when,
        summary="",
        category="company",
    )
    settings = Settings(
        universe=UniverseConfig(float_max=20_000_000),
        data_sources=DataSourcesSettings(
            news_lookback_hours=96,
            news_max_age_hours_for_classify=72,
        ),
    )
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub(news={"ACME": [fresh_item]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].catalyst == "earnings_beat"


# ---------- Phase 5.3 market-data enrichment ---------- #


@pytest.mark.asyncio
async def test_enrichment_populates_hit_fields() -> None:
    """Enrichment fills ``price``/``change_pct``/``volume`` from the 2-bar daily pull."""
    ibkr = _mock_ibkr(["ACME"])
    # [yesterday=10.00, today=12.50] → +25% on 500k volume
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[
            _fake_daily_bar(close=10.0, volume=0),
            _fake_daily_bar(close=12.5, volume=500_000),
        ]
    )
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"ACME": [_news("ACME tops estimates")]}),
        settings=_settings(),
        float_source=_float_source({"ACME": 5_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    hit = hits[0]
    assert hit.price == pytest.approx(12.5)
    assert hit.change_pct == pytest.approx(25.0)
    assert hit.volume == 500_000


@pytest.mark.asyncio
async def test_enrichment_timeout_leaves_fields_none() -> None:
    """If the IBKR call doesn't return in time, the hit still renders with dashes."""
    ibkr = _mock_ibkr(["SLOWP"])

    async def never_returns(*_args: object, **_kwargs: object) -> list[object]:
        await asyncio.sleep(5.0)
        return []

    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=never_returns)
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"SLOWP": [_news("SLOWP tops estimates")]}),
        settings=_settings(),
        float_source=_float_source({"SLOWP": 5_000_000}),
        enrichment_timeout_seconds=0.05,
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    hit = hits[0]
    assert hit.price is None
    assert hit.change_pct is None
    assert hit.volume is None
    assert any(row.get("event") == "scanner.enrichment_timeout" for row in logs)


@pytest.mark.asyncio
async def test_enrichment_failure_does_not_block_watchlist() -> None:
    """One symbol raising must not prevent others from appearing in the watchlist."""
    ibkr = _mock_ibkr(["GOOD", "BADXX"])

    async def mixed(contract: object, **_kwargs: object) -> list[object]:
        if getattr(contract, "symbol", "") == "BADXX":
            raise RuntimeError("IBKR error 162: historical data service error")
        return [
            _fake_daily_bar(close=4.0, volume=0),
            _fake_daily_bar(close=5.0, volume=1_000_000),
        ]

    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=mixed)
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(
            news={
                "GOOD": [_news("GOOD tops estimates")],
                "BADXX": [_news("BADXX tops estimates")],
            }
        ),
        settings=_settings(),
        float_source=_float_source({"GOOD": 5_000_000, "BADXX": 5_000_000}),
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    assert len(hits) == 2
    by_sym = {h.symbol: h for h in hits}
    assert by_sym["GOOD"].price == pytest.approx(5.0)
    assert by_sym["GOOD"].change_pct == pytest.approx(25.0)
    assert by_sym["BADXX"].price is None
    assert by_sym["BADXX"].change_pct is None
    assert any(
        row.get("event") == "scanner.enrichment_failed" and row.get("symbol") == "BADXX"
        for row in logs
    )


@pytest.mark.asyncio
async def test_enrichment_runs_only_after_filters() -> None:
    """High-float (dropped) and no-catalyst hits must not consume an IBKR round-trip.

    Phase 6.11: NOCAT is now dropped from hits entirely (the pre-6.11
    behaviour surfaced it with ``reasons=["no_catalyst"]`` but didn't
    enrich it — the invariant here, "enrichment round-trip only for
    catalyst-bearing symbols", is unchanged and strengthened: NOCAT
    never reaches enrichment AND never reaches the returned hits.
    """
    ibkr = _mock_ibkr(["HIGHF", "NOCAT", "GOODX"])
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[
            _fake_daily_bar(close=3.0, volume=0),
            _fake_daily_bar(close=4.5, volume=250_000),
        ]
    )
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(
            news={
                "NOCAT": [],  # no catalyst → dropped in Phase 6.11
                "GOODX": [_news("GOODX tops estimates")],
                # HIGHF: float filter drops it before news is even fetched
            }
        ),
        settings=_settings(float_max=10_000_000),
        float_source=_float_source({"HIGHF": 50_000_000, "NOCAT": 5_000_000, "GOODX": 5_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    # HIGHF dropped by float filter, NOCAT dropped by Phase 6.11, GOODX survives.
    symbols = {h.symbol for h in hits}
    assert symbols == {"GOODX"}
    # Only GOODX triggers enrichment — invariant from Phase 5.3 holds.
    assert ibkr.ib.reqHistoricalDataAsync.await_count == 1
    called_contract = ibkr.ib.reqHistoricalDataAsync.await_args.args[0]
    assert getattr(called_contract, "symbol", None) == "GOODX"


@pytest.mark.asyncio
async def test_enrichment_parallelizes() -> None:
    """Three symbols × 0.3 s delay each must finish in < N × delay (proves asyncio.gather)."""
    import time

    ibkr = _mock_ibkr(["AAA", "BBB", "CCC"])

    async def slow(*_args: object, **_kwargs: object) -> list[object]:
        await asyncio.sleep(0.3)
        return [
            _fake_daily_bar(close=9.0, volume=0),
            _fake_daily_bar(close=10.0, volume=100_000),
        ]

    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=slow)
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(
            news={s: [_news(f"{s} tops estimates")] for s in ("AAA", "BBB", "CCC")}
        ),
        settings=_settings(),
        float_source=_float_source({s: 5_000_000 for s in ("AAA", "BBB", "CCC")}),
    )
    start = time.perf_counter()
    hits = await scanner.scan_top_gappers()
    elapsed = time.perf_counter() - start
    assert len(hits) == 3
    # Serial would be ~0.9 s; parallel should be ~0.3 s. Allow generous headroom.
    assert elapsed < 0.7, f"enrichment not parallelized: took {elapsed:.2f}s for 3 × 0.3s"


# ---------- Phase 5.4 subscription-lifecycle hygiene ---------- #


@pytest.mark.asyncio
async def test_scanner_cancels_scanner_subscription_after_scan() -> None:
    """The TOP_PERC_GAIN subscription must be cancelled after reqScannerDataAsync returns."""
    ibkr = _mock_ibkr(["ACME"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={}),
        settings=_settings(),
        float_source=_float_source({"ACME": 5_000_000}),
    )
    await scanner.scan_top_gappers()
    ibkr.ib.cancelScannerSubscription.assert_called_once()
    # Registry must be empty after the sweep.
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_fetch_quote_timeout_cancels_on_wire() -> None:
    """On timeout the enrichment must call cancelHistoricalData (not just cancel the coroutine)."""
    ibkr = _mock_ibkr(["SLOW"])
    # The ib_async shape: reqHistoricalDataAsync is awaitable but the
    # underlying IB resolves the future to the partial BarDataList. Simulate
    # by returning a list after the timeout; when our task-based cancel path
    # races ``asyncio.wait_for``, the task completes with that list and our
    # cancel-on-wire path picks it up.
    bar_list_marker: list[object] = []

    async def slow(*_args: object, **_kwargs: object) -> list[object]:
        await asyncio.sleep(0.2)
        return bar_list_marker

    ibkr.ib.reqHistoricalDataAsync = AsyncMock(side_effect=slow)
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"SLOW": [_news("SLOW tops estimates")]}),
        settings=_settings(),
        float_source=_float_source({"SLOW": 5_000_000}),
        enrichment_timeout_seconds=0.01,
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].price is None
    # Wait long enough for the backgrounded task to complete so our cancel
    # path has a BarDataList to fire on.
    await asyncio.sleep(0.3)
    # cancelHistoricalData fires from _cancel_enrichment_task once the task
    # resolves to the partial BDL.
    assert ibkr.ib.cancelHistoricalData.call_args_list, (
        "expected cancelHistoricalData to fire on the wire after timeout"
    )
    # Registry must be empty — even the timed-out sub was unregistered.
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_fetch_quote_error_unregisters() -> None:
    """On IBKR error the enrichment must unregister its subscription (no registry leak)."""
    ibkr = _mock_ibkr(["BAD"])
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(
        side_effect=RuntimeError("IBKR error 162: historical data service error")
    )
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"BAD": [_news("BAD tops estimates")]}),
        settings=_settings(),
        float_source=_float_source({"BAD": 5_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].price is None
    assert len(ibkr.subscriptions) == 0


@pytest.mark.asyncio
async def test_fetch_quote_success_unregisters() -> None:
    """A successful enrichment call must unregister once the BarDataList lands."""
    ibkr = _mock_ibkr(["OK"])
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[
            _fake_daily_bar(close=5.0, volume=0),
            _fake_daily_bar(close=6.0, volume=100_000),
        ]
    )
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"OK": [_news("OK tops estimates")]}),
        settings=_settings(),
        float_source=_float_source({"OK": 5_000_000}),
    )
    await scanner.scan_top_gappers()
    assert len(ibkr.subscriptions) == 0


# ---------- Phase 6.8: manual catalyst override application ---------- #


def _settings_with_override_flag(*, allow: bool, float_max: int = 20_000_000) -> Settings:
    """Scanner-facing Settings with ``testing.allow_catalyst_overrides`` toggled."""
    s = _settings(float_max=float_max)
    return s.model_copy(
        update={"testing": s.testing.model_copy(update={"allow_catalyst_overrides": allow})}
    )


def _write_override_file(
    path: Path,
    *,
    symbol: str,
    category: str,
    expires_at: datetime,
    injected_at: datetime | None = None,
    note: str | None = None,
) -> None:
    """Write a single-entry override file to ``path`` for scanner-side tests."""
    import json as _json

    payload = [
        {
            "symbol": symbol,
            "category": category,
            "expires_at": expires_at.isoformat(),
            "injected_at": (injected_at or datetime.now(UTC)).isoformat(),
            "note": note,
            "injected_by": "cli",
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_scanner_applies_valid_override(tmp_path: Path, monkeypatch: Any) -> None:
    """An active override for AGPU injects its category; Finnhub is NOT called for that symbol."""
    override_path = tmp_path / "overrides.json"
    _write_override_file(
        override_path,
        symbol="AGPU",
        category="contract_or_m&a",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        note="paper-trading test",
    )
    monkeypatch.setattr("bot.scanning.scanner.load_active_overrides_map", _load_from_path(override_path))

    ibkr = _mock_ibkr(["AGPU"])
    finnhub = _mock_finnhub(news={})  # deliberately empty — override must bypass
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=finnhub,
        settings=_settings_with_override_flag(allow=True),
        float_source=_float_source({"AGPU": 4_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].catalyst == "contract_or_m&a"
    assert "no_catalyst" not in hits[0].reasons
    # Finnhub's company_news must not have been called for AGPU — the
    # override short-circuits the fetch, saving a rate-limit quota.
    finnhub_symbols = [c.args[0] for c in finnhub.company_news.call_args_list]
    assert "AGPU" not in finnhub_symbols


@pytest.mark.asyncio
async def test_scanner_skips_expired_override(tmp_path: Path, monkeypatch: Any) -> None:
    """An override whose ``expires_at`` is past is ignored; normal classifier runs."""
    override_path = tmp_path / "overrides.json"
    _write_override_file(
        override_path,
        symbol="AGPU",
        category="contract_or_m&a",
        expires_at=datetime.now(UTC) - timedelta(hours=1),  # already expired
        injected_at=datetime.now(UTC) - timedelta(hours=3),
    )
    monkeypatch.setattr("bot.scanning.scanner.load_active_overrides_map", _load_from_path(override_path))

    ibkr = _mock_ibkr(["AGPU"])
    finnhub = _mock_finnhub(news={"AGPU": [_news("AGPU signs a new contract with partner")]})
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=finnhub,
        settings=_settings_with_override_flag(allow=True),
        float_source=_float_source({"AGPU": 4_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    # Classifier must have run → catalyst comes from the Finnhub item, not the expired override.
    assert hits[0].catalyst == "contract_or_m&a"
    # Finnhub DID get called for AGPU (normal path).
    finnhub_symbols = [c.args[0] for c in finnhub.company_news.call_args_list]
    assert "AGPU" in finnhub_symbols


@pytest.mark.asyncio
async def test_scanner_ignores_overrides_when_flag_disabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Stale override file on disk but flag off: scanner never consults it."""
    override_path = tmp_path / "overrides.json"
    _write_override_file(
        override_path,
        symbol="AGPU",
        category="contract_or_m&a",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
    )

    calls: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr("bot.scanning.scanner.load_active_overrides_map", _spy)

    ibkr = _mock_ibkr(["AGPU"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={"AGPU": [_news("AGPU news unrelated to catalyst")]}),
        settings=_settings_with_override_flag(allow=False),
        float_source=_float_source({"AGPU": 4_000_000}),
    )
    hits = await scanner.scan_top_gappers()
    # Override file was NOT read — defence in depth.
    assert calls == []
    # Classifier ran normally on the non-green headline → catalyst is None
    # → Phase 6.11 drops the hit entirely (not surfaced with "no_catalyst"
    # reason as it was pre-6.11).
    assert hits == []


@pytest.mark.asyncio
async def test_manual_override_applied_logged(tmp_path: Path, monkeypatch: Any) -> None:
    """Applying an override fires a distinct ``catalyst.manual_override_applied`` event."""
    from structlog.testing import capture_logs  # noqa: PLC0415

    override_path = tmp_path / "overrides.json"
    _write_override_file(
        override_path,
        symbol="AGPU",
        category="clinical",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
        note="operator saw FDA headline",
    )
    monkeypatch.setattr("bot.scanning.scanner.load_active_overrides_map", _load_from_path(override_path))

    ibkr = _mock_ibkr(["AGPU"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(news={}),
        settings=_settings_with_override_flag(allow=True),
        float_source=_float_source({"AGPU": 4_000_000}),
    )
    with capture_logs() as captured:
        await scanner.scan_top_gappers()

    applied = [e for e in captured if e.get("event") == "catalyst.manual_override_applied"]
    assert len(applied) == 1
    evt = applied[0]
    assert evt["symbol"] == "AGPU"
    assert evt["category"] == "clinical"
    assert evt["note"] == "operator saw FDA headline"
    # The organic classifier event must NOT also fire for this symbol.
    organic = [
        e
        for e in captured
        if e.get("event") == "catalyst.item_matched" and e.get("symbol") == "AGPU"
    ]
    assert organic == []


def _load_from_path(path: Path) -> Any:
    """Build a ``load_active_overrides_map`` stand-in that reads from ``path``.

    The scanner accepts a keyword-only ``now`` so the stub honours the
    same signature — expired entries are filtered using the provided
    ``now`` rather than wall-clock, which keeps the expired-override
    test deterministic.
    """
    from bot.scanning.catalyst_overrides import load_active_overrides_map  # noqa: PLC0415

    def _loader(*, now: datetime) -> dict[str, Any]:
        return load_active_overrides_map(now=now, path=path)

    return _loader
