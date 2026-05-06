"""Phase 12 scanner integration tests — LLM classifier wiring + parallel calls + watchlist hook.

These tests live in their own file so they don't pollute
``tests/test_scanner.py``'s keyword-mode fixtures (the keyword path is
its own concern). Mocks the LLM classifier at the
``LLMCatalystClassifier.classify`` method boundary so we don't need
the full LLM-client / cache / cost-tracker stack to drive scanner
behaviour assertions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.brokerage.ibkr_client import SubscriptionRegistry
from bot.config import (
    CatalystClassifierConfig,
    CatalystClassifierKeywordConfig,
    CatalystClassifierLLMConfig,
    Settings,
    UniverseConfig,
)
from bot.scanning.finnhub_client import NewsItem
from bot.scanning.float_source import FloatSource
from bot.scanning.llm_catalyst_classifier import (
    CatalystClassification,
    ClassificationResult,
    LLMCatalystClassifier,
)
from bot.scanning.scanner import IBKRScanner

# ---------------- shared fixtures ---------------- #


def _phase12_settings(
    *,
    llm_enabled: bool = True,
    keyword_enabled: bool = False,
    float_max: int = 20_000_000,
) -> Settings:
    """Build Settings with the requested classifier-mode toggles."""
    base = Settings(universe=UniverseConfig(float_max=float_max))
    return base.model_copy(
        update={
            "catalyst_classifier": CatalystClassifierConfig(
                llm=CatalystClassifierLLMConfig(enabled=llm_enabled),
                keyword=CatalystClassifierKeywordConfig(enabled=keyword_enabled),
            ),
        }
    )


def _news(headline: str = "Earnings beat") -> NewsItem:
    return NewsItem(
        headline=headline,
        source="test",
        url="https://example.com/x",
        datetime=datetime.now(UTC),
        summary="",
        category="company",
    )


def _fake_scan_row(symbol: str) -> SimpleNamespace:
    contract = SimpleNamespace(
        symbol=symbol,
        secType="STK",
        currency="USD",
        exchange="SMART",
    )
    details = SimpleNamespace(contract=contract)
    return SimpleNamespace(contractDetails=details)


def _mock_ibkr(symbols: Iterable[str]) -> MagicMock:
    ibkr = MagicMock(name="IBKRClient")
    ibkr.ib = MagicMock(name="IB")
    ibkr.ib.reqScannerDataAsync = AsyncMock(return_value=[_fake_scan_row(s) for s in symbols])
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
    ibkr.ib.cancelScannerSubscription = MagicMock()
    ibkr.ib.cancelHistoricalData = MagicMock()
    ibkr.subscriptions = SubscriptionRegistry()
    ibkr.get_longname = AsyncMock(return_value="")
    return ibkr


def _mock_finnhub(news: dict[str, list[NewsItem]]) -> MagicMock:
    finnhub = MagicMock(name="FinnhubClient")

    async def get_news(symbol: str, hours_back: int = 24) -> list[NewsItem]:
        return news.get(symbol, [])

    finnhub.company_news = AsyncMock(side_effect=get_news)
    finnhub.company_profile = AsyncMock(return_value=None)
    return finnhub


def _float_source(yf_map: dict[str, int | None]) -> FloatSource:
    def fetch(symbol: str) -> int | None:
        return yf_map.get(symbol)

    return FloatSource(finnhub=None, yfinance_fetcher=fetch)


def _stub_classification(category: str = "earnings_beat") -> CatalystClassification:
    return CatalystClassification(
        qualifies=True,
        category=category,  # type: ignore[arg-type]
        confidence=0.85,
        reasoning="r",
        concerns=tuple(),
    )


def _stub_qualifying_result(symbol: str, category: str = "earnings_beat") -> ClassificationResult:
    return ClassificationResult(
        ticker=symbol,
        qualifies=True,
        classification=_stub_classification(category=category),
        reason="llm_classified",
        cost_usd=0.005,
        duration_seconds=1.0,
    )


def _stub_non_qualifying_result(symbol: str, reason: str = "stale_news") -> ClassificationResult:
    return ClassificationResult(ticker=symbol, qualifies=False, reason=reason)


def _classifier_returning(
    results: dict[str, ClassificationResult],
) -> MagicMock:
    """Build a MagicMock LLMCatalystClassifier whose ``classify`` returns the right ClassificationResult per ticker."""
    classifier = MagicMock(spec=LLMCatalystClassifier)

    async def classify(
        ticker: str, news_items: list[NewsItem], **_kwargs: Any
    ) -> ClassificationResult:
        return results.get(ticker, _stub_non_qualifying_result(ticker, reason="not_in_test_map"))

    classifier.classify = classify  # type: ignore[assignment]
    classifier.on_watchlist_removal = MagicMock()
    return classifier


# ---------------- mode resolution ---------------- #


@pytest.mark.asyncio
async def test_llm_mode_uses_classifier_categories() -> None:
    """``llm.enabled=true`` + classifier wired → hits carry the LLM's category."""
    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    classifier = _classifier_returning(
        {"ACME": _stub_qualifying_result("ACME", category="clinical_data")}
    )
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub({"ACME": [_news("Phase 3 readout")]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
        llm_classifier=classifier,
    )
    hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    assert hits[0].symbol == "ACME"
    assert hits[0].catalyst == "clinical_data"


@pytest.mark.asyncio
async def test_llm_mode_drops_non_qualifying_tickers() -> None:
    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    classifier = _classifier_returning(
        {
            "GOOD": _stub_qualifying_result("GOOD"),
            "BAD": _stub_non_qualifying_result("BAD", reason="sympathy_only"),
        }
    )
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["GOOD", "BAD"]),
        finnhub=_mock_finnhub(
            {"GOOD": [_news("Earnings beat")], "BAD": [_news("Generic market wrap")]}
        ),
        settings=settings,
        float_source=_float_source({"GOOD": 5_000_000, "BAD": 5_000_000}),
        llm_classifier=classifier,
    )
    hits = await scanner.scan_top_gappers()
    symbols = {h.symbol for h in hits}
    assert "GOOD" in symbols
    assert "BAD" not in symbols


@pytest.mark.asyncio
async def test_llm_enabled_but_no_classifier_falls_back_silently() -> None:
    """``llm.enabled=true`` but ``llm_classifier=None`` → resolved mode = keyword path or none.

    With keyword.enabled=False (default in this fixture), no catalyst
    evaluation runs; everything drops on no_catalyst. The scanner logs
    a warning so the operator sees the silent degradation.
    """
    from structlog.testing import capture_logs

    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub({"ACME": [_news("Earnings beat raised guidance")]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
        llm_classifier=None,
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    events = [e.get("event") for e in captured]
    assert "scanner.classifier_llm_bootstrap_missing" in events


@pytest.mark.asyncio
async def test_both_modes_disabled_qualifies_nothing() -> None:
    from structlog.testing import capture_logs

    settings = _phase12_settings(llm_enabled=False, keyword_enabled=False)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub({"ACME": [_news("Earnings beat")]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    events = [e.get("event") for e in captured]
    assert "scanner.classifier_no_mode_enabled" in events


@pytest.mark.asyncio
async def test_both_modes_enabled_prefers_llm_with_warning() -> None:
    from structlog.testing import capture_logs

    settings = _phase12_settings(llm_enabled=True, keyword_enabled=True)
    classifier = _classifier_returning({"ACME": _stub_qualifying_result("ACME")})
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["ACME"]),
        finnhub=_mock_finnhub({"ACME": [_news("Earnings beat")]}),
        settings=settings,
        float_source=_float_source({"ACME": 5_000_000}),
        llm_classifier=classifier,
    )
    with capture_logs() as captured:
        hits = await scanner.scan_top_gappers()
    assert len(hits) == 1
    events = [e.get("event") for e in captured]
    assert "scanner.classifier_both_modes_enabled" in events


# ---------------- parallel execution ---------------- #


@pytest.mark.asyncio
async def test_multiple_tickers_classified_in_parallel() -> None:
    """All ``classify`` calls overlap in a single ``asyncio.gather``.

    Drives the assertion via a synchronization barrier — each tracking
    ``classify`` call waits on a shared event; if calls were serial,
    only the first would advance and the test would deadlock. We
    use a ``timeout=2.0`` on ``scan_top_gappers`` to fail loudly
    rather than hanging forever in the deadlocked case.
    """
    barrier_count = 3
    started_count = 0
    proceed_event = asyncio.Event()
    started_event = asyncio.Event()

    async def parallel_classify(
        ticker: str, news_items: list[NewsItem], **_kwargs: Any
    ) -> ClassificationResult:
        nonlocal started_count
        started_count += 1
        if started_count == barrier_count:
            started_event.set()
        # Wait until all three have started — confirms parallelism.
        await started_event.wait()
        proceed_event.set()
        return _stub_qualifying_result(ticker)

    classifier = MagicMock(spec=LLMCatalystClassifier)
    classifier.classify = parallel_classify  # type: ignore[assignment]
    classifier.on_watchlist_removal = MagicMock()

    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["A", "B", "C"]),
        finnhub=_mock_finnhub({"A": [_news()], "B": [_news()], "C": [_news()]}),
        settings=settings,
        float_source=_float_source({"A": 5_000_000, "B": 5_000_000, "C": 5_000_000}),
        llm_classifier=classifier,
    )
    hits = await asyncio.wait_for(scanner.scan_top_gappers(), timeout=2.0)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_one_ticker_failure_does_not_fail_batch() -> None:
    """A classifier raising on one ticker leaves the other tickers' results intact."""
    classifier = MagicMock(spec=LLMCatalystClassifier)

    async def maybe_raise(
        ticker: str, news_items: list[NewsItem], **_kwargs: Any
    ) -> ClassificationResult:
        if ticker == "BAD":
            raise RuntimeError("classifier crashed for BAD")
        return _stub_qualifying_result(ticker)

    classifier.classify = maybe_raise  # type: ignore[assignment]
    classifier.on_watchlist_removal = MagicMock()

    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["GOOD", "BAD"]),
        finnhub=_mock_finnhub({"GOOD": [_news()], "BAD": [_news()]}),
        settings=settings,
        float_source=_float_source({"GOOD": 5_000_000, "BAD": 5_000_000}),
        llm_classifier=classifier,
    )
    hits = await scanner.scan_top_gappers()
    symbols = {h.symbol for h in hits}
    assert "GOOD" in symbols
    assert "BAD" not in symbols  # crash → no_catalyst → drop


# ---------------- watchlist removal hook ---------------- #


def test_scanner_on_watchlist_removal_forwards_to_classifier() -> None:
    classifier = MagicMock(spec=LLMCatalystClassifier)
    classifier.on_watchlist_removal = MagicMock()
    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr([]),
        finnhub=_mock_finnhub({}),
        settings=settings,
        float_source=_float_source({}),
        llm_classifier=classifier,
    )
    scanner.on_watchlist_removal("ACME")
    classifier.on_watchlist_removal.assert_called_once_with("ACME")


def test_scanner_on_watchlist_removal_no_classifier_is_noop() -> None:
    """When no LLM classifier is wired, on_watchlist_removal is a clean no-op."""
    settings = _phase12_settings(llm_enabled=False, keyword_enabled=True)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr([]),
        finnhub=_mock_finnhub({}),
        settings=settings,
        float_source=_float_source({}),
    )
    # Must not raise.
    scanner.on_watchlist_removal("ACME")


# ---------------- pillar ordering ---------------- #


@pytest.mark.asyncio
async def test_high_float_symbols_never_reach_classifier() -> None:
    """Phase 12 spec: news fetch + classify run only after deterministic pillars.

    A high-float symbol fails the float filter; the scanner must NOT
    invoke the classifier for it (saves the LLM call for a doomed
    ticker).
    """
    seen_tickers: list[str] = []
    classifier = MagicMock(spec=LLMCatalystClassifier)

    async def record(
        ticker: str, news_items: list[NewsItem], **_kwargs: Any
    ) -> ClassificationResult:
        seen_tickers.append(ticker)
        return _stub_qualifying_result(ticker)

    classifier.classify = record  # type: ignore[assignment]
    classifier.on_watchlist_removal = MagicMock()

    settings = _phase12_settings(llm_enabled=True, keyword_enabled=False, float_max=10_000_000)
    scanner = IBKRScanner(
        ibkr=_mock_ibkr(["LOWF", "HIGHF"]),
        finnhub=_mock_finnhub({"LOWF": [_news()], "HIGHF": [_news()]}),
        settings=settings,
        float_source=_float_source({"LOWF": 3_000_000, "HIGHF": 50_000_000}),
        llm_classifier=classifier,
    )
    await scanner.scan_top_gappers()
    # HIGHF was dropped at the float filter; classifier never saw it.
    assert "HIGHF" not in seen_tickers
    assert "LOWF" in seen_tickers
