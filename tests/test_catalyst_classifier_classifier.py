"""Behavioural tests for ``LLMCatalystClassifier`` — the Phase 12 classifier core.

Mock the ``AnthropicCatalystClient`` at the .call boundary so we drive
deterministic LLM responses without hitting the network. Tests focus on
the classifier's contracts: session-qualified short-circuit, cache
integration, self-disable, cost-cap behaviour, watchlist removal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from bot.scanning.finnhub_client import NewsItem
from bot.scanning.llm_catalyst_classifier.cache import ClassificationCache
from bot.scanning.llm_catalyst_classifier.classifier import (
    LLMCatalystClassifier,
)
from bot.scanning.llm_catalyst_classifier.cost_tracker import CatalystCostTracker
from bot.scanning.llm_catalyst_classifier.llm_client import (
    CatalystClassification,
    LLMCallResult,
)


def _news(headline: str, hours_ago: float = 1.0) -> NewsItem:
    return NewsItem(
        headline=headline,
        source="test",
        url="https://example.com/x",
        datetime=datetime(2026, 5, 5, 14, 0, tzinfo=UTC),
        summary="",
        category="company",
    )


def _classification(
    qualifies: bool = True, category: str = "earnings_beat"
) -> CatalystClassification:
    return CatalystClassification(
        qualifies=qualifies,
        category=category,  # type: ignore[arg-type]
        confidence=0.8,
        reasoning="test reasoning",
        concerns=tuple(),
    )


@dataclass
class _FakeLLMClient:
    """Returns the next queued ``LLMCallResult`` per ``call`` — same shape as exit advisor tests."""

    queued: list[LLMCallResult] = field(default_factory=list)
    calls_seen: list[tuple[str, str]] = field(default_factory=list)

    def call(
        self, system_prompt: str, user_message: str, tool_schema: dict[str, Any]
    ) -> LLMCallResult:
        self.calls_seen.append((system_prompt[:30], user_message[:50]))
        if not self.queued:
            return LLMCallResult(
                success=False,
                classification=None,
                cost_usd=0.0,
                duration_seconds=0.0,
                failure_reason="no_queued_response",
            )
        return self.queued.pop(0)


def _success_result(
    qualifies: bool = True,
    category: str = "earnings_beat",
    cost_usd: float = 0.005,
) -> LLMCallResult:
    return LLMCallResult(
        success=True,
        classification=_classification(qualifies=qualifies, category=category),
        cost_usd=cost_usd,
        duration_seconds=1.2,
    )


def _failure_result(reason: str = "llm_timeout") -> LLMCallResult:
    return LLMCallResult(
        success=False,
        classification=None,
        cost_usd=0.0,
        duration_seconds=0.5,
        failure_reason=reason,
    )


def _make_classifier(
    llm: _FakeLLMClient,
    *,
    soft_cap_usd: float = 100.0,
    hard_cap_usd: float = 1000.0,
    self_disable_min_calls: int = 5,
    self_disable_failure_rate: float = 0.5,
    cache_capacity: int = 200,
    notify_callback: Callable[[str], None] | None = None,
) -> LLMCatalystClassifier:
    return LLMCatalystClassifier(
        llm_client=llm,  # type: ignore[arg-type]
        cache=ClassificationCache(ttl_seconds=1800.0, capacity=cache_capacity),
        cost_tracker=CatalystCostTracker(soft_cap_usd=soft_cap_usd, hard_cap_usd=hard_cap_usd),
        self_disable_failure_rate=self_disable_failure_rate,
        self_disable_min_calls=self_disable_min_calls,
        notify_callback=notify_callback,
    )


# ---------------- empty-news short-circuit ---------------- #


@pytest.mark.asyncio
async def test_empty_news_returns_no_news_without_calling_llm() -> None:
    llm = _FakeLLMClient(queued=[_success_result()])
    classifier = _make_classifier(llm)
    result = await classifier.classify("AAA", news_items=[])
    assert result.qualifies is False
    assert result.reason == "no_news"
    assert llm.calls_seen == []


@pytest.mark.asyncio
async def test_empty_news_emits_structured_event() -> None:
    """Silent no-news drops must surface as a log event for operator audit."""
    from structlog.testing import capture_logs

    llm = _FakeLLMClient(queued=[_success_result()])
    classifier = _make_classifier(llm)
    with capture_logs() as captured:
        await classifier.classify("ERNA", news_items=[])
    no_news_events = [e for e in captured if e.get("event") == "catalyst_classifier.no_news"]
    assert len(no_news_events) == 1
    assert no_news_events[0]["ticker"] == "ERNA"


# ---------------- success / cache integration ---------------- #


@pytest.mark.asyncio
async def test_classify_qualifies_true_calls_llm_and_caches() -> None:
    llm = _FakeLLMClient(queued=[_success_result(qualifies=True)])
    classifier = _make_classifier(llm)
    news = [_news("Earnings beat raised guidance")]
    result = await classifier.classify("AAA", news_items=news)
    assert result.qualifies is True
    assert result.classification is not None
    assert result.classification.category == "earnings_beat"
    assert result.cached is False
    # Second call with identical news → cache hit, no LLM call.
    result2 = await classifier.classify("AAA", news_items=news)
    # Note: second call short-circuits via _qualified_this_session BEFORE
    # the cache, so reason is "previously_qualified", not "cache_hit".
    assert result2.qualifies is True
    assert result2.reason == "previously_qualified"
    assert len(llm.calls_seen) == 1


@pytest.mark.asyncio
async def test_classify_qualifies_false_caches_but_no_session_admit() -> None:
    """Non-qualifying classifications are cached, but the ticker stays out of the session set."""
    llm = _FakeLLMClient(queued=[_success_result(qualifies=False, category="stale_news")])
    classifier = _make_classifier(llm)
    news = [_news("Something old")]
    result = await classifier.classify("BBB", news_items=news)
    assert result.qualifies is False
    assert "BBB" not in classifier.qualified_this_session()
    # Second call → cache hit (NOT session-skip), still hits cache layer.
    result2 = await classifier.classify("BBB", news_items=news)
    assert result2.qualifies is False
    assert result2.cached is True
    assert result2.reason == "cache_hit"
    assert len(llm.calls_seen) == 1


@pytest.mark.asyncio
async def test_classify_different_headlines_misses_cache() -> None:
    llm = _FakeLLMClient(
        queued=[_success_result(qualifies=False), _success_result(qualifies=False)]
    )
    classifier = _make_classifier(llm)
    await classifier.classify("AAA", news_items=[_news("first headline")])
    await classifier.classify("AAA", news_items=[_news("second headline")])
    assert len(llm.calls_seen) == 2


# ---------------- session-qualified set + on_watchlist_removal ---------------- #


@pytest.mark.asyncio
async def test_qualified_ticker_short_circuits_on_subsequent_pass() -> None:
    llm = _FakeLLMClient(queued=[_success_result(qualifies=True)])
    classifier = _make_classifier(llm)
    news = [_news("first pass headline")]
    await classifier.classify("AAA", news_items=news)
    assert "AAA" in classifier.qualified_this_session()
    # Different headlines, but session-qualified → short-circuit BEFORE cache.
    different_news = [_news("totally different headline")]
    result = await classifier.classify("AAA", news_items=different_news)
    assert result.qualifies is True
    assert result.reason == "previously_qualified"
    assert len(llm.calls_seen) == 1, "session-qualified short-circuit must skip LLM"


@pytest.mark.asyncio
async def test_non_qualifying_does_not_admit_to_session_set() -> None:
    llm = _FakeLLMClient(queued=[_success_result(qualifies=False)])
    classifier = _make_classifier(llm)
    await classifier.classify("AAA", news_items=[_news("h")])
    assert classifier.qualified_this_session() == set()


@pytest.mark.asyncio
async def test_on_watchlist_removal_evicts_from_session_set() -> None:
    llm = _FakeLLMClient(queued=[_success_result(qualifies=True), _success_result(qualifies=True)])
    classifier = _make_classifier(llm)
    await classifier.classify("AAA", news_items=[_news("h1")])
    assert "AAA" in classifier.qualified_this_session()
    classifier.on_watchlist_removal("AAA")
    assert "AAA" not in classifier.qualified_this_session()
    # Next call with same news fires LLM again because the session memory is gone.
    # (Cache might still have the result though — depends on whether we want
    # cache to survive a watchlist removal. The spec says yes — only the
    # session memory clears on removal; the cache persists.)
    result = await classifier.classify("AAA", news_items=[_news("h1")])
    assert result.qualifies is True
    # The classification came from cache (not LLM), but it re-admits to the set.
    assert result.cached is True
    assert "AAA" in classifier.qualified_this_session()


def test_on_watchlist_removal_for_unqualified_ticker_is_noop() -> None:
    """No exception, no log spam, just a clean no-op."""
    llm = _FakeLLMClient()
    classifier = _make_classifier(llm)
    # Never qualified — removal must not raise.
    classifier.on_watchlist_removal("NEVER_QUALIFIED")


def test_qualified_set_empty_after_construction() -> None:
    llm = _FakeLLMClient()
    classifier = _make_classifier(llm)
    assert classifier.qualified_this_session() == set()
    assert not classifier.is_self_disabled()


# ---------------- self-disable ---------------- #


@pytest.mark.asyncio
async def test_self_disable_after_failure_rate_threshold() -> None:
    notifications: list[str] = []
    llm = _FakeLLMClient(queued=[_failure_result() for _ in range(10)])
    classifier = _make_classifier(
        llm,
        self_disable_min_calls=3,
        self_disable_failure_rate=0.5,
        notify_callback=notifications.append,
    )
    # Fire 5 calls with 5 failures → rate 100% > 50% threshold.
    for i in range(5):
        await classifier.classify(f"SYM{i}", news_items=[_news(f"h{i}")])
    assert classifier.is_self_disabled()
    assert any("self-disabled" in m for m in notifications)


@pytest.mark.asyncio
async def test_self_disabled_short_circuits_subsequent_calls() -> None:
    llm = _FakeLLMClient(queued=[_failure_result() for _ in range(20)])
    classifier = _make_classifier(
        llm,
        self_disable_min_calls=2,
        self_disable_failure_rate=0.4,
    )
    # Trip self-disable.
    for i in range(3):
        await classifier.classify(f"SYM{i}", news_items=[_news(f"h{i}")])
    assert classifier.is_self_disabled()
    calls_before = len(llm.calls_seen)
    # Subsequent classify must NOT invoke LLM.
    later = await classifier.classify("ZZZ", news_items=[_news("h_late")])
    assert later.qualifies is False
    assert later.reason == "classifier_self_disabled"
    assert len(llm.calls_seen) == calls_before


@pytest.mark.asyncio
async def test_self_disabled_does_not_admit_to_session_set_via_cache() -> None:
    """Cached qualifies=True after self-disable shouldn't re-admit the ticker.

    Path: pre-disable classify produces qualifies=True (admitted to session
    set). After eviction via on_watchlist_removal AND self-disable trips,
    the same news triggers a cache hit but the ticker shouldn't return to
    the qualified set (defensive behaviour: self-disabled means no
    qualification of any kind).
    """
    llm = _FakeLLMClient(
        queued=[
            _success_result(qualifies=True),  # AAA qualifies fresh
            _failure_result(),
            _failure_result(),
            _failure_result(),
            _failure_result(),
            _failure_result(),  # 5 failures → trips self-disable
        ]
    )
    classifier = _make_classifier(llm, self_disable_min_calls=5, self_disable_failure_rate=0.5)
    # Cache the AAA qualifies-true.
    news_aaa = [_news("h_aaa")]
    await classifier.classify("AAA", news_items=news_aaa)
    classifier.on_watchlist_removal("AAA")
    # Drive 5 failures on other tickers to trip self-disable.
    for i in range(5):
        await classifier.classify(f"FAIL{i}", news_items=[_news(f"f{i}")])
    assert classifier.is_self_disabled()
    # Now AAA re-classifies. Self-disable short-circuit happens BEFORE cache
    # lookup, so it returns qualifies=False, never admits to the set.
    result = await classifier.classify("AAA", news_items=news_aaa)
    assert result.qualifies is False
    assert result.reason == "classifier_self_disabled"
    assert "AAA" not in classifier.qualified_this_session()


# ---------------- cost cap ---------------- #


@pytest.mark.asyncio
async def test_cost_hard_cap_short_circuits_subsequent_calls() -> None:
    llm = _FakeLLMClient(queued=[_success_result(cost_usd=100.0)])
    classifier = _make_classifier(llm, soft_cap_usd=10.0, hard_cap_usd=20.0)
    # First call trips the hard cap.
    await classifier.classify("AAA", news_items=[_news("h")])
    calls_before = len(llm.calls_seen)
    # Second call short-circuits without invoking LLM.
    result = await classifier.classify("BBB", news_items=[_news("h2")])
    assert result.qualifies is False
    assert result.reason == "cost_hard_capped"
    assert len(llm.calls_seen) == calls_before


# ---------------- failure handling ---------------- #


@pytest.mark.asyncio
async def test_failure_does_not_admit_to_session_set() -> None:
    llm = _FakeLLMClient(queued=[_failure_result("api_error: boom")])
    classifier = _make_classifier(llm)
    result = await classifier.classify("AAA", news_items=[_news("h")])
    assert result.qualifies is False
    assert result.reason == "llm_call_failed"
    assert result.failure_reason is not None
    assert "api_error" in result.failure_reason
    assert "AAA" not in classifier.qualified_this_session()


@pytest.mark.asyncio
async def test_failure_does_not_cache() -> None:
    """Failed calls aren't cached — the next call retries the LLM."""
    llm = _FakeLLMClient(queued=[_failure_result(), _success_result(qualifies=True)])
    classifier = _make_classifier(llm)
    news = [_news("h")]
    first = await classifier.classify("AAA", news_items=news)
    assert first.qualifies is False
    second = await classifier.classify("AAA", news_items=news)
    assert second.qualifies is True
    # Two real LLM calls — failure didn't short-circuit the second.
    assert len(llm.calls_seen) == 2
