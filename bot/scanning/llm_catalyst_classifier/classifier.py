"""Phase 12 LLM-driven catalyst classifier.

Glues the LLM client, cache, cost tracker, and session-scoped
qualification memory together. The scanner constructs one instance per
session and calls :meth:`classify` for each surviving (post-float-filter)
ticker. Failure on one ticker doesn't fail the batch — the scanner
parallelises calls via ``asyncio.gather`` over per-ticker tasks.

Lookup priority on each ``classify`` call:

1. **Session-qualified set short-circuit**. If the ticker has been
   qualified earlier this session AND is still on the watchlist (the
   scanner calls :meth:`on_watchlist_removal` to evict on drop), return
   a synthetic ``ClassificationResult`` with ``qualifies=True,
   reason="previously_qualified"`` without touching cache or LLM.
2. **Self-disable / cost cap short-circuit**. If the failure-rate
   threshold has tripped or today's hard cap is exhausted, return
   ``qualifies=False`` with the corresponding reason.
3. **Cache lookup**. Hash the sorted headlines; if a non-expired entry
   exists, return that classification (cached entries are always
   trusted, including ``qualifies=False`` results).
4. **Live LLM call**. Construct the user message, call the API,
   validate the tool-use response. On success: cache, optionally
   admit to qualified set, return classification. On failure:
   increment failure counters, return ``qualifies=False``.

The class is internally synchronous around the LLM call (the underlying
``AnthropicCatalystClient`` is sync) but exposes an async ``classify()``
method so the scanner can ``asyncio.gather`` multiple instances.
``classify`` itself runs the sync LLM call in a thread executor so it
doesn't block the event loop while waiting on the API.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from bot.scanning.finnhub_client import NewsItem
from bot.scanning.llm_catalyst_classifier.cache import (
    ClassificationCache,
    hash_headlines,
)
from bot.scanning.llm_catalyst_classifier.cost_tracker import CatalystCostTracker
from bot.scanning.llm_catalyst_classifier.llm_client import (
    AnthropicCatalystClient,
    CatalystClassification,
    LLMCallResult,
)
from bot.scanning.llm_catalyst_classifier.prompts import (
    CATALYST_CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFY_CATALYST_TOOL,
    render_user_message,
)

_log = structlog.get_logger("bot.scanning.llm_catalyst_classifier")
_NY = ZoneInfo("America/New_York")

# Heuristic ceiling on prompt size — see config.yaml ``max_input_tokens``.
# Approximate: 1 token ≈ 4 chars for English text. We truncate the
# headline list (oldest dropped) when the rendered user message exceeds
# this character budget.
_DEFAULT_MAX_INPUT_CHARS = 16_000  # ~4000 tokens

DEFAULT_SELF_DISABLE_FAILURE_RATE = 0.5
DEFAULT_SELF_DISABLE_MIN_CALLS = 5


@dataclass(frozen=True)
class ClassificationResult:
    """Public result of one :meth:`LLMCatalystClassifier.classify` call.

    Always populated. ``qualifies=False`` results carry a ``reason``
    that explains why the ticker was rejected (or why the classifier
    declined to evaluate). The scanner attaches the result to its
    ``ScanHit`` row regardless.
    """

    ticker: str
    qualifies: bool
    classification: CatalystClassification | None = None
    reason: str = ""
    failure_reason: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    cached: bool = False


@dataclass
class _CallCounters:
    """Per-classifier rolling counters used by the self-disable check."""

    total_calls: int = 0
    failed_calls: int = 0


@dataclass
class _QualifiedRecord:
    """Marker for a ticker that has been admitted to the session set."""

    qualified_at: datetime
    classification: CatalystClassification


class LLMCatalystClassifier:
    """The Phase 12 classifier.

    One instance per session. Construct via
    :func:`bot.scanning.llm_catalyst_classifier.bootstrap.bootstrap_catalyst_classifier`
    in production, or directly with mocked dependencies in tests.
    """

    def __init__(
        self,
        *,
        llm_client: AnthropicCatalystClient,
        cache: ClassificationCache,
        cost_tracker: CatalystCostTracker,
        max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
        self_disable_failure_rate: float = DEFAULT_SELF_DISABLE_FAILURE_RATE,
        self_disable_min_calls: int = DEFAULT_SELF_DISABLE_MIN_CALLS,
        notify_callback: Any = None,
    ) -> None:
        if not 0.0 < self_disable_failure_rate <= 1.0:
            raise ValueError(
                f"self_disable_failure_rate must be in (0.0, 1.0]; got {self_disable_failure_rate}"
            )
        if self_disable_min_calls <= 0:
            raise ValueError(f"self_disable_min_calls must be > 0; got {self_disable_min_calls}")
        if max_input_chars <= 0:
            raise ValueError(f"max_input_chars must be > 0; got {max_input_chars}")
        self._llm_client = llm_client
        self._cache = cache
        self._cost_tracker = cost_tracker
        self._max_input_chars = max_input_chars
        self._self_disable_failure_rate = self_disable_failure_rate
        self._self_disable_min_calls = self_disable_min_calls
        self._notify_callback = notify_callback
        self._counters = _CallCounters()
        self._self_disabled = False
        self._qualified_this_session: dict[str, _QualifiedRecord] = {}

    # ---------------- public API ---------------- #

    async def classify(
        self,
        ticker: str,
        news_items: list[NewsItem],
        *,
        market_cap_usd: float | None = None,
        recent_raise_count: int | None = None,
        now: datetime | None = None,
    ) -> ClassificationResult:
        """Classify a single ticker. Always returns a ``ClassificationResult``.

        ``news_items`` is the raw Finnhub output. Empty list short-circuits
        to ``qualifies=False, reason="no_news"`` without touching cache
        or LLM — there's nothing to classify.

        ``market_cap_usd`` and ``recent_raise_count`` are optional context
        passed through to the LLM prompt. Omitted when ``None``.

        ``now`` is injectable so tests can drive deterministic prompt
        rendering. Defaults to ``datetime.now(_NY)``.
        """
        # 1. Session-qualified short-circuit.
        record = self._qualified_this_session.get(ticker)
        if record is not None:
            _log.info(
                "catalyst_classifier.skip_already_qualified",
                ticker=ticker,
                qualified_at=record.qualified_at.isoformat(),
                category=record.classification.category,
            )
            return ClassificationResult(
                ticker=ticker,
                qualifies=True,
                classification=record.classification,
                reason="previously_qualified",
            )

        # 2. Self-disable / cost cap short-circuit.
        if self._self_disabled:
            return ClassificationResult(
                ticker=ticker,
                qualifies=False,
                reason="classifier_self_disabled",
            )
        if self._cost_tracker.is_hard_capped():
            _log.info("catalyst_classifier.cost_cap_short_circuit", ticker=ticker)
            return ClassificationResult(
                ticker=ticker,
                qualifies=False,
                reason="cost_hard_capped",
            )

        # 3. Empty news → no catalyst.
        if not news_items:
            # Emit a structured event so the operator can audit silent
            # no-news drops without having to grep for the *absence* of
            # ``catalyst_classifier.evaluation``. ERNA on 2026-05-06 was
            # dropped this way despite a clinical readout that day —
            # Finnhub free-tier coverage missed the headline entirely
            # and we only noticed when AJ flagged it manually.
            _log.info("catalyst_classifier.no_news", ticker=ticker)
            return ClassificationResult(
                ticker=ticker,
                qualifies=False,
                reason="no_news",
            )

        # 4. Cache lookup.
        headlines_only = [item.headline for item in news_items]
        h = hash_headlines(headlines_only)
        cached_classification = self._cache.get(ticker, h)
        if cached_classification is not None:
            _log.info(
                "catalyst_classifier.cache_hit",
                ticker=ticker,
                qualifies=cached_classification.qualifies,
                category=cached_classification.category,
            )
            if cached_classification.qualifies:
                self._admit_to_session_set(ticker, cached_classification, now=now)
            return ClassificationResult(
                ticker=ticker,
                qualifies=cached_classification.qualifies,
                classification=cached_classification,
                reason="cache_hit",
                cached=True,
            )

        # 5. Live LLM call (run sync API in a thread executor so we don't
        #    block the event loop).
        result = await asyncio.to_thread(
            self._call_llm,
            ticker=ticker,
            news_items=news_items,
            market_cap_usd=market_cap_usd,
            recent_raise_count=recent_raise_count,
            now=now or datetime.now(_NY),
        )

        # Transient failures (Anthropic 529 OverloadedError) are invisible
        # to the self-disable accounting -- excluded from BOTH numerator
        # and denominator so the rate reflects only outcomes we control.
        # The next 5-minute rescan naturally recovers once Anthropic
        # restores capacity, so a short-lived blip shouldn't take the
        # pillar offline for the rest of the session. The failure still
        # logs as ``catalyst_classifier.failure`` (with ``transient=True``)
        # so the operator can audit how often this happens.
        if not result.transient:
            self._counters.total_calls += 1
        self._cost_tracker.record_cost(result.cost_usd)

        if not result.success:
            if not result.transient:
                self._counters.failed_calls += 1
            _log.warning(
                "catalyst_classifier.failure",
                ticker=ticker,
                failure_reason=result.failure_reason,
                duration_seconds=round(result.duration_seconds, 3),
                transient=result.transient,
            )
            if not result.transient:
                self._check_self_disable()
            return ClassificationResult(
                ticker=ticker,
                qualifies=False,
                reason="llm_call_failed",
                failure_reason=result.failure_reason,
                duration_seconds=result.duration_seconds,
            )

        classification = result.classification
        assert classification is not None  # success path narrows for mypy
        self._cache.put(ticker, h, classification)
        if classification.qualifies and not self._self_disabled:
            self._admit_to_session_set(ticker, classification, now=now)

        _log.info(
            "catalyst_classifier.evaluation",
            ticker=ticker,
            qualifies=classification.qualifies,
            category=classification.category,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            concerns=list(classification.concerns),
            cost_usd=round(result.cost_usd, 6),
            duration_seconds=round(result.duration_seconds, 3),
            market_cap_usd=market_cap_usd,
            recent_raise_count=recent_raise_count,
            headline_count=len(news_items),
        )
        return ClassificationResult(
            ticker=ticker,
            qualifies=classification.qualifies,
            classification=classification,
            reason="llm_classified",
            cost_usd=result.cost_usd,
            duration_seconds=result.duration_seconds,
        )

    def on_watchlist_removal(self, ticker: str) -> None:
        """Evict ``ticker`` from the session-qualified set.

        Idempotent — calling for a ticker that was never qualified is a
        no-op (does NOT log a warning, that would spam every removal of
        a non-qualified ticker).
        """
        if ticker in self._qualified_this_session:
            del self._qualified_this_session[ticker]
            _log.info("catalyst_classifier.watchlist_removal", ticker=ticker)

    # ---------------- introspection ---------------- #

    def is_self_disabled(self) -> bool:
        return self._self_disabled

    def qualified_this_session(self) -> set[str]:
        """Snapshot of currently-qualified tickers (forensics + tests)."""
        return set(self._qualified_this_session.keys())

    # ---------------- internals ---------------- #

    def _call_llm(
        self,
        *,
        ticker: str,
        news_items: list[NewsItem],
        market_cap_usd: float | None,
        recent_raise_count: int | None,
        now: datetime,
    ) -> LLMCallResult:
        """Sync LLM call (called from ``asyncio.to_thread`` in :meth:`classify`)."""
        headlines = self._build_headline_tuples(news_items)
        today_iso = now.astimezone(_NY).date().isoformat()
        now_iso_et = now.astimezone(_NY).isoformat(timespec="seconds")
        user_message = render_user_message(
            symbol=ticker,
            headlines=headlines,
            today_iso=today_iso,
            now_iso_et=now_iso_et,
            market_cap_usd=market_cap_usd,
            recent_raise_count=recent_raise_count,
        )
        # Truncation: if the message is over budget, drop the oldest
        # headline-tuples (sorted desc by published_at — the helper
        # builds in chronological order with newest first).
        while len(user_message) > self._max_input_chars and len(headlines) > 1:
            headlines = headlines[:-1]  # drop oldest tail
            user_message = render_user_message(
                symbol=ticker,
                headlines=headlines,
                today_iso=today_iso,
                now_iso_et=now_iso_et,
                market_cap_usd=market_cap_usd,
                recent_raise_count=recent_raise_count,
            )
        return self._llm_client.call(
            CATALYST_CLASSIFIER_SYSTEM_PROMPT,
            user_message,
            CLASSIFY_CATALYST_TOOL,
        )

    @staticmethod
    def _build_headline_tuples(
        news_items: list[NewsItem],
    ) -> list[tuple[str | None, str, str]]:
        """Convert ``NewsItem`` rows into ``(published_at, headline, summary)`` tuples.

        Newest first. Finnhub's ``NewsItem.datetime`` is the publish
        timestamp (tz-aware UTC) — coerced from the Finnhub unix-seconds
        integer at parse time.
        """
        sorted_items = sorted(
            news_items,
            key=lambda n: n.datetime,
            reverse=True,
        )
        return [
            (
                n.datetime.isoformat(),
                n.headline,
                n.summary or "",
            )
            for n in sorted_items
        ]

    def _admit_to_session_set(
        self,
        ticker: str,
        classification: CatalystClassification,
        *,
        now: datetime | None,
    ) -> None:
        """Add a qualified ticker to the session set. Skips when self-disabled."""
        if self._self_disabled:
            return  # defensive; the caller guards too
        self._qualified_this_session[ticker] = _QualifiedRecord(
            qualified_at=now if now is not None else datetime.now(_NY),
            classification=classification,
        )

    def _check_self_disable(self) -> None:
        """Trip the session-wide kill switch if failure rate is too high."""
        total = self._counters.total_calls
        failed = self._counters.failed_calls
        if total < self._self_disable_min_calls:
            return
        rate = failed / total
        if rate > self._self_disable_failure_rate:
            self._self_disabled = True
            message = (
                f"catalyst-classifier self-disabled for session: "
                f"failure rate {rate:.0%} > threshold "
                f"{self._self_disable_failure_rate:.0%} "
                f"after {total} calls ({failed} failed)"
            )
            _log.error(
                "catalyst_classifier.self_disabled",
                failure_rate=round(rate, 3),
                threshold=self._self_disable_failure_rate,
                total_calls=total,
                failed_calls=failed,
            )
            self._notify(message)

    def _notify(self, message: str) -> None:
        if self._notify_callback is None:
            return
        with contextlib.suppress(Exception):
            self._notify_callback(message)


__all__ = [
    "ClassificationResult",
    "LLMCatalystClassifier",
]
