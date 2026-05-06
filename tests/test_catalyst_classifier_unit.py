"""Unit tests for the Phase 12 catalyst classifier — cache, cost, prompts.

The lower-level building blocks. Higher-level behavioural tests
(qualified-memory, watchlist removal, scanner integration) live in
``tests/test_catalyst_classifier_*.py`` files alongside this one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bot.scanning.llm_catalyst_classifier.cache import (
    DEFAULT_CAPACITY,
    DEFAULT_TTL_SECONDS,
    ClassificationCache,
    hash_headlines,
)
from bot.scanning.llm_catalyst_classifier.cost_tracker import CatalystCostTracker
from bot.scanning.llm_catalyst_classifier.llm_client import CatalystClassification
from bot.scanning.llm_catalyst_classifier.prompts import (
    CATALYST_CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFY_CATALYST_TOOL,
    CLASSIFY_CATALYST_TOOL_NAME,
    render_user_message,
)

# ---------------- hash_headlines ---------------- #


def test_hash_headlines_stable_under_order_permutation() -> None:
    """Sorting + dedup means {A, B} == {B, A}."""
    a = hash_headlines(["earnings beat", "stock up"])
    b = hash_headlines(["stock up", "earnings beat"])
    assert a == b


def test_hash_headlines_canonical_lower_strip() -> None:
    """Whitespace + casing normalised to coalesce trivial variants."""
    a = hash_headlines(["  Earnings BEAT  "])
    b = hash_headlines(["earnings beat"])
    assert a == b


def test_hash_headlines_distinguishes_different_content() -> None:
    """Different headline sets produce different hashes."""
    a = hash_headlines(["earnings beat"])
    b = hash_headlines(["stock fell"])
    assert a != b


def test_hash_headlines_handles_empty_and_whitespace() -> None:
    """Empty / whitespace-only items are dropped."""
    a = hash_headlines(["earnings beat", "", "   "])
    b = hash_headlines(["earnings beat"])
    assert a == b


# ---------------- ClassificationCache ---------------- #


def _classification(qualifies: bool = True) -> CatalystClassification:
    return CatalystClassification(
        qualifies=qualifies,
        category="earnings_beat" if qualifies else "non_qualifying_other",
        confidence=0.8,
        reasoning="r",
        concerns=tuple(),
    )


def test_cache_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        ClassificationCache(ttl_seconds=0)
    with pytest.raises(ValueError, match="capacity"):
        ClassificationCache(capacity=0)


def test_cache_hit_within_ttl() -> None:
    cache = ClassificationCache(ttl_seconds=60.0, capacity=10)
    cache.put("AAA", "h1", _classification())
    assert cache.get("AAA", "h1") is not None
    # Different ticker, same hash → miss.
    assert cache.get("BBB", "h1") is None
    # Same ticker, different hash → miss.
    assert cache.get("AAA", "h2") is None


def test_cache_ttl_expiry_with_injected_clock() -> None:
    """Drive an explicit clock past the TTL; subsequent get returns None."""
    now = [100.0]

    def clock() -> float:
        return now[0]

    cache = ClassificationCache(ttl_seconds=30.0, capacity=10, clock=clock)
    cache.put("AAA", "h1", _classification())
    assert cache.get("AAA", "h1") is not None
    now[0] += 31.0  # past TTL
    assert cache.get("AAA", "h1") is None
    assert len(cache) == 0  # expired entry was evicted on lookup


def test_cache_lru_eviction_at_capacity() -> None:
    """Inserting past capacity evicts the LRU entry."""
    cache = ClassificationCache(ttl_seconds=60.0, capacity=3)
    cache.put("A", "h", _classification())
    cache.put("B", "h", _classification())
    cache.put("C", "h", _classification())
    cache.get("A", "h")  # touch A → A becomes most recent
    cache.put("D", "h", _classification())  # evicts B (oldest)
    assert cache.get("A", "h") is not None
    assert cache.get("B", "h") is None  # evicted
    assert cache.get("C", "h") is not None
    assert cache.get("D", "h") is not None


def test_cache_caches_qualifies_false_too() -> None:
    """Cache the non-qualify dispositions too so identical news doesn't re-evaluate."""
    cache = ClassificationCache(ttl_seconds=60.0, capacity=10)
    cache.put("AAA", "h1", _classification(qualifies=False))
    cached = cache.get("AAA", "h1")
    assert cached is not None
    assert cached.qualifies is False


def test_cache_clear_empties_store() -> None:
    cache = ClassificationCache(ttl_seconds=60.0, capacity=10)
    cache.put("A", "h", _classification())
    cache.put("B", "h", _classification())
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_cache_default_constants_match_spec() -> None:
    """Phase 12 spec: 30 minutes / 200 entries."""
    assert DEFAULT_TTL_SECONDS == 1800
    assert DEFAULT_CAPACITY == 200


# ---------------- CatalystCostTracker ---------------- #


def test_cost_tracker_constructor_validates() -> None:
    with pytest.raises(ValueError):
        CatalystCostTracker(soft_cap_usd=0.0, hard_cap_usd=1.0)
    with pytest.raises(ValueError):
        CatalystCostTracker(soft_cap_usd=1.0, hard_cap_usd=0.0)
    with pytest.raises(ValueError, match="strictly less"):
        CatalystCostTracker(soft_cap_usd=10.0, hard_cap_usd=10.0)


def test_cost_tracker_starts_zero() -> None:
    tracker = CatalystCostTracker()
    assert tracker.cost_today_usd() == 0.0
    assert not tracker.is_hard_capped()
    assert not tracker.soft_warning_fired_today()


def test_cost_tracker_record_accumulates_within_day() -> None:
    tracker = CatalystCostTracker(soft_cap_usd=5.0, hard_cap_usd=25.0)
    tracker.record_cost(1.5)
    tracker.record_cost(2.0)
    assert tracker.cost_today_usd() == pytest.approx(3.5)


def test_cost_tracker_soft_warning_fires_once_per_day() -> None:
    seen: list[str] = []
    tracker = CatalystCostTracker(soft_cap_usd=2.0, hard_cap_usd=10.0, notify_callback=seen.append)
    tracker.record_cost(1.0)
    assert not tracker.soft_warning_fired_today()
    tracker.record_cost(1.5)  # crosses 2.0
    assert tracker.soft_warning_fired_today()
    assert any("soft warning" in m for m in seen)
    # Subsequent crossings within same day don't re-fire.
    tracker.record_cost(0.5)
    soft_msgs = [m for m in seen if "soft warning" in m]
    assert len(soft_msgs) == 1


def test_cost_tracker_hard_cap_latches_for_day() -> None:
    seen: list[str] = []
    tracker = CatalystCostTracker(soft_cap_usd=1.0, hard_cap_usd=2.0, notify_callback=seen.append)
    tracker.record_cost(2.5)  # past both caps in one shot
    assert tracker.is_hard_capped()
    hard_msgs = [m for m in seen if "HARD CAP" in m]
    assert len(hard_msgs) == 1
    # Subsequent records ignored once capped.
    tracker.record_cost(100.0)
    assert tracker.cost_today_usd() == pytest.approx(2.5)


def test_cost_tracker_day_rollover_resets() -> None:
    """Inject a now_provider that returns yesterday, then today, to drive rollover."""
    days = [datetime(2026, 5, 5, 10, 0, tzinfo=UTC)]

    def now_provider() -> datetime:
        return days[0]

    tracker = CatalystCostTracker(soft_cap_usd=1.0, hard_cap_usd=5.0, now_provider=now_provider)
    tracker.record_cost(6.0)
    assert tracker.is_hard_capped()
    # Roll into the next day.
    days[0] = datetime(2026, 5, 6, 10, 0, tzinfo=UTC)
    assert not tracker.is_hard_capped()  # new day, fresh state
    assert tracker.cost_today_usd() == 0.0


# ---------------- prompts ---------------- #


def test_system_prompt_loads_and_is_nontrivial() -> None:
    """The system prompt is a non-empty string of meaningful length."""
    assert isinstance(CATALYST_CLASSIFIER_SYSTEM_PROMPT, str)
    assert len(CATALYST_CLASSIFIER_SYSTEM_PROMPT) > 1000
    # Spot-check a couple of strategy-specific phrases the operator
    # explicitly wanted encoded.
    assert "small-cap momentum" in CATALYST_CLASSIFIER_SYSTEM_PROMPT
    assert "classify_catalyst" in CATALYST_CLASSIFIER_SYSTEM_PROMPT


def test_tool_schema_shape_matches_anthropic_format() -> None:
    """The tool dict has the structure Anthropic expects: name + description + input_schema."""
    assert CLASSIFY_CATALYST_TOOL["name"] == CLASSIFY_CATALYST_TOOL_NAME
    assert "description" in CLASSIFY_CATALYST_TOOL
    schema = CLASSIFY_CATALYST_TOOL["input_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"qualifies", "category", "confidence", "reasoning"}
    # Every category in the spec is in the enum.
    enum = schema["properties"]["category"]["enum"]
    for category in (
        "earnings_beat",
        "clinical_data",
        "fda_approval",
        "m_a_definitive",
        "contract_win",
        "regulatory_milestone",
        "fundamental_inflection",
        "sympathy_only",
        "stale_news",
        "announcement_only",
        "routine_filings",
        "pump_indicators",
        "non_qualifying_other",
    ):
        assert category in enum


def test_render_user_message_includes_required_fields() -> None:
    msg = render_user_message(
        symbol="ACME",
        headlines=[("2026-05-05T10:00:00+00:00", "Earnings beat", "Revenue +20%")],
        today_iso="2026-05-05",
        now_iso_et="2026-05-05T06:00:00-04:00",
    )
    assert "Ticker: ACME" in msg
    assert "Earnings beat" in msg
    assert "Revenue +20%" in msg
    assert "Today's date: 2026-05-05" in msg
    assert "Current time: 2026-05-05T06:00:00-04:00" in msg


def test_render_user_message_omits_optional_fields_when_none() -> None:
    msg = render_user_message(
        symbol="ACME",
        headlines=[],
        today_iso="2026-05-05",
        now_iso_et="2026-05-05T06:00:00-04:00",
    )
    assert "Market cap" not in msg
    assert "Recent capital raises" not in msg


def test_render_user_message_includes_optional_fields_when_provided() -> None:
    msg = render_user_message(
        symbol="ACME",
        headlines=[],
        today_iso="2026-05-05",
        now_iso_et="2026-05-05T06:00:00-04:00",
        market_cap_usd=85_000_000.0,
        recent_raise_count=3,
    )
    assert "Market cap: $85,000,000" in msg
    assert "Recent capital raises (last 6 months): 3" in msg


def test_render_user_message_handles_missing_publish_time() -> None:
    """Headlines without a publish timestamp render with an explicit placeholder."""
    msg = render_user_message(
        symbol="ACME",
        headlines=[(None, "Some headline", "summary")],
        today_iso="2026-05-05",
        now_iso_et="2026-05-05T06:00:00-04:00",
    )
    assert "[unknown publish time] Some headline" in msg
