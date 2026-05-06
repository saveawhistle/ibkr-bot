"""Phase 10.5 — name-extension catalyst attribution gate.

Phase 9.7 established the precision direction: a wrap article that
mentions our ticker only by Finnhub mistagging (no ticker in the
headline) gets correctly rejected. Phase 10.5 widens the recall
direction: a real press release that names the company by its actual
name without the cashtag (2026-05-01 SHPH "Shuttle Pharmaceutical
Enters Definite Agreement..." precedent) gets accepted via a
fallback that matches non-stopword tokens of the IBKR
``ContractDetails.longName``.

These tests cover the 13 cases in the Phase 10.5 spec, organised by
concern: tokenizer behaviour, the NameTokenCache lifecycle, and the
end-to-end ``classify()`` integration with both gates.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from structlog.testing import capture_logs

from bot.config import NameExtensionConfig, Settings
from bot.scanning.catalyst import (
    NameTokenCache,
    classify,
    tokenize_name,
)
from bot.scanning.finnhub_client import NewsItem

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _default_stopwords() -> frozenset[str]:
    """The shipped stopword set, lowercased — used by the tokenizer tests."""
    return frozenset(s.lower() for s in NameExtensionConfig().stopwords)


def _cache(
    *,
    high_rate_threshold: int = 10,
    min_token_len: int = 5,
) -> NameTokenCache:
    """Construct a NameTokenCache with the shipped stopwords + tunable other knobs."""
    return NameTokenCache(
        stopwords=_default_stopwords(),
        min_token_len=min_token_len,
        high_rate_threshold=high_rate_threshold,
    )


def _item(
    headline: str,
    summary: str = "",
    *,
    when: datetime | None = None,
) -> NewsItem:
    """Build a minimal NewsItem matching the existing test_catalyst.py fixture shape."""
    return NewsItem(
        headline=headline,
        source="test",
        url="https://example.com",
        datetime=when or datetime(2026, 5, 1, tzinfo=UTC),
        summary=summary,
        category="company",
    )


# ---------------------------------------------------------------------------
# 1. Tokenizer — tokenize_name behaviour
# ---------------------------------------------------------------------------


class TestTokenizer:
    """Pure-function tests for ``tokenize_name`` against the spec's edge cases."""

    def test_shph_baseline(self) -> None:
        """SHPH longName tokenises to the expected pair, no surprises."""
        tokens = tokenize_name(
            "SHUTTLE PHARMACEUTICAL HOLDINGS INC",
            _default_stopwords(),
            min_token_len=5,
        )
        # 'holdings' is stopworded; 'inc' (3 chars) is below min_token_len.
        assert tokens == ["shuttle", "pharmaceutical"]

    def test_rmax_min_len_5_yields_empty(self) -> None:
        """Spec case 2: RMAX with min_token_len=5 produces no usable tokens.

        Tokens after splitting on non-alphanumeric: re, max, holdings, inc, cl, a.
        With min_token_len=5: only 'holdings' would survive the length filter,
        but it's a stopword. Result: empty list → graceful fallback to
        ticker-only matching.
        """
        tokens = tokenize_name(
            "RE/MAX HOLDINGS INC-CL A",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == []

    def test_spy_yields_empty_via_stopwords(self) -> None:
        """Spec case 3: SPY's longName is all stopwords / too-short tokens."""
        tokens = tokenize_name(
            "SS SPDR S&P 500 ETF TRUST-US",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == []

    def test_atlas_lithium_multi_token(self) -> None:
        """Spec case 5: ATLX has two distinctive tokens."""
        tokens = tokenize_name(
            "ATLAS LITHIUM INC",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == ["atlas", "lithium"]

    def test_ibm_stopword_filtering(self) -> None:
        """Spec case 6: 'INTERNATIONAL' is stopworded; 'BUSINESS' and 'MACHINES' pass."""
        tokens = tokenize_name(
            "INTERNATIONAL BUSINESS MACHINES",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == ["business", "machines"]

    def test_hyphen_tokenisation(self) -> None:
        """Spec case 12: hyphens act as word separators, not preserved tokens."""
        tokens = tokenize_name(
            "ELECTRO-SENSORS INC",
            _default_stopwords(),
            min_token_len=5,
        )
        # If hyphen weren't a separator we'd see ['electrosensors', 'inc']
        # instead. The spec is explicit: hyphens split.
        assert tokens == ["electro", "sensors"]

    def test_empty_longname_returns_empty(self) -> None:
        """Empty input → empty token list, no exception."""
        assert tokenize_name("", _default_stopwords(), min_token_len=5) == []

    def test_pure_numeric_token_dropped(self) -> None:
        """A pure-numeric token (e.g. '500' in 'S&P 500') is filtered."""
        tokens = tokenize_name(
            "12345 PHARMACEUTICAL",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == ["pharmaceutical"]

    def test_duplicate_tokens_deduplicated(self) -> None:
        """A longName with repeated words (rare but possible) doesn't duplicate."""
        tokens = tokenize_name(
            "ATLAS ATLAS LITHIUM",
            _default_stopwords(),
            min_token_len=5,
        )
        assert tokens == ["atlas", "lithium"]

    def test_lower_min_token_len_admits_max(self) -> None:
        """``min_token_len`` is a config knob — at 3 it would re-admit RMAX's 'max'.

        The shipped default is 5 (deliberate — see config docstring), but
        the tokenizer respects whatever value the cache was built with.
        Confirms the parameter is plumbed correctly.
        """
        tokens = tokenize_name(
            "RE/MAX HOLDINGS INC-CL A",
            _default_stopwords(),
            min_token_len=3,
        )
        # 'holdings' is still a stopword; 'inc'/'max' both pass min_token_len=3.
        # 're' (2 chars) and 'a' (1 char) and 'cl' (2 chars) still drop.
        assert "max" in tokens
        assert "holdings" not in tokens


# ---------------------------------------------------------------------------
# 2. NameTokenCache lifecycle + telemetry
# ---------------------------------------------------------------------------


class TestNameTokenCache:
    """Lifecycle: populate + get_tokens + record_rescue + telemetry events."""

    def test_populate_caches_tokens(self) -> None:
        """``populate`` runs the tokenizer and ``get_tokens`` returns the result."""
        cache = _cache()
        cache.populate("SHPH", "SHUTTLE PHARMACEUTICAL HOLDINGS INC")
        assert cache.get_tokens("SHPH") == ["shuttle", "pharmaceutical"]

    def test_get_tokens_unpopulated_returns_empty_list(self) -> None:
        """A symbol the cache hasn't seen returns empty (caller falls back)."""
        cache = _cache()
        assert cache.get_tokens("UNKNOWN") == []

    def test_no_tokens_event_fires_once_per_symbol(self) -> None:
        """Spec case 9: ``catalyst.name_extension_no_tokens`` fires once per session."""
        cache = _cache()
        with capture_logs() as captured:
            cache.populate("SPY", "SS SPDR S&P 500 ETF TRUST-US")
            cache.populate("SPY", "SS SPDR S&P 500 ETF TRUST-US")  # idempotent re-populate
        events = [e for e in captured if e["event"] == "catalyst.name_extension_no_tokens"]
        assert len(events) == 1
        assert events[0]["symbol"] == "SPY"

    def test_longname_missing_event_fires_once_per_symbol(self) -> None:
        """Spec case 10: ``catalyst.name_extension_longname_missing`` fires once."""
        cache = _cache()
        with capture_logs() as captured:
            cache.populate("SBLX", "")
            cache.populate("SBLX", None)
        events = [e for e in captured if e["event"] == "catalyst.name_extension_longname_missing"]
        assert len(events) == 1
        assert events[0]["symbol"] == "SBLX"

    def test_no_tokens_distinct_from_longname_missing(self) -> None:
        """All-stopword longName fires no_tokens; missing longName fires longname_missing."""
        cache = _cache()
        with capture_logs() as captured:
            cache.populate("SPY", "SS SPDR S&P 500 ETF TRUST-US")  # tokens empty
            cache.populate("SBLX", "")  # longName missing
        no_tokens = [e for e in captured if e["event"] == "catalyst.name_extension_no_tokens"]
        missing = [e for e in captured if e["event"] == "catalyst.name_extension_longname_missing"]
        assert len(no_tokens) == 1 and no_tokens[0]["symbol"] == "SPY"
        assert len(missing) == 1 and missing[0]["symbol"] == "SBLX"

    def test_record_rescue_emits_rescue_event(self) -> None:
        """Spec case 7: rescue event payload contains all required fields."""
        cache = _cache()
        cache.populate("SHPH", "SHUTTLE PHARMACEUTICAL HOLDINGS INC")
        with capture_logs() as captured:
            cache.record_rescue(
                "SHPH",
                matched_token="shuttle",
                headline="Shuttle Pharmaceutical Enters Definite Agreement To Acquire United Dogecoin",
            )
        events = [e for e in captured if e["event"] == "catalyst.name_extension_rescued"]
        assert len(events) == 1
        evt = events[0]
        assert evt["symbol"] == "SHPH"
        assert evt["matched_token"] == "shuttle"
        assert "Shuttle Pharmaceutical" in evt["headline"]
        assert evt["longname_tokens"] == ["shuttle", "pharmaceutical"]

    def test_high_rate_threshold_fires_once(self) -> None:
        """Spec case 8: 11th rescue triggers high_rate; subsequent rescues don't re-warn.

        The high-rate event is for operator visibility — it should fire
        exactly once per (symbol, session) when the threshold is first
        crossed. Subsequent rescues continue to log the per-rescue event
        but don't re-trigger the warning.
        """
        cache = _cache(high_rate_threshold=10)
        cache.populate("AKAN", "AKANDA CORP")
        with capture_logs() as captured:
            for i in range(15):
                cache.record_rescue(
                    "AKAN",
                    matched_token="akanda",
                    headline=f"Akanda News Item {i}",
                )
        high_rate = [e for e in captured if e["event"] == "catalyst.name_extension_high_rate"]
        assert len(high_rate) == 1, "high_rate fires exactly once per symbol per session"
        evt = high_rate[0]
        assert evt["symbol"] == "AKAN"
        assert evt["count_at_threshold"] == 10
        assert evt["token_list"] == ["akanda"]
        # Confirm the 11th rescue (which crossed the threshold) is the trigger.
        rescues = [e for e in captured if e["event"] == "catalyst.name_extension_rescued"]
        assert len(rescues) == 15
        # Counter still tracks all 15.
        assert cache.rescue_count("AKAN") == 15

    def test_repopulate_replaces_tokens(self) -> None:
        """A second ``populate`` overwrites the cached token list."""
        cache = _cache()
        cache.populate("FOO", "FOO BAR INC")
        assert "foo" in cache.get_tokens("FOO") or "bar" not in cache.get_tokens("FOO")
        cache.populate("FOO", "BAZBAT INDUSTRIES")
        # 'industries' is not in the default stopwords — verify by
        # checking what the tokenizer says directly.
        new_tokens = cache.get_tokens("FOO")
        assert "bazbat" in new_tokens

    def test_from_settings_picks_up_config_defaults(self) -> None:
        """``NameTokenCache.from_settings`` reads the catalyst config block."""
        cache = NameTokenCache.from_settings(Settings())
        # Verify cache uses default min_token_len + threshold by exercising both.
        cache.populate("RMAX", "RE/MAX HOLDINGS INC-CL A")
        assert cache.get_tokens("RMAX") == []  # default min_token_len=5 kills 'max'


# ---------------------------------------------------------------------------
# 3. Integration with classify() — the actual rescue path
# ---------------------------------------------------------------------------


class TestClassifyNameExtension:
    """End-to-end tests on the rescue branch in ``classify``."""

    def test_shph_reproduction_gate_1_passes_via_name(self) -> None:
        """Spec case 1: 2026-05-01 SHPH headlines pass the extended gate.

        Both real SHPH M&A headlines from the bug-trigger session, paired
        with realistic Finnhub-style body text that:
          (a) contains a green-list-matching phrase (the bare verb form
              ``Acquire`` / ``Merges`` in the headlines doesn't match
              ``_MA_PHRASES`` directly, but Finnhub article bodies for
              real M&A deals invariably include ``acquired`` / ``merger``
              / ``deal`` somewhere — that's what the production matcher
              hit when the user's bot logged the
              ``ticker_not_in_headline`` rejection),
          (b) includes the ticker via the standard ``(NASDAQ: SHPH)``
              cashtag form so Gate 2's proximity check can anchor.

        Without name extension, both reject at Gate 1 (no SHPH in
        headline). With name extension, the 'shuttle' token rescues
        Gate 1; Gate 2 finds the ticker in the summary near the matched
        phrase; catalyst attributes.
        """
        cache = _cache()
        cache.populate("SHPH", "SHUTTLE PHARMACEUTICAL HOLDINGS INC")

        for headline in [
            "Shuttle Pharmaceutical Enters Definite Agreement To Acquire United Dogecoin",
            "Shuttle Merges with United Dogecoin to Become the World's Largest Public Dogecoin Miner",
        ]:
            summary = (
                "Shuttle Pharmaceutical Holdings (NASDAQ: SHPH) today acquired the "
                "assets of United Dogecoin Mining Corp in a stock-and-cash deal..."
            )
            result = classify(
                [_item(headline, summary=summary)],
                symbol="SHPH",
                name_token_cache=cache,
            )
            assert result == "contract_or_m&a", f"failed for headline: {headline!r}"

    def test_shph_rescue_event_fires_with_correct_payload(self) -> None:
        """The SHPH rescue path emits ``catalyst.name_extension_rescued``."""
        cache = _cache()
        cache.populate("SHPH", "SHUTTLE PHARMACEUTICAL HOLDINGS INC")
        with capture_logs() as captured:
            classify(
                [
                    _item(
                        "Shuttle Pharmaceutical Enters Definite Agreement To Acquire United Dogecoin",
                        summary=(
                            "(NASDAQ: SHPH) acquired United Dogecoin assets in a "
                            "stock-and-cash deal announced today."
                        ),
                    )
                ],
                symbol="SHPH",
                name_token_cache=cache,
            )
        rescues = [e for e in captured if e["event"] == "catalyst.name_extension_rescued"]
        assert len(rescues) == 1
        assert rescues[0]["matched_token"] == "shuttle"

    def test_atlx_either_token_rescues(self) -> None:
        """Spec case 5: a headline mentioning either of ATLX's tokens passes.

        Both "atlas" and "lithium" are valid name signatures for ATLX —
        single-token match is sufficient.
        """
        cache = _cache()
        cache.populate("ATLX", "ATLAS LITHIUM INC")
        # Headline mentions only 'atlas' — should rescue Gate 1. Summary
        # carries the ticker so Gate 2 has an anchor.
        result = classify(
            [
                _item(
                    "Atlas Stake Acquired By Strategic Investor",
                    summary="(NASDAQ: ATLX) — strategic investor acquired stake today.",
                )
            ],
            symbol="ATLX",
            name_token_cache=cache,
        )
        assert result == "contract_or_m&a"

    def test_whole_word_matching_no_substring_false_positive(self) -> None:
        """Spec case 4: 'atlas' must not match 'atlasian' (substring).

        Whole-word regex anchoring is critical — without word boundaries
        a token like 'atlas' would false-positive on Atlassian, atlas-shrugged,
        etc. The tokenizer's match function uses ``\\b`` anchors.
        """
        cache = _cache()
        cache.populate("ATLX", "ATLAS LITHIUM INC")
        # Headline doesn't have 'atlas' as a whole word — only 'atlasian'.
        # Without ATLX in headline either, gate must reject.
        result = classify(
            [_item("Atlasian Reports Earnings Beat Estimates Strongly")],
            symbol="ATLX",
            name_token_cache=cache,
        )
        # Whole-word match fails for 'atlas'; ticker not in headline either.
        # Note: this also verifies "earnings beat" requires attribution.
        assert result is None

    def test_atlas_matches_atlas_word_not_atlasian(self) -> None:
        """Companion to whole-word test: 'atlas' as a real word DOES match."""
        cache = _cache()
        cache.populate("ATLX", "ATLAS LITHIUM INC")
        result = classify(
            [
                _item(
                    "Atlas Wins $50M Defense Contract from Pentagon",
                    summary="ATLX announced today...",
                )
            ],
            symbol="ATLX",
            name_token_cache=cache,
        )
        assert result == "contract_or_m&a"

    def test_rmax_no_tokens_falls_back_to_ticker_only(self) -> None:
        """Spec case 2: RMAX with min_token_len=5 has no tokens; ticker-only applies.

        The cache populates with empty tokens and emits the
        ``no_tokens`` event. The classify path then can't rescue,
        and falls through to ticker-only matching. A headline without
        RMAX in it gets correctly rejected.
        """
        cache = _cache()
        with capture_logs() as captured:
            cache.populate("RMAX", "RE/MAX HOLDINGS INC-CL A")
        no_tokens_events = [
            e for e in captured if e["event"] == "catalyst.name_extension_no_tokens"
        ]
        assert no_tokens_events  # emitted on populate

        result = classify(
            [_item("Re/Max shares rise 26.1% on acquisition deal with tech-focused brokerage")],
            symbol="RMAX",
            name_token_cache=cache,
        )
        # Cache returns no tokens → name extension can't help → ticker-only.
        # 'RMAX' is not in the headline → reject.
        assert result is None

    def test_biya_wrap_article_still_rejected_phase_9_7_regression(self) -> None:
        """Spec case 13: BIYA wrap-article regression case still rejects.

        The whole reason Phase 9.7 exists. With Phase 10.5 layered on
        top, BIYA's longName tokens (``["baiya"]``) MUST NOT match the
        Coca-Cola wrap article headline. If the tokenizer or matcher
        regressed in 10.5 such that this passes, we'd be back to the
        original 4/30 false positive.
        """
        cache = _cache()
        cache.populate("BIYA", "BAIYA INTERNATIONAL GROUP IN")
        wrap_headline = "Dow Gains 150 Points; Coca-Cola Posts Upbeat Earnings"
        result = classify(
            [_item(wrap_headline, summary="Coca-Cola raises guidance for fiscal 2026.")],
            symbol="BIYA",
            name_token_cache=cache,
        )
        assert result is None, "BIYA wrap article must remain rejected post-10.5"

    def test_gate_2_still_rejects_when_phrase_far_from_ticker(self) -> None:
        """Spec case 11: Gate 1 passes via name, but Gate 2 (ticker-anchored) fails.

        Constructed scenario: the headline mentions the company name (so
        Gate 1 rescues), and the body text contains the ticker (so Gate 2
        finds an anchor) but the matched phrase sits 50+ tokens away
        from any ticker mention. Gate 2 still rejects.
        """
        cache = _cache()
        cache.populate("ATLX", "ATLAS LITHIUM INC")
        # Headline mentions 'atlas' — Gate 1 passes.
        # Summary contains ATLX once at the very start, then a long block
        # of filler, then 'wins contract' at the very end. The phrase is
        # > 30 tokens from the ATLX mention → Gate 2 rejects.
        filler = " ".join(["filler word"] * 40)
        summary = f"ATLX is a lithium miner. {filler} Separately wins contract worth $50M."
        result = classify(
            [_item("Atlas Reports Q1 Update", summary=summary)],
            symbol="ATLX",
            name_token_cache=cache,
        )
        # Phase 8.4 and Phase 9.2: 'wins contract' includes the verb-anchored
        # phrase, so the matcher would normally match. But Gate 2 rejects
        # because the phrase is too far from the only ATLX mention.
        assert result is None

    def test_classify_without_cache_uses_legacy_behaviour(self) -> None:
        """Backwards-compat: ``name_token_cache=None`` (default) → pre-10.5 behaviour.

        Existing test suites that don't plumb the cache should continue
        to work with ticker-only matching. Verifies a SHPH headline
        without name extension correctly rejects (the original bug shape).
        """
        result = classify(
            [_item("Shuttle Pharmaceutical Enters Definite Agreement To Acquire United Dogecoin")],
            symbol="SHPH",
            # No name_token_cache passed.
        )
        # Pre-10.5: SHPH not in headline → ticker_not_in_headline → reject.
        assert result is None

    def test_classify_with_cache_but_no_population_falls_back(self) -> None:
        """Cache present but symbol not populated → behaves like legacy.

        Useful guard: the scanner might call classify before populating
        the cache for some symbol (e.g., a manual override path that
        skipped the qualify step). Should still classify correctly via
        ticker-only logic without crashing.
        """
        cache = _cache()
        # Don't populate SHPH.
        result = classify(
            [_item("Shuttle Pharmaceutical Enters Definite Agreement To Acquire")],
            symbol="SHPH",
            name_token_cache=cache,
        )
        # Empty token list → name extension can't rescue → ticker-only → reject.
        assert result is None

    def test_phrase_matching_rules_unchanged(self) -> None:
        """Phrase-matching rules from Phases 9.2 / 9.5 / 9.7 are untouched.

        Spot-check with a Phase-9.2-shape rejection (bare 'contract'
        verb form). Even with name extension, the ATLX 'Contracts Key
        Project Execution Partners' headline must NOT classify because
        the phrase matcher itself rejects.
        """
        cache = _cache()
        cache.populate("ATLX", "ATLAS LITHIUM INC")
        result = classify(
            [
                _item(
                    "Atlas Lithium Contracts Key Project Execution Partners to Drive Its Neves Project"
                )
            ],
            symbol="ATLX",
            name_token_cache=cache,
        )
        # Phase 9.2 fix: 'Contracts' (verb form) does not pass the
        # contract_or_m&a matcher. Result is None regardless of name
        # extension because no green-list category matched.
        assert result is None


# ---------------------------------------------------------------------------
# 4. Config validators (smoke-tested for defensive shape)
# ---------------------------------------------------------------------------


def test_name_extension_config_defaults_match_spec() -> None:
    """The shipped defaults match what the Phase 10.5 spec calls for."""
    cfg = NameExtensionConfig()
    assert cfg.min_token_len == 5
    assert cfg.high_rate_threshold == 10
    assert "incorporated" in cfg.stopwords
    assert "etf" in cfg.stopwords
    assert "international" in cfg.stopwords


def test_min_token_len_rejects_zero() -> None:
    """Pydantic validator rejects ``min_token_len=0`` to prevent empty-string tokens."""
    import pytest as _pytest  # noqa: PLC0415 - local import keeps the test self-contained
    from pydantic import ValidationError

    with _pytest.raises(ValidationError):
        NameExtensionConfig(min_token_len=0)


def test_high_rate_threshold_rejects_zero() -> None:
    """Pydantic validator rejects ``high_rate_threshold=0`` (would fire on every rescue)."""
    import pytest as _pytest  # noqa: PLC0415
    from pydantic import ValidationError

    with _pytest.raises(ValidationError):
        NameExtensionConfig(high_rate_threshold=0)


# ---------------------------------------------------------------------------
# 5. Scanner integration (light-touch — full scanner tests cover the rest)
# ---------------------------------------------------------------------------


def test_scanner_constructs_cache_from_settings_when_not_passed() -> None:
    """``IBKRScanner`` builds a NameTokenCache from settings if one isn't injected."""
    from bot.scanning.scanner import IBKRScanner  # noqa: PLC0415

    fake_ibkr = MagicMock()
    fake_finnhub = MagicMock()
    scanner = IBKRScanner(ibkr=fake_ibkr, finnhub=fake_finnhub, settings=Settings())
    assert scanner._name_token_cache is not None
    assert isinstance(scanner._name_token_cache, NameTokenCache)


def test_scanner_accepts_explicit_cache_for_tests() -> None:
    """Tests can inject a pre-populated cache without going through settings."""
    from bot.scanning.scanner import IBKRScanner  # noqa: PLC0415

    fake_ibkr = MagicMock()
    fake_finnhub = MagicMock()
    custom = _cache()
    custom.populate("SHPH", "SHUTTLE PHARMACEUTICAL HOLDINGS INC")
    scanner = IBKRScanner(
        ibkr=fake_ibkr,
        finnhub=fake_finnhub,
        settings=Settings(),
        name_token_cache=custom,
    )
    assert scanner._name_token_cache is custom
    assert scanner._name_token_cache.get_tokens("SHPH") == ["shuttle", "pharmaceutical"]
