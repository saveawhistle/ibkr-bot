"""Tests for ``bot.scanning.catalyst.classify`` — 5 green-list categories + 3 negative cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from structlog.testing import capture_logs

from bot.scanning.catalyst import classify
from bot.scanning.finnhub_client import NewsItem


def _item(
    headline: str,
    summary: str = "",
    *,
    when: datetime | None = None,
) -> NewsItem:
    """Build a minimal NewsItem for classifier input."""
    return NewsItem(
        headline=headline,
        source="test",
        url="https://example.com",
        datetime=when or datetime(2026, 4, 16, tzinfo=UTC),
        summary=summary,
        category="company",
    )


def test_earnings_beat_category() -> None:
    """A headline containing an earnings-beat phrase maps to earnings_beat."""
    assert classify([_item("ACME tops estimates on Q1 earnings")]) == "earnings_beat"


def test_clinical_category() -> None:
    """A headline mentioning an FDA approval maps to clinical."""
    assert classify([_item("BIOTECH announces FDA approval of lead asset")]) == "clinical"


def test_contract_or_ma_category() -> None:
    """A headline about an acquisition maps to contract_or_m&a."""
    assert classify([_item("LARGECO acquires SMALLCO in $2B deal")]) == "contract_or_m&a"


def test_analyst_upgrade_category() -> None:
    """A headline about a ratings upgrade maps to analyst_upgrade."""
    assert (
        classify([_item("Morgan Stanley upgrade: ACME raised to overweight")]) == "analyst_upgrade"
    )


def test_match_looks_at_summary_not_just_headline() -> None:
    """Matches must hit against both headline and summary text (phase 3 positive)."""
    item = _item(
        headline="ACME pipeline update",
        summary="Company announced results from its phase 3 trial today.",
    )
    assert classify([item]) == "clinical"


def test_black_list_hit_returns_none() -> None:
    """A black-listed phrase with no green match returns None (no catalyst)."""
    assert classify([_item("ACME announces dilution via secondary offering")]) is None


def test_no_match_returns_none() -> None:
    """Generic filler with no green or black keyword returns None."""
    assert classify([_item("ACME CEO speaks at industry conference about market outlook")]) is None


def test_black_list_overrides_green_list() -> None:
    """When a black-listed phrase co-occurs with a green match, black wins — return None."""
    item = _item(
        headline="ACME wins FDA approval",
        summary="Company also disclosed plans for a reverse split in Q3.",
    )
    assert classify([item]) is None


# ---------- Phase 5.1 timestamp filter ---------- #


def test_classify_without_max_age_keeps_all_items() -> None:
    """Backwards-compat: no ``max_age_hours`` means no filtering, matching pre-5.1 behavior."""
    stale = _item("ACME tops estimates", when=datetime(2020, 1, 1, tzinfo=UTC))
    assert classify([stale]) == "earnings_beat"


def test_classify_filters_stale_news_beyond_max_age() -> None:
    """A green-list headline older than ``max_age_hours`` is ignored → None."""
    reference = datetime(2026, 4, 20, 9, 30, tzinfo=UTC)
    stale = _item(
        "ACME tops estimates",
        when=reference - timedelta(hours=80),  # outside a 72-hour window
    )
    assert (
        classify(
            [stale],
            max_age_hours=72,
            reference_time=reference,
        )
        is None
    )


def test_classify_keeps_news_within_max_age() -> None:
    """A green-list headline inside the window still classifies normally."""
    reference = datetime(2026, 4, 20, 9, 30, tzinfo=UTC)
    fresh = _item(
        "ACME tops estimates",
        when=reference - timedelta(hours=60),  # inside the 72-hour window
    )
    assert (
        classify(
            [fresh],
            max_age_hours=72,
            reference_time=reference,
        )
        == "earnings_beat"
    )


def test_classify_filtered_stale_news_log_event_fires() -> None:
    """Dropped items emit ``catalyst.filtered_stale_news`` with counts + symbol."""
    reference = datetime(2026, 4, 20, 9, 30, tzinfo=UTC)
    stale = _item("ACME tops estimates", when=reference - timedelta(hours=80))
    fresh = _item("ACME signs new contract", when=reference - timedelta(hours=2))
    with capture_logs() as captured:
        classify(
            [stale, fresh],
            max_age_hours=72,
            reference_time=reference,
            symbol="ACME",
        )
    matching = [e for e in captured if e.get("event") == "catalyst.filtered_stale_news"]
    assert matching, "expected catalyst.filtered_stale_news log event"
    event = matching[0]
    assert event["symbol"] == "ACME"
    assert event["filtered"] == 1
    assert event["kept"] == 1


def test_classify_mixed_fresh_and_stale_weekend_scenario() -> None:
    """Saturday EO catalyst (within 72h) survives; stale Thursday dilution (also within) wins blacklist.

    Models the 2026-04-20 ENVB scenario with a blacklist item thrown in:
    reference_time is Monday 10:00 ET, a Saturday green-list headline is
    fresh enough to match, but a Thursday dilution item is *also* fresh
    and the blacklist overrides → None.
    """
    reference = datetime(2026, 4, 20, 14, 0, tzinfo=UTC)  # ~10:00 ET Monday
    saturday_green = _item(
        "ACME signs new contract with DOE",
        when=reference - timedelta(hours=48),
    )
    thursday_black = _item(
        "ACME announces offering",
        when=reference - timedelta(hours=96),  # outside 72h — filtered out
    )
    assert (
        classify(
            [saturday_green, thursday_black],
            max_age_hours=72,
            reference_time=reference,
            symbol="ACME",
        )
        == "contract_or_m&a"
    )


# ---------- Phase 5.2 item-level scoping + regulatory bucket ---------- #


def test_item_level_blacklist_does_not_poison_other_items() -> None:
    """Phase 5.2: a black-listed item is dropped individually; other items still classify.

    Under Phase 1's document-level concatenation this returned None because
    ``offering`` poisoned the whole corpus. Item-level scoping lets the
    clean FDA-approval item still match ``clinical``.
    """
    green = _item(headline="ACME wins FDA approval of lead asset")
    black = _item(headline="ACME announces dilution via secondary offering")
    assert classify([green, black]) == "clinical"


def test_item_matched_event_fires_with_category_and_phrase() -> None:
    """Each green-matching item emits ``catalyst.item_matched`` with category + phrase."""
    item = _item(headline="ACME tops estimates on Q1 earnings")
    with capture_logs() as captured:
        classify([item], symbol="ACME")
    matched = [e for e in captured if e.get("event") == "catalyst.item_matched"]
    assert matched, "expected catalyst.item_matched event"
    event = matched[0]
    assert event["symbol"] == "ACME"
    assert event["category"] == "earnings_beat"
    assert event["matched_phrase"] == "tops estimates"


def test_item_blacklisted_event_fires_with_matched_phrase() -> None:
    """Black-listed items emit ``catalyst.item_blacklisted`` with the offending phrase."""
    item = _item(headline="ACME announces dilution via secondary offering")
    with capture_logs() as captured:
        classify([item], symbol="ACME")
    blacklisted = [e for e in captured if e.get("event") == "catalyst.item_blacklisted"]
    assert blacklisted, "expected catalyst.item_blacklisted event"
    event = blacklisted[0]
    assert event["symbol"] == "ACME"
    # "dilution" is earlier in the black list than "offering" — either is acceptable,
    # but we assert the matched_phrase is one of the actual black-list entries.
    assert event["matched_phrase"] in {"dilution", "offering"}


def test_generic_executive_order_no_regulatory_match() -> None:
    """An EO with no drug/FDA/biotech Group-B anchor must NOT classify as regulatory."""
    item = _item(headline="Trump signs executive order on tariffs")
    assert classify([item]) is None


def test_layoffs_expedite_no_regulatory_match() -> None:
    """``expedite`` alone is not enough — requires a Group-B domain anchor."""
    item = _item(headline="Company expedites layoffs and restructuring")
    assert classify([item]) is None


def test_regulatory_requires_both_group_a_and_group_b() -> None:
    """Group B alone (e.g. ``fda`` in an earnings context) does NOT match regulatory."""
    item = _item(headline="FDA spokesperson comments on agency staffing")
    assert classify([item]) is None


def test_regulatory_matches_fast_track_with_biotech_anchor() -> None:
    """``fast-track`` (Group A) + ``biotech`` (Group B) anchors the regulatory bucket."""
    item = _item(headline="Regulator fast-tracks biotech review of rare-disease therapy")
    assert classify([item]) == "regulatory"


def test_regulatory_matches_rescheduling_with_dea_anchor() -> None:
    """``rescheduling`` (Group A) + ``dea`` (Group B) captures DEA schedule changes."""
    item = _item(headline="DEA rescheduling of psilocybin advances to public-comment phase")
    assert classify([item]) == "regulatory"


class TestEnvbCanonicalCases:
    """Regression suite for the 2026-04-20 ENVB catalyst miss — the four real items.

    Item texts are copied verbatim from the 2026-04-20 Finnhub diagnostic run
    (see ``_diag_envb.py`` transcript in conversation history). The mixed-corpus
    test is the critical proof of Phase 5.2: the regulatory catalyst items
    survive alongside the dilutive placement item instead of being masked.
    """

    def test_classifies_trump_psychedelic_eo_as_regulatory(self) -> None:
        """Item 4 — Mon 2:36 AM ET Finnhub article driving ENVB's +92% premarket."""
        item = _item(
            headline="Psychedelic drug makers rally as Trump orders FDA to expedite reviews",
            summary=(
                "Shares of psychedelic drug developers rose in premarket trading on "
                "Monday after U.S. President Donald Trump signed an executive order "
                "directing health regulators to speed up reviews of psychedelic..."
            ),
        )
        assert classify([item]) == "regulatory"

    def test_classifies_cbs_eo_scoop_as_regulatory(self) -> None:
        """Item 18 — Thu 7:56 AM ET Benzinga/CBS scoop of the same EO."""
        item = _item(
            headline=(
                "'Trump To Sign Executive Order On Psychedelic Drug Used Abroad "
                "To Treat PTSD' - CBS News Exclusive"
            ),
            summary=(
                "https://www.cbsnews.com/news/psychedelic-drug-ibogaine-ptsd-"
                "trump-to-sign-executive-order/"
            ),
        )
        assert classify([item]) == "regulatory"

    def test_private_placement_correctly_returns_none(self) -> None:
        """Item 6 — Friday's $13.9M private placement; no green match (dilutive)."""
        item = _item(
            headline=(
                "Enveric Biosciences announces closing of up to $13.9 million "
                "private placement priced at-the-market under Nasdaq rules"
            ),
        )
        assert classify([item]) is None

    def test_aggregator_article_no_match(self) -> None:
        """Item 1 — ChartMill aggregator; no green phrase, no black phrase."""
        item = _item(
            headline="Top movers in Monday's pre-market session",
            summary=(
                "Before the opening bell on Monday, let's take a glimpse of the US "
                "markets and explore the top gainers and losers in today's pre-market session."
            ),
        )
        assert classify([item]) is None

    def test_envb_mixed_corpus_returns_regulatory(self) -> None:
        """CRITICAL — item-level scoping lets regulatory wins survive alongside a dilutive item.

        This is the Phase 5.2 raison d'être. Under Phase 1's document-level
        classifier the corpus returned None because the private-placement
        item's keywords poisoned the blob. With item-level evaluation the
        regulatory EO items (4 and 18) classify correctly even when the
        placement item is in the same list.
        """
        trump_eo = _item(
            headline="Psychedelic drug makers rally as Trump orders FDA to expedite reviews",
            summary=(
                "Shares of psychedelic drug developers rose in premarket trading on "
                "Monday after U.S. President Donald Trump signed an executive order "
                "directing health regulators to speed up reviews of psychedelic..."
            ),
        )
        cbs_scoop = _item(
            headline=(
                "'Trump To Sign Executive Order On Psychedelic Drug Used Abroad "
                "To Treat PTSD' - CBS News Exclusive"
            ),
        )
        placement = _item(
            headline=(
                "Enveric Biosciences announces closing of up to $13.9 million "
                "private placement priced at-the-market under Nasdaq rules"
            ),
        )
        aggregator = _item(headline="Top movers in Monday's pre-market session")
        assert classify([trump_eo, cbs_scoop, placement, aggregator]) == "regulatory"


# ---------- Phase 9.2: contract category requires award context ---------- #


def test_contract_award_phrase_matches_awarded_contract() -> None:
    """``Awarded $50M Defense Contract`` — verb + dollar amount + contract."""
    assert classify([_item("Company X Awarded $50M Defense Contract")]) == "contract_or_m&a"


def test_contract_award_phrase_matches_wins_contract() -> None:
    """``Wins Major Contract`` — verb + adjective + contract."""
    assert classify([_item("Company X Wins Major Contract from Government")]) == "contract_or_m&a"


def test_contract_award_phrase_matches_signs_contract() -> None:
    """``Signs Multi-Year Contract`` — verb + adjective + contract."""
    assert (
        classify([_item("Company X Signs Multi-Year Contract with Customer Y")])
        == "contract_or_m&a"
    )


def test_contract_award_phrase_matches_dollar_value() -> None:
    """``Receives $25 Million Contract`` — verb + dollar amount + contract."""
    assert classify([_item("Company X Receives $25 Million Contract")]) == "contract_or_m&a"


def test_contract_award_phrase_matches_contract_worth() -> None:
    """``Contract Worth $100M`` — noun-phrase anchor."""
    assert classify([_item("Company X Announces Contract Worth $100M")]) == "contract_or_m&a"


def test_contract_verb_form_does_not_match_atlx_pattern() -> None:
    """ATLX 2026-04-27 false positive — verb-form ``Contracts`` must NOT match.

    Pre-9.2 the bare ``"contract"`` substring in ``"contracts"`` triggered
    a contract_or_m&a classification on a headline where the company was
    *hiring* service providers (issuer, not recipient). This contributed
    to onboarding ATLX onto the watchlist and a -$18.84 trade.
    """
    assert (
        classify(
            [
                _item(
                    "Atlas Lithium Contracts Key Project Execution Partners to "
                    "Drive Its Neves Project Toward Production"
                )
            ]
        )
        is None
    )


def test_contract_verb_form_does_not_match_hires_pattern() -> None:
    """``Contracts Engineering Firm`` — verb form, company is the issuer."""
    assert classify([_item("Company X Contracts Engineering Firm for Project")]) is None


def test_contractor_word_does_not_match() -> None:
    """``Contractor`` is operational staffing, not a tradeable award."""
    assert classify([_item("Company X Names New Contractor for Operations")]) is None


def test_contract_negotiations_does_not_match() -> None:
    """``Enters Contract Negotiations`` — no award yet."""
    assert classify([_item("Company X Enters Contract Negotiations")]) is None


def test_acquisition_phrase_still_matches() -> None:
    """M&A category preserved — ``acquires`` keyword still classifies."""
    assert classify([_item("Company X Acquires Company Y")]) == "contract_or_m&a"


def test_merger_phrase_still_matches() -> None:
    """M&A category preserved — ``merger`` keyword still classifies."""
    assert classify([_item("Company X and Company Y Announce Merger")]) == "contract_or_m&a"


def test_contract_match_logs_specific_phrase_not_bare_word() -> None:
    """``catalyst.item_matched`` must report the verb-anchored span, not bare ``contract``.

    Pre-9.2 logged ``matched_phrase: "contract"`` — too generic for triage.
    The verb-anchored span lets an operator grep for the exact award pattern.
    """
    item = _item("Company X Wins $50M Defense Contract from Pentagon")
    with capture_logs() as captured:
        classify([item], symbol="X")
    matched = [e for e in captured if e.get("event") == "catalyst.item_matched"]
    assert matched
    event = matched[0]
    assert event["category"] == "contract_or_m&a"
    # Span includes the verb + dollar amount + "contract"; not the bare word.
    assert event["matched_phrase"] != "contract"
    assert "wins" in event["matched_phrase"]
    assert "contract" in event["matched_phrase"]


# ---------- Phase 9.5: earnings_beat record/growth axis + tightened upside ---------- #


def test_record_revenue_yoy_growth_matches_sagt_pattern() -> None:
    """SAGT 2026-04-29 verbatim — record revenue + YoY growth must classify earnings_beat.

    Pre-9.5 the bare phrase set ("earnings beat", "beats estimates", "tops
    estimates", "raises guidance", "upside") missed this framing entirely
    and SAGT was dropped at scanner.dropped_no_catalyst despite a clean
    49% YoY growth + record-revenue earnings catalyst.
    """
    item = _item(
        "Sagtec Global Limited Achieves Record Revenue of US$19.1 Million "
        "in Fiscal Year 2025, Marking 49% Year-over-Year Growth"
    )
    assert classify([item]) == "earnings_beat"


def test_record_revenue_alone_matches() -> None:
    """``Reports Record Revenue`` — substring anchor in the record axis."""
    assert classify([_item("Company X Reports Record Revenue for Q4")]) == "earnings_beat"


def test_record_quarterly_earnings_matches() -> None:
    """``Posts Record Quarterly Earnings`` — record-axis substring."""
    assert classify([_item("Company X Posts Record Quarterly Earnings")]) == "earnings_beat"


def test_yoy_growth_with_percentage_matches() -> None:
    """``50% Year-over-Year Growth`` — YoY regex with numeric anchor."""
    assert classify([_item("Company X Sees 50% Year-over-Year Growth")]) == "earnings_beat"


def test_yoy_increase_with_percentage_matches() -> None:
    """``30% YoY Increase in Revenue`` — YoY regex, abbreviated form."""
    assert classify([_item("Company X Reports 30% YoY Increase in Revenue")]) == "earnings_beat"


def test_existing_beats_estimates_still_matches() -> None:
    """Regression: pre-9.5 ``tops estimates`` headline still classifies."""
    assert classify([_item("ACME tops estimates on Q1 earnings")]) == "earnings_beat"


def test_existing_raises_guidance_still_matches() -> None:
    """Regression: ``raises guidance`` headline still classifies."""
    assert classify([_item("ACME raises guidance for the full year")]) == "earnings_beat"


def test_exceeds_estimates_now_matches() -> None:
    """New variant ``exceeds estimates`` (added in Phase 9.5) classifies."""
    assert classify([_item("Company X Exceeds Estimates on Strong Demand")]) == "earnings_beat"


def test_beat_expectations_now_matches() -> None:
    """New variant ``beat expectations`` (past-tense) classifies."""
    assert classify([_item("Company X Beat Expectations with $0.50 EPS")]) == "earnings_beat"


def test_limited_upside_does_not_match() -> None:
    """``Limited Upside`` is bearish analyst language — must NOT classify earnings_beat.

    Pre-9.5 the bare ``"upside"`` keyword fired here. Phase 9.5 dropped
    bare upside in favour of anchored variants (``earnings upside``,
    ``upside surprise``, ``surprise to the upside``, ``upside to estimates``).
    """
    assert classify([_item("Analysts See Limited Upside for Company X")]) is None


def test_upside_potential_does_not_match() -> None:
    """``Upside Potential`` is speculative analyst language, not actual results."""
    assert classify([_item("Company X Has Upside Potential According to Analysts")]) is None


def test_negative_yoy_does_not_match_growth() -> None:
    """``30% Year-over-Year Decline`` — ``decline`` not in positive verb list."""
    assert classify([_item("Company X Reports 30% Year-over-Year Decline")]) is None


def test_misses_estimates_does_not_match() -> None:
    """Bearish earnings (``misses estimates``) intentionally not in the green list."""
    assert classify([_item("Company X Misses Estimates")]) is None


def test_lowered_guidance_does_not_match() -> None:
    """Bearish guidance (``lowers guidance``) intentionally not in the green list."""
    assert classify([_item("Company X Lowers Guidance")]) is None


def test_yoy_growth_without_percentage_does_not_match() -> None:
    """``Year-over-Year Growth`` without a numeric anchor must NOT classify.

    The YoY regex requires a leading percentage so vague "growth"
    mentions don't become a catalyst. A press release citing actual
    growth almost always quantifies it.
    """
    assert classify([_item("Company X Sees Year-over-Year Growth")]) is None


def test_record_high_does_not_match() -> None:
    """``Record High`` is price-action chatter, not an earnings catalyst.

    The record-axis phrases are anchored on ``revenue``/``earnings``/
    ``profit``/``sales``, so ``record high`` falls through.
    """
    assert classify([_item("Stock Hits Record High")]) is None


def test_earnings_beat_match_logs_specific_phrase() -> None:
    """``catalyst.item_matched`` reports the matched span, not bare ``earnings_beat``.

    Headline carries the ``(NASDAQ: SAGT)`` cashtag form so the Phase 9.7
    symbol-attribution gate accepts it. Real Finnhub press releases for
    sub-$20 names typically include this pattern verbatim.
    """
    item = _item(
        "Sagtec Global Limited (NASDAQ: SAGT) Achieves Record Revenue of US$19.1 "
        "Million in Fiscal Year 2025, Marking 49% Year-over-Year Growth"
    )
    with capture_logs() as captured:
        classify([item], symbol="SAGT")
    matched = [e for e in captured if e.get("event") == "catalyst.item_matched"]
    assert matched
    event = matched[0]
    assert event["category"] == "earnings_beat"
    # Highest-priority phrase wins — ``record revenue`` comes before the
    # YoY regex in the matcher's iteration order.
    assert event["matched_phrase"] == "record revenue"


# ---------- Phase 9.7: symbol-attribution gate ---------- #


class TestSymbolAttributionGate:
    """Phase 9.7 — defends against Finnhub mistagging a wrap article to a ticker.

    2026-04-30 BIYA precedent: ``"Dow Gains 150 Points; Coca-Cola Posts Upbeat
    Earnings"`` was returned by Finnhub's ``company-news`` endpoint for BIYA
    and matched ``"raises guidance"`` in the body. The gate rejects this two
    ways: gate 1 (ticker missing from headline) and gate 2 (matched phrase
    far from any ticker mention).
    """

    def test_biya_2026_04_30_wrap_article_rejected(self) -> None:
        """BIYA regression — ``raises guidance`` in a Coca-Cola wrap article must NOT classify.

        Verbatim from session_2026-04-30.jsonl line 60: the headline is a
        generic Dow market-wrap, the matched phrase came from the body text
        (almost certainly referring to KO, not BIYA). Gate 1 rejects because
        ``BIYA`` is absent from the headline.
        """
        item = _item(
            headline="Dow Gains 150 Points; Coca-Cola Posts Upbeat Earnings",
            summary="Coca-Cola raises guidance for fiscal 2026 after a strong Q1.",
        )
        assert classify([item], symbol="BIYA") is None

    def test_ticker_in_headline_phrase_close_classifies(self) -> None:
        """Gate 1 + gate 2 both pass — earnings_beat fires normally."""
        item = _item("BIYA Reports Q1 2026 Earnings: Tops Estimates on Strong Demand")
        assert classify([item], symbol="BIYA") == "earnings_beat"

    def test_cashtag_form_in_headline_classifies(self) -> None:
        """``(NASDAQ: TICKER)`` boilerplate counts as a headline ticker mention.

        Most sub-$20 press releases follow ``Company X (NASDAQ: TICKER) ...``;
        the word-boundary regex matches the bare ticker inside the parens.
        """
        item = _item("Baiya International Group (NASDAQ: BIYA) Raises Guidance for FY26")
        assert classify([item], symbol="BIYA") == "earnings_beat"

    def test_ticker_only_in_summary_rejected_by_gate_1(self) -> None:
        """Headline missing the ticker, but summary mentions it — gate 1 still rejects.

        The user's first rule: "the headline to require mention of the
        symbol's name/ticker". A ticker buried in the body is not enough.
        """
        item = _item(
            headline="Markets Rally on Tech Strength",
            summary="Among today's gainers, BIYA tops estimates on Q1 earnings.",
        )
        assert classify([item], symbol="BIYA") is None

    def test_ticker_substring_inside_word_does_not_match(self) -> None:
        """Word-boundary regex prevents ``ARM`` from matching inside ``alarm``.

        Without the ``\\b…\\b`` anchors, a ticker like ``ARM`` would falsely
        satisfy gate 1 against any headline containing ``alarm`` / ``armor``.
        """
        item = _item(
            "Alarm Sounds on Sector Tops Estimates Reading",
            summary="alarm bells beat estimates today.",
        )
        assert classify([item], symbol="ARM") is None

    def test_phrase_too_far_from_ticker_rejected_by_gate_2(self) -> None:
        """Ticker in headline, matched phrase 60+ tokens away in body — gate 2 rejects.

        Models a long article that mentions the ticker once at the top and
        an unrelated earnings-beat phrase deep in the body. The headline
        passes gate 1 but proximity gate 2 fails.
        """
        # Build a long summary: ticker mention near the top, phrase far away.
        filler = " ".join(["lorem ipsum dolor sit amet"] * 20)  # ~100 tokens
        item = _item(
            headline="BIYA Q1 Update Released",
            summary=f"Other ticker mentioned briefly. {filler} elsewhere KO raises guidance today.",
        )
        # ~100 tokens between BIYA mention and ``raises guidance`` — well
        # outside the 30-token proximity window.
        assert classify([item], symbol="BIYA") is None

    def test_phrase_within_window_in_summary_passes(self) -> None:
        """Ticker in headline, matched phrase later in the same sentence in summary — passes."""
        item = _item(
            headline="BIYA Provides Business Update",
            summary="The company today announced it raises guidance for fiscal 2026.",
        )
        assert classify([item], symbol="BIYA") == "earnings_beat"

    def test_no_symbol_passed_skips_gate_legacy_behavior(self) -> None:
        """Backwards compat: ``symbol=None`` (or omitted) bypasses the gate.

        Pre-9.7 callers (and tests) that don't plumb a symbol keep their
        behaviour. The scanner always passes ``symbol``, so production
        traffic always runs the gate.
        """
        item = _item("Generic ACME tops estimates wrap article")
        # No symbol → gate skipped → matches as before.
        assert classify([item]) == "earnings_beat"

    def test_regulatory_exempt_from_gate_envb_eo_still_classifies(self) -> None:
        """Phase 9.7 deliberately exempts ``regulatory`` — sector-wide EO catalysts
        (ENVB 2026-04-20 precedent) by design don't name the affected ticker.

        The bucket already requires both Group A (policy verb) AND Group B
        (FDA / drug / biotech anchor) — that AND-match is itself a strong
        attribution filter.
        """
        item = _item(
            headline="Psychedelic drug makers rally as Trump orders FDA to expedite reviews",
            summary=(
                "Shares of psychedelic drug developers rose in premarket trading on "
                "Monday after the executive order directing health regulators to "
                "speed up reviews of psychedelic therapies."
            ),
        )
        # ENVB is not in the headline — but regulatory is exempt, so it
        # still classifies even with symbol provided.
        assert classify([item], symbol="ENVB") == "regulatory"

    def test_off_topic_rejection_emits_log_event(self) -> None:
        """Gate 1 rejection fires ``catalyst.item_matched_rejected_off_topic``."""
        item = _item(
            headline="Dow Gains 150 Points; Coca-Cola Posts Upbeat Earnings",
            summary="Coca-Cola raises guidance for fiscal 2026.",
        )
        with capture_logs() as captured:
            classify([item], symbol="BIYA")
        rejected = [
            e for e in captured if e.get("event") == "catalyst.item_matched_rejected_off_topic"
        ]
        assert rejected, "expected catalyst.item_matched_rejected_off_topic event"
        event = rejected[0]
        assert event["symbol"] == "BIYA"
        assert event["category"] == "earnings_beat"
        assert event["matched_phrase"] == "raises guidance"
        assert event["reason"] == "ticker_not_in_headline"

    def test_ticker_case_insensitive(self) -> None:
        """Lowercase ticker in headline still passes gate 1 (everything is lowercased)."""
        item = _item("biya tops estimates on Q1")
        assert classify([item], symbol="BIYA") == "earnings_beat"

    def test_proximity_rejection_lets_other_categories_try(self) -> None:
        """Gate 2 rejection on one category doesn't poison the item for other categories.

        Earnings_beat phrase is far from the ticker; clinical phrase is
        right next to it. The earnings_beat match is rejected by gate 2
        (logged), and the clinical match wins.
        """
        filler = " ".join(["filler word here"] * 25)  # push phrase out of window
        item = _item(
            headline="ACME Announces FDA Approval of Lead Asset",
            summary=f"{filler}. Separately, an analyst note today raises guidance on peers.",
        )
        # FDA approval is in the headline next to ACME → clinical passes.
        # ``raises guidance`` is far from ACME → earnings_beat rejected,
        # but classify still returns clinical because the loop continues.
        assert classify([item], symbol="ACME") == "clinical"
