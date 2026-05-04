"""Keyword-based catalyst classifier — Phase 5.2 item-level scoping + regulatory bucket.

Phase 1 shipped a document-level classifier: all news items for a symbol were
concatenated into one haystack, then checked against a black list and a
priority-ordered green list. Paper-trading day 1 (2026-04-20) proved that
pattern was wrong on two fronts:

* One black-listed item (e.g. a dilutive placement) poisoned the entire
  corpus, masking legitimate green matches on other items.
* No category covered sector/policy catalysts — a presidential EO directing
  the FDA to expedite drug reviews is a real bullish driver (ENVB +92%) but
  matched none of the company-specific green buckets.

Phase 5.2 rewrites ``classify`` to per-item evaluation and adds a
``regulatory`` bucket that AND-matches a policy-verb group against a
biotech/drug-context group, blocking false positives on generic
executive-order headlines (tariffs, layoffs, etc.).

Phase 9.7 adds a symbol-attribution gate to defend against Finnhub returning
generic market-wrap articles tagged to a ticker the article doesn't actually
discuss (2026-04-30 BIYA: ``"Dow Gains 150 Points; Coca-Cola Posts Upbeat
Earnings"`` matched ``"raises guidance"`` in the body, classifying BIYA as
``earnings_beat`` even though the article was about KO). The gate has two
parts and is applied only to *company-specific* categories (earnings_beat,
clinical, contract_or_m&a, analyst_upgrade) — ``regulatory`` is exempt by
design because sector-wide policy catalysts (ENVB EO precedent) intentionally
don't name the affected tickers.

  * Gate 1 — the ticker (word-boundary, case-insensitive) must appear in
    the *headline*. Generic wrap articles that mention only the macro
    benchmark or a different company fail here.
  * Gate 2 — the matched phrase must appear within ``_PROXIMITY_MAX_TOKENS``
    tokens of any ticker mention in headline + summary, so a long article
    with the ticker mentioned once at the top and an unrelated "raises
    guidance" two paragraphs down still rejects.

Both gates are skipped when ``symbol=None`` (so existing tests that don't
plumb a symbol keep working). The scanner always passes ``symbol`` so
production traffic always runs the gate.

Priority order (first match wins across surviving items):
    earnings_beat > clinical > contract_or_m&a > regulatory > analyst_upgrade

Phase 5.1's timestamp filter still runs first — stale items are dropped
before any per-item black/green evaluation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from bot.scanning.finnhub_client import NewsItem

if TYPE_CHECKING:
    from bot.config import Settings

_log = structlog.get_logger("bot.scanning.catalyst")

_Matcher = Callable[[str], str | None]
"""Lowercased-text matcher. Returns the matched phrase(s) for log forensics, else None."""


def _any_of(*phrases: str) -> _Matcher:
    """Build a matcher that returns the first phrase found in ``text`` (or None)."""

    def match(text: str) -> str | None:
        for phrase in phrases:
            if phrase in text:
                return phrase
        return None

    return match


def _both_groups(group_a: tuple[str, ...], group_b: tuple[str, ...]) -> _Matcher:
    """Build a matcher requiring one hit from each group; reports ``"<a> + <b>"``.

    Used by the ``regulatory`` bucket: Group A is the policy-action verb
    (``executive order``, ``expedite``, ``rescheduling``…), Group B is the
    drug/FDA/biotech context anchor. Both must land on the same item.
    """

    def match(text: str) -> str | None:
        hit_a = next((p for p in group_a if p in text), None)
        if hit_a is None:
            return None
        hit_b = next((p for p in group_b if p in text), None)
        if hit_b is None:
            return None
        return f"{hit_a} + {hit_b}"

    return match


# Phase 5.2 regulatory/policy keyword groups.
# Group A: the regulatory action verb/noun — the "something is changing" signal.
# Group B: the domain anchor — prevents generic-EO false positives (tariffs, layoffs).
_REG_GROUP_A: tuple[str, ...] = (
    "executive order",
    "expedite",
    "fast-track",
    "fast track",
    "accelerate review",
    "accelerate approval",
    "policy change",
    "rescheduling",
)
_REG_GROUP_B: tuple[str, ...] = (
    "fda",
    "dea",
    "drug",  # substring also covers "drugs"
    "biotech",
    "pharma",  # substring also covers "pharmaceutical"
    "medical",
    "health regulator",  # substring also covers "health regulators"
    "psychedelic",
    "clinical",
)

# Phase 9.2 — "contract" award context. Bare ``"contract"`` substring matched
# the verb form too: ATLX's "Atlas Lithium *Contracts* Key Project Execution
# Partners" classified as ``contract_or_m&a`` even though the company was the
# one *hiring* (false positive that helped onboard a -$18 trade on Day 6).
# Now we require either:
#   (a) a verb that implies the company *received* a contract, followed by
#       ``contract`` with up to four intervening words for adjectives /
#       dollar amounts ("wins major contract", "signs $50M defense contract"),
#       OR
#   (b) a noun phrase that anchors an award context ("contract worth",
#       "contract valued at", "contract award").
# Word boundary on ``\bcontract\b`` (singular) intentionally excludes
# ``contracts`` verb form and ``contractor`` operational staffing.
_CONTRACT_AWARD_PATTERN = re.compile(
    r"\b(?:awarded|wins|won|receives|received|secures|secured|"
    r"signs|signed|announces|announced|selected for)\b"
    r"(?:\s+\S+){0,4}?\s+\bcontract\b"
)
_CONTRACT_NOUN_PHRASES: tuple[str, ...] = (
    "contract worth",
    "contract valued at",
    "contract award",
)
_MA_PHRASES: tuple[str, ...] = (
    "partnership",
    "deal",
    "acquires",
    "acquired",
    "merger",
)


def _contract_or_ma_matcher(text: str) -> str | None:
    """Phase 9.2 — match ``contract_or_m&a`` only with award/M&A context.

    Returns the specific span that triggered the match so logs can
    distinguish ``"wins $50m defense contract"`` from ``"acquires"`` from
    ``"contract worth"``. Replaces the prior bare-substring matcher whose
    ``"contract"`` substring fired on ATLX's verb-form headline.
    """
    award_hit = _CONTRACT_AWARD_PATTERN.search(text)
    if award_hit is not None:
        return award_hit.group(0)
    for phrase in _CONTRACT_NOUN_PHRASES:
        if phrase in text:
            return phrase
    for phrase in _MA_PHRASES:
        if phrase in text:
            return phrase
    return None


# Phase 9.5 — "earnings_beat" expanded to cover the record/growth axis
# (SAGT 2026-04-29: "Achieves Record Revenue ... Marking 49% Year-over-Year
# Growth" was dropped at scanner because the prior keyword set only handled
# estimates-beat language). Same three-layer design as the contract matcher:
# (a) substring phrases for explicit "beat / raise / lift" framings,
# (b) substring phrases for "record revenue / earnings / profit / sales",
# (c) regex for "<N>% year-over-year <growth|increase|...>" so an unanchored
# ``"upside"`` (which matched bearish "limited upside" / "upside risk") is
# replaced with anchored variants ("earnings upside", "upside surprise").
_EARNINGS_BEAT_PHRASES: tuple[str, ...] = (
    # Estimates / expectations beat (existing, plus past-tense + variants).
    "earnings beat",
    "beats estimates",
    "beat estimates",
    "tops estimates",
    "topped estimates",
    "exceeds estimates",
    "exceeded estimates",
    "beats expectations",
    "beat expectations",
    "exceeds expectations",
    "exceeded expectations",
    # Guidance.
    "raises guidance",
    "raised guidance",
    "lifts guidance",
    "lifted guidance",
    # Upside — anchored to remove "limited upside" / "upside potential" /
    # "upside risk" false positives.
    "earnings upside",
    "upside surprise",
    "surprise to the upside",
    "upside to estimates",
    # Record / growth axis (new).
    "record revenue",
    "record quarterly revenue",
    "record earnings",
    "record quarterly earnings",
    "record profit",
    "record quarterly profit",
    "record sales",
    "record quarterly sales",
)

# "<N>% year-over-year <positive verb>" — captures "49% year-over-year growth",
# "50% YoY increase", "30% year-over-year jump in revenue", etc. The 0-30 char
# window allows for connector words ("growth in revenue", "increase in EPS").
# Negative directions ("decline", "drop", "decrease") deliberately omitted
# from the verb alternation so a negative-YoY framing won't classify.
_YOY_GROWTH_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%\s*(?:year[- ]over[- ]year|yoy)\b"
    r".{0,30}?"
    r"\b(?:growth|increase|gain|jump|rise|surge)\b"
)


def _earnings_beat_matcher(text: str) -> str | None:
    """Phase 9.5 — match ``earnings_beat`` across estimates/record/YoY framings.

    Returns the specific phrase or regex span that triggered so logs
    distinguish ``"record revenue"`` from ``"raised guidance"`` from
    ``"49% year-over-year growth"``. The bare ``"upside"`` keyword is
    intentionally absent — only anchored upside variants qualify.
    """
    for phrase in _EARNINGS_BEAT_PHRASES:
        if phrase in text:
            return phrase
    yoy_hit = _YOY_GROWTH_PATTERN.search(text)
    if yoy_hit is not None:
        return yoy_hit.group(0)
    return None


# Priority order: entries earlier in the tuple win ties on a single item.
# Phase 5.2 inserts ``regulatory`` between ``contract_or_m&a`` and
# ``analyst_upgrade`` — policy tailwinds are real catalysts but often
# drive sympathy chop, so they rank below company-specific deals.
#
# Phase 9.7 — third tuple element ``requires_symbol_attribution`` toggles
# the headline-ticker + phrase-proximity gate. Company-specific buckets
# require attribution (defends against Finnhub mistagging a wrap article
# to an unrelated ticker, 2026-04-30 BIYA precedent). ``regulatory``
# opts out: sector-wide EO catalysts (ENVB 2026-04-20 precedent) by
# design don't mention the affected tickers, and the bucket already
# requires both a policy-verb (Group A) AND a domain anchor (Group B)
# match, which is itself a stringent attribution-equivalent filter.
_GREEN_LIST: tuple[tuple[str, _Matcher, bool], ...] = (
    (
        "earnings_beat",
        _earnings_beat_matcher,
        True,
    ),
    (
        "clinical",
        _any_of("fda approval", "fda clears", "phase 3", "clinical trial", "breakthrough"),
        True,
    ),
    (
        "contract_or_m&a",
        _contract_or_ma_matcher,
        True,
    ),
    (
        "regulatory",
        _both_groups(_REG_GROUP_A, _REG_GROUP_B),
        False,
    ),
    (
        "analyst_upgrade",
        _any_of("upgrade", "price target raised"),
        True,
    ),
)

_BLACK_LIST: tuple[str, ...] = (
    "reverse split",
    "offering",
    "dilution",
    "going concern",
    "bankruptcy",
    "delisting",
    "sec investigation",
    "fraud",
)

# Phase 6.8 — public list of valid green-category names for the manual
# catalyst override CLI. Derived from _GREEN_LIST so there is exactly one
# source of truth: adding a category to the classifier automatically
# makes it injectable, and renaming one surfaces as a test/validation
# error on any injections still referencing the old name.
VALID_CATEGORIES: frozenset[str] = frozenset(entry[0] for entry in _GREEN_LIST)

_HEADLINE_LOG_MAX = 100

# Phase 9.7 — symbol-attribution proximity window. A press release usually
# carries the ticker in the headline plus the first sentence of the summary;
# 30 whitespace-separated tokens covers that envelope generously without
# admitting matches buried two paragraphs into a 500-word body. Tightening
# (e.g. to 15) would start dropping legitimate headlines whose summary
# mentions the catalyst phrase before the ticker; loosening (e.g. to 100)
# defeats the gate's purpose for long wrap articles. Re-tune from session
# JSONLs by grepping ``catalyst.item_matched_rejected_off_topic`` and
# ``catalyst.item_skipped_off_topic_headline`` for false-negative shapes.
_PROXIMITY_MAX_TOKENS = 30


def _token_index_at(text: str, char_pos: int) -> int:
    """Return the index of the whitespace-delimited token containing ``char_pos``.

    Whitespace positions snap to the nearest preceding token (or 0 if the
    position falls before any token). Used by the Phase 9.7 proximity gate
    to convert character offsets (from ``str.find`` / regex match start)
    into token-distance comparable units.
    """
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return 0
    for i, t in enumerate(tokens):
        if t.start() <= char_pos < t.end():
            return i
        if t.start() > char_pos:
            return max(0, i - 1)
    return len(tokens) - 1


def _phrase_near_ticker(
    text_lc: str,
    matched_phrase: str,
    ticker_lc: str,
    *,
    max_tokens: int = _PROXIMITY_MAX_TOKENS,
) -> bool:
    """Phase 9.7 — True iff ``matched_phrase`` is within ``max_tokens`` of any ticker mention.

    Both ``text_lc`` and ``ticker_lc`` must already be lowercased (matches
    the existing ``classify`` normalisation). ``matched_phrase`` is the
    exact substring/span returned by a green-list matcher; we re-locate it
    via ``str.find`` because matchers don't currently expose offsets and
    re-finding the first occurrence is sufficient for this gate.
    """
    phrase_pos = text_lc.find(matched_phrase)
    if phrase_pos < 0:
        return False
    ticker_pat = re.compile(rf"\b{re.escape(ticker_lc)}\b")
    ticker_positions = [m.start() for m in ticker_pat.finditer(text_lc)]
    if not ticker_positions:
        return False
    phrase_tok = _token_index_at(text_lc, phrase_pos)
    return any(
        abs(_token_index_at(text_lc, tp) - phrase_tok) <= max_tokens
        for tp in ticker_positions
    )


def _ticker_in_headline(headline_lc: str, ticker_lc: str) -> bool:
    """Phase 9.7 — Gate 1: word-boundary match for the ticker in the headline."""
    return re.search(rf"\b{re.escape(ticker_lc)}\b", headline_lc) is not None


def tokenize_name(
    longname: str,
    stopwords: frozenset[str],
    min_token_len: int,
) -> list[str]:
    """Phase 10.5 — derive name-signature tokens from an IBKR longName.

    Splits on any run of non-alphanumeric characters (whitespace, hyphen,
    slash, comma, etc.) so e.g. ``RE/MAX HOLDINGS INC-CL A`` becomes
    ``["re", "max", "holdings", "inc", "cl", "a"]`` and ``ELECTRO-SENSORS
    INC`` becomes ``["electro", "sensors", "inc"]``.

    Then drops:
      * tokens shorter than ``min_token_len`` (default 5 — keeps "akanda"
        and "republic" but kills generic 3-char tokens like "max" that
        2026-05-01 ``measure_longname_match_rate.py`` analysis flagged
        as too high false-positive surface),
      * tokens in ``stopwords`` (corporate suffixes like "holdings"
        / "incorporated", fund markers like "etf" / "spdr", and
        generic financial words like "international" / "global" that
        appear in too many unrelated company names to anchor on),
      * pure-numeric tokens.

    Order is preserved (typically the leftmost word of the company name
    is the most distinctive — "Shuttle" in "SHUTTLE PHARMACEUTICAL
    HOLDINGS INC", "Electro" in "ELECTRO-SENSORS INC").

    Returns ``[]`` for empty input or when every token gets filtered out
    (the SPY case: ``"SS SPDR S&P 500 ETF TRUST-US"`` → all stopwords or
    too-short, leaving nothing). Empty result means "name extension
    can't help this symbol — fall back to ticker-only matching".
    """
    if not longname:
        return []
    parts = re.split(r"[^a-zA-Z0-9]+", longname.lower())
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if len(part) < min_token_len:
            continue
        if part in stopwords:
            continue
        if part.isdigit():
            continue
        if part in seen:
            continue
        seen.add(part)
        tokens.append(part)
    return tokens


class NameTokenCache:
    """Phase 10.5 — per-session cache of name-extension tokens + telemetry.

    Lifecycle: one instance per session, populated as the scanner
    qualifies new symbols, queried by :func:`classify` at every catalyst
    evaluation. Cleared implicitly on bot restart (in-memory only).

    The cache holds three kinds of state:

    * ``_tokens[symbol]`` — the tokenised longName for each symbol seen.
      Empty list means "tried but no usable tokens"; missing key means
      "not yet populated" (caller treats both the same — fall back to
      ticker-only matching).
    * Per-session "emitted-once" sets for the informational events:
      ``catalyst.name_extension_no_tokens`` (the longName tokenised to
      empty) and ``catalyst.name_extension_longname_missing`` (longName
      itself was empty). Operators see one event per affected symbol per
      session, not one per news evaluation.
    * Per-symbol rescue counter + a once-per-session
      ``catalyst.name_extension_high_rate`` warning when a symbol's
      rescue count crosses ``high_rate_threshold``. The count keeps
      incrementing past the threshold; we just don't re-warn. The
      warning's payload includes the token list so the operator can
      decide whether to add a too-generic token to the stopword list.
    """

    def __init__(
        self,
        *,
        stopwords: frozenset[str],
        min_token_len: int,
        high_rate_threshold: int,
    ) -> None:
        """Snapshot the per-session config; allocate empty registries."""
        self._stopwords = stopwords
        self._min_token_len = min_token_len
        self._high_rate_threshold = high_rate_threshold
        self._tokens: dict[str, list[str]] = {}
        self._rescue_counts: dict[str, int] = {}
        self._high_rate_emitted: set[str] = set()
        self._no_tokens_emitted: set[str] = set()
        self._missing_emitted: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> NameTokenCache:
        """Build a cache from a ``Settings`` instance — picks up defaults from
        ``catalyst.name_extension``."""
        cfg = settings.catalyst.name_extension
        return cls(
            stopwords=frozenset(s.lower() for s in cfg.stopwords),
            min_token_len=cfg.min_token_len,
            high_rate_threshold=cfg.high_rate_threshold,
        )

    def populate(self, symbol: str, longname: str | None) -> None:
        """Tokenise + cache; emit the right one-shot informational event.

        Idempotent: a second populate for the same symbol re-tokenises
        and replaces the cached list. The one-shot informational events
        (no-tokens / longname-missing) only fire on the first populate
        per symbol per session — subsequent populates with the same
        edge-case shape are silent.
        """
        if not longname:
            self._tokens[symbol] = []
            if symbol not in self._missing_emitted:
                self._missing_emitted.add(symbol)
                _log.info(
                    "catalyst.name_extension_longname_missing",
                    symbol=symbol,
                    hint="Symbol falls back to ticker-only matching for catalyst attribution.",
                )
            return
        tokens = tokenize_name(longname, self._stopwords, self._min_token_len)
        self._tokens[symbol] = tokens
        if not tokens and symbol not in self._no_tokens_emitted:
            self._no_tokens_emitted.add(symbol)
            _log.info(
                "catalyst.name_extension_no_tokens",
                symbol=symbol,
                longname=longname,
                hint=(
                    "Every token in this symbol's longName was filtered (too short or "
                    "in stopword list). Symbol falls back to ticker-only matching."
                ),
            )

    def get_tokens(self, symbol: str) -> list[str]:
        """Return the cached token list (possibly empty) or ``[]`` if not populated."""
        return self._tokens.get(symbol, [])

    def record_rescue(self, symbol: str, matched_token: str, headline: str) -> None:
        """Log a rescue + bump the per-symbol counter; warn once if it crosses threshold.

        Called from :func:`classify` after both gates pass via the name-
        extension path (i.e. ticker not in headline, name token matched,
        and Gate 2 still satisfied). Always logs the rescue; only warns
        on the FIRST rescue that crosses ``high_rate_threshold``.
        """
        _log.info(
            "catalyst.name_extension_rescued",
            symbol=symbol,
            matched_token=matched_token,
            headline=headline[:_HEADLINE_LOG_MAX],
            longname_tokens=list(self._tokens.get(symbol, [])),
        )
        count = self._rescue_counts.get(symbol, 0) + 1
        self._rescue_counts[symbol] = count
        if count > self._high_rate_threshold and symbol not in self._high_rate_emitted:
            self._high_rate_emitted.add(symbol)
            _log.warning(
                "catalyst.name_extension_high_rate",
                symbol=symbol,
                count_at_threshold=self._high_rate_threshold,
                token_list=list(self._tokens.get(symbol, [])),
                hint=(
                    "Per-session rescue count exceeded the threshold. Likely cause: a "
                    "name token matches too many unrelated headlines. Consider adding "
                    "the noisy token to catalyst.name_extension.stopwords."
                ),
            )

    def rescue_count(self, symbol: str) -> int:
        """Return the per-session rescue count for ``symbol`` — used by tests + status."""
        return self._rescue_counts.get(symbol, 0)


def _name_token_in_headline(name_tokens: list[str], headline_lc: str) -> str | None:
    """Phase 10.5 — return the first name token matching the headline (word-boundary).

    Whole-word match is critical — substring matching would false-positive
    ("max" matching "maximum"). Each token is escaped before insertion
    into the regex so longNames with regex-meta characters (rare but
    possible — the ``$`` in some ETF names) don't crash.
    """
    for token in name_tokens:
        if re.search(rf"\b{re.escape(token)}\b", headline_lc):
            return token
    return None


def classify(
    news_items: list[NewsItem],
    *,
    max_age_hours: int | None = None,
    reference_time: datetime | None = None,
    symbol: str | None = None,
    name_token_cache: NameTokenCache | None = None,
) -> str | None:
    """Return the highest-priority green category matched by any surviving item, or None.

    Order of operations:
      1. Phase 5.1 timestamp filter — items older than ``max_age_hours`` are
         dropped before any green/black evaluation.
      2. Phase 5.2 item-level black list — any surviving item whose combined
         headline + summary contains a black-list phrase is excluded, and
         only that item. Other items continue to the green check.
      3. Phase 5.2 item-level green list — each remaining item is matched
         against the category matchers in priority order; the first category
         matching any item wins at the corpus level.

    Phase 10.5 — when ``name_token_cache`` is provided, the
    ticker-not-in-headline rejection branch in Gate 1 falls back to a
    name-token match (any non-stopword token of the symbol's
    ``ContractDetails.longName`` appearing as a whole word in the
    headline). Gate 2 (ticker-anchored proximity) is unchanged: a
    headline that passes Gate 1 via the name-extension path still has
    to satisfy the body-text proximity check against the ticker. The
    rescue is logged via the cache's :meth:`record_rescue` method so
    operators can audit per-session rescue volume.

    Emits the following INFO events for forensic session review:

      * ``catalyst.item_blacklisted`` — one item was dropped due to a
        black-list phrase. Carries ``symbol``, ``headline`` (truncated to
        100 chars), and ``matched_phrase``.
      * ``catalyst.item_matched`` — one item matched a green category.
        Carries ``symbol``, ``category``, ``headline`` (truncated), and
        ``matched_phrase`` (e.g. ``"expedite + fda"`` for regulatory).
      * ``catalyst.item_skipped_off_topic_headline`` — Phase 9.7 gate 1
        rejected an item because the ticker is absent from the headline.
        Fired once per off-topic item (no per-category dispatch).
      * ``catalyst.item_matched_rejected_off_topic`` — Phase 9.7 gate 2
        rejected a per-category match because the matched phrase sits more
        than ``_PROXIMITY_MAX_TOKENS`` tokens from any ticker mention.
        Fired per (item, category) so an item that fails the proximity
        gate on earnings_beat may still match a closer-anchored category.
      * ``catalyst.name_extension_rescued`` — Phase 10.5: a per-(item, category)
        rescue fired. Carries ``matched_token`` and ``longname_tokens`` so
        the operator can audit which name signature did the work.
    """
    if max_age_hours is not None:
        ref = reference_time or datetime.now(UTC)
        cutoff = ref - timedelta(hours=max_age_hours)
        fresh = [item for item in news_items if item.datetime >= cutoff]
        filtered_out = len(news_items) - len(fresh)
        if filtered_out > 0:
            _log.debug(
                "catalyst.filtered_stale_news",
                symbol=symbol,
                filtered=filtered_out,
                kept=len(fresh),
                max_age_hours=max_age_hours,
            )
        news_items = fresh

    best_idx: int | None = None
    best_category: str | None = None
    ticker_lc = symbol.lower() if symbol else None

    for item in news_items:
        headline_lc = item.headline.lower()
        text = f"{item.headline} {item.summary}".lower()
        matched_black = next((p for p in _BLACK_LIST if p in text), None)
        if matched_black is not None:
            _log.info(
                "catalyst.item_blacklisted",
                symbol=symbol,
                headline=item.headline[:_HEADLINE_LOG_MAX],
                matched_phrase=matched_black,
            )
            continue

        # Phase 9.7 — Gate 1 (item-wide). When a symbol is provided we require
        # the ticker to appear in the headline before any green-list matcher
        # runs. Skipped when ``symbol is None`` (legacy callers + test
        # fixtures that don't plumb a ticker keep their old behaviour).
        # Gate 1 is bypassed for items that *might* match the regulatory
        # bucket, since that bucket is sector-wide by design — we'll let the
        # green-list loop run normally and apply the per-category attribution
        # check inside it.
        ticker_in_head = (
            ticker_lc is not None and _ticker_in_headline(headline_lc, ticker_lc)
        )

        for idx, (category, matcher, requires_attribution) in enumerate(_GREEN_LIST):
            hit = matcher(text)
            if hit is None:
                continue
            rescue_token: str | None = None
            if ticker_lc is not None and requires_attribution:
                # Gate 1 — ticker in headline OR (Phase 10.5) name token in headline.
                if not ticker_in_head:
                    if name_token_cache is not None and symbol is not None:
                        tokens = name_token_cache.get_tokens(symbol)
                        rescue_token = _name_token_in_headline(tokens, headline_lc)
                    if rescue_token is None:
                        _log.info(
                            "catalyst.item_matched_rejected_off_topic",
                            symbol=symbol,
                            category=category,
                            matched_phrase=hit,
                            headline=item.headline[:_HEADLINE_LOG_MAX],
                            reason="ticker_not_in_headline",
                        )
                        continue
                # Gate 2 — ticker-anchored proximity, unchanged. Even when Gate 1
                # passed via the name-extension fallback, Gate 2 still requires
                # the matched phrase to sit near a ticker mention in the
                # body text. A press release that names the company in the
                # headline almost always also references the ticker
                # somewhere in the body, so this rarely rejects the
                # rescued path — but a wrap article that happens to share
                # a name token with our symbol AND has the matched phrase
                # nowhere near our ticker is still correctly rejected.
                if not _phrase_near_ticker(text, hit, ticker_lc):
                    _log.info(
                        "catalyst.item_matched_rejected_off_topic",
                        symbol=symbol,
                        category=category,
                        matched_phrase=hit,
                        headline=item.headline[:_HEADLINE_LOG_MAX],
                        reason="phrase_too_far_from_ticker",
                        max_tokens=_PROXIMITY_MAX_TOKENS,
                    )
                    continue
                # Phase 10.5 — both gates passed via the name-extension path.
                # Log the rescue + bump the per-symbol counter so the
                # operator can audit volume and catch a too-generic
                # token via the high-rate warning.
                if rescue_token is not None and name_token_cache is not None:
                    name_token_cache.record_rescue(symbol, rescue_token, item.headline)  # type: ignore[arg-type]
            _log.info(
                "catalyst.item_matched",
                symbol=symbol,
                category=category,
                matched_phrase=hit,
                headline=item.headline[:_HEADLINE_LOG_MAX],
            )
            if best_idx is None or idx < best_idx:
                best_idx = idx
                best_category = category
            break

    return best_category
