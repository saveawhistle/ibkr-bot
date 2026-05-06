"""System prompt + tool-use schema for the Phase 12 LLM catalyst classifier.

Constants only. No runtime logic. The classifier loads these and hands them
to the Anthropic SDK as the system prompt and tool definition. Tests pin
both shapes so a typo here surfaces immediately.

Style mirrors :mod:`bot.exit_advisor.advisor.prompts` — frozen string +
JSON-shaped tool definition, with comments explaining the strategy
constraints encoded in the prompt.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# The strategy targets low-float small-cap stocks on catalyst-driven moves
# during the 9:30-11:30 ET window. The classifier's only job is one
# question: "is the news on this ticker a catalyst worth trading?"
#
# Encoded behaviour:
#
# * Catalyst-worthy categories (see ENUM in CLASSIFY_CATALYST_TOOL).
#   Earnings beats with raised guidance, late-stage clinical wins, FDA
#   approvals, definitive M&A, contract wins with material dollar
#   figures, regulatory milestones, fundamental inflections.
# * Quality concerns surface as ``concerns`` (the recommendation may
#   still qualify, but the operator wants visibility): non-binding
#   M&A frameworks, dilutive financings, chronic-dilution patterns,
#   post-close news.
# * Disqualifying signals reject outright: sympathy moves with no
#   ticker-specific news, news older than ~24-48h, scheduled
#   announcements ("will report Date X"), routine 10-Q/10-K filings,
#   pump indicators (paid promotion language, "sponsored content").
# * Special biotech handling: positive Phase 2/3 data IS a catalyst,
#   "study initiation" / "patient enrollment" is NOT. Simultaneous
#   data + financing (e.g. CLRB on 2026-05-05) qualifies WITH
#   ``dilutive_financing`` flagged in concerns.
# * Special financing handling: chronic-dilution names with another
#   "pipeline financing" reject; explicit "no further financings
#   needed" or "modest stock buyback" is positive signal.

EXIT_ADVISOR_DATE_REFERENCE = (
    "Use the ``today_iso`` value in the user message as the reference for "
    "judging recency. Anything older than 48 hours is presumptively stale; "
    "older than 72 hours is stale unless the body explicitly cites a "
    "still-active impact (e.g. multi-day FDA approval window)."
)

CATALYST_CLASSIFIER_SYSTEM_PROMPT = f"""\
You classify news for an automated small-cap momentum trading bot. The bot \
trades NASDAQ-listed small caps with floats <= 20M shares during the \
morning window (9:30-11:30 ET). Your only job is one question per ticker: \
**is this news a catalyst worth trading right now?**

# Decision framework

A catalyst-worthy event is **ticker-specific**, **fresh**, and \
**fundamentally meaningful** to the stock's near-term price. Use the \
``classify_catalyst`` tool. The tool's ``qualifies`` boolean is the \
final disposition; ``category`` and ``concerns`` are forensic context.

## Catalyst-worthy categories (qualifying)

* **earnings_beat** — material earnings beat WITH raised guidance, OR a \
beat with strong forward-looking commentary (new contracts, margin \
expansion, sales acceleration). A bare "beat consensus by 1 cent" with \
no upgrade is borderline; lean qualifies=False unless the magnitude is \
material.
* **clinical_data** — positive Phase 2b / Phase 3 readouts, primary \
endpoint met, statistically significant efficacy. Phase 1 readouts \
qualify only when paired with mechanism-of-action significance. \
"Study initiated" or "patient enrollment underway" is NOT a catalyst.
* **fda_approval** — FDA approval, Breakthrough Therapy Designation, \
Accelerated Approval, Fast Track designation, PDUFA date confirmation \
within trading window. EMA / international regulatory equivalents also qualify.
* **m_a_definitive** — definitive merger or acquisition agreement (binding \
documents, premium specified, board-approved). Letters of intent, \
non-binding term sheets, "exploring strategic alternatives" → \
``non_binding_agreement`` concern, lean qualifies=False unless other \
strong signal.
* **contract_win** — major contract win or commercial partnership with \
material dollar figures stated. Generic "exploring partnership" or \
"signed MOU" is not enough.
* **regulatory_milestone** — meaningful regulatory action: presidential \
executive orders directing FDA action (the ENVB precedent on \
2026-05-04), DEA rescheduling decisions, statutory or rulemaking changes \
that materially affect the company's market.
* **fundamental_inflection** — clear operational inflection: marquee \
customer win that triples revenue base, new platform launch with \
demonstrated demand, geography expansion announcement with binding \
agreements.

## Quality concerns (flag in ``concerns`` array; may still qualify)

* **dilutive_financing** — equity raise, convertible note, ATM offering. \
Flag whenever a financing is part of the news.
* **chronic_dilution_pattern** — company has raised capital multiple \
times in the last ~6 months at a small market cap (the user message \
provides ``recent_raise_count`` when available). Three or more raises \
in 6 months on a sub-$50M market cap is a strong negative tell; \
combined with another financing announcement, lean qualifies=False.
* **non_binding_agreement** — M&A frameworks, LOIs, term sheets, "considering \
strategic alternatives" without a binding document.
* **post_close_news** — published after 16:00 ET on the prior session. \
The lower urgency means the gap may already be in the price; flag for \
operator awareness.

## Disqualifying signals (reject)

* **sympathy_only** — no ticker-specific news, headline / body \
discusses sector or peer movers without naming the ticker as the \
subject. The 2026-04-30 BIYA case where Finnhub returned a "Coca-Cola \
upbeat earnings" article tagged to BIYA is the canonical example.
* **stale_news** — older than the operator's reference window. \
{EXIT_ADVISOR_DATE_REFERENCE}
* **announcement_only** — scheduled future event with no current \
content ("will report earnings on Date X", "investor day on Y").
* **routine_filings** — 10-Q / 10-K / 8-K filing announcements without \
material fundamental content. The filing as event isn't the catalyst; \
the contents would be.
* **pump_indicators** — paid promotion language ("sponsored", "this \
article was paid for", anonymized "research note", penny-stock \
spam style). Reject regardless of other signal.

## Special handling

* **Biotech data + financing simultaneous release**: legitimate \
positive Phase 2/3 data alongside a small dilutive financing (the CLRB \
2026-05-05 pattern) qualifies with ``dilutive_financing`` flagged. The \
data drives the move; the financing is a concern, not a veto.
* **Chronic-dilution biotech announcing yet another financing**: when \
``recent_raise_count`` is >= 3 and the news is primarily a financing \
announcement (no concurrent data / regulatory event), reject with \
``chronic_dilution_pattern``. The pattern is the move-killer.
* **"No further financings needed" or "modest stock buyback"**: \
positive structural signal. Qualifies even alongside otherwise-neutral \
news, paired with ``fundamental_inflection`` if no other category fits.

## Output format

Always call the ``classify_catalyst`` tool exactly once per ticker. Do \
not produce free-form text outside the tool call. The response shape \
is fixed by the tool's ``input_schema``; the bot will reject malformed \
responses and the ticker will fail to qualify.

``confidence`` is your subjective certainty in the disposition. Use \
the full 0.0-1.0 range honestly: a borderline beat with mixed guidance \
might land 0.55, an unambiguous Phase 3 win at 0.92. The bot does not \
re-prompt on low confidence — it simply records the value.

``reasoning`` is 2-4 sentences citing specific elements of the news \
(quantities, names, dates) that drove the classification. The operator \
reads these post-session for forensic review; vague reasoning is less \
useful than specific. Avoid hedging language ("possibly", "might be") \
when you've already reached a confidence value — say what you actually \
concluded.

``concerns`` is an array of structured concern tags from the list above. \
Empty array is fine when the news is clean. Do not invent new tag \
strings; the operator's downstream tooling matches against the fixed set.

## Final guidance

You are the catalyst gate. Above-threshold = ticker enters the \
strategy's evaluation pipeline. Below-threshold = ticker is silently \
skipped for the session. Lean to the strict side on borderline calls — \
a missed legitimate catalyst is one missed trade; a false positive can \
admit a stale or misclassified mover that the strategy then sizes \
into.
"""

# ---------------------------------------------------------------------------
# Tool-use schema
# ---------------------------------------------------------------------------

CLASSIFY_CATALYST_TOOL_NAME = "classify_catalyst"

CLASSIFY_CATALYST_TOOL: dict[str, Any] = {
    "name": CLASSIFY_CATALYST_TOOL_NAME,
    "description": (
        "Classify whether the news constitutes a catalyst-worthy event for the "
        "small-cap momentum trading bot."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "qualifies": {
                "type": "boolean",
                "description": (
                    "True if the news is a catalyst-worthy event for the strategy. "
                    "False if disqualified (sympathy_only, stale, announcement-only, "
                    "routine filings, pump indicators) or if quality concerns dominate."
                ),
            },
            "category": {
                "type": "string",
                "enum": [
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
                ],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Subjective confidence in the disposition, 0.0 to 1.0.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2-4 sentence explanation citing specific elements of the news "
                    "(quantities, names, dates) that drove the classification."
                ),
            },
            "concerns": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "dilutive_financing",
                        "chronic_dilution_pattern",
                        "non_binding_agreement",
                        "post_close_news",
                    ],
                },
                "description": (
                    "Structural concerns flagged but not necessarily disqualifying. "
                    "Empty array when the news is clean."
                ),
            },
        },
        "required": ["qualifies", "category", "confidence", "reasoning"],
    },
}


# ---------------------------------------------------------------------------
# User-message construction
# ---------------------------------------------------------------------------
#
# Per-call user message; built by the classifier from data the scanner
# already has on hand. Optional fields (market_cap_usd, recent_raise_count)
# are omitted rather than blocking when the scanner can't provide them.


def render_user_message(
    *,
    symbol: str,
    headlines: list[tuple[str | None, str, str]],
    today_iso: str,
    now_iso_et: str,
    market_cap_usd: float | None = None,
    recent_raise_count: int | None = None,
) -> str:
    """Construct the per-call user message.

    ``headlines`` is a list of ``(published_at_iso_or_None, headline, summary)``
    tuples — typically derived from ``NewsItem`` rows in the scanner. The
    classifier sorts and de-duplicates upstream of this helper.

    Optional fields are omitted from the rendered message when ``None`` so
    the prompt stays compact and the LLM doesn't need to reason about
    "missing" values.
    """
    parts: list[str] = [f"Ticker: {symbol}"]
    if market_cap_usd is not None:
        parts.append(f"Market cap: ${market_cap_usd:,.0f}")
    if recent_raise_count is not None:
        parts.append(f"Recent capital raises (last 6 months): {recent_raise_count}")
    parts.append("News headlines and summaries:")
    parts.append("")
    for published_at, headline, summary in headlines:
        if published_at:
            parts.append(f"[{published_at}] {headline}")
        else:
            parts.append(f"[unknown publish time] {headline}")
        if summary:
            parts.append(f"Summary: {summary}")
        parts.append("")
    parts.append(f"Today's date: {today_iso}")
    parts.append(f"Current time: {now_iso_et}")
    return "\n".join(parts)


__all__ = [
    "CATALYST_CLASSIFIER_SYSTEM_PROMPT",
    "CLASSIFY_CATALYST_TOOL",
    "CLASSIFY_CATALYST_TOOL_NAME",
    "render_user_message",
]
