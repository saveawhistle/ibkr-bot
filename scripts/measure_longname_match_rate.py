"""One-off probe — would extending Phase 9.7's catalyst gate to accept the
IBKR ``ContractDetails.longName`` rescue legitimate catalysts that the
ticker-only gate currently rejects?

Today's gate requires the ticker (word-boundary, case-insensitive) in the
news headline. Press releases for sub-$20 names usually do include the
ticker via the ``(NASDAQ: TICKER)`` cashtag form, but a real press release
that names *only* the company (e.g. "Baiya International Group Provides
Business Update") would be rejected. The hypothetical extension: also
accept headlines that mention any non-stopword token of the IBKR longName.

This probe walks ``logs/session_*.jsonl``, extracts every historical
``catalyst.item_matched`` event (these are the headlines the pre-9.7
classifier accepted — our universe of "things the catalyst chain saw"),
qualifies each unique symbol via ``reqContractDetailsAsync`` to get its
longName, then retroactively applies BOTH gates:

  * **Current gate (Phase 9.7)**: ticker word-boundary in headline +
    matched_phrase within 30 tokens of any ticker mention.
  * **Hypothetical name-extended gate**: same as current PLUS accept
    when any non-stopword longName token appears in the headline AND
    the matched_phrase is within 30 tokens of that token.

Buckets the per-event outcomes:

  * **both_accept** — current gate accepts; extension changes nothing.
  * **rescue** — current rejects, extension accepts. Each rescue is
    listed for human triage to decide if the original
    ``catalyst.item_matched`` was on-topic (true rescue) or off-topic
    (would be a new false positive).
  * **both_reject** — both gates reject. The current gate already
    catches these as off-topic; extension wouldn't help.

Reports also include longName coverage (how many symbols have a
populated longName at all) and a per-rescue dump suitable for grep.

Run any time TWS is reachable:
    uv run python scripts/measure_longname_match_rate.py

Defaults:
  * Reads every ``logs/session_*.jsonl`` file (operator can comment-out
    the glob to scope to a specific day).
  * Uses clientId 95 to avoid colliding with the running bot (17) or
    the other probes (96, 97, 98, 99).
  * Output is plain text to stdout — pipe to a file if you want to
    archive the rescue list for triage.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ib_async import IB, Stock

# Reuse the production gate helpers so the probe's "current gate" is
# exactly what the bot runs in prod — no drift between probe and reality.
from bot.scanning.catalyst import (  # noqa: PLC2701 - private helpers used as ground truth
    _PROXIMITY_MAX_TOKENS,
    _phrase_near_ticker,
    _ticker_in_headline,
    _token_index_at,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_GLOB = "logs/session_*.jsonl"
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 95


# ---------------------------------------------------------------------------
# Name-token tokeniser (the hypothetical gate's input)
# ---------------------------------------------------------------------------

# Common corporate suffixes / class indicators / generic tokens that
# would create rampant false positives if used as the "name signature".
# Keep it conservative — better to miss a few rescues than admit floods
# of "Holdings"/"Group"/"Inc"-only matches.
_NAME_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # Corporate suffixes
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "llc",
        "co",
        "company",
        "group",
        "grp",
        "holdings",
        "hldgs",
        "international",
        "intl",
        "industries",
        "enterprises",
        "trust",
        # Geographic / listing tags
        "us",
        "usa",
        "uk",
        "ca",
        "cn",
        # Class / share-type indicators
        "a",
        "b",
        "c",
        "ord",
        "adr",
        "spdr",
        "etf",
        "fund",
        # Filler
        "the",
        "and",
        "of",
        "on",
        "for",
    }
)

# Minimum token length for a word to be considered "name signature."
# 3 chars catches "BIO", "TEC", but drops "AI"/"IT". Adjustable.
_MIN_TOKEN_LEN: Final[int] = 3


def _name_tokens(longname: str) -> list[str]:
    """Tokenise an IBKR longName into matchable name-signature tokens.

    Lowercase, split on any non-alphanumeric, drop stopwords + short tokens
    + pure-numeric tokens. Returns the de-duplicated list in order so the
    matcher tries the most specific first (typically the leftmost word
    of the company name).
    """
    if not longname:
        return []
    parts = re.split(r"[^a-zA-Z0-9]+", longname.lower())
    seen: set[str] = set()
    tokens: list[str] = []
    for p in parts:
        if len(p) < _MIN_TOKEN_LEN:
            continue
        if p in _NAME_STOPWORDS:
            continue
        if p.isdigit():
            continue
        if p in seen:
            continue
        seen.add(p)
        tokens.append(p)
    return tokens


def _name_token_in_headline(name_tokens: list[str], headline_lc: str) -> str | None:
    """Return the first name token that matches the headline (word-boundary)."""
    for token in name_tokens:
        if re.search(rf"\b{re.escape(token)}\b", headline_lc):
            return token
    return None


def _phrase_near_any_anchor(
    text_lc: str,
    matched_phrase: str,
    anchors_lc: list[str],
) -> bool:
    """True iff ``matched_phrase`` is within ``_PROXIMITY_MAX_TOKENS`` of any anchor.

    Anchors are word-boundary substrings (ticker + name tokens). Used by
    the hypothetical name-extended gate's relaxed Gate 2.
    """
    phrase_pos = text_lc.find(matched_phrase)
    if phrase_pos < 0:
        return False
    phrase_tok = _token_index_at(text_lc, phrase_pos)
    for anchor in anchors_lc:
        pat = re.compile(rf"\b{re.escape(anchor)}\b")
        for m in pat.finditer(text_lc):
            anchor_tok = _token_index_at(text_lc, m.start())
            if abs(anchor_tok - phrase_tok) <= _PROXIMITY_MAX_TOKENS:
                return True
    return False


# ---------------------------------------------------------------------------
# Log scraping
# ---------------------------------------------------------------------------


@dataclass
class CatalystEvent:
    """One historical ``catalyst.item_matched`` row from session JSONL."""

    symbol: str
    headline: str
    matched_phrase: str
    category: str
    session_date: str  # filename stem for forensic linkback


def _load_catalyst_events(repo_root: Path) -> list[CatalystEvent]:
    """Walk session_*.jsonl files and extract every catalyst.item_matched row."""
    events: list[CatalystEvent] = []
    for path in sorted(repo_root.glob(LOG_GLOB)):
        date_stem = path.stem.replace("session_", "")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                if "catalyst.item_matched" not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") != "catalyst.item_matched":
                    continue
                symbol = row.get("symbol") or ""
                headline = row.get("headline") or ""
                matched_phrase = row.get("matched_phrase") or ""
                category = row.get("category") or ""
                if not (symbol and headline and matched_phrase):
                    continue
                events.append(
                    CatalystEvent(
                        symbol=symbol,
                        headline=headline,
                        matched_phrase=matched_phrase,
                        category=category,
                        session_date=date_stem,
                    )
                )
    return events


# ---------------------------------------------------------------------------
# Contract details lookup
# ---------------------------------------------------------------------------


async def _fetch_longname(ib: IB, symbol: str) -> str:
    """Return ContractDetails.longName for ``symbol``, or empty string on miss.

    Catches every failure shape (delisted symbol, foreign listing, IBKR
    timeout) and returns "" so the probe can cleanly bucket "no longName"
    coverage without exception handling at every call site.
    """
    contract = Stock(symbol, "SMART", "USD")
    try:
        details_list = await asyncio.wait_for(ib.reqContractDetailsAsync(contract), timeout=10.0)
    except (TimeoutError, Exception) as exc:  # noqa: BLE001 - probe; show everything
        print(f"  [warn] {symbol}: contract details lookup failed — {exc!r}")
        return ""
    if not details_list:
        return ""
    return str(getattr(details_list[0], "longName", "") or "")


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


@dataclass
class GateOutcome:
    """Per-event outcome under both gates."""

    event: CatalystEvent
    longname: str
    name_tokens: list[str]
    current_accepts: bool
    extended_accepts: bool
    extension_matched_token: str | None  # which name token rescued it (None if not a rescue)


def _evaluate(event: CatalystEvent, longname: str) -> GateOutcome:
    """Apply both gates retroactively to one event."""
    headline_lc = event.headline.lower()
    text_lc = headline_lc  # we don't have the summary in the log; headline is conservative
    ticker_lc = event.symbol.lower()

    # --- Current gate (Phase 9.7) ---
    current_g1 = _ticker_in_headline(headline_lc, ticker_lc)
    current_g2 = (
        _phrase_near_ticker(text_lc, event.matched_phrase, ticker_lc) if current_g1 else False
    )
    current_accepts = current_g1 and current_g2

    # --- Hypothetical name-extended gate ---
    name_tokens = _name_tokens(longname)
    extension_matched: str | None = None
    extended_accepts = current_accepts
    if not current_accepts:
        # Try the name-token branch.
        token = _name_token_in_headline(name_tokens, headline_lc)
        if token is not None:
            anchors = [ticker_lc, *name_tokens]
            if _phrase_near_any_anchor(text_lc, event.matched_phrase, anchors):
                extended_accepts = True
                extension_matched = token

    return GateOutcome(
        event=event,
        longname=longname,
        name_tokens=name_tokens,
        current_accepts=current_accepts,
        extended_accepts=extended_accepts,
        extension_matched_token=extension_matched,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _delim(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _report(outcomes: list[GateOutcome], longnames: dict[str, str]) -> None:
    total = len(outcomes)
    if total == 0:
        print("No catalyst events found in session logs — nothing to report.")
        return

    both_accept = sum(1 for o in outcomes if o.current_accepts and o.extended_accepts)
    rescues = [o for o in outcomes if not o.current_accepts and o.extended_accepts]
    both_reject = sum(1 for o in outcomes if not o.current_accepts and not o.extended_accepts)

    _delim("HEADLINE COUNT BY GATE OUTCOME")
    print(f"  total catalyst.item_matched events: {total}")
    print(
        f"  both gates accept              : {both_accept:5d}  ({both_accept / total * 100:5.1f}%)"
    )
    print(
        f"  RESCUE (current rej, ext acc)  : {len(rescues):5d}  "
        f"({len(rescues) / total * 100:5.1f}%)"
    )
    print(
        f"  both gates reject              : {both_reject:5d}  ({both_reject / total * 100:5.1f}%)"
    )

    _delim("LONGNAME COVERAGE")
    have_name = sum(1 for n in longnames.values() if n)
    no_name = sum(1 for n in longnames.values() if not n)
    print(f"  unique symbols seen in logs : {len(longnames)}")
    print(f"  with non-empty longName     : {have_name}  ({have_name / len(longnames) * 100:.1f}%)")
    print(f"  with empty longName         : {no_name}")
    if no_name:
        empties = sorted(s for s, n in longnames.items() if not n)
        print(
            f"  empty-name symbols          : {', '.join(empties[:30])}{'...' if len(empties) > 30 else ''}"
        )

    _delim("LONGNAME SAMPLE — first 25 symbols (raw IBKR data)")
    for symbol in sorted(longnames)[:25]:
        ln = longnames[symbol]
        toks = _name_tokens(ln)
        print(f"  {symbol:<8s}  longName={ln!r:<45s}  tokens={toks}")

    _delim(f"RESCUE CANDIDATES — {len(rescues)} cases for human triage")
    if not rescues:
        print("  (none)")
    else:
        # Sort by symbol so the operator can scan together-grouped rescues.
        rescues.sort(key=lambda o: (o.event.symbol, o.event.session_date))
        for o in rescues:
            print()
            print(f"  symbol      = {o.event.symbol}")
            print(f"  session     = {o.event.session_date}")
            print(f"  category    = {o.event.category}")
            print(f"  matched     = {o.event.matched_phrase!r}")
            print(
                f"  rescued by  = {o.extension_matched_token!r}  (longName tokens: {o.name_tokens})"
            )
            print(f"  longName    = {o.longname!r}")
            print(f"  headline    = {o.event.headline!r}")

    _delim("PER-SYMBOL RESCUE COUNTS — top 20")
    counts: dict[str, int] = defaultdict(int)
    for o in rescues:
        counts[o.event.symbol] += 1
    for symbol, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        ln = longnames.get(symbol, "")
        print(f"  {symbol:<8s}  rescues={n:3d}  longName={ln!r}")

    _delim("DECISION HEURISTIC")
    rescue_pct = len(rescues) / total * 100
    print(f"  Rescue rate: {rescue_pct:.1f}% of historical catalyst.item_matched events would")
    print("  flip from rejected (current Phase 9.7 gate) to accepted (name-extended gate).")
    print()
    if rescue_pct < 1.0:
        print("  -> Below 1% — the BIYA-shape false positives the current gate already catches")
        print("    are basically all of them. Name extension would add complexity for")
        print("    marginal recall gain. RECOMMEND: stay with ticker-only.")
    elif rescue_pct < 5.0:
        print("  -> 1-5% — modest recall gain. Worth implementing IF the rescue list above")
        print("    looks dominated by genuine on-topic catalysts (e.g. press releases that")
        print("    name only the company, no ticker). If the list is mostly noise, skip.")
    else:
        print("  -> >5% — substantial recall gain. Likely worth implementing — but read the")
        print("    rescue list carefully for false-positive patterns before committing.")
    print()
    print(f"  Hard cases (both reject): {both_reject} headlines that even name-extension")
    print("  doesn't help. These would need a different approach (e.g. NER, longName")
    print("  ALIAS table, summary-text matching) to recover.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Reading session logs from {repo_root / 'logs'}/session_*.jsonl ...")
    events = _load_catalyst_events(repo_root)
    print(f"  loaded {len(events)} catalyst.item_matched events")

    unique_symbols = sorted({e.symbol for e in events})
    print(f"  unique symbols: {len(unique_symbols)}")

    if not unique_symbols:
        print("Nothing to look up — exiting.")
        return

    print()
    print(f"Connecting to TWS at {TWS_HOST}:{TWS_PORT} (clientId={CLIENT_ID})...")
    ib = IB()
    await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(1)

    longnames: dict[str, str] = {}
    print(f"Fetching longName for {len(unique_symbols)} symbols (sequential, ~1s each)...")
    for i, symbol in enumerate(unique_symbols, start=1):
        ln = await _fetch_longname(ib, symbol)
        longnames[symbol] = ln
        if i % 10 == 0 or i == len(unique_symbols):
            print(f"  [{i:3d}/{len(unique_symbols):3d}] {symbol:<8s} longName={ln!r}")

    ib.disconnect()

    print()
    print(f"Evaluating both gates against {len(events)} events...")
    outcomes = [_evaluate(e, longnames.get(e.symbol, "")) for e in events]

    _report(outcomes, longnames)


if __name__ == "__main__":
    asyncio.run(main())
