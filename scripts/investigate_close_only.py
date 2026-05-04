"""One-off probe: does the IBKR API surface the close-only restriction
flag that TWS renders as a scanner-window icon?

Phase 9.6 added a per-symbol broker-rejection lockout that catches
close-only / restricted symbols *after* placement (e.g. error 10349 on
NASDAQ.SCM for RPGL). We want to reject these symbols at scanner time
instead. HCAI and FATN both currently show the close-only icon in the
TWS scanner with different restriction reasons; SPY is the unrestricted
control.

Run: ``uv run python scripts/investigate_close_only.py``

The script connects through the project's IBKRClient (so host/port/clientId
come from config.yaml — no duplicated credentials), runs four probes per
symbol, and prints a per-field diff at the end. Probe (d) (fundamental
data, slow + rate-limited) only runs if (a)–(c) yielded nothing
restriction-shaped.

Do NOT run this in production — it is for paper-account inspection only.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import fields, is_dataclass
from typing import Any

from ib_async import ScannerSubscription, Stock, TagValue

from bot.brokerage.ibkr_client import IBKRClient
from bot.config import get_settings

# Symbols to probe. HCAI and FATN are confirmed close-only in TWS at the
# time of writing (different restriction reasons each); SPY is the
# unrestricted control row used to anchor the per-field diff.
TEST_SYMBOLS = ["HCAI", "FATN"]
CONTROL_SYMBOL = "SPY"
ALL_SYMBOLS = [*TEST_SYMBOLS, CONTROL_SYMBOL]

# Generic tick list for probe (b). 236=Shortable, 318=LastRTHTrade,
# 588=IBDividends — the three ticks adjacent to "is the symbol
# trade-restricted?" semantics. ib_async populates ``ticker.shortable``
# and ``ticker.shortableShares`` from 236.
GENERIC_TICKS = "236,318,588"
MKTDATA_WAIT_SECONDS = 5.0

# Words that, if found anywhere in a probe payload, are likely the API's
# expression of the close-only state. Used both to flag (a)–(c) as
# "found something" (so we can skip probe d) and to grep the fundamental
# data XML.
RESTRICTION_KEYWORDS = (
    "restrict",
    "close",   # "close-only"
    "halt",
    "compliance",
    "shortable",
    "tradeable",
    "tradable",
    "easy_to_borrow",
    "etb",
    "htb",
    "rejected",
)

# Error codes that can arrive on the errorEvent stream rather than as
# return data. 10349 is the NASDAQ.SCM close-only reject seen in Phase
# 9.6; the rest are adjacent broker / market-data permission errors
# worth surfacing if TWS spits them back during the probes.
INTERESTING_ERROR_CODES = {200, 354, 10090, 10167, 10197, 10349}


def _delim(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def _kv_dump(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten ``obj`` into ``{field_name: repr-friendly value}``.

    Uses dataclasses.fields when possible (the ib_async types are dataclasses);
    falls back to ``vars()`` otherwise. Nested dataclasses get a
    dotted-prefix expansion one level deep — that's enough for ContractDetails
    → Contract without recursing into list-of-TagValue noise.
    """
    out: dict[str, Any] = {}
    if obj is None:
        return out
    if is_dataclass(obj):
        for f in fields(obj):
            value = getattr(obj, f.name, None)
            key = f"{prefix}{f.name}"
            if is_dataclass(value) and not isinstance(value, type):
                # one-level recurse so Contract inside ContractDetails is visible
                nested = _kv_dump(value, prefix=f"{key}.")
                out.update(nested)
            else:
                out[key] = value
        return out
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            out[f"{prefix}{k}"] = v
    return out


def _looks_restriction_shaped(blob: str) -> list[str]:
    """Return the keywords from ``RESTRICTION_KEYWORDS`` that appear in ``blob``."""
    lc = blob.lower()
    return [w for w in RESTRICTION_KEYWORDS if w in lc]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


async def probe_contract_details(
    client: IBKRClient, symbol: str
) -> tuple[dict[str, Any], list[str]]:
    """(a) reqContractDetails — dump every attribute, including nested Contract."""
    _sub(f"{symbol} | (a) reqContractDetailsAsync")
    contract = Stock(symbol, "SMART", "USD")
    try:
        details_list = await asyncio.wait_for(
            client.ib.reqContractDetailsAsync(contract), timeout=10.0
        )
    except Exception as exc:  # noqa: BLE001 - this is investigation; show everything
        print(f"  ERROR: {exc!r}")
        return {}, []
    if not details_list:
        print("  reqContractDetails returned no rows — symbol unknown to TWS?")
        return {}, []
    if len(details_list) > 1:
        print(f"  note: {len(details_list)} ContractDetails returned; using first.")
    details = details_list[0]
    flat = _kv_dump(details)
    for k in sorted(flat):
        print(f"  {k}={flat[k]!r}")
    blob = "\n".join(f"{k}={v}" for k, v in flat.items())
    hits = _looks_restriction_shaped(blob)
    if hits:
        print(f"  >> restriction-shaped tokens in (a): {hits}")
    return flat, hits


async def probe_mkt_data(
    client: IBKRClient, symbol: str
) -> tuple[dict[str, Any], list[str]]:
    """(b) reqMktData with generic ticks 236,318,588 — capture the streaming ticker."""
    _sub(f"{symbol} | (b) reqMktData(generic={GENERIC_TICKS}) for {MKTDATA_WAIT_SECONDS}s")
    contract = Stock(symbol, "SMART", "USD")
    try:
        await asyncio.wait_for(client.ib.qualifyContractsAsync(contract), timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR (qualify): {exc!r}")
        return {}, []
    ticks_seen: list[tuple[str, Any]] = []
    ticker = client.ib.reqMktData(
        contract,
        genericTickList=GENERIC_TICKS,
        snapshot=False,
        regulatorySnapshot=False,
    )

    def _on_update(t: Any) -> None:
        for td in list(t.ticks):
            ticks_seen.append((f"tickType={td.tickType}", td))

    ticker.updateEvent += _on_update
    try:
        await asyncio.sleep(MKTDATA_WAIT_SECONDS)
    finally:
        with contextlib.suppress(Exception):
            client.ib.cancelMktData(contract)
        with contextlib.suppress(Exception):
            ticker.updateEvent -= _on_update

    flat = _kv_dump(ticker)
    # Strip noisy collections that aren't field-like; keep restriction-shaped
    # scalars (halted, shortable, shortableShares, snapshotPermissions, etc.).
    skip = {"ticks", "tickByTicks", "domBids", "domAsks", "domTicks",
            "domBidsDict", "domAsksDict", "updateEvent", "defaults",
            "created", "contract"}
    printable = {k: v for k, v in flat.items() if k not in skip}
    for k in sorted(printable):
        v = printable[k]
        # NaN floats should still print so the diff catches the *absence*
        # of a value as a signal.
        print(f"  {k}={v!r}")
    if ticks_seen:
        print(f"  raw tick events captured: {len(ticks_seen)}")
        for label, td in ticks_seen[:30]:
            print(f"    {label} td={td!r}")
    else:
        print("  raw tick events captured: 0 (market closed or no updates in window)")

    blob_parts = [f"{k}={v}" for k, v in printable.items()]
    blob_parts.extend(repr(td) for _, td in ticks_seen)
    hits = _looks_restriction_shaped("\n".join(blob_parts))
    if hits:
        print(f"  >> restriction-shaped tokens in (b): {hits}")
    return printable, hits


async def probe_scanner(
    client: IBKRClient, symbols: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """(c) reqScannerSubscription — read scanner params from project config.

    Returns ``(per_symbol_flat, per_symbol_hits)``. Symbols absent from
    the scan get an explicit ``{"_absent_": True}`` row — absence is itself
    a signal we want to record in the diff.
    """
    _sub("(c) reqScannerSubscription [project TOP_PERC_GAIN config]")
    settings = get_settings()
    u = settings.universe
    sub = ScannerSubscription(
        instrument="STK",
        locationCode="STK.US.MAJOR",
        scanCode="TOP_PERC_GAIN",
    )
    tag_filters = [
        TagValue("priceAbove", str(u.price_min)),
        TagValue("priceBelow", str(u.price_max)),
        TagValue("changePercAbove", str(u.gap_pct_min)),
        TagValue("volumeAbove", str(u.premarket_vol_min)),
    ]
    try:
        scan_rows = await asyncio.wait_for(
            client.ib.reqScannerDataAsync(
                sub, scannerSubscriptionFilterOptions=tag_filters
            ),
            timeout=15.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc!r}")
        return {s: {} for s in symbols}, {s: [] for s in symbols}
    finally:
        with contextlib.suppress(Exception):
            # symmetrical with bot.scanning.scanner: TOP_PERC_GAIN is streaming
            client.ib.cancelScannerSubscription(scan_rows)  # type: ignore[arg-type]

    rows_by_symbol: dict[str, Any] = {}
    print(f"  scanner returned {len(scan_rows)} rows")
    for row in scan_rows:
        sym = getattr(getattr(row.contractDetails, "contract", None), "symbol", None)
        if sym in symbols:
            rows_by_symbol[sym] = row

    per_symbol_flat: dict[str, dict[str, Any]] = {}
    per_symbol_hits: dict[str, list[str]] = {}
    for sym in symbols:
        row = rows_by_symbol.get(sym)
        if row is None:
            print(f"  {sym}: ABSENT from scan results (signal — symbol may be filtered "
                  "out by TWS server-side close-only handling)")
            per_symbol_flat[sym] = {"_absent_from_scan_": True}
            per_symbol_hits[sym] = []
            continue
        flat = _kv_dump(row)
        # Expand contractDetails one more level
        cd = getattr(row, "contractDetails", None)
        if cd is not None:
            flat.update(_kv_dump(cd, prefix="contractDetails."))
        print(f"  {sym}: present in scan, fields:")
        for k in sorted(flat):
            print(f"    {k}={flat[k]!r}")
        blob = "\n".join(f"{k}={v}" for k, v in flat.items())
        hits = _looks_restriction_shaped(blob)
        if hits:
            print(f"    >> restriction-shaped tokens for {sym} in (c): {hits}")
        per_symbol_flat[sym] = flat
        per_symbol_hits[sym] = hits
    return per_symbol_flat, per_symbol_hits


async def probe_fundamental(
    client: IBKRClient, symbol: str
) -> tuple[str, list[str]]:
    """(d) reqFundamentalData(ReportsFinSummary) — only run if (a)–(c) were silent."""
    _sub(f"{symbol} | (d) reqFundamentalDataAsync(ReportsFinSummary)")
    contract = Stock(symbol, "SMART", "USD")
    try:
        await asyncio.wait_for(client.ib.qualifyContractsAsync(contract), timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR (qualify): {exc!r}")
        return "", []
    try:
        xml = await asyncio.wait_for(
            client.ib.reqFundamentalDataAsync(contract, reportType="ReportsFinSummary"),
            timeout=15.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR (fundamental): {exc!r}")
        return "", []
    print(f"  payload length: {len(xml)} chars")
    hits = _looks_restriction_shaped(xml)
    if hits:
        print(f"  >> restriction-shaped tokens in (d): {hits}")
        # Print 3 lines around each match for context
        for kw in hits:
            for m in re.finditer(re.escape(kw), xml, flags=re.IGNORECASE):
                start = max(0, m.start() - 80)
                end = min(len(xml), m.end() + 80)
                print(f"    [{kw}] ...{xml[start:end]}...")
    else:
        print("  no restriction-shaped tokens in fundamental payload")
    return xml, hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _value_repr(v: Any) -> str:
    """Stable string repr for diff comparison (handles NaN consistently)."""
    if isinstance(v, float):
        # NaN != NaN, normalise so the diff doesn't flag every NaN field as different
        if v != v:  # noqa: PLR0124 - intentional NaN check
            return "NaN"
    return repr(v)


def diff_fields(per_symbol: dict[str, dict[str, Any]]) -> None:
    """Print every flat field whose value differs across the probed symbols."""
    _delim("CROSS-SYMBOL DIFF (fields whose value differs between symbols)")
    all_keys: set[str] = set()
    for flat in per_symbol.values():
        all_keys.update(flat.keys())
    diffs: list[tuple[str, dict[str, str]]] = []
    for key in sorted(all_keys):
        values = {sym: _value_repr(per_symbol.get(sym, {}).get(key, "<MISSING>"))
                  for sym in per_symbol}
        unique = set(values.values())
        if len(unique) <= 1:
            continue
        diffs.append((key, values))
    if not diffs:
        print("  (no fields differ across symbols)")
        return
    print(f"  {len(diffs)} differing fields across {list(per_symbol)}:")
    for key, values in diffs:
        print(f"\n  {key}")
        for sym, val in values.items():
            print(f"    {sym}: {val}")


async def main() -> None:
    captured_errors: list[tuple[int, int, str]] = []

    settings = get_settings()
    client = IBKRClient(settings=settings)

    def _on_error(
        reqId: int,  # noqa: N803 - mirror ib_async signature
        errorCode: int,
        errorString: str,
        contract: object = None,
    ) -> None:
        # Capture restriction-shaped error codes plus anything containing
        # restriction keywords, so we don't miss novel codes.
        msg = errorString or ""
        if errorCode in INTERESTING_ERROR_CODES or _looks_restriction_shaped(msg):
            captured_errors.append((reqId, errorCode, msg))

    client.ib.errorEvent += _on_error

    await client.connect()
    try:
        # Per-symbol flat field maps; later combined for the diff.
        per_symbol_combined: dict[str, dict[str, Any]] = {s: {} for s in ALL_SYMBOLS}
        per_symbol_hits: dict[str, list[str]] = {s: [] for s in ALL_SYMBOLS}

        # (a) and (b) per symbol
        for sym in ALL_SYMBOLS:
            _delim(f"SYMBOL: {sym}")
            flat_a, hits_a = await probe_contract_details(client, sym)
            for k, v in flat_a.items():
                per_symbol_combined[sym][f"a.{k}"] = v
            per_symbol_hits[sym].extend(hits_a)

            flat_b, hits_b = await probe_mkt_data(client, sym)
            for k, v in flat_b.items():
                per_symbol_combined[sym][f"b.{k}"] = v
            per_symbol_hits[sym].extend(hits_b)

        # (c) one shared scan, per-symbol row extraction
        scan_flat, scan_hits = await probe_scanner(client, ALL_SYMBOLS)
        for sym in ALL_SYMBOLS:
            for k, v in scan_flat.get(sym, {}).items():
                per_symbol_combined[sym][f"c.{k}"] = v
            per_symbol_hits[sym].extend(scan_hits.get(sym, []))

        # (d) only for symbols where (a)-(c) was silent. SPY can stay silent
        # (control); we mainly care that HCAI/FATN got something earlier.
        any_test_signal = any(per_symbol_hits[s] for s in TEST_SYMBOLS)
        if any_test_signal:
            print("\n[skip d] (a)-(c) already returned restriction-shaped tokens "
                  "for at least one test symbol; skipping fundamental-data probe.")
        else:
            _delim("Probe (a)-(c) silent — running (d) reqFundamentalData")
            for sym in ALL_SYMBOLS:
                xml, hits_d = await probe_fundamental(client, sym)
                if xml:
                    per_symbol_combined[sym]["d.fundamental_xml_length"] = len(xml)
                    per_symbol_combined[sym]["d.fundamental_keyword_hits"] = ",".join(hits_d)
                per_symbol_hits[sym].extend(hits_d)

        # Captured error stream
        _delim("CAPTURED IB errorEvent ENTRIES (filtered to restriction-shaped)")
        if captured_errors:
            for reqId, code, msg in captured_errors:
                print(f"  reqId={reqId} code={code} msg={msg}")
        else:
            print("  (none captured during probe window)")

        diff_fields(per_symbol_combined)

        _delim("KEYWORD-HIT SUMMARY")
        for sym in ALL_SYMBOLS:
            hits = sorted(set(per_symbol_hits[sym]))
            print(f"  {sym}: {hits or '(none)'}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
