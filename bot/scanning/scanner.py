"""IBKR TOP_PERC_GAIN scan → Finnhub float + news enrichment → ranked morning watchlist.

The scanner implements the non-strategy portion of the 5-Pillar filter (price,
% change, premarket volume, float, rvol, catalyst). Price / change% / premarket
volume are enforced on the IBKR side via scanner TagValues; float, rvol, and
catalyst are applied post-scan in cheapest-first order so the LLM catalyst call
only runs against symbols that have already cleared every cheaper pillar.

Phase 12.1: quote enrichment (``price`` / ``change_pct`` / ``volume`` via
``reqHistoricalData``) runs immediately after the float filter -- the rvol
pillar uses ``volume`` as its numerator, and pruning before catalyst saves LLM
spend on symbols that wouldn't qualify anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from ib_async import Contract, ScannerSubscription, TagValue

from bot.brokerage.ibkr_client import ActiveSubscription, IBKRClient, ref_req_id
from bot.config import Settings, get_settings
from bot.scanning.catalyst import NameTokenCache, classify
from bot.scanning.catalyst_overrides import CatalystOverride, load_active_overrides_map
from bot.scanning.finnhub_client import FinnhubClient, NewsItem
from bot.scanning.float_source import FloatData, FloatSource
from bot.scanning.llm_catalyst_classifier import (
    ClassificationResult,
    LLMCatalystClassifier,
)
from bot.scanning.manual_watchlist import (
    MANUAL_CATALYST_SENTINEL,
    ManualWatchlistEntry,
)
from bot.scanning.manual_watchlist import (
    load_active_entries as load_active_manual_watchlist,
)
from bot.scanning.yfinance_news import fetch_yfinance_news

_ENRICHMENT_TIMEOUT_SECONDS_DEFAULT = 2.0
"""Per-symbol IBKR market-data timeout. 2s × 10-15 symbols parallelized stays
under ~3s total wall time — acceptable for premarket responsiveness."""

_log = structlog.get_logger("bot.scanning.scanner")


@dataclass
class ScanHit:
    """One ranked morning-watchlist row with populated 5-Pillar context.

    ``price``/``change_pct``/``volume`` are Optional because IBKR's scanner
    snapshot does not expose them in Phase 2; rows are ordered by the scanner's
    own TOP_PERC_GAIN rank when numerical change_pct is not available.

    ``rvol`` (Phase 12.1) is today's session volume divided by yfinance's
    10-day average daily volume. Populated when both inputs are known; the
    rvol pillar drops the symbol upstream when either is missing or the
    ratio is below ``universe.rvol_min``.
    """

    symbol: str
    price: float | None
    change_pct: float | None
    volume: int | None
    float_shares: int | None
    catalyst: str | None
    float_source: str | None = None
    news_items: list[NewsItem] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    rvol: float | None = None
    avg_daily_volume: int | None = None
    manual: bool = False
    """True when this hit came from ``data/manual_watchlist.json`` rather
    than the IBKR scanner snapshot. Manual hits bypass float / rvol /
    catalyst gates upstream; downstream code uses this flag to log
    operator-injected entries distinctly from organic scanner hits."""


class IBKRScanner:
    """Runs the IBKR TOP_PERC_GAIN scan and enriches each hit with Finnhub float + news."""

    def __init__(
        self,
        ibkr: IBKRClient,
        finnhub: FinnhubClient,
        settings: Settings | None = None,
        *,
        float_source: FloatSource | None = None,
        enrichment_timeout_seconds: float = _ENRICHMENT_TIMEOUT_SECONDS_DEFAULT,
        name_token_cache: NameTokenCache | None = None,
        llm_classifier: LLMCatalystClassifier | None = None,
    ) -> None:
        """Wire an ``IBKRClient``, ``FinnhubClient``, and ``FloatSource`` together.

        ``enrichment_timeout_seconds`` bounds each per-symbol market-data call
        at Phase 5.3 enrichment time. Exposed as a constructor kwarg so tests
        can dial it down without waiting the real 2 s.

        Phase 10.5 ``name_token_cache`` (defaults to one built from
        ``settings.catalyst.name_extension``) feeds the catalyst
        attribution gate's name-extension fallback. Tests can pass an
        explicit cache (or ``None`` to disable name extension entirely
        — useful for pre-10.5 regression tests).

        Phase 12 ``llm_classifier`` is the LLM-driven catalyst classifier
        constructed via ``bootstrap_catalyst_classifier``. When ``None``
        AND ``settings.catalyst_classifier.llm.enabled=True``, the
        scanner logs a warning and silently falls back to the keyword
        classifier — defence in depth so a missing API key (or bootstrap
        failure) doesn't crash the scanner. When ``llm_classifier`` is
        provided AND the LLM mode is enabled, news fetch + classification
        run in parallel via ``asyncio.gather`` for the surviving symbols.
        """
        self._ibkr = ibkr
        self._finnhub = finnhub
        self._settings = settings or get_settings()
        self._float_source = float_source or FloatSource(finnhub=finnhub)
        self._enrichment_timeout_seconds = enrichment_timeout_seconds
        self._name_token_cache = (
            name_token_cache
            if name_token_cache is not None
            else NameTokenCache.from_settings(self._settings)
        )
        self._llm_classifier = llm_classifier
        # Cache the resolved classifier mode for one scan pass — the result is
        # logged once on first use rather than per-symbol so the JSONL stays
        # readable.
        self._classifier_mode_logged = False

    # ------------------------------------------------------------------
    # Phase 12 public hooks
    # ------------------------------------------------------------------

    @property
    def llm_classifier(self) -> LLMCatalystClassifier | None:
        """Return the bound LLM classifier (or ``None`` when keyword path active)."""
        return self._llm_classifier

    def on_watchlist_removal(self, ticker: str) -> None:
        """Forward to the LLM classifier so a re-entered ticker re-evaluates fresh.

        Idempotent — the classifier itself short-circuits when the ticker
        was never qualified. The orchestrator calls this from
        ``_apply_watchlist_diff`` whenever a symbol drops off the
        watchlist, regardless of cause (rescan churn, position closed,
        end-of-day).
        """
        if self._llm_classifier is None:
            return
        self._llm_classifier.on_watchlist_removal(ticker)

    def _resolve_classifier_mode(self) -> str:
        """Decide which classifier path runs this scan and warn on misconfig.

        Returns one of ``"llm"``, ``"keyword"``, or ``"none"``. Only emits
        the resolution log once per ``IBKRScanner`` lifetime so a long
        session doesn't repeat-spam.
        """
        cfg = self._settings.catalyst_classifier
        llm_on = cfg.llm.enabled and self._llm_classifier is not None
        keyword_on = cfg.keyword.enabled
        # Warn when LLM is configured-on but bootstrap returned None
        # (missing API key, construction failed). This happens once per
        # session and tells the operator the scanner silently degraded.
        bootstrap_missing = cfg.llm.enabled and self._llm_classifier is None
        if not self._classifier_mode_logged:
            if bootstrap_missing:
                _log.warning(
                    "scanner.classifier_llm_bootstrap_missing",
                    hint=(
                        "catalyst_classifier.llm.enabled=true but no "
                        "LLMCatalystClassifier was wired. Falling back to "
                        "keyword classifier if its enabled=true; otherwise "
                        "no catalyst evaluation runs."
                    ),
                )
            if llm_on and keyword_on:
                _log.warning(
                    "scanner.classifier_both_modes_enabled",
                    hint="catalyst_classifier.llm + keyword both enabled; preferring LLM.",
                )
            if not llm_on and not keyword_on:
                _log.warning(
                    "scanner.classifier_no_mode_enabled",
                    hint=(
                        "Both catalyst_classifier.llm and .keyword are off; "
                        "the scanner will qualify nothing this pass."
                    ),
                )
            self._classifier_mode_logged = True
        if llm_on:
            return "llm"
        if keyword_on:
            return "keyword"
        return "none"

    async def scan_top_gappers(self) -> list[ScanHit]:
        """Run the scan and return the ranked morning watchlist as ``ScanHit`` rows.

        Phase 6.8: when ``testing.allow_catalyst_overrides`` is on, the
        scanner consults ``data/test_catalyst_overrides.json`` before
        fetching Finnhub news. Symbols with an active injection skip
        the fetch entirely (saves one company-news quota per override)
        and inherit the injected category as their catalyst; everything
        downstream (ScanHit, strategy evaluation, executor) runs
        identically to an organically-classified hit.

        Phase 12: when ``catalyst_classifier.llm.enabled=true`` AND a
        classifier was wired (bootstrap succeeded), per-ticker news fetch
        + classification run concurrently via ``asyncio.gather``. The
        keyword classifier remains as a fallback when the LLM path is
        disabled OR the bootstrap returned ``None``.

        Phase 12.1 pillar order (cheap -> expensive, prune as we go):
        1. IBKR scanner snapshot (price / gap% / premarket vol enforced
           by ``ScannerSubscription`` TagValues).
        2. Float filter (yfinance + Finnhub fallback).
        3. Quote fetch (one IBKR ``reqHistoricalData`` per float-survivor,
           parallelized).
        4. Rvol filter (today's session volume / 10-day avg daily volume
           >= ``universe.rvol_min``). Drops on either-unknown.
        5. Catalyst classification (LLM or keyword) -- only the most
           expensive call runs, and only for symbols past every cheaper
           pillar.
        """
        contracts = await self._fetch_ibkr_gappers()
        if not contracts:
            _log.info("scanner.empty_scan")
            # Manual watchlist entries (operator escape hatch) still need
            # to seed the watchlist even when IBKR's TOP_PERC_GAIN returned
            # nothing -- the whole point of manual entries is they bypass
            # the IBKR scan.
            return self._merge_manual_watchlist([])
        floats = await self._fetch_floats([c.symbol for c in contracts])
        survivors = self._apply_float_filter(contracts, floats)
        if not survivors:
            return self._merge_manual_watchlist([])
        # Phase 12.1 — quote enrichment runs BEFORE catalyst classification
        # so the rvol pillar can prune candidates the LLM would otherwise
        # be billed to evaluate. Best-effort: failures populate (None, None,
        # None) and the rvol filter treats those as ``rvol_unknown``.
        quotes = await self._fetch_quotes_map(survivors)
        survivors, rvol_map = self._apply_rvol_filter(survivors, quotes, floats)
        if not survivors:
            return self._merge_manual_watchlist([])
        # Phase 6.8 override map — empty dict when the flag is off, so
        # every symbol follows the Finnhub path and the store is never
        # touched. Expired entries are filtered inside the helper.
        overrides = self._load_active_overrides()
        symbols_needing_news = [c.symbol for c in survivors if c.symbol not in overrides]
        news_map = await self._fetch_news(symbols_needing_news)
        # Phase 10.5 — populate the name-token cache for symbols whose news
        # we'll classify. Concurrent fetch via ``asyncio.gather`` so the
        # 10-15 ``reqContractDetails`` round-trips overlap (~30ms each
        # serially → bounded by slowest in parallel). Failures are
        # absorbed inside ``IBKRClient.get_longname`` (returns "" with a
        # warning log); the cache treats empty as "no name extension
        # available, fall back to ticker-only matching".
        await self._populate_name_token_cache(symbols_needing_news)
        # Phase 12 — when the LLM path is active, classify all
        # surviving non-overridden symbols concurrently. Override
        # symbols skip the classifier entirely (their ``catalyst``
        # comes straight from the override dict). The map carries
        # ``None`` for tickers whose evaluation didn't qualify.
        mode = self._resolve_classifier_mode()
        llm_results: dict[str, ClassificationResult] = {}
        if mode == "llm":
            llm_results = await self._classify_via_llm(
                symbols=symbols_needing_news,
                news_map=news_map,
            )
        raw_hits = [
            self._build_hit(
                c,
                floats.get(c.symbol),
                news_map.get(c.symbol, []),
                override=overrides.get(c.symbol),
                llm_result=llm_results.get(c.symbol),
                mode=mode,
                quote=quotes.get(c.symbol),
                rvol=rvol_map.get(c.symbol),
            )
            for c in survivors
        ]
        # Phase 6.11: drop symbols whose catalyst never landed. The
        # 5-pillar rule treats news as mandatory — subscribing bars for
        # a no-catalyst symbol just burns an IBKR slot (cap=10) that the
        # next rescan's catalyst-bearing candidate could use. The 5-min
        # rescan interval recovers any symbol where news lands after
        # first scan. Operator manual overrides (Phase 6.8) attach a
        # synthetic catalyst BEFORE this filter so injected symbols
        # always survive.
        hits: list[ScanHit] = []
        for hit in raw_hits:
            if hit.catalyst is None:
                _log.info(
                    "scanner.dropped_no_catalyst",
                    symbol=hit.symbol,
                    float_shares=hit.float_shares,
                    float_source=hit.float_source,
                )
                continue
            hits.append(hit)
        # Preserve IBKR scanner rank when we can't compute change_pct numerically.
        indexed = list(enumerate(hits))
        indexed.sort(
            key=lambda pair: (
                -(pair[1].change_pct if pair[1].change_pct is not None else 0.0),
                pair[0],
            )
        )
        ranked = [hit for _, hit in indexed]
        # Manual watchlist entries (operator escape hatch) are prepended so
        # they always lead the ranked watchlist regardless of whether IBKR's
        # TOP_PERC_GAIN scan returned them. They bypass every scanner gate
        # (price / gap / premarket-vol / float / rvol / catalyst); the risk
        # engine is the only safety net once a strategy fires a signal.
        # Symbols already returned by the IBKR scan keep their organic
        # ScanHit (with real float / rvol / catalyst) -- the manual entry
        # is a no-op upgrade for those.
        return self._merge_manual_watchlist(ranked)

    async def _fetch_ibkr_gappers(self) -> list[Contract]:
        """Issue one TOP_PERC_GAIN scan snapshot and return the returned contracts."""
        u = self._settings.universe
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
        scan_rows = await self._ibkr.ib.reqScannerDataAsync(
            sub, scannerSubscriptionFilterOptions=tag_filters
        )
        # TOP_PERC_GAIN is a streaming scanner subscription on the TWS side —
        # even after reqScannerDataAsync returns the initial snapshot, TWS
        # keeps the slot allocated until cancelScannerSubscription fires.
        # Register-then-cancel keeps the accounting symmetrical and gives
        # cancel_all_subscriptions a chance to sweep on abnormal shutdown.
        scanner_req_id = ref_req_id(scan_rows)
        await self._ibkr.subscriptions.register(
            ActiveSubscription(
                req_id=scanner_req_id,
                kind="scanner",
                symbol=None,
                ref=scan_rows,
            )
        )
        try:
            self._ibkr.ib.cancelScannerSubscription(scan_rows)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            _log.warning("scanner.cancel_scanner_failed", error=str(exc))
        await self._ibkr.subscriptions.unregister(scanner_req_id)
        contracts: list[Contract] = []
        for row in scan_rows:
            details = getattr(row, "contractDetails", None)
            contract = getattr(details, "contract", None) if details is not None else None
            if contract is not None and getattr(contract, "symbol", None):
                contracts.append(contract)
        _log.info("scanner.ibkr_scan_complete", count=len(contracts))
        return contracts

    async def _fetch_floats(self, symbols: list[str]) -> dict[str, FloatData | None]:
        """Resolve float data for every symbol concurrently via the FloatSource chain."""

        async def one(symbol: str) -> tuple[str, FloatData | None]:
            try:
                return symbol, await self._float_source.get_float(symbol)
            except Exception as exc:  # noqa: BLE001 - log + degrade to "float unknown"
                _log.warning("scanner.float_failed", symbol=symbol, error=str(exc))
                return symbol, None

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(results)

    def _apply_float_filter(
        self,
        contracts: list[Contract],
        floats: dict[str, FloatData | None],
    ) -> list[Contract]:
        """Drop symbols whose float is unknown or exceeds ``universe.float_max``.

        Phase 6.3: symbols with ``float_shares=None`` (both yfinance and
        Finnhub returned nothing — typically leveraged ETFs like UVIX/MSTU or
        newly-listed / foreign-domiciled tickers) are dropped outright.
        Pre-6.3 these passed through flagged ``float_unknown`` and burned an
        IBKR bar subscription for symbols the strategies couldn't size anyway.
        Operators investigating a specific drop can grep
        ``scanner.dropped_float_unknown`` in the session JSONL.
        """
        float_max = self._settings.universe.float_max
        survivors: list[Contract] = []
        for contract in contracts:
            data = floats.get(contract.symbol)
            if data is None:
                _log.info(
                    "scanner.dropped_float_unknown",
                    symbol=contract.symbol,
                    sources_attempted=["yfinance", "finnhub"],
                )
                continue
            if data.float_shares > float_max:
                _log.info(
                    "scanner.dropped_high_float",
                    symbol=contract.symbol,
                    float_shares=data.float_shares,
                    float_source=data.source,
                    float_max=float_max,
                )
                continue
            survivors.append(contract)
        return survivors

    def _apply_rvol_filter(
        self,
        contracts: list[Contract],
        quotes: dict[str, tuple[float | None, float | None, int | None]],
        floats: dict[str, FloatData | None],
    ) -> tuple[list[Contract], dict[str, float]]:
        """Drop symbols whose rvol is unknown or below ``universe.rvol_min``.

        rvol = today's session volume / 10-day average daily volume. Both
        inputs come from earlier pillars: today-volume from the IBKR quote
        fetch (premarket included via ``useRTH=False``), avg-volume from
        yfinance's ``averageVolume10days`` field carried on ``FloatData``.

        Symmetric with the float pillar's drop-on-unknown policy: if either
        numerator or denominator is missing, the symbol is dropped with
        ``scanner.dropped_rvol_unknown`` rather than silently passing
        through. The 5-min rescan recovers any ticker whose data lands
        between passes.

        ``rvol_min <= 0`` disables the filter entirely (every symbol passes,
        rvol still computed when both inputs are available). Useful for
        backtest harnesses and for tests that don't care about rvol.
        """
        threshold = self._settings.universe.rvol_min
        survivors: list[Contract] = []
        rvol_map: dict[str, float] = {}
        for contract in contracts:
            symbol = contract.symbol
            today_volume = quotes.get(symbol, (None, None, None))[2]
            float_data = floats.get(symbol)
            avg_volume = float_data.avg_daily_volume if float_data is not None else None
            if threshold <= 0:
                # Pillar disabled — populate rvol when possible but don't drop.
                if today_volume and avg_volume and avg_volume > 0:
                    rvol_map[symbol] = today_volume / avg_volume
                survivors.append(contract)
                continue
            if not today_volume or not avg_volume or avg_volume <= 0:
                _log.info(
                    "scanner.dropped_rvol_unknown",
                    symbol=symbol,
                    today_volume=today_volume,
                    avg_daily_volume=avg_volume,
                )
                continue
            rvol = today_volume / avg_volume
            if rvol < threshold:
                _log.info(
                    "scanner.dropped_low_rvol",
                    symbol=symbol,
                    rvol=round(rvol, 2),
                    rvol_min=threshold,
                    today_volume=today_volume,
                    avg_daily_volume=avg_volume,
                )
                continue
            rvol_map[symbol] = rvol
            survivors.append(contract)
        return survivors, rvol_map

    async def _populate_name_token_cache(self, symbols: list[str]) -> None:
        """Phase 10.5 — fetch ``longName`` per symbol concurrently and seed the cache.

        Idempotent — repeated calls for the same symbol are cheap because
        :meth:`IBKRClient.get_longname` is itself per-process cached, so a
        rescan tick doesn't re-roundtrip TWS for symbols already qualified.
        Failures inside ``get_longname`` are swallowed there; here we
        only need to feed whatever string came back into the cache (the
        cache emits the right one-shot informational event for empty
        longName via :meth:`NameTokenCache.populate`).
        """
        if not symbols:
            return

        async def _one(symbol: str) -> tuple[str, str]:
            try:
                return symbol, await self._ibkr.get_longname(symbol)
            except Exception as exc:  # noqa: BLE001 - degrade to no-name
                _log.warning("scanner.longname_fetch_failed", symbol=symbol, error=str(exc))
                return symbol, ""

        results = await asyncio.gather(*(_one(s) for s in symbols))
        for symbol, longname in results:
            self._name_token_cache.populate(symbol, longname)

    async def _fetch_news(self, symbols: list[str]) -> dict[str, list[NewsItem]]:
        """Fetch news for all surviving symbols concurrently.

        Finnhub is the primary source. When Finnhub returns an empty list
        (HTTP 200 with no items — typical for small-cap biotechs and
        recently-listed Chinese tickers on the free tier), we fall back
        to yfinance's news endpoint sequentially per affected symbol.

        ERNA on 2026-05-06 was the trigger: a clinical readout published
        that day was missed by Finnhub's free-tier ``/company-news``,
        the classifier silently short-circuited on ``no_news``, and the
        symbol was dropped without an LLM call. yfinance.news catches
        most of these gaps without a paid API key. The fallback is
        sequential rather than parallel-then-merge so we don't double
        the API load on the common case where Finnhub has coverage.
        """
        lookback_hours = self._settings.data_sources.news_lookback_hours

        async def one(symbol: str) -> tuple[str, list[NewsItem]]:
            try:
                primary = await self._finnhub.company_news(
                    symbol, hours_back=lookback_hours
                )
            except Exception as exc:  # noqa: BLE001 - log + fall back to "no news"
                _log.warning("scanner.news_failed", symbol=symbol, error=str(exc))
                primary = []
            if primary:
                return symbol, primary
            fallback = await fetch_yfinance_news(symbol, hours_back=lookback_hours)
            if fallback:
                _log.info(
                    "scanner.news_yfinance_fallback_used",
                    symbol=symbol,
                    item_count=len(fallback),
                )
            return symbol, fallback

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(results)

    async def _classify_via_llm(
        self,
        *,
        symbols: list[str],
        news_map: dict[str, list[NewsItem]],
    ) -> dict[str, ClassificationResult]:
        """Phase 12 — classify every surviving non-overridden ticker in parallel.

        One ``asyncio.gather`` over per-ticker tasks. Each task wraps a
        single ``LLMCatalystClassifier.classify`` call. Failures inside
        ``classify`` are reported via the result's ``failure_reason``
        rather than raising, so one ticker's failure doesn't fail the
        batch. Returns ``{symbol: ClassificationResult}`` covering every
        input symbol.
        """
        classifier = self._llm_classifier
        if classifier is None:
            # Resolved mode said llm but classifier isn't wired — defensive,
            # the resolver should have logged the warning already.
            return {}

        async def _one(symbol: str) -> tuple[str, ClassificationResult]:
            try:
                result = await classifier.classify(
                    symbol,
                    news_map.get(symbol, []),
                )
            except Exception as exc:  # noqa: BLE001 - one ticker's bug must not fail the batch
                _log.error(
                    "scanner.classifier_call_unexpected_error",
                    symbol=symbol,
                    error=str(exc),
                )
                result = ClassificationResult(
                    ticker=symbol,
                    qualifies=False,
                    reason="classifier_unexpected_error",
                    failure_reason=str(exc),
                )
            return symbol, result

        if not symbols:
            return {}
        gathered = await asyncio.gather(*(_one(s) for s in symbols))
        return dict(gathered)

    def _build_hit(
        self,
        contract: Contract,
        float_data: FloatData | None,
        news_items: list[NewsItem],
        *,
        override: CatalystOverride | None = None,
        llm_result: ClassificationResult | None = None,
        mode: str = "keyword",
        quote: tuple[float | None, float | None, int | None] | None = None,
        rvol: float | None = None,
    ) -> ScanHit:
        """Assemble a single ``ScanHit`` with catalyst classified and reasons populated.

        Phase 6.8: when ``override`` is provided the classifier is
        bypassed entirely — the injected category becomes the hit's
        catalyst and a dedicated ``catalyst.manual_override_applied``
        event fires so post-session review can distinguish organic
        classifier matches from operator injections.

        Phase 12: when ``mode == "llm"`` AND the classifier produced a
        ``llm_result`` for this symbol, the hit's catalyst category
        comes from the LLM (via ``ClassificationResult.classification``)
        rather than the keyword classifier. ``mode == "none"`` (both
        flags off) leaves catalyst as ``None`` so the no-catalyst drop
        filter dispatches normally.
        """
        float_shares = float_data.float_shares if float_data is not None else None
        float_source = float_data.source if float_data is not None else None
        if override is not None:
            catalyst: str | None = override.category
            _log.info(
                "catalyst.manual_override_applied",
                symbol=contract.symbol,
                category=override.category,
                expires_at=override.expires_at.isoformat(),
                note=override.note,
                injected_at=override.injected_at.isoformat(),
                injected_by=override.injected_by,
            )
        elif mode == "llm" and llm_result is not None:
            if llm_result.qualifies and llm_result.classification is not None:
                catalyst = llm_result.classification.category
            else:
                catalyst = None
        elif mode == "keyword":
            catalyst = (
                classify(
                    news_items,
                    symbol=contract.symbol,
                    max_age_hours=self._settings.data_sources.news_max_age_hours_for_classify,
                    reference_time=datetime.now(UTC),
                    # Phase 10.5 — pass the per-session cache so the
                    # attribution gate can fall back to name-token matching
                    # when the ticker isn't in the headline.
                    name_token_cache=self._name_token_cache,
                )
                if news_items
                else None
            )
        else:
            # mode == "none" or mode == "llm" without llm_result (defensive).
            # No catalyst evaluation runs; the hit drops at the
            # no-catalyst filter downstream.
            catalyst = None
        reasons: list[str] = []
        if float_shares is None:
            reasons.append("float_unknown")
        if catalyst is None:
            reasons.append("no_catalyst")
        price, change_pct, volume = quote if quote is not None else (None, None, None)
        avg_daily_volume = float_data.avg_daily_volume if float_data is not None else None
        return ScanHit(
            symbol=contract.symbol,
            price=price,
            change_pct=change_pct,
            volume=volume,
            float_shares=float_shares,
            catalyst=catalyst,
            float_source=float_source,
            news_items=news_items,
            reasons=reasons,
            rvol=rvol,
            avg_daily_volume=avg_daily_volume,
        )

    def _merge_manual_watchlist(self, ranked: list[ScanHit]) -> list[ScanHit]:
        """Prepend operator-injected manual watchlist entries to the ranked list.

        Gated on ``settings.testing.allow_catalyst_overrides`` -- defence
        in depth so a stale ``data/manual_watchlist.json`` on the operator's
        machine can't influence a live run if the flag is off (mirrors the
        catalyst_overrides pattern). Symbols already in ``ranked`` are
        skipped: the organic scanner hit (with real float / rvol /
        catalyst) takes precedence; the manual entry is a no-op upgrade.

        Logs ``scanner.manual_watchlist_merged`` once per scan with the
        list of operator-added symbols so the JSONL audit trail captures
        every operator decision.
        """
        if not self._settings.testing.allow_catalyst_overrides:
            return ranked
        active = load_active_manual_watchlist(now=datetime.now(UTC))
        if not active:
            return ranked
        existing_symbols = {hit.symbol for hit in ranked}
        manual_hits: list[ScanHit] = []
        for entry in active:
            if entry.symbol in existing_symbols:
                continue
            manual_hits.append(self._manual_entry_to_hit(entry))
        if not manual_hits:
            return ranked
        _log.info(
            "scanner.manual_watchlist_merged",
            symbols=[h.symbol for h in manual_hits],
            count=len(manual_hits),
        )
        return manual_hits + ranked

    def _manual_entry_to_hit(self, entry: ManualWatchlistEntry) -> ScanHit:
        """Build a synthetic ScanHit for a manual watchlist entry.

        No float / rvol / volume enrichment runs here -- the operator
        explicitly bypassed those pillars. ``catalyst`` is the
        ``MANUAL_CATALYST_SENTINEL`` so the no-catalyst drop filter
        downstream lets the symbol survive (the filter checks for
        ``catalyst is None``, not for any specific category).
        """
        return ScanHit(
            symbol=entry.symbol,
            price=None,
            change_pct=None,
            volume=None,
            float_shares=None,
            catalyst=MANUAL_CATALYST_SENTINEL,
            float_source=None,
            news_items=[],
            reasons=["manual_watchlist"],
            rvol=None,
            avg_daily_volume=None,
            manual=True,
        )

    def _load_active_overrides(self) -> dict[str, CatalystOverride]:
        """Return ``{symbol: override}`` when the testing gate is on, else ``{}``.

        Defence in depth: even if a stale override file sits on disk,
        flipping ``testing.allow_catalyst_overrides`` off instantly
        makes the scanner ignore it. No cleanup required.
        """
        if not self._settings.testing.allow_catalyst_overrides:
            return {}
        return load_active_overrides_map(now=datetime.now(UTC))

    async def _fetch_quotes_map(
        self,
        contracts: list[Contract],
    ) -> dict[str, tuple[float | None, float | None, int | None]]:
        """Fetch ``(price, change_pct, volume)`` for every contract, parallelized.

        Phase 12.1: this runs immediately after the float filter so the rvol
        pillar can use ``volume`` as its numerator before any catalyst spend.
        Per-symbol failures yield ``(None, None, None)`` -- the rvol filter
        treats those as ``rvol_unknown`` and drops the symbol with a
        structured log.
        """
        if not contracts:
            return {}
        results = await asyncio.gather(
            *(self._fetch_quote(c.symbol, c) for c in contracts)
        )
        return {c.symbol: result for c, result in zip(contracts, results, strict=True)}

    async def _cancel_enrichment_task(
        self,
        symbol: str,
        task: asyncio.Task[Any],
        req_id: int,
    ) -> None:
        """Release a timed-out / errored enrichment request on the IBKR wire.

        ``asyncio.wait_for`` only cancels the awaiting coroutine — the real
        IBKR request keeps its market-data line allocated until
        ``cancelHistoricalData(bar_list)`` lands on the wire. The ``bar_list``
        reference is only available once the task resolves, so we schedule a
        done-callback that fires the wire-cancel + registry unregister when
        ``ib_async`` finally delivers the BDL.

        We deliberately do **not** call ``task.cancel()`` — cancelling the
        Python coroutine would leave us without the BDL and the TWS-side
        subscription would stay allocated until the socket dies. Letting the
        task finish in the background is cheap (one-shot daily request,
        resolves within the normal IBKR backoff window) and gives us a
        concrete ref to cancel.
        """
        loop = asyncio.get_running_loop()

        def _on_done(t: asyncio.Task[Any]) -> None:
            bar_list: Any = None
            if not t.cancelled() and t.exception() is None:
                bar_list = t.result()
            if bar_list is not None:
                with contextlib.suppress(Exception):
                    self._ibkr.ib.cancelHistoricalData(bar_list)
            # Unregister — can't await from a done-callback, schedule it.
            loop.create_task(self._ibkr.subscriptions.unregister(req_id))

        if task.done():
            _on_done(task)
        else:
            task.add_done_callback(_on_done)

    async def _fetch_quote(
        self,
        symbol: str,
        contract: Contract,
    ) -> tuple[float | None, float | None, int | None]:
        """Return ``(price, change_pct, volume)`` via a 2-bar daily history pull.

        Request semantics (Phase 12.1): ``"2 D"`` of ``"1 day"`` bars with
        ``useRTH=False`` so today's bar includes premarket activity. Without
        this, premarket scans returned yesterday's RTH volume as the
        ``today_volume`` proxy, which made the rvol pillar always read ~1.0
        and useless. Today returns ``[yesterday, today]`` during RTH and
        just ``[yesterday]`` very early in the premarket session.

        During the early-premarket case we surface yesterday's close as
        ``price`` and ``0.0`` as ``change_pct`` so the watchlist renders a
        real value instead of dashes; a
        ``scanner.enrichment_premarket_unavailable`` event flags the row
        for operator review.
        """
        # ``asyncio.wait_for`` only cancels the awaiting coroutine — the
        # IBKR-side subscription keeps its market-data line allocated until an
        # explicit ``cancelHistoricalData`` lands. Day-1 paper trading observed
        # timeout leaks accumulating lines across restarts. Wrap the request in
        # ``asyncio.create_task`` + ``asyncio.shield`` so a timeout can surface
        # the partial ``BarDataList`` (``ib_async`` resolves the future to the
        # BDL even on cancel) and we can cancel it on the wire.
        task = asyncio.create_task(
            self._ibkr.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
                keepUpToDate=False,
            )
        )
        req_id = ref_req_id(task)
        await self._ibkr.subscriptions.register(
            ActiveSubscription(
                req_id=req_id,
                kind="historical",
                symbol=symbol,
                ref=task,
            )
        )
        try:
            bars = await asyncio.wait_for(
                asyncio.shield(task), timeout=self._enrichment_timeout_seconds
            )
        except TimeoutError:
            _log.info("scanner.enrichment_timeout", symbol=symbol)
            await self._cancel_enrichment_task(symbol, task, req_id)
            return None, None, None
        except Exception as exc:  # noqa: BLE001 - IB surfaces many error shapes
            _log.warning("scanner.enrichment_failed", symbol=symbol, error=str(exc))
            await self._cancel_enrichment_task(symbol, task, req_id)
            return None, None, None
        else:
            # Success — the BarDataList is the result; update the registry ref
            # to the concrete BDL (in case any later sweep has to cancel) and
            # unregister now that the one-shot request is complete on TWS side.
            await self._ibkr.subscriptions.unregister(req_id)

        if not bars:
            _log.warning(
                "scanner.enrichment_failed",
                symbol=symbol,
                error="no_bars_returned",
            )
            return None, None, None
        if len(bars) == 1:
            _log.info("scanner.enrichment_premarket_unavailable", symbol=symbol)
            yest = bars[-1]
            return float(yest.close), 0.0, int(yest.volume)

        prev_close = float(bars[-2].close)
        today = bars[-1]
        last = float(today.close)
        volume = int(today.volume)
        change_pct = (last - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
        return last, change_pct, volume
