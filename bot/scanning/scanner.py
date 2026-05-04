"""IBKR TOP_PERC_GAIN scan → Finnhub float + news enrichment → ranked morning watchlist.

The scanner implements the non-strategy portion of the 5-Pillar filter (price,
% change, volume, float, catalyst). Price / change% / volume are enforced on the
IBKR side via scanner TagValues; float + catalyst are applied post-scan from
Finnhub. Phase 5.3 adds a numeric market-data enrichment step after the float +
catalyst filter pass, populating ``price``, ``change_pct``, and ``volume`` via
``reqHistoricalData`` ("1 day" × 2 bars) for rows that passed both filters. The
existing sort at :meth:`IBKRScanner.scan_top_gappers` already prefers numeric
``change_pct`` descending when present, so populating these fields upgrades the
watchlist ordering automatically.
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

    async def scan_top_gappers(self) -> list[ScanHit]:
        """Run the scan and return the ranked morning watchlist as ``ScanHit`` rows.

        Phase 6.8: when ``testing.allow_catalyst_overrides`` is on, the
        scanner consults ``data/test_catalyst_overrides.json`` before
        fetching Finnhub news. Symbols with an active injection skip
        the fetch entirely (saves one company-news quota per override)
        and inherit the injected category as their catalyst; everything
        downstream (ScanHit, strategy evaluation, executor) runs
        identically to an organically-classified hit.
        """
        contracts = await self._fetch_ibkr_gappers()
        if not contracts:
            _log.info("scanner.empty_scan")
            return []
        floats = await self._fetch_floats([c.symbol for c in contracts])
        survivors = self._apply_float_filter(contracts, floats)
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
        raw_hits = [
            self._build_hit(
                c,
                floats.get(c.symbol),
                news_map.get(c.symbol, []),
                override=overrides.get(c.symbol),
            )
            for c in survivors
        ]
        # Phase 6.11: drop symbols whose catalyst never landed. the # 5-pillar rule treats news as mandatory — subscribing bars for
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
        # Phase 5.3: enrich only the rows that survived both the float filter
        # (already dropped above) and the catalyst filter (``catalyst is not
        # None``). Enrichment is best-effort — failures populate None and the
        # watchlist renderer falls back to dashes.
        contract_by_symbol = {c.symbol: c for c in survivors}
        await self._enrich_market_data(hits, contract_by_symbol)
        # Preserve IBKR scanner rank when we can't compute change_pct numerically.
        indexed = list(enumerate(hits))
        indexed.sort(
            key=lambda pair: (
                -(pair[1].change_pct if pair[1].change_pct is not None else 0.0),
                pair[0],
            )
        )
        return [hit for _, hit in indexed]

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
                _log.warning(
                    "scanner.longname_fetch_failed", symbol=symbol, error=str(exc)
                )
                return symbol, ""

        results = await asyncio.gather(*(_one(s) for s in symbols))
        for symbol, longname in results:
            self._name_token_cache.populate(symbol, longname)

    async def _fetch_news(self, symbols: list[str]) -> dict[str, list[NewsItem]]:
        """Fetch Finnhub news for all surviving symbols concurrently."""

        lookback_hours = self._settings.data_sources.news_lookback_hours

        async def one(symbol: str) -> tuple[str, list[NewsItem]]:
            try:
                return symbol, await self._finnhub.company_news(symbol, hours_back=lookback_hours)
            except Exception as exc:  # noqa: BLE001 - log + fall back to "no news"
                _log.warning("scanner.news_failed", symbol=symbol, error=str(exc))
                return symbol, []

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(results)

    def _build_hit(
        self,
        contract: Contract,
        float_data: FloatData | None,
        news_items: list[NewsItem],
        *,
        override: CatalystOverride | None = None,
    ) -> ScanHit:
        """Assemble a single ``ScanHit`` with catalyst classified and reasons populated.

        Phase 6.8: when ``override`` is provided the classifier is
        bypassed entirely — the injected category becomes the hit's
        catalyst and a dedicated ``catalyst.manual_override_applied``
        event fires so post-session review can distinguish organic
        classifier matches from operator injections.
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
        else:
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
        reasons: list[str] = []
        if float_shares is None:
            reasons.append("float_unknown")
        if catalyst is None:
            reasons.append("no_catalyst")
        return ScanHit(
            symbol=contract.symbol,
            price=None,
            change_pct=None,
            volume=None,
            float_shares=float_shares,
            catalyst=catalyst,
            float_source=float_source,
            news_items=news_items,
            reasons=reasons,
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

    async def _enrich_market_data(
        self,
        hits: list[ScanHit],
        contracts: dict[str, Contract],
    ) -> None:
        """Populate ``price``/``change_pct``/``volume`` in-place for catalyst-bearing hits.

        Only rows with a non-None ``catalyst`` are enriched — high-float rows
        are already dropped upstream, and ``no_catalyst`` rows aren't worth the
        IBKR round-trip. All per-symbol calls are launched concurrently via
        ``asyncio.gather`` so total wall time is bounded by the slowest symbol
        (or the 2 s timeout), not by N × 2 s.
        """
        enrichable = [h for h in hits if h.catalyst is not None]
        if not enrichable:
            return
        results = await asyncio.gather(
            *(self._fetch_quote(h.symbol, contracts[h.symbol]) for h in enrichable)
        )
        for hit, (price, change_pct, volume) in zip(enrichable, results, strict=True):
            hit.price = price
            hit.change_pct = change_pct
            hit.volume = volume

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

        Request semantics: ``"2 D"`` of ``"1 day"`` RTH bars → typically returns
        ``[yesterday, today]`` during RTH and just ``[yesterday]`` premarket.
        During premarket we surface yesterday's close as ``price`` and ``0.0``
        as ``change_pct`` so the watchlist renders a real value instead of
        dashes; a ``scanner.enrichment_premarket_unavailable`` event flags the
        row for operator review.
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
                useRTH=True,
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
