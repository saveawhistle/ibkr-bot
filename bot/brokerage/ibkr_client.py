"""Async wrapper around ``ib_async.IB`` — connection lifecycle, reconnect, account + contract helpers.

Phase 5.4 adds a :class:`SubscriptionRegistry` that tracks every outstanding
``reqHistoricalData`` / ``reqScannerData`` / ``reqMktData`` call placed against
TWS, plus a :meth:`IBKRClient.cancel_all_subscriptions` sweep invoked from
:meth:`IBKRClient.disconnect`. Day-1 paper trading observed 127 TWS market-data
lines persisting across four bot restarts because a socket ``disconnect()``
leaves active subscriptions allocated on the TWS side — only a TWS restart
cleared them. The registry gives us a single point of truth to cancel them
by kind (historical/scanner/market_data) before the socket goes away.

Phase 10.3 adds two latency caches that hang off the same ``IBKRClient``
instance:

* ``_contract_cache`` — per-symbol qualified Stock contracts. Day-7 paper
  trading (BIYA 2026-04-30) showed ~37 ms per ``qualify_stock`` round-trip
  on the executor's signal hot path, despite the same contract having
  already been qualified at scanner-enrichment + market-data-subscribe
  time. Cache hit serves immediately.
* ``_account_summary_cache`` — TTL-bounded copy of the most recent
  ``accountSummaryAsync`` result. Same session showed ~94 ms per call;
  the executor invokes it on every signal even though
  ``AvailableFunds`` / ``BuyingPower`` change only on fills. Cache TTL
  defaults to ``_ACCOUNT_SUMMARY_TTL_SECONDS`` (30 s); explicit
  invalidation hooks fire from the executor on ``position.filled`` /
  ``position.closed`` so the next entry sees fresh values even within
  the TTL window.

Both caches are cleared on ``disconnect`` so a reconnected session does
not serve stale data from a prior socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

import structlog
from ib_async import IB, Contract, Stock

from bot.config import Settings, get_settings

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

SubscriptionKind = Literal["historical", "market_data", "scanner", "tick_by_tick"]

# Phase 10.3 — default TTL for the account-summary cache. 30 s is a balance
# between staleness (post-fill ``AvailableFunds`` lags by up to TTL) and
# round-trip elimination on rapid-fire signals. Belt-and-suspenders:
# explicit invalidation on fills/closes via
# :meth:`IBKRClient.invalidate_account_summary_cache` makes the post-fill
# state fresh on the next entry regardless of the TTL.
_ACCOUNT_SUMMARY_TTL_SECONDS: Final[float] = 30.0


@dataclass
class ActiveSubscription:
    """Bookkeeping for one outstanding IBKR data request.

    ``ref`` holds whatever object ``ib_async`` needs to cancel the request:
    the ``BarDataList`` for historical, the ``ScanDataList``-shaped object
    for scanner, and the ``Contract`` for market_data (``cancelMktData`` takes
    the contract, not the reqId integer). Kind-dispatch lives in
    :meth:`IBKRClient.cancel_all_subscriptions`.
    """

    req_id: int
    kind: SubscriptionKind
    symbol: str | None = None
    ref: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SubscriptionRegistry:
    """Thread-safe-async registry of outstanding IBKR subscriptions.

    Uses :class:`asyncio.Lock` so concurrent ``asyncio.gather`` fan-outs in
    :meth:`bot.scanning.scanner.IBKRScanner._enrich_market_data` (10–15 symbols at
    once) can register/unregister without clobbering the shared dict.
    ``__len__`` is sync by design — operators want a cheap size peek.
    """

    def __init__(self) -> None:
        """Initialise an empty registry with its own asyncio lock."""
        self._subs: dict[int, ActiveSubscription] = {}
        self._lock = asyncio.Lock()

    async def register(self, sub: ActiveSubscription) -> None:
        """Insert ``sub`` under its ``req_id``; last write wins on collision."""
        async with self._lock:
            self._subs[sub.req_id] = sub

    async def unregister(self, req_id: int) -> ActiveSubscription | None:
        """Pop the subscription with ``req_id`` if present; return it (or None)."""
        async with self._lock:
            return self._subs.pop(req_id, None)

    async def list_active(self) -> list[ActiveSubscription]:
        """Snapshot the currently-registered subscriptions as a list."""
        async with self._lock:
            return list(self._subs.values())

    async def drain(self) -> list[ActiveSubscription]:
        """Atomically clear and return every tracked subscription."""
        async with self._lock:
            drained = list(self._subs.values())
            self._subs.clear()
            return drained

    def __len__(self) -> int:
        """Approximate size — readable without acquiring the lock."""
        return len(self._subs)


def ref_req_id(ref: Any) -> int:
    """Best-effort reqId extraction: ``ref.reqId`` when present, else ``id(ref)``.

    ``BarDataList`` / ``ScanDataList`` expose ``.reqId``; anything else we
    touch (contract refs for market_data) falls back to the object's Python
    id — unique for the lifetime of the registered object, which is what
    the registry needs for key stability.
    """
    value = getattr(ref, "reqId", None)
    if isinstance(value, int) and value != 0:
        return value
    return id(ref)


class IBKRClient:
    """Async IBKR client covering connect, reconnect, account summary, and contract qualification.

    Exposes a minimal surface for Phase 1. Later phases layer market data, orders,
    and streaming on top of the same ``IB`` instance held by this client.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        ib: IB | None = None,
        *,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
    ) -> None:
        """Create a client bound to ``settings`` (defaults to the process singleton)."""
        self._settings = settings or get_settings()
        self._ib: IB = ib or IB()
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._intentional_disconnect = False
        self._reconnect_task: asyncio.Task[None] | None = None
        self._log: BoundLogger = structlog.get_logger("bot.brokerage.ibkr_client")
        self._subscriptions = SubscriptionRegistry()
        self._disconnecting = False
        self._ib.disconnectedEvent += self._on_disconnect
        # Phase 10.3 — per-process Contract cache. Symbol → qualified Stock
        # contract; populated on first qualify_stock() call and re-served
        # for every subsequent call. Day-7 paper trading (BIYA) showed
        # ~37 ms per qualify_stock round-trip on the entry hot path; cache
        # eliminates that for the second-and-later qualify on the same
        # symbol (every entry signal after the scanner subscribed bars
        # already qualified the contract). Cleared on disconnect so a
        # reconnected session re-qualifies fresh.
        self._contract_cache: dict[str, Contract] = {}
        # Phase 10.3 — TTL-bounded account-summary cache. Tuple of
        # ``(snapshot_dict, monotonic_cached_at)`` or ``None`` when not yet
        # populated. ``account_summary()`` is called on every entry signal
        # by the executor; the underlying ``accountSummaryAsync()`` is a
        # ~94 ms TWS round-trip. Cache served when fresh, refreshed on
        # expiry or explicit invalidation.
        self._account_summary_cache: tuple[dict[str, str], float] | None = None
        self._account_summary_lock = asyncio.Lock()
        # Phase 10.5 — per-process longName cache. Symbol → ContractDetails
        # ``longName`` populated lazily by :meth:`get_longname` and consumed
        # by the scanner to populate the catalyst :class:`NameTokenCache`.
        # Cached value is the IBKR string (or empty string for delisted /
        # foreign symbols where ``reqContractDetailsAsync`` returns Error 200,
        # the SBLX 2026-05-01 case). Empty string is a real cache entry —
        # ``"not yet looked up"`` is encoded by absence of the key. Cleared
        # on disconnect so a reconnected session re-fetches fresh.
        self._longname_cache: dict[str, str] = {}

    @property
    def subscriptions(self) -> SubscriptionRegistry:
        """Expose the registry — used by scanner/market_data to track their requests."""
        return self._subscriptions

    @property
    def ib(self) -> IB:
        """Expose the underlying ``ib_async.IB`` instance for later-phase modules."""
        return self._ib

    def is_connected(self) -> bool:
        """Return True when the underlying IB socket is live."""
        return bool(self._ib.isConnected())

    async def connect(self) -> None:
        """Connect to TWS/Gateway using host/port/client_id from settings."""
        self._intentional_disconnect = False
        host = self._settings.ibkr.host
        port = self._settings.ibkr.port
        client_id = self._settings.ibkr.client_id
        self._log.info(
            "ibkr.connecting",
            host=host,
            port=port,
            client_id=client_id,
        )
        await self._ib.connectAsync(host=host, port=port, clientId=client_id)
        self._ib.reqMarketDataType(1)  # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
        self._log.info("ibkr.connected", host=host, port=port, client_id=client_id)

    async def disconnect(self) -> None:
        """Disconnect cleanly: sweep subscriptions, cancel reconnect, close socket.

        Idempotent — a second call is a no-op. The subscription sweep runs
        *before* ``self._ib.disconnect()`` because TWS only releases
        market-data lines when an explicit cancel lands on the wire; a
        raw socket close leaves them allocated against the clientId.
        """
        if self._disconnecting:
            return
        self._disconnecting = True
        self._intentional_disconnect = True
        task = self._reconnect_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self.cancel_all_subscriptions()
        # Phase 10.3 — clear per-session caches on disconnect so a
        # reconnect path re-qualifies / re-fetches fresh (defensive;
        # conIds are stable per-symbol so reuse would also be safe,
        # but caches are bound to the open socket conceptually).
        # Phase 10.5 — same treatment for the longName cache.
        self._contract_cache.clear()
        self._account_summary_cache = None
        self._longname_cache.clear()
        if self._ib.isConnected():
            self._ib.disconnect()
        self._log.info("ibkr.disconnected")

    async def cancel_all_subscriptions(self) -> None:
        """Cancel every outstanding IBKR data subscription the bot placed.

        Dispatches per ``kind`` to the matching ``cancel*`` method on
        ``ib_async.IB``. Each cancel is wrapped — a single bad ref (e.g. a
        stale bar list for an already-closed symbol) must not block the
        rest. Tolerates a dead socket: if the connection is gone,
        ``ib_async`` typically raises; we swallow, log, and keep sweeping.

        Emits ``ibkr.subscriptions_swept`` with the counts for forensic
        review so operators can correlate a shutdown with a drop in the
        TWS market-data-lines counter.
        """
        drained = await self._subscriptions.drain()
        if not drained:
            self._log.info("ibkr.subscriptions_swept", cancelled=0, failed=0)
            return
        cancelled = 0
        failures: list[str] = []
        for sub in drained:
            try:
                self._cancel_one(sub)
            except Exception as exc:  # noqa: BLE001 - many IBKR shapes on cancel
                failures.append(f"{sub.kind}:{sub.symbol or sub.req_id}:{exc}")
                self._log.warning(
                    "ibkr.subscription_cancel_failed",
                    kind=sub.kind,
                    symbol=sub.symbol,
                    req_id=sub.req_id,
                    error=str(exc),
                )
                continue
            cancelled += 1
        self._log.info(
            "ibkr.subscriptions_swept",
            cancelled=cancelled,
            failed=len(failures),
            kinds={
                k: sum(1 for s in drained if s.kind == k)
                for k in ("historical", "market_data", "scanner")
            },
        )

    def _cancel_one(self, sub: ActiveSubscription) -> None:
        """Dispatch a single cancel call to the right IBKR method."""
        if sub.ref is None:
            return
        if sub.kind == "historical":
            self._ib.cancelHistoricalData(sub.ref)
        elif sub.kind == "scanner":
            self._ib.cancelScannerSubscription(sub.ref)
        elif sub.kind == "market_data":
            self._ib.cancelMktData(sub.ref)
        elif sub.kind == "tick_by_tick":
            # Phase 7.5: cancelTickByTickData takes (contract, tickType).
            # The MarketData-level unsubscribe_ticks is the primary path;
            # this sweep-cancel is a safety net during emergency shutdown.
            with contextlib.suppress(Exception):
                self._ib.cancelTickByTickData(sub.ref.contract, "Last")

    async def account_summary(self, *, refresh: bool = False) -> dict[str, str]:
        """Return the current account summary as a flat ``tag -> value`` mapping.

        Phase 10.3: cached for up to ``_ACCOUNT_SUMMARY_TTL_SECONDS``
        (30 s default) to skip the per-signal TWS round-trip on the
        executor's hot path. The risk engine consumes only
        ``AvailableFunds`` / ``BuyingPower`` / ``DayTradesRemaining``,
        none of which need penny-precision; the margin gate
        (``AvailableFunds × 0.95``) is coarse enough that staleness up
        to one TTL is benign.

        Pass ``refresh=True`` to force a fresh fetch — used by the CLI
        ``status`` command where the operator expects current values.
        Explicit invalidation via
        :meth:`invalidate_account_summary_cache` is hooked from the
        executor on fills and closes so the next entry sees fresh
        values within-TTL.

        ``asyncio.Lock`` serialises concurrent fetches; without it,
        rapid-fire signals would cause N coroutines to all miss the
        cache and each fire its own ``accountSummaryAsync``, defeating
        the cache and adding artificial contention.
        """
        if not self.is_connected():
            raise RuntimeError("IBKRClient.account_summary called before connect()")
        async with self._account_summary_lock:
            if not refresh and self._account_summary_cache is not None:
                snapshot, cached_at = self._account_summary_cache
                if (time.monotonic() - cached_at) < _ACCOUNT_SUMMARY_TTL_SECONDS:
                    return dict(snapshot)
            rows = await self._ib.accountSummaryAsync()
            snapshot = {row.tag: row.value for row in rows}
            self._account_summary_cache = (snapshot, time.monotonic())
            return dict(snapshot)

    def invalidate_account_summary_cache(self) -> None:
        """Phase 10.3 — drop the cached summary so the next ``account_summary()`` re-fetches.

        Called from the executor on ``position.filled`` (entry reduces
        ``AvailableFunds``) and ``position.closed`` (exit restores it)
        so the risk engine's next gate sees fresh margin/buying-power
        values even inside the TTL window. Safe to call when no cache
        is populated — a no-op in that case.
        """
        self._account_summary_cache = None

    async def qualify_stock(self, symbol: str) -> Contract:
        """Qualify a US stock symbol on SMART routing and return the resolved contract.

        Phase 10.3: serves from per-symbol cache when available so the
        executor's signal hot path doesn't re-roundtrip TWS for a contract
        already qualified at scanner-enrichment / market-data-subscribe
        time. Cache is cleared on disconnect (see ``disconnect``); a
        symbol's qualified contract is stable for the life of a session.
        """
        if not self.is_connected():
            raise RuntimeError("IBKRClient.qualify_stock called before connect()")
        cached = self._contract_cache.get(symbol)
        if cached is not None:
            self._log.debug("ibkr.qualified_cache_hit", symbol=symbol, con_id=cached.conId)
            return cached
        contract = Stock(symbol, "SMART", "USD")
        qualified = await self._ib.qualifyContractsAsync(contract)
        resolved = qualified[0] if qualified else None
        if not isinstance(resolved, Contract):
            raise ValueError(f"Could not qualify stock symbol: {symbol!r}")
        self._contract_cache[symbol] = resolved
        self._log.info("ibkr.qualified", symbol=symbol, con_id=resolved.conId)
        return resolved

    async def get_longname(self, symbol: str) -> str:
        """Phase 10.5 — fetch ``ContractDetails.longName`` for ``symbol``, cached.

        Returns the IBKR longName string (e.g. ``"SHUTTLE PHARMACEUTICAL
        HOLDINGS INC"``) or empty string when the symbol can't be
        resolved (delisted, foreign listing, IBKR Error 200 — the SBLX
        2026-05-01 case). Empty-string is itself a cache entry so a
        repeated lookup of an unresolvable symbol doesn't re-roundtrip.

        Used by the scanner to populate
        :class:`bot.scanning.catalyst.NameTokenCache` before classifying news;
        the catalyst attribution gate's name-extension fallback (Phase
        10.5) uses the resulting tokens.

        Failure modes are absorbed silently and logged at WARNING
        level — the caller treats empty-string as "name extension
        unavailable for this symbol; fall back to ticker-only matching".
        """
        cached = self._longname_cache.get(symbol)
        if cached is not None:
            self._log.debug("ibkr.longname_cache_hit", symbol=symbol)
            return cached
        if not self.is_connected():
            raise RuntimeError("IBKRClient.get_longname called before connect()")
        contract = Stock(symbol, "SMART", "USD")
        try:
            details_list = await asyncio.wait_for(
                self._ib.reqContractDetailsAsync(contract), timeout=10.0
            )
        except (TimeoutError, Exception) as exc:  # noqa: BLE001 - probe-shape failures
            self._log.warning(
                "ibkr.longname_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            self._longname_cache[symbol] = ""
            return ""
        longname = ""
        if details_list:
            longname = str(getattr(details_list[0], "longName", "") or "")
        self._longname_cache[symbol] = longname
        self._log.info(
            "ibkr.longname_fetched",
            symbol=symbol,
            longname=longname,
            had_details=bool(details_list),
        )
        return longname

    def _on_disconnect(self) -> None:
        """Handle socket drops by scheduling a reconnect unless the drop was intentional."""
        if self._intentional_disconnect:
            return
        self._log.warning("ibkr.disconnected_unexpectedly")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = loop.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Retry ``connect`` with exponential backoff until it succeeds or disconnect is requested."""
        delay = self._reconnect_initial_delay
        while not self._intentional_disconnect:
            await asyncio.sleep(delay)
            if self._intentional_disconnect:
                return
            try:
                await self.connect()
                return
            except Exception as exc:  # noqa: BLE001 - log and retry; IB can raise many shapes
                self._log.warning(
                    "ibkr.reconnect_failed",
                    error=str(exc),
                    next_delay_s=min(delay * 2, self._reconnect_max_delay),
                )
                delay = min(delay * 2, self._reconnect_max_delay)
