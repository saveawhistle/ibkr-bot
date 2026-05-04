"""Tests for the Phase 10.3 entry-path latency fixes.

Three independently-testable changes:

1. ``apply_default_tif`` — every order constructed for placement now ships
   with ``tif="DAY"`` so TWS doesn't apply its preset via a Cancelled →
   PreSubmitted cycle (Day-7 paper trading observed ~400 ms broker-side
   per entry from the 10349 cancel/resubmit loop).
2. ``IBKRClient.qualify_stock`` memoises per-symbol qualified contracts.
   Re-asks for the same symbol within a session no longer round-trip TWS.
3. ``IBKRClient.account_summary`` is TTL-cached (default 30 s) with
   explicit invalidation on fills/closes from the executor. The CLI
   ``ping`` path opts out via ``refresh=True``.

The integration paths (executor placing real orders, CLI exercising
ping) are covered by the existing ``test_executor`` / ``test_ibkr_client``
suites; this file isolates the new caching surfaces.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from ib_async import (
    LimitOrder,
    MarketOrder,
    Order,
    Stock,
    StopLimitOrder,
    StopOrder,
)

from bot.brokerage.ibkr_client import IBKRClient
from bot.config import IBKRConfig, Settings
from bot.execution.executor import apply_default_tif


def _stock(symbol: str, con_id: int) -> Stock:
    """Build a real ``Stock`` Contract — ``IBKRClient.qualify_stock`` checks isinstance."""
    contract = Stock(symbol, "SMART", "USD")
    contract.conId = con_id
    return contract


def _settings() -> Settings:
    """IBKRConfig stub used across the cache tests."""
    return Settings(ibkr=IBKRConfig(host="x", port=1, client_id=42))


def _mock_ib() -> MagicMock:
    """``ib_async.IB`` stand-in with the methods IBKRClient touches."""
    ib = MagicMock(name="IB")
    ib.connectAsync = AsyncMock(return_value=None)
    ib.disconnect = MagicMock(return_value=None)
    ib.isConnected = MagicMock(return_value=True)
    ib.reqMarketDataType = MagicMock(return_value=None)
    ib.disconnectedEvent = MagicMock()
    ib.disconnectedEvent.__iadd__ = lambda self, handler: self  # type: ignore[assignment]
    return ib


# ---------------------------------------------------------------------------
# Part 1 — apply_default_tif
# ---------------------------------------------------------------------------


class TestApplyDefaultTif:
    """Phase 10.3 — every order ships with TIF=DAY explicitly.

    ib_async's ``Order.tif`` defaults to ``""`` (empty string); when an
    order leaves the bot with empty tif, TWS applies its account preset
    via a Cancelled → PreSubmitted cycle (Error 10349). Setting tif on
    construction skips the cycle.
    """

    def test_sets_tif_to_day_in_place(self) -> None:
        """Mutates the order and returns it for chaining."""
        order = LimitOrder("BUY", 100, 10.0)
        assert order.tif == ""
        returned = apply_default_tif(order)
        assert order.tif == "DAY"
        assert returned is order

    def test_works_on_every_order_class_used_by_the_bot(self) -> None:
        """Each order subclass we construct must accept the tif assignment."""
        for order in [
            LimitOrder("BUY", 100, 10.0),
            StopOrder("SELL", 100, 9.5),
            StopLimitOrder("BUY", 100, 10.10, 10.00),
            MarketOrder("SELL", 100),
            Order(action="SELL", totalQuantity=100, orderType="TRAIL", auxPrice=0.5),
        ]:
            apply_default_tif(order)
            assert order.tif == "DAY", f"tif not set on {type(order).__name__}"

    def test_overwrites_a_pre_existing_tif(self) -> None:
        """Pre-existing tif (e.g. someone tried IOC) gets replaced — last write wins.

        We don't currently use anything but DAY; the helper's contract
        is "set tif to the project default", not "set if unset".
        """
        order = LimitOrder("BUY", 100, 10.0)
        order.tif = "IOC"
        apply_default_tif(order)
        assert order.tif == "DAY"


# ---------------------------------------------------------------------------
# Part 2 — qualify_stock contract cache
# ---------------------------------------------------------------------------


class TestContractCache:
    """Phase 10.3 — IBKRClient.qualify_stock memoises per-symbol contracts."""

    @pytest.mark.asyncio
    async def test_first_call_round_trips_then_caches(self) -> None:
        """First qualify_stock hits the wire; second is served from cache."""
        ib = _mock_ib()
        contract = _stock("BIYA", 842015853)
        ib.qualifyContractsAsync = AsyncMock(return_value=[contract])
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        first = await client.qualify_stock("BIYA")
        second = await client.qualify_stock("BIYA")
        assert first is second
        ib.qualifyContractsAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_different_symbols_each_round_trip_once(self) -> None:
        """Cache is per-symbol; two distinct symbols miss independently."""
        ib = _mock_ib()
        contracts = [_stock("BIYA", 1), _stock("ZENA", 2)]
        ib.qualifyContractsAsync = AsyncMock(side_effect=[[contracts[0]], [contracts[1]]])
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.qualify_stock("BIYA")
        await client.qualify_stock("ZENA")
        await client.qualify_stock("BIYA")  # cache hit
        await client.qualify_stock("ZENA")  # cache hit
        assert ib.qualifyContractsAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_clears_cache(self) -> None:
        """A reconnect path must not serve a stale contract from a prior socket."""
        ib = _mock_ib()
        contract_a = _stock("BIYA", 1)
        contract_b = _stock("BIYA", 2)
        ib.qualifyContractsAsync = AsyncMock(side_effect=[[contract_a], [contract_b]])
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        first = await client.qualify_stock("BIYA")
        assert first is contract_a
        await client.disconnect()
        # Re-connect for the second qualify — IBKRClient guards on is_connected().
        ib.isConnected = MagicMock(return_value=True)
        # IBKRClient sets _disconnecting=True in disconnect(); reset to allow connect.
        client._disconnecting = False
        await client.connect()
        second = await client.qualify_stock("BIYA")
        assert second is contract_b
        assert ib.qualifyContractsAsync.await_count == 2


# ---------------------------------------------------------------------------
# Part 3 — account_summary TTL cache + invalidation
# ---------------------------------------------------------------------------


def _account_row(tag: str, value: str) -> Any:
    """Mock an ``AccountValue``-shape object with .tag/.value attributes."""
    row = MagicMock()
    row.tag = tag
    row.value = value
    return row


class TestAccountSummaryCache:
    """Phase 10.3 — TTL cache on accountSummaryAsync with explicit invalidation."""

    @pytest.mark.asyncio
    async def test_first_call_round_trips(self) -> None:
        """First call hits ``accountSummaryAsync`` and populates the cache."""
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        out = await client.account_summary()
        assert out["AvailableFunds"] == "10000"
        ib.accountSummaryAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_within_ttl_serves_cached(self) -> None:
        """Second call within TTL window returns the cached snapshot, no round-trip."""
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.account_summary()
        await client.account_summary()
        await client.account_summary()
        ib.accountSummaryAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_true_forces_round_trip(self) -> None:
        """``refresh=True`` always re-fetches even when the cache is fresh.

        Used by the CLI ``ping`` command where the operator expects current
        values, not a possibly-cached snapshot.
        """
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.account_summary()
        await client.account_summary(refresh=True)
        assert ib.accountSummaryAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_forces_next_call_to_round_trip(self) -> None:
        """``invalidate_account_summary_cache`` drops the cache; next call re-fetches."""
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.account_summary()
        client.invalidate_account_summary_cache()
        await client.account_summary()
        assert ib.accountSummaryAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_when_cache_unset_is_safe(self) -> None:
        """Calling invalidate before the cache has been populated must not raise."""
        client = IBKRClient(settings=_settings(), ib=_mock_ib())
        # No cache populated; should not raise.
        client.invalidate_account_summary_cache()

    @pytest.mark.asyncio
    async def test_returned_dict_is_a_copy(self) -> None:
        """Mutating the returned dict must not corrupt the cached snapshot.

        Important because the risk engine consumes the dict and passes it
        through; if it ever mutated entries, a subsequent cached read
        would see the mutation.
        """
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        first = await client.account_summary()
        first["AvailableFunds"] = "0"
        first["EXTRA_KEY"] = "tampered"
        second = await client.account_summary()
        assert second["AvailableFunds"] == "10000"
        assert "EXTRA_KEY" not in second

    @pytest.mark.asyncio
    async def test_disconnect_clears_cache(self) -> None:
        """Disconnecting drops the cache so a reconnected session re-fetches."""
        ib = _mock_ib()
        ib.accountSummaryAsync = AsyncMock(
            return_value=[_account_row("AvailableFunds", "10000")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.account_summary()
        await client.disconnect()
        assert client._account_summary_cache is None

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_one_round_trip(self) -> None:
        """Two coroutines awaiting account_summary at once should share one fetch.

        Without the asyncio.Lock guard each would miss the cache on its
        first read and fire its own ``accountSummaryAsync`` round-trip.
        """
        import asyncio  # noqa: PLC0415 - local import keeps the helper self-contained

        ib = _mock_ib()

        async def slow_fetch() -> list[Any]:
            await asyncio.sleep(0.01)
            return [_account_row("AvailableFunds", "10000")]

        ib.accountSummaryAsync = AsyncMock(side_effect=slow_fetch)
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        results = await asyncio.gather(
            client.account_summary(),
            client.account_summary(),
            client.account_summary(),
        )
        for result in results:
            assert result["AvailableFunds"] == "10000"
        assert ib.accountSummaryAsync.await_count == 1


# ---------------------------------------------------------------------------
# Part 4 — Phase 10.5 longName cache (lives in IBKRClient alongside Phase 10.3)
# ---------------------------------------------------------------------------


def _contract_details(longname: str) -> Any:
    """Mock a ``ContractDetails``-shape object with .longName attribute."""
    cd = MagicMock()
    cd.longName = longname
    return cd


class TestLongnameCache:
    """Phase 10.5 — IBKRClient.get_longname memoises per-symbol longName lookups."""

    @pytest.mark.asyncio
    async def test_first_call_round_trips_then_caches(self) -> None:
        """First get_longname hits ``reqContractDetails``; second is cached."""
        ib = _mock_ib()
        ib.reqContractDetailsAsync = AsyncMock(
            return_value=[_contract_details("SHUTTLE PHARMACEUTICAL HOLDINGS INC")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        first = await client.get_longname("SHPH")
        second = await client.get_longname("SHPH")
        assert first == "SHUTTLE PHARMACEUTICAL HOLDINGS INC"
        assert first == second
        ib.reqContractDetailsAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_response_caches_empty_string(self) -> None:
        """A symbol IBKR can't resolve (e.g. delisted) caches as empty string.

        Phase 10.5: SBLX 2026-05-01 returned ``Error 200: No security
        definition``. Empty-string is itself a real cache entry so
        repeated lookups don't re-roundtrip.
        """
        ib = _mock_ib()
        ib.reqContractDetailsAsync = AsyncMock(return_value=[])
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        result = await client.get_longname("SBLX")
        assert result == ""
        # Second call should hit cache, not re-fetch.
        await client.get_longname("SBLX")
        ib.reqContractDetailsAsync.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_swallowed_caches_empty(self) -> None:
        """A TimeoutError or other exception caches empty + warns; doesn't propagate."""
        ib = _mock_ib()
        ib.reqContractDetailsAsync = AsyncMock(side_effect=RuntimeError("boom"))
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        result = await client.get_longname("BORKED")
        assert result == ""

    @pytest.mark.asyncio
    async def test_different_symbols_each_round_trip_once(self) -> None:
        """Cache is per-symbol; two distinct symbols miss independently."""
        ib = _mock_ib()
        ib.reqContractDetailsAsync = AsyncMock(
            side_effect=[
                [_contract_details("SHUTTLE PHARMACEUTICAL HOLDINGS INC")],
                [_contract_details("ATLAS LITHIUM INC")],
            ]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.get_longname("SHPH")
        await client.get_longname("ATLX")
        await client.get_longname("SHPH")  # cached
        await client.get_longname("ATLX")  # cached
        assert ib.reqContractDetailsAsync.await_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_clears_cache(self) -> None:
        """Reconnect path must not serve stale longName from prior socket."""
        ib = _mock_ib()
        ib.reqContractDetailsAsync = AsyncMock(
            return_value=[_contract_details("FIRST NAME")]
        )
        client = IBKRClient(settings=_settings(), ib=ib)
        await client.connect()
        await client.get_longname("X")
        await client.disconnect()
        assert client._longname_cache == {}

    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self) -> None:
        """Calling before connect surfaces the same RuntimeError as qualify_stock."""
        client = IBKRClient(settings=_settings(), ib=_mock_ib())
        # _mock_ib.isConnected returns True; flip to simulate disconnected client.
        client._ib.isConnected = MagicMock(return_value=False)
        with pytest.raises(RuntimeError):
            await client.get_longname("X")
