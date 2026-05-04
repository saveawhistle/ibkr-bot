"""Unit test (mocked) + integration test (requires paper TWS) for ``IBKRClient``."""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.brokerage.ibkr_client import (
    ActiveSubscription,
    IBKRClient,
    SubscriptionRegistry,
    ref_req_id,
)
from bot.config import IBKRConfig, Settings


def _build_settings() -> Settings:
    """Return a Settings instance with a known-recognisable IBKR block for assertions."""
    return Settings(
        ibkr=IBKRConfig(host="mock-host.local", port=14999, client_id=42),
    )


def _build_mock_ib() -> MagicMock:
    """Produce a MagicMock that mimics the ``ib_async.IB`` surface used by IBKRClient."""
    mock_ib = MagicMock(name="IB")
    mock_ib.connectAsync = AsyncMock(return_value=None)
    mock_ib.disconnect = MagicMock(return_value=None)
    mock_ib.isConnected = MagicMock(return_value=True)
    # disconnectedEvent supports ``+=`` to register handlers (ib_async.Event).
    mock_ib.disconnectedEvent = MagicMock()
    mock_ib.disconnectedEvent.__iadd__ = lambda self, handler: self  # type: ignore[assignment]
    return mock_ib


@pytest.mark.asyncio
async def test_connect_passes_settings_to_ib() -> None:
    """IBKRClient.connect() should forward host/port/client_id from settings to ib_async."""
    settings = _build_settings()
    mock_ib = _build_mock_ib()

    client = IBKRClient(settings=settings, ib=mock_ib)
    await client.connect()

    mock_ib.connectAsync.assert_awaited_once_with(
        host="mock-host.local",
        port=14999,
        clientId=42,
    )


def _paper_tws_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to the IBKR paper TWS port succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ping_paper_account() -> None:
    """End-to-end smoke test against a running paper TWS.

    Requires:
      * TWS (or IB Gateway) running locally with the API enabled.
      * Logged into the IBKR **paper** account.
      * Socket listening on the host/port/client-id from ``config.yaml`` (defaults to
        127.0.0.1:7497, client_id 17).

    The test auto-skips if the configured socket is closed so CI and dev machines
    without TWS can still run ``pytest`` cleanly.
    """
    from bot.config import get_settings

    settings = get_settings()
    if not _paper_tws_reachable(settings.ibkr.host, settings.ibkr.port):
        pytest.skip(
            f"Paper TWS not reachable on {settings.ibkr.host}:{settings.ibkr.port}; "
            "start TWS with the API enabled to run this test."
        )

    client = IBKRClient(settings=settings)
    await client.connect()
    try:
        assert client.is_connected()
        summary: dict[str, Any] = await client.account_summary()
        assert summary, "expected at least one row in the account summary"
        assert "NetLiquidation" in summary
    finally:
        await client.disconnect()
    assert not client.is_connected()


# ---------- Phase 5.4: SubscriptionRegistry + cancel_all_subscriptions ----------


@pytest.mark.asyncio
async def test_subscription_registry_register_unregister_roundtrip() -> None:
    """Register, list, unregister — the key matches the ``req_id`` and ``__len__`` tracks size."""
    registry = SubscriptionRegistry()
    sub = ActiveSubscription(req_id=1, kind="historical", symbol="AAA", ref=object())
    await registry.register(sub)
    assert len(registry) == 1
    listed = await registry.list_active()
    assert listed == [sub]
    popped = await registry.unregister(1)
    assert popped is sub
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_subscription_registry_drain_returns_and_clears() -> None:
    """``drain`` atomically returns every sub and empties the registry."""
    registry = SubscriptionRegistry()
    await registry.register(ActiveSubscription(req_id=1, kind="historical", symbol="A"))
    await registry.register(ActiveSubscription(req_id=2, kind="scanner", symbol=None))
    drained = await registry.drain()
    assert {s.req_id for s in drained} == {1, 2}
    assert len(registry) == 0
    # Second drain is a no-op.
    assert await registry.drain() == []


def test_ref_req_id_prefers_req_id_attribute() -> None:
    """``ref_req_id`` uses ``.reqId`` when present and non-zero; else falls back to ``id()``."""

    class _HasReqId:
        reqId = 77  # noqa: N815 - mirror ib_async's attribute name

    class _Zero:
        reqId = 0  # noqa: N815 - mirror ib_async's attribute name

    class _NoReqId:
        pass

    assert ref_req_id(_HasReqId()) == 77
    zero_obj = _Zero()
    assert ref_req_id(zero_obj) == id(zero_obj)
    no_attr_obj = _NoReqId()
    assert ref_req_id(no_attr_obj) == id(no_attr_obj)


@pytest.mark.asyncio
async def test_cancel_all_subscriptions_dispatches_per_kind() -> None:
    """cancel_all_subscriptions routes each ``kind`` to the matching IB method."""
    settings = _build_settings()
    mock_ib = _build_mock_ib()
    mock_ib.cancelHistoricalData = MagicMock()
    mock_ib.cancelScannerSubscription = MagicMock()
    mock_ib.cancelMktData = MagicMock()

    client = IBKRClient(settings=settings, ib=mock_ib)
    hist_ref = object()
    scan_ref = object()
    mkt_ref = object()
    await client.subscriptions.register(
        ActiveSubscription(req_id=1, kind="historical", symbol="A", ref=hist_ref)
    )
    await client.subscriptions.register(
        ActiveSubscription(req_id=2, kind="scanner", symbol=None, ref=scan_ref)
    )
    await client.subscriptions.register(
        ActiveSubscription(req_id=3, kind="market_data", symbol="B", ref=mkt_ref)
    )

    await client.cancel_all_subscriptions()

    mock_ib.cancelHistoricalData.assert_called_once_with(hist_ref)
    mock_ib.cancelScannerSubscription.assert_called_once_with(scan_ref)
    mock_ib.cancelMktData.assert_called_once_with(mkt_ref)
    assert len(client.subscriptions) == 0


@pytest.mark.asyncio
async def test_cancel_all_continues_on_individual_failure() -> None:
    """A single ``cancel*`` raising must not block the sweep for the rest."""
    settings = _build_settings()
    mock_ib = _build_mock_ib()
    mock_ib.cancelHistoricalData = MagicMock(side_effect=RuntimeError("boom"))
    mock_ib.cancelScannerSubscription = MagicMock()
    mock_ib.cancelMktData = MagicMock()

    client = IBKRClient(settings=settings, ib=mock_ib)
    await client.subscriptions.register(
        ActiveSubscription(req_id=1, kind="historical", symbol="A", ref=object())
    )
    await client.subscriptions.register(
        ActiveSubscription(req_id=2, kind="scanner", symbol=None, ref=object())
    )

    await client.cancel_all_subscriptions()

    mock_ib.cancelScannerSubscription.assert_called_once()
    assert len(client.subscriptions) == 0


@pytest.mark.asyncio
async def test_disconnect_calls_cancel_all_before_socket_close() -> None:
    """``disconnect`` must sweep subscriptions before ``ib.disconnect`` hits the socket."""
    settings = _build_settings()
    mock_ib = _build_mock_ib()
    calls: list[str] = []
    mock_ib.cancelHistoricalData = MagicMock(side_effect=lambda ref: calls.append("cancel"))
    mock_ib.disconnect = MagicMock(side_effect=lambda: calls.append("disconnect"))

    client = IBKRClient(settings=settings, ib=mock_ib)
    await client.subscriptions.register(
        ActiveSubscription(req_id=1, kind="historical", symbol="A", ref=object())
    )
    await client.disconnect()

    assert calls == ["cancel", "disconnect"]


@pytest.mark.asyncio
async def test_disconnect_is_idempotent() -> None:
    """Double-disconnect is a no-op on the second call (no double-cancel, no double-close)."""
    settings = _build_settings()
    mock_ib = _build_mock_ib()
    mock_ib.cancelHistoricalData = MagicMock()

    client = IBKRClient(settings=settings, ib=mock_ib)
    await client.subscriptions.register(
        ActiveSubscription(req_id=1, kind="historical", symbol="A", ref=object())
    )
    await client.disconnect()
    await client.disconnect()

    mock_ib.disconnect.assert_called_once()
    mock_ib.cancelHistoricalData.assert_called_once()
