"""L2StreamAdapter tests with mocked ib_async objects (no live IBKR)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.exit_advisor.market.l2_adapter import (
    L2StreamAdapter,
    translate_depth_event,
    translate_print_event,
)
from bot.exit_advisor.market.l2_events import L2BookUpdate, L2Print


def _ts(s: int) -> datetime:
    return datetime(2026, 5, 5, 13, 30, s, tzinfo=UTC)


# --- translate_depth_event ---


def test_translate_depth_int_operation_and_side() -> None:
    """ib_async forwards IBKR's int codes (operation 0/1/2, side 0/1).
    The translator maps both to canonical literals."""
    raw = SimpleNamespace(operation=0, side=1, position=0, price=10.00, size=100, time=_ts(0))
    out = translate_depth_event(raw, "X")
    assert out is not None
    assert out.operation == "insert"
    assert out.side == "bid"
    assert out.price == 10.00
    assert out.size == 100


def test_translate_depth_string_operation_passes_through() -> None:
    """If a future ib_async release switches to string operation/side,
    the translator already handles them — no breaking change required."""
    raw = SimpleNamespace(
        operation="update", side="ask", position=0, price=10.05, size=50, time=_ts(0)
    )
    out = translate_depth_event(raw, "X")
    assert out is not None
    assert out.operation == "update"
    assert out.side == "ask"


def test_translate_depth_unknown_operation_returns_none(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    raw = SimpleNamespace(operation=99, side=1, position=0, price=10.00, size=100, time=_ts(0))
    with caplog.at_level(logging.WARNING):
        assert translate_depth_event(raw, "X") is None


def test_translate_depth_missing_field_returns_none(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    raw = SimpleNamespace(operation=0, position=0, price=10.00, size=100)  # no side
    with caplog.at_level(logging.WARNING):
        assert translate_depth_event(raw, "X") is None


# --- translate_print_event ---


def test_translate_print_with_book_state() -> None:
    """When book state is supplied, aggressor side gets derived."""
    from collections import deque

    from bot.exit_advisor.market.book_state import BookLevel, BookState

    state = BookState(
        bids=[BookLevel(10.00, 100, "insert", _ts(0))],
        asks=[BookLevel(10.05, 100, "insert", _ts(0))],
        recent_prints=deque(),
        cumulative_volume_at_level={},
        spread=0.05,
    )
    raw = SimpleNamespace(price=10.05, size=200, time=_ts(1))
    out = translate_print_event(raw, "X", state)
    assert out is not None
    assert out.aggressor_side == "buy"


def test_translate_print_without_book_state() -> None:
    """No book state → aggressor unknown."""
    raw = SimpleNamespace(price=10.05, size=200, time=_ts(1))
    out = translate_print_event(raw, "X", None)
    assert out is not None
    assert out.aggressor_side == "unknown"


def test_translate_print_skips_zero_size() -> None:
    raw = SimpleNamespace(price=10.05, size=0, time=_ts(1))
    assert translate_print_event(raw, "X", None) is None


def test_translate_print_skips_zero_price() -> None:
    raw = SimpleNamespace(price=0, size=100, time=_ts(1))
    assert translate_print_event(raw, "X", None) is None


# --- adapter integration with mocked client ---


class _MockClient:
    """Minimal stand-in for IBKRClient. Tests don't need start/stop
    flows — they call ``feed_*_update`` directly to exercise the
    translation + emit path."""

    def __init__(self) -> None:
        self.ib = SimpleNamespace()


def test_adapter_emits_book_update_via_feed() -> None:
    received: list[object] = []
    adapter = L2StreamAdapter(
        ibkr_client=_MockClient(),
        symbol="X",
        consumer=received.append,
    )
    raw = SimpleNamespace(operation=0, side=1, position=0, price=10.00, size=100, time=_ts(0))
    adapter.feed_depth_update(raw)
    assert len(received) == 1
    assert isinstance(received[0], L2BookUpdate)
    assert received[0].price == 10.00


def test_adapter_emits_print_with_derived_aggressor() -> None:
    """The adapter maintains its own minimal book mirror so it can
    derive aggressor for prints arriving after the book is established."""
    received: list[object] = []
    adapter = L2StreamAdapter(
        ibkr_client=_MockClient(),
        symbol="X",
        consumer=received.append,
    )
    # Establish book first.
    adapter.feed_depth_update(
        SimpleNamespace(operation=0, side=1, position=0, price=10.00, size=100, time=_ts(0))
    )
    adapter.feed_depth_update(
        SimpleNamespace(operation=0, side=0, position=0, price=10.05, size=100, time=_ts(0))
    )
    # Now print at the ask → aggressor buy.
    adapter.feed_print_update(SimpleNamespace(price=10.05, size=50, time=_ts(1)))
    prints = [evt for evt in received if isinstance(evt, L2Print)]
    assert len(prints) == 1
    assert prints[0].aggressor_side == "buy"


def test_adapter_print_before_book_is_unknown_aggressor() -> None:
    received: list[object] = []
    adapter = L2StreamAdapter(
        ibkr_client=_MockClient(),
        symbol="X",
        consumer=received.append,
    )
    adapter.feed_print_update(SimpleNamespace(price=10.00, size=100, time=_ts(0)))
    prints = [evt for evt in received if isinstance(evt, L2Print)]
    assert len(prints) == 1
    assert prints[0].aggressor_side == "unknown"


def test_adapter_translates_malformed_event_silently() -> None:
    """A malformed raw event logs a warning and is dropped — adapter
    must not crash on a single bad row."""
    received: list[object] = []
    adapter = L2StreamAdapter(
        ibkr_client=_MockClient(),
        symbol="X",
        consumer=received.append,
    )
    adapter.feed_depth_update(SimpleNamespace())  # no fields
    assert received == []


# --- start() / reqMktDepth wiring ---


def _make_async_qualify(contract: Any) -> Any:
    """Build an awaitable mock for ``IBKRClient.qualify_stock(symbol)``."""

    async def _qualify(_symbol: str) -> Any:
        return contract

    return _qualify


def _make_start_ready_client(contract: Any) -> Any:
    """Construct a MagicMock IBKR client that supports ``L2StreamAdapter.start()``."""
    client = MagicMock(name="IBKRClient")
    client.qualify_stock = _make_async_qualify(contract)
    client.ib = MagicMock(name="IB")
    client.ib.reqMktDepth = MagicMock(return_value=SimpleNamespace(updateEvent=_FakeEvent()))
    client.ib.reqTickByTickData = MagicMock(return_value=SimpleNamespace(updateEvent=_FakeEvent()))
    return client


class _FakeEvent:
    """``+= handler`` accumulator — same pattern as the market_data tests."""

    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _FakeEvent:
        self.handlers.append(handler)
        return self


@pytest.mark.asyncio
async def test_start_passes_is_smart_depth_true_for_smart_routed_contract() -> None:
    """SMART-routed contract MUST receive ``isSmartDepth=True``.

    Regression for the 2026-05-04 CNSP probe finding: ``reqMktDepth``
    against a SMART-routed contract returns IBKR Error 10092 ("Deep
    market data is not supported for this combination of security
    type/exchange") unless ``isSmartDepth=True`` is set. The adapter
    used to omit the flag, which would have produced zero depth events
    in production regardless of L2 entitlement state.
    """
    contract = SimpleNamespace(
        symbol="CNSP", conId=800594194, exchange="SMART", primaryExchange="NASDAQ"
    )
    client = _make_start_ready_client(contract)
    adapter = L2StreamAdapter(
        ibkr_client=client, symbol="CNSP", consumer=lambda _evt: None, num_depth_rows=10
    )
    await adapter.start()
    client.ib.reqMktDepth.assert_called_once()
    args, kwargs = client.ib.reqMktDepth.call_args
    # Positional: (contract,); keyword: numRows + isSmartDepth.
    assert args[0] is contract
    assert kwargs["numRows"] == 10
    assert kwargs["isSmartDepth"] is True, (
        "isSmartDepth must be True for SMART-routed contracts. Without it, IBKR "
        "rejects with Error 10092 even when L2 entitlement is fully active."
    )


@pytest.mark.asyncio
async def test_start_passes_is_smart_depth_false_for_direct_routed_contract() -> None:
    """Contract with explicit non-SMART routing must NOT receive ``isSmartDepth=True``.

    If a future code path qualifies a contract directly to a venue (e.g.
    ``exchange="ISLAND"`` for raw NASDAQ depth), the request goes
    straight to that venue's book and ``isSmartDepth`` must be False.
    Setting True against a real venue would produce a different error
    from IBKR or behave undefined.
    """
    contract = SimpleNamespace(
        symbol="AAPL", conId=265598, exchange="ISLAND", primaryExchange="NASDAQ"
    )
    client = _make_start_ready_client(contract)
    adapter = L2StreamAdapter(
        ibkr_client=client, symbol="AAPL", consumer=lambda _evt: None, num_depth_rows=10
    )
    await adapter.start()
    client.ib.reqMktDepth.assert_called_once()
    _args, kwargs = client.ib.reqMktDepth.call_args
    assert kwargs["isSmartDepth"] is False, (
        "isSmartDepth must be False when the contract has explicit direct-venue "
        "routing — the depth request goes straight to that venue's book."
    )


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    """Re-calling ``start()`` is a no-op (no duplicate subscriptions)."""
    contract = SimpleNamespace(symbol="X", conId=1, exchange="SMART", primaryExchange="NASDAQ")
    client = _make_start_ready_client(contract)
    adapter = L2StreamAdapter(ibkr_client=client, symbol="X", consumer=lambda _evt: None)
    await adapter.start()
    await adapter.start()
    client.ib.reqMktDepth.assert_called_once()
    client.ib.reqTickByTickData.assert_called_once()
