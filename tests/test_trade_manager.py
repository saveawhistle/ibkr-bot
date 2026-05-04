"""Tests for ``bot.execution.trade_manager`` — scale-out + trailing exits on the tail half."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.brokerage.ibkr_client import IBKRClient
from bot.config import AccountConfig, ExecutionConfig, RiskConfig, Settings
from bot.execution.executor import Executor, _BracketTrades
from bot.execution.position_state import Position, PositionStore
from bot.execution.trade_manager import TradeManager, _evaluate_trailing_exit  # noqa: PLC2701
from bot.persistence.journal import Journal
from bot.risk import RiskEngine


class _StubEvent:
    """Eventkit-style handler registration — trade_manager tests only need the ``+=`` side."""

    def __init__(self) -> None:
        """Keep a list of subscribers registered via ``+=``."""
        self._handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _StubEvent:
        """Register a handler (matches eventkit ``event += handler``)."""
        self._handlers.append(handler)
        return self


class _TradeStub:
    """Minimal ``ib_async.Trade`` stand-in for trade_manager tests."""

    _next_id = 9000

    def __init__(self, order: Any, contract: Any) -> None:
        """Stash the order + contract; assign a unique orderId if missing."""
        self.order = order
        self.contract = contract
        self.fills: list[Any] = []
        # Phase 4k: executor's subscribe_commission hooks this on each placed trade.
        self.commissionReportEvent = _StubEvent()
        self._done = False
        if not getattr(order, "orderId", 0):
            order.orderId = _TradeStub._next_id
            _TradeStub._next_id += 1

    def isDone(self) -> bool:  # noqa: N802 - mirrors ib_async
        """Cancel flag for idempotent double-cancels."""
        return self._done


class _FakeIB:
    """``ib_async.IB`` stand-in that records placed + cancelled orders."""

    def __init__(self) -> None:
        """Keep a record of every placeOrder + cancelOrder for assertion."""
        self.placed: list[_TradeStub] = []
        self.cancelled: list[Any] = []

    def placeOrder(self, contract: Any, order: Any) -> _TradeStub:  # noqa: N802
        """Record + return the stubbed Trade."""
        trade = _TradeStub(order=order, contract=contract)
        self.placed.append(trade)
        return trade

    def cancelOrder(self, order: Any, manualCancelOrderTime: str = "") -> None:  # noqa: N802, N803
        """Record the cancel."""
        self.cancelled.append(order)


def _fake_ibkr() -> tuple[MagicMock, _FakeIB]:
    """Build an IBKRClient MagicMock wired to a fresh _FakeIB."""
    ib = _FakeIB()
    client = MagicMock(spec=IBKRClient)
    client.ib = ib
    contract = MagicMock(symbol="TEST")
    client.qualify_stock = AsyncMock(return_value=contract)
    client.account_summary = AsyncMock(
        return_value={
            "AvailableFunds": "1000000",
            "BuyingPower": "2000000",
            "NetLiquidation": "1000000",
            "DayTradesRemaining": "-1",
        }
    )
    return client, ib


def _settings(
    *,
    post_scaleout_stop_mode: str = "adjustable_to_trail",
    runner_target_enabled: bool = False,
) -> Settings:
    """Default Settings with Phase 4b risk defaults.

    ``post_scaleout_stop_mode`` (Phase 6.14) selects the post-scale-out
    stop shape: one of ``"static_breakeven"`` (Phase 4e),
    ``"adjustable_to_trail"`` (Phase 4h), or ``"immediate_trail"``
    (Phase 6.14 default in production — overridden here to preserve
    the pre-6.14 test behaviour). ``runner_target_enabled`` opts into
    the Phase 4i runner LMT leg — default off per the methodology.
    """
    base = Settings()
    return base.model_copy(
        update={
            "account": AccountConfig(mode="paper"),
            "execution": ExecutionConfig(
                rth_only=True,
                require_paper_mode=True,
                post_scaleout_stop_mode=post_scaleout_stop_mode,  # type: ignore[arg-type]
                runner_target_enabled=runner_target_enabled,
            ),
            "risk": RiskConfig(),
        }
    )


def _build_position(*, shares: int = 100, entry: float = 10.0, stop: float = 9.0) -> Position:
    """Pre-built ``open`` Position for TradeManager tests (bypasses the exec path)."""
    return Position(
        symbol="TEST",
        strategy="gap_and_go",
        shares=shares,
        avg_price=entry,
        stop_price=stop,
        scale_out_price=entry + 2 * (entry - stop),
        runner_target_price=entry + 3 * (entry - stop),
        parent_order_id=100,
        stop_order_id=101,
        target_order_id=102,
        opened_at=datetime.now(UTC),
        status="open",
    )


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Render a tiny OHLC frame; index is sequential 1-min timestamps."""
    now = datetime(2026, 4, 16, 9, 31, tzinfo=UTC)
    index = [now + timedelta(minutes=i) for i in range(len(rows))]
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"], index=pd.DatetimeIndex(index)
    )


@pytest.fixture
def factory(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Per-test factory yielding a freshly-wired (executor, store, tm, ib) quartet.

    The ``post_scaleout_stop_mode`` kwarg (Phase 6.14) selects the stop
    shape: ``"adjustable_to_trail"`` for Phase 4h semantics (the pre-6.14
    default in this suite), ``"static_breakeven"`` for Phase 4e fallback,
    or ``"immediate_trail"`` for the Phase 6.14 TRAIL-at-scale behaviour.
    """

    async def _make(
        *,
        post_scaleout_stop_mode: str = "adjustable_to_trail",
        runner_target_enabled: bool = False,
    ) -> tuple[Executor, PositionStore, TradeManager, _FakeIB, Journal]:
        ibkr, ib = _fake_ibkr()
        store = PositionStore()
        journal = Journal(db_path=tmp_path / "trades.db")
        settings = _settings(
            post_scaleout_stop_mode=post_scaleout_stop_mode,
            runner_target_enabled=runner_target_enabled,
        )
        risk_engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
        executor = Executor(
            ibkr=cast("IBKRClient", ibkr),
            position_store=store,
            journal=journal,
            risk_engine=risk_engine,
            settings=settings,
        )
        market_data = MagicMock()
        # Phase 7.5: TradeManager now calls unsubscribe_ticks on teardown
        # and subscribe_ticks / subscribe_bars during start_tracking.
        # MagicMock's default returns a MagicMock (not awaitable), so give
        # these an AsyncMock with a bar-stream-shaped return value.
        _fake_stream = MagicMock()
        _fake_stream._bar_list = MagicMock()
        _fake_stream._bar_list.updateEvent = MagicMock()
        market_data.subscribe_bars = AsyncMock(return_value=_fake_stream)
        market_data.unsubscribe_ticks = AsyncMock()
        market_data.subscribe_ticks = AsyncMock()
        trade_manager = TradeManager(
            ibkr=ibkr,
            store=store,
            market_data=market_data,
            executor=executor,
            journal=journal,
            settings=settings,
        )
        return executor, store, trade_manager, ib, journal

    return _make


# ---------- Scale-out behavior ---------- #


@pytest.mark.asyncio
async def test_scale_out_cancels_both_children_and_markets_half(factory: Any) -> None:
    """At +1R, cancel STP + target LMT, market-sell half, install breakeven STP.

    Pins Phase 4e static-breakeven behaviour by disabling the Phase 4h
    adjustable-trail toggle — exercises the classic scale-out path.
    """
    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        # Seed an executor bracket so TradeManager can cancel its children.
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        bars = _bars([(11.9, 12.1, 11.8, 12.0)])  # last_close = 12.0 == entry + 2R
        await tm.on_bar_update(position, bars)

        # Both original children cancelled; one market SELL + one breakeven STP placed.
        assert stop_order in ib.cancelled
        assert target_order in ib.cancelled
        # placeOrder-side types are derived from ib_async's own classes — we just assert count + side.
        assert len(ib.placed) == 2
        market_sell = ib.placed[0]
        new_stop = ib.placed[1]
        assert market_sell.order.action == "SELL"
        assert market_sell.order.totalQuantity == 50
        assert new_stop.order.action == "SELL"
        assert new_stop.order.totalQuantity == 50
        assert new_stop.order.auxPrice == pytest.approx(10.0)

        updated = store.get_active("TEST")
        assert updated is not None
        assert updated.scaled_out is True
        assert updated.shares == 50
        assert updated.scale_partial_pnl == pytest.approx(100.0)  # (12 - 10) * 50
        assert updated.stop_price == pytest.approx(10.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_tick_driven_scale_out_fires_on_first_qualifying_tick(factory: Any) -> None:
    """Phase 7.5: a tick ≥ scale_out_price executes scale-out immediately.

    ``start_tracking`` wires a tick-handler via ``market_data.subscribe_ticks``.
    This test captures the handler the TradeManager registers, then invokes
    it with a tick whose price crosses ``scale_out_price``. Expected outcome
    is identical to the bar-close scale-out path: children cancelled, half
    market-sold, breakeven STP installed, position flagged scaled_out.
    """
    from types import SimpleNamespace

    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        # Capture the on_tick callback TradeManager.start_tracking registers
        # with market_data.subscribe_ticks.
        captured: dict[str, Any] = {}

        async def _capture(symbol: str, on_tick: Any = None, tick_type: str = "Last") -> Any:
            captured["symbol"] = symbol
            captured["on_tick"] = on_tick
            # Return a TickStream-shaped object so handlers that read
            # scale_out_fired work.
            return SimpleNamespace(symbol=symbol, scale_out_fired=False)

        tm._market_data.subscribe_ticks = _capture  # type: ignore[method-assign]
        # Expose a _ticks dict so the handler's _tick_stream_for resolves.
        tick_stream = SimpleNamespace(scale_out_fired=False)
        tm._market_data._ticks = {"TEST": tick_stream}  # type: ignore[attr-defined]

        await tm.start_tracking(position)
        assert "on_tick" in captured, "start_tracking should have wired on_tick"

        # Fire a trade tick whose price crosses scale_out_price (=$12.0).
        tick = SimpleNamespace(price=12.05, size=100, exchange="ARCA", specialConditions="")
        await captured["on_tick"](tick)

        # Same assertions as the bar-close scale-out path.
        assert stop_order in ib.cancelled
        assert target_order in ib.cancelled
        assert len(ib.placed) == 2
        market_sell = ib.placed[0]
        new_stop = ib.placed[1]
        assert market_sell.order.action == "SELL"
        assert market_sell.order.totalQuantity == 50
        assert new_stop.order.auxPrice == pytest.approx(10.0)

        updated = store.get_active("TEST")
        assert updated is not None
        assert updated.scaled_out is True
        assert updated.shares == 50
        # Latch set so duplicate ticks in the same batch don't schedule again.
        assert tick_stream.scale_out_fired is True
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_tick_scale_out_skipped_when_scale_lmt_is_live(factory: Any) -> None:
    """Phase 7.5.1: tick path defers to the Phase 6.14 post-fill scale LMT.

    When ``bracket.scale_lmt`` is present, IBKR holds the order server-side
    and will fill it at exchange latency. Firing our own MKT sell in
    parallel would double-sell (LMT fills, our MKT sell fills, 100% flat
    instead of 50%). The tick handler must no-op in this state.
    """
    from types import SimpleNamespace

    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)

        stop_order = MagicMock(orderId=301, orderType="STP")
        # scale_lmt present — simulates the Phase 6.14 post-fill LMT
        # having been planted after the parent MKT filled.
        scale_lmt_order = MagicMock(orderId=302, orderType="LMT")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
            scale_lmt=cast(
                "Any", _TradeStub(order=scale_lmt_order, contract=MagicMock(symbol="TEST"))
            ),
        )

        captured: dict[str, Any] = {}

        async def _capture(symbol: str, on_tick: Any = None, tick_type: str = "Last") -> Any:
            captured["on_tick"] = on_tick
            return SimpleNamespace(symbol=symbol, scale_out_fired=False)

        tm._market_data.subscribe_ticks = _capture  # type: ignore[method-assign]
        tm._market_data._ticks = {  # type: ignore[attr-defined]
            "TEST": SimpleNamespace(scale_out_fired=False)
        }

        await tm.start_tracking(position)

        # Fire a tick that crosses scale_out_price — handler must still no-op
        # because scale_lmt is on the book.
        tick = SimpleNamespace(price=12.05, size=100, exchange="ARCA", specialConditions="")
        await captured["on_tick"](tick)

        # Nothing placed, nothing cancelled, no scaled_out transition.
        assert ib.placed == []
        assert ib.cancelled == []
        assert store.get_active("TEST").scaled_out is False  # type: ignore[union-attr]
        # Scale_lmt still alive on the bracket.
        assert executor.active_trades["TEST"].scale_lmt is not None
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_tick_below_scale_out_does_nothing(factory: Any) -> None:
    """Phase 7.5: a tick below ``scale_out_price`` must NOT trigger scale-out."""
    from types import SimpleNamespace

    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=201, orderType="STP")
        target_order = MagicMock(orderId=202, orderType="LMT")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=cast("Any", _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))),
        )

        captured: dict[str, Any] = {}

        async def _capture(symbol: str, on_tick: Any = None, tick_type: str = "Last") -> Any:
            captured["on_tick"] = on_tick
            return SimpleNamespace(symbol=symbol, scale_out_fired=False)

        tm._market_data.subscribe_ticks = _capture  # type: ignore[method-assign]
        tm._market_data._ticks = {  # type: ignore[attr-defined]
            "TEST": SimpleNamespace(scale_out_fired=False)
        }

        await tm.start_tracking(position)

        tick = SimpleNamespace(price=11.5, size=100, exchange="ARCA", specialConditions="")
        await captured["on_tick"](tick)

        assert ib.placed == []
        assert store.get_active("TEST").scaled_out is False  # type: ignore[union-attr]
    finally:
        await journal.close()


# ---------- Phase 7.8: pre-scale red-candle exit ---------- #


@pytest.mark.asyncio
async def test_pre_scale_red_candle_exit_fires_on_ross_rule(factory: Any) -> None:
    """Phase 7.8: just-closed bar is red AND closes below prior close → full market-close.

    Three-bar frame: bar[-3] green (close 10.5), bar[-2] red with close 10.2
    (red body AND below prior close 10.5), bar[-1] the freshly-appended next
    minute (arbitrary). ``on_bar_update`` must cancel the bracket, market-sell
    all 100 shares, mark the position closed, and journal the exit as
    ``pre_scale_red_candle``.
    """
    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()  # entry=10.0, stop=9.0, scale_out=12.0, shares=100
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        scale_lmt_order = MagicMock(orderId=102, orderType="LMT")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
            scale_lmt=cast(
                "Any", _TradeStub(order=scale_lmt_order, contract=MagicMock(symbol="TEST"))
            ),
        )

        # bar[-3] green, bar[-2] RED (open 10.5 close 10.2 < prev close 10.5),
        # bar[-1] is the nascent next-minute bar (arbitrary).
        bars = _bars(
            [
                (10.3, 10.5, 10.25, 10.5),  # [-3] closes 10.5 (prior)
                (10.5, 10.52, 10.1, 10.2),  # [-2] RED body + below prior close
                (10.2, 10.21, 10.18, 10.19),  # [-1] in-progress next-minute
            ]
        )

        await tm.on_bar_update(position, bars)

        # Bracket stop + scale LMT both cancelled.
        assert stop_order in ib.cancelled
        assert scale_lmt_order in ib.cancelled
        # One MKT sell placed for the full position (not half — pre-scale).
        assert len(ib.placed) == 1
        close_order = ib.placed[0]
        assert close_order.order.action == "SELL"
        assert close_order.order.totalQuantity == 100

        # Position marked closed with pre-scale-red exit type.
        closed = store.get("TEST")
        assert closed is not None
        assert closed.status == "closed"
        # pnl = (10.2 − 10.0) × 100 = 20.0 (no scale_partial_pnl component).
        assert closed.exit_price == pytest.approx(10.2)
        assert closed.realized_pnl == pytest.approx(20.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_pre_scale_no_exit_when_closed_bar_is_green(factory: Any) -> None:
    """Phase 7.8: green-body just-closed bar must NOT fire the pre-scale exit."""
    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=201, orderType="STP")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
        )
        # bar[-2] is GREEN (close > open), close also above prior close.
        bars = _bars(
            [
                (10.0, 10.2, 9.95, 10.1),
                (10.1, 10.4, 10.05, 10.3),  # [-2] green, close 10.3 > 10.1
                (10.3, 10.31, 10.29, 10.3),  # [-1] in-progress
            ]
        )
        await tm.on_bar_update(position, bars)
        assert ib.placed == []
        assert ib.cancelled == []
        assert store.get_active("TEST").scaled_out is False  # type: ignore[union-attr]
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_pre_scale_no_exit_when_red_body_but_above_prior_close(factory: Any) -> None:
    """Phase 7.8: red-body bar that still closes above prior close does NOT fire.

    Both conditions must hold — red body AND close below prior close.
    """
    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=301, orderType="STP")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
        )
        # bar[-3] close 10.0, bar[-2] open 10.5 close 10.3 (red body) but 10.3 > 10.0.
        bars = _bars(
            [
                (9.9, 10.0, 9.8, 10.0),
                (10.5, 10.55, 10.25, 10.3),  # red body, close 10.3 > prior 10.0
                (10.3, 10.31, 10.29, 10.3),
            ]
        )
        await tm.on_bar_update(position, bars)
        assert ib.placed == []
        assert ib.cancelled == []
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_pre_scale_red_candle_suppressed_post_scale(factory: Any) -> None:
    """Phase 7.8: once scaled_out, the red-candle pre-scale check must NOT fire.

    Post-scale runner is managed by the TRAIL stop (or, in static_breakeven
    mode, by ``_evaluate_trailing_exit``'s extension/ema branches — the
    red-candle branch there is independently suppressed by
    ``red_candle_exit_suppressed=True`` set at ``mark_scaled``).
    """
    executor, store, tm, ib, journal = await factory(post_scaleout_stop_mode="static_breakeven")
    try:
        position = _build_position()
        store.insert_reconciled(position)
        # Transition the store to scaled_out state.
        store.mark_scaled(
            "TEST",
            remaining_shares=50,
            scale_partial_pnl=100.0,
            new_stop_price=10.0,
            new_stop_order_id=999,
            post_scaleout_stop_type="static_breakeven",
            post_scaleout_adjustment_trigger_price=None,
        )
        scaled = store.get_active("TEST")
        assert scaled is not None
        assert scaled.scaled_out is True

        stop_order = MagicMock(orderId=401, orderType="STP")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
        )
        # Red candle that would have fired pre-scale.
        bars = _bars(
            [
                (10.3, 10.5, 10.25, 10.5),
                (10.5, 10.52, 10.1, 10.2),  # red, below prior close
                (10.2, 10.21, 10.18, 10.19),
            ]
        )
        await tm.on_bar_update(scaled, bars)
        # No pre-scale close fired; _evaluate_trailing_exit's red-candle
        # branch is separately suppressed; extension/ema_break don't fire
        # on this frame either.
        assert ib.placed == []
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_pre_scale_red_candle_disabled_by_config(factory: Any) -> None:
    """Phase 7.8: ``pre_scale_red_candle_exit_enabled=False`` disables the trigger."""
    executor, store, tm, ib, journal = await factory(
        post_scaleout_stop_mode="static_breakeven",
    )
    # Rebuild trade_manager with a settings override disabling the feature.
    from bot.execution.trade_manager import TradeManager

    overridden = tm._settings.model_copy(  # type: ignore[attr-defined]
        update={
            "execution": tm._settings.execution.model_copy(  # type: ignore[attr-defined]
                update={"pre_scale_red_candle_exit_enabled": False}
            )
        }
    )
    tm_disabled = TradeManager(
        ibkr=tm._ibkr,  # type: ignore[attr-defined]
        store=store,
        market_data=tm._market_data,  # type: ignore[attr-defined]
        executor=executor,
        journal=journal,
        settings=overridden,
    )
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=501, orderType="STP")
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None,
            stop=cast("Any", _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))),
            target=None,
        )
        bars = _bars(
            [
                (10.3, 10.5, 10.25, 10.5),
                (10.5, 10.52, 10.1, 10.2),
                (10.2, 10.21, 10.18, 10.19),
            ]
        )
        await tm_disabled.on_bar_update(position, bars)
        assert ib.placed == []
        assert ib.cancelled == []
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_no_scale_out_below_first_target(factory: Any) -> None:
    """Price below +1R → no orders, no mutation."""
    executor, store, tm, ib, journal = await factory()
    try:
        position = _build_position()
        store.insert_reconciled(position)
        bars = _bars([(10.0, 10.5, 9.9, 10.4)])  # 10.4 < 11.0
        await tm.on_bar_update(position, bars)
        assert ib.placed == []
        assert ib.cancelled == []
        assert store.get_active("TEST").scaled_out is False  # type: ignore[union-attr]
    finally:
        await journal.close()


# ---------- Trailing-exit triggers ---------- #


def test_trailing_exit_dollar_extension_bar_fires() -> None:
    """Phase 7.9: extension fires on the *just-closed* bar (iloc[-2]).

    Frame shape models production: iloc[-1] is the nascent next-minute
    bar appended by IBKR on ``has_new_bar=True``; iloc[-2] is the bar
    that just finalized — and is the one we evaluate.

    Just-closed bar: 100 × ($12.55 − $10.05) = $250 ≥ $200 threshold.
    """
    rows = [
        (10.0, 10.1, 9.95, 10.05),  # prev closed
        (10.05, 12.55, 10.00, 12.30),  # closed_bar — extension candidate
        (12.30, 12.31, 12.29, 12.30),  # nascent next-minute bar (iloc[-1])
    ]
    bars = _bars(rows)
    trigger = _evaluate_trailing_exit(
        bars,
        entry_price=10.05,
        position_shares=100,
        extension_dollar_threshold=200.0,
    )
    assert trigger == "extension_bar"


def test_trailing_exit_dollar_extension_bar_silent_below_threshold() -> None:
    """Phase 7.9: closed-bar gain below threshold → no extension trigger."""
    rows = [
        (10.0, 10.1, 9.95, 10.05),
        (10.05, 11.55, 10.00, 11.30),  # closed_bar gain = $150 < $200
        (11.30, 11.31, 11.29, 11.30),  # nascent
    ]
    bars = _bars(rows)
    trigger = _evaluate_trailing_exit(
        bars,
        entry_price=10.05,
        position_shares=100,
        extension_dollar_threshold=200.0,
    )
    assert trigger is None


def test_trailing_exit_does_not_fire_on_nascent_bar_spike() -> None:
    """Phase 7.9 regression: a spike on the nascent bar (iloc[-1]) does NOT fire.

    Pre-7.9 this was the buggy failure mode — every call to
    ``_evaluate_trailing_exit`` looked at iloc[-1], so a loud nascent
    bar (or, commonly, a 1-tick nascent bar where high≈open≈close) was
    what got evaluated. Now we evaluate the closed bar, so a quiet
    just-closed bar combined with a gigantic nascent bar is a no-op.
    """
    rows = [
        (10.0, 10.1, 9.95, 10.05),  # prev
        (10.05, 10.10, 10.0, 10.05),  # closed_bar: flat, no extension
        (10.05, 15.00, 10.04, 14.90),  # nascent: huge fake spike
    ]
    bars = _bars(rows)
    trigger = _evaluate_trailing_exit(
        bars,
        entry_price=10.05,
        position_shares=100,
        extension_dollar_threshold=200.0,
    )
    assert trigger is None, "extension must not fire on the nascent trailing bar"


def test_trailing_exit_detects_ema_break_below_entry() -> None:
    """Phase 7.9: 9-EMA break on the just-closed bar, EMA over closed series.

    Builds a rising run (9+ bars) so the EMA tracks high, a "closed_bar"
    that drops below both EMA and entry, then a nascent trailing bar.
    EMA is computed over ``bars.iloc[:-1]`` — the nascent bar's
    transient close doesn't skew the EMA.
    """
    rising = [(10.0 + 0.1 * i, 10.1 + 0.1 * i, 9.95 + 0.1 * i, 10.08 + 0.1 * i) for i in range(15)]
    closed_bar = (9.6, 9.7, 9.5, 9.65)  # iloc[-2] — under EMA and entry
    nascent = (9.65, 9.66, 9.64, 9.65)  # iloc[-1] — quiet nascent
    bars = _bars([*rising, closed_bar, nascent])
    trigger = _evaluate_trailing_exit(bars, entry_price=10.0)
    assert trigger == "ema_break"


def test_trailing_exit_does_not_fire_on_nascent_ema_break() -> None:
    """Phase 7.9 regression: an EMA-break on the *nascent* bar must not fire.

    Pre-7.9 this was a latent issue — if the nascent first tick came
    in below the EMA, the break would fire mid-minute on a bar that
    might recover before its real close.
    """
    rising = [(10.0 + 0.1 * i, 10.1 + 0.1 * i, 9.95 + 0.1 * i, 10.08 + 0.1 * i) for i in range(15)]
    # Closed bar stays above EMA + above entry → no break on the closed bar.
    closed_bar = (11.3, 11.4, 11.29, 11.35)
    # Nascent bar dips below both — must be IGNORED.
    nascent = (11.35, 11.36, 9.0, 9.20)
    bars = _bars([*rising, closed_bar, nascent])
    trigger = _evaluate_trailing_exit(bars, entry_price=10.0)
    assert trigger is None, "ema_break must not fire on the nascent bar"


@pytest.mark.asyncio
async def test_scale_out_adjustable_stop_attributes_set(factory: Any) -> None:
    """Phase 4i default — scale-out places an adjustable STP with the spec'd attributes.

    Entry=10, stop=9 → initial_risk=1. Phase 4i defaults: scale-out at +2R (12.0),
    activation at +1R *relative to scale-out* (so trigger = 12 + 1 = 13), amount=1R.
    Expected: triggerPrice=13, adjustedStopPrice=12, adjustedTrailingAmount=1,
    adjustableTrailingUnit=0, adjustedOrderType=``TRAIL``.
    """
    executor, store, tm, ib, journal = await factory(runner_target_enabled=True)
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        bars = _bars([(11.9, 12.1, 11.8, 12.0)])  # +2R trigger (the methodology 2:1)
        await tm.on_bar_update(position, bars)

        # Three orders land: market-sell half, new STP, new runner LMT.
        assert len(ib.placed) == 3
        _market_sell, new_stop, new_target = ib.placed

        # Adjustable STP attrs.
        assert new_stop.order.orderType == "STP"
        assert new_stop.order.auxPrice == pytest.approx(10.0)  # breakeven base
        assert new_stop.order.triggerPrice == pytest.approx(13.0)
        assert new_stop.order.adjustedOrderType == "TRAIL"
        assert new_stop.order.adjustedStopPrice == pytest.approx(12.0)
        assert new_stop.order.adjustedTrailingAmount == pytest.approx(1.0)
        assert new_stop.order.adjustableTrailingUnit == 0

        # Runner LMT sits at entry + 3R by default (runner_target_multiple=3.0).
        assert new_target.order.orderType == "LMT"
        assert new_target.order.lmtPrice == pytest.approx(13.0)

        # OCA linkage: STP + LMT share group, ocaType=1.
        assert new_stop.order.ocaGroup == new_target.order.ocaGroup
        assert new_stop.order.ocaGroup.startswith("scaleout_TEST_")
        assert new_stop.order.ocaType == 1
        assert new_target.order.ocaType == 1

        # Position carries the Phase 4h bookkeeping.
        updated = store.get_active("TEST")
        assert updated is not None
        assert updated.post_scaleout_stop_type == "adjustable_to_trail"
        assert updated.post_scaleout_adjustment_trigger_price == pytest.approx(13.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_scale_out_skipped_when_position_already_closing(factory: Any) -> None:
    """Phase 4h defensive guard — an OCA fill that flipped us to closing must abort scale-out.

    Without this guard a market SELL on already-exited shares would flip the
    account short. The handler must log ``exit_skipped_position_inactive`` and
    place zero new orders.
    """
    executor, store, tm, ib, journal = await factory()
    try:
        position = _build_position()
        store.insert_reconciled(position)
        store.mark_closing("TEST", reason="stop_hit")
        bars = _bars([(11.9, 12.1, 11.8, 12.0)])  # would normally fire scale-out at +2R

        with capture_logs() as captured:
            await tm.on_bar_update(position, bars)

        assert ib.placed == []
        assert ib.cancelled == []
        events = [e.get("event") for e in captured]
        assert "trade_manager.exit_skipped_position_inactive" in events
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_trailing_exit_skipped_when_position_already_closed(factory: Any) -> None:
    """Defensive guard in trailing exit — server-side OCA winner beat us, no re-sell."""
    executor, store, tm, ib, journal = await factory()
    try:
        scaled = Position(
            symbol="TEST",
            strategy="gap_and_go",
            shares=50,
            avg_price=10.0,
            stop_price=10.0,
            scale_out_price=12.0,
            runner_target_price=13.0,
            parent_order_id=100,
            stop_order_id=201,
            target_order_id=0,
            opened_at=datetime.now(UTC),
            status="open",
            scaled_out=True,
            scale_partial_pnl=50.0,
        )
        store.insert_reconciled(scaled)
        store.mark_closed("TEST", exit_price=10.0, pnl=50.0, closed_at=datetime.now(UTC))
        # Extension-bar frame that would normally fire trailing_exit:
        # closed_bar gain = (15 − 10) × 50 = $250 ≥ $200 threshold.
        # Plus a trailing nascent bar for closed-bar semantics (Phase 7.9).
        bars = _bars(
            [
                (10.0, 10.1, 9.9, 10.0),
                (10.0, 15.0, 9.95, 14.5),  # closed_bar: extension trigger
                (14.5, 14.51, 14.49, 14.5),  # nascent
            ]
        )
        with capture_logs() as captured:
            await tm.on_bar_update(scaled, bars)

        assert ib.placed == []
        events = [e.get("event") for e in captured]
        assert "trade_manager.exit_skipped_position_inactive" in events
    finally:
        await journal.close()


def test_trailing_exit_extension_bar_fires_post_scaleout() -> None:
    """Phase 7.9: extension-bar on the just-closed bar fires post-scale.

    _evaluate_trailing_exit is only invoked post-scale today — the suppression of the red-candle trigger after scaling is enforced by
    the absence of a red-candle branch in the function entirely
    (Phase 7.9 removal). Extension and EMA are the only checks.
    """
    rows = [
        (10.0, 10.1, 9.95, 10.05),
        (10.05, 12.55, 10.00, 12.30),  # closed_bar: $2.50 × 100 = $250 ≥ $200
        (12.30, 12.31, 12.29, 12.30),  # nascent
    ]
    bars = _bars(rows)
    trigger = _evaluate_trailing_exit(
        bars,
        entry_price=10.05,
        position_shares=100,
        extension_dollar_threshold=200.0,
    )
    assert trigger == "extension_bar"


def test_trailing_exit_ema_break_fires_post_scaleout() -> None:
    """Phase 7.9: EMA-break on the just-closed bar fires post-scale."""
    rising = [(10.0 + 0.1 * i, 10.1 + 0.1 * i, 9.95 + 0.1 * i, 10.08 + 0.1 * i) for i in range(15)]
    closed_bar = (9.6, 9.7, 9.5, 9.65)  # dives under EMA + entry
    nascent = (9.65, 9.66, 9.64, 9.65)
    bars = _bars([*rising, closed_bar, nascent])
    trigger = _evaluate_trailing_exit(bars, entry_price=10.0)
    assert trigger == "ema_break"


@pytest.mark.asyncio
async def test_scale_out_places_only_stp_when_runner_disabled(factory: Any) -> None:
    """Phase 4i default — runner off: scale-out places market SELL + lone STP, no LMT, no OCA."""
    executor, store, tm, ib, journal = await factory(runner_target_enabled=False)
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        bars = _bars([(11.9, 12.1, 11.8, 12.0)])
        await tm.on_bar_update(position, bars)

        # Exactly two new orders: market-sell half + lone adjustable STP.
        assert len(ib.placed) == 2
        market_sell, new_stop = ib.placed
        assert market_sell.order.action == "SELL"
        assert new_stop.order.orderType == "STP"
        assert new_stop.order.transmit is True
        # Lone STP: no OCA ties because there is nothing to cancel against.
        assert getattr(new_stop.order, "ocaGroup", "") == ""
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_scale_out_adjustable_trigger_equals_scale_out_plus_activation(factory: Any) -> None:
    """Phase 4i trigger formula: ``position.scale_out_price + activation × initial_risk``.

    Build a position with scale_out_price=12.0 and initial_risk=(10-9)=1.0;
    default activation multiple is 1.0 → trigger=13.0. The position record
    should carry that value under ``post_scaleout_adjustment_trigger_price``.
    """
    executor, store, tm, ib, journal = await factory()
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        bars = _bars([(11.9, 12.1, 11.8, 12.0)])
        await tm.on_bar_update(position, bars)

        updated = store.get_active("TEST")
        assert updated is not None
        assert updated.post_scaleout_adjustment_trigger_price == pytest.approx(13.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_scale_out_sets_red_candle_exit_suppressed_on_position(factory: Any) -> None:
    """Phase 4i: the adjustable-trail scale-out path flips the suppression flag."""
    executor, store, tm, ib, journal = await factory()
    try:
        position = _build_position()
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=101, orderType="STP")
        target_order = MagicMock(orderId=102, orderType="LMT")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        target_trade = _TradeStub(order=target_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=cast("Any", target_trade)
        )

        bars = _bars([(11.9, 12.1, 11.8, 12.0)])  # +2R trigger
        await tm.on_bar_update(position, bars)

        updated = store.get_active("TEST")
        assert updated is not None
        assert updated.scaled_out is True
        assert updated.red_candle_exit_suppressed is True
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_execute_trailing_exit_closes_and_journals(factory: Any) -> None:
    """Trailing-exit trigger → cancel breakeven STP, market-close remainder, mark closed + journal."""
    executor, store, tm, ib, journal = await factory()
    try:
        # Scaled-out position: 50 shares left, $50 already banked.
        position = Position(
            symbol="TEST",
            strategy="gap_and_go",
            shares=50,
            avg_price=10.0,
            stop_price=10.0,
            scale_out_price=12.0,
            runner_target_price=13.0,
            parent_order_id=100,
            stop_order_id=201,
            target_order_id=0,
            opened_at=datetime.now(UTC),
            status="open",
            scaled_out=True,
            scale_partial_pnl=50.0,
        )
        store.insert_reconciled(position)
        stop_order = MagicMock(orderId=201, orderType="STP")
        stop_trade = _TradeStub(order=stop_order, contract=MagicMock(symbol="TEST"))
        executor.active_trades["TEST"] = _BracketTrades(
            parent=None, stop=cast("Any", stop_trade), target=None
        )
        # Extension-bar on the just-closed bar → trailing exit fires.
        # Phase 7.9: iloc[-2] is the target; iloc[-1] is the nascent bar.
        # closed_bar gain = (15 − 10) × 50 = $250 ≥ $200 threshold.
        bars = _bars(
            [
                (10.0, 10.1, 9.9, 10.0),
                (10.0, 15.0, 9.95, 14.5),  # closed_bar: extension trigger
                (14.5, 14.51, 14.49, 14.5),  # nascent
            ]
        )
        with capture_logs() as captured:
            await tm.on_bar_update(position, bars)

        assert stop_order in ib.cancelled
        # Expect a market SELL for the remaining 50.
        assert len(ib.placed) == 1
        assert ib.placed[0].order.action == "SELL"
        assert ib.placed[0].order.totalQuantity == 50

        closed = store.get("TEST")
        assert closed is not None
        assert closed.status == "closed"
        # Total PnL: (14.5 − 10.0) × 50 + 50 banked = $275
        assert closed.realized_pnl == pytest.approx(4.5 * 50 + 50.0)
        events = [e.get("event") for e in captured]
        assert "trade_manager.trailing_exit" in events
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_execute_scale_out_short_circuits_if_already_scaled_via_lmt(factory: Any) -> None:
    """Phase 6.14 — ``_execute_scale_out`` guards against double-scale from the LMT path.

    If the MKT atomic bracket's scale-out LMT has already fired
    (event-driven, setting ``scaled_out=True``), a direct call into
    ``_execute_scale_out`` must no-op. Defensive guard: ``on_bar_update``
    already checks ``scaled_out`` and routes to trailing-exit, but the
    guard inside ``_execute_scale_out`` protects any callers that
    bypass ``on_bar_update``.
    """
    from dataclasses import replace  # noqa: PLC0415

    executor, store, tm, ib, journal = await factory()
    try:
        raw = _build_position(shares=100)
        scaled = replace(raw, shares=50, scaled_out=True, scale_partial_pnl=50.0)
        store.insert_reconciled(scaled)

        with capture_logs() as captured:
            # Call the guarded method directly with a last-close well above
            # scale_out_price. The guard should short-circuit before any
            # MKT-sell is placed.
            await tm._execute_scale_out(scaled, fill_price=14.0)  # noqa: SLF001

        assert len(ib.placed) == 0, "short-circuit must prevent any new orders"
        events = [e.get("event") for e in captured]
        assert "trade_manager.scale_out_skipped_already_scaled_via_lmt" in events
    finally:
        await journal.close()
