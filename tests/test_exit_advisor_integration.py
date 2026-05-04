"""Phase 11 — exit-advisor hook integration through TradeManager.on_bar_update.

These tests exercise the actual wiring: a registered advisor receives
``BarFinalizedEvent`` notifications during bar evaluation, an
actionable recommendation flows through the applier, and (when
``hook_acts=true``) results in a SELL order placed via the executor's
existing exit primitives.

Smaller in scope than ``test_trade_manager.py``'s exhaustive matrix —
the goal here is "the hook fires at the right places end-to-end with
the right safety contract", not "every exit code path".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from structlog.testing import capture_logs

from bot.brokerage.ibkr_client import IBKRClient
from bot.config import (
    AccountConfig,
    ExecutionConfig,
    ExitAdvisorConfig,
    RiskConfig,
    Settings,
)
from bot.execution.executor import Executor
from bot.execution.position_state import Position, PositionStore
from bot.execution.trade_manager import TradeManager
from bot.exit_advisor import (
    AdvisorResponse,
    ExitRecommendation,
    register_exit_advisor,
    unregister_exit_advisor,
)
from bot.persistence.journal import Journal
from bot.risk import RiskEngine


class _StubEvent:
    def __init__(self) -> None:
        self._handlers: list[Any] = []

    def __iadd__(self, handler: Any) -> _StubEvent:
        self._handlers.append(handler)
        return self


class _TradeStub:
    _next_id = 9000

    def __init__(self, order: Any, contract: Any) -> None:
        self.order = order
        self.contract = contract
        self.fills: list[Any] = []
        self.commissionReportEvent = _StubEvent()
        self._done = False
        if not getattr(order, "orderId", 0):
            order.orderId = _TradeStub._next_id
            _TradeStub._next_id += 1

    def isDone(self) -> bool:  # noqa: N802
        return self._done


class _FakeIB:
    def __init__(self) -> None:
        self.placed: list[_TradeStub] = []
        self.cancelled: list[Any] = []

    def placeOrder(self, contract: Any, order: Any) -> _TradeStub:  # noqa: N802
        trade = _TradeStub(order=order, contract=contract)
        self.placed.append(trade)
        return trade

    def cancelOrder(self, order: Any, manualCancelOrderTime: str = "") -> None:  # noqa: N802, N803
        self.cancelled.append(order)


def _settings(*, enabled: bool, hook_acts: bool) -> Settings:
    """Settings tuned for the integration tests with overridable exit_advisor block."""
    base = Settings()
    return base.model_copy(
        update={
            "account": AccountConfig(mode="paper"),
            "execution": ExecutionConfig(
                rth_only=True,
                require_paper_mode=True,
                # Deterministic post-scale shape; we don't exercise scale-out here.
                post_scaleout_stop_mode="static_breakeven",
                pre_scale_red_candle_exit_enabled=False,
            ),
            "risk": RiskConfig(),
            "exit_advisor": ExitAdvisorConfig(
                enabled=enabled,
                hook_acts=hook_acts,
                timeout_seconds=2.0,
                log_skipped_events=True,
            ),
        }
    )


def _build_position(*, shares: int = 100, entry: float = 10.0, stop: float = 9.0) -> Position:
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


def _bars(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    """OHLCV frame; index is sequential 1-min timestamps in UTC."""
    now = datetime(2026, 4, 30, 13, 31, tzinfo=UTC)
    index = [now + timedelta(minutes=i) for i in range(len(rows))]
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex(index),
    )


@pytest.fixture
async def make_tm(tmp_path: Path, monkeypatch: Any):  # type: ignore[no-untyped-def]
    """Per-test factory that builds a wired (executor, store, tm, ib) quartet
    parameterised on the exit_advisor enabled/hook_acts pair.

    Also monkeypatches ``bot.exit_advisor.hook.registry.get_settings`` to return the
    test's Settings so the close-notification path (which calls through
    ``position_state.mark_closed`` without an explicit settings arg) sees
    the test config rather than the repo's ``config.yaml`` defaults.
    """
    teardown_callbacks: list[Any] = []

    async def _make(
        *, enabled: bool, hook_acts: bool
    ) -> tuple[Executor, PositionStore, TradeManager, _FakeIB]:
        ib = _FakeIB()
        ibkr_mock = MagicMock(spec=IBKRClient)
        ibkr_mock.ib = ib
        ibkr_mock.qualify_stock = AsyncMock(return_value=MagicMock(symbol="TEST"))
        ibkr_mock.account_summary = AsyncMock(
            return_value={
                "AvailableFunds": "1000000",
                "BuyingPower": "2000000",
                "NetLiquidation": "1000000",
                "DayTradesRemaining": "-1",
            }
        )
        ibkr_mock.invalidate_account_summary_cache = MagicMock()

        store = PositionStore()
        journal = Journal(db_path=tmp_path / "trades.db")
        settings = _settings(enabled=enabled, hook_acts=hook_acts)
        # Hook's notify functions call get_settings() when no explicit
        # settings is passed (e.g. from position_state.mark_closed); patch
        # the import so test config takes effect on those code paths.
        monkeypatch.setattr("bot.exit_advisor.hook.registry.get_settings", lambda: settings)
        risk_engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
        executor = Executor(
            ibkr=cast("IBKRClient", ibkr_mock),
            position_store=store,
            journal=journal,
            risk_engine=risk_engine,
            settings=settings,
        )
        market_data = MagicMock()
        market_data.unsubscribe_ticks = AsyncMock()
        market_data.subscribe_ticks = AsyncMock()
        tm = TradeManager(
            ibkr=cast("IBKRClient", ibkr_mock),
            store=store,
            market_data=market_data,
            executor=executor,
            journal=journal,
            settings=settings,
        )
        teardown_callbacks.append(unregister_exit_advisor)
        return executor, store, tm, ib

    yield _make
    for cb in teardown_callbacks:
        cb()


class _RecordingAdvisor:
    """Records calls + returns a configurable AdvisorResponse from on_event."""

    def __init__(self, response: AdvisorResponse | None = None) -> None:
        self.response = response or AdvisorResponse()
        self.protected_calls: list[Any] = []
        self.event_calls: list[Any] = []
        self.closed_calls: list[tuple[Any, float]] = []

    def on_position_protected(self, position: Any) -> None:
        self.protected_calls.append(position)

    def on_event(self, position: Any, event: Any) -> AdvisorResponse:
        self.event_calls.append(event)
        return self.response

    def on_position_closed(self, position: Any, final_pnl: float) -> None:
        self.closed_calls.append((position, final_pnl))


# ---------------------------------------------------------------------------
# Disabled hook ⇒ identical to pre-Phase-11 behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_hook_does_not_call_advisor_on_bar_update(make_tm: Any) -> None:
    """``enabled=false`` ⇒ on_bar_update never calls the registered advisor."""
    _executor, store, tm, ib = await make_tm(enabled=False, hook_acts=False)
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    bars = _bars(
        [
            (10.0, 10.1, 9.95, 10.05, 1000.0),
            (10.05, 10.15, 10.0, 10.10, 1100.0),
        ]
    )
    await tm.on_bar_update(position, bars)

    assert advisor.event_calls == []
    assert ib.placed == []  # no orders placed on a normal bar with hook off


# ---------------------------------------------------------------------------
# Enabled hook + hook_acts=False ⇒ log-only mode (no orders)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enabled_log_only_mode_calls_advisor_but_does_not_act(make_tm: Any) -> None:
    """``enabled=true, hook_acts=false`` ⇒ advisor sees event, but no order placed."""
    _executor, store, tm, ib = await make_tm(enabled=True, hook_acts=False)
    rec = ExitRecommendation(action="exit_full", reason="9ema break", source="test")
    advisor = _RecordingAdvisor(
        AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
    )
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    bars = _bars(
        [
            (10.0, 10.1, 9.95, 10.05, 1000.0),
            (10.05, 10.15, 10.0, 10.10, 1100.0),
        ]
    )
    with capture_logs() as captured:
        await tm.on_bar_update(position, bars)

    assert len(advisor.event_calls) == 1  # advisor saw the bar
    assert ib.placed == []  # but no SELL placed in log-only mode
    actionable_logs = [e for e in captured if e["event"] == "exit_advisor.event_actionable"]
    assert len(actionable_logs) == 1


# ---------------------------------------------------------------------------
# Enabled hook + hook_acts=True ⇒ actionable recommendation triggers exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actionable_exit_full_places_market_sell_when_hook_acts_true(
    make_tm: Any,
) -> None:
    """``enabled=true, hook_acts=true`` + actionable exit_full ⇒ market SELL placed,
    position transitions to closed, journal/history updated, executor untracked."""
    executor, store, tm, ib = await make_tm(enabled=True, hook_acts=True)
    rec = ExitRecommendation(action="exit_full", reason="9ema break", source="test")
    advisor = _RecordingAdvisor(
        AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
    )
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    bars = _bars(
        [
            (10.0, 10.1, 9.95, 10.05, 1000.0),
            (10.05, 10.15, 10.0, 10.10, 1100.0),
        ]
    )
    with capture_logs() as captured:
        await tm.on_bar_update(position, bars)

    # SELL order placed at full size (no scale-out yet, so position.shares=100).
    assert len(ib.placed) == 1
    sell = ib.placed[0]
    assert sell.order.action == "SELL"
    assert sell.order.totalQuantity == 100

    # Position closed; advisor's on_position_closed fired with final PnL.
    closed = store.get("TEST")
    assert closed is not None
    assert closed.status == "closed"
    assert len(advisor.closed_calls) == 1
    closed_pos, final_pnl = advisor.closed_calls[0]
    assert closed_pos.symbol == "TEST"

    # Forensic logs present.
    log_events = [e["event"] for e in captured]
    assert "exit_advisor.event_actionable" in log_events
    assert "exit_advisor.applied_exit_full" in log_events
    assert "trade_manager.advisor_exit" in log_events


# ---------------------------------------------------------------------------
# Hold response ⇒ falls through to bot's normal logic (no exit, no order)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hold_response_falls_through_to_normal_bot_logic(make_tm: Any) -> None:
    """An advisor that returns a 'hold' recommendation does NOT short-circuit
    the bot's existing logic — but on a normal non-trigger bar, neither the
    advisor's hold nor the bot's logic places an order."""
    _executor, store, tm, ib = await make_tm(enabled=True, hook_acts=True)
    rec = ExitRecommendation(action="hold", reason="all green")
    advisor = _RecordingAdvisor(
        AdvisorResponse(recommendation=rec, evaluation_performed=True, reasoning="...")
    )
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    bars = _bars(
        [
            (10.0, 10.1, 9.95, 10.05, 1000.0),
            (10.05, 10.15, 10.0, 10.10, 1100.0),
        ]
    )
    await tm.on_bar_update(position, bars)

    assert len(advisor.event_calls) == 1
    assert ib.placed == []  # hold + no scale-out trigger ⇒ no orders


# ---------------------------------------------------------------------------
# Position closed notification fires from mark_closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_closed_notification_fires_from_mark_closed(make_tm: Any) -> None:
    """Calling ``store.mark_closed`` directly should fire the close hook (with enabled)."""
    _executor, store, _tm, _ib = await make_tm(enabled=True, hook_acts=False)
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    store.mark_closing("TEST", reason="test")
    store.mark_closed("TEST", exit_price=10.20, pnl=20.0, closed_at=datetime.now(UTC))

    assert len(advisor.closed_calls) == 1
    pos, pnl = advisor.closed_calls[0]
    assert pos.symbol == "TEST"
    assert pos.status == "closed"
    assert pnl == 20.0


@pytest.mark.asyncio
async def test_position_closed_notification_silent_when_disabled(make_tm: Any) -> None:
    """With enabled=false, mark_closed must NOT call the advisor."""
    _executor, store, _tm, _ib = await make_tm(enabled=False, hook_acts=False)
    advisor = _RecordingAdvisor()
    register_exit_advisor(advisor)

    position = _build_position()
    store.insert_reconciled(position)
    store.mark_closing("TEST", reason="test")
    store.mark_closed("TEST", exit_price=10.20, pnl=20.0, closed_at=datetime.now(UTC))

    assert advisor.closed_calls == []
