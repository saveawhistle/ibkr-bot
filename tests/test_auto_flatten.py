"""Tests for ``bot.orchestrator.AutoFlattenScheduler`` — 15:55 ET hard flatten."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from freezegun import freeze_time
from structlog.testing import capture_logs

from bot.config import Settings
from bot.execution.position_state import Position, PositionStore
from bot.orchestrator import AutoFlattenScheduler, _parse_hh_mm


def _build_active_position(symbol: str = "AAA") -> Position:
    """Minimal ``open`` Position for the scheduler callback to iterate."""
    return Position(
        symbol=symbol,
        strategy="gap_and_go",
        shares=100,
        avg_price=10.0,
        stop_price=9.0,
        scale_out_price=12.0,
        runner_target_price=13.0,
        parent_order_id=1,
        stop_order_id=2,
        target_order_id=3,
        opened_at=datetime.now(UTC),
        status="open",
    )


@pytest.mark.asyncio
async def test_flatten_all_active_fires_for_every_active_position() -> None:
    """Scheduler callback calls ``executor.flatten_symbol`` once per active symbol."""
    store = PositionStore()
    store.insert_reconciled(_build_active_position("AAA"))
    store.insert_reconciled(_build_active_position("BBB"))

    executor = MagicMock()
    executor.flatten_symbol = AsyncMock()

    scheduler = AutoFlattenScheduler(
        executor=cast("Any", executor), store=store, settings=Settings()
    )
    flattened = await scheduler.flatten_all_active()

    assert flattened == 2
    assert executor.flatten_symbol.call_count == 2
    reasons = {call.kwargs["reason"] for call in executor.flatten_symbol.call_args_list}
    assert reasons == {"session_auto_flatten"}


@pytest.mark.asyncio
async def test_flatten_all_active_runs_even_when_halted() -> None:
    """Hard rule: halt state must not block the flatten callback.

    We don't plumb halt state into the scheduler at all; this test is a
    regression guardrail — if someone adds a halt check here, this test
    fails.
    """
    store = PositionStore()
    store.insert_reconciled(_build_active_position("AAA"))
    executor = MagicMock()
    executor.flatten_symbol = AsyncMock()

    scheduler = AutoFlattenScheduler(
        executor=cast("Any", executor), store=store, settings=Settings()
    )
    with capture_logs() as captured:
        await scheduler.flatten_all_active()
    assert executor.flatten_symbol.called
    events = [e.get("event") for e in captured]
    assert "session.auto_flatten" in events


@pytest.mark.asyncio
async def test_scheduler_registers_cron_trigger_at_configured_time() -> None:
    """``start()`` adds a cron job matching ``settings.session.flatten_all`` HH:MM."""
    injected = MagicMock(spec=AsyncIOScheduler)
    store = PositionStore()
    executor = MagicMock()
    settings = Settings()
    scheduler = AutoFlattenScheduler(
        executor=cast("Any", executor),
        store=store,
        settings=settings,
        scheduler=injected,
    )
    with freeze_time("2026-04-17 13:00:00"):
        scheduler.start()
    # add_job called once with a CronTrigger at the configured time.
    assert injected.add_job.call_count == 1
    kwargs = injected.add_job.call_args.kwargs
    trigger = kwargs["trigger"]
    # CronTrigger stores hour/minute as RangeExpression objects indexed by field name.
    fields_by_name = {f.name: f for f in trigger.fields}
    assert str(fields_by_name["hour"].expressions[0]) == "15"
    assert str(fields_by_name["minute"].expressions[0]) == "55"
    assert kwargs["id"] == "session.auto_flatten"
    injected.start.assert_called_once()


def test_parse_hh_mm_round_trip() -> None:
    """Utility parser accepts ``HH:MM`` and rejects malformed input."""
    assert _parse_hh_mm("15:55") == (15, 55)
    assert _parse_hh_mm("09:30") == (9, 30)
    with pytest.raises(ValueError):
        _parse_hh_mm("25:00")
    with pytest.raises(ValueError):
        _parse_hh_mm("noon")


# ---------- Phase 4j: pending-entry cancellation on auto-flatten ---------- #


def _build_pending_entry_trigger_position(symbol: str = "STPL") -> Position:
    """Phase 4j resting STP-LMT parent — no shares, no fill yet."""
    return Position(
        symbol=symbol,
        strategy="gap_and_go",
        shares=100,
        avg_price=0.0,
        stop_price=9.0,
        scale_out_price=12.0,
        runner_target_price=None,
        parent_order_id=1,
        stop_order_id=0,
        target_order_id=0,
        opened_at=datetime.now(UTC),
        status="pending_entry_trigger",
        entry_order_type="STP_LMT",
        entry_trigger_price=10.0,
    )


@pytest.mark.asyncio
async def test_auto_flatten_iterates_pending_entry_trigger_positions() -> None:
    """Scheduler callback flattens resting STP-LMT parents just like filled positions."""
    store = PositionStore()
    store.insert_reconciled(_build_pending_entry_trigger_position("STPL"))
    store.insert_reconciled(_build_active_position("LIVE"))

    executor = MagicMock()
    executor.flatten_symbol = AsyncMock()

    scheduler = AutoFlattenScheduler(
        executor=cast("Any", executor), store=store, settings=Settings()
    )
    flattened = await scheduler.flatten_all_active()

    assert flattened == 2
    flattened_symbols = {call.args[0] for call in executor.flatten_symbol.call_args_list}
    assert flattened_symbols == {"STPL", "LIVE"}
