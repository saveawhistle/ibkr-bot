"""Tests for ``bot.persistence.journal`` — open/update/recent across a tmp-path SQLite file."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bot.execution.position_state import Position
from bot.persistence.journal import Journal

_OPEN_TS = datetime(2026, 4, 16, 9, 31, tzinfo=UTC)
_CLOSE_TS = datetime(2026, 4, 16, 9, 45, tzinfo=UTC)


def _position(symbol: str, *, parent_order_id: int, opened_at: datetime = _OPEN_TS) -> Position:
    """Build an ``open``-status Position — ready for ``open_trade``."""
    return Position(
        symbol=symbol,
        strategy="gap_and_go",
        shares=20,
        avg_price=10.05,
        stop_price=9.0,
        scale_out_price=12.0,
        runner_target_price=13.0,
        parent_order_id=parent_order_id,
        stop_order_id=parent_order_id + 1,
        target_order_id=parent_order_id + 2,
        opened_at=opened_at,
        status="open",
        reasons=["break_of_premarket_high", "above_vwap"],
    )


@pytest.mark.asyncio
async def test_open_trade_persists_row(tmp_path: Path) -> None:
    """``open_trade`` writes a row whose fields round-trip cleanly."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        trade_id = await journal.open_trade(_position("AAA", parent_order_id=100))
        assert trade_id > 0
        recent = await journal.recent_trades()
        assert len(recent) == 1
        row = recent[0]
        assert row.symbol == "AAA"
        assert row.strategy == "gap_and_go"
        assert row.shares == 20
        assert row.entry_price == pytest.approx(10.05)
        assert row.exit_price is None
        assert row.pnl is None
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_update_exit_rewrites_same_row(tmp_path: Path) -> None:
    """``update_exit`` updates the existing row, not a new one."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = _position("BBB", parent_order_id=200)
        await journal.open_trade(position)
        closed = replace(position, status="closed", closed_at=_CLOSE_TS)
        await journal.update_exit(closed, exit_price=13.0, pnl=59.0)
        recent = await journal.recent_trades()
        assert len(recent) == 1
        row = recent[0]
        assert row.exit_price == pytest.approx(13.0)
        assert row.pnl == pytest.approx(59.0)
        assert row.closed_at == _CLOSE_TS
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_open_trade_persists_scale_out_and_runner_multiple(tmp_path: Path) -> None:
    """Phase 4e: new columns round-trip with the executor-supplied multiple.

    ``target_price`` carries the runner ceiling (position.runner_target_price),
    ``scale_out_price`` carries the +1R anchor (position.scale_out_price), and
    ``runner_target_multiple_used`` carries whatever the executor passed in.
    """
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(
            _position("ZZZ", parent_order_id=999),
            runner_target_multiple_used=3.0,
        )
        recent = await journal.recent_trades()
        assert len(recent) == 1
        row = recent[0]
        assert row.target_price == pytest.approx(13.0)
        assert row.scale_out_price == pytest.approx(12.0)
        assert row.runner_target_multiple_used == pytest.approx(3.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_open_trade_without_multiple_persists_null(tmp_path: Path) -> None:
    """Legacy callers that don't pass the multiple land a SQL NULL — stays query-compatible."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(_position("YYY", parent_order_id=777))
        recent = await journal.recent_trades()
        assert recent[0].runner_target_multiple_used is None
        assert recent[0].scale_out_price == pytest.approx(12.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_update_exit_persists_post_scaleout_fields(tmp_path: Path) -> None:
    """Phase 4h — post-scale-out stop variant + trigger price round-trip on update_exit."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = _position("PSC", parent_order_id=444)
        await journal.open_trade(position)
        closed = replace(
            position,
            status="closed",
            closed_at=_CLOSE_TS,
            scaled_out=True,
            scale_partial_pnl=50.0,
            post_scaleout_stop_type="adjustable_to_trail",
            post_scaleout_adjustment_trigger_price=12.0,
        )
        await journal.update_exit(
            closed, exit_price=12.5, pnl=75.0, exit_type="scale_out_then_trail"
        )

        async with (await journal._ensure_engine())[1]() as session:  # noqa: SLF001 - test access
            from sqlalchemy import select

            from bot.persistence.journal import TradeRecord

            row = await session.scalar(
                select(TradeRecord).where(TradeRecord.parent_order_id == 444)
            )
        assert row is not None
        assert row.post_scaleout_stop_type == "adjustable_to_trail"
        assert row.post_scaleout_adjustment_trigger_price == pytest.approx(12.0)
        assert row.exit_type == "scale_out_then_trail"
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_phase_4h_migration_is_idempotent_on_legacy_db(tmp_path: Path) -> None:
    """Pre-4h ``trades`` DBs get the new columns added on open without clobbering rows."""
    import aiosqlite

    db_path = tmp_path / "trades.db"
    # Build a pre-4h schema by hand — mirrors a DB written under Phase 4e: the
    # ``scale_out_price`` + ``runner_target_multiple_used`` columns exist, but the
    # Phase 4h columns don't.
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(16) NOT NULL,
                strategy VARCHAR(32) NOT NULL,
                shares INTEGER NOT NULL,
                entry_price FLOAT NOT NULL,
                stop_price FLOAT NOT NULL,
                target_price FLOAT NOT NULL,
                scale_out_price FLOAT,
                runner_target_multiple_used FLOAT,
                exit_price FLOAT,
                pnl FLOAT,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP,
                parent_order_id INTEGER NOT NULL,
                reasons_json TEXT NOT NULL DEFAULT '[]',
                exit_type VARCHAR(32)
            )
            """
        )
        await conn.execute(
            "INSERT INTO trades (symbol, strategy, shares, entry_price, stop_price, "
            "target_price, opened_at, parent_order_id, reasons_json) "
            "VALUES ('LEG', 'gap_and_go', 20, 10.0, 9.0, 13.0, "
            "'2026-04-16 09:31:00.000000+00:00', 555, '[]')"
        )
        await conn.commit()

    journal = Journal(db_path=db_path)
    try:
        # Open-engine path runs the 4h migration; a legacy row must still be readable.
        recent = await journal.recent_trades()
        assert len(recent) == 1
        assert recent[0].symbol == "LEG"
        assert recent[0].post_scaleout_stop_type is None
        assert recent[0].post_scaleout_adjustment_trigger_price is None

        # Re-running ``_ensure_engine`` (fresh Journal against the same file) is a no-op.
        journal2 = Journal(db_path=db_path)
        try:
            again = await journal2.recent_trades()
            assert len(again) == 1
        finally:
            await journal2.close()
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_open_trade_handles_null_runner_target_price(tmp_path: Path) -> None:
    """Phase 4i: ``runner_target_price=None`` writes 0.0 sentinel (column is NOT NULL)."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        no_runner = replace(
            _position("NRT", parent_order_id=880),
            runner_target_price=None,
        )
        await journal.open_trade(no_runner)
        recent = await journal.recent_trades()
        assert len(recent) == 1
        assert recent[0].target_price == pytest.approx(0.0)
        assert recent[0].scale_out_price == pytest.approx(12.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_update_exit_persists_red_candle_exit_suppressed(tmp_path: Path) -> None:
    """Phase 4i: a scaled-out close persists the suppression flag."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = _position("RCS", parent_order_id=881)
        await journal.open_trade(position)
        closed = replace(
            position,
            status="closed",
            closed_at=_CLOSE_TS,
            scaled_out=True,
            scale_partial_pnl=100.0,
            red_candle_exit_suppressed=True,
            post_scaleout_stop_type="adjustable_to_trail",
            post_scaleout_adjustment_trigger_price=13.0,
        )
        await journal.update_exit(
            closed, exit_price=12.7, pnl=135.0, exit_type="scale_out_then_trail"
        )

        async with (await journal._ensure_engine())[1]() as session:  # noqa: SLF001 - test access
            from sqlalchemy import select

            from bot.persistence.journal import TradeRecord

            row = await session.scalar(
                select(TradeRecord).where(TradeRecord.parent_order_id == 881)
            )
        assert row is not None
        assert row.red_candle_exit_suppressed is True
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_update_exit_leaves_red_candle_exit_suppressed_null_pre_scaleout(
    tmp_path: Path,
) -> None:
    """Phase 4i: a pre-scale stop-out does not touch the suppression column."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = _position("PRE", parent_order_id=882)
        await journal.open_trade(position)
        closed = replace(position, status="closed", closed_at=_CLOSE_TS)
        await journal.update_exit(closed, exit_price=9.0, pnl=-21.0, exit_type="stop_hit")
        async with (await journal._ensure_engine())[1]() as session:  # noqa: SLF001 - test access
            from sqlalchemy import select

            from bot.persistence.journal import TradeRecord

            row = await session.scalar(
                select(TradeRecord).where(TradeRecord.parent_order_id == 882)
            )
        assert row is not None
        assert row.red_candle_exit_suppressed is None
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_recent_trades_returns_reverse_chronological(tmp_path: Path) -> None:
    """Most recently inserted row comes first — the dashboard needs newest-on-top."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(_position("AAA", parent_order_id=1))
        # Tiny gap so timestamps are ordered even on fast runners.
        await asyncio.sleep(0.01)
        await journal.open_trade(_position("BBB", parent_order_id=2))
        await asyncio.sleep(0.01)
        await journal.open_trade(_position("CCC", parent_order_id=3))
        recent = await journal.recent_trades(n=2)
        assert [r.symbol for r in recent] == ["CCC", "BBB"]
    finally:
        await journal.close()


# ---------- Phase 4j entry_order_type column ---------- #


@pytest.mark.asyncio
async def test_journal_persists_entry_order_type(tmp_path: Path) -> None:
    """Phase 4j — ``entry_order_type`` from the position round-trips through the DB."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = replace(
            _position("STPL", parent_order_id=500),
            entry_order_type="STP_LMT",
        )
        await journal.open_trade(position)
        recent = await journal.recent_trades()
        assert recent[0].entry_order_type == "STP_LMT"
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_journal_legacy_position_defaults_entry_order_type(tmp_path: Path) -> None:
    """Callers that leave ``entry_order_type`` at its LMT default persist that value."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(_position("LEG", parent_order_id=600))
        recent = await journal.recent_trades()
        assert recent[0].entry_order_type == "LMT"
    finally:
        await journal.close()


# ---------- Phase 4k commission columns ---------- #


@pytest.mark.asyncio
async def test_add_commission_accumulates_per_leg(tmp_path: Path) -> None:
    """Multiple ``add_commission`` calls sum onto the same column."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(_position("CMM", parent_order_id=700))
        await journal.add_commission(700, leg="entry", amount=1.00)
        await journal.add_commission(700, leg="entry", amount=0.50)
        await journal.add_commission(700, leg="scale", amount=0.75)
        await journal.add_commission(700, leg="exit", amount=1.25)
        recent = await journal.recent_trades()
        assert recent[0].entry_commission == pytest.approx(1.50)
        assert recent[0].scale_commission == pytest.approx(0.75)
        assert recent[0].exit_commission == pytest.approx(1.25)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_add_commission_skips_nonpositive_amounts(tmp_path: Path) -> None:
    """Zero / negative amounts are no-ops so paper's $0 simulated reports don't land."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        await journal.open_trade(_position("ZRO", parent_order_id=701))
        await journal.add_commission(701, leg="entry", amount=0.0)
        await journal.add_commission(701, leg="entry", amount=-0.50)
        recent = await journal.recent_trades()
        assert recent[0].entry_commission is None
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_add_commission_missing_row_is_warning_not_error(tmp_path: Path) -> None:
    """A commission report for an unknown parent_order_id logs but does not raise."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        # Force the engine open so the warning path is exercised against a valid schema.
        await journal.open_trade(_position("XXX", parent_order_id=800))
        await journal.add_commission(99_999, leg="entry", amount=1.0)  # no such parent
        # Original row untouched.
        recent = await journal.recent_trades()
        assert recent[0].entry_commission is None
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_update_exit_does_not_write_commissions(tmp_path: Path) -> None:
    """Commissions are journaled via ``add_commission``, not ``update_exit``."""
    journal = Journal(db_path=tmp_path / "trades.db")
    try:
        position = _position("UPD", parent_order_id=802)
        await journal.open_trade(position)
        # Simulate commissions landing BEFORE update_exit (can happen on fast paths).
        await journal.add_commission(802, leg="entry", amount=1.00)
        from dataclasses import replace as _replace

        closed = _replace(position, status="closed", closed_at=_CLOSE_TS)
        await journal.update_exit(closed, exit_price=11.0, pnl=19.0)
        recent = await journal.recent_trades()
        # update_exit must not overwrite commission columns.
        assert recent[0].entry_commission == pytest.approx(1.00)
        assert recent[0].pnl == pytest.approx(19.0)
    finally:
        await journal.close()


@pytest.mark.asyncio
async def test_phase_4k_migration_idempotent_on_pre_4k_db(tmp_path: Path) -> None:
    """Pre-4k journal DBs get the three commission columns added without clobbering rows."""
    import aiosqlite

    db_path = tmp_path / "trades.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(16) NOT NULL,
                strategy VARCHAR(32) NOT NULL,
                shares INTEGER NOT NULL,
                entry_price FLOAT NOT NULL,
                stop_price FLOAT NOT NULL,
                target_price FLOAT NOT NULL,
                scale_out_price FLOAT,
                runner_target_multiple_used FLOAT,
                exit_price FLOAT,
                pnl FLOAT,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP,
                parent_order_id INTEGER NOT NULL,
                reasons_json TEXT NOT NULL DEFAULT '[]',
                exit_type VARCHAR(32),
                post_scaleout_stop_type VARCHAR(32),
                post_scaleout_adjustment_trigger_price FLOAT,
                red_candle_exit_suppressed BOOLEAN,
                entry_order_type VARCHAR(16)
            )
            """
        )
        await conn.execute(
            "INSERT INTO trades (symbol, strategy, shares, entry_price, stop_price, "
            "target_price, opened_at, parent_order_id, reasons_json) "
            "VALUES ('OLD', 'gap_and_go', 20, 10.0, 9.0, 13.0, "
            "'2026-04-16 09:31:00.000000+00:00', 900, '[]')"
        )
        await conn.commit()

    journal = Journal(db_path=db_path)
    try:
        recent = await journal.recent_trades()
        assert len(recent) == 1
        assert recent[0].entry_commission is None
        # Migration ran, column exists — add_commission should work on legacy row.
        await journal.add_commission(900, leg="exit", amount=0.60)
        recent_after = await journal.recent_trades()
        assert recent_after[0].exit_commission == pytest.approx(0.60)
    finally:
        await journal.close()
