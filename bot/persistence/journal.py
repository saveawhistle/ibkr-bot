"""SQLite trade journal — append-only historical record, not primary state.

The ``PositionStore`` holds the bot's live view of what's open; this module
writes each trade's lifecycle events to ``logs/trades.db`` for post-session
review, tax records, and Phase 4b's performance dashboards. If the journal
diverges from the ``PositionStore`` at any time, the ``PositionStore`` wins.

Why SQLAlchemy: PLAN.md §5 commits to it as the SQLite ORM; the async engine
(``sqlalchemy.ext.asyncio``) backed by ``aiosqlite`` gives us await-friendly
calls without wrestling with ``asyncio.to_thread``. Database file is created
on first use — there is no separate migration step in v1 (``create_all`` runs
idempotently).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from datetime import date as date_cls
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    TypeDecorator,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

if TYPE_CHECKING:
    from bot.execution.position_state import Position

_log = structlog.get_logger("bot.persistence.journal")

_DEFAULT_DB_PATH = Path("logs") / "trades.db"


class _UTCDateTime(TypeDecorator[datetime]):
    """SQLite loses tzinfo on round-trip; this decorator reattaches UTC on read.

    We only ever journal UTC timestamps, so the tzinfo is not ambiguous — but
    tests (and any downstream consumer) compare against tz-aware values, so we
    restore UTC here instead of scattering ``.replace(tzinfo=UTC)`` at callsites.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        """Normalize naive-or-tz-aware input to a UTC datetime before writing."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:  # noqa: ARG002
        """Reattach UTC to values that came back naive from the SQLite layer."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


async def _migrate_add_phase_4e_columns(conn: Any) -> None:
    """Additively add Phase 4e columns to a pre-existing ``trades`` table.

    SQLAlchemy's ``create_all`` is idempotent for missing tables but does
    *not* alter existing ones, so a journal DB written under Phase 4d (no
    ``scale_out_price`` / ``runner_target_multiple_used`` columns) will
    fail reads after the upgrade. Both new columns are nullable so legacy
    rows keep their NULLs; ``ADD COLUMN`` is guarded by an introspection
    check so the migration is idempotent across restarts.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(trades)")
    existing = {row[1] for row in result.fetchall()}
    if "scale_out_price" not in existing:
        await conn.execute(text("ALTER TABLE trades ADD COLUMN scale_out_price FLOAT"))
        _log.info("journal.migrated_column", column="scale_out_price")
    if "runner_target_multiple_used" not in existing:
        await conn.execute(text("ALTER TABLE trades ADD COLUMN runner_target_multiple_used FLOAT"))
        _log.info("journal.migrated_column", column="runner_target_multiple_used")


async def _migrate_add_phase_4h_columns(conn: Any) -> None:
    """Additively add Phase 4h post-scale-out stop columns to ``trades``.

    Mirrors the Phase 4e pattern: both columns are nullable so legacy rows
    (written pre-4h, or written post-4h but pre-scale-out) keep their NULLs.
    ``post_scaleout_stop_type`` is a short enum string
    (``static_breakeven`` | ``adjustable_to_trail``); the trigger price is
    only populated for the adjustable variant. Guarded by a ``PRAGMA
    table_info`` check so restarts skip a no-op re-migration.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(trades)")
    existing = {row[1] for row in result.fetchall()}
    if "post_scaleout_stop_type" not in existing:
        await conn.execute(
            text("ALTER TABLE trades ADD COLUMN post_scaleout_stop_type VARCHAR(32)")
        )
        _log.info("journal.migrated_column", column="post_scaleout_stop_type")
    if "post_scaleout_adjustment_trigger_price" not in existing:
        await conn.execute(
            text("ALTER TABLE trades ADD COLUMN post_scaleout_adjustment_trigger_price FLOAT")
        )
        _log.info("journal.migrated_column", column="post_scaleout_adjustment_trigger_price")


async def _migrate_add_phase_4i_columns(conn: Any) -> None:
    """Additively add the Phase 4i red-candle-suppression flag to ``trades``.

    Nullable boolean (stored as SQLite INTEGER) so legacy rows keep their
    NULLs and read as ``False`` under the Phase 4i in-memory default. Only
    populated for post-scale-out closes that carried the the methodology suppression
    flag; pre-scale exits (stop-hit before the scale-out anchor) leave it
    NULL. Guarded by a ``PRAGMA table_info`` introspection so restarts
    skip a no-op re-migration.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(trades)")
    existing = {row[1] for row in result.fetchall()}
    if "red_candle_exit_suppressed" not in existing:
        await conn.execute(text("ALTER TABLE trades ADD COLUMN red_candle_exit_suppressed BOOLEAN"))
        _log.info("journal.migrated_column", column="red_candle_exit_suppressed")


async def _migrate_add_phase_4j_columns(conn: Any) -> None:
    """Additively add the Phase 4j entry-order-type marker to ``trades``.

    Legacy rows (pre-4j) are all LMT parent entries — we leave them NULL
    and read-time defaulting happens at the consumer (``None`` → ``"LMT"``).
    Guarded by ``PRAGMA table_info`` so restarts skip the ALTER.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(trades)")
    existing = {row[1] for row in result.fetchall()}
    if "entry_order_type" not in existing:
        await conn.execute(text("ALTER TABLE trades ADD COLUMN entry_order_type VARCHAR(16)"))
        _log.info("journal.migrated_column", column="entry_order_type")


async def _migrate_add_phase_4k_columns(conn: Any) -> None:
    """Additively add the Phase 4k per-leg commission columns to ``trades``.

    Three nullable FLOAT columns — ``entry_commission``, ``scale_commission``,
    ``exit_commission`` — sum per-leg IBKR CommissionReport values. Nullable
    so pre-4k rows stay readable with NULL, and analytics ``COALESCE`` to
    0.0. Guarded by ``PRAGMA table_info`` for idempotent restart.
    """
    result = await conn.exec_driver_sql("PRAGMA table_info(trades)")
    existing = {row[1] for row in result.fetchall()}
    for column in ("entry_commission", "scale_commission", "exit_commission"):
        if column not in existing:
            await conn.execute(text(f"ALTER TABLE trades ADD COLUMN {column} FLOAT"))
            _log.info("journal.migrated_column", column=column)


class _Base(DeclarativeBase):
    """SQLAlchemy declarative base — exists only to anchor ``TradeRecord``."""


class TradeRecord(_Base):
    """One row per placed trade, updated once on exit.

    ``exit_price`` and ``pnl`` are nullable — they remain ``NULL`` until the
    stop or target fires and ``update_exit`` rewrites the row. ``reasons_json``
    is the strategy's emitted reasons list serialised for grepability.

    Phase 4e columns: ``scale_out_price`` + ``runner_target_multiple_used``
    are nullable to preserve read-compatibility with rows written before the
    split landed. On those legacy rows, ``target_price`` is the full runner
    ceiling (historically emitted at 2R by the strategy); new rows carry
    ``scale_out_price`` at +1R and ``target_price`` at the config-driven
    runner ceiling. ``runner_target_multiple_used`` is the value of
    ``execution.runner_target_multiple`` that was live when the trade opened,
    so post-hoc analytics can bucket outcomes by risk regime.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_price: Mapped[float] = mapped_column(Float, nullable=False)
    # Phase 4i: ``target_price`` is the runner-ceiling LMT. When
    # ``execution.runner_target_enabled`` is false (default —
    # "no hard profit ceilings") there is no bracket LMT and we persist
    # 0.0 as the "no runner target" sentinel so the column stays NOT NULL
    # across fresh + legacy DBs. Downstream analytics can detect the
    # sentinel by ``target_price == 0.0`` or by ``post_scaleout_stop_type``.
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    scale_out_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    runner_target_multiple_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(_UTCDateTime(), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(_UTCDateTime(), nullable=True)
    parent_order_id: Mapped[int] = mapped_column(Integer, nullable=False)
    reasons_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    # Phase 4d: classify the exit so ``rebuild_symbol_histories_from_journal`` can
    # restore ``SymbolHistory.last_exit_type`` after a crash-restart. One of
    # ``target_hit``, ``stop_hit``, ``scale_out_then_trail``, ``auto_flatten``;
    # NULL for still-open trades.
    exit_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Phase 4h: how the post-scale-out stop was placed (``static_breakeven`` |
    # ``adjustable_to_trail``) plus the triggerPrice at which the adjustable
    # variant converts STP → TRAIL server-side. Both NULL pre-scale-out and on
    # legacy rows written before Phase 4h landed.
    post_scaleout_stop_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    post_scaleout_adjustment_trigger_price: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    # Phase 4i: red-candle suppression marker on the tail.
    # NULL for pre-scale exits; True once ``mark_scaled`` flipped the flag
    # and the suppression governed the tail's exit path.
    red_candle_exit_suppressed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Phase 4j: parent-entry order type (``LMT`` | ``STP_LMT``). Nullable
    # because pre-4j rows were all LMT — consumers default ``None`` → ``LMT``.
    entry_order_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Phase 4k: per-leg IBKR commission totals. NULL on legacy rows + on
    # rows whose leg never fired (e.g. unfilled scale on a straight stop-out).
    # Analytics should ``COALESCE(col, 0.0)``. Stored as summed account-currency
    # floats (we're USD-only in v1 — no FX conversion).
    entry_commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    scale_commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_commission: Mapped[float | None] = mapped_column(Float, nullable=True)


class Journal:
    """Thin async facade over a single-file SQLite database.

    Callers construct once per process (or per test) and hand it to the
    executor; all operations are awaitable and short-lived. The engine is kept
    open for the journal's lifetime — close it via ``close()`` on shutdown.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Point the journal at ``db_path`` (default ``logs/trades.db``)."""
        self._path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_engine(self) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
        """Lazily create the engine + tables on first use.

        The lock matters: executor fill handlers (parent-fill + child-fill) and
        the main loop can all call into the journal concurrently on the very
        first event. Without serialization two tasks race ``create_all`` and
        SQLite raises ``table already exists``.
        """
        if self._engine is not None and self._session_factory is not None:
            return self._engine, self._session_factory
        async with self._init_lock:
            if self._engine is not None and self._session_factory is not None:
                return self._engine, self._session_factory
            self._path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite+aiosqlite:///{self._path.as_posix()}"
            engine = create_async_engine(url, future=True)
            async with engine.begin() as conn:
                await conn.run_sync(_Base.metadata.create_all)
                await _migrate_add_phase_4e_columns(conn)
                await _migrate_add_phase_4h_columns(conn)
                await _migrate_add_phase_4i_columns(conn)
                await _migrate_add_phase_4j_columns(conn)
                await _migrate_add_phase_4k_columns(conn)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            self._engine = engine
            self._session_factory = factory
            _log.info("journal.opened", path=str(self._path))
            return engine, factory

    async def open_trade(
        self, position: Position, *, runner_target_multiple_used: float | None = None
    ) -> int:
        """Persist a freshly-opened trade; returns the new row id.

        ``runner_target_multiple_used`` is the executor's active
        ``execution.runner_target_multiple`` at open time. It's kept out of
        the ``Position`` dataclass (which doesn't know about config) and
        passed explicitly so the journal can freeze the risk regime alongside
        the trade. ``None`` is accepted for back-compat callers (adoption
        paths, tests) and lands as SQL NULL.
        """
        _, factory = await self._ensure_engine()
        async with factory() as session:
            # Phase 4i: ``runner_target_price`` is None when
            # ``execution.runner_target_enabled`` is false — persist 0.0 as the
            # "no runner target" sentinel to keep the column NOT NULL and
            # legacy-DB compatible.
            target_price_to_write = (
                position.runner_target_price if position.runner_target_price is not None else 0.0
            )
            record = TradeRecord(
                symbol=position.symbol,
                strategy=position.strategy,
                shares=position.shares,
                entry_price=position.avg_price,
                stop_price=position.stop_price,
                target_price=target_price_to_write,
                scale_out_price=position.scale_out_price,
                runner_target_multiple_used=runner_target_multiple_used,
                opened_at=position.opened_at,
                parent_order_id=position.parent_order_id,
                reasons_json=json.dumps(position.reasons),
                entry_order_type=position.entry_order_type,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            _log.info(
                "journal.trade_opened",
                trade_id=record.id,
                symbol=record.symbol,
                strategy=record.strategy,
                shares=record.shares,
                runner_target_multiple_used=runner_target_multiple_used,
            )
            return record.id

    async def update_exit(
        self,
        position: Position,
        *,
        exit_price: float,
        pnl: float,
        exit_type: str | None = None,
    ) -> None:
        """Write the exit leg onto the existing row for this parent order.

        Looked up by ``parent_order_id`` rather than position object identity —
        safer across process restarts where the in-memory ``Position`` may
        have been reconstructed from IBKR. ``exit_type`` is a Phase 4d
        classification (``target_hit`` / ``stop_hit`` / ``scale_out_then_trail``
        / ``auto_flatten``) persisted so ``rebuild_symbol_histories_from_journal``
        can restore the re-entry gate state.
        """
        _, factory = await self._ensure_engine()
        async with factory() as session:
            record = await session.scalar(
                select(TradeRecord).where(TradeRecord.parent_order_id == position.parent_order_id)
            )
            if record is None:
                _log.warning(
                    "journal.update_exit_missing_row",
                    symbol=position.symbol,
                    parent_order_id=position.parent_order_id,
                )
                return
            record.exit_price = exit_price
            record.pnl = pnl
            record.closed_at = position.closed_at
            if exit_type is not None:
                record.exit_type = exit_type
            # Phase 4h: persist the post-scale-out stop variant + (optional)
            # adjustable-trigger price if the position carried them. Pre-scale
            # exits (stop-hit before +1R) leave both NULL.
            if position.post_scaleout_stop_type is not None:
                record.post_scaleout_stop_type = position.post_scaleout_stop_type
            if position.post_scaleout_adjustment_trigger_price is not None:
                record.post_scaleout_adjustment_trigger_price = (
                    position.post_scaleout_adjustment_trigger_price
                )
            # Phase 4i: persist suppression marker only on scaled-out closes;
            # pre-scale exits leave the column NULL.
            if position.scaled_out:
                record.red_candle_exit_suppressed = position.red_candle_exit_suppressed
            # NB: Phase 4k commissions are NOT written here — IBKR fires the
            # commissionReport *after* the fill that triggers update_exit, so
            # they're journaled via ``add_commission`` on each report instead.
            await session.commit()
            _log.info(
                "journal.trade_closed",
                trade_id=record.id,
                symbol=record.symbol,
                pnl=round(pnl, 2),
                exit_type=exit_type,
            )

    async def add_commission(
        self,
        parent_order_id: int,
        *,
        leg: Literal["entry", "scale", "exit"],
        amount: float,
    ) -> None:
        """Phase 4k — additively bump the per-leg commission column on a trade row.

        IBKR's ``commissionReportEvent`` fires after each fill, sometimes
        after the fill that closed the position and triggered ``update_exit``.
        We bypass the Position dataclass and write directly to the journal so
        late-arriving reports still land on the correct trade. Looked up by
        ``parent_order_id`` — the only identifier that's stable across fills,
        scale-outs, and reconciled restarts. Silently no-ops on negative or
        zero amounts so paper accounts with $0 simulated commissions don't
        litter the DB with zero-value updates.
        """
        if amount <= 0.0:
            return
        column = f"{leg}_commission"
        _, factory = await self._ensure_engine()
        # Atomic UPDATE — avoids the read-modify-write race when two fills
        # fire commission reports back-to-back (both reading 0.0 before
        # either commits). ``COALESCE`` makes the add additive on legacy NULL rows.
        async with factory() as session:
            result = await session.execute(
                text(
                    f"UPDATE trades SET {column} = COALESCE({column}, 0.0) + :amt "
                    "WHERE parent_order_id = :pid"
                ),
                {"amt": amount, "pid": parent_order_id},
            )
            rowcount = getattr(result, "rowcount", 0)
            if rowcount == 0:
                _log.warning(
                    "journal.commission_missing_row",
                    parent_order_id=parent_order_id,
                    leg=leg,
                    amount=amount,
                )
                return
            await session.commit()

    async def recent_trades(self, n: int = 20) -> list[TradeRecord]:
        """Return the ``n`` most recently opened trades (reverse-chronological)."""
        _, factory = await self._ensure_engine()
        async with factory() as session:
            result = await session.scalars(
                select(TradeRecord).order_by(TradeRecord.id.desc()).limit(n)
            )
            return list(result.all())

    async def trades_for_session(
        self,
        session_date: date_cls,
        timezone: str = "America/New_York",
    ) -> list[TradeRecord]:
        """Return today's rows in chronological order — feeds the Phase 4d rebuild.

        ``opened_at`` is stored as UTC, so the filter converts each row's
        timestamp to the configured session timezone and compares calendar
        dates. Returns oldest-first so ``rebuild_symbol_histories_from_journal``
        sees the same ordering as the live session would have produced.
        Filtered in Python rather than SQL because SQLite lacks a native
        timezone-aware date predicate.
        """
        _, factory = await self._ensure_engine()
        async with factory() as session:
            # Upper bound is deliberately loose — one session will not exceed
            # ``max_trades_per_day`` (default 5) by any margin, so 500 is ample.
            result = await session.scalars(
                select(TradeRecord).order_by(TradeRecord.id.asc()).limit(500)
            )
            tz = ZoneInfo(timezone)
            return [
                row for row in result.all() if row.opened_at.astimezone(tz).date() == session_date
            ]

    async def close(self) -> None:
        """Dispose of the engine; safe to call multiple times."""
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        _log.info("journal.closed", path=str(self._path))
