"""In-memory position state — the executor's source of truth.

The ``PositionStore`` is authoritative for "does the bot have a position on
``X``?". The SQLite journal (``bot.persistence.journal``) is write-only historical
persistence; on crash/restart the executor calls ``reconcile()`` against IBKR
(the authority for "does the account have a position?") and writes the
reconciled state into a fresh ``PositionStore``.

Status machine — one-way, no reversals:

    pending_entry_trigger → pending_entry → open → closing → closed

Any attempt to move backwards raises ``InvalidPositionTransitionError``. The
``pending_entry_trigger`` state (Phase 4j) is the BUY STP-LMT
resting state: the parent order sits on IBKR's servers waiting for price to
tick through ``signal.entry``. Once IBKR triggers + fills, the state machine
advances either directly to ``open`` (full fill) or through ``pending_entry``
first (partial-fill window).

There is no ``cancelled`` terminal state — if a STP-LMT parent never
triggers (e.g. 15:55 auto-flatten fires first), the state machine
transitions to ``closed`` with ``closing_reason="entry_never_triggered"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Final, Literal, Protocol

import structlog

_log = structlog.get_logger("bot.execution.position_state")

PositionStatus = Literal["pending_entry_trigger", "pending_entry", "open", "closing", "closed"]

# Phase 4d re-entry exit classification. ``auto_flatten`` is terminal for the
# session — RiskEngine.check_reentry refuses a new entry on a symbol whose
# most recent exit was the 15:55 forced flatten (trading day is ending).
ExitType = Literal[
    "target_hit",
    "stop_hit",
    "scale_out_then_trail",
    "auto_flatten",
    # Phase 7.8 — pre-scale red-candle close exit (the "first red candle"
    # rule). Classified distinctly so re-entry gating and journaling can
    # tell the difference between a stop-hit and a bar-close discretionary
    # exit. The outcome may be profitable or negative depending on where
    # the red bar closed relative to the entry fill.
    "pre_scale_red_candle",
    # Phase 11 — exit triggered by an external advisor implementation
    # via the exit-advisor hook (``hook_acts=true`` mode). Distinct from
    # all bot-internal exit types so the journal can attribute outcomes
    # to advisor decisions vs the bot's own rule set during analysis.
    "advisor_exit",
]

# Phase 4h / 6.14 — how the post-scale-out stop was placed.
# ``static_breakeven``: Phase 4e fallback (flat STP at entry).
# ``adjustable_to_trail``: Phase 4h STP-that-converts-to-TRAIL at scale_out+1R.
# ``immediate_trail``: Phase 6.14 immediate TRAIL order (no conversion wait).
PostScaleoutStopType = Literal["static_breakeven", "adjustable_to_trail", "immediate_trail"]

_ACTIVE_STATUSES: Final[frozenset[PositionStatus]] = frozenset(
    {"pending_entry_trigger", "pending_entry", "open", "closing"}
)
_VALID_TRANSITIONS: Final[dict[PositionStatus, frozenset[PositionStatus]]] = {
    "pending_entry_trigger": frozenset({"pending_entry", "open", "closing", "closed"}),
    "pending_entry": frozenset({"open", "closing", "closed"}),
    "open": frozenset({"closing", "closed"}),
    "closing": frozenset({"closed"}),
    "closed": frozenset(),
}


class InvalidPositionTransitionError(RuntimeError):
    """Raised when a status transition violates the one-way state machine."""


class UnknownPositionError(KeyError):
    """Raised when a mutation references a symbol with no active position."""


@dataclass(frozen=True)
class Position:
    """A single in-flight bracket trade tracked by the executor.

    ``shares`` is the *parent* quantity; until fill it's the requested size and
    after fill it's the executed size (they can differ on partial fills, which
    Phase 4a does not split — a partial stays at ``pending_entry`` until the
    remainder comes in).

    ``avg_price`` is 0.0 until the parent fills; thereafter it is the execution
    weighted-average price supplied by IBKR on the fill event.

    ``realized_pnl`` is 0.0 until the position closes via stop or target; it is
    a signed dollar amount (positive on winners, negative on losers).
    """

    symbol: str
    strategy: str
    shares: int
    avg_price: float
    stop_price: float
    scale_out_price: float
    runner_target_price: float | None
    parent_order_id: int
    stop_order_id: int
    target_order_id: int
    opened_at: datetime
    status: PositionStatus = "pending_entry"
    closing_reason: str | None = None
    closed_at: datetime | None = None
    exit_price: float = 0.0
    realized_pnl: float = 0.0
    reasons: list[str] = field(default_factory=list)
    adopted_from_reconcile: bool = False
    scaled_out: bool = False
    scale_partial_pnl: float = 0.0
    # Phase 4h / 6.14 — post-scale-out stop bookkeeping. ``None`` pre-
    # scale-out; set by ``mark_scaled`` to one of
    # {``static_breakeven``, ``adjustable_to_trail``, ``immediate_trail``}
    # per the ``execution.post_scaleout_stop_mode`` config.
    # ``post_scaleout_adjustment_trigger_price`` is the IBKR triggerPrice at
    # which the adjustable STP converts to a TRAIL (None for
    # ``static_breakeven`` and ``immediate_trail`` — the latter is
    # already a TRAIL, no conversion needed).
    post_scaleout_stop_type: PostScaleoutStopType | None = None
    post_scaleout_adjustment_trigger_price: float | None = None
    # Phase 4i — the methodology: *"If I've already sold 1/2, I'll hold through red candles
    # as long as my breakeven stop doesn't hit."* Set to True by ``mark_scaled``
    # so the trade manager's bar-close loop stops treating a single red bar as
    # an exit trigger on the surviving runner half. Extension-bar + EMA-break
    # still fire — they signal structural breakdown, not a one-bar shakeout.
    red_candle_exit_suppressed: bool = False
    # Phase 4j — which parent-entry order type IBKR is (or was) working for
    # this position. ``LMT`` matches the Phase 4i marketable-limit behavior
    # (atomic bracket); ``STP_LMT`` is the default where IBKR
    # triggers server-side on the breakout tick and the bot plants
    # protection children only after the parent fills.
    entry_order_type: str = "LMT"
    # Phase 4j — the intended breakout-trigger price. Equals ``signal.entry``
    # on both the LMT and STP-LMT paths (the LMT limit price + the STP-LMT
    # stop trigger). Distinct from ``avg_price`` (0.0 pre-fill, actual fill
    # post-fill) so status surfaces can show the plan while the parent is
    # still resting. 0.0 on reconciled-orphan adoptions that predate 4j.
    entry_trigger_price: float = 0.0
    # Phase 4k — per-leg commission accumulators. IBKR fires
    # ``commissionReportEvent`` on each Trade after its fillEvent; the
    # executor sums them into these three buckets so a scaled-out trade
    # journals three distinct commission legs (entry, scale, exit).
    # Values are in account currency (USD) and always non-negative.
    entry_commission: float = 0.0
    scale_commission: float = 0.0
    exit_commission: float = 0.0
    # Phase 6.5 — bar timestamp of the signal that placed this entry. The
    # orchestrator's per-symbol loop calls ``Executor.expire_unfilled_entry``
    # on every iteration; once the latest-bar timestamp advances strictly
    # past this value AND the parent has zero fills, the resting entry is
    # cancelled. ``None`` on legacy/reconciled positions where the bar
    # timestamp wasn't recorded — those skip the auto-expire path.
    placement_bar_ts: datetime | None = None


@dataclass
class SymbolHistory:
    """Per-symbol-per-session re-entry bookkeeping (Phase 4d).

    Distinct from ``Position`` — a symbol's ``SymbolHistory`` outlives any one
    Position and tracks how many brackets have been opened on it today plus the
    classification of the most recent exit. RiskEngine.check_reentry reads this
    to decide whether a second / third signal on the same symbol is allowed
    and at what size multiplier.

    Reset at session start (09:30 ET) by ``PositionStore.reset_symbol_histories``
    and rebuilt from the journal on crash-restart via
    ``rebuild_symbol_histories_from_journal``.
    """

    symbol: str
    entries_count: int = 0
    last_exit_time: datetime | None = None
    last_exit_pnl: float | None = None
    last_exit_type: ExitType | None = None

    def record_entry(self) -> None:
        """Bump the entries counter — called on each bracket open."""
        self.entries_count += 1

    def record_exit(
        self,
        *,
        exit_time: datetime,
        pnl: float,
        exit_type: ExitType,
    ) -> None:
        """Overwrite the ``last_exit_*`` trio; ``entries_count`` is unaffected.

        For ``scale_out_then_trail``, callers must pass the *total* realized
        PnL (scale-out fill plus trailing close) so the profitable-prior-exit
        gate sees the correct sign.
        """
        self.last_exit_time = exit_time
        self.last_exit_pnl = pnl
        self.last_exit_type = exit_type


class _TradeLike(Protocol):
    """Duck-typed view of ``bot.persistence.journal.TradeRecord`` used by rebuild.

    Declared here rather than importing TradeRecord so PositionStore stays
    free of the journal dependency. Callers (the Executor) know the real
    type and pass it in directly.
    """

    symbol: str
    opened_at: datetime
    closed_at: datetime | None
    pnl: float | None
    exit_type: str | None


class PositionStore:
    """In-memory registry of active + recently-closed positions, keyed by symbol.

    At most one ``active`` position per symbol at a time — the executor's
    ``has_active`` check is the single-position-per-symbol guardrail. Closed
    positions remain queryable via ``get`` until the next ``clear_closed()``
    but do not count toward ``has_active``.
    """

    def __init__(self) -> None:
        """Start empty — state is rebuilt via ``reconcile`` on startup."""
        self._positions: dict[str, Position] = {}
        # Phase 4d: per-symbol re-entry state. Survives position closes within
        # a session; cleared by ``reset_symbol_histories`` at 09:30 ET.
        self._histories: dict[str, SymbolHistory] = {}

    def open_position(
        self,
        *,
        symbol: str,
        strategy: str,
        shares: int,
        stop_price: float,
        scale_out_price: float,
        runner_target_price: float | None,
        parent_order_id: int,
        stop_order_id: int,
        target_order_id: int,
        opened_at: datetime,
        reasons: list[str] | None = None,
        status: PositionStatus = "pending_entry",
        entry_order_type: str = "LMT",
        entry_trigger_price: float = 0.0,
        placement_bar_ts: datetime | None = None,
    ) -> Position:
        """Record a pending-entry position; raise if one is already active for ``symbol``.

        ``status`` defaults to ``pending_entry`` (Phase 4i LMT path). The
        Phase 4j STP-LMT path passes ``pending_entry_trigger`` and
        ``entry_order_type="STP_LMT"`` so the state machine knows the
        parent is resting on IBKR's servers waiting on the breakout tick.

        Phase 6.5 ``placement_bar_ts`` is the timestamp of the signal's
        triggering bar; the executor's auto-expire path compares this to
        the latest closed bar to cancel entries that didn't fill within
        their breakout bar's price action.
        """
        if self.has_active(symbol):
            raise InvalidPositionTransitionError(
                f"Cannot open position for {symbol!r}: already active."
            )
        position = Position(
            symbol=symbol,
            strategy=strategy,
            shares=shares,
            avg_price=0.0,
            stop_price=stop_price,
            scale_out_price=scale_out_price,
            runner_target_price=runner_target_price,
            parent_order_id=parent_order_id,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            opened_at=opened_at,
            status=status,
            reasons=list(reasons or []),
            entry_order_type=entry_order_type,
            entry_trigger_price=entry_trigger_price,
            placement_bar_ts=placement_bar_ts,
        )
        self._positions[symbol] = position
        _log.info(
            "position.opened",
            symbol=symbol,
            strategy=strategy,
            shares=shares,
            stop=stop_price,
            scale_out=scale_out_price,
            runner_target=runner_target_price,
            parent_order_id=parent_order_id,
        )
        return position

    def mark_filled(self, symbol: str, *, fill_price: float, filled_shares: int) -> Position:
        """Transition ``pending_entry`` / ``pending_entry_trigger`` → ``open`` on parent fill."""
        position = self._require(symbol)
        new_status: PositionStatus = "open"
        self._check_transition(position.status, new_status, symbol)
        updated = replace(
            position,
            status=new_status,
            avg_price=fill_price,
            shares=filled_shares,
        )
        self._positions[symbol] = updated
        _log.info(
            "position.filled",
            symbol=symbol,
            fill_price=fill_price,
            filled_shares=filled_shares,
        )
        return updated

    def mark_entry_never_triggered(self, symbol: str, *, closed_at: datetime) -> Position:
        """Close a pending-entry position that never filled (any pending status).

        Phase 4j-original: fired when 15:55 auto-flatten (or a manual
        flatten) cancels a resting BUY STP-LMT parent before IBKR ever
        triggered it. Phase 6.5 widens the accepted source statuses to
        include ``pending_entry`` so the LMT-bracket path can also call
        this when the orchestrator's auto-expire cancels an unfilled
        parent on the next-bar boundary.

        Both source statuses transition straight to ``closed`` with
        realized PnL of zero and ``closing_reason="entry_never_triggered"``
        so downstream analytics can distinguish a never-filled order from
        actual winners/losers regardless of which entry order type was used.
        """
        position = self._require(symbol)
        if position.status not in {"pending_entry_trigger", "pending_entry"}:
            raise InvalidPositionTransitionError(
                f"Cannot mark {symbol!r} as entry-never-triggered: "
                f"status must be 'pending_entry_trigger' or 'pending_entry', "
                f"got {position.status!r}"
            )
        updated = replace(
            position,
            status="closed",
            closing_reason="entry_never_triggered",
            closed_at=closed_at,
            exit_price=0.0,
            realized_pnl=0.0,
        )
        self._positions[symbol] = updated
        _log.info("position.entry_never_triggered", symbol=symbol)
        return updated

    def mark_closing(self, symbol: str, *, reason: str) -> Position:
        """Flag the position as in-flight closing (stop or target trigger received)."""
        position = self._require(symbol)
        new_status: PositionStatus = "closing"
        self._check_transition(position.status, new_status, symbol)
        updated = replace(position, status=new_status, closing_reason=reason)
        self._positions[symbol] = updated
        _log.info("position.closing", symbol=symbol, reason=reason)
        return updated

    def mark_closed(
        self,
        symbol: str,
        *,
        exit_price: float,
        pnl: float,
        closed_at: datetime,
    ) -> Position:
        """Terminal transition → ``closed``. Records exit price and realized PnL."""
        position = self._require(symbol)
        new_status: PositionStatus = "closed"
        self._check_transition(position.status, new_status, symbol)
        updated = replace(
            position,
            status=new_status,
            exit_price=exit_price,
            realized_pnl=pnl,
            closed_at=closed_at,
        )
        self._positions[symbol] = updated
        _log.info(
            "position.closed",
            symbol=symbol,
            exit_price=exit_price,
            pnl=round(pnl, 2),
            reason=updated.closing_reason,
        )
        # Phase 11 — fire the exit-advisor close notification at the
        # single canonical "position closed" transition. The hook
        # function short-circuits when ``exit_advisor.enabled=false``
        # (production main default) so this is a no-op in normal
        # operation. Local import keeps position_state.py free of an
        # unconditional dependency on the exit_advisor package at
        # module import time.
        from bot.exit_advisor.hook.registry import notify_position_closed

        notify_position_closed(updated, pnl)
        return updated

    def mark_scaled(
        self,
        symbol: str,
        *,
        remaining_shares: int,
        scale_partial_pnl: float,
        new_stop_price: float,
        new_stop_order_id: int,
        post_scaleout_stop_type: PostScaleoutStopType | None = None,
        post_scaleout_adjustment_trigger_price: float | None = None,
        red_candle_exit_suppressed: bool = True,
    ) -> Position:
        """Record a Phase 4b first-target scale-out on an ``open`` position.

        Mutates three interlinked fields at once:

        * ``shares`` → remaining shares after the partial sell (not the
          original entry size); subsequent fill-handler PnL math uses this.
        * ``stop_price`` / ``stop_order_id`` → the new breakeven stop (sized
          for the remaining shares, armed at the entry price).
        * ``scaled_out`` + ``scale_partial_pnl`` → bookkeeping for the
          already-banked half so the final close can add it to the tail PnL.

        Phase 4h adds ``post_scaleout_stop_type`` +
        ``post_scaleout_adjustment_trigger_price`` so the position records
        whether the new stop is a plain static breakeven STP or an
        adjustable STP that server-side converts to a TRAIL once price tags
        the recorded trigger. Both are optional for back-compat with callers
        (legacy tests, adoption paths) that predate Phase 4h.

        Stays in ``open`` status — the trailing-exit logic in
        ``TradeManager`` handles the eventual transition to ``closed``.
        """
        position = self._require(symbol)
        if position.status != "open":
            raise InvalidPositionTransitionError(
                f"Cannot scale out {symbol!r}: status must be 'open', got {position.status!r}"
            )
        if position.scaled_out:
            raise InvalidPositionTransitionError(f"Cannot scale out {symbol!r}: already scaled out")
        updated = replace(
            position,
            shares=remaining_shares,
            stop_price=new_stop_price,
            stop_order_id=new_stop_order_id,
            scaled_out=True,
            scale_partial_pnl=scale_partial_pnl,
            post_scaleout_stop_type=post_scaleout_stop_type,
            post_scaleout_adjustment_trigger_price=post_scaleout_adjustment_trigger_price,
            red_candle_exit_suppressed=red_candle_exit_suppressed,
        )
        self._positions[symbol] = updated
        _log.info(
            "position.scaled_out",
            symbol=symbol,
            remaining_shares=remaining_shares,
            scale_partial_pnl=round(scale_partial_pnl, 2),
            new_stop_price=new_stop_price,
            post_scaleout_stop_type=post_scaleout_stop_type,
            post_scaleout_adjustment_trigger_price=post_scaleout_adjustment_trigger_price,
            red_candle_exit_suppressed=red_candle_exit_suppressed,
        )
        return updated

    def add_entry_commission(self, symbol: str, amount: float) -> Position:
        """Phase 4k — accumulate a commission report onto the parent-entry leg."""
        return self._add_commission(symbol, amount, leg="entry")

    def add_scale_commission(self, symbol: str, amount: float) -> Position:
        """Phase 4k — accumulate commission onto the +1R scale-out market-SELL leg."""
        return self._add_commission(symbol, amount, leg="scale")

    def add_exit_commission(self, symbol: str, amount: float) -> Position:
        """Phase 4k — accumulate commission on stop/target/trailing close legs."""
        return self._add_commission(symbol, amount, leg="exit")

    def _add_commission(
        self, symbol: str, amount: float, *, leg: Literal["entry", "scale", "exit"]
    ) -> Position:
        """Shared additive bump; rejects negative amounts (commissions are costs, never negative)."""
        if amount < 0.0:
            raise ValueError(f"Commission must be non-negative, got {amount!r} for {symbol!r}")
        position = self._require(symbol)
        if leg == "entry":
            updated = replace(position, entry_commission=position.entry_commission + amount)
        elif leg == "scale":
            updated = replace(position, scale_commission=position.scale_commission + amount)
        else:
            updated = replace(position, exit_commission=position.exit_commission + amount)
        self._positions[symbol] = updated
        return updated

    def update_fill_anchored_prices(
        self,
        symbol: str,
        *,
        new_stop_price: float,
        new_scale_out_price: float,
        new_stop_order_id: int,
    ) -> Position:
        """Phase 8.3 — replace ``stop_price`` + ``scale_out_price`` with fill-anchored values.

        After a parent fill, executor cancels the loose signal-anchored STP
        and plants a fill-anchored OCA pair (new STP + scale_lmt). This
        method atomically updates both prices on the Position so downstream
        readers (bar-close scale-out backup, tick-driven scale-out backup,
        Phase 7.6 trail trigger derivations from Position state) see the
        same values that are now resting on the exchange. ``new_stop_order_id``
        replaces the cancelled STP's id.

        Status guard: must be ``open`` (parent already filled). Raises
        ``InvalidPositionTransitionError`` otherwise so callers don't
        accidentally re-anchor a pending or closed position.
        """
        position = self._require(symbol)
        if position.status != "open":
            raise InvalidPositionTransitionError(
                f"update_fill_anchored_prices requires status=open "
                f"(symbol={symbol}, status={position.status})"
            )
        updated = replace(
            position,
            stop_price=new_stop_price,
            scale_out_price=new_scale_out_price,
            stop_order_id=new_stop_order_id,
        )
        self._positions[symbol] = updated
        return updated

    def attach_protection_children(
        self,
        symbol: str,
        *,
        stop_order_id: int,
        target_order_id: int,
        runner_target_price: float | None = None,
    ) -> Position:
        """Phase 4j — persist protection-child order IDs onto an ``open`` position.

        The STP-LMT path places the parent alone first (state
        ``pending_entry_trigger``); when the parent fills the executor
        transitions to ``open`` and plants the stop + optional runner
        LMT. Those children's order IDs are unknown at ``open_position``
        time — this method writes them onto the Position so reconcile +
        CLI status surfaces see the right numbers. ``runner_target_price``
        is only overwritten when caller passes a non-None value so a
        disabled runner (``target_order_id=0``) leaves the original None.
        """
        position = self._require(symbol)
        updated = replace(
            position,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            runner_target_price=(
                runner_target_price
                if runner_target_price is not None
                else position.runner_target_price
            ),
        )
        self._positions[symbol] = updated
        return updated

    def has_active(self, symbol: str) -> bool:
        """True iff an active (pending/open/closing) position exists for ``symbol``."""
        position = self._positions.get(symbol)
        return position is not None and position.status in _ACTIVE_STATUSES

    def get_active(self, symbol: str) -> Position | None:
        """Return the active position for ``symbol`` or None."""
        position = self._positions.get(symbol)
        if position is None or position.status not in _ACTIVE_STATUSES:
            return None
        return position

    def get(self, symbol: str) -> Position | None:
        """Return the current record for ``symbol`` regardless of status (or None)."""
        return self._positions.get(symbol)

    def list_active(self) -> list[Position]:
        """Return all active positions in insertion order."""
        return [p for p in self._positions.values() if p.status in _ACTIVE_STATUSES]

    def insert_reconciled(self, position: Position) -> None:
        """Directly insert a ``Position`` built from IBKR's authoritative snapshot.

        The normal path is ``open_position`` → ``mark_filled`` → ``mark_closed``.
        ``reconcile()`` on the executor bypasses that because the position was
        opened by a prior process; we just want to record what IBKR already
        knows. Overwrites any existing record for the symbol.
        """
        self._positions[position.symbol] = position

    def clear_closed(self) -> None:
        """Evict terminal records so the store doesn't grow unboundedly across sessions."""
        self._positions = {
            symbol: position
            for symbol, position in self._positions.items()
            if position.status != "closed"
        }

    # ---------- Phase 4d: symbol history ---------- #

    def symbol_history(self, symbol: str) -> SymbolHistory:
        """Return the session's history for ``symbol``, creating a blank one on first touch.

        ``entries_count == 0`` + no ``last_exit_*`` means "first signal on this
        symbol today"; the risk engine maps that to multiplier ``size_multipliers[0]``.
        """
        history = self._histories.get(symbol)
        if history is None:
            history = SymbolHistory(symbol=symbol)
            self._histories[symbol] = history
        return history

    def reset_symbol_histories(self) -> None:
        """Drop every per-symbol history — called at session start (09:30 ET)."""
        self._histories.clear()

    def list_symbol_histories(self) -> list[SymbolHistory]:
        """Return histories in deterministic (insertion) order — used by CLI status."""
        return list(self._histories.values())

    def rebuild_symbol_histories_from_journal(
        self,
        trades: Iterable[_TradeLike],
    ) -> None:
        """Reconstruct ``_histories`` from today's journal rows after a crash-restart.

        Caller must pre-filter the iterable to trades belonging to the current
        session (``opened_at`` on-or-after 09:30 ET today). Ordering within a
        symbol matters: later-opened trades overwrite ``last_exit_*``, so pass
        rows in chronological order (ascending ``opened_at``).

        Each input row bumps ``entries_count``. Rows with a populated
        ``closed_at`` additionally update ``last_exit_time`` / ``last_exit_pnl``
        / ``last_exit_type``. Rows with an unrecognised ``exit_type`` are
        counted as an entry but skip the exit-metadata update — the risk gate
        will then treat the symbol as "no known prior exit" and fall through to
        the default multiplier (conservative).
        """
        self._histories.clear()
        valid_types: frozenset[str] = frozenset(
            {
                "target_hit",
                "stop_hit",
                "scale_out_then_trail",
                "auto_flatten",
                "pre_scale_red_candle",
                # Phase 11 — advisor-driven full exits via the exit-advisor
                # hook. Recognised here so journal-replay on crash-restart
                # restores SymbolHistory.last_exit_type accurately.
                "advisor_exit",
            }
        )
        for record in trades:
            history = self.symbol_history(record.symbol)
            history.record_entry()
            if record.closed_at is None or record.pnl is None:
                continue
            exit_type = record.exit_type
            if exit_type not in valid_types:
                continue
            # exit_type is a runtime str here; narrow for the Literal by cast-via-match.
            # Each value of ``valid_types`` above must have an explicit branch here
            # — the ``else`` is reserved for ``"auto_flatten"`` only. Phase 7.8
            # ``pre_scale_red_candle`` was added to ``valid_types`` and ``ExitType``
            # but missed this chain initially, so the fallthrough silently
            # downgraded it to ``auto_flatten`` (which the risk engine treats as
            # a session-ending terminal classification, blocking re-entries
            # post-restart that the live path would have allowed). When adding
            # a new ``ExitType`` member, update both the frozenset above AND
            # this chain in lockstep.
            typed_exit: ExitType
            if exit_type == "target_hit":
                typed_exit = "target_hit"
            elif exit_type == "stop_hit":
                typed_exit = "stop_hit"
            elif exit_type == "scale_out_then_trail":
                typed_exit = "scale_out_then_trail"
            elif exit_type == "pre_scale_red_candle":
                typed_exit = "pre_scale_red_candle"
            elif exit_type == "advisor_exit":
                typed_exit = "advisor_exit"
            else:
                typed_exit = "auto_flatten"
            history.record_exit(
                exit_time=record.closed_at,
                pnl=float(record.pnl),
                exit_type=typed_exit,
            )
        _log.info(
            "position_store.histories_rebuilt",
            symbols=len(self._histories),
        )

    def _require(self, symbol: str) -> Position:
        """Fetch a record or raise ``UnknownPositionError``."""
        position = self._positions.get(symbol)
        if position is None:
            raise UnknownPositionError(symbol)
        return position

    @staticmethod
    def _check_transition(current: PositionStatus, target: PositionStatus, symbol: str) -> None:
        """Enforce the one-way state machine; closed is terminal."""
        if target not in _VALID_TRANSITIONS[current]:
            raise InvalidPositionTransitionError(
                f"Invalid transition for {symbol!r}: {current} → {target}"
            )
