"""Converts ``Signal`` objects into IBKR bracket orders and tracks fills.

Phase 4b scope:

* Place bracket (parent LMT + stop-loss STP + profit-taker LMT) on signal,
  sized by ``RiskEngine.check_entry`` ($ max-loss rule).
* Track pending → open → closed transitions in ``PositionStore``.
* Route exit PnL into ``RiskEngine.on_fill_closed`` so daily-loss,
  profit-goal, and give-back halts can trip.
* Write the trade to the SQLite journal on entry fill and exit fill.
* Reconcile on startup so a crash-restart can't double-open a symbol.

The 4a ``require_paper_mode`` hard gate has moved to the CLI (the ``trade``
command refuses to enable live without an interactive CONFIRM). The
executor itself is mode-agnostic; whichever mode the config says to use is
what it drives.

Reconcile filtering asymmetry: ``reqAllOpenOrdersAsync()`` results are
filtered by ``clientId`` so we only touch orders this bot placed —
a user's manual TWS orders must never be cancelled or adopted. No such
filter applies to ``reqPositionsAsync()``, because positions are
account-level at IBKR: they carry no ``clientId`` attribution (a share
lot is owned by the account regardless of which client opened it). We
therefore adopt every unknown account position, but only manage orders
tagged with our own client id.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from typing import TYPE_CHECKING, Any, Literal

import structlog
from ib_async import (
    Contract,
    LimitOrder,
    MarketOrder,
    Order,
    StopLimitOrder,
    StopOrder,
    Trade,
)

from bot.brokerage.ibkr_client import ref_req_id
from bot.config import Settings, get_settings
from bot.execution.position_state import (
    ExitType,
    InvalidPositionTransitionError,
    Position,
    PositionStatus,
    PositionStore,
    PostScaleoutStopType,
    UnknownPositionError,
)
from bot.persistence.journal import Journal
from bot.risk import Approved, Rejected, RiskEngine

if TYPE_CHECKING:
    from bot.brokerage.ibkr_client import IBKRClient
    from bot.notify import Notifier
    from bot.strategies.base import Signal

_log = structlog.get_logger("bot.execution.executor")

# ib_async encodes "unset" float-valued Order attributes with ``sys.float_info.max``.
# Used by the reconcile path to distinguish an adjustable STP (triggerPrice set)
# from a plain STP (triggerPrice still at the sentinel).
_IBKR_UNSET_FLOAT_SENTINEL = 1.7976931348623157e308

# Phase 10.3 — explicit Time-In-Force on every order. ``ib_async.Order.tif``
# defaults to the empty string, so without an explicit value TWS applies
# its account preset and rewrites it to ``"DAY"`` via a Cancelled→Resubmitted
# cycle. Day-7 paper trading (2026-04-30 BIYA) showed ~400 ms of broker-side
# round-trip per entry from this preset cancel + resubmit (Error 10349:
# "Order TIF was set to DAY based on order preset"). Setting tif explicitly
# at construction skips the cycle entirely. ``"DAY"`` matches the existing
# behaviour we were getting via the preset; if a future leg needs ``"IOC"``
# or ``"GTC"`` (e.g. an end-of-day rebalancing stop), set it locally on
# that order rather than threading a config knob through every site.
_DEFAULT_ORDER_TIF = "DAY"

# Used by ``expire_unfilled_entry`` to compare bar timestamps against the
# gap-and-go window_end (specified in ET regardless of the bar's own tz).
_NY_TZ = ZoneInfo("America/New_York")


def apply_default_tif(order: Order) -> Order:
    """Phase 10.3 — assign the default TIF in place; return the order for chaining.

    Used immediately after every ``LimitOrder`` / ``StopOrder`` /
    ``StopLimitOrder`` / ``MarketOrder`` / ``Order`` construction so the
    placement leg ships with TIF already set. See ``_DEFAULT_ORDER_TIF``
    for the rationale.
    """
    order.tif = _DEFAULT_ORDER_TIF
    return order


def _round_to_tick(price: float) -> float:
    """Round ``price`` to the US-equity minimum tick per Reg NMS Rule 612.

    * Price ≥ $1.00 → whole pennies ($0.01). Sub-penny submissions on
      >= $1 stocks are rejected by IBKR with Error 110 *"The price does
      not conform to the minimum price variation for this contract"*.
    * Price < $1.00 → $0.0001 ticks (NMS Rule 612 sub-penny exception for
      quotes below $1).

    Trail *amounts* and *deltas* (not absolute prices) inherit the same
    rule here — IBKR rejects a trailing distance like $0.045 at prices
    ≥ $1.00 with the same error. Exchange-layer concern, applied at
    order-construction time so strategies can emit theoretical values
    unchanged.
    """
    if price >= 1.0:
        return round(price, 2)
    return round(price, 4)


@dataclass(frozen=True)
class _LmtBufferBreakdown:
    """Phase 10.6 — every input and output of the LMT buffer clamp chain.

    Surfaced into ``executor.lmt_bracket_placed`` so an operator can grep
    which clamp bound, what the floor and ceiling values were at the
    evaluation, and what raw % buffer the strategy would have produced
    without any clamps.
    """

    final: float
    pct_raw: float
    floor_value: float
    ceiling_value: float
    clamp: str  # "floor" | "ceiling" | "none"


def _compute_lmt_buffer_breakdown(
    entry_price: float,
    buffer_pct: float,
    buffer_floor_usd: float,
    buffer_cap_usd: float,
    max_pct: float,
    anchor_price: float | None = None,
) -> _LmtBufferBreakdown:
    """Phase 8.2 + 10.6 + 12.5 — full clamp resolution for the LMT entry buffer.

    Pipeline:

    * ``pct_raw = entry × buffer_pct / 100`` — the raw percentage buffer.
    * ``floor_value = buffer_floor_usd`` — penny-stock spread floor.
    * ``ceiling_value = min(buffer_cap_usd, ANCHOR × max_pct / 100)`` —
      the binding upper bound. Combines the legacy fixed-dollar
      slippage cap (Phase 8.2) with the percentage cap (Phase 10.6).
      Phase 12.5: the percentage cap is anchored on
      ``min(entry, anchor_price)`` rather than ``entry`` alone --
      when the strategy's entry sits well above the most recent
      market quote (breakout bar's close > prior offer), the original
      entry-anchored ceiling could still produce LMTs above IBKR's
      ~9.78%-above-current-market threshold and trigger Error 202.
      ``anchor_price=None`` falls back to the legacy entry-only
      ceiling (preserves pre-12.5 numerics for tests).
    * Apply: ``buffer = min(max(pct_raw, floor_value), ceiling_value)``.

    Clamp determination (priority order — ceiling wins ties since it's
    the binding *safety* constraint when both floor and ceiling could
    apply):

    * "ceiling" — buffer was lowered by the ceiling.
    * "floor"   — floor raised the raw % above ``pct_raw`` and ceiling
      did not bind.
    * "none"    — raw % already sat between floor and ceiling.
    """
    pct_raw = entry_price * (buffer_pct / 100.0)
    floor_value = buffer_floor_usd
    if anchor_price is not None and anchor_price > 0 and anchor_price < entry_price:
        ceiling_anchor = anchor_price
    else:
        ceiling_anchor = entry_price
    pct_ceiling = ceiling_anchor * (max_pct / 100.0)
    ceiling_value = min(buffer_cap_usd, pct_ceiling)

    pre_ceiling = max(floor_value, pct_raw)
    if pre_ceiling > ceiling_value:
        final = ceiling_value
        clamp = "ceiling"
    elif floor_value > pct_raw:
        final = floor_value
        clamp = "floor"
    else:
        final = pct_raw
        clamp = "none"

    return _LmtBufferBreakdown(
        final=final,
        pct_raw=pct_raw,
        floor_value=floor_value,
        ceiling_value=ceiling_value,
        clamp=clamp,
    )


def _compute_lmt_buffer(
    entry_price: float,
    buffer_pct: float,
    buffer_floor_usd: float,
    buffer_cap_usd: float,
    max_pct: float = 100.0,
) -> float:
    """Phase 8.2 + 10.6 — scalar wrapper around :func:`_compute_lmt_buffer_breakdown`.

    Returns just the clamped buffer for callers that don't need the
    full breakdown. ``max_pct`` defaults to 100% (effectively disabling
    the percentage ceiling) so existing call sites and tests written
    pre-Phase-10.6 retain identical numerics.
    """
    return _compute_lmt_buffer_breakdown(
        entry_price=entry_price,
        buffer_pct=buffer_pct,
        buffer_floor_usd=buffer_floor_usd,
        buffer_cap_usd=buffer_cap_usd,
        max_pct=max_pct,
    ).final


# Phase 12.5 — IBKR Error 202 message regex. Sample:
#   "Order Canceled - reason:We cannot accept an order at a limit price at or
#    more aggressive than 1.6046172. Please submit your order using a limit
#    price that is closer to the current market price of 1.4614. ..."
# We extract the FIRST decimal -- that's IBKR's accepted ceiling.
_AGGRESSIVE_LIMIT_PRICE_PATTERN = re.compile(
    r"limit price at or more aggressive than\s+([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)


def _parse_aggressive_limit_ceiling(error_string: str) -> float | None:
    """Extract IBKR's accepted-ceiling price from an Error 202 message, or None.

    Returns ``None`` when the message doesn't match the expected pattern
    (e.g. a different 202 reason, or IBKR changed the wording in a future
    TWS release). Callers treat ``None`` as "can't recover, give up".
    """
    if not error_string:
        return None
    match = _AGGRESSIVE_LIMIT_PRICE_PATTERN.search(error_string)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


@dataclass
class _LmtRetryContext:
    """Phase 12.5 — captured per-placement state for Error 202 retry.

    Stored in ``Executor._lmt_retry_contexts`` keyed on parent_order_id at
    placement time. The ``ib.errorEvent`` handler looks up by the cancelled
    order's reqId; when found AND ``retried`` is False, it computes a new
    LMT from the IBKR-supplied ceiling and re-submits the bracket once.

    ``retried`` flips True before the re-submission so a second 202 on the
    new bracket can't trigger another retry (avoids an infinite loop if
    market keeps moving against us between submissions).
    """

    symbol: str
    contract: Contract
    entry: float
    stop: float
    target: float | None
    shares: int
    strategy: str
    original_limit_price: float
    market_anchor_price: float | None
    placement_bar_ts: datetime
    retried: bool = False


@dataclass
class _BracketTrades:
    """Container for the Trade objects that make up a bracket.

    Fields are populated conditionally by entry type:

    * ``parent`` — the BUY order (LMT / STP-LMT / MKT). Always present
      once the bracket is placed.
    * ``stop`` — full-size SELL STP at ``signal.stop`` (initial
      protection). Present on LMT path (atomic), STP-LMT (post-fill),
      and MKT (atomic, Phase 6.14).
    * ``target`` — full-size SELL LMT at ``runner_target_price``.
      Present only when ``runner_target_enabled: true`` and entry type
      is LMT or STP-LMT.
    * ``scale_lmt`` — half-size SELL LMT at ``scale_out_price``. Phase
      6.14: present only on MKT entries (atomic). Other paths scale
      out via TradeManager's bar-close poll.

    After a crash-restart, ``reconcile`` may adopt a partially-filled
    bracket where e.g. the parent is already done and only the stop
    child remains — any field may then be ``None``. Fill-handler wiring
    + cancel helpers all treat ``None`` as a no-op.
    """

    parent: Trade | None
    stop: Trade | None
    target: Trade | None
    scale_lmt: Trade | None = None


class Executor:
    """Owns bracket placement, fill handling, and position reconciliation.

    One instance per session. The executor does not subscribe to the signal
    bus itself — the orchestrator drains the bus and calls
    ``handle_signal`` per winner. That separation keeps the executor
    testable without spinning up a bus.
    """

    def __init__(
        self,
        *,
        ibkr: IBKRClient,
        position_store: PositionStore,
        journal: Journal,
        risk_engine: RiskEngine,
        notifier: Notifier | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire dependencies; ``notifier=None`` silently skips Telegram fills."""
        self._ibkr = ibkr
        self._store = position_store
        self._journal = journal
        self._risk_engine = risk_engine
        self._notifier = notifier
        self._settings = settings or get_settings()
        # Keep Trade refs alive so eventkit callbacks are not garbage-collected.
        self._active_trades: dict[str, _BracketTrades] = {}
        # Pending fill-handler tasks; tracked so tests (and ``flatten``) can
        # deterministically wait for them to complete instead of racing the
        # aiosqlite commit thread with ``asyncio.sleep(0)``.
        self._pending_fill_tasks: set[asyncio.Task[None]] = set()
        # Phase 12.5 — per-parent-order retry context for IBKR Error 202
        # (aggressive-LMT cancel) recovery. Keyed on parent_order_id of the
        # initial entry-LMT placement. Populated on placement, consumed by
        # the ``ib.errorEvent`` handler when 202 fires for a tracked order,
        # mutated on retry (``retried=True`` blocks a second pass for the
        # same order chain).
        self._lmt_retry_contexts: dict[int, _LmtRetryContext] = {}
        # Subscribe to IBKR's error event stream so the handler runs inline
        # when 202 fires, NOT after the next bar tick (which would be too
        # late: the breakout window is ~1 minute and the strategy may not
        # re-emit on the next bar). Defensive ``getattr``: synthetic test
        # fakes don't carry an ``errorEvent`` attribute, and we don't want
        # the executor to refuse construction in those cases. The Phase
        # 12.5 retry tests inject their own dispatch path.
        if self._settings.execution.lmt_aggressive_limit_retry:
            error_event = getattr(self._ibkr.ib, "errorEvent", None)
            if error_event is not None:
                error_event += self._on_ibkr_error_event

    @property
    def risk_engine(self) -> RiskEngine:
        """Expose the risk engine for the orchestrator + CLI status surface."""
        return self._risk_engine

    @property
    def store(self) -> PositionStore:
        """Expose the PositionStore so the orchestrator can sync TradeManager tracking."""
        return self._store

    @property
    def journal(self) -> Journal:
        """Expose the journal so the orchestrator can run the Phase 4d session rebuild."""
        return self._journal

    @property
    def active_trades(self) -> dict[str, _BracketTrades]:
        """Expose live bracket trades so TradeManager can cancel/replace legs on scale-out."""
        return self._active_trades

    @property
    def notifier(self) -> Notifier | None:
        """Expose the optional Telegram notifier so TradeManager can push scale-out fills."""
        return self._notifier

    def cancel_trade_silently(self, trade: Trade | None) -> None:
        """Public facade over the internal cancel — used by TradeManager."""
        self._cancel_trade_silently(trade)

    def place_adjustable_post_scaleout_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
        initial_risk: float,
    ) -> tuple[Trade, Trade | None, float]:
        """Public facade — used by TradeManager at scale-out time.

        Returns ``(stp_trade, target_trade, trigger_price)`` so the caller
        can rebuild ``active_trades`` and persist the trigger price onto
        the position. ``target_trade`` is ``None`` when
        ``execution.runner_target_enabled`` is false (Phase 4i default).
        """
        return self._place_adjustable_post_scaleout_stop(
            contract=contract,
            position=position,
            remaining_shares=remaining_shares,
            initial_risk=initial_risk,
        )

    def place_static_breakeven_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
    ) -> Trade:
        """Public facade — Phase 4e fallback used when trail toggle is off.

        Places a plain STP at ``position.avg_price`` for ``remaining_shares``.
        No OCA, no adjustment attributes; TradeManager's bar-close loop drives
        the trailing exit path from here.
        """
        return self._place_static_breakeven_stop(
            contract=contract, position=position, remaining_shares=remaining_shares
        )

    def place_immediate_trail_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
        initial_risk: float,
    ) -> Trade:
        """Phase 6.14 — public facade for the immediate TRAIL post-scale-out stop.

        Places an IBKR TRAIL SELL at scale-out time with
        ``trailingAmount = trail_amount_r_multiple × initial_risk``, and the
        starting stop position pre-seeded to ``scale_out_price - trail_amount``
        so the tail locks in at least ~+1R profit the moment scale-out fires.
        No conversion trigger — the TRAIL is live immediately.
        """
        return self._place_immediate_trail_stop(
            contract=contract,
            position=position,
            remaining_shares=remaining_shares,
            initial_risk=initial_risk,
        )

    async def handle_signal(self, signal: Signal) -> None:
        """Route a single Signal through the risk gate, sizing, and bracket placement.

        Sizing + halt + PDT-advisory + margin checks live in ``RiskEngine``;
        the executor only acts on ``Approved``. On ``Rejected``, no orders
        are placed — the rejection has already been logged by the engine
        with ``risk.entry_rejected`` + structured gate context.
        """
        symbol = signal.symbol

        if self._store.has_active(symbol):
            existing = self._store.get_active(symbol)
            _log.info(
                "signal.superseded_open_position",
                symbol=symbol,
                incoming_strategy=signal.strategy,
                existing_strategy=existing.strategy if existing else None,
                existing_status=existing.status if existing else None,
            )
            return

        try:
            account_summary = await self._ibkr.account_summary()
        except Exception as exc:  # noqa: BLE001 - IB can raise many shapes
            _log.error("executor.account_summary_failed", symbol=symbol, error=str(exc))
            return

        decision = await self._risk_engine.check_entry(signal, self._store, account_summary)
        if isinstance(decision, Rejected):
            detail = {k: v for k, v in decision.detail.items() if k != "symbol"}
            _log.info(
                "signal.rejected",
                symbol=symbol,
                strategy=signal.strategy,
                stage="risk_gate",
                reason=decision.reason,
                **detail,
            )
            return
        assert isinstance(decision, Approved)  # noqa: S101 - narrowing for mypy
        shares = decision.shares

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001 - qualify can raise many shapes
            _log.error("executor.qualify_failed", symbol=symbol, error=str(exc))
            return

        # Phase 4i: runner target is optional. The methodology doesn't plant
        # hard profit ceilings on the runner, so when
        # ``execution.runner_target_enabled`` is false we place a 2-leg
        # bracket (parent + stop) and let trade_manager's bar-close loop
        # drive the tail exit. The scale-out price from the signal is still
        # the R:R anchor and the executor's scale-out trigger.
        initial_risk = signal.entry - signal.stop
        runner_multiple = self._settings.execution.runner_target_multiple
        runner_enabled = self._settings.execution.runner_target_enabled
        runner_target: float | None = (
            round(signal.entry + initial_risk * runner_multiple, 4) if runner_enabled else None
        )

        # Prefer the signal's own order-type override (e.g. gap-and-go standing
        # STP_LMT order placed before price breaks the trigger) over the global
        # config default. ``None`` falls back to ``execution.entry_order_type``.
        entry_order_type = signal.preferred_order_type or self._settings.execution.entry_order_type

        if entry_order_type == "LMT":
            # Phase 4i path — atomic 2/3-leg bracket placed in one shot.
            bracket = self._place_bracket(
                contract=contract,
                entry=signal.entry,
                stop=signal.stop,
                target=runner_target,
                shares=shares,
                # Phase 12.5 — pass the strategy's market anchor through to
                # the buffer-ceiling calculation. None falls back to entry.
                market_anchor_price=signal.market_anchor_price,
            )
            assert bracket.parent is not None
            assert bracket.stop is not None
            if runner_enabled:
                assert bracket.target is not None
            open_status: PositionStatus = "pending_entry"
            parent_order_id = bracket.parent.order.orderId
            stop_order_id = bracket.stop.order.orderId
            target_order_id = bracket.target.order.orderId if bracket.target is not None else 0
            # Phase 12.5 — record the retry context so the ``ib.errorEvent``
            # handler can recover from an Error 202 cancel without losing
            # the trade. Only registered when the retry is enabled in
            # config (otherwise the handler isn't subscribed).
            if self._settings.execution.lmt_aggressive_limit_retry:
                self._lmt_retry_contexts[parent_order_id] = _LmtRetryContext(
                    symbol=symbol,
                    contract=contract,
                    entry=signal.entry,
                    stop=signal.stop,
                    target=runner_target,
                    shares=shares,
                    strategy=signal.strategy,
                    # ``lmtPrice`` is typed as ``float | Decimal | None`` upstream
                    # but always concrete on the LMT path we just placed.
                    original_limit_price=float(bracket.parent.order.lmtPrice or 0.0),
                    market_anchor_price=signal.market_anchor_price,
                    placement_bar_ts=signal.timestamp,
                )
        elif entry_order_type == "MKT":
            # Phase 6.14.1 path — atomic 2-leg [parent MKT + full-size
            # STP]. The 50%-share scale-out LMT is planted post-fill
            # via ``_handle_parent_fill`` because IBKR auto-normalizes
            # bracket-child quantities to match the parent (live AKAN
            # test confirmed 83-share LMT got rewritten to 166 on the
            # wire). No LMT_runner — tail is managed post-scale-out
            # via the immediate TRAIL stop per ``post_scaleout_stop_mode``.
            bracket = self._place_mkt_bracket(
                contract=contract,
                entry=signal.entry,
                stop=signal.stop,
                shares=shares,
            )
            assert bracket.parent is not None
            assert bracket.stop is not None
            open_status = "pending_entry"
            parent_order_id = bracket.parent.order.orderId
            stop_order_id = bracket.stop.order.orderId
            target_order_id = 0  # no runner LMT on the MKT atomic bracket
        else:
            # Phase 4j path (``STP_LMT``) — parent BUY STP-LMT transmitted
            # alone; children planted post-fill in ``_handle_parent_fill``.
            # The reason to keep the parent-alone + children-on-fill
            # pattern for STP_LMT: IBKR's server-side adjustable stop
            # can auto-convert OCA children when the parent converts,
            # and atomic placement with children attached can race the
            # trigger tick.
            # Pass the resolved ``entry_order_type`` explicitly so the builder
            # uses STP_LMT even when the global config says ``LMT`` (e.g.
            # gap-and-go standing orders with ``signal.preferred_order_type``).
            parent_order = self._build_parent_entry_order(
                signal=signal, shares=shares, order_type=entry_order_type
            )
            parent_trade = self._ibkr.ib.placeOrder(contract, parent_order)
            bracket = _BracketTrades(parent=parent_trade, stop=None, target=None)
            open_status = "pending_entry_trigger"
            parent_order_id = parent_trade.order.orderId
            stop_order_id = 0
            target_order_id = 0

        try:
            position = self._store.open_position(
                symbol=symbol,
                strategy=signal.strategy,
                shares=shares,
                stop_price=signal.stop,
                scale_out_price=signal.scale_out_price,
                runner_target_price=runner_target,
                parent_order_id=parent_order_id,
                stop_order_id=stop_order_id,
                target_order_id=target_order_id,
                opened_at=_now_utc(),
                reasons=list(signal.reasons),
                status=open_status,
                entry_order_type=entry_order_type,
                entry_trigger_price=signal.entry,
                # Phase 6.5 — anchor the auto-expire timer to the bar that
                # produced this signal. The orchestrator compares the next
                # closed-bar timestamp against this value to decide whether
                # the breakout's price action ended without a fill.
                placement_bar_ts=signal.timestamp,
            )
        except InvalidPositionTransitionError as exc:
            # Race: another signal slipped between has_active and open_position.
            # Unlikely in a single-task orchestrator, but we must not double-open.
            _log.error("executor.open_position_race", symbol=symbol, error=str(exc))
            self._cancel_trade_silently(bracket.parent)
            self._cancel_trade_silently(bracket.stop)
            self._cancel_trade_silently(bracket.target)
            return

        self._active_trades[symbol] = bracket
        self._wire_fill_handlers(symbol, bracket)

        # Phase 4d: bump the symbol's re-entry counter. This happens on bracket
        # placement (not on fill) so a pending-entry that never fills still
        # counts — otherwise the ``max_trades_per_day`` guard would double-serve
        # as a re-entry cap, and a rejected fill would quietly reset the budget.
        history = self._store.symbol_history(symbol)
        history.record_entry()

        _log.info(
            "order.placed",
            symbol=symbol,
            strategy=signal.strategy,
            shares=shares,
            entry=signal.entry,
            stop=signal.stop,
            scale_out=signal.scale_out_price,
            runner_target=runner_target,
            runner_target_enabled=runner_enabled,
            runner_target_multiple=runner_multiple,
            parent_order_id=position.parent_order_id,
            stop_order_id=position.stop_order_id,
            target_order_id=position.target_order_id,
            rth_only=self._settings.execution.rth_only,
            entries_count=history.entries_count,
            entry_order_type=entry_order_type,
            status=open_status,
        )

    async def flatten_symbol(self, symbol: str, *, reason: str) -> None:
        """Cancel open bracket legs for ``symbol`` and market-close any live shares.

        Intended for the manual ``flatten`` CLI and Phase 4b's risk kill
        switches. Cancellation is idempotent — already-filled / already-dead
        orders are no-ops.
        """
        position = self._store.get_active(symbol)
        if position is None:
            _log.info("flatten.no_position", symbol=symbol, reason=reason)
            return

        bracket = self._active_trades.get(symbol)
        if bracket is not None:
            self._cancel_trade_silently(bracket.parent)
            self._cancel_trade_silently(bracket.stop)
            self._cancel_trade_silently(bracket.target)

        # Phase 4j: ``pending_entry_trigger`` → the STP-LMT parent is resting
        # on IBKR's servers and never filled. No shares exist, so no market
        # close is needed; just cancel the parent and transition the record.
        # The scheduler-driven path gets a dedicated log so operators can
        # distinguish "parent cancelled at session end" from "filled bracket
        # flattened" in the audit trail.
        if position.status == "pending_entry_trigger":
            if reason == "session_auto_flatten":
                _log.info(
                    "auto_flatten.cancelled_pending_entry",
                    symbol=symbol,
                    parent_order_id=position.parent_order_id,
                    entry_order_type=position.entry_order_type,
                    entry_trigger_price=position.entry_trigger_price,
                )
            else:
                _log.info(
                    "flatten.cancelled_pending_entry",
                    symbol=symbol,
                    parent_order_id=position.parent_order_id,
                    entry_trigger_price=position.entry_trigger_price,
                    reason=reason,
                )
            with contextlib.suppress(InvalidPositionTransitionError, UnknownPositionError):
                self._store.mark_entry_never_triggered(symbol, closed_at=_now_utc())
            self._active_trades.pop(symbol, None)
            return

        # If the parent filled, we own shares → send a market close.
        if position.status in {"open", "closing"} and position.shares > 0:
            try:
                contract = await self._ibkr.qualify_stock(symbol)
            except Exception as exc:  # noqa: BLE001
                _log.error("flatten.qualify_failed", symbol=symbol, error=str(exc))
                return
            close_order = MarketOrder("SELL", position.shares)
            close_order.outsideRth = not self._settings.execution.rth_only
            apply_default_tif(close_order)
            self._ibkr.ib.placeOrder(contract, close_order)
            _log.warning(
                "flatten.market_close_sent",
                symbol=symbol,
                shares=position.shares,
                reason=reason,
            )

        # Already terminal — nothing to do.
        closed_at = _now_utc()
        with contextlib.suppress(InvalidPositionTransitionError):
            self._store.mark_closed(
                symbol,
                exit_price=position.avg_price,
                pnl=0.0,  # TODO(Phase 4b): attribute PnL from market-close fill
                closed_at=closed_at,
            )
        # Phase 10.3 — flatten frees margin; refresh the cache so the next
        # entry signal sees the restored ``AvailableFunds``.
        self._ibkr.invalidate_account_summary_cache()
        self._active_trades.pop(symbol, None)

        # Phase 4d: record ``auto_flatten`` only for the session-end scheduler
        # firing — manual flattens are one-off operator actions and should
        # not permanently block re-entry for the rest of the day.
        if reason == "session_auto_flatten":
            history = self._store.symbol_history(symbol)
            history.record_exit(
                exit_time=closed_at,
                pnl=0.0,
                exit_type="auto_flatten",
            )
            # Persist the classification so rebuild after a restart recognises
            # the terminal block. Best-effort — a journal failure must not
            # abort the flatten path.
            closed_position = self._store.get(symbol)
            if closed_position is not None:
                try:
                    await self._journal.update_exit(
                        closed_position,
                        exit_price=position.avg_price,
                        pnl=0.0,
                        exit_type="auto_flatten",
                    )
                except Exception as exc:  # noqa: BLE001 - journal is observational
                    _log.error(
                        "executor.journal_auto_flatten_failed",
                        symbol=symbol,
                        error=str(exc),
                    )

    def expire_unfilled_entry(self, symbol: str, current_bar_ts: datetime) -> bool:
        """Phase 6.5 — cancel a resting entry order whose breakout bar is gone.

        Called by the orchestrator on each new closed bar for ``symbol``.
        Cancels the resting parent (and any bracket children placed
        atomically with it) when:

        1. The position is still ``pending_entry`` or ``pending_entry_trigger``.
        2. ``placement_bar_ts`` is recorded (skips legacy/reconciled rows).
        3. Normal path (LMT/MKT): ``current_bar_ts > placement_bar_ts`` —
           strict inequality. The same-bar evaluation must not cancel; only
           T+1's close onward.
        4. Standing STP_LMT path (gap-and-go strategy): the order sits resting
           until the gap-and-go window closes at 10:00 ET rather than
           expiring after just one bar. The order is cancelled when
           ``current_bar_ts.time() >= gap_and_go.window_end``. The
           session-end auto-flatten handles any orders that survive to 15:55.
        5. The parent has zero filled shares. Any partial fill exempts the
           order — partial positions are valid under existing risk
           management and ``status == 'open'`` would already exclude them,
           but we re-check the trade fills as belt-and-suspenders against
           a race between IBKR's fill event and the store transition.

        Bracket children are explicitly cancelled too. IBKR's ``parentId``
        linkage *should* cascade-cancel them when the parent is cancelled,
        but cancelling them ourselves is idempotent and hardens against
        the rare case where a child's status hasn't propagated yet. STP-LMT
        parents have no pre-fill children, so the extra cancels are no-ops
        on that path.

        Returns True iff a cancel was actually issued. The orchestrator
        doesn't act on the return value today; it's exposed for tests.
        """
        position = self._store.get_active(symbol)
        if position is None:
            return False
        if position.status not in {"pending_entry", "pending_entry_trigger"}:
            return False
        if position.placement_bar_ts is None:
            return False

        # Standing STP_LMT gap-and-go orders live until the strategy window
        # closes (default 10:00 ET), not just one bar. Always protect the same
        # bar (same-bar call must never cancel) then check the window clock.
        # ``current_bar_ts`` may be UTC or NY-local; we convert to NY for
        # comparison with ``window_end`` which is specified in ET.
        if position.entry_order_type == "STP_LMT" and position.strategy == "gap_and_go":
            if current_bar_ts <= position.placement_bar_ts:
                return False  # same-bar protection (matches non-STP_LMT behaviour)
            gng_window_end = self._settings.strategies.gap_and_go.window_end
            h, m = int(gng_window_end.split(":")[0]), int(gng_window_end.split(":")[1])
            window_minutes = h * 60 + m
            if current_bar_ts.tzinfo is not None:
                ny_ts = current_bar_ts.astimezone(_NY_TZ)
            else:
                ny_ts = current_bar_ts  # assume caller already localised
            bar_minutes = ny_ts.hour * 60 + ny_ts.minute
            if bar_minutes < window_minutes:
                return False  # still within the gap-and-go window, keep alive
            # At or past window_end — fall through to cancel below.
        elif current_bar_ts <= position.placement_bar_ts:
            return False

        bracket = self._active_trades.get(symbol)
        # Belt-and-suspenders fill check: if any shares filled between the
        # last status sync and this call, the position will transition to
        # ``open`` momentarily and we must not race that by cancelling.
        if bracket is not None and bracket.parent is not None:
            _, filled_shares = _extract_fill(bracket.parent)
            if filled_shares > 0:
                return False

        parent_already_done = (
            bracket is not None and bracket.parent is not None and bracket.parent.isDone()
        )
        # Cancel the bracket — parent first (which on IBKR's side cancels
        # the children atomically when transmit chains are intact), then
        # explicit child cancels as belt-and-suspenders for the LMT path
        # where stop+target sit on IBKR's books pre-fill.
        if bracket is not None:
            self._cancel_trade_silently(bracket.parent)
            self._cancel_trade_silently(bracket.stop)
            self._cancel_trade_silently(bracket.target)

        try:
            self._store.mark_entry_never_triggered(symbol, closed_at=_now_utc())
        except (InvalidPositionTransitionError, UnknownPositionError) as exc:
            # Race: the position transitioned out from under us between the
            # status check above and the close call (e.g. a fill landed in
            # the same tick). Log defensively and leave state alone.
            _log.warning(
                "executor.entry_expired_close_failed",
                symbol=symbol,
                error=str(exc),
            )
            self._active_trades.pop(symbol, None)
            return False

        self._active_trades.pop(symbol, None)
        if parent_already_done:
            error_code = _extract_broker_error_code(bracket.parent if bracket else None)
            _log.info(
                "executor.entry_already_cancelled",
                symbol=symbol,
                strategy=position.strategy,
                parent_order_id=position.parent_order_id,
                placement_bar_ts=position.placement_bar_ts.isoformat(),
                current_bar_ts=current_bar_ts.isoformat(),
                entry_order_type=position.entry_order_type,
                error_code=error_code,
            )
            # Phase 9.6: a parent that finished without our explicit cancel
            # came from one of two sources — broker auto-cancel (carries an
            # error code, e.g. 10349 for SCM eligibility) or operator manual
            # cancel in TWS (no error code). We only treat the broker case
            # as a rejection counted toward the per-symbol lockout — operator
            # cancels are an explicit human override, not a structural
            # signal that the symbol is untradeable.
            if error_code is not None:
                asyncio.create_task(
                    self._record_broker_rejection(symbol, error_code=error_code),
                    name=f"broker_rejection:{symbol}",
                )
        else:
            _log.info(
                "executor.entry_expired",
                symbol=symbol,
                strategy=position.strategy,
                parent_order_id=position.parent_order_id,
                placement_bar_ts=position.placement_bar_ts.isoformat(),
                current_bar_ts=current_bar_ts.isoformat(),
                entry_order_type=position.entry_order_type,
                reason="not_filled_in_breakout_bar",
            )
        return True

    def _on_ibkr_error_event(
        self,
        req_id: int,
        error_code: int,
        error_string: str,
        contract: Contract | None = None,
    ) -> None:
        """Phase 12.5 — IBKR error-event listener for the Error 202 retry path.

        Fires synchronously from ib_async's eventkit dispatcher when TWS
        sends an error message. We filter for code 202 with the
        "limit price at or more aggressive" reason; everything else is a
        no-op. Note that 202 carries other reasons (e.g. "Cancelled by
        user") which we MUST NOT treat as a retry trigger -- the regex
        match against the price-suggestion text is the discriminator.

        On match: look up the retry context by ``reqId`` (== parent order
        id at placement time); if not found OR already retried, skip.
        Parse the IBKR-supplied ceiling, compute a new LMT one tick below
        it, re-submit the bracket once, and update the position store
        with the new parent order id so ``expire_unfilled_entry`` keys
        on the right trade.

        Defensive: any exception in the handler is caught and logged.
        ib_async's eventkit will keep firing other listeners; a crash in
        our handler must not poison the broker connection.
        """
        if error_code != 202:
            return
        suggested_ceiling = _parse_aggressive_limit_ceiling(error_string)
        if suggested_ceiling is None:
            # Code 202 with a non-aggressive-limit reason (e.g. operator-
            # cancelled, after-hours rejection). Not our retry case.
            return
        context = self._lmt_retry_contexts.get(req_id)
        if context is None:
            # Error 202 fired for an order we don't track (e.g. a
            # post-scaleout LMT, or a different client's order on a
            # shared session). No-op.
            return
        if context.retried:
            _log.warning(
                "executor.lmt_aggressive_limit_retry_failed",
                symbol=context.symbol,
                strategy=context.strategy,
                parent_order_id=req_id,
                reason="already_retried",
                suggested_ceiling=suggested_ceiling,
            )
            self._lmt_retry_contexts.pop(req_id, None)
            return
        try:
            self._do_lmt_retry(req_id, context, suggested_ceiling, error_string)
        except Exception as exc:  # noqa: BLE001 - retry must never poison the connection
            _log.error(
                "executor.lmt_aggressive_limit_retry_crashed",
                symbol=context.symbol,
                parent_order_id=req_id,
                error=str(exc),
            )

    def _do_lmt_retry(
        self,
        original_parent_id: int,
        context: _LmtRetryContext,
        suggested_ceiling: float,
        error_string: str,
    ) -> None:
        """Phase 12.5 — execute the actual retry: cancel leftovers + re-place.

        Split out from ``_on_ibkr_error_event`` so the try/except wrapper
        stays narrow and the happy path reads cleanly.
        """
        # Compute the corrected LMT: one tick below IBKR's accepted
        # ceiling, but never below ``entry`` (a sub-entry LMT would
        # never fill marketable). Tick rounding handled inside
        # ``_place_bracket`` for the placed price; we round here for
        # the log + the >= entry guard.
        new_lmt = _round_to_tick(suggested_ceiling - 0.01)
        if new_lmt <= context.entry:
            _log.warning(
                "executor.lmt_aggressive_limit_retry_failed",
                symbol=context.symbol,
                strategy=context.strategy,
                parent_order_id=original_parent_id,
                reason="suggested_ceiling_below_entry",
                suggested_ceiling=suggested_ceiling,
                entry=context.entry,
                hint=(
                    "IBKR's market-anchored ceiling sits below the strategy's "
                    "entry; the breakout already faded past the entry price. "
                    "Skip retry -- next bar's strategy decision applies."
                ),
            )
            self._lmt_retry_contexts.pop(original_parent_id, None)
            return

        _log.info(
            "executor.lmt_aggressive_limit_detected",
            symbol=context.symbol,
            strategy=context.strategy,
            original_parent_order_id=original_parent_id,
            original_limit_price=context.original_limit_price,
            suggested_ceiling=suggested_ceiling,
            entry=context.entry,
            error_string=error_string,
        )

        # Mark retried BEFORE the new placement so a fast second 202 on
        # the new bracket can't race us into a third attempt.
        context.retried = True

        # Cancel any leftover trades for this symbol. IBKR has already
        # cancelled the original parent and stop (that's why we got the
        # 202); these are belt-and-suspenders in case the bracket sat
        # in some intermediate state.
        existing_bracket = self._active_trades.get(context.symbol)
        if existing_bracket is not None:
            self._cancel_trade_silently(existing_bracket.parent)
            self._cancel_trade_silently(existing_bracket.stop)
            self._cancel_trade_silently(existing_bracket.target)

        # Re-place the bracket at the corrected LMT. Force-override
        # bypasses the buffer chain so we land exactly at IBKR's ceiling
        # minus one tick.
        new_bracket = self._place_bracket(
            contract=context.contract,
            entry=context.entry,
            stop=context.stop,
            target=context.target,
            shares=context.shares,
            market_anchor_price=context.market_anchor_price,
            force_limit_price=new_lmt,
        )
        # ``parent`` is non-None right after a successful ``_place_bracket``;
        # mypy can't narrow the dataclass field on its own.
        assert new_bracket.parent is not None
        new_parent_id = new_bracket.parent.order.orderId

        # Swap the active-trades pointer to the new bracket and rewire
        # fill handlers. Without this, the post-fill flow would still
        # think the cancelled trade is the live one.
        self._active_trades[context.symbol] = new_bracket
        self._wire_fill_handlers(context.symbol, new_bracket)

        # The position record's ``parent_order_id`` field is observability
        # metadata; ``expire_unfilled_entry`` resolves the live trade via
        # ``self._active_trades.get(symbol)`` which we already swapped
        # above. So we don't need a position-store mutation -- the new
        # bracket is the canonical "live order" lookup. The position's
        # original parent_order_id remains in the journal for forensic
        # correlation with the cancel event.

        # Move the retry context to the new parent_order_id so any
        # subsequent 202 on the new order is recognized as already-
        # retried and rejected with the proper log.
        self._lmt_retry_contexts.pop(original_parent_id, None)
        self._lmt_retry_contexts[new_parent_id] = context

        _log.info(
            "executor.lmt_aggressive_limit_retry",
            symbol=context.symbol,
            strategy=context.strategy,
            original_parent_order_id=original_parent_id,
            new_parent_order_id=new_parent_id,
            original_limit_price=context.original_limit_price,
            new_limit_price=new_lmt,
            suggested_ceiling=suggested_ceiling,
            entry=context.entry,
        )

    async def _record_broker_rejection(self, symbol: str, *, error_code: int | None) -> None:
        """Phase 9.6 — schedule the rejection counter update + drop event.

        Runs in its own task because ``expire_unfilled_entry`` is sync but
        ``on_broker_rejection`` needs the risk engine's async lock. Errors
        here must not propagate — the rejection accounting is observability
        and lockout machinery, not on the order-cancel critical path.
        """
        try:
            just_blocked = await self._risk_engine.on_broker_rejection(
                symbol, error_code=error_code
            )
        except Exception as exc:  # noqa: BLE001 - book-keeping must not crash the bar loop
            _log.error("executor.broker_rejection_record_failed", symbol=symbol, error=str(exc))
            return
        if just_blocked:
            _log.warning(
                "orchestrator.watchlist_symbol_dropped",
                symbol=symbol,
                reason="repeated_broker_rejection",
                rejection_count=self._risk_engine.state.broker_rejection_count.get(symbol, 0),
            )

    async def reconcile(self) -> None:
        """Sync ``PositionStore`` + pending brackets with IBKR's authoritative state.

        Runs once on startup. Two authoritative IBKR sources are combined:

        * ``reqPositionsAsync`` — open share lots (no clientId, account-level).
        * ``reqAllOpenOrdersAsync`` — every open order on the account, filtered
          here to just the orders placed by this bot's ``clientId`` so user
          manual TWS orders are left untouched.

        After filtering, bot-owned open orders are grouped by ``parentId`` to
        reconstruct crash-orphaned brackets. Four buckets of outcomes:

        1. **Full bracket pending** — parent + children all open. Adopt as
           ``pending_entry`` so the fill handler re-activates and a second
           signal for the same symbol is deduped.
        2. **Parent filled, children pending** — children exist but parent
           isn't in the open-orders list (already filled) *and* IBKR reports
           a live share lot. Adopt as ``open`` with the stop/target IDs
           populated.
        3. **Legacy IBKR position, no open orders** — account has shares but
           no orders. Adopt with strategy=``reconciled`` and no children so
           the operator sees "go flatten this by hand".
        4. **Lone orphan order** — bot-owned open order not part of a bracket
           (either a parent with no children, or children with no matching
           IBKR lot). These shouldn't exist in practice; cancel defensively.

        Store-active symbols not reflected in IBKR are marked ``closed``.
        """
        ib = self._ibkr.ib
        try:
            ibkr_positions = await ib.reqPositionsAsync()
            open_trades = await ib.reqAllOpenOrdersAsync()
        except Exception as exc:  # noqa: BLE001 - reconcile must never crash startup
            _log.error("reconcile.request_failed", error=str(exc))
            return

        bot_client_id = self._settings.ibkr.client_id
        bot_trades: list[Trade] = []
        non_bot_count = 0
        for trade in open_trades:
            if getattr(trade.order, "clientId", None) == bot_client_id:
                bot_trades.append(trade)
            else:
                non_bot_count += 1
        if non_bot_count:
            _log.info("reconcile.filtered_non_bot_orders", count=non_bot_count)

        ibkr_by_symbol = {p.contract.symbol: p for p in ibkr_positions if p.position != 0}
        _log.info(
            "reconcile.snapshot",
            ibkr_positions=len(ibkr_by_symbol),
            ibkr_open_orders=len(bot_trades),
            non_bot_open_orders=non_bot_count,
            store_active=len(self._store.list_active()),
        )

        parents_by_id: dict[int, Trade] = {}
        children_by_parent: dict[int, list[Trade]] = {}
        for trade in bot_trades:
            parent_id = int(getattr(trade.order, "parentId", 0) or 0)
            if parent_id == 0:
                parents_by_id[int(trade.order.orderId)] = trade
            else:
                children_by_parent.setdefault(parent_id, []).append(trade)

        handled_symbols: set[str] = set()

        # Bucket 1: parent + children present → pending_entry adoption.
        # Phase 4j bucket: a BUY STP-LMT parent alone (no children yet) is
        # the resting pre-trigger state — adopt as ``pending_entry_trigger``
        # so the scheduler + flatten paths see it and the state machine is
        # consistent across a crash-restart mid-Phase-4j.
        for parent_id, parent_trade in parents_by_id.items():
            symbol = parent_trade.contract.symbol
            children = children_by_parent.pop(parent_id, [])
            if self._store.has_active(symbol):
                handled_symbols.add(symbol)
                continue
            parent_order_type = getattr(parent_trade.order, "orderType", "") or ""
            if not children and parent_order_type == "STP LMT":
                shares = int(getattr(parent_trade.order, "totalQuantity", 0) or 0)
                trigger_price = float(getattr(parent_trade.order, "auxPrice", 0.0) or 0.0)
                limit_price = float(getattr(parent_trade.order, "lmtPrice", 0.0) or 0.0)
                self._adopt_pending_entry_trigger(
                    symbol=symbol,
                    parent_trade=parent_trade,
                    shares=shares,
                    trigger_price=trigger_price,
                    limit_price=limit_price,
                )
                handled_symbols.add(symbol)
                continue
            stop_trade, target_trade = _split_bracket_children(children, parent_trade)
            if stop_trade is None and target_trade is None:
                _log.warning(
                    "reconcile.orphan_single_order",
                    order_id=parent_id,
                    symbol=symbol,
                    order_type=parent_order_type or "?",
                    hint="Bot-owned order not part of a bracket; cancelling.",
                )
                self._cancel_trade_silently(parent_trade)
                continue
            self._adopt_bracket(
                symbol=symbol,
                status="pending_entry",
                shares=int(getattr(parent_trade.order, "totalQuantity", 0) or 0),
                avg_price=0.0,
                stop_trade=stop_trade,
                target_trade=target_trade,
                parent_trade=parent_trade,
            )
            handled_symbols.add(symbol)

        # Bucket 2: parent already filled (children only) + live IBKR lot → open adoption.
        for parent_id, children in list(children_by_parent.items()):
            symbol = children[0].contract.symbol
            if self._store.has_active(symbol):
                handled_symbols.add(symbol)
                continue
            ib_pos = ibkr_by_symbol.get(symbol)
            if ib_pos is None:
                for child in children:
                    _log.warning(
                        "reconcile.orphan_single_order",
                        order_id=int(child.order.orderId),
                        symbol=symbol,
                        order_type=getattr(child.order, "orderType", "?"),
                        hint="Bot-owned child with no matching IBKR lot; cancelling.",
                    )
                    self._cancel_trade_silently(child)
                continue
            stop_trade = next(
                (c for c in children if getattr(c.order, "orderType", None) == "STP"), None
            )
            target_trade = next(
                (c for c in children if getattr(c.order, "orderType", None) == "LMT"), None
            )
            shares = int(abs(ib_pos.position))
            avg_cost = float(ib_pos.avgCost)
            self._adopt_bracket(
                symbol=symbol,
                status="open",
                shares=shares,
                avg_price=avg_cost,
                stop_trade=stop_trade,
                target_trade=target_trade,
                parent_trade=None,
                parent_order_id=parent_id,
            )
            handled_symbols.add(symbol)

        # Bucket 3: IBKR position with no bot-owned orders → legacy-style adoption.
        for symbol, ib_pos in ibkr_by_symbol.items():
            if symbol in handled_symbols or self._store.has_active(symbol):
                continue
            shares = int(abs(ib_pos.position))
            avg_cost = float(ib_pos.avgCost)
            synth = self._synthesize_reconciled_position(
                symbol=symbol, shares=shares, avg_cost=avg_cost
            )
            self._store.insert_reconciled(synth)
            _log.warning(
                "reconcile.ibkr_position_unknown_to_store",
                symbol=symbol,
                shares=shares,
                avg_cost=avg_cost,
                hint="IBKR shows shares but no bot-owned orders; operator should flatten.",
            )

        # Store-active symbols not in IBKR → terminate.
        for position in list(self._store.list_active()):
            if (
                position.symbol in ibkr_by_symbol
                or position.symbol in handled_symbols
                or position.status != "open"
            ):
                continue
            _log.warning(
                "reconcile.store_position_unknown_to_ibkr",
                symbol=position.symbol,
                status=position.status,
                hint="Store will mark closed; IBKR shows no shares.",
            )
            with contextlib.suppress(UnknownPositionError, InvalidPositionTransitionError):
                self._store.mark_closed(
                    position.symbol,
                    exit_price=position.avg_price,
                    pnl=0.0,
                    closed_at=_now_utc(),
                )

    def _build_parent_entry_order(
        self, *, signal: Signal, shares: int, order_type: str | None = None
    ) -> Order:
        """Phase 4j — build the parent entry order per ``execution.entry_order_type``.

        ``LMT`` returns a plain marketable-limit BUY at ``signal.entry``.
        ``STP_LMT`` returns a BUY stop-limit with ``stopPrice = signal.entry``
        (trigger) and ``lmtPrice = signal.entry + entry_limit_buffer_usd``
        (ceiling).

        ``order_type`` overrides the global config when the caller has already
        resolved the effective order type (e.g. ``handle_signal`` after applying
        ``signal.preferred_order_type``). ``None`` falls back to the configured
        ``execution.entry_order_type``.

        Phase 10.3: TIF is applied uniformly at the end via
        ``apply_default_tif`` so all three entry types ship with TIF=DAY
        on the wire. The Phase 4j-original STP-LMT branch had the only
        explicit assignment; the LMT and MKT branches inherited the
        ib_async default of ``""`` and were the source of the 10349
        cancel/resubmit cycle on entry.
        """
        rth_only = self._settings.execution.rth_only
        entry_type = order_type or self._settings.execution.entry_order_type
        # Phase 6.9: round to US-equity tick size. IBKR rejects sub-penny
        # prices on stocks >= $1.00 with Error 110. Applied on both the
        # LMT and the STP-LMT paths (trigger + limit).
        entry_price = _round_to_tick(signal.entry)
        if entry_type == "LMT":
            order: Order = LimitOrder("BUY", shares, entry_price)
        elif entry_type == "MKT":
            # Phase 6.12 — the manual hotkey flow: market buy at signal bar
            # close, no server-side trigger wait, no $0.10 limit buffer
            # to blow through. Slippage is uncapped — relies on the
            # Phase 4c ``max_pct_of_bar_volume`` guardrail to avoid
            # sweeping an illiquid name through multiple price levels.
            order = MarketOrder("BUY", shares)
        else:
            buffer = self._settings.execution.entry_limit_buffer_usd
            limit_price = _round_to_tick(entry_price + buffer)
            order = StopLimitOrder("BUY", shares, limit_price, entry_price)
            # Phase 6.10: fire on EITHER a last print OR a bid/ask
            # excursion past the trigger — whichever lands first. The
            # IBKR default for stocks is ``Last`` (triggerMethod=2);
            # Day-4 paper trading showed a GP breakout where the LAST
            # price clearly printed above the trigger multiple times
            # and the stop never converted. Scoped to the BUY entry
            # stop only: SELL protection stops stay on ``Last`` so a
            # wick below stop doesn't eject us from a position that
            # never really traded there.
            order.triggerMethod = 7
        order.outsideRth = not rth_only
        apply_default_tif(order)
        return order

    def _apply_initial_stop_adjustable_fields(
        self,
        stop_order: Any,
        *,
        entry: float,
        stop_price: float,
    ) -> None:
        """Phase 7.6: encode server-side auto-convert STP → TRAIL at +Nx R.

        When ``initial_stop_adjustable_enabled`` is true, the initial
        protective STP gets four extra fields so IBKR auto-converts it
        to a TRAIL when the market tags ``entry + trigger_R × R``, with
        a ``trail_R × R`` trailing distance. ``adjustableTrailingUnit=0``
        means the trail is in price units (dollars), not percent.

        Zero bot-side code runs at the trigger — the conversion happens
        server-side at exchange latency, even if the bot is briefly
        offline. The OCA group (scale_lmt ↔ stop) is preserved across
        the conversion because the TRAIL is the same orderId as the STP.
        """
        exec_cfg = self._settings.execution
        if not exec_cfg.initial_stop_adjustable_enabled:
            return
        # Phase 7.6 mode gate. ``server_adjustable`` is the original wiring
        # (encode adjustment fields and let IBKR convert at trigger).
        # ``bot_driven`` keeps the initial STP plain — the bot watches the
        # +1R bar-close in TradeManager and cancels/plants the TRAIL itself,
        # avoiding IBKR's FIX-PEGGED substitution observed on 2026-05-05.
        if exec_cfg.initial_stop_trail_mode != "server_adjustable":
            return
        initial_risk = entry - stop_price
        if initial_risk <= 0.0:
            # Defensive: strategies already guard this, but a wider-than-entry
            # stop would produce a nonsensical negative trigger_price. Skip
            # encoding and let the plain STP stand as-is.
            return
        trigger_r = exec_cfg.initial_stop_trigger_r_multiple
        trail_r = exec_cfg.initial_stop_trail_r_multiple
        trigger_price = _round_to_tick(entry + initial_risk * trigger_r)
        trail_amount = _round_to_tick(initial_risk * trail_r)
        adjusted_stop_price = _round_to_tick(trigger_price - trail_amount)
        stop_order.triggerPrice = trigger_price
        stop_order.adjustedOrderType = "TRAIL"
        stop_order.adjustedStopPrice = adjusted_stop_price
        stop_order.adjustedTrailingAmount = trail_amount
        stop_order.adjustableTrailingUnit = 0  # 0 = price units

    def _place_bracket(
        self,
        *,
        contract: Contract,
        entry: float,
        stop: float,
        target: float | None,
        shares: int,
        market_anchor_price: float | None = None,
        force_limit_price: float | None = None,
    ) -> _BracketTrades:
        """Place a 2- or 3-leg bracket with the mandatory ``transmit`` flag sequence.

        The flag sequence is load-bearing — get it wrong and IBKR accepts
        the parent but discards the children, which is a uniquely painful
        way to have a stopless position in production. The *final* leg
        always carries ``transmit=True``:

            parent.transmit       = False  (children not yet registered)
            profit_taker.transmit = False  (only when target is not None)
            stop_loss.transmit    = True   (final leg — sends the bracket)

        Phase 4i: when ``target`` is ``None`` we emit a 2-leg bracket
        (parent + stop). The stop remains the transmit leg.

        Phase 8.2: the parent LMT price is ``signal.entry + buffer`` where
        ``buffer = _compute_lmt_buffer(...)`` (percent of entry, clamped
        to ``[lmt_buffer_usd_floor, lmt_buffer_usd_cap]``). This makes the
        order marketable on a normal ask while capping slippage on halt
        reopens / fast spikes — anything above ``signal.entry + buffer``
        sits unfilled and the Phase 6.5 next-bar auto-cancel sweeps it.
        Children (STP, optional target LMT) and the Phase 7.6 adjustable
        STP fields all anchor to ``signal.entry`` (not the LMT ceiling)
        so the +1R trigger reflects the strategy's intended entry.
        """
        rth_only = self._settings.execution.rth_only
        exec_cfg = self._settings.execution
        # Phase 6.9: tick-round every absolute price before IBKR sees it.
        entry_px = _round_to_tick(entry)
        stop_px = _round_to_tick(stop)
        target_px = _round_to_tick(target) if target is not None else None

        # Phase 8.2 + 10.6: lift the LMT above ``signal.entry`` by the
        # configured buffer so the order is marketable but capped.
        # The Phase 10.6 percentage ceiling (``lmt_buffer_max_pct``)
        # prevents the floor from producing LMTs above IBKR's
        # aggressive-LMT cap on low-priced names. Tick-round AFTER
        # buffer addition to land on a valid grid.
        buffer_breakdown = _compute_lmt_buffer_breakdown(
            entry_price=entry_px,
            buffer_pct=exec_cfg.lmt_buffer_pct,
            buffer_floor_usd=exec_cfg.lmt_buffer_usd_floor,
            buffer_cap_usd=exec_cfg.lmt_buffer_usd_cap,
            max_pct=exec_cfg.lmt_buffer_max_pct,
            anchor_price=market_anchor_price,
        )
        buffer = buffer_breakdown.final
        # Phase 12.5 — when the retry path forces an explicit LMT (because
        # IBKR told us its accepted ceiling), bypass the buffer chain and
        # use the override directly. ``buffer`` becomes the implied
        # back-computed value for log observability only.
        if force_limit_price is not None:
            limit_px = _round_to_tick(force_limit_price)
            buffer = round(limit_px - entry_px, 4)
        else:
            limit_px = _round_to_tick(entry_px + buffer)

        parent = LimitOrder("BUY", shares, limit_px)
        parent.transmit = False
        parent.outsideRth = not rth_only
        apply_default_tif(parent)

        stop_order = StopOrder("SELL", shares, stop_px)
        stop_order.transmit = True
        stop_order.outsideRth = not rth_only
        apply_default_tif(stop_order)
        # Phase 7.6: encode server-side auto-convert STP → TRAIL at +Nx R
        # when enabled. Must be set BEFORE placement; IBKR rejects
        # modifications to these fields on a live order. Anchor stays on
        # ``signal.entry``, NOT the LMT ceiling — the +R trigger should
        # reflect the strategy's intended entry, not the protective
        # ceiling above it.
        self._apply_initial_stop_adjustable_fields(stop_order, entry=entry_px, stop_price=stop_px)

        target_order: LimitOrder | None = None
        if target_px is not None:
            target_order = LimitOrder("SELL", shares, target_px)
            target_order.transmit = False
            target_order.outsideRth = not rth_only
            apply_default_tif(target_order)

        ib = self._ibkr.ib
        parent_trade = ib.placeOrder(contract, parent)
        # Link children to the parent via IBKR's ``parentId``.
        stop_order.parentId = parent_trade.order.orderId
        target_trade: Trade | None = None
        if target_order is not None:
            target_order.parentId = parent_trade.order.orderId
            target_trade = ib.placeOrder(contract, target_order)
        stop_trade = ib.placeOrder(contract, stop_order)
        # Phase 8.2 + 10.6: log the full buffer clamp state. The
        # ``buffer_clamp`` enum is {"floor", "ceiling", "none"} —
        # "ceiling" wins when both floor and ceiling could apply, since
        # the ceiling is the binding safety constraint against IBKR's
        # aggressive-LMT cap.
        _log.info(
            "executor.lmt_bracket_placed",
            symbol=contract.symbol,
            parent_order_id=parent_trade.order.orderId,
            shares=shares,
            entry_price=entry_px,
            limit_price=limit_px,
            buffer=round(buffer, 4),
            buffer_pct_raw=round(buffer_breakdown.pct_raw, 4),
            buffer_clamp=buffer_breakdown.clamp,
            buffer_floor_value=round(buffer_breakdown.floor_value, 4),
            buffer_ceiling_value=round(buffer_breakdown.ceiling_value, 4),
            final_buffer=round(buffer, 4),
            stop_price=stop_px,
            target_price=target_px,
            # Phase 12.5: which anchor drove the percentage ceiling so an
            # operator can grep for "ceiling anchored on prior_close" runs.
            market_anchor_price=(
                round(market_anchor_price, 4) if market_anchor_price is not None else None
            ),
            # Phase 12.5: when the retry path forced an explicit LMT (after
            # an Error 202 cancel of the original placement), this is the
            # IBKR-supplied ceiling (minus a tick) we used. ``None`` for
            # initial placements -- only the retry sets it.
            forced_limit_price=(
                round(force_limit_price, 4) if force_limit_price is not None else None
            ),
        )
        return _BracketTrades(parent=parent_trade, stop=stop_trade, target=target_trade)

    def _place_mkt_bracket(
        self,
        *,
        contract: Contract,
        entry: float,
        stop: float,
        shares: int,
    ) -> _BracketTrades:
        """Phase 6.14.1 — atomic BUY MKT parent + full-size STP child.

        Submits two linked orders in one network round trip:

        * Parent = ``MarketOrder("BUY", shares)`` — fills at top-of-book
          ask on arrival (the manual hotkey semantics).
        * STP SELL = full ``shares`` at ``stop`` — initial protection for
          the entire position. Attached to parent via ``parentId``;
          transmit chain parent=False, stop=True.

        The 50%-share scale-out LMT is NOT in this atomic bracket. IBKR
        auto-normalizes bracket-child quantities to match the parent's —
        live AKAN test on 2026-04-22 confirmed a scale LMT submitted at
        ``shares // 2`` was immediately rewritten to ``shares`` on
        IBKR's side. The scale LMT is instead planted post-fill via
        ``_place_post_fill_scale_out_lmt``, OCA-linked with this stop
        (no parentId → no quantity normalization).

        Returns ``_BracketTrades(parent, stop, target=None,
        scale_lmt=None)``. The scale_lmt field is populated later when
        ``_handle_parent_fill`` plants the post-fill LMT.
        """
        rth_only = self._settings.execution.rth_only
        stop_px = _round_to_tick(stop)

        parent = MarketOrder("BUY", shares)
        parent.outsideRth = not rth_only
        parent.transmit = False
        apply_default_tif(parent)

        stop_order = StopOrder("SELL", shares, stop_px)
        stop_order.outsideRth = not rth_only
        stop_order.transmit = True  # final leg transmits the pair
        apply_default_tif(stop_order)
        # Phase 7.6: encode server-side auto-convert STP → TRAIL at +Nx R.
        # Anchor to signal.entry; actual MKT fill price may slip a few
        # cents higher, so the true +1R trigger may fire marginally
        # earlier than a fill-anchored calculation — acceptable, and
        # conservative (earlier protection).
        self._apply_initial_stop_adjustable_fields(
            stop_order, entry=_round_to_tick(entry), stop_price=stop_px
        )

        ib = self._ibkr.ib
        parent_trade = ib.placeOrder(contract, parent)
        stop_order.parentId = parent_trade.order.orderId
        stop_trade = ib.placeOrder(contract, stop_order)
        _log.info(
            "executor.mkt_bracket_placed",
            symbol=contract.symbol,
            parent_order_id=parent_trade.order.orderId,
            parent_shares=shares,
            stop_order_id=stop_trade.order.orderId,
            stop_shares=shares,
            stop_price=stop_px,
        )
        return _BracketTrades(parent=parent_trade, stop=stop_trade, target=None)

    def _place_adjustable_post_scaleout_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
        initial_risk: float,
    ) -> tuple[Trade, Trade | None, float]:
        """Place a server-side adjustable STP (breakeven → TRAIL), optional runner LMT.

        IBKR's *adjustable stop* feature lets a single STP auto-convert to
        a TRAIL order once the market tags ``triggerPrice``. Encoding it
        directly in the order object means the bot does not have to watch
        price and swap orders itself — TWS handles the conversion even if
        the bot is briefly offline.

        Phase 4i trigger formula: the flat base STP sits at
        ``position.avg_price`` (breakeven). The trigger is
        ``scale_out_price + trail_activation_r_multiple × initial_risk``
        (default scale_out + 1R, i.e. entry + 3R when scale-out is 2R).
        At trigger IBKR converts to a TRAIL with distance
        ``trail_amount_r_multiple × initial_risk`` (default 1R). The
        initial adjusted stop is ``triggerPrice - trail_amount`` so the
        tail locks in profit at conversion.

        When ``execution.runner_target_enabled`` is true, a runner-target
        LMT is OCA-linked with the STP so a ceiling fill cancels the
        trailing exit and vice versa. When disabled (the documented default —
        "no hard profit ceilings on the runner"), only the STP is placed
        and ``target_trade`` is ``None``. Returns
        ``(stp_trade, target_trade, trigger_price)``.
        """
        rth_only = self._settings.execution.rth_only
        # Phase 6.9: breakeven base is the filled entry price; tick-round
        # in case ib reports a sub-penny weighted-average fill.
        entry_price = _round_to_tick(position.avg_price)
        activation_mult = self._settings.execution.trail_activation_r_multiple
        amount_mult = self._settings.execution.trail_amount_r_multiple
        runner_enabled = self._settings.execution.runner_target_enabled
        runner_mult = self._settings.execution.runner_target_multiple

        # All four price fields are validated by IBKR's tick check; round
        # each separately because the trail_amount is a *delta* that also
        # must sit on a tick boundary for stocks >= $1.
        trigger_price = _round_to_tick(position.scale_out_price + initial_risk * activation_mult)
        trail_amount = _round_to_tick(initial_risk * amount_mult)
        adjusted_stop_price = _round_to_tick(trigger_price - trail_amount)

        # Unique per symbol + parent id so concurrent positions on other
        # symbols (future multi-position phase) and re-entries on the same
        # symbol never share an OCA group. Only populated when the runner
        # LMT is also placed — a lone STP does not need an OCA anchor.
        oca_group = f"scaleout_{position.symbol}_{position.parent_order_id}"

        stp_order = Order(
            action="SELL",
            totalQuantity=remaining_shares,
            orderType="STP",
            auxPrice=entry_price,
            outsideRth=not rth_only,
            triggerPrice=trigger_price,
            adjustedOrderType="TRAIL",
            adjustedStopPrice=adjusted_stop_price,
            adjustedTrailingAmount=trail_amount,
            adjustableTrailingUnit=0,  # 0 = trail in price units (not percent)
            # ``transmit=True`` for single-leg; gets flipped to False below
            # when we also place a runner LMT (final leg transmits both).
            transmit=not runner_enabled,
        )
        apply_default_tif(stp_order)

        target_trade: Trade | None = None
        runner_target_price: float | None = None
        if runner_enabled:
            stp_order.ocaGroup = oca_group
            stp_order.ocaType = 1  # 1 = cancel remaining block
            runner_target_price = _round_to_tick(entry_price + initial_risk * runner_mult)
            target_order = LimitOrder("SELL", remaining_shares, runner_target_price)
            target_order.outsideRth = not rth_only
            target_order.ocaGroup = oca_group
            target_order.ocaType = 1
            target_order.transmit = True  # final leg transmits both
            apply_default_tif(target_order)

        ib = self._ibkr.ib
        stp_trade = ib.placeOrder(contract, stp_order)
        self._subscribe_commission(
            stp_trade,
            symbol=position.symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )
        if runner_enabled and runner_target_price is not None:
            target_trade = ib.placeOrder(contract, target_order)
            self._subscribe_commission(
                target_trade,
                symbol=position.symbol,
                leg="exit",
                parent_order_id=position.parent_order_id,
            )
        _log.info(
            "executor.adjustable_stop_placed",
            symbol=position.symbol,
            remaining_shares=remaining_shares,
            base_stop=entry_price,
            trigger_price=trigger_price,
            adjusted_stop_price=adjusted_stop_price,
            trail_amount=trail_amount,
            runner_target=runner_target_price,
            runner_target_enabled=runner_enabled,
            oca_group=oca_group if runner_enabled else None,
        )
        return stp_trade, target_trade, trigger_price

    def _place_static_breakeven_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
    ) -> Trade:
        """Phase 4e fallback — plain STP at breakeven, no OCA, no adjustment.

        Used when ``execution.post_scaleout_stop_mode == "static_breakeven"``.
        Keeps the pre-4h behaviour (TradeManager drives the trailing exit on
        bar closes) so the mode is a clean on/off switch for server-side
        trailing entirely.
        """
        # Phase 6.9: tick-round the breakeven price — ib's reported
        # avg fill can be sub-penny on a partially-filled parent.
        new_stop = StopOrder("SELL", remaining_shares, _round_to_tick(position.avg_price))
        new_stop.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(new_stop)
        trade = self._ibkr.ib.placeOrder(contract, new_stop)
        self._subscribe_commission(
            trade,
            symbol=position.symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )
        return trade

    async def plant_initial_trail(
        self,
        *,
        symbol: str,
        last_close: float,
    ) -> Trade | None:
        """Phase 7.6 (bot_driven mode) — replace the initial STP with a plain TRAIL.

        Called by ``TradeManager.on_bar_update`` once when the bar close
        crosses ``entry + initial_stop_trigger_r_multiple × R`` and the
        position has ``initial_trail_planted=False``. Cancels the
        existing STP, places a plain ``TRAIL`` order in the same OCA
        group as the scale-out LMT (so a scale-out fill still cancels
        the trail), and marks the position so the bar-close guard
        won't re-fire.

        ``last_close`` anchors the TRAIL's initial trigger — IBKR will
        ratchet it upward as price advances. We seed at
        ``last_close - trail_amount`` so a same-bar reversal would fire
        immediately at the configured trail distance below the trigger
        bar's close (rather than seeding at entry, which would let the
        trail rest below the trigger and produce slack).

        Returns the new TRAIL ``Trade``, or ``None`` if the position is
        no longer ``open`` (defensive — a same-bar STP fill could have
        closed it before this call ran).
        """
        position = self._store.get_active(symbol)
        if position is None or position.status != "open":
            _log.info(
                "executor.plant_initial_trail_skipped_inactive",
                symbol=symbol,
                status=position.status if position is not None else None,
            )
            return None
        if position.initial_trail_planted:
            _log.info(
                "executor.plant_initial_trail_skipped_already_planted",
                symbol=symbol,
                stop_order_id=position.stop_order_id,
            )
            return None

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001 - qualify-stock failures are non-fatal here
            _log.error(
                "executor.plant_initial_trail_qualify_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None

        exec_cfg = self._settings.execution
        rth_only = exec_cfg.rth_only
        initial_risk = position.avg_price - position.stop_price
        if initial_risk <= 0.0:
            # Defensive: should never happen post-fill on a long but a
            # widened-stop reconciliation could land here. Skip rather than
            # plant a TRAIL with an absurd trail amount.
            _log.warning(
                "executor.plant_initial_trail_skipped_nonpositive_risk",
                symbol=symbol,
                avg_price=position.avg_price,
                stop_price=position.stop_price,
            )
            return None
        trail_amount = _round_to_tick(initial_risk * exec_cfg.initial_stop_trail_r_multiple)
        initial_trail_stop = _round_to_tick(last_close - trail_amount)

        # Cancel the existing STP first so OCA bookkeeping on IBKR's side
        # doesn't see two protective orders briefly. The new TRAIL inherits
        # the same OCA group as the scale-out LMT (Phase 8.3 fill-anchored
        # plant uses ``f"scale_{symbol}_{parent_order_id}"``); a TRAIL fill
        # then cancels the LMT and vice versa, matching the original STP
        # ↔ scale_lmt OCA semantics.
        bracket = self._active_trades.get(symbol)
        if bracket is not None and bracket.stop is not None:
            self._cancel_trade_silently(bracket.stop)

        oca_group = f"scale_{symbol}_{position.parent_order_id}"
        trail_order = Order(
            action="SELL",
            totalQuantity=position.shares,
            orderType="TRAIL",
            auxPrice=trail_amount,
            trailStopPrice=initial_trail_stop,
            outsideRth=not rth_only,
            ocaGroup=oca_group,
            ocaType=1,  # 1 = cancel remaining block (matches scale-out OCA)
            transmit=True,
        )
        apply_default_tif(trail_order)
        trade = self._ibkr.ib.placeOrder(contract, trail_order)
        self._subscribe_commission(
            trade,
            symbol=symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )

        # Wire the same fill handler the rest of the bot uses for stops so
        # a TRAIL fill closes the position correctly (mark_closing → mark_closed
        # → journal), with sibling cancel disabled because the OCA group already
        # handles cancelling the scale-out LMT on a TRAIL fill.
        self._wire_post_scale_stop_handler(symbol, trade)

        # Replace the bracket's stop slot in-place so subsequent code that
        # reaches for ``bracket.stop`` sees the new TRAIL (not the cancelled STP).
        if bracket is not None:
            self._active_trades[symbol] = _BracketTrades(
                parent=bracket.parent,
                stop=trade,
                target=bracket.target,
                scale_lmt=bracket.scale_lmt,
            )

        try:
            self._store.mark_initial_trail_planted(
                symbol,
                new_stop_order_id=ref_req_id(trade),
                new_stop_price=initial_trail_stop,
            )
        except Exception as exc:  # noqa: BLE001 - state-mutator errors must not orphan the live TRAIL
            _log.error(
                "executor.plant_initial_trail_mark_failed",
                symbol=symbol,
                error=str(exc),
                hint="TRAIL is live on IBKR but in-memory store didn't update; restart or reconcile.",
            )

        _log.info(
            "executor.initial_trail_planted",
            symbol=symbol,
            mode="bot_driven",
            shares=position.shares,
            trail_amount=trail_amount,
            initial_trail_stop=initial_trail_stop,
            last_close=last_close,
            initial_risk=round(initial_risk, 4),
            trail_r_multiple=exec_cfg.initial_stop_trail_r_multiple,
            oca_group=oca_group,
        )
        return trade

    def _place_immediate_trail_stop(
        self,
        *,
        contract: Contract,
        position: Position,
        remaining_shares: int,
        initial_risk: float,
    ) -> Trade:
        """Phase 6.14 — IBKR TRAIL order planted immediately at scale-out.

        Distinct from Phase 4h's ``adjustable_to_trail`` which keeps a
        breakeven STP resting until price tags a conversion trigger, and
        only then becomes a TRAIL. ``immediate_trail`` fires the TRAIL
        logic on the spot: the stop starts at ``scale_out_price -
        trail_amount`` (≈ +1R above entry under default settings) and
        follows the runner upward, locking in profit on every new high.

        The ``trail_amount`` honours ``trail_amount_r_multiple`` from
        config (1.0 × initial_risk by default = 1R trail distance).
        Tick-rounded at the executor boundary so IBKR's tick check
        accepts the absolute stop price on >= $1 stocks (Phase 6.9).
        """
        rth_only = self._settings.execution.rth_only
        amount_mult = self._settings.execution.trail_amount_r_multiple
        trail_amount = _round_to_tick(initial_risk * amount_mult)
        # Initial stop position: scale-out price minus the trail distance.
        # At scale-out moment current price ≈ scale_out_price, so this
        # pre-seeds the trail such that a same-tick reversal would fire
        # the stop at +1R above entry (with defaults). IBKR will move
        # this upward as price advances.
        initial_trail_stop = _round_to_tick(position.scale_out_price - trail_amount)
        trail_order = Order(
            action="SELL",
            totalQuantity=remaining_shares,
            orderType="TRAIL",
            auxPrice=trail_amount,
            trailStopPrice=initial_trail_stop,
            outsideRth=not rth_only,
            transmit=True,
        )
        apply_default_tif(trail_order)
        trade = self._ibkr.ib.placeOrder(contract, trail_order)
        self._subscribe_commission(
            trade,
            symbol=position.symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )
        _log.info(
            "executor.immediate_trail_placed",
            symbol=position.symbol,
            remaining_shares=remaining_shares,
            trail_amount=trail_amount,
            initial_trail_stop=initial_trail_stop,
            scale_out_price=position.scale_out_price,
            trail_amount_r_multiple=amount_mult,
        )
        return trade

    def _wire_fill_handlers(self, symbol: str, bracket: _BracketTrades) -> None:
        """Hook ``filledEvent`` on each leg; child fills cross-cancel the sibling.

        Adopted brackets from ``reconcile`` may be missing a leg (e.g. the
        parent already filled before the crash) — skip subscription when the
        leg is ``None``. The cross-cancel in ``_handle_child_fill`` tolerates
        a ``None`` sibling.

        Phase 4k: each leg also gets a ``commissionReportEvent`` subscription
        so IBKR's per-fill commission is banked against the right leg on the
        journal row.
        """

        def _spawn(coro: Any) -> None:
            task = asyncio.create_task(coro)
            self._pending_fill_tasks.add(task)
            task.add_done_callback(self._pending_fill_tasks.discard)

        if bracket.parent is not None:

            def _on_parent_filled(trade: Trade) -> None:
                _spawn(self._handle_parent_fill(symbol, trade))

            bracket.parent.filledEvent += _on_parent_filled
            self._subscribe_commission(bracket.parent, symbol=symbol, leg="entry")

        if bracket.stop is not None:
            # Phase 6.14: on MKT atomic bracket the stop's sibling is
            # the scale-out LMT (50%-share take-profit), not a runner
            # target. Cancel it on stop-out so the scale LMT doesn't
            # linger when the position has already exited at loss.
            stop_sibling = bracket.target if bracket.target is not None else bracket.scale_lmt

            def _on_stop_filled(trade: Trade) -> None:
                _spawn(
                    self._handle_child_fill(
                        symbol=symbol,
                        filled_trade=trade,
                        sibling_trade=stop_sibling,
                        fill_type="stop",
                    )
                )

            bracket.stop.filledEvent += _on_stop_filled
            self._subscribe_commission(bracket.stop, symbol=symbol, leg="exit")

        if bracket.target is not None:

            def _on_target_filled(trade: Trade) -> None:
                _spawn(
                    self._handle_child_fill(
                        symbol=symbol,
                        filled_trade=trade,
                        sibling_trade=bracket.stop,
                        fill_type="target",
                    )
                )

            bracket.target.filledEvent += _on_target_filled
            self._subscribe_commission(bracket.target, symbol=symbol, leg="exit")

        if bracket.scale_lmt is not None:
            # Phase 6.14 — half-size scale-out LMT. On fill, transition
            # to ``scaled_out``: cancel the full-size stop, plant the
            # new post-scale stop (per ``post_scaleout_stop_mode``),
            # journal the scale leg. Distinct from runner-target fills
            # because the position is NOT closed — the remaining half
            # keeps running under the new trail/breakeven stop.
            def _on_scale_filled(trade: Trade) -> None:
                _spawn(self._handle_scale_out_lmt_fill(symbol, trade))

            bracket.scale_lmt.filledEvent += _on_scale_filled
            self._subscribe_commission(bracket.scale_lmt, symbol=symbol, leg="scale")

    def subscribe_commission(
        self,
        trade: Trade | None,
        *,
        symbol: str,
        leg: Literal["entry", "scale", "exit"],
        parent_order_id: int,
    ) -> None:
        """Public wrapper so ``TradeManager`` can wire commissions on scale/exit market sells."""
        self._subscribe_commission(trade, symbol=symbol, leg=leg, parent_order_id=parent_order_id)

    def _subscribe_commission(
        self,
        trade: Trade | None,
        *,
        symbol: str,
        leg: Literal["entry", "scale", "exit"],
        parent_order_id: int | None = None,
    ) -> None:
        """Phase 4k — bank each ``CommissionReport`` against the right leg on the journal.

        IBKR fires ``commissionReportEvent(trade, fill, report)`` after the
        fill event; we accumulate on the in-memory Position (best-effort,
        observational) AND on the journal row additively so late-arriving
        reports on already-closed trades still land. ``parent_order_id`` is
        optional for the standard bracket path (we resolve it from the
        active Position) but required for post-scale-out legs where the
        caller already has it in hand.
        """
        if trade is None:
            return

        def _on_commission(_trade: Any, _fill: Any, report: Any) -> None:
            amount_raw = getattr(report, "commission", 0.0)
            try:
                amount = float(amount_raw or 0.0)
            except (TypeError, ValueError):
                amount = 0.0
            if amount <= 0.0:
                return
            # Best-effort: update the in-memory position if still present. A
            # report that arrives after ``clear_closed`` runs will miss this
            # update but still land on the journal row below.
            with contextlib.suppress(UnknownPositionError):
                if leg == "entry":
                    self._store.add_entry_commission(symbol, amount)
                elif leg == "scale":
                    self._store.add_scale_commission(symbol, amount)
                else:
                    self._store.add_exit_commission(symbol, amount)
            # Journal additively — ``add_commission`` is the source of truth
            # for post-session analytics.
            resolved_parent_id = parent_order_id
            if resolved_parent_id is None:
                pos = self._store.get(symbol)
                if pos is None:
                    _log.warning(
                        "executor.commission_unresolved_parent_id",
                        symbol=symbol,
                        leg=leg,
                        amount=amount,
                    )
                    return
                resolved_parent_id = pos.parent_order_id

            async def _persist() -> None:
                try:
                    await self._journal.add_commission(resolved_parent_id, leg=leg, amount=amount)
                except Exception as exc:  # noqa: BLE001 - journaling is observational
                    _log.error(
                        "executor.commission_journal_failed",
                        symbol=symbol,
                        leg=leg,
                        error=str(exc),
                    )

            task = asyncio.create_task(_persist())
            self._pending_fill_tasks.add(task)
            task.add_done_callback(self._pending_fill_tasks.discard)

        trade.commissionReportEvent += _on_commission

    async def drain_pending_fills(self) -> None:
        """Await every currently-scheduled fill-handler task.

        Eventkit fires synchronously, so fill handlers are scheduled via
        ``asyncio.create_task`` — but aiosqlite's thread-backed commits need
        more than a single event-loop tick to finish. Tests (and any caller
        that needs the journal row visible) should ``await`` this before
        asserting.
        """
        while self._pending_fill_tasks:
            tasks = tuple(self._pending_fill_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_parent_fill(self, symbol: str, trade: Trade) -> None:
        """Parent fill → transition to ``open``, persist to journal, notify.

        Phase 4j: when the parent was a BUY STP-LMT, the bracket was
        transmitted alone — this is where we plant the stop + optional
        runner LMT so the fill doesn't leave a naked long on the account.
        """
        fill_price, filled_shares = _extract_fill(trade)
        if filled_shares <= 0:
            _log.warning("executor.parent_fill_zero_shares", symbol=symbol)
            return
        try:
            position = self._store.mark_filled(
                symbol, fill_price=fill_price, filled_shares=filled_shares
            )
        except (UnknownPositionError, InvalidPositionTransitionError) as exc:
            _log.error("executor.parent_fill_transition_failed", symbol=symbol, error=str(exc))
            return

        # Phase 10.3 — invalidate the cached account summary so the next
        # entry signal's risk gate sees post-fill ``AvailableFunds`` /
        # ``BuyingPower``. The Phase 4b max_concurrent_positions=1 default
        # makes back-to-back signals rare in practice, but a re-entry on
        # the same symbol after a partial scale-out exit can land within
        # the cache TTL — the invalidation makes the staleness window
        # zero in the cases that matter.
        self._ibkr.invalidate_account_summary_cache()

        # Phase 9.6: trades_today increments at fill, not approval. ``mark_filled``
        # only succeeds for the first fill of a position (subsequent partial
        # fills don't re-enter this path), so on_first_fill is called exactly
        # once per opened position. Rejection counter is also cleared here so
        # a clean fill resets any prior transient broker rejections.
        try:
            await self._risk_engine.on_first_fill(symbol)
        except Exception as exc:  # noqa: BLE001 - book-keeping must not block the fill flow
            _log.error("executor.on_first_fill_failed", symbol=symbol, error=str(exc))

        # Phase 4j (``STP_LMT``): parent transmitted alone, full
        # protection children planted post-fill. Failure to plant is
        # serious — we market-close the orphan long to avoid carrying
        # shares without a stop.
        # Phase 8.3 (``LMT`` and ``MKT``): the loose signal-anchored
        # STP placed at signal time covers the fill-notification
        # window. Here we cancel it and plant a fill-anchored OCA
        # pair (new STP at ``fill - intended_R`` + scale_lmt at
        # ``fill + N×intended_R``). LMT uses this path now in addition
        # to MKT — same drift fix applies to both.
        if position.entry_order_type == "STP_LMT":
            placed = await self._place_entry_protection_children(symbol=symbol, position=position)
            if placed is None:
                return
            position = placed
        elif position.entry_order_type in {"MKT", "LMT"}:
            updated = await self._place_post_fill_scale_out_lmt(symbol=symbol, position=position)
            if updated is not None:
                position = updated

        try:
            await self._journal.open_trade(
                position,
                runner_target_multiple_used=self._settings.execution.runner_target_multiple,
            )
        except Exception as exc:  # noqa: BLE001 - journaling is observational
            _log.error("executor.journal_open_failed", symbol=symbol, error=str(exc))
        # Phase 4d: ``entries_count`` was bumped at ``open_position`` time, so
        # the current value is the 1-indexed number for the filled entry.
        entry_number = self._store.symbol_history(symbol).entries_count
        await self._send_fill(position, "entry", entry_number=entry_number)

        # Phase 11 — fire the exit-advisor "position protected"
        # notification. By this point the parent has filled and the
        # path-specific protection placement has run (STP_LMT post-fill
        # plant or LMT/MKT fill-anchored re-plant), so the position is
        # genuinely protected. The notify call short-circuits when
        # ``exit_advisor.enabled=false`` (production main default).
        from bot.exit_advisor.hook.registry import notify_position_protected

        notify_position_protected(position)

    async def _place_post_fill_scale_out_lmt(
        self, *, symbol: str, position: Position
    ) -> Position | None:
        """Phase 8.3 — cancel the loose signal-anchored STP and plant fill-anchored protection.

        Runs from ``_handle_parent_fill`` on both MKT and LMT entries (Phase
        6.14.1 originally MKT-only; Phase 8.3 extends to LMT for the same
        drift fix). The atomic STP placed at signal time used
        ``signal.stop`` — which means realized risk = ``actual_fill -
        signal.stop``, NOT the strategy's intended R when the fill slips
        above ``signal.entry``.

        This function cancels the original STP and re-plants the
        protection OCA pair anchored to ``actual_fill``:

        * ``intended_R`` is derived from the position's signal-time
          ``scale_out_price - stop_price``, divided by ``(1 + N)`` where
          ``N = scale_out_multiple``. (We don't store ``signal.entry``
          on Position; the derivation is exact for the current
          ``scale_out = entry + N×R`` formula.)
        * New STP at ``fill - intended_R`` (full size).
        * New scale-out LMT at ``fill + N × intended_R`` (half size).
        * Both OCA-linked with type 2 ("Reduce with block") so the
          LMT filling auto-shrinks the STP to the runner half.
        * Phase 7.6 adjustable fields anchored to ``actual_fill`` so the
          ``+1R`` auto-convert-to-TRAIL trigger reflects the actual
          cost basis, not the signal-time entry.

        Position store is updated via ``update_fill_anchored_prices`` so
        downstream readers (bar-close + tick-driven scale-out backups,
        which key on ``position.scale_out_price``) see the same
        threshold that's resting on the exchange.

        Returns the updated Position on success (or ``None`` on placement
        failure / pre-flight skip — the original STP is still resting in
        most failure paths, so the account isn't naked).

        Brief unprotected window (~50ms) between cancel of original STP
        and submit of new STP. Acceptable per existing MKT precedent.
        """
        bracket = self._active_trades.get(symbol)
        if bracket is None or bracket.stop is None:
            _log.warning(
                "executor.post_fill_protection_no_bracket",
                symbol=symbol,
                parent_order_id=position.parent_order_id,
            )
            return None
        shares_to_scale = position.shares // 2
        if shares_to_scale <= 0:
            _log.info(
                "executor.post_fill_protection_skipped_tiny_size",
                symbol=symbol,
                total_shares=position.shares,
            )
            return None

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001 - qualify can raise many shapes
            _log.error(
                "executor.post_fill_protection_qualify_failed", symbol=symbol, error=str(exc)
            )
            return None

        # --- Phase 8.3 fill-anchored price derivation ---
        # Phase 8.4: prefer the direct lookup ``entry_trigger_price - stop_price``
        # (Phase 4j stored ``entry_trigger_price = signal.entry`` on the
        # Position). This is robust to dynamic scale-out targets like
        # the Phase 8.4 premarket-high cap, where the ``scale_out =
        # entry + N×R`` assumption no longer holds. Fallback: if
        # ``entry_trigger_price <= 0`` (reconciled-from-IBKR positions
        # that pre-date Phase 4j or were adopted across a restart),
        # derive from the formula. Less accurate when a cap is binding,
        # but reconciled positions never have a cap applied anyway.
        rth_only = self._settings.execution.rth_only
        scale_out_multiple = self._settings.execution.scale_out_multiple
        signal_stop = float(position.stop_price)
        signal_scale_out = float(position.scale_out_price)
        signal_entry = float(position.entry_trigger_price)
        fill_price = float(position.avg_price)
        if signal_entry > 0.0:
            intended_r = signal_entry - signal_stop
            r_source = "entry_trigger_price"
        else:
            # Formula fallback for reconciled positions without a known signal entry.
            intended_r = (signal_scale_out - signal_stop) / (1.0 + scale_out_multiple)
            r_source = "formula_fallback"
            _log.warning(
                "executor.intended_r_formula_fallback",
                symbol=symbol,
                signal_scale_out=signal_scale_out,
                signal_stop=signal_stop,
                derived_r=round(intended_r, 4),
                hint="entry_trigger_price <= 0 (likely reconciled/adopted position).",
            )
        new_stop_px = _round_to_tick(fill_price - intended_r)
        new_scale_out_px = _round_to_tick(fill_price + scale_out_multiple * intended_r)
        oca_group = f"scale_{symbol}_{position.parent_order_id}"

        # Cancel original loose STP. ~50ms unprotected window between
        # this cancel and the new STP being acked by IBKR.
        original_stop_trade = bracket.stop
        self._cancel_trade_silently(original_stop_trade)

        new_stop_order = StopOrder("SELL", position.shares, new_stop_px)
        new_stop_order.outsideRth = not rth_only
        new_stop_order.ocaGroup = oca_group
        new_stop_order.ocaType = 2  # Reduce with block
        apply_default_tif(new_stop_order)
        # Phase 7.6 (re-applied): anchor the +1R trigger to actual_fill,
        # not signal.entry. Initial risk derived as ``fill - new_stop_px``
        # which equals ``intended_R`` by construction.
        self._apply_initial_stop_adjustable_fields(
            new_stop_order,
            entry=_round_to_tick(fill_price),
            stop_price=new_stop_px,
        )

        scale_order = LimitOrder("SELL", shares_to_scale, new_scale_out_px)
        scale_order.outsideRth = not rth_only
        scale_order.ocaGroup = oca_group
        scale_order.ocaType = 2
        apply_default_tif(scale_order)

        ib = self._ibkr.ib
        try:
            new_stop_trade = ib.placeOrder(contract, new_stop_order)
            scale_trade = ib.placeOrder(contract, scale_order)
        except Exception as exc:  # noqa: BLE001 - IB can raise many shapes
            _log.error(
                "executor.post_fill_protection_place_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None

        # Update Position state with fill-anchored prices + new STP id.
        # Bar-close / tick-driven scale-out backups read these, so they
        # must match what's resting on the exchange.
        try:
            updated_position = self._store.update_fill_anchored_prices(
                symbol,
                new_stop_price=new_stop_px,
                new_scale_out_price=new_scale_out_px,
                new_stop_order_id=int(new_stop_trade.order.orderId),
            )
        except (UnknownPositionError, InvalidPositionTransitionError) as exc:
            _log.error(
                "executor.post_fill_protection_state_update_failed",
                symbol=symbol,
                error=str(exc),
            )
            updated_position = position

        # Refresh the bracket: parent (still filled/done), new stop, scale LMT.
        self._active_trades[symbol] = _BracketTrades(
            parent=bracket.parent,
            stop=new_stop_trade,
            target=bracket.target,
            scale_lmt=scale_trade,
        )
        self._wire_post_fill_mkt_handlers(symbol, new_stop_trade, scale_trade)
        _log.info(
            "executor.protection_fill_anchored",
            symbol=symbol,
            parent_order_id=position.parent_order_id,
            entry_order_type=position.entry_order_type,
            fill_price=fill_price,
            signal_entry=signal_entry,
            signal_stop=signal_stop,
            signal_scale_out=signal_scale_out,
            intended_r=round(intended_r, 4),
            r_source=r_source,
            new_stop_order_id=int(new_stop_trade.order.orderId),
            new_stop_shares=position.shares,
            new_stop_price=new_stop_px,
            scale_lmt_order_id=int(scale_trade.order.orderId),
            scale_lmt_shares=shares_to_scale,
            scale_lmt_price=new_scale_out_px,
            oca_group=oca_group,
        )
        return updated_position

    def _wire_post_fill_mkt_handlers(
        self, symbol: str, stop_trade: Trade, scale_trade: Trade
    ) -> None:
        """Wire filledEvent on the post-fill MKT stop + scale LMT pair.

        The original atomic-bracket stop already had a handler wired from
        ``_wire_fill_handlers``; we cancelled that trade and replaced it
        with an OCA-linked fresh one. Wire the new stop's filledEvent so
        stop-out still closes the position, and wire the scale LMT's
        filledEvent to trigger the scale-out transition.
        """

        def _spawn(coro: Any) -> None:
            task = asyncio.create_task(coro)
            self._pending_fill_tasks.add(task)
            task.add_done_callback(self._pending_fill_tasks.discard)

        def _on_stop_filled(trade: Trade) -> None:
            _spawn(
                self._handle_child_fill(
                    symbol=symbol,
                    filled_trade=trade,
                    sibling_trade=scale_trade,
                    fill_type="stop",
                )
            )

        def _on_scale_filled(trade: Trade) -> None:
            _spawn(self._handle_scale_out_lmt_fill(symbol, trade))

        stop_trade.filledEvent += _on_stop_filled
        self._subscribe_commission(stop_trade, symbol=symbol, leg="exit")
        scale_trade.filledEvent += _on_scale_filled
        self._subscribe_commission(scale_trade, symbol=symbol, leg="scale")

    async def _place_entry_protection_children(
        self, *, symbol: str, position: Position
    ) -> Position | None:
        """Phase 4j — plant the stop + optional runner LMT after a STP-LMT parent fills.

        Children are OCA-linked when the runner target is enabled so a
        ceiling fill cancels the stop and vice versa. When the runner is
        disabled (the documented default — "no hard profit ceilings") only the STP
        is placed. On placement failure we market-close the long rather
        than carry it unprotected; returning ``None`` signals the caller to
        abort the post-fill flow.
        """
        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001 - qualify can raise many shapes
            _log.error("executor.protection_qualify_failed", symbol=symbol, error=str(exc))
            await self._emergency_market_close(symbol, position)
            return None

        rth_only = self._settings.execution.rth_only
        runner_enabled = self._settings.execution.runner_target_enabled
        shares = position.shares

        # Phase 6.9: tick-round protection-child prices. The signal-time
        # stop_price / runner_target_price come straight from the
        # strategy's round(x, 4) emit and can hold sub-penny precision.
        stop_px = _round_to_tick(position.stop_price)
        stop_order = StopOrder("SELL", shares, stop_px)
        stop_order.outsideRth = not rth_only
        apply_default_tif(stop_order)
        # Phase 7.6: anchor the +R trigger to the actual fill price, which
        # is known here (post-fill path for STP_LMT entries).
        self._apply_initial_stop_adjustable_fields(
            stop_order, entry=_round_to_tick(position.avg_price), stop_price=stop_px
        )

        target_order: LimitOrder | None = None
        runner_target_price = (
            _round_to_tick(position.runner_target_price)
            if position.runner_target_price is not None
            else None
        )
        if runner_enabled and runner_target_price is not None:
            oca_group = f"entry_{symbol}_{position.parent_order_id}"
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1
            target_order = LimitOrder("SELL", shares, runner_target_price)
            target_order.outsideRth = not rth_only
            target_order.ocaGroup = oca_group
            target_order.ocaType = 1
            apply_default_tif(target_order)

        try:
            ib = self._ibkr.ib
            stop_trade = ib.placeOrder(contract, stop_order)
            target_trade: Trade | None = None
            if target_order is not None:
                target_trade = ib.placeOrder(contract, target_order)
        except Exception as exc:  # noqa: BLE001 - IB can raise many shapes
            _log.error(
                "executor.protection_place_failed",
                symbol=symbol,
                error=str(exc),
            )
            await self._emergency_market_close(symbol, position)
            return None

        updated = self._store.attach_protection_children(
            symbol,
            stop_order_id=int(stop_trade.order.orderId),
            target_order_id=(int(target_trade.order.orderId) if target_trade is not None else 0),
        )

        bracket = self._active_trades.get(symbol)
        if bracket is not None:
            bracket.stop = stop_trade
            bracket.target = target_trade
            self._wire_child_fill_handlers(symbol, bracket)

        _log.info(
            "executor.protection_children_placed",
            symbol=symbol,
            parent_order_id=position.parent_order_id,
            stop_order_id=int(stop_trade.order.orderId),
            target_order_id=(int(target_trade.order.orderId) if target_trade is not None else None),
            runner_target_enabled=runner_enabled,
        )
        return updated

    async def _emergency_market_close(self, symbol: str, position: Position) -> None:
        """Phase 4j — market-close an unprotected long if child placement fails.

        Called only from ``_place_entry_protection_children`` when the STP
        order could not be planted. Leaves the position state machine at
        ``open`` (the parent *did* fill) but fires a loud log + best-effort
        market SELL so the operator sees the naked long and the account
        isn't left carrying it while the scheduler catches up at 15:55.
        """
        _log.error(
            "executor.unprotected_long_emergency_close",
            symbol=symbol,
            shares=position.shares,
            parent_order_id=position.parent_order_id,
            hint="STP-LMT parent filled but protection child placement failed.",
        )
        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("executor.emergency_close_qualify_failed", symbol=symbol, error=str(exc))
            return
        close_order = MarketOrder("SELL", position.shares)
        close_order.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(close_order)
        with contextlib.suppress(Exception):
            self._ibkr.ib.placeOrder(contract, close_order)

    def _wire_child_fill_handlers(self, symbol: str, bracket: _BracketTrades) -> None:
        """Phase 4j — wire only stop + target fill handlers (parent already fired).

        Used after a STP-LMT parent fills and we plant children post-hoc;
        re-running ``_wire_fill_handlers`` would double-subscribe the parent
        (whose ``filledEvent`` already dispatched) so we limit the hook-up
        to the two new legs.
        """

        def _spawn(coro: Any) -> None:
            task = asyncio.create_task(coro)
            self._pending_fill_tasks.add(task)
            task.add_done_callback(self._pending_fill_tasks.discard)

        if bracket.stop is not None:

            def _on_stop_filled(trade: Trade) -> None:
                _spawn(
                    self._handle_child_fill(
                        symbol=symbol,
                        filled_trade=trade,
                        sibling_trade=bracket.target,
                        fill_type="stop",
                    )
                )

            bracket.stop.filledEvent += _on_stop_filled
            self._subscribe_commission(bracket.stop, symbol=symbol, leg="exit")

        if bracket.target is not None:

            def _on_target_filled(trade: Trade) -> None:
                _spawn(
                    self._handle_child_fill(
                        symbol=symbol,
                        filled_trade=trade,
                        sibling_trade=bracket.stop,
                        fill_type="target",
                    )
                )

            bracket.target.filledEvent += _on_target_filled
            self._subscribe_commission(bracket.target, symbol=symbol, leg="exit")

    async def _handle_scale_out_lmt_fill(self, symbol: str, filled_trade: Trade) -> None:
        """Phase 6.14 — half-size scale-out LMT filled; transition to scaled_out.

        Equivalent to ``TradeManager._execute_scale_out`` but driven by
        the LMT fill event instead of a 1-min bar close — the LMT fires
        the instant price touches ``scale_out_price``, eliminating the
        bar-close lag. TradeManager's scale-out path becomes a
        belt-and-suspenders fallback for the unlikely case the LMT
        didn't fill (thin book, halt, data gap) but a subsequent bar
        close confirmed price reached scale_out.

        Steps:
        1. Guard against double-scale (``position.scaled_out`` already).
        2. Cancel the full-size protection STP (now oversized for the
           remaining half).
        3. Call ``mark_scaled`` (records partial PnL, updates
           ``position.shares`` to remaining half, stamps post-scale
           stop type).
        4. Plant the new post-scale stop per
           ``execution.post_scaleout_stop_mode`` (immediate_trail |
           adjustable_to_trail | static_breakeven).
        5. Journal the scale leg.
        6. Notifier push.
        """
        fill_price, cum_qty = _extract_fill(filled_trade)
        if cum_qty <= 0:
            _log.warning("executor.scale_out_lmt_zero_shares", symbol=symbol)
            return
        position = self._store.get_active(symbol)
        if position is None or position.status != "open":
            _log.info(
                "executor.scale_out_lmt_skipped_inactive",
                symbol=symbol,
                status=position.status if position is not None else None,
            )
            return
        if position.scaled_out:
            _log.info("executor.scale_out_lmt_skipped_already_scaled", symbol=symbol)
            return

        bracket = self._active_trades.get(symbol)
        shares_sold = cum_qty
        remaining = position.shares - shares_sold
        if remaining <= 0:
            _log.warning(
                "executor.scale_out_lmt_full_fill_unexpected",
                symbol=symbol,
                shares_sold=shares_sold,
                total_shares=position.shares,
            )
            return

        # Cancel the full-size protection stop — it's oversized for
        # the remaining half and must be replaced with the post-scale stop.
        if bracket is not None:
            self._cancel_trade_silently(bracket.stop)

        # Qualify the contract for the new stop placement. Cheap: IB
        # caches contract qualification server-side.
        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("executor.scale_out_lmt_qualify_failed", symbol=symbol, error=str(exc))
            return

        initial_risk = position.avg_price - position.stop_price
        mode = self._settings.execution.post_scaleout_stop_mode
        stop_type: PostScaleoutStopType
        trigger_price: float | None = None
        if mode == "adjustable_to_trail" and initial_risk > 0.0:
            new_stop_trade, _runner_trade, trigger_price = (
                self._place_adjustable_post_scaleout_stop(
                    contract=contract,
                    position=position,
                    remaining_shares=remaining,
                    initial_risk=initial_risk,
                )
            )
            stop_type = "adjustable_to_trail"
        elif mode == "immediate_trail" and initial_risk > 0.0:
            new_stop_trade = self._place_immediate_trail_stop(
                contract=contract,
                position=position,
                remaining_shares=remaining,
                initial_risk=initial_risk,
            )
            stop_type = "immediate_trail"
        else:
            new_stop_trade = self._place_static_breakeven_stop(
                contract=contract, position=position, remaining_shares=remaining
            )
            stop_type = "static_breakeven"

        partial_pnl = (fill_price - position.avg_price) * shares_sold
        try:
            self._store.mark_scaled(
                symbol,
                remaining_shares=remaining,
                scale_partial_pnl=partial_pnl,
                new_stop_price=position.avg_price,
                new_stop_order_id=int(new_stop_trade.order.orderId),
                post_scaleout_stop_type=stop_type,
                post_scaleout_adjustment_trigger_price=trigger_price,
            )
        except InvalidPositionTransitionError as exc:
            _log.error("executor.scale_out_lmt_mark_scaled_failed", symbol=symbol, error=str(exc))
            return

        # Replace the bracket's stop with the newly-planted post-scale stop.
        # scale_lmt is now done (its own fill). target stays as-is.
        if bracket is not None:
            self._active_trades[symbol] = _BracketTrades(
                parent=bracket.parent,
                stop=new_stop_trade,
                target=bracket.target,
                scale_lmt=None,
            )
            self._wire_post_scale_stop_handler(symbol, new_stop_trade)

        _log.info(
            "executor.scale_out_lmt_filled",
            symbol=symbol,
            shares_sold=shares_sold,
            remaining=remaining,
            fill_price=round(fill_price, 4),
            partial_pnl=round(partial_pnl, 2),
            post_scaleout_stop_type=stop_type,
            adjustment_trigger_price=trigger_price,
        )

        # Fresh snapshot so the notifier renders the post-scale fields.
        updated = self._store.get_active(symbol)
        if updated is not None and self._notifier is not None:
            try:
                await self._notifier.send_fill(updated, "scale_out")
            except Exception as exc:  # noqa: BLE001 — notifications must not crash the fill loop
                _log.error("executor.scale_out_notify_failed", symbol=symbol, error=str(exc))

    def _wire_post_scale_stop_handler(self, symbol: str, stop_trade: Trade) -> None:
        """Wire filledEvent on a post-scale stop so tail stop-out closes the position.

        Phase 6.14 — mirrors the pre-Phase-6.14 wiring that TradeManager
        did when it placed the post-scale stop itself. Re-uses the
        existing ``_handle_child_fill`` logic with ``fill_type="stop"``
        (scale_out_then_trail classification is derived from the
        position's scaled_out flag).
        """

        def _spawn(coro: Any) -> None:
            task = asyncio.create_task(coro)
            self._pending_fill_tasks.add(task)
            task.add_done_callback(self._pending_fill_tasks.discard)

        def _on_post_scale_stop_filled(trade: Trade) -> None:
            _spawn(
                self._handle_child_fill(
                    symbol=symbol,
                    filled_trade=trade,
                    sibling_trade=None,  # post-scale bracket has no sibling to cancel
                    fill_type="stop",
                )
            )

        stop_trade.filledEvent += _on_post_scale_stop_filled

    async def _handle_child_fill(
        self,
        *,
        symbol: str,
        filled_trade: Trade,
        sibling_trade: Trade | None,
        fill_type: str,
    ) -> None:
        """Stop/target fill → mark closed, cancel sibling, journal, notify."""
        exit_price, exit_shares = _extract_fill(filled_trade)
        if exit_shares <= 0:
            _log.warning("executor.child_fill_zero_shares", symbol=symbol, fill_type=fill_type)
            return
        position = self._store.get(symbol)
        if position is None:
            _log.error("executor.child_fill_unknown_symbol", symbol=symbol)
            return
        # Include already-banked scale-out profit so the day's ledger is whole.
        pnl = (exit_price - position.avg_price) * position.shares + position.scale_partial_pnl

        # Already closing/closed — another leg beat us here. Not an error.
        with contextlib.suppress(InvalidPositionTransitionError):
            self._store.mark_closing(symbol, reason=fill_type)

        # Belt-and-suspenders cancel — IBKR's OCA should handle this, but
        # redundant cancellation is idempotent and hardens against edge cases
        # where the sibling stays "Submitted" after the winner fills.
        self._cancel_trade_silently(sibling_trade)

        try:
            closed = self._store.mark_closed(
                symbol, exit_price=exit_price, pnl=pnl, closed_at=_now_utc()
            )
        except InvalidPositionTransitionError as exc:
            _log.error("executor.close_transition_failed", symbol=symbol, error=str(exc))
            return

        # Phase 10.3 — invalidate the cached account summary so the next
        # entry signal's risk gate sees post-close ``AvailableFunds`` /
        # ``BuyingPower``. A close restores margin; the next entry should
        # see that capital available immediately, not after the TTL expires.
        self._ibkr.invalidate_account_summary_cache()

        # Phase 4d exit classification. A stop-fill *after* scale-out means the
        # breakeven stop on the tail caught it — we treat that as
        # "scale_out_then_trail" because the first half was banked and the tail
        # exited via our trailing logic (even if TradeManager's in-memory path
        # didn't run — e.g. a gap fill outside the poll loop).
        exit_type: ExitType
        if fill_type == "target":
            exit_type = "target_hit"
        elif position.scaled_out:
            exit_type = "scale_out_then_trail"
        else:
            exit_type = "stop_hit"

        try:
            await self._journal.update_exit(
                closed, exit_price=exit_price, pnl=pnl, exit_type=exit_type
            )
        except Exception as exc:  # noqa: BLE001 - journaling is observational
            _log.error("executor.journal_update_failed", symbol=symbol, error=str(exc))

        history = self._store.symbol_history(symbol)
        history.record_exit(
            exit_time=closed.closed_at or _now_utc(),
            pnl=pnl,
            exit_type=exit_type,
        )

        self._active_trades.pop(symbol, None)
        await self._send_fill(closed, fill_type)
        try:
            await self._risk_engine.on_fill_closed(closed, pnl)
        except Exception as exc:  # noqa: BLE001 - risk accounting must not crash the fill loop
            _log.error("executor.risk_on_fill_failed", symbol=symbol, error=str(exc))

    async def _send_fill(
        self,
        position: Position,
        fill_type: str,
        *,
        entry_number: int | None = None,
    ) -> None:
        """Best-effort Telegram push on fills; ``None`` notifier is the dev default.

        ``entry_number`` (Phase 4d) is the 1-indexed re-entry count for
        ``entry`` fills — passed through to the notifier so the Telegram
        header reads ``ENTRY FILL — Entry #2``.
        """
        if self._notifier is None:
            return
        try:
            await self._notifier.send_fill(position, fill_type, entry_number=entry_number)
        except Exception as exc:  # noqa: BLE001 - notifications must never crash execution
            _log.error(
                "executor.notify_fill_failed",
                symbol=position.symbol,
                fill_type=fill_type,
                error=str(exc),
            )

    def _cancel_trade_silently(self, trade: Trade | None) -> None:
        """Cancel a Trade with idempotent swallowing; already-done orders are no-ops."""
        if trade is None or trade.isDone():
            return
        try:
            self._ibkr.ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001 - IB raises many shapes on cancel
            _log.warning(
                "executor.cancel_failed",
                order_id=trade.order.orderId,
                error=str(exc),
            )

    def _synthesize_reconciled_position(
        self, *, symbol: str, shares: int, avg_cost: float
    ) -> Position:
        """Build a minimal ``Position`` from an IBKR-reported orphan, marked ``open``.

        We don't know the original stop/target/strategy — fill sentinels so
        the operator can see "something's here, go sort it out." The
        ``reasons`` list carries a breadcrumb marking the provenance.
        """
        return Position(
            symbol=symbol,
            strategy="reconciled",
            shares=shares,
            avg_price=avg_cost,
            stop_price=0.0,
            scale_out_price=0.0,
            runner_target_price=None,
            parent_order_id=0,
            stop_order_id=0,
            target_order_id=0,
            opened_at=_now_utc(),
            status="open",
            reasons=["reconciled_from_ibkr"],
            adopted_from_reconcile=True,
        )

    def _adopt_pending_entry_trigger(
        self,
        *,
        symbol: str,
        parent_trade: Trade,
        shares: int,
        trigger_price: float,
        limit_price: float,
    ) -> None:
        """Phase 4j — adopt a resting BUY STP-LMT parent as ``pending_entry_trigger``.

        The pre-trigger state only has the parent on IBKR's books; there
        are no children to rebuild and no fill to journal. We synthesize a
        minimal Position so the flatten + CLI paths treat it identically
        to a fresh Phase 4j placement. ``stop_price`` is carried forward
        as ``trigger_price`` (the intended entry), which is enough for
        TradeManager to compute R-multiples once the fill lands.
        """
        position = Position(
            symbol=symbol,
            strategy="reconciled",
            shares=shares,
            avg_price=0.0,
            stop_price=0.0,
            scale_out_price=0.0,
            runner_target_price=None,
            parent_order_id=int(parent_trade.order.orderId),
            stop_order_id=0,
            target_order_id=0,
            opened_at=_now_utc(),
            status="pending_entry_trigger",
            reasons=["reconciled_from_ibkr"],
            adopted_from_reconcile=True,
            entry_order_type="STP_LMT",
            entry_trigger_price=trigger_price,
        )
        self._store.insert_reconciled(position)
        bracket = _BracketTrades(parent=parent_trade, stop=None, target=None)
        self._active_trades[symbol] = bracket
        self._wire_fill_handlers(symbol, bracket)
        _log.warning(
            "reconcile.adopted_pending_entry_trigger",
            symbol=symbol,
            parent_order_id=int(parent_trade.order.orderId),
            shares=shares,
            trigger_price=trigger_price,
            limit_price=limit_price,
        )

    def _adopt_bracket(
        self,
        *,
        symbol: str,
        status: PositionStatus,
        shares: int,
        avg_price: float,
        stop_trade: Trade | None,
        target_trade: Trade | None,
        parent_trade: Trade | None,
        parent_order_id: int | None = None,
    ) -> None:
        """Insert an adopted bracket into the store + wire fill handlers.

        Used for both pending_entry (parent still open) and open (parent
        already filled) adoptions. When ``parent_trade`` is present we use
        its ``orderId`` + ``lmtPrice`` + ``totalQuantity``; when it's None
        we fall back to ``parent_order_id`` and the IBKR-reported share lot.
        """
        resolved_parent_id = int(
            parent_order_id
            if parent_order_id is not None
            else (parent_trade.order.orderId if parent_trade is not None else 0)
        )
        stop_price = (
            float(getattr(stop_trade.order, "auxPrice", 0.0) or 0.0)
            if stop_trade is not None
            else 0.0
        )
        target_price = (
            float(getattr(target_trade.order, "lmtPrice", 0.0) or 0.0)
            if target_trade is not None
            else 0.0
        )
        stop_order_id = int(stop_trade.order.orderId) if stop_trade is not None else 0
        target_order_id = int(target_trade.order.orderId) if target_trade is not None else 0

        # Phase 4h: detect an adjustable STP and populate the post-scale-out
        # bookkeeping. ib_async encodes "unset" adjustment fields with a
        # float-max sentinel (1.7976931348623157e+308), so a triggerPrice
        # below that + ``adjustedOrderType == "TRAIL"`` identifies an
        # adjustable stop planted by a previous process.
        post_scaleout_type: PostScaleoutStopType | None = None
        adjustment_trigger: float | None = None
        if stop_trade is not None:
            trigger_raw = float(getattr(stop_trade.order, "triggerPrice", 0.0) or 0.0)
            adjusted_type = getattr(stop_trade.order, "adjustedOrderType", "") or ""
            if _IBKR_UNSET_FLOAT_SENTINEL > trigger_raw > 0.0 and adjusted_type == "TRAIL":
                post_scaleout_type = "adjustable_to_trail"
                adjustment_trigger = round(trigger_raw, 4)
                _log.info(
                    "reconcile.adopted_adjustable_stop",
                    symbol=symbol,
                    stop_order_id=stop_order_id,
                    trigger_price=adjustment_trigger,
                    adjusted_order_type=adjusted_type,
                )

        # Phase 4e: on an orphan bracket we only see IBKR's live numbers — we
        # don't know the original scale-out. When a runner LMT is present,
        # use it as both ``runner_target_price`` (that *is* what IBKR is
        # working) and a conservative ``scale_out_price`` fallback so
        # TradeManager's scale math still has a sensible anchor if the
        # tail never reaches it. Phase 4i: when the LMT is absent (either
        # runner_target_enabled=false or post-scale-out adjustable STP-only
        # state), runner_target_price becomes None and scale_out_price
        # falls back to the stop price so the +N R trigger can still be
        # reconstructed upstream.
        #
        # If the bracket's target implies a runner multiple that disagrees
        # with the configured one (e.g. prior session ran a 2R ceiling, now
        # we're on 3R), warn so an operator can notice the drift before the
        # next re-entry on the same symbol books against stale economics.
        parent_entry_price = (
            float(getattr(parent_trade.order, "lmtPrice", 0.0) or 0.0)
            if parent_trade is not None
            else avg_price
        )
        if (
            stop_trade is not None
            and target_trade is not None
            and stop_price > 0.0
            and target_price > 0.0
            and parent_entry_price > stop_price
        ):
            implied_risk = parent_entry_price - stop_price
            implied_multiple = (target_price - parent_entry_price) / implied_risk
            configured_multiple = self._settings.execution.runner_target_multiple
            if abs(implied_multiple - configured_multiple) > 0.05:
                _log.warning(
                    "reconcile.target_multiple_mismatch",
                    symbol=symbol,
                    parent_order_id=resolved_parent_id,
                    implied_multiple=round(implied_multiple, 3),
                    configured_multiple=configured_multiple,
                    target_price=target_price,
                    hint="Adopted bracket's runner target differs from "
                    "execution.runner_target_multiple; preserving IBKR-reported values.",
                )

        runner_target_value: float | None = target_price if target_trade is not None else None
        scale_out_value = target_price if target_trade is not None else stop_price
        # Phase 4i: an adopted adjustable STP implies we're already in
        # post-scale-out state, so the tail should inherit the # red-candle suppression.
        red_candle_exit_suppressed = post_scaleout_type == "adjustable_to_trail"

        position = Position(
            symbol=symbol,
            strategy="reconciled",
            shares=shares,
            avg_price=avg_price,
            stop_price=stop_price,
            scale_out_price=scale_out_value,
            runner_target_price=runner_target_value,
            parent_order_id=resolved_parent_id,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            opened_at=_now_utc(),
            status=status,
            reasons=["reconciled_from_ibkr"],
            adopted_from_reconcile=True,
            post_scaleout_stop_type=post_scaleout_type,
            post_scaleout_adjustment_trigger_price=adjustment_trigger,
            red_candle_exit_suppressed=red_candle_exit_suppressed,
        )
        self._store.insert_reconciled(position)
        bracket = _BracketTrades(parent=parent_trade, stop=stop_trade, target=target_trade)
        self._active_trades[symbol] = bracket
        self._wire_fill_handlers(symbol, bracket)
        _log.warning(
            "reconcile.adopted_orphan_bracket",
            symbol=symbol,
            parent_order_id=resolved_parent_id,
            stop_order_id=stop_order_id,
            target_order_id=target_order_id,
            status=status,
            shares=shares,
        )


def _split_bracket_children(
    children: list[Trade], parent_trade: Trade
) -> tuple[Trade | None, Trade | None]:
    """Separate a bracket's children into (stop, target) by order type + action.

    Target is a LMT with the opposite action of the parent (BUY parent → SELL
    target). STP is the stop-loss. Returns ``(None, None)`` slots for legs
    that aren't present.
    """
    parent_action = getattr(parent_trade.order, "action", None)
    stop_trade = next((c for c in children if getattr(c.order, "orderType", None) == "STP"), None)
    target_trade = next(
        (
            c
            for c in children
            if getattr(c.order, "orderType", None) == "LMT"
            and getattr(c.order, "action", None) != parent_action
        ),
        None,
    )
    return stop_trade, target_trade


def _extract_fill(trade: Trade) -> tuple[float, int]:
    """Pull (avgPrice, cumQty) from the most recent execution on ``trade``.

    Returns ``(0.0, 0)`` if there are no fills — callers treat that as a no-op.
    """
    if not trade.fills:
        return 0.0, 0
    last = trade.fills[-1].execution
    price = float(getattr(last, "avgPrice", 0.0) or 0.0)
    shares = int(getattr(last, "cumQty", 0) or 0)
    return price, shares


def _extract_broker_error_code(trade: Trade | None) -> int | None:
    """Phase 9.6 — pull the most recent error code from a trade's log, or None.

    ib_async's ``Trade.log`` accumulates ``TradeLogEntry`` records as the
    order progresses; broker auto-cancels carry an ``errorCode`` field
    (e.g. 10349 for the Day 8 RPGL SCM eligibility reject). Returns the
    last non-zero, non-None code; ``None`` when no error is recorded
    (e.g. operator-cancelled in TWS or a pure status transition).
    """
    if trade is None or not getattr(trade, "log", None):
        return None
    for entry in reversed(trade.log):
        code = getattr(entry, "errorCode", None)
        if code:
            try:
                return int(code)
            except (TypeError, ValueError):
                return None
    return None


def _now_utc() -> datetime:
    """UTC-now as a tz-aware datetime (SQLite roundtrips expect tz-aware)."""
    return datetime.now(UTC)
