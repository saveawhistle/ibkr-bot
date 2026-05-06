"""Phase 10.1 — naked-position watchdog.

Detection-only safety floor: every tick, classify each tracked open position
into ``PROTECTED`` / ``PROTECTED_PENDING`` / ``UNDERPROTECTED`` / ``NAKED``
by comparing the bot's view of the position size against the working SELL
orders the executor (or operator) has placed on IBKR. Underprotected and
naked classifications fire an operator alert via the existing Telegram
notifier; the alert carries an inline-keyboard "Ack" button whose tap
suppresses re-fires for that ``(symbol, classification)`` until the
position transitions out of the bad state, the size changes, or the
trading day rolls over.

The watchdog does not place or cancel any orders. Auto-remediation is a
later phase decision; this module establishes the safety floor and the
audit trail.

Triggering precedent (2026-04-30 BIYA, session_2026-04-30.jsonl):
1. Phase 8.3 protection planted correctly: STP 83 shares + LMT 41 shares.
2. Trail stop partial-filled 42/83 shares on the 09:34 down-bar.
3. IBKR's ``ocaType=2`` (REDUCE_WITH_BLOCK) proportionally reduced the
   LMT scale-out from 41 → 20 shares.
4. Both orders later cancelled at 13:43:36 with no bot-side event (manual
   TWS cancel or preset auto-cancel).
5. Remaining 41-share position sat naked for the rest of the session.

The watchdog catches the moment in step 5 — and equivalent failure modes
regardless of cause — by polling IBKR's working-orders cache and bot
state on every orchestrator iteration (self-throttled to a configurable
minimum interval, default 5 s).

Design notes:

* **Order-type-only protection discriminator.** A SELL LMT placed above
  market is a take-profit, not protection — the BIYA scenario is exactly
  this shape (working SELL LMT @ $2.77, no working stop, position naked).
  We classify on ``order.orderType`` alone; do not infer from price
  comparison since live mid-prices race the classifier.

* **Bot-state vs. IBKR-state mismatch is a separate alert class.** If
  ``executor.store`` says 100 shares but IBKR says 50 (or vice versa),
  we emit ``watchdog.position_state_mismatch`` *and* still classify the
  bot's view — both signals are useful and the operator wants to see
  them as distinct concerns.

* **Suppression is per ``(symbol, classification)``.** A ``NAKED`` ack
  does not silence a future ``UNDERPROTECTED`` (different shape, may
  imply different remediation). Position-size changes blanket-rearm so
  the operator sees fresh state on partial fills / scale-outs.

* **Shadow mode** (``watchdog.shadow_mode=True``) runs all detection +
  event emission, but suppresses Telegram sends. Default-on for the
  initial deployment; flip off after one clean session to enable live
  alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

import structlog

from bot.config import Settings, get_settings

if TYPE_CHECKING:
    from bot.brokerage.ibkr_client import IBKRClient
    from bot.execution.position_state import Position, PositionStore
    from bot.notify import Notifier

_log = structlog.get_logger("bot.execution.watchdog")


# Order-type strings (uppercased) that count as protective stops on the SELL
# side. ``STP`` = plain stop (Phase 4j initial protection). ``STP LMT`` = stop
# with limit ceiling (rare for protection but supported). ``TRAIL`` = server-
# side trailing stop (Phase 6.14 immediate trail + Phase 7.6 adjustable
# conversion target). A SELL LMT is a take-profit, not protection — see the
# module docstring for the BIYA precedent.
_PROTECTIVE_SELL_ORDER_TYPES: Final[frozenset[str]] = frozenset(
    {"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"}
)

# IBKR order statuses that mean "this order is no longer working on the
# wire". Anything else (PendingSubmit, PreSubmitted, Submitted, Inactive
# in the active sense) counts as still-resting and should be considered
# in protection accounting.
_INACTIVE_ORDER_STATUSES: Final[frozenset[str]] = frozenset(
    {"Cancelled", "Filled", "ApiCancelled", "PendingCancel", "Inactive"}
)

# Truncate alert messages so a runaway state machine can't spam Telegram
# with a 4 KB payload. 600 chars covers the structured fields below
# comfortably.
_ALERT_TEXT_MAX = 600


class Classification(StrEnum):
    """Per-position protection-state buckets.

    Ordered by severity (NAKED worst, PROTECTED best) so callers can
    reason about transitions without consulting a separate ranking.
    """

    PROTECTED = "PROTECTED"
    PROTECTED_PENDING = "PROTECTED_PENDING"
    UNDERPROTECTED = "UNDERPROTECTED"
    NAKED = "NAKED"


_ALERTABLE: Final[frozenset[Classification]] = frozenset(
    {Classification.UNDERPROTECTED, Classification.NAKED}
)


@dataclass(frozen=True)
class WorkingSellOrder:
    """Snapshot of one bot-owned SELL order found in IBKR's working orders.

    ``is_protective`` is True iff ``order_type`` is in
    :data:`_PROTECTIVE_SELL_ORDER_TYPES`. Stored explicitly so downstream
    consumers (alert formatter, log payload) don't have to re-look-up the
    set.
    """

    order_id: int
    order_type: str
    quantity: int
    aux_price: float | None
    lmt_price: float | None
    is_protective: bool

    def describe(self) -> str:
        """One-line human description for inclusion in a Telegram alert."""
        kind = "protective" if self.is_protective else "take-profit (not protective)"
        if self.order_type in {"STP", "TRAIL"} and self.aux_price is not None:
            price_part = f"@ ${self.aux_price:.2f} aux"
        elif self.order_type == "STP LMT" and self.aux_price is not None:
            lmt = self.lmt_price if self.lmt_price is not None else 0.0
            price_part = f"@ ${self.aux_price:.2f} aux / ${lmt:.2f} lmt"
        elif self.order_type == "LMT" and self.lmt_price is not None:
            price_part = f"@ ${self.lmt_price:.2f}"
        else:
            price_part = ""
        return f"{self.quantity} SELL {self.order_type} {price_part} ({kind})".strip()


@dataclass
class _SymbolState:
    """Per-symbol watchdog memory: when first seen open, last classification, last size."""

    first_seen_open_at: datetime
    last_classification: Classification | None = None
    last_position_size: int | None = None
    # Per (symbol, classification) — first time we observed this bad state in
    # the current arming cycle. Used in alert messages so the operator sees
    # how long the position has been unprotected.
    first_unprotected_at: dict[Classification, datetime] = field(default_factory=dict)


class Watchdog:
    """Naked-position detector + operator alerter.

    The orchestrator calls :meth:`tick` once per loop iteration; this method
    self-throttles to ``settings.watchdog.check_interval_seconds`` and is a
    no-op when ``settings.watchdog.enabled`` is False.

    Detection only — never places or cancels orders.
    """

    def __init__(
        self,
        *,
        ibkr: IBKRClient,
        position_store: PositionStore,
        notifier: Notifier | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire dependencies; ``notifier=None`` runs in detection-only mode (no Telegram path)."""
        self._ibkr = ibkr
        self._store = position_store
        self._notifier = notifier
        self._settings = settings or get_settings()
        self._cfg = self._settings.watchdog
        self._tz = ZoneInfo(self._settings.session.timezone)
        self._symbols: dict[str, _SymbolState] = {}
        self._suppressed: set[tuple[str, Classification]] = set()
        # Tracks (symbol, classification) tuples whose ack we've already
        # logged via ``watchdog.alert_acked``. The notifier may hold the
        # ack indefinitely; we only want to log the *first* observation.
        self._ack_logged: set[tuple[str, Classification]] = set()
        self._last_ran_at: datetime | None = None
        # NY-local date the most recent tick observed; rollover clears
        # suppressions and the underlying ack registry on the notifier.
        self._last_trading_day: date | None = None

    # ------------------------------------------------------------------
    # Public API — orchestrator calls tick(); tests may call _evaluate
    # directly with synthetic state.
    # ------------------------------------------------------------------

    async def tick(self) -> None:
        """Self-throttled evaluation. Safe to call on every orchestrator iteration."""
        now = datetime.now(UTC)
        if self._last_ran_at is not None:
            since = (now - self._last_ran_at).total_seconds()
            if since < self._cfg.check_interval_seconds:
                return
        self._last_ran_at = now

        if not self._cfg.enabled:
            return

        self._maybe_handle_day_rollover(now)

        try:
            await self._evaluate_once(now)
        except Exception as exc:  # noqa: BLE001 - watchdog faults must not crash the loop
            _log.error("watchdog.tick_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _evaluate_once(self, now: datetime) -> None:
        """Read state once and evaluate every tracked position."""
        bot_positions = self._store.list_active()
        # Drop per-symbol memory for symbols no longer tracked. This
        # cleans up across position close → re-entry on the same symbol
        # (re-arms naturally) and across watchlist eviction.
        tracked_symbols = {p.symbol for p in bot_positions}
        for stale in [s for s in self._symbols if s not in tracked_symbols]:
            self._symbols.pop(stale, None)
            for c in list(Classification):
                self._suppressed.discard((stale, c))
                self._clear_notifier_ack(stale, c)
        if not bot_positions:
            return

        # IBKR-side state. ``ib.positions()`` and ``ib.openTrades()`` are
        # both in-memory caches kept up-to-date by ib_async's event
        # callbacks — synchronous and free, no network round-trip per tick.
        ib = self._ibkr.ib
        try:
            ibkr_positions = ib.positions()
            open_trades = ib.openTrades()
        except Exception as exc:  # noqa: BLE001 - swallow + skip this tick
            _log.warning("watchdog.ibkr_read_failed", error=str(exc))
            return

        ibkr_size_by_symbol: dict[str, int] = {}
        for ibp in ibkr_positions:
            try:
                size = int(ibp.position)
            except (TypeError, ValueError):
                continue
            if size == 0:
                continue
            sym = getattr(getattr(ibp, "contract", None), "symbol", None)
            if sym:
                ibkr_size_by_symbol[sym] = size

        sells_by_symbol = self._collect_bot_sell_orders(open_trades)

        for position in bot_positions:
            self._evaluate_position(
                position=position,
                ibkr_size_by_symbol=ibkr_size_by_symbol,
                sells_by_symbol=sells_by_symbol,
                now=now,
            )

    def _collect_bot_sell_orders(self, open_trades: list[Any]) -> dict[str, list[WorkingSellOrder]]:
        """Filter ``open_trades`` to bot-owned, currently-resting SELL orders, keyed by symbol."""
        bot_client_id = self._settings.ibkr.client_id
        out: dict[str, list[WorkingSellOrder]] = {}
        for trade in open_trades:
            order = getattr(trade, "order", None)
            contract = getattr(trade, "contract", None)
            order_status = getattr(trade, "orderStatus", None)
            if order is None or contract is None:
                continue
            if getattr(order, "clientId", None) != bot_client_id:
                continue
            if getattr(order, "action", "") != "SELL":
                continue
            status = getattr(order_status, "status", "") or ""
            if status in _INACTIVE_ORDER_STATUSES:
                continue
            symbol = getattr(contract, "symbol", "") or ""
            if not symbol:
                continue
            order_type = (getattr(order, "orderType", "") or "").upper()
            try:
                qty = int(getattr(order, "totalQuantity", 0) or 0)
            except (TypeError, ValueError):
                qty = 0
            # Treat partially-filled remaining quantity correctly: if the
            # order has fills, the wire-side resting size is
            # ``totalQuantity - filled``. ``orderStatus.remaining`` carries
            # this when present.
            remaining = getattr(order_status, "remaining", None)
            if isinstance(remaining, (int, float)) and remaining > 0:
                qty = int(remaining)
            aux = getattr(order, "auxPrice", 0.0)
            lmt = getattr(order, "lmtPrice", 0.0)
            wso = WorkingSellOrder(
                order_id=int(getattr(order, "orderId", 0) or 0),
                order_type=order_type,
                quantity=qty,
                aux_price=float(aux) if aux else None,
                lmt_price=float(lmt) if lmt else None,
                is_protective=order_type in _PROTECTIVE_SELL_ORDER_TYPES,
            )
            out.setdefault(symbol, []).append(wso)
        return out

    def _evaluate_position(
        self,
        *,
        position: Position,
        ibkr_size_by_symbol: dict[str, int],
        sells_by_symbol: dict[str, list[WorkingSellOrder]],
        now: datetime,
    ) -> None:
        """Classify one position and dispatch alerts/events."""
        symbol = position.symbol
        # Only evaluate ``open`` positions. Pending entries don't have
        # shares on the wire yet; closing positions are by definition
        # being unwound and don't need protection.
        if position.status != "open":
            return

        state = self._ensure_symbol_state(symbol, now)

        # Re-arm on any size change.
        if state.last_position_size is not None and state.last_position_size != position.shares:
            self._rearm(
                symbol=symbol,
                reason="position_size_changed",
                prev=state.last_position_size,
                new=position.shares,
            )
            state = self._symbols[symbol]
        state.last_position_size = position.shares

        # Bot vs IBKR mismatch — emitted as a separate concern.
        ibkr_size = ibkr_size_by_symbol.get(symbol)
        if ibkr_size is None or ibkr_size != position.shares:
            self._emit_mismatch(
                symbol=symbol, bot_shares=position.shares, ibkr_shares=ibkr_size, now=now
            )

        sells = sells_by_symbol.get(symbol, [])
        protective_qty = sum(s.quantity for s in sells if s.is_protective)

        in_grace = (now - state.first_seen_open_at).total_seconds() < self._cfg.entry_grace_seconds

        if protective_qty >= position.shares:
            classification = Classification.PROTECTED
        elif in_grace:
            classification = Classification.PROTECTED_PENDING
        elif sells:
            # Any working SELL exists (LMT take-profit, partial-cover stop,
            # etc.) but combined protective quantity falls short. The BIYA
            # 2026-04-30 case lands here — a SELL LMT @ $2.77 but no
            # protective stop on the wire.
            classification = Classification.UNDERPROTECTED
        else:
            classification = Classification.NAKED

        # Track first-observation of bad state per (symbol, classification)
        # for the alert payload.
        if classification in _ALERTABLE and classification not in state.first_unprotected_at:
            state.first_unprotected_at[classification] = now

        # Transition events.
        if state.last_classification != classification:
            self._emit_transition(
                symbol=symbol,
                position=position,
                classification=classification,
                sells=sells,
                protective_qty=protective_qty,
            )
            state.last_classification = classification
            # Auto-resolve: if we just transitioned into PROTECTED, clear
            # any prior suppressions so a subsequent regression re-alerts.
            if classification == Classification.PROTECTED:
                self._clear_all_suppressions_for_symbol(symbol, reason="auto_resolved")
                state.first_unprotected_at.clear()

        if classification in _ALERTABLE:
            self._maybe_alert(
                symbol=symbol,
                position=position,
                classification=classification,
                sells=sells,
                protective_qty=protective_qty,
                now=now,
            )

    # ------------------------------------------------------------------
    # Per-symbol state helpers
    # ------------------------------------------------------------------

    def _ensure_symbol_state(self, symbol: str, now: datetime) -> _SymbolState:
        """Lazily allocate per-symbol memory the first time we see it on a tick."""
        state = self._symbols.get(symbol)
        if state is None:
            state = _SymbolState(first_seen_open_at=now)
            self._symbols[symbol] = state
        return state

    def _rearm(self, *, symbol: str, reason: str, prev: int, new: int) -> None:
        """Clear all suppressions + first_unprotected_at memory for ``symbol``.

        Position-size changes (partial fill, scale-out) re-arm so the
        operator gets a fresh alert if the new state is still bad. Also
        clears the underlying ack ids on the notifier so a subsequent
        identical-shape alert is delivered.
        """
        cleared: list[str] = []
        for c in list(Classification):
            key = (symbol, c)
            if key in self._suppressed:
                self._suppressed.discard(key)
                cleared.append(c.value)
            self._ack_logged.discard(key)
            self._clear_notifier_ack(symbol, c)
        state = self._symbols.get(symbol)
        if state is not None:
            state.first_unprotected_at.clear()
        if cleared:
            _log.info(
                "watchdog.suppressions_cleared",
                symbol=symbol,
                reason=reason,
                cleared=cleared,
                prev_size=prev,
                new_size=new,
            )

    def _clear_all_suppressions_for_symbol(self, symbol: str, *, reason: str) -> None:
        """Clear suppressions for ``symbol`` without touching the size-change re-arm path."""
        cleared: list[str] = []
        for c in list(Classification):
            key = (symbol, c)
            if key in self._suppressed:
                self._suppressed.discard(key)
                cleared.append(c.value)
            self._ack_logged.discard(key)
            self._clear_notifier_ack(symbol, c)
        if cleared:
            _log.info(
                "watchdog.suppressions_cleared",
                symbol=symbol,
                reason=reason,
                cleared=cleared,
            )

    def _maybe_handle_day_rollover(self, now: datetime) -> None:
        """If the NY-local date has changed since last tick, blanket-clear suppressions."""
        ny_today = now.astimezone(self._tz).date()
        if self._last_trading_day is None:
            self._last_trading_day = ny_today
            return
        if ny_today == self._last_trading_day:
            return
        prev_day = self._last_trading_day
        self._last_trading_day = ny_today
        # Clear every suppressed pair + any acked ids.
        cleared = [(s, c.value) for (s, c) in self._suppressed]
        self._suppressed.clear()
        self._ack_logged.clear()
        for state in self._symbols.values():
            state.first_unprotected_at.clear()
        for symbol in self._symbols:
            for c in list(Classification):
                self._clear_notifier_ack(symbol, c)
        if cleared:
            _log.info(
                "watchdog.suppressions_cleared",
                reason="trading_day_rollover",
                prev_day=prev_day.isoformat(),
                new_day=ny_today.isoformat(),
                cleared_count=len(cleared),
            )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_transition(
        self,
        *,
        symbol: str,
        position: Position,
        classification: Classification,
        sells: list[WorkingSellOrder],
        protective_qty: int,
    ) -> None:
        """Fire ``watchdog.position_*`` for a classification transition."""
        event_name = {
            Classification.PROTECTED: "watchdog.position_protected",
            Classification.PROTECTED_PENDING: "watchdog.position_protected_pending",
            Classification.UNDERPROTECTED: "watchdog.position_underprotected",
            Classification.NAKED: "watchdog.position_naked",
        }[classification]
        _log.info(
            event_name,
            symbol=symbol,
            shares=position.shares,
            protective_quantity=protective_qty,
            working_sell_orders=[s.describe() for s in sells],
        )

    def _emit_mismatch(
        self, *, symbol: str, bot_shares: int, ibkr_shares: int | None, now: datetime
    ) -> None:
        """Fire ``watchdog.position_state_mismatch`` (separate concern from naked alerts).

        This is throttled to one fire per (symbol, mismatch-shape) to avoid
        log floods when the desync is persistent. The shape key is
        ``(symbol, bot_shares, ibkr_shares)``; if any value changes, a new
        event fires.
        """
        state = self._ensure_symbol_state(symbol, now)
        # Reuse first_unprotected_at to dedup mismatch shapes; key with
        # a synthetic Classification-shaped tuple via stringly stash.
        # Simpler: track on the state itself.
        shape = (bot_shares, ibkr_shares)
        prev = getattr(state, "_last_mismatch_shape", None)
        if prev == shape:
            return
        state._last_mismatch_shape = shape  # type: ignore[attr-defined]
        _log.warning(
            "watchdog.position_state_mismatch",
            symbol=symbol,
            bot_shares=bot_shares,
            ibkr_shares=ibkr_shares,
        )

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def _maybe_alert(
        self,
        *,
        symbol: str,
        position: Position,
        classification: Classification,
        sells: list[WorkingSellOrder],
        protective_qty: int,
        now: datetime,
    ) -> None:
        """Send a Telegram alert if not already suppressed.

        Suppression sources:
          * Operator tapped Ack on a prior alert for the same
            ``(symbol, classification)``.
          * The watchdog already sent an alert this arming cycle and the
            classification hasn't changed (one alert per cycle by design;
            re-arm conditions are size change / day rollover / auto-resolve).
        """
        key = (symbol, classification)

        # Operator-acked path: persists across re-fires until re-armed.
        if self._notifier is not None and self._notifier.is_alert_acked(
            self._build_ack_id(symbol, classification)
        ):
            if key not in self._ack_logged:
                self._ack_logged.add(key)
                ack_ts = self._notifier.ack_timestamp(self._build_ack_id(symbol, classification))
                _log.info(
                    "watchdog.alert_acked",
                    symbol=symbol,
                    classification=classification.value,
                    acked_at=(ack_ts or now).isoformat(),
                )
            self._suppressed.add(key)
            return

        # Already suppressed (e.g., we sent an alert earlier this cycle and
        # the operator hasn't acked yet — don't re-fire on every tick).
        if key in self._suppressed:
            _log.info(
                "watchdog.alert_suppressed",
                symbol=symbol,
                classification=classification.value,
                reason="prior_alert_pending_ack",
            )
            return

        ack_id = self._build_ack_id(symbol, classification)
        first_at = self._symbols[symbol].first_unprotected_at.get(classification, now)
        text = self._format_alert(
            symbol=symbol,
            position=position,
            classification=classification,
            sells=sells,
            protective_qty=protective_qty,
            first_observed_at=first_at,
        )

        if self._cfg.shadow_mode:
            _log.info(
                "watchdog.shadow_alert_skipped",
                symbol=symbol,
                classification=classification.value,
                first_observed_at=first_at.isoformat(),
                ack_id=ack_id,
                shares=position.shares,
                protective_quantity=protective_qty,
            )
            self._suppressed.add(key)
            return

        if self._notifier is None:
            # No Telegram wired — log the alert text so it's still visible
            # in session JSONL, then suppress to avoid every-tick repeats.
            _log.warning(
                "watchdog.alert_no_notifier",
                symbol=symbol,
                classification=classification.value,
                text=text[:_ALERT_TEXT_MAX],
            )
            self._suppressed.add(key)
            return

        # Live alert dispatch. Errors are logged by the notifier; we
        # always-suppress after attempting so a Telegram outage doesn't
        # produce a tick-rate flood when service comes back.
        self._suppressed.add(key)
        _log.info(
            "watchdog.alert_sent",
            symbol=symbol,
            classification=classification.value,
            ack_id=ack_id,
            first_observed_at=first_at.isoformat(),
        )
        # Notifier returns a coroutine. The orchestrator doesn't await us
        # in a place where exceptions matter (we run inside tick(), itself
        # try/excepted), so fire-and-forget is fine — the notifier swallows
        # its own errors per its existing contract.
        import asyncio  # local import to avoid pulling asyncio into hot path  # noqa: PLC0415

        asyncio.create_task(self._notifier.send_alert_with_ack(text=text, ack_id=ack_id))

    def _build_ack_id(self, symbol: str, classification: Classification) -> str:
        """Compose the ack id used as the inline-keyboard callback_data."""
        return f"watchdog:{symbol}:{classification.value}"

    def _clear_notifier_ack(self, symbol: str, classification: Classification) -> None:
        """Best-effort ack clear; safe when notifier is None."""
        if self._notifier is None:
            return
        ack_id = self._build_ack_id(symbol, classification)
        self._notifier.clear_alert_ack(ack_id)

    def _format_alert(
        self,
        *,
        symbol: str,
        position: Position,
        classification: Classification,
        sells: list[WorkingSellOrder],
        protective_qty: int,
        first_observed_at: datetime,
    ) -> str:
        """Render the Telegram-bound alert body. Tight; operator should act on it directly."""
        emoji = "🟥" if classification == Classification.NAKED else "🟧"
        header = f"{emoji} {classification.value} — ${symbol}"
        first_local = first_observed_at.astimezone(self._tz).strftime("%H:%M:%S ET")
        lines = [
            header,
            f"Position: {position.shares} shares @ avg ${position.avg_price:.2f}",
            f"Protective qty on wire: {protective_qty} (need ≥ {position.shares})",
        ]
        if sells:
            lines.append("Working SELL orders:")
            for s in sells:
                lines.append(f"  • {s.describe()}")
        else:
            lines.append("Working SELL orders: 0")
        lines.append(f"First observed: {first_local}")
        body = "\n".join(lines)
        if len(body) > _ALERT_TEXT_MAX:
            body = body[: _ALERT_TEXT_MAX - 3] + "..."
        return body
