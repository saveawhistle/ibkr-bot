"""Phase 4b trade management: scale-out at the signal's anchor, then trailing exits.

Per PLAN §2.4, after a bracket's parent fills and the position is ``open``,
this module takes over the remaining lifecycle on every 1-min bar close:

1. **Pre-scale red-candle exit (Phase 7.8)** — any bar that closes red
   (close < open) AND below the prior close triggers a full-position
   market-close before scale-out. the "first red candle close" rule:
   "if the first red candle closes against me, take the loss." Belt-and-
   suspenders on top of the server-side STP (paper TWS STPs sometimes
   miss; this covers it). Gated by
   ``execution.pre_scale_red_candle_exit_enabled`` (default True).

2. **First-target scale-out** — when the bar close reaches
   ``position.scale_out_price`` (Phase 4i: the strategy-time anchor,
   ``entry + scale_out_multiple × initial_risk``, defaults to +2R under
   the 2:1 R:R rule), cancel both original bracket children (stop + target
   — both are full-share orders that would over-sell if left alive),
   market-sell 50% of the shares to bank profit, and install a new STP
   at entry (breakeven) sized for the remaining 50%. Flag
   ``position.scaled_out = True`` +
   ``position.red_candle_exit_suppressed = True``.

   Reading from ``position.scale_out_price`` (rather than recomputing
   from realized fill + stop) keeps the trigger consistent with the
   strategy's signal-time anchor the journal recorded. A parent-fill slip
   of a couple of cents therefore banks a touch more or less than 2R of
   *realized* dollars — acceptable for the simpler invariant: "we always
   scale at the price the signal told us we would."

   Phase 4e's ``runner_target_price`` is the bracket-LMT ceiling on the
   runner half; TradeManager doesn't reach that ceiling via its own code
   path — IBKR fills the runner LMT directly when price hits it, if
   enabled. Phase 4i: ``runner_target_enabled`` defaults to false, so the
   runner normally has no hard ceiling.
3. **Trailing exit on remaining half (post-scale)** — after scale-out,
   each new bar close is evaluated for two textbook runner exits:
   (a) dollar-based extension bar (``(high-open) × shares`` ≥
       max_loss × trigger multiple),
   (b) close below 9-EMA.
   The red-candle trigger is *suppressed* after scale-out (the methodology holds
   through a single red on the runner as long as breakeven holds);
   ``red_candle_exit_suppressed`` flips true on ``mark_scaled`` and
   ``_evaluate_trailing_exit`` short-circuits the (a) branch there.
   Any firing trigger → cancel the post-scale stop, market-sell the
   remainder, log ``trade_manager.trailing_exit`` with the reason,
   close the position, and journal the exit.

**Known behavior to monitor in Phase 5 paper-trading** — bar-close exits on
a 1-min chart are aggressive. professional discretionary traders trade that window but he's
reading the tape live; our detector may exit on a fake-out red candle that
reverses the next bar. We document that here rather than "tuning" with no
data. Phase 5 will revisit after 2+ weeks of paper runs.

Thread-safety: each symbol's handler reads its own row from PositionStore
and mutates only that row. No shared state across symbols. Orchestrator is
the sole caller of ``start_tracking`` / ``stop_tracking``.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
import structlog
from ib_async import MarketOrder

from bot.config import Settings, get_settings
from bot.execution.executor import apply_default_tif
from bot.execution.position_state import (
    InvalidPositionTransitionError,
    Position,
    PositionStore,
    PostScaleoutStopType,
)
from bot.exit_advisor.core.types import BarFinalizedEvent, PositionLike
from bot.exit_advisor.hook.apply import RecommendationApplier
from bot.exit_advisor.hook.registry import notify_event
from bot.indicators import ema, is_extension_bar_dollar

if TYPE_CHECKING:
    from bot.brokerage.ibkr_client import IBKRClient
    from bot.brokerage.market_data import BarStream, MarketData, TickStream
    from bot.execution.executor import Executor
    from bot.persistence.journal import Journal

_log = structlog.get_logger("bot.execution.trade_manager")

_EMA_LENGTH = 9


class TradeManager:
    """Drives scale-out + trailing exits for each ``open`` position.

    Subscribes to 1-min bars per symbol once the position is tracked and
    evaluates on every bar update. Bar-close logic only (no intrabar
    decisions) — the on-bar callback checks whether a new bar closed before
    acting.
    """

    def __init__(
        self,
        *,
        ibkr: IBKRClient,
        store: PositionStore,
        market_data: MarketData,
        executor: Executor,
        journal: Journal,
        settings: Settings | None = None,
    ) -> None:
        """Wire the dependencies; tracked-symbol map starts empty."""
        self._ibkr = ibkr
        self._store = store
        self._market_data = market_data
        self._executor = executor
        self._journal = journal
        self._settings = settings or get_settings()
        # Each symbol → (stream, last_bar_timestamp seen). The timestamp is
        # how we detect "new bar closed" without consuming intra-bar updates.
        self._tracked: dict[str, _TrackedSymbol] = {}
        # Phase 11 — applier is constructed lazily on first use because the
        # vast majority of sessions run with ``exit_advisor.enabled=false``
        # and never need it. Constructed against ``self`` so the applier
        # can route ``exit_full`` recommendations through
        # ``execute_advisor_exit`` without a circular import.
        self._advisor_applier: RecommendationApplier | None = None

    async def start_tracking(self, position: Position) -> None:
        """Subscribe to the position's bar feed + wire a per-bar callback.

        Idempotent — re-tracking an already-tracked symbol is a no-op.

        Phase 7.5: also subscribes to tick-by-tick ``Last`` prints so the
        scale-out target check can fire on the first qualifying trade
        instead of waiting for the next 1-min bar close (typical saving:
        0.1-0.3 s vs ~1-60 s). Post-scale trailing exits stay on the bar
        path — they're bar-close concepts (red-candle, 9-EMA, extension
        bar) and tick semantics don't apply.
        """
        symbol = position.symbol
        if symbol in self._tracked:
            return
        stream = await self._market_data.subscribe_bars(symbol)
        tracked = _TrackedSymbol(stream=stream, last_bar_time=None)
        self._tracked[symbol] = tracked

        def _on_update(bars: object, has_new_bar: bool) -> None:
            # We don't even snapshot bars here — just note that a new bar
            # closed. The caller polls ``on_bar_update`` from the orchestrator
            # loop on each tick, which keeps the async work off eventkit's
            # synchronous dispatch path.
            if has_new_bar:
                tracked.pending_new_bar = True

        stream._bar_list.updateEvent += _on_update  # noqa: SLF001 - private access for callback wiring
        await self._subscribe_tick_scale_out(position)
        _log.info("trade_manager.tracking", symbol=symbol)

    async def _subscribe_tick_scale_out(self, position: Position) -> None:
        """Phase 7.5: wire the tick-driven scale-out handler for ``position``.

        One subscription per symbol, cancelled after scale-out or position
        close. The handler is single-fire by construction via the
        ``scale_out_fired`` latch on the returned ``TickStream``.
        """
        symbol = position.symbol

        async def _on_tick(tick: Any) -> None:
            # TickByTickAllLast — tick.price is the trade print.
            fresh = self._store.get_active(symbol)
            if fresh is None or fresh.status != "open":
                return
            if fresh.scaled_out:
                return
            if tick.price < fresh.scale_out_price:
                return
            # Phase 7.5.1: the tick path is a BACKUP for the Phase 6.14
            # post-fill scale LMT. When the LMT is live on the exchange
            # (``bracket.scale_lmt is not None``), server-side fill is
            # faster and race-free — defer. Firing here while the LMT
            # is live would double-sell (LMT fills server-side, our
            # MKT sell lands at the same time, total 100% flat instead
            # of the intended 50%). The tick path only engages when the
            # LMT failed to place, was cancelled, or the entry type
            # never planted one (LMT / STP_LMT paths today).
            bracket = self._executor.active_trades.get(symbol)
            if bracket is not None and bracket.scale_lmt is not None:
                return
            tick_stream = self._tick_stream_for(symbol)
            if tick_stream is None or tick_stream.scale_out_fired:
                return
            tick_stream.scale_out_fired = True  # latch before awaiting
            _log.warning(
                "trade_manager.tick_scale_out_backup_fired",
                symbol=symbol,
                tick_price=tick.price,
                scale_out_price=fresh.scale_out_price,
                exchange=getattr(tick, "exchange", None),
                entry_order_type=self._settings.execution.entry_order_type,
                reason="scale_lmt_missing",
            )
            try:
                await self._execute_scale_out(fresh, fill_price=float(tick.price))
            except Exception as exc:  # noqa: BLE001 — one symbol must not break others
                _log.error(
                    "trade_manager.tick_scale_out_failed",
                    symbol=symbol,
                    error=str(exc),
                )
            finally:
                # Tick feed no longer needed — post-scale uses bar path.
                await self._market_data.unsubscribe_ticks(symbol)

        await self._market_data.subscribe_ticks(symbol, on_tick=_on_tick)

    def _tick_stream_for(self, symbol: str) -> TickStream | None:
        """Resolve the MarketData-owned ``TickStream`` for ``symbol`` (None if unsubscribed)."""
        # Reaches through to MarketData's private dict; tests inject a mock
        # MarketData, so we guard for the attribute being absent.
        ticks: dict[str, TickStream] | None = getattr(self._market_data, "_ticks", None)
        if ticks is None:
            return None
        return ticks.get(symbol)

    async def stop_tracking(self, symbol: str) -> None:
        """Stop following ``symbol``; leaves the bar subscription to be managed elsewhere.

        Phase 7.5: also cancels the tick-by-tick scale-out subscription if
        one is still active (scale-out may not have fired yet when the
        position is closed some other way — stop-out, manual flatten,
        15:55 auto-flatten).
        """
        self._tracked.pop(symbol, None)
        await self._market_data.unsubscribe_ticks(symbol)
        _log.info("trade_manager.untracked", symbol=symbol)

    def is_tracking(self, symbol: str) -> bool:
        """True iff ``start_tracking`` has been called and no matching stop."""
        return symbol in self._tracked

    async def poll(self) -> None:
        """Evaluate every tracked symbol against its latest bars.

        Called by the orchestrator loop each tick. Positions no longer in
        the store (closed, flattened) are untracked on the fly. Exceptions
        in one symbol's evaluation are logged but do not block the others.
        """
        for symbol in list(self._tracked):
            position = self._store.get_active(symbol)
            if position is None or position.status == "closed":
                await self.stop_tracking(symbol)
                continue
            if position.status != "open":
                # pending_entry → nothing to do yet; closing → let the close
                # handlers do their work.
                continue
            tracked = self._tracked[symbol]
            if not tracked.pending_new_bar:
                continue
            tracked.pending_new_bar = False
            bars = tracked.stream.bars
            try:
                await self.on_bar_update(position, bars)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not halt others
                _log.error(
                    "trade_manager.poll_symbol_failed",
                    symbol=symbol,
                    error=str(exc),
                )

    async def on_bar_update(self, position: Position, bars: pd.DataFrame) -> None:
        """Core bar-close evaluation: pre-scale guards, then scale-out, then trailing exits.

        Public so tests can drive it directly without setting up live
        subscriptions. Expects ``bars`` indexed by tz-aware timestamps with
        at least ``open``, ``high``, ``low``, ``close`` columns.

        Phase 11 — runs the exit-advisor hook *before* the bot's own
        bar-close exit logic. When the hook is enabled and an advisor
        is registered, a :class:`BarFinalizedEvent` is forwarded; an
        actionable recommendation that gets applied terminates the
        update early so the bot's existing rules don't fire on top.
        With ``exit_advisor.enabled=false`` (production main default),
        :func:`notify_event` returns immediately and the original
        bot-only path runs unchanged.
        """
        if bars.empty:
            return
        last_close = float(bars["close"].iloc[-1])

        # Phase 11 — fire the bar-finalized event into the advisor hook
        # before the bot's own bar-close exit logic. The ``enabled``
        # check short-circuits *before* constructing the BarFinalizedEvent
        # so the disabled-default path adds zero overhead and zero new
        # column dependencies on ``bars`` (production main + the
        # historical test fixtures that don't carry a volume column).
        # ``has_new_bar=True`` means iloc[-2] is the bar that just
        # finalized; iloc[-1] is the nascent next-minute bar. We feed
        # the advisor the just-closed bar so it reasons on settled
        # OHLCV, not the in-progress tick.
        if self._settings.exit_advisor.enabled and len(bars) >= 2:
            closed_idx = -2
            closed_ts = bars.index[closed_idx]
            bar_event = BarFinalizedEvent(
                timestamp=closed_ts.to_pydatetime()
                if hasattr(closed_ts, "to_pydatetime")
                else closed_ts,
                symbol=position.symbol,
                open=float(bars["open"].iloc[closed_idx]),
                high=float(bars["high"].iloc[closed_idx]),
                low=float(bars["low"].iloc[closed_idx]),
                close=float(bars["close"].iloc[closed_idx]),
                volume=float(bars["volume"].iloc[closed_idx]) if "volume" in bars.columns else 0.0,
            )
            response = notify_event(position, bar_event, settings=self._settings)
            if (
                response.is_actionable
                and self._settings.exit_advisor.hook_acts
                and response.recommendation is not None
            ):
                if self._advisor_applier is None:
                    self._advisor_applier = RecommendationApplier(self)
                acted = await self._advisor_applier.apply(
                    response.recommendation,
                    position,
                    exit_price=float(bar_event.close),
                )
                if acted:
                    return

        if not position.scaled_out:
            # Phase 7.6 (bot_driven mode) — replace the initial STP with a
            # plain TRAIL when the bar close crosses
            # ``entry + initial_stop_trigger_r_multiple × R``. This is the
            # opposite of the ``server_adjustable`` mode where IBKR handles
            # the conversion via encoded fields on the original STP — that
            # path is silently substituted with FIX PEGGED on SCM stocks
            # (2026-05-05 ENVB finding). bot_driven keeps every order at
            # its native type. Idempotent via ``initial_trail_planted``;
            # short-circuits unconditionally when mode != ``bot_driven``
            # so the server-side path retains its original behaviour.
            await self._maybe_plant_initial_trail(position, last_close=last_close)
            # Phase 4i: trigger is the signal-time anchor recorded on the
            # Position (entry + scale_out_multiple × initial_risk, default +2R).
            if last_close >= position.scale_out_price:
                await self._execute_scale_out(position, fill_price=last_close)
                return
            # Phase 7.8: the pre-scale red-candle exit. Fires when the
            # latest bar closed red AND below the prior bar's close. Catches
            # the "entry rolled over" scenario before the server-side STP
            # trips (paper TWS STPs can miss on a close-through without
            # hitting; this is the belt-and-suspenders backup).
            if self._settings.execution.pre_scale_red_candle_exit_enabled:
                red = _pre_scale_red_candle_fired(bars)
                if red is not None:
                    await self._execute_pre_scale_red_candle_exit(position, exit_price=red.close)
                    return
            return

        # Post-scale: evaluate trailing exit triggers on the just-closed bar.
        # Phase 7.9 — closed-bar semantics. ``last_close`` above is the
        # nascent next-minute bar's first-tick close; for the post-scale
        # exit path we want the bar that just finalized, which sits at
        # ``iloc[-2]`` when ``has_new_bar=True`` triggered this call.
        if len(bars) < 2:
            return
        closed_bar_close = float(bars["close"].iloc[-2])
        extension_threshold = (
            self._settings.risk.max_loss_per_trade_usd
            * self._settings.risk.extension_bar_trigger_multiple
        )
        trigger = _evaluate_trailing_exit(
            bars,
            entry_price=position.avg_price,
            position_shares=position.shares,
            extension_dollar_threshold=extension_threshold,
        )
        if trigger is not None:
            await self._execute_trailing_exit(
                position, exit_price=closed_bar_close, trigger=trigger
            )

    async def _maybe_plant_initial_trail(self, position: Position, *, last_close: float) -> None:
        """Phase 7.6 (bot_driven mode) — fire ``Executor.plant_initial_trail`` once at +R.

        Short-circuits in three cases (in order):

        1. ``initial_stop_trail_mode`` is not ``"bot_driven"`` —
           ``server_adjustable`` mode encoded the conversion server-side
           at placement time and the bot does not observe the trigger.
        2. ``position.initial_trail_planted`` is already True — the
           replacement TRAIL was planted on a prior bar close; this
           bar is post-trigger and should evaluate the runner-trail
           rules instead.
        3. The bar close has not yet reached
           ``position.avg_price + initial_stop_trigger_r_multiple ×
           initial_risk`` — trigger condition not met. The bot uses
           ``avg_price`` (the actual fill price) rather than
           ``signal.entry`` so the trigger reflects what was paid,
           not the strategy's intended entry. Materially identical
           on LMT fills at the limit; differs only on fast-market
           overruns where IBKR fills above the LMT.

        Wraps the executor call in ``try/except`` so a transient
        IBKR / qualify-stock failure logs and is retried on the next
        bar (the position is still ``open`` and the guard hasn't
        flipped, so the next bar's call repeats the attempt).
        """
        exec_cfg = self._settings.execution
        if exec_cfg.initial_stop_trail_mode != "bot_driven":
            return
        if position.initial_trail_planted:
            return
        if not exec_cfg.initial_stop_adjustable_enabled:
            # Master kill-switch: if Phase 7.6 is disabled entirely
            # neither mode does anything. Honour that here too so the
            # config has a single switch for "no +R trail at all".
            return
        initial_risk = position.avg_price - position.stop_price
        if initial_risk <= 0.0:
            return
        trigger_price = position.avg_price + initial_risk * exec_cfg.initial_stop_trigger_r_multiple
        if last_close < trigger_price:
            return
        try:
            await self._executor.plant_initial_trail(symbol=position.symbol, last_close=last_close)
        except Exception as exc:  # noqa: BLE001 - never let a TRAIL plant failure crash the bar loop
            _log.error(
                "trade_manager.plant_initial_trail_failed",
                symbol=position.symbol,
                last_close=last_close,
                trigger_price=round(trigger_price, 4),
                error=str(exc),
                hint="position.initial_trail_planted stays False; next bar retries.",
            )

    async def _execute_scale_out(self, position: Position, *, fill_price: float) -> None:
        """Cancel original bracket, market-sell 50%, install new post-scale stop on remainder.

        Cancels both the original STP and the target LMT: both were sized
        for the full share count; leaving either alive after a partial sell
        risks an over-sell (we'd be short) when the remaining leg fires.

        Phase 4h / 6.14: the replacement stop shape is driven by
        ``execution.post_scaleout_stop_mode``:

        * ``adjustable_to_trail`` (Phase 4h) — IBKR adjustable STP at
          breakeven that server-side converts to TRAIL at
          +``trail_activation_r_multiple`` R, OCA-linked with a fresh
          runner-target LMT when the runner is enabled.
        * ``immediate_trail`` (Phase 6.14 default) — IBKR TRAIL order
          planted immediately at ``scale_out - trail_amount``, follows
          the runner upward, no conversion wait.
        * ``static_breakeven`` (Phase 4e fallback) — plain STP at entry,
          TradeManager's bar-close exits drive the tail.
        """
        symbol = position.symbol
        # Defensive guard: a server-side OCA fill (stop or target) between the
        # bar close and this handler can push the position to closing/closed.
        # Don't issue a fresh market-sell on a position the account has already
        # exited — that would flip us short. Fresh state is cheap; stale state
        # is expensive.
        fresh = self._store.get_active(symbol)
        if fresh is None or fresh.status != "open":
            _log.info(
                "trade_manager.exit_skipped_position_inactive",
                symbol=symbol,
                status=fresh.status if fresh is not None else None,
                operation="scale_out",
            )
            return
        # Phase 6.14: when the MKT atomic bracket's scale-out LMT has
        # already fired (event-driven, via ``_handle_scale_out_lmt_fill``),
        # the position is already ``scaled_out`` and a second MKT-sell
        # from this bar-close path would flip us short on the remaining
        # half. This short-circuit makes TradeManager a belt-and-suspenders
        # fallback for paths that don't use the atomic scale-out LMT
        # (LMT / STP_LMT entries).
        if fresh.scaled_out:
            _log.info(
                "trade_manager.scale_out_skipped_already_scaled_via_lmt",
                symbol=symbol,
            )
            return
        shares_to_sell = position.shares // 2
        remaining = position.shares - shares_to_sell
        if shares_to_sell <= 0 or remaining <= 0:
            _log.warning(
                "trade_manager.scale_out_skipped_tiny_size",
                symbol=symbol,
                total_shares=position.shares,
            )
            return

        bracket = self._executor.active_trades.get(symbol)
        if bracket is not None:
            self._executor.cancel_trade_silently(bracket.stop)
            self._executor.cancel_trade_silently(bracket.target)

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.qualify_failed", symbol=symbol, error=str(exc))
            return

        sell_order = MarketOrder("SELL", shares_to_sell)
        sell_order.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(sell_order)
        scale_trade = self._ibkr.ib.placeOrder(contract, sell_order)
        self._executor.subscribe_commission(
            scale_trade,
            symbol=symbol,
            leg="scale",
            parent_order_id=position.parent_order_id,
        )

        initial_risk = position.avg_price - position.stop_price
        mode = self._settings.execution.post_scaleout_stop_mode
        new_target_trade = None
        stop_type: PostScaleoutStopType
        trigger_price: float | None = None
        if mode == "adjustable_to_trail" and initial_risk > 0.0:
            new_stop_trade, new_target_trade, trigger_price = (
                self._executor.place_adjustable_post_scaleout_stop(
                    contract=contract,
                    position=position,
                    remaining_shares=remaining,
                    initial_risk=initial_risk,
                )
            )
            stop_type = "adjustable_to_trail"
        elif mode == "immediate_trail" and initial_risk > 0.0:
            new_stop_trade = self._executor.place_immediate_trail_stop(
                contract=contract,
                position=position,
                remaining_shares=remaining,
                initial_risk=initial_risk,
            )
            stop_type = "immediate_trail"
        else:
            # static_breakeven, or a degenerate zero-initial-risk edge case
            # where trailing math would be nonsensical — fall back to the
            # flat breakeven STP.
            new_stop_trade = self._executor.place_static_breakeven_stop(
                contract=contract, position=position, remaining_shares=remaining
            )
            stop_type = "static_breakeven"

        partial_pnl = (fill_price - position.avg_price) * shares_to_sell
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
            _log.error("trade_manager.mark_scaled_failed", symbol=symbol, error=str(exc))
            return

        # Replace the executor's bracket reference. When adjustable-trail is
        # on the LMT is live (OCA runner ceiling); when off it's None.
        from bot.execution.executor import (
            _BracketTrades,  # noqa: PLC0415 - late import to avoid cycle
        )

        self._executor.active_trades[symbol] = _BracketTrades(
            parent=bracket.parent if bracket is not None else None,
            stop=new_stop_trade,
            target=new_target_trade,
        )
        _log.info(
            "trade_manager.scale_out",
            symbol=symbol,
            shares_sold=shares_to_sell,
            remaining=remaining,
            fill_price=round(fill_price, 2),
            partial_pnl=round(partial_pnl, 2),
            new_stop=position.avg_price,
            post_scaleout_stop_type=stop_type,
            adjustment_trigger_price=trigger_price,
        )

        # Fresh snapshot so the notifier renders the post-scale-out fields
        # written by ``mark_scaled`` above (the ``position`` argument is stale).
        updated = self._store.get_active(symbol)
        notifier = self._executor.notifier
        if updated is not None and notifier is not None:
            try:
                await notifier.send_fill(updated, "scale_out")
            except Exception as exc:  # noqa: BLE001 - notifications must never crash execution
                _log.error("trade_manager.notify_scale_out_failed", symbol=symbol, error=str(exc))

    async def _execute_pre_scale_red_candle_exit(
        self, position: Position, *, exit_price: float
    ) -> None:
        """Phase 7.8: market-close the full position on the pre-scale red-candle rule.

        Invariant: runs only before scale-out. Cancels both the OCA stop
        and (if MKT post-fill planted it) the half-size scale LMT, then
        market-sells ``position.shares`` (full size). The defensive guard
        re-reads ``fresh`` from the store — if the server-side STP has
        already fired between the bar close and this handler, the position
        is ``closing``/``closed`` and we short-circuit rather than
        double-sell.
        """
        symbol = position.symbol
        fresh = self._store.get_active(symbol)
        if fresh is None or fresh.status != "open":
            _log.info(
                "trade_manager.exit_skipped_position_inactive",
                symbol=symbol,
                status=fresh.status if fresh is not None else None,
                operation="pre_scale_red_candle",
            )
            return
        if fresh.scaled_out:
            # Defensive: scale LMT filled server-side between the bar close
            # and this handler. Post-scale has its own exit path.
            _log.info(
                "trade_manager.pre_scale_exit_skipped_already_scaled",
                symbol=symbol,
            )
            return

        bracket = self._executor.active_trades.get(symbol)
        if bracket is not None:
            self._executor.cancel_trade_silently(bracket.stop)
            self._executor.cancel_trade_silently(bracket.target)
            self._executor.cancel_trade_silently(bracket.scale_lmt)

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.qualify_failed", symbol=symbol, error=str(exc))
            return

        close_order = MarketOrder("SELL", position.shares)
        close_order.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(close_order)
        close_trade = self._ibkr.ib.placeOrder(contract, close_order)
        self._executor.subscribe_commission(
            close_trade,
            symbol=symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )

        # Pre-scale — no scale_partial_pnl component to fold in.
        total_pnl = (exit_price - position.avg_price) * position.shares

        with contextlib.suppress(InvalidPositionTransitionError):
            self._store.mark_closing(symbol, reason="pre_scale_red_candle")
        try:
            closed = self._store.mark_closed(
                symbol,
                exit_price=exit_price,
                pnl=total_pnl,
                closed_at=datetime.now(UTC),
            )
        except InvalidPositionTransitionError as exc:
            _log.error("trade_manager.mark_closed_failed", symbol=symbol, error=str(exc))
            return

        try:
            await self._journal.update_exit(
                closed,
                exit_price=exit_price,
                pnl=total_pnl,
                exit_type="pre_scale_red_candle",
            )
        except Exception as exc:  # noqa: BLE001 - journaling is observational
            _log.error("trade_manager.journal_update_failed", symbol=symbol, error=str(exc))

        history = self._store.symbol_history(symbol)
        history.record_exit(
            exit_time=closed.closed_at or datetime.now(UTC),
            pnl=total_pnl,
            exit_type="pre_scale_red_candle",
        )

        self._executor.active_trades.pop(symbol, None)
        await self.stop_tracking(symbol)
        try:
            await self._executor.risk_engine.on_fill_closed(closed, total_pnl)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.risk_on_fill_failed", symbol=symbol, error=str(exc))

        _log.warning(
            "trade_manager.pre_scale_red_candle_exit",
            symbol=symbol,
            exit_price=round(exit_price, 4),
            pnl=round(total_pnl, 2),
            shares=position.shares,
            avg_price=round(position.avg_price, 4),
        )

    async def _execute_trailing_exit(
        self, position: Position, *, exit_price: float, trigger: str
    ) -> None:
        """Cancel the breakeven stop, market-sell remainder, close + journal the position."""
        symbol = position.symbol
        # Defensive guard: the Phase 4h adjustable stop may have fired server-side
        # between this bar's close and the handler running. Re-read store state;
        # if we're no longer ``open`` the OCA winner already exited us.
        fresh = self._store.get_active(symbol)
        if fresh is None or fresh.status != "open":
            _log.info(
                "trade_manager.exit_skipped_position_inactive",
                symbol=symbol,
                status=fresh.status if fresh is not None else None,
                operation="trailing_exit",
            )
            return
        bracket = self._executor.active_trades.get(symbol)
        if bracket is not None:
            self._executor.cancel_trade_silently(bracket.stop)
            self._executor.cancel_trade_silently(bracket.target)

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.qualify_failed", symbol=symbol, error=str(exc))
            return

        close_order = MarketOrder("SELL", position.shares)
        close_order.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(close_order)
        close_trade = self._ibkr.ib.placeOrder(contract, close_order)
        self._executor.subscribe_commission(
            close_trade,
            symbol=symbol,
            leg="exit",
            parent_order_id=position.parent_order_id,
        )

        total_pnl = (exit_price - position.avg_price) * position.shares + position.scale_partial_pnl
        with contextlib.suppress(InvalidPositionTransitionError):
            self._store.mark_closing(symbol, reason=f"trailing_{trigger}")
        try:
            closed = self._store.mark_closed(
                symbol,
                exit_price=exit_price,
                pnl=total_pnl,
                closed_at=datetime.now(UTC),
            )
        except InvalidPositionTransitionError as exc:
            _log.error("trade_manager.mark_closed_failed", symbol=symbol, error=str(exc))
            return

        try:
            await self._journal.update_exit(
                closed,
                exit_price=exit_price,
                pnl=total_pnl,
                exit_type="scale_out_then_trail",
            )
        except Exception as exc:  # noqa: BLE001 - journaling is observational
            _log.error("trade_manager.journal_update_failed", symbol=symbol, error=str(exc))

        # Phase 4d — classify the exit so ``RiskEngine.check_reentry`` can see a
        # profitable scale-out-then-trail outcome. ``total_pnl`` already folds
        # the scale-out fill with the trailing close, so the profitable-prior
        # gate reads the correct sign even on a slightly-negative tail.
        history = self._store.symbol_history(symbol)
        history.record_exit(
            exit_time=closed.closed_at or datetime.now(UTC),
            pnl=total_pnl,
            exit_type="scale_out_then_trail",
        )

        self._executor.active_trades.pop(symbol, None)
        await self.stop_tracking(symbol)
        try:
            await self._executor.risk_engine.on_fill_closed(closed, total_pnl)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.risk_on_fill_failed", symbol=symbol, error=str(exc))

        _log.warning(
            "trade_manager.trailing_exit",
            symbol=symbol,
            trigger=trigger,
            exit_price=round(exit_price, 2),
            pnl=round(total_pnl, 2),
            scale_partial_pnl=round(position.scale_partial_pnl, 2),
        )

    async def execute_advisor_exit(
        self,
        position: PositionLike,
        *,
        exit_price: float,
        reason: str,
    ) -> bool:
        """Phase 11 — full-position market-close on advisor request.

        Mirrors the cancel-children → market-sell-remainder → mark-closed
        → journal/history pattern of the existing pre-scale and trailing
        exits, but classifies the exit as ``advisor_exit`` end-to-end so
        downstream analysis (re-entry gating, journal queries) can
        distinguish advisor-driven exits from the bot's own rules.

        Defensive guard: re-reads the position from the store. If the
        server-side stop fired between the bar close and this handler,
        the position is already ``closing``/``closed`` and we
        short-circuit rather than double-sell.

        Returns True iff a market-SELL order was submitted.
        """
        symbol = position.symbol
        fresh = self._store.get_active(symbol)
        if fresh is None or fresh.status != "open":
            _log.info(
                "trade_manager.exit_skipped_position_inactive",
                symbol=symbol,
                status=fresh.status if fresh is not None else None,
                operation="advisor_exit",
            )
            return False

        bracket = self._executor.active_trades.get(symbol)
        if bracket is not None:
            self._executor.cancel_trade_silently(bracket.stop)
            self._executor.cancel_trade_silently(bracket.target)
            self._executor.cancel_trade_silently(bracket.scale_lmt)

        try:
            contract = await self._ibkr.qualify_stock(symbol)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.qualify_failed", symbol=symbol, error=str(exc))
            return False

        close_order = MarketOrder("SELL", fresh.shares)
        close_order.outsideRth = not self._settings.execution.rth_only
        apply_default_tif(close_order)
        close_trade = self._ibkr.ib.placeOrder(contract, close_order)
        self._executor.subscribe_commission(
            close_trade,
            symbol=symbol,
            leg="exit",
            parent_order_id=fresh.parent_order_id,
        )

        # PnL math accounts for whether the position was already scaled
        # out (post-scale → fold in scale_partial_pnl) or pre-scale
        # (full size, no scale partial yet).
        total_pnl = (exit_price - fresh.avg_price) * fresh.shares + fresh.scale_partial_pnl

        with contextlib.suppress(InvalidPositionTransitionError):
            self._store.mark_closing(symbol, reason=f"advisor_exit:{reason}")
        try:
            closed = self._store.mark_closed(
                symbol,
                exit_price=exit_price,
                pnl=total_pnl,
                closed_at=datetime.now(UTC),
            )
        except InvalidPositionTransitionError as exc:
            _log.error("trade_manager.mark_closed_failed", symbol=symbol, error=str(exc))
            return False

        try:
            await self._journal.update_exit(
                closed,
                exit_price=exit_price,
                pnl=total_pnl,
                exit_type="advisor_exit",
            )
        except Exception as exc:  # noqa: BLE001 - journaling is observational
            _log.error("trade_manager.journal_update_failed", symbol=symbol, error=str(exc))

        history = self._store.symbol_history(symbol)
        history.record_exit(
            exit_time=closed.closed_at or datetime.now(UTC),
            pnl=total_pnl,
            exit_type="advisor_exit",
        )

        self._executor.active_trades.pop(symbol, None)
        await self.stop_tracking(symbol)
        try:
            await self._executor.risk_engine.on_fill_closed(closed, total_pnl)
        except Exception as exc:  # noqa: BLE001
            _log.error("trade_manager.risk_on_fill_failed", symbol=symbol, error=str(exc))

        _log.warning(
            "trade_manager.advisor_exit",
            symbol=symbol,
            reason=reason,
            exit_price=round(exit_price, 4),
            pnl=round(total_pnl, 2),
            scale_partial_pnl=round(fresh.scale_partial_pnl, 2),
            scaled_out_at_exit=fresh.scaled_out,
        )
        return True


class _TrackedSymbol:
    """Per-symbol bookkeeping used by ``TradeManager.poll`` to coalesce updates."""

    __slots__ = ("last_bar_time", "pending_new_bar", "stream")

    def __init__(self, stream: BarStream, last_bar_time: datetime | None) -> None:
        """Hold the underlying bar stream + a 'new bar pending' latch."""
        self.stream = stream
        self.last_bar_time = last_bar_time
        self.pending_new_bar = True  # first poll always evaluates

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        """Short repr for structlog bindings."""
        return f"_TrackedSymbol(pending={self.pending_new_bar})"


class _RedCandleBar(NamedTuple):
    """Phase 7.8 — the just-closed bar's OHLC when the red-candle rule fires."""

    open: float
    close: float
    prev_close: float


def _pre_scale_red_candle_fired(bars: pd.DataFrame) -> _RedCandleBar | None:
    """Phase 7.8 — does the just-closed bar meet the red-candle close?

    Returns the ``_RedCandleBar`` of the firing bar, or ``None`` if the
    check doesn't apply (insufficient history) or doesn't fire.

    When ``TradeManager.poll`` runs it was triggered by ``has_new_bar=True``
    — so ``bars.iloc[-1]`` is the freshly-appended next-minute bar
    (1 tick of data) and ``bars.iloc[-2]`` is the bar that just finalized.
    We evaluate on ``[-2]`` (just-closed) against ``[-3]``'s close
    (prior finalized). Matches Phase 7.4 "evaluate closed bars" semantics
    — never fires on an in-progress bar's transient close.

    Fires iff:
      * ``bars.iloc[-2].close < bars.iloc[-2].open`` (red body), AND
      * ``bars.iloc[-2].close < bars.iloc[-3].close``   (close below prior).
    """
    if len(bars) < 3:
        return None
    closed = bars.iloc[-2]
    prior = bars.iloc[-3]
    closed_open = float(closed["open"])
    closed_close = float(closed["close"])
    prior_close = float(prior["close"])
    if closed_close >= closed_open:
        return None  # not red
    if closed_close >= prior_close:
        return None  # red body but closed above prior close
    return _RedCandleBar(
        open=closed_open,
        close=closed_close,
        prev_close=prior_close,
    )


def _evaluate_trailing_exit(
    bars: pd.DataFrame,
    entry_price: float,
    *,
    position_shares: int = 0,
    extension_dollar_threshold: float = 0.0,
) -> str | None:
    """Return the name of the firing post-scale runner exit (or None).

    Checks, in priority order:

    * ``extension_bar`` — the just-closed bar's unrealized gain
      (``(high - open) * shares``) clears
      ``extension_dollar_threshold`` (default $max_loss × 2). the "$200-$400 spike" rule.
    * ``ema_break`` — the just-closed bar's close is below the 9-EMA
      **and** below entry. The below-entry gate avoids whipping out on
      early-session EMA noise above entry.

    Phase 7.9 — closed-bar semantics. When ``TradeManager.poll`` fires
    this function it was triggered by ``has_new_bar=True`` on the
    underlying stream, which means IBKR appended a brand-new
    next-minute bar at ``bars.iloc[-1]`` with ~1 tick of data. The
    evaluation target is therefore ``bars.iloc[-2]`` (the bar that
    just finalized), and the EMA is computed over ``bars.iloc[:-1]``
    to drop the nascent bar's transient close from the series.

    Pre-Phase-7.9 this looked at ``iloc[-1]``, which meant in
    production: extension never fired (1-tick bars have
    ``high ≈ open``) and the EMA-break check read the nascent close
    against an EMA that included its own transient value.

    Red-candle exit is handled separately: the pre-scale path lives
    at ``_pre_scale_red_candle_fired`` (Phase 7.8), and the post-
    scale runner rule suppresses it (hold through a single red as
    long as breakeven holds). So this function no longer has a
    red-candle branch.

    ``position_shares`` / ``extension_dollar_threshold`` default to 0
    so callers that don't care about the extension check can opt out.
    """
    if len(bars) < 2:
        return None
    closed_bar = bars.iloc[-2]
    closed_close = float(closed_bar["close"])

    # (a) Dollar-denominated extension bar on the just-closed bar.
    if is_extension_bar_dollar(closed_bar, position_shares, extension_dollar_threshold):
        return "extension_bar"

    # (b) Close below 9-EMA on the just-closed bar. Compute EMA over
    # the series *excluding* the nascent trailing bar so the latest
    # EMA value reflects finalized data only.
    closed_series = bars["close"].iloc[:-1]
    if len(closed_series) >= _EMA_LENGTH:
        ema_series = ema(closed_series, length=_EMA_LENGTH)
        if not ema_series.empty:
            last_ema = float(ema_series.iloc[-1])
            if closed_close < last_ema and closed_close < entry_price:
                return "ema_break"

    return None


__all__ = ["TradeManager"]
