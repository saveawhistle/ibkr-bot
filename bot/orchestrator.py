"""Strategy loop + Phase 4b session wiring (RiskEngine + TradeManager + auto-flatten).

Two public surfaces:

* ``run_strategy_loop`` — the Phase 3/4a detector+executor loop. Still the
  single-entry-point used by the ``watch`` command (detector-only) and
  now extended so the ``trade`` command can pass a ``TradeManager`` that
  gets tracked per fresh ``open`` position and polled every tick.

* ``Orchestrator`` — the Phase 4b session owner. Wraps the loop + halt
  check + the 15:55 ET auto-flatten scheduler. ``Orchestrator.run`` is
  what ``cli.trade`` calls; it also exposes ``flatten_all_active`` so
  tests can drive the scheduler callback directly without spinning an
  event loop to wall-clock 15:55.

The auto-flatten is a hard rule: it fires **even when the RiskEngine is
halted**. A halt blocks new entries, not open positions. Leaving a live
share lot overnight is a bigger risk than any halt signal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from bot.brokerage.market_data import BarStream, MarketData
from bot.config import Settings, get_settings
from bot.execution.executor import Executor
from bot.scanning.scanner import ScanHit
from bot.signal_bus import SignalBus
from bot.strategies.base import Signal, Strategy
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy

if TYPE_CHECKING:
    from bot.execution.position_state import PositionStore
    from bot.execution.trade_manager import TradeManager
    from bot.execution.watchdog import Watchdog
    from bot.notify import Notifier
    from bot.risk.rehab import RehabEngine
    from bot.scanning.scanner import IBKRScanner

_log = structlog.get_logger("bot.orchestrator")

# Phase 6.1: per-symbol consecutive-stall count at which ``orchestrator.loop_stall``
# (INFO) upgrades to ``orchestrator.persistent_stall`` (WARNING). Three in a row
# on a 60-second poll means three minutes of silent stalling — operator-visible.
_PERSISTENT_STALL_THRESHOLD = 3

# Phase 6.4: how often the loop sweeps ``last_evaluated_bar_ts`` for entries
# whose symbol is no longer subscribed. Keeps dict growth bounded over very
# long sessions where many unique symbols cycle through the watchlist.
# Correctness-irrelevant — the cursor check already no-ops when a symbol's
# bar advances past the recorded timestamp; this just prevents unbounded
# memory use across a multi-week runtime.
_CURSOR_SWEEP_INTERVAL = 100


async def _fetch_bars(stream: BarStream) -> pd.DataFrame:
    """Per-symbol bar-snapshot hook; async so it composes with ``asyncio.wait_for``.

    Reading ``stream.bars`` is a dataclass attribute — cheap, non-blocking —
    but routing it through an async helper gives the orchestrator a single
    wrap point for timeouts and lets tests simulate an IBKR-side stall by
    monkeypatching this function. Any future per-iteration IBKR bar fetch
    (level 2, tape, higher-timeframe confirmations) should flow through the
    same seam so timeouts apply uniformly.
    """
    return stream.bars


async def _apply_watchlist_diff(
    new_hits: list[ScanHit],
    *,
    streams: dict[str, BarStream],
    market_data: MarketData,
    position_store: PositionStore | None,
    max_size: int,
    bar_source: str = "ibkr_1min",
    on_symbol_dropped: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Reconcile ``streams`` with the current scan via one declarative diff; return (added, removed).

    Phase 6.4 replaces the Phase 6.2 procedural add-then-evict logic.
    That version subscribed new symbols one at a time and evicted the
    oldest non-position subscription each time the cap was exceeded —
    which cap-thrashed when the scanner returned more symbols than
    ``max_size``: a symbol at scan rank 3 could be subscribed early in
    the add pass and then evicted later in the same call when a scan-
    rank-11 symbol was processed. BATL at rank 3 got subscribed + evicted
    every rescan tick.

    The declarative form computes the survivor set first, then applies
    the diff once:

      * ``target = top-N scan ∪ active positions``, capped at ``max_size``
        with active positions always winning (non-negotiable — a symbol
        we hold must keep its bars).
      * Each symbol is subscribed or unsubscribed **at most once** per
        call; the cap is never transiently exceeded during the diff.
      * When every slot holds an active position, new scan hits are
        rejected with a single WARNING event — the operator sees the
        rejection count without per-symbol spam.

    ``position_symbols`` is scoped to the currently-subscribed set; we
    never resurrect a stored-but-unsubscribed position here (that's
    reconcile's job on startup).
    """
    new_symbols_ordered = [hit.symbol for hit in new_hits]
    desired_top_n = new_symbols_ordered[:max_size]
    desired_set = set(desired_top_n)

    position_symbols: set[str] = set()
    if position_store is not None:
        position_symbols = {s for s in streams if position_store.has_active(s)}

    target: set[str] = desired_set | position_symbols
    if len(target) > max_size:
        remaining_slots = max_size - len(position_symbols)
        if remaining_slots > 0:
            scan_keepers = [s for s in desired_top_n if s not in position_symbols][:remaining_slots]
            target = set(position_symbols) | set(scan_keepers)
        else:
            target = set(position_symbols)
            _log.warning(
                "orchestrator.watchlist_full_no_eviction_possible",
                new_scan_hits_rejected=len(new_symbols_ordered),
                watchlist_size=len(streams),
                max_size=max_size,
            )

    # Observability: position protection is only load-bearing when the
    # symbol would otherwise have been dropped (i.e., not in the scan).
    # Emitting per-rescan keeps the JSONL audit trail consistent with
    # Phase 6.2 so operators can grep for active-book continuity.
    for symbol in position_symbols:
        if symbol not in desired_set:
            _log.info("orchestrator.watchlist_kept_for_position", symbol=symbol)

    current = set(streams.keys())
    to_drop = [s for s in streams if s not in target]
    to_add = [s for s in desired_top_n if s in target and s not in current]

    removed: list[str] = []
    for symbol in to_drop:
        await market_data.unsubscribe(symbol)
        streams.pop(symbol, None)
        _log.info("orchestrator.watchlist_symbol_dropped", symbol=symbol, reason="not_in_scan")
        if on_symbol_dropped is not None:
            try:
                on_symbol_dropped(symbol)
            except Exception as exc:  # noqa: BLE001 - hook bug must not break diff
                _log.error(
                    "orchestrator.on_symbol_dropped_failed",
                    symbol=symbol,
                    error=str(exc),
                )
        removed.append(symbol)

    added: list[str] = []
    for symbol in to_add:
        # Phase 10.4 — dispatch to the configured bar source. Rescan-added
        # symbols don't receive the orchestrator's event-driven on_new_bar
        # callback (pre-existing limitation; they're picked up by the next
        # poll iteration's ``streams.items()`` sweep). The bar-source
        # choice still applies so all subscribed symbols use the same
        # finalization path.
        if bar_source == "rtbars_5sec_aggregated":
            stream = await market_data.subscribe_bars_5sec_aggregated(symbol)
        else:
            stream = await market_data.subscribe_bars(symbol)
        streams[symbol] = stream
        _log.info("orchestrator.watchlist_symbol_added", symbol=symbol)
        added.append(symbol)

    return added, removed


def _flush_file_handlers() -> None:
    """Flush every ``FileHandler`` on the root logger.

    Day-2 paper trading showed post-stall events can be lost if the process
    is killed mid-hang: default ``FileHandler`` buffering holds records in
    memory until flush. Flushing at each iteration boundary guarantees the
    last forensic event (``orchestrator.loop_stall``, ``loop_iteration``) is
    on disk before the next — potentially wedging — operation starts.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()


@dataclass
class StrategyLoopResult:
    """Return payload of ``run_strategy_loop`` — signals collected during the run."""

    signals: list[Signal]


def build_default_strategies(settings: Settings | None = None) -> list[Strategy]:
    """Return the Phase 3 default strategy set filtered by ``enabled`` flags.

    Phase 6.6 — both strategies receive ``extended_from_vwap_atr_multiple``
    from per-strategy config (calibrated default 5.0) and a shared
    ``log_extension_check_passes`` toggle from ``StrategiesConfig``.
    """
    s = settings or get_settings()
    log_passes = s.strategies.log_extension_check_passes
    # Phase 10.2 — stop-distance floor is shared across both strategies
    # (the ZENA pathology is a property of the breakout pattern, not the
    # per-strategy stop reference). Hoist out of the per-strategy
    # constructors so any future consumer reads from one place.
    floor_min_abs = s.strategies.stop_floor.min_abs
    floor_min_pct = s.strategies.stop_floor.min_pct
    strategies: list[Strategy] = []
    if s.strategies.gap_and_go.enabled:
        trading_start_h, trading_start_m = _parse_hh_mm(s.session.trading_start)
        gng_cfg = s.strategies.gap_and_go
        gng_end_h, gng_end_m = _parse_hh_mm(gng_cfg.window_end)
        strategies.append(
            GapAndGoStrategy(
                vwap_extension_grace_minutes=gng_cfg.vwap_extension_grace_minutes,
                trading_start=time(trading_start_h, trading_start_m),
                extended_from_vwap_atr_multiple=gng_cfg.extended_from_vwap_atr_multiple,
                log_extension_check_passes=log_passes,
                window_end=time(gng_end_h, gng_end_m),
                premarket_high_cap_enabled=s.strategies.premarket_high_cap_enabled,
                stop_floor_min_abs=floor_min_abs,
                stop_floor_min_pct=floor_min_pct,
            )
        )
    if s.strategies.momentum.enabled:
        mom_cfg = s.strategies.momentum
        mom_end_h, mom_end_m = _parse_hh_mm(mom_cfg.window_end)
        strategies.append(
            MomentumStrategy(
                flag_max_pullback_pct=mom_cfg.flag_max_pullback_pct,
                extended_from_vwap_atr_multiple=mom_cfg.extended_from_vwap_atr_multiple,
                log_extension_check_passes=log_passes,
                window_end=time(mom_end_h, mom_end_m),
                premarket_high_cap_enabled=s.strategies.premarket_high_cap_enabled,
                stop_floor_min_abs=floor_min_abs,
                stop_floor_min_pct=floor_min_pct,
            )
        )
    return strategies


async def run_strategy_loop(
    watchlist: list[ScanHit],
    market_data: MarketData,
    signal_bus: SignalBus,
    *,
    strategies: list[Strategy] | None = None,
    executor: Executor | None = None,
    trade_manager: TradeManager | None = None,
    rehab_engine: RehabEngine | None = None,
    notifier: Notifier | None = None,
    watchdog: Watchdog | None = None,
    rehab_check_interval_seconds: float = 600.0,
    duration_minutes: float | None = None,
    poll_interval: float = 5.0,
    settings: Settings | None = None,
    shutdown_event: asyncio.Event | None = None,
    scanner: IBKRScanner | None = None,
    position_store: PositionStore | None = None,
    evaluate_on_closed_bar_only: bool = True,
) -> StrategyLoopResult:
    """Evaluate strategies against the watchlist's live bars until the session deadline.

    Phase 5.1: ``duration_minutes=None`` (the CLI default) derives the
    deadline from today's ``session.flatten_all`` in NY-local time minus a
    60-second safety buffer — i.e., the loop naturally runs to the end of
    the session instead of capping at the prior 120-minute default that
    caused the 2026-04-20 early exit at 09:58 ET. When a caller passes an
    explicit ``duration_minutes`` (tests, operational override via
    ``--duration``), that value wins and is treated exactly like before.

    If ``duration_minutes is None`` and ``flatten_all`` has already passed
    on today's NY clock, the loop exits immediately with event
    ``orchestrator.flatten_all_already_passed`` — subscribing bars for a
    session that's over would burn rate-limit without any chance of
    emitting a signal.

    Duplicate signals for the same (symbol, strategy, bar-timestamp) are
    suppressed. When both strategies fire on one bar the bus' ``put_batch``
    collapses them to the Gap-and-Go winner.

    If ``executor`` is provided each published signal is handed to
    ``handle_signal`` in the same task. If ``trade_manager`` is also
    provided, after every poll we:

    * ``start_tracking`` any newly-``open`` position the executor opened
      during the pass (drives scale-out + trailing-exit on that symbol).
    * ``poll()`` the trade manager so bar-close triggers fire.

    Halted sessions: the RiskEngine blocks new entries at the executor
    boundary; the loop itself doesn't peek at the halt flag — it's always
    the same path whether we're halted or not. That keeps the loop small
    and puts the policy in exactly one place.
    """
    strategies = strategies or build_default_strategies(settings)
    if not strategies:
        _log.warning("orchestrator.no_strategies_enabled")
        return StrategyLoopResult(signals=[])
    if not watchlist:
        # Phase 9.3: when a scanner is wired (production CLI flow), an empty
        # initial scan is transient — the rescan tick can populate the
        # watchlist mid-session. Without a scanner (most unit tests, the
        # ``--dry-run-signal`` CLI path) there is no path to candidates, so
        # short-circuit to avoid an idle spin.
        if scanner is None:
            _log.info("orchestrator.empty_watchlist")
            return StrategyLoopResult(signals=[])
        _log.warning(
            "orchestrator.empty_initial_scan",
            message="initial scan returned no candidates; awaiting periodic rescan",
            rescan_interval_seconds=(
                settings.session.watchlist_rescan_interval_seconds if settings is not None else None
            ),
        )

    resolved_settings = settings or get_settings()
    tz = ZoneInfo(resolved_settings.session.timezone)
    trading_start_h, trading_start_m = _parse_hh_mm(resolved_settings.session.trading_start)

    # Phase 5.1: derive duration from session.flatten_all when not specified.
    # The AutoFlattenScheduler fires at flatten_all itself; our 60-second
    # buffer lets the loop exit cleanly before flatten runs so the two
    # aren't racing for the same positions.
    if duration_minutes is None:
        duration_seconds = _derive_duration_seconds_to_flatten(resolved_settings, datetime.now(tz))
        if duration_seconds <= 0:
            _log.warning(
                "orchestrator.flatten_all_already_passed",
                flatten_all=resolved_settings.session.flatten_all,
                timezone=resolved_settings.session.timezone,
            )
            return StrategyLoopResult(signals=[])
        deadline_source = "flatten_all"
    else:
        duration_seconds = duration_minutes * 60
        deadline_source = "explicit_duration"

    streams: dict[str, BarStream] = {}
    _log.info(
        "orchestrator.loop_started",
        symbols=list(streams),
        strategies=[s.name for s in strategies],
        duration_minutes=round(duration_seconds / 60.0, 2),
        deadline_source=deadline_source,
        executor_wired=executor is not None,
        trade_manager_wired=trade_manager is not None,
    )
    seen: set[tuple[str, str, str]] = set()
    collected: list[Signal] = []
    # Phase 4d session-start latch: reset per-symbol re-entry state once when
    # NY wall-clock first crosses the trading-start boundary. If the loop
    # already launches inside RTH, startup's rebuild-from-journal already
    # reconstructed today's state — skip the reset so we don't wipe it.
    session_reset_fired = _is_at_or_past_ny(datetime.now(tz), trading_start_h, trading_start_m)

    # Phase 6.1 stall tracking: per-symbol consecutive stall count. Resets to
    # zero the first time a bar fetch for that symbol returns within the
    # timeout. The outer iteration index is carried on every stall event so
    # post-session analysis can correlate a specific stall with surrounding
    # log context.
    op_timeout = resolved_settings.session.loop_operation_timeout_seconds
    stall_counts: dict[str, int] = {}
    iteration_index = 0

    # Phase 6.2 rescan state: in-flight scan task + last-kicked-off timestamp.
    # Kicking off via ``asyncio.create_task`` keeps the scan concurrent with
    # bar evaluation; the diff applies on the iteration the task completes.
    rescan_interval = float(resolved_settings.session.watchlist_rescan_interval_seconds)
    max_watchlist_size = resolved_settings.session.watchlist_max_size
    rescan_task: asyncio.Task[list[ScanHit]] | None = None
    rescan_enabled = scanner is not None

    # Phase 6.4: per-(symbol, strategy) last-evaluated-bar cursor. A key is
    # entered the first time a strategy evaluates a given bar (accepted or
    # rejected by a strategy gate); on the next iteration with the same bar
    # timestamp, evaluation is skipped entirely — no log, no call. This
    # prevents the Day-2 paper-trading pathology where a halted symbol's
    # last bar was re-evaluated every poll (e.g., 11×/minute on a 5.5s
    # poll), flooding the JSONL with ``signal.rejected`` events for a bar
    # that was never going to pass anyway. The cursor is loop-local; it
    # survives resubscribe cycles (key is on symbol+strategy, not stream
    # identity) so backfilled bars after a resubscribe don't re-fire.
    last_evaluated_bar_ts: dict[tuple[str, str], pd.Timestamp] = {}
    bar_staleness_threshold = int(resolved_settings.session.bar_staleness_threshold_seconds)

    async def _evaluate_symbol(
        symbol: str, bars: pd.DataFrame, latest_bar_ts: pd.Timestamp | None
    ) -> None:
        """Strategy evaluation + dispatch for one symbol's latest bar.

        Shared between the poll-driven path (fetched under stall-timeout)
        and the Phase 7.3 event-driven path (fired by ``has_new_bar=True``
        from IBKR). The cursor + ``seen`` dedup make concurrent invocations
        idempotent: whichever path stamps the cursor first wins, the other
        short-circuits.
        """
        if bars.empty:
            return
        if executor is not None and latest_bar_ts is not None:
            executor.expire_unfilled_entry(symbol, latest_bar_ts.to_pydatetime())
        batch: list[Signal] = []
        for strategy in strategies:
            if latest_bar_ts is not None:
                cursor_key = (symbol, strategy.name)
                last_seen = last_evaluated_bar_ts.get(cursor_key)
                if last_seen is not None and latest_bar_ts <= last_seen:
                    continue
                staleness_seconds = (
                    datetime.now(tz) - latest_bar_ts.to_pydatetime()
                ).total_seconds()
                if staleness_seconds > bar_staleness_threshold:
                    _log.info(
                        "strategy.bar_stale",
                        symbol=symbol,
                        strategy=strategy.name,
                        latest_bar_ts=latest_bar_ts.isoformat(),
                        staleness_seconds=round(staleness_seconds, 2),
                        threshold_seconds=bar_staleness_threshold,
                    )
                    last_evaluated_bar_ts[cursor_key] = latest_bar_ts
                    continue
            signal = strategy.evaluate(symbol, bars)
            if latest_bar_ts is not None:
                last_evaluated_bar_ts[(symbol, strategy.name)] = latest_bar_ts
            if signal is None:
                continue
            # Phase 8.1: post-emission R:R filter removed. Scale-out is
            # constructed as entry + scale_out_multiple × risk, so R:R is
            # pinned to scale_out_multiple by definition; the old check
            # against a matching rr_min floor was a tautology whose only
            # observed effect was floating-point-drift rejections.
            # Risk-engine and sizing continue to gate downstream.
            key = (symbol, strategy.name, signal.timestamp.isoformat())
            if key in seen:
                continue
            seen.add(key)
            batch.append(signal)
        if not batch:
            return
        await signal_bus.put_batch(batch)
        winners = _pick_per_bar_winners(batch)
        collected.extend(winners)
        if executor is not None:
            for winner in winners:
                await executor.handle_signal(winner)

    def _latest_bar_ts(bars: pd.DataFrame) -> pd.Timestamp | None:
        """Phase 7.3: shared helper for both paths. NY-tz timestamp of the latest bar, or None.

        Tests use integer-indexed frames which return None here; the eval
        closure then skips the cursor + staleness branches entirely.
        """
        if not isinstance(bars.index, pd.DatetimeIndex) or len(bars.index) == 0:
            return None
        raw_ts = bars.index[-1]
        if raw_ts.tzinfo is None:
            return raw_ts.tz_localize(tz)
        return raw_ts.tz_convert(tz)

    def _bars_for_evaluation(bars: pd.DataFrame) -> pd.DataFrame:
        """Phase 7.4: drop the in-progress trailing bar in live evaluation.

        IBKR's ``keepUpToDate=True`` keeps ``bars[-1]`` live-updating until
        the minute rolls — at which point a new in-progress bar is
        appended and the previous ``bars[-1]`` (now at ``bars[-2]``) has
        its final OHLCV frozen. Strategies are bar-close strategies, so
        they must evaluate the just-closed bar (``iloc[:-1]``'s last row)
        rather than the freshly-started one. Single-row frames are passed
        through unchanged so synthetic-frame tests that don't model the
        in-progress bar still work when they opt-out of this behaviour.
        """
        if evaluate_on_closed_bar_only and len(bars) >= 2:
            return bars.iloc[:-1]
        return bars

    async def _on_new_bar_for(symbol: str) -> None:
        """Phase 7.3 event-driven entrypoint — IBKR rolled a new bar for ``symbol``.

        Fires in the same event-loop turn the ``has_new_bar=True`` update
        arrives, eliminating the poll-wait component of end-to-end latency.
        The poll path stays as a backstop; the cursor prevents duplicate
        evaluation if the poll reaches the same bar later.

        Phase 7.4: the incoming frame's trailing bar is the freshly-started
        next-minute bar (just-appended by IBKR). ``_bars_for_evaluation``
        slices it off so strategies see the just-closed bar at ``iloc[-1]``.
        """
        stream = streams.get(symbol)
        if stream is None:
            return
        eval_bars = _bars_for_evaluation(stream.bars)
        await _evaluate_symbol(symbol, eval_bars, _latest_bar_ts(eval_bars))

    # Phase 10.4 — bar-source dispatch. ``ibkr_1min`` keeps the pre-10.4
    # ``reqHistoricalData(keepUpToDate=True)`` path; ``rtbars_5sec_aggregated``
    # routes through the new ``RollingMinuteAggregator`` for ~5 s lower
    # bar-finalization latency. See ``session.bar_source`` in config.
    bar_source = resolved_settings.session.bar_source

    for hit in watchlist:

        def _make_callback(sym: str) -> Callable[[], Coroutine[Any, Any, None]]:
            async def _cb() -> None:
                await _on_new_bar_for(sym)

            return _cb

        if bar_source == "rtbars_5sec_aggregated":
            streams[hit.symbol] = await market_data.subscribe_bars_5sec_aggregated(
                hit.symbol, on_new_bar=_make_callback(hit.symbol)
            )
        else:
            streams[hit.symbol] = await market_data.subscribe_bars(
                hit.symbol, on_new_bar=_make_callback(hit.symbol)
            )

    deadline = asyncio.get_event_loop().time() + duration_seconds
    last_rescan_at = asyncio.get_event_loop().time()
    last_rehab_check = asyncio.get_event_loop().time()
    # Phase 10.1 — start the Telegram ack listener so the watchdog's alerts
    # carry an actionable Ack button. Notifier no-ops when credentials are
    # missing or an injected (test) bot is in use; lifecycle is paired with
    # the loop so a Ctrl-C / shutdown_event triggers stop_ack_listener via
    # the finally clause below.
    if notifier is not None and watchdog is not None:
        await notifier.start_ack_listener()
    try:
        while asyncio.get_event_loop().time() < deadline:
            iteration_index += 1
            if shutdown_event is not None and shutdown_event.is_set():
                _log.warning("orchestrator.shutdown_requested")
                break

            # Phase 9.6 — drop any symbols the risk engine has just locked out
            # for repeated broker auto-cancels. Runs once per loop iteration so
            # a symbol that crossed the threshold mid-bar gets unsubscribed
            # within ``poll_interval`` of the lockout. The watchlist_dropped
            # event was already emitted by the executor when the threshold
            # crossed; here we just clean up the market-data subscription.
            if executor is not None:
                blocked = set(executor.risk_engine.state.blocked_symbols)
                for symbol in blocked & set(streams):
                    await market_data.unsubscribe(symbol)
                    streams.pop(symbol, None)
                    stall_counts.pop(symbol, None)
                    last_evaluated_bar_ts = {
                        k: v for k, v in last_evaluated_bar_ts.items() if k[0] != symbol
                    }

            if not session_reset_fired and _is_at_or_past_ny(
                datetime.now(tz), trading_start_h, trading_start_m
            ):
                if executor is not None:
                    executor.store.reset_symbol_histories()
                _log.info(
                    "session.symbol_histories_reset",
                    trading_start=resolved_settings.session.trading_start,
                    timezone=resolved_settings.session.timezone,
                )
                session_reset_fired = True

            # Phase 6.2 rescan tick. Two parts:
            # 1. If a previously-kicked scan finished, apply its diff now.
            # 2. If the interval has elapsed and nothing is in flight, kick
            #    off a new scan via create_task so bar evaluation for the
            #    current iteration is not delayed by the Finnhub round trip.
            if rescan_enabled and scanner is not None:
                if rescan_task is not None and rescan_task.done():
                    try:
                        new_hits = rescan_task.result()
                    except Exception as exc:  # noqa: BLE001 — rescan faults must not halt the loop
                        _log.warning("orchestrator.rescan_failed", error=str(exc))
                        new_hits = []
                    rescan_task = None
                    if new_hits:
                        # Phase 12: forward watchlist drops to the LLM
                        # catalyst classifier so a re-entered ticker
                        # gets fresh evaluation. ``getattr`` shields
                        # tests that pass a bare-mocked scanner without
                        # the ``on_watchlist_removal`` method.
                        on_drop = getattr(scanner, "on_watchlist_removal", None)
                        added, removed = await _apply_watchlist_diff(
                            new_hits,
                            streams=streams,
                            market_data=market_data,
                            position_store=position_store,
                            max_size=max_watchlist_size,
                            bar_source=bar_source,
                            on_symbol_dropped=on_drop,
                        )
                        for symbol in removed:
                            stall_counts.pop(symbol, None)
                        _log.info(
                            "orchestrator.watchlist_rescanned",
                            scan_count=len(new_hits),
                            current_watchlist_size=len(streams),
                            symbols_added=added,
                            symbols_removed=removed,
                        )
                if (
                    rescan_task is None
                    and asyncio.get_event_loop().time() - last_rescan_at >= rescan_interval
                ):
                    rescan_task = asyncio.create_task(scanner.scan_top_gappers())
                    last_rescan_at = asyncio.get_event_loop().time()

            if rehab_engine is not None and (
                asyncio.get_event_loop().time() - last_rehab_check >= rehab_check_interval_seconds
            ):
                last_rehab_check = asyncio.get_event_loop().time()
                await _run_rehab_check(rehab_engine, notifier)

            for symbol, stream in streams.items():
                try:
                    bars = await asyncio.wait_for(_fetch_bars(stream), timeout=op_timeout)
                except TimeoutError:
                    stall_counts[symbol] = stall_counts.get(symbol, 0) + 1
                    count = stall_counts[symbol]
                    if count >= _PERSISTENT_STALL_THRESHOLD:
                        _log.warning(
                            "orchestrator.persistent_stall",
                            symbol=symbol,
                            operation="bar_fetch",
                            timeout_seconds=op_timeout,
                            iteration_index=iteration_index,
                            consecutive_stall_count=count,
                        )
                    else:
                        _log.info(
                            "orchestrator.loop_stall",
                            symbol=symbol,
                            operation="bar_fetch",
                            timeout_seconds=op_timeout,
                            iteration_index=iteration_index,
                        )
                    continue
                stall_counts[symbol] = 0
                if bars.empty:
                    continue

                # Phase 7.3: poll path shares ``_evaluate_symbol`` with the
                # event-driven on_new_bar callback. Event path typically wins
                # the race at minute roll (latency-optimal); poll path is a
                # backstop for bars that would otherwise never fire a
                # has_new_bar event (e.g., first bar after subscription, or
                # integer-indexed test frames).
                # Phase 7.4: slice off the in-progress trailing bar so
                # strategies evaluate the just-closed bar, not a
                # freshly-born one with near-zero ticks.
                eval_bars = _bars_for_evaluation(bars)
                await _evaluate_symbol(symbol, eval_bars, _latest_bar_ts(eval_bars))

            if trade_manager is not None and executor is not None:
                _sync_trade_manager_tracking(executor, trade_manager)
                await trade_manager.poll()

            # Phase 10.1 — naked-position watchdog. Self-throttled to its
            # configured ``check_interval_seconds``; safe to call every
            # iteration. Detection-only (never places or cancels orders).
            if watchdog is not None:
                await watchdog.tick()

            # Phase 6.1: flush file handlers at each iteration boundary so the
            # last event before a potential hang is on disk. Default FileHandler
            # buffering loses buffered records on SIGKILL / abrupt termination.
            _flush_file_handlers()

            # Phase 6.4: periodic cursor sweep. Cursor entries for symbols
            # that have been unsubscribed (and haven't come back) accumulate
            # indefinitely over a long session as the watchlist rotates
            # through new gappers. Drop them on a throttled cadence so the
            # dict stays bounded without paying the check on every iteration.
            if iteration_index % _CURSOR_SWEEP_INTERVAL == 0 and last_evaluated_bar_ts:
                subscribed = set(streams)
                stale_keys = [k for k in last_evaluated_bar_ts if k[0] not in subscribed]
                for k in stale_keys:
                    last_evaluated_bar_ts.pop(k, None)

            # Wake early if a shutdown is requested mid-sleep so Ctrl-C feels
            # responsive instead of lagging by up to ``poll_interval`` seconds.
            if shutdown_event is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)
            else:
                await asyncio.sleep(poll_interval)
    finally:
        if rescan_task is not None and not rescan_task.done():
            rescan_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await rescan_task
        # Phase 10.1 — stop the ack listener before unsubscribing market
        # data so a dangling Bot connection doesn't outlive the session.
        if notifier is not None and watchdog is not None:
            with contextlib.suppress(Exception):
                await notifier.stop_ack_listener()
        for symbol in list(streams):
            await market_data.unsubscribe(symbol)
        _log.info("orchestrator.loop_complete", signal_count=len(collected))

    return StrategyLoopResult(signals=collected)


async def _run_rehab_check(
    rehab_engine: RehabEngine,
    notifier: Notifier | None,
) -> None:
    """Recompute the rehab tier; notify on transition. Never raises.

    Intra-session rehab checks shouldn't crash the loop on a journal
    hiccup or Telegram flake, so any exception becomes an error log and
    a silent return. The session-start equivalent lives on
    ``Orchestrator.startup`` and has the same defensive contract.
    """
    try:
        transition = await rehab_engine.check_transitions()
    except Exception as exc:  # noqa: BLE001 - engine faults shouldn't halt the loop
        _log.error("rehab.check_failed", error=str(exc))
        return
    if transition is None:
        return
    _log.warning(
        "rehab.transition",
        old=transition.old_tier.value,
        new=transition.new_tier.value,
        reason=transition.reason,
    )
    if notifier is None:
        return
    try:
        await notifier.send_rehab_tier_change(
            old=transition.old_tier.value,
            new=transition.new_tier.value,
            reason=transition.reason,
        )
    except Exception as exc:  # noqa: BLE001 - Telegram outage must not halt trading
        _log.error("rehab.notify_failed", error=str(exc))


def _sync_trade_manager_tracking(executor: Executor, trade_manager: TradeManager) -> None:
    """Start tracking newly-open positions; stop tracking closed ones.

    Idempotent on both ends — ``start_tracking`` short-circuits on
    re-track, ``stop_tracking`` pops on missing. Runs synchronously
    against the PositionStore snapshot; the bar subscription itself is
    async but the tracking map mutation is not.
    """
    for position in executor.store.list_active():
        if position.status == "open" and not trade_manager.is_tracking(position.symbol):
            # schedule start_tracking; we can't await in this sync hop, so spawn
            asyncio.create_task(trade_manager.start_tracking(position))


def _pick_per_bar_winners(batch: list[Signal]) -> list[Signal]:
    """Mirror ``SignalBus._dedupe_co_signals`` to know which signals actually went out.

    Duplicating this logic in-module (rather than importing a private helper)
    keeps the bus encapsulated and lets each module be read in isolation. The
    preference rule is the same: Gap-and-Go wins ties.
    """
    groups: dict[tuple[str, str], list[Signal]] = {}
    order: list[tuple[str, str]] = []
    for signal in batch:
        key = (signal.symbol, signal.timestamp.isoformat())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(signal)

    winners: list[Signal] = []
    for key in order:
        bucket = groups[key]
        winner = next((s for s in bucket if s.strategy == "gap_and_go"), bucket[0])
        winners.append(winner)
    return winners


# ---------- Phase 4b auto-flatten scheduler ---------- #


class AutoFlattenScheduler:
    """Fires ``flatten_all_active`` every trading day at ``session.flatten_all`` NY-local.

    Hard rule (PLAN §4 + design consultation): runs even when halted.
    Closing positions is never risky; leaving them open overnight is.

    The scheduler itself is just an apscheduler ``AsyncIOScheduler`` with
    a single cron job. The job callback reads the session time from
    ``settings.session.flatten_all`` (``HH:MM``) at ``start()`` and wires
    a ``CronTrigger`` for that exact minute, Monday–Friday, with the
    configured timezone.
    """

    def __init__(
        self,
        *,
        executor: Executor,
        store: PositionStore,
        settings: Settings | None = None,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        """Wire the executor + store; caller can inject a scheduler for tests."""
        self._executor = executor
        self._store = store
        self._settings = settings or get_settings()
        self._scheduler = scheduler or AsyncIOScheduler(timezone=self._settings.session.timezone)
        self._started = False

    def start(self) -> None:
        """Register the cron job and start the scheduler.

        Idempotent — subsequent calls are no-ops. Parses ``HH:MM`` once
        into hour + minute components; apscheduler handles DST by
        treating wall-clock (the NY trading day's local time).
        """
        if self._started:
            return
        hour, minute = _parse_hh_mm(self._settings.session.flatten_all)
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=hour,
            minute=minute,
            timezone=self._settings.session.timezone,
        )
        self._scheduler.add_job(
            self.flatten_all_active,
            trigger=trigger,
            id="session.auto_flatten",
            replace_existing=True,
        )
        self._scheduler.start()
        self._started = True
        _log.info(
            "session.auto_flatten_scheduled",
            fire_at=self._settings.session.flatten_all,
            timezone=self._settings.session.timezone,
        )

    def shutdown(self) -> None:
        """Stop the scheduler; safe to call multiple times."""
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False

    async def flatten_all_active(self) -> int:
        """Close every active position; returns the count actually flattened.

        One ``flatten_symbol`` per position, each wrapped in its own
        try/except so a single IBKR failure doesn't block the rest of
        the book. Emits ``session.auto_flatten`` with the observed count
        regardless of success — the operator wants a daily log entry
        proving the scheduler fired even when there were no positions.
        """
        active = list(self._store.list_active())
        flattened = 0
        for position in active:
            try:
                await self._executor.flatten_symbol(position.symbol, reason="session_auto_flatten")
                flattened += 1
            except Exception as exc:  # noqa: BLE001 - one bad symbol mustn't block the rest
                _log.error(
                    "session.auto_flatten_symbol_failed",
                    symbol=position.symbol,
                    error=str(exc),
                )
        _log.warning(
            "session.auto_flatten",
            positions_seen=len(active),
            positions_flattened=flattened,
        )
        return flattened


_FLATTEN_SAFETY_BUFFER_SECONDS = 60.0


def _derive_duration_seconds_to_flatten(
    settings: Settings,
    now_ny: datetime,
    *,
    safety_buffer_seconds: float = _FLATTEN_SAFETY_BUFFER_SECONDS,
) -> float:
    """Return seconds from ``now_ny`` until today's ``flatten_all`` minus a safety buffer.

    Returns a non-positive value if the flatten time has already passed
    on today's NY clock — the caller uses that as the "exit immediately"
    signal. The safety buffer (60s by default) ensures the loop exits
    cleanly before the ``AutoFlattenScheduler`` cron fires, so the two
    aren't racing on the same open positions.
    """
    hour, minute = _parse_hh_mm(settings.session.flatten_all)
    tz = ZoneInfo(settings.session.timezone)
    flatten_at = datetime(now_ny.year, now_ny.month, now_ny.day, hour, minute, tzinfo=tz)
    return (flatten_at - now_ny).total_seconds() - safety_buffer_seconds


def _parse_hh_mm(value: str) -> tuple[int, int]:
    """Parse a ``HH:MM`` string into (hour, minute); raises on malformed input."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM, got {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid HH:MM value {value!r}")
    return hour, minute


def _is_at_or_past_ny(now_ny: datetime, hour: int, minute: int) -> bool:
    """True when the NY-local ``now_ny`` is at-or-after ``hour:minute`` on its own day.

    Used by the Phase 4d session-reset latch to fire exactly once per day at
    the trading-start boundary.
    """
    return (now_ny.hour, now_ny.minute) >= (hour, minute)


# ---------- Phase 4b session owner ---------- #


class Orchestrator:
    """Session-level wrapper: halt adoption on startup + scheduler lifecycle.

    Minimal surface because the CLI still builds the dependency graph
    (``Executor``, ``RiskEngine``, ``TradeManager``, ``MarketData``,
    ``SignalBus``) and hands it over. Having a thin owner keeps the
    wiring linear in the CLI module and the scheduler + halt-check in
    exactly one place.
    """

    def __init__(
        self,
        *,
        executor: Executor,
        store: PositionStore,
        trade_manager: TradeManager | None = None,
        settings: Settings | None = None,
        auto_flatten: AutoFlattenScheduler | None = None,
        rehab_engine: RehabEngine | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        """Wire the dependencies; auto-flatten is auto-constructed unless injected."""
        self._executor = executor
        self._store = store
        self._trade_manager = trade_manager
        self._settings = settings or get_settings()
        self._auto_flatten = auto_flatten or AutoFlattenScheduler(
            executor=executor, store=store, settings=self._settings
        )
        self._rehab = rehab_engine
        self._notifier = notifier

    @property
    def auto_flatten(self) -> AutoFlattenScheduler:
        """Expose the scheduler for CLI status + tests."""
        return self._auto_flatten

    @property
    def rehab_engine(self) -> RehabEngine | None:
        """Expose the rehab engine (if wired) for CLI status + tests."""
        return self._rehab

    async def startup(self) -> dict[str, Any]:
        """Reconcile with IBKR + adopt the halt flag + rebuild re-entry state + schedule.

        Returns a dict the CLI can inspect to decide what to print and
        whether to bail (e.g. same-day halt flag → refuse to trade).

        Phase 4d ordering: reconcile first (IBKR is authoritative for
        positions), then adopt halt flag, then rebuild ``SymbolHistory``
        from today's journal rows so a crash-restart recovers
        ``entries_count`` + ``last_exit_type``. The halt flag and the
        re-entry state are orthogonal — a halted session still needs
        correct history in case ``reset-halt`` is run mid-day.
        """
        await self._executor.reconcile()
        halt_record = await self._executor.risk_engine.apply_halt_flag_if_current()
        await self._rebuild_symbol_histories()
        rehab_tier = await self._session_start_rehab_check()
        self._auto_flatten.start()
        return {
            "halted": self._executor.risk_engine.is_halted(),
            "halt_record": halt_record,
            "rehab_tier": rehab_tier,
        }

    async def _session_start_rehab_check(self) -> str | None:
        """Load + recompute + notify on any tier change; return the active tier name.

        Defensive: exceptions become error logs and a None return so a
        broken journal never prevents the session from starting. Returns
        the active tier string (e.g. ``"REHAB"``) or ``None`` when the
        engine is not wired — lets the CLI print a concise one-line
        session summary.
        """
        if self._rehab is None:
            return None
        try:
            self._rehab.load_state()
            await _run_rehab_check(self._rehab, self._notifier)
        except Exception as exc:  # noqa: BLE001 - startup must not crash on rehab errors
            _log.error("rehab.startup_check_failed", error=str(exc))
            return None
        return self._rehab.state.tier.value

    async def _rebuild_symbol_histories(self) -> None:
        """Populate ``PositionStore`` histories from today's journal rows."""
        tz = ZoneInfo(self._settings.session.timezone)
        today_ny = datetime.now(tz).date()
        trades = await self._executor.journal.trades_for_session(
            today_ny, self._settings.session.timezone
        )
        self._store.rebuild_symbol_histories_from_journal(trades)
        _log.info(
            "session.symbol_histories_rebuilt",
            session_date=today_ny.isoformat(),
            trades_seen=len(trades),
            symbols=len(self._store.list_symbol_histories()),
        )

    async def shutdown(self) -> None:
        """Tear down the scheduler; does not disconnect IBKR (CLI owns that)."""
        self._auto_flatten.shutdown()


__all__ = [
    "AutoFlattenScheduler",
    "Orchestrator",
    "StrategyLoopResult",
    "build_default_strategies",
    "run_strategy_loop",
]
