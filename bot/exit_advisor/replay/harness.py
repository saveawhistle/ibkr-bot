"""Trade-replay harness.

Replays one closed long trade end-to-end against an :class:`ExitPolicy`.
Layer 1 supports bar-close granularity only (no intra-bar tick synthesis).
Long positions only — the bot does not short, and the harness mirrors that.

Sacred-ground precondition: ``policy.on_event`` is never called before
``PositionProtected`` has fired. That mirrors the production rule that
the entry-protection sequence (parent BUY + initial STP + initial scale-out)
must be in place before any advisor-initiated action runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from bot.exit_advisor.core.events import (
    DrawdownFromPeak,
    Event,
    GateChainResult,
    GateRejection,
    MaxFavorableExcursionUpdate,
    OrderRejectionEvent,
    PartialFillEvent,
    PositionProtected,
    ReplayTerminalTick,
    RMultipleReached,
    TimeInTradeMilestone,
    TimeOfDayMilestone,
)
from bot.exit_advisor.decision.gates import (
    Gate,
    GateContext,
    apply_gate_chain,
    build_default_gate_chain,
)
from bot.exit_advisor.decision.policy import ExitDecision, MaxHoldTimePolicy, TradeState
from bot.exit_advisor.detectors.bar_shape import BarShapeDetector
from bot.exit_advisor.detectors.moving_averages import MovingAveragesDetector
from bot.exit_advisor.detectors.price_levels import PriceLevelsDetector
from bot.exit_advisor.detectors.volume import VolumeDetector

from .bar_history import BarHistory

if TYPE_CHECKING:
    from bot.config import ExitEventsConfig, ExitGatesConfig
    from bot.exit_advisor.decision.policy import ExitPolicy

    from .replay_source import Bar, TradeReplayData


@dataclass
class ReplayResult:
    final_pnl: float
    final_position_size: int
    exit_price: float | None
    exit_timestamp: datetime | None
    decisions: list[tuple[Event, ExitDecision]] = field(default_factory=list)
    events_emitted: list[Event] = field(default_factory=list)
    bars_consumed: int = 0
    notes: list[str] = field(default_factory=list)
    gate_rejections: list[GateRejection] = field(default_factory=list)
    gate_chain_results: list[GateChainResult] = field(default_factory=list)
    decisions_proposed: int = 0
    decisions_accepted: int = 0
    decisions_rejected: int = 0


from bot.exit_advisor.core.timeutil import rth_open_for as _rth_open_for  # noqa: E402


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


class _NoopPolicy:
    """Placeholder for the empty-policy-list edge case in the harness.
    Returns None on every event — no decisions produced."""

    def on_event(
        self, trade_state: TradeState, event: Event
    ) -> ExitDecision | None:  # pragma: no cover - trivial
        return None


class TradeReplayHarness:
    def __init__(
        self,
        replay_data: TradeReplayData,
        policy: ExitPolicy | list[ExitPolicy],
        config: ExitEventsConfig,
        gates_config: ExitGatesConfig | None = None,
    ) -> None:
        self.replay_data = replay_data
        # Normalize: layer 1+2 callers pass a single policy; layer 3 supports
        # a list. ``self.policy`` retains the public single-policy view (the
        # advisor under test), while ``self._all_policies`` is the dispatch
        # list including the built-in MaxHoldTimePolicy when gates are on.
        if isinstance(policy, list):
            self._advisor_policies: list[ExitPolicy] = list(policy)
            self.policy = policy[0] if policy else _NoopPolicy()
        else:
            self._advisor_policies = [policy]
            self.policy = policy
        self.config = config
        self.gates_config = gates_config

        # Built-in MaxHoldTimePolicy + gate chain are layer-3 features; only
        # active when a gates_config was supplied AND it's enabled. When
        # disabled, the harness behaves exactly as in layer 1+2 (single
        # advisor policy, no gate chain, no force-exit-on-time-decay).
        self._gates_enabled = gates_config is not None and gates_config.enabled
        if self._gates_enabled:
            assert gates_config is not None  # narrow for mypy
            self._max_hold_policy: MaxHoldTimePolicy | None = MaxHoldTimePolicy(
                max_hold_minutes=gates_config.max_hold_minutes
            )
            self._gates: list[Gate] = build_default_gate_chain(gates_config)
        else:
            self._max_hold_policy = None
            self._gates = []
        self._recent_decisions: list[tuple[datetime, ExitDecision]] = []

        self._fired_r_multiples: set[tuple[float, str]] = set()
        self._fired_drawdowns: set[float] = set()
        self._fired_tod_milestones: set[int] = set()
        self._fired_in_trade_milestones: set[int] = set()
        self._protection_emitted = False

        self._events_log: list[Event] = []
        self._decisions_log: list[tuple[Event, ExitDecision]] = []
        self._notes: list[str] = []
        self._gate_rejections: list[GateRejection] = []
        self._gate_chain_results: list[GateChainResult] = []
        self._decisions_proposed = 0
        self._decisions_accepted = 0
        self._decisions_rejected = 0

        self._history = BarHistory()
        self._detectors: list[
            PriceLevelsDetector
            | MovingAveragesDetector
            | VolumeDetector
            | BarShapeDetector
        ] = []
        self._ma_detector: MovingAveragesDetector | None = None

    # --- public ---

    def run(self) -> ReplayResult:
        rd = self.replay_data
        symbol = rd.symbol

        # Resolve entry price + initial protection levels from the recorded
        # events. Prefer the post-fill anchored stop/scale-out (those are
        # what the bot actually had working in IBKR) and fall back to the
        # pre-fill bracket levels if the fill anchor never logged.
        if rd.fill_event:
            entry_price = float(rd.fill_event["fill_price"])
            position_size = int(rd.fill_event["filled_shares"])
        else:
            entry_price = float(rd.bracket_event["entry_price"])
            position_size = int(rd.bracket_event["shares"])
            self._notes.append("no position.filled event; using bracket entry_price")

        # Use the bracket's original stop, not the post-fill anchored level.
        # The bracket stop is the stop the bot first armed and the level
        # the recorded exit prices anchor to (e.g. ZENA 2026-04-30 closed
        # at 2.16 = bracket stop, even though the post-fill anchor moved
        # it to 2.17 — the cancel/resubmit race meant the original 2.16
        # stop fired). Replay reproduces the bot's actual outcome, not
        # the intended one. ``initial_scale_out`` prefers the anchored
        # level (it's denser to the actual working order), falling back
        # to the entry event.
        initial_stop = float(rd.bracket_event["stop_price"])
        if rd.protection_anchored_event:
            initial_scale_out = float(rd.protection_anchored_event["scale_lmt_price"])
        else:
            initial_scale_out = float(rd.entry_event.get("scale_out", 0.0))
            self._notes.append("no protection_fill_anchored event; using pre-fill scale-out")

        entry_ts = self._resolve_entry_timestamp()

        state = TradeState(
            symbol=symbol,
            entry_price=entry_price,
            entry_timestamp=entry_ts,
            current_position_size=position_size,
            initial_position_size=position_size,
            initial_stop=initial_stop,
            initial_scale_out=initial_scale_out,
            current_stop=initial_stop,
            realized_pnl=0.0,
            is_protected=False,
            peak_price=entry_price,
            current_price=entry_price,
        )

        # Build layer-2 detectors and feed them the pre-trade bar backfill
        # so VWAP / EMA / rolling-volume baselines are warm when the trade
        # starts. Backfill events ARE logged (one-shot data-availability
        # warnings, HOD/LOD touches that already happened, etc.) but the
        # policy is suppressed because state.is_protected is still False —
        # sacred-ground gate stays intact.
        self._build_detectors(symbol, rd)
        for bar in rd.pre_trade_bars:
            self._history.add_bar(bar)
            self._emit_detector_events(state, bar)

        # Order-state events: any rejections recorded BEFORE protection
        # was confirmed get logged but never call the policy.
        self._emit_order_rejections_pre_protection(symbol)

        # Protection fires once the bracket children are confirmed working.
        protection_ts = self._resolve_protection_timestamp(entry_ts)
        protected_event = PositionProtected(
            timestamp=protection_ts,
            symbol=symbol,
            entry_price=entry_price,
            initial_stop=initial_stop,
            initial_scale_out=initial_scale_out,
            position_size=position_size,
        )
        self._emit(protected_event, state)
        state.is_protected = True
        self._protection_emitted = True

        # Bar loop. Events use bar-close timestamps (bar_time is the bar
        # START; the bar itself spans bar_time → bar_time + 60s). For an
        # entry bar that fired mid-interval, this puts derived events
        # after the protection moment, where the policy is allowed to act.
        bars_consumed = 0
        exit_taken = False
        for bar in rd.bars:
            bar_close_ts = _to_utc(bar.timestamp) + timedelta(minutes=1)
            prior_peak = self._update_state_for_bar(state, bar)
            self._history.add_bar(bar)
            self._emit_bar_derived_events(state, bar, bar_close_ts, prior_peak)
            self._emit_detector_events(state, bar)
            bars_consumed += 1

            # Stop simulation: if the bar's low pierced the current stop, the
            # mechanical layer would have filled the stop.
            if bar.low <= state.current_stop:
                exit_price = state.current_stop
                self._apply_full_exit(
                    state, exit_price, bar_close_ts, "stop fill (bar low <= stop)"
                )
                exit_taken = True
                break

            decision = self._latest_decision_for_bar()
            if decision is not None and decision.action == "exit_full":
                exit_price = (
                    decision.fill_price if decision.fill_price is not None else bar.close
                )
                self._apply_full_exit(
                    state, exit_price, bar_close_ts, decision.reason or "exit_full"
                )
                exit_taken = True
                break

        # Terminal tick: gives replay-anchored policies a chance to fire even
        # when the recorded exit doesn't align with a bar boundary.
        if not exit_taken:
            terminal = ReplayTerminalTick(
                timestamp=rd.recorded_exit_timestamp, symbol=symbol
            )
            decision = self._emit(terminal, state)
            if decision is not None and decision.action == "exit_full":
                exit_price = (
                    decision.fill_price
                    if decision.fill_price is not None
                    else state.current_price
                )
                self._apply_full_exit(
                    state,
                    exit_price,
                    rd.recorded_exit_timestamp,
                    decision.reason or "exit_full (terminal)",
                )
                exit_taken = True

        # Compute final PnL.
        final_pnl = state.realized_pnl
        return ReplayResult(
            final_pnl=final_pnl,
            final_position_size=state.current_position_size,
            exit_price=self._exit_price,
            exit_timestamp=self._exit_ts,
            decisions=list(self._decisions_log),
            events_emitted=list(self._events_log),
            bars_consumed=bars_consumed,
            notes=list(self._notes),
            gate_rejections=list(self._gate_rejections),
            gate_chain_results=list(self._gate_chain_results),
            decisions_proposed=self._decisions_proposed,
            decisions_accepted=self._decisions_accepted,
            decisions_rejected=self._decisions_rejected,
        )

    # --- internals ---

    _exit_price: float | None = None
    _exit_ts: datetime | None = None

    def _resolve_entry_timestamp(self) -> datetime:
        rd = self.replay_data
        if rd.fill_event and "timestamp" in rd.fill_event:
            return _parse_event_ts(rd.fill_event["timestamp"])
        return _parse_event_ts(rd.entry_event["timestamp"])

    def _resolve_protection_timestamp(self, entry_ts: datetime) -> datetime:
        rd = self.replay_data
        if rd.protection_anchored_event and "timestamp" in rd.protection_anchored_event:
            return _parse_event_ts(rd.protection_anchored_event["timestamp"])
        return entry_ts

    def _build_detectors(self, symbol: str, rd: TradeReplayData) -> None:
        """Instantiate the layer-2 detectors enabled in the config.

        ``today_open`` for the gap-fill computation comes from the first
        bar in the pre-trade backfill if available, else the first
        trade-window bar. Detectors not enabled in config simply aren't
        built — saves the dispatch cost in the bar loop.
        """
        cfg = self.config

        today_open: float | None = None
        if rd.pre_trade_bars:
            today_open = rd.pre_trade_bars[0].open
        elif rd.bars:
            today_open = rd.bars[0].open

        if cfg.price_levels.enabled:
            self._detectors.append(
                PriceLevelsDetector(
                    symbol=symbol,
                    hod_lod_enabled=cfg.price_levels.hod_lod,
                    prior_day_high_low_enabled=cfg.price_levels.prior_day_high_low,
                    prior_day_close_enabled=cfg.price_levels.prior_day_close,
                    gap_fill_enabled=cfg.price_levels.gap_fill,
                    prior_day_high=rd.prior_day_session_high,
                    prior_day_low=rd.prior_day_session_low,
                    prior_day_close=rd.prior_day_session_close,
                    today_open=today_open,
                    gap_threshold_pct=cfg.price_levels.gap_threshold_pct,
                )
            )

        if cfg.moving_averages.enabled:
            ma = MovingAveragesDetector(
                symbol=symbol,
                vwap_enabled=cfg.moving_averages.vwap,
                ema_9_enabled=cfg.moving_averages.ema_9,
            )
            self._detectors.append(ma)
            self._ma_detector = ma

        if cfg.volume.enabled:
            # Layer L2-A: try the multi-day cache for the RVOL curve.
            # Falls back to None when the cache isn't populated, in
            # which case VolumeDetector emits its existing
            # RVolDataUnavailable warning.
            from .replay_source import load_prior_n_day_volume_curve

            prior_curve, days_used = load_prior_n_day_volume_curve(
                symbol=symbol,
                trade_date=rd.trade_date,
                n_days=cfg.volume.rvol_lookback_days,
            )
            self._detectors.append(
                VolumeDetector(
                    symbol=symbol,
                    spike_threshold_x_avg=cfg.volume.spike_threshold_x_avg,
                    dryup_threshold_x_avg=cfg.volume.dryup_threshold_x_avg,
                    baseline_window_bars=cfg.volume.baseline_window_bars,
                    rvol_milestones=list(cfg.volume.rvol_milestones),
                    prior_day_cum_volume_by_minute=prior_curve or None,
                    rvol_prior_days_used=days_used,
                    rvol_prior_days_configured=cfg.volume.rvol_lookback_days,
                )
            )

        if cfg.bar_shape.enabled:
            self._detectors.append(
                BarShapeDetector(
                    symbol=symbol,
                    enabled_shapes=tuple(cfg.bar_shape.shapes),
                    wick_threshold_pct=cfg.bar_shape.wick_threshold_pct,
                    consecutive_bars_threshold=cfg.bar_shape.consecutive_bars_threshold,
                )
            )

    def _emit_detector_events(self, state: TradeState, bar: Bar) -> None:
        for det in self._detectors:
            for event in det.on_bar(bar, self._history):
                self._emit(event, state)

    def _emit_order_rejections_pre_protection(self, symbol: str) -> None:
        """Order rejections logged in the structured events stream.

        Layer 1 reads the structured events only; ib_async repr-style
        ``Cancelled`` / Error 10349 lines are not parseable JSON and are
        skipped at load time. As a result the harness sees no rejections
        for ZENA's 2026-04-30 trade — the TIF auto-cancel/resubmit dance
        IBKR did is not currently a structured event. Captured in
        PROGRESS.md as input to layer 2.
        """
        if not self.config.order_state.enabled or not self.config.order_state.order_rejections:
            return
        for raw in self.replay_data.order_events:
            if raw.get("event") != "order.rejected":  # not present in current logs
                continue
            event = OrderRejectionEvent(
                timestamp=_parse_event_ts(raw["timestamp"]),
                symbol=symbol,
                order_id=int(raw.get("order_id", 0)),
                error_code=raw.get("error_code"),
                reason=str(raw.get("reason", "")),
            )
            self._events_log.append(event)

    def _update_state_for_bar(self, state: TradeState, bar: Bar) -> float:
        """Update mutable state from this bar; return the prior peak_price
        so MFE detection can compare new vs old."""
        prior_peak = state.peak_price
        state.current_price = bar.close
        if bar.high > state.peak_price:
            state.peak_price = bar.high
        return prior_peak

    def _emit_bar_derived_events(
        self, state: TradeState, bar: Bar, bar_close_ts: datetime, prior_peak: float
    ) -> None:
        # time class
        if self.config.time.enabled:
            self._maybe_emit_time_milestones(state, bar_close_ts)

        # pnl class — R-multiples, drawdowns, MFE
        if self.config.pnl.enabled:
            self._maybe_emit_pnl_events(state, bar, bar_close_ts, prior_peak)

    def _maybe_emit_time_milestones(self, state: TradeState, bar_ts: datetime) -> None:
        cfg = self.config.time

        rth_open = _rth_open_for(bar_ts)
        minutes_after_open = int((bar_ts - rth_open).total_seconds() // 60)
        for milestone in cfg.milestones_minutes_after_open:
            if minutes_after_open >= milestone and milestone not in self._fired_tod_milestones:
                self._fired_tod_milestones.add(milestone)
                self._emit(
                    TimeOfDayMilestone(
                        timestamp=bar_ts,
                        symbol=state.symbol,
                        minutes_after_open=milestone,
                    ),
                    state,
                )

        minutes_in_trade = int((bar_ts - state.entry_timestamp).total_seconds() // 60)
        for milestone in cfg.time_in_trade_milestones:
            if minutes_in_trade >= milestone and milestone not in self._fired_in_trade_milestones:
                self._fired_in_trade_milestones.add(milestone)
                self._emit(
                    TimeInTradeMilestone(
                        timestamp=bar_ts,
                        symbol=state.symbol,
                        minutes_in_trade=milestone,
                    ),
                    state,
                )

    def _maybe_emit_pnl_events(
        self, state: TradeState, bar: Bar, bar_ts: datetime, prior_peak: float
    ) -> None:
        cfg = self.config.pnl

        risk_per_share = state.entry_price - state.initial_stop
        if risk_per_share <= 0:
            return  # degenerate: no R denominator

        prior_peak_r = (prior_peak - state.entry_price) / risk_per_share
        new_peak_r = (state.peak_price - state.entry_price) / risk_per_share
        if cfg.track_mfe and new_peak_r > prior_peak_r and new_peak_r > 0:
            self._emit(
                MaxFavorableExcursionUpdate(
                    timestamp=bar_ts,
                    symbol=state.symbol,
                    new_peak_r_multiple=round(new_peak_r, 4),
                    previous_peak_r_multiple=round(prior_peak_r, 4),
                ),
                state,
            )

        # R-multiple thresholds based on this bar's high (favorable) and low
        # (adverse). Each (multiple, direction) tuple fires at most once.
        favorable_r = (bar.high - state.entry_price) / risk_per_share
        adverse_r = (bar.low - state.entry_price) / risk_per_share
        for r in cfg.r_multiples:
            up_key = (r, "up")
            if favorable_r >= r and up_key not in self._fired_r_multiples:
                self._fired_r_multiples.add(up_key)
                self._emit(
                    RMultipleReached(
                        timestamp=bar_ts,
                        symbol=state.symbol,
                        r_multiple=r,
                        direction="up",
                    ),
                    state,
                )
            down_key = (r, "down")
            # adverse R is negative for losses; treat r_multiple as |R| reached
            if adverse_r <= -r and down_key not in self._fired_r_multiples:
                self._fired_r_multiples.add(down_key)
                self._emit(
                    RMultipleReached(
                        timestamp=bar_ts,
                        symbol=state.symbol,
                        r_multiple=r,
                        direction="down",
                    ),
                    state,
                )

        # Drawdown from peak.
        current_r = (state.current_price - state.entry_price) / risk_per_share
        peak_r = (state.peak_price - state.entry_price) / risk_per_share
        if peak_r > 0:
            drawdown = (peak_r - current_r) / peak_r
            for threshold in cfg.drawdown_pct_from_peak:
                if drawdown >= threshold and threshold not in self._fired_drawdowns:
                    self._fired_drawdowns.add(threshold)
                    self._emit(
                        DrawdownFromPeak(
                            timestamp=bar_ts,
                            symbol=state.symbol,
                            drawdown_pct=threshold,
                            peak_r_multiple=round(peak_r, 4),
                            current_r_multiple=round(current_r, 4),
                        ),
                        state,
                    )

    def _emit(self, event: Event, state: TradeState) -> ExitDecision | None:
        """Log the event, dispatch to all policies, resolve priority,
        run the gate chain, and return the final (post-gate) decision.

        Layer 3 changes the decision pipeline shape:
        1. Each event goes to every policy (advisor + built-in MaxHoldTime).
        2. Priority resolution: MaxHoldTimePolicy force-exit beats anything else.
        3. The resolved decision passes through the gate chain.
        4. Gate-rejected decisions become ``hold`` (None) but are still logged
           via ``GateChainResult`` + ``GateRejection`` for forensics.
        """
        self._events_log.append(event)
        if not state.is_protected:
            return None

        # Dispatch to all policies (advisor + max-hold-time when active).
        proposals: list[ExitDecision] = []
        for advisor in self._advisor_policies:
            proposed = advisor.on_event(state, event)
            if proposed is not None:
                proposals.append(proposed)
        if self._max_hold_policy is not None:
            forced = self._max_hold_policy.on_event(state, event)
            if forced is not None:
                proposals.append(forced)

        decision = self._resolve_priority(proposals)
        if decision is None or decision.action == "hold":
            return None

        self._decisions_proposed += 1

        # Layer 3 gate chain (no-op when gates disabled).
        final: ExitDecision | None
        if not self._gates_enabled:
            final = decision
        else:
            ctx = GateContext(
                current_timestamp=event.timestamp,
                recent_decisions=list(self._recent_decisions),
                events_emitted_this_trade=len(self._events_log),
                bar_count_in_trade=self._history.bar_count(),
            )
            final, gate_results = apply_gate_chain(self._gates, state, decision, ctx)
            chain_evt = GateChainResult(
                timestamp=event.timestamp,
                symbol=state.symbol,
                original_decision=decision,
                final_decision=final,
                gate_results=[
                    (name, r.accepted, r.rejection_reason) for name, r in gate_results
                ],
            )
            self._events_log.append(chain_evt)
            self._gate_chain_results.append(chain_evt)
            for name, r in gate_results:
                if not r.accepted:
                    rej = GateRejection(
                        timestamp=event.timestamp,
                        symbol=state.symbol,
                        gate_name=name,
                        rejection_reason=r.rejection_reason,
                        rejection_detail=dict(r.rejection_detail),
                        original_decision=decision,
                    )
                    self._events_log.append(rej)
                    self._gate_rejections.append(rej)

        if final is None:
            self._decisions_rejected += 1
            return None

        self._decisions_accepted += 1
        # Track for RecencyThrottleGate's lookback window. Cap at 10 to
        # bound memory; only the most recent matters for the gate today.
        self._recent_decisions.insert(0, (event.timestamp, final))
        del self._recent_decisions[10:]

        self._decisions_log.append((event, final))
        if (
            final.action == "tighten_stop"
            and final.new_stop_price is not None
            and final.new_stop_price > state.current_stop
        ):
            # Sacred ground: never loosen the stop.
            state.current_stop = final.new_stop_price
        return final

    @staticmethod
    def _resolve_priority(proposals: list[ExitDecision]) -> ExitDecision | None:
        """MaxHoldTimePolicy force-exit beats advisor decisions; otherwise
        the first non-None proposal (which by construction is the advisor)."""
        if not proposals:
            return None
        for p in proposals:
            if p.action == "exit_full" and p.reason.startswith("max_hold_time_reached"):
                return p
        return proposals[0]

    def _latest_decision_for_bar(self) -> ExitDecision | None:
        """Return the most recent decision logged in this run, if any."""
        if not self._decisions_log:
            return None
        return self._decisions_log[-1][1]

    def _apply_full_exit(
        self,
        state: TradeState,
        exit_price: float,
        exit_ts: datetime,
        reason: str,
    ) -> None:
        shares = state.current_position_size
        pnl = (exit_price - state.entry_price) * shares
        state.realized_pnl += pnl
        state.current_position_size = 0
        state.current_price = exit_price
        self._exit_price = exit_price
        self._exit_ts = exit_ts
        self._notes.append(f"full exit: {shares}@{exit_price:.4f} ({reason})")


def _parse_event_ts(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


# expose PartialFillEvent for re-export so test files can patch behavior
__all__ = [
    "PartialFillEvent",
    "ReplayResult",
    "TradeReplayHarness",
]
