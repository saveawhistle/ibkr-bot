"""Typed configuration: merges ``config.yaml`` (strategy/risk) with ``.env`` (secrets)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ConfigurationError(RuntimeError):
    """Raised when required configuration (e.g. an API key) is missing at a point where it's needed."""


class AccountConfig(BaseModel):
    """Account mode (paper/live).

    Phase 7.7: the Phase-0 ``min_equity_usd`` PDT floor was removed — it was
    never enforced by any gate (no validator, no pre-trade check) and the
    PDT rule itself was eliminated by FINRA Rule 4210 amendment
    SR-FINRA-2025-017 (SEC approved April 14, 2026). See
    ``bot/risk.py`` module docstring.
    """

    mode: Literal["paper", "live"] = "paper"


class ReEntryConfig(BaseModel):
    """Phase 4d — controls for re-entering the same symbol on a new pullback.

    *"I buy the first and second pullback most aggressively and
    be a bit cautious on later pullbacks."* Relaxes the prior "one entry per
    symbol per day" rule while keeping "one active position per symbol at a
    time" intact via ``RiskConfig.max_concurrent_positions`` (still 1).

    Defaults are intentionally conservative — cooldown + profitable-prior-exit
    are both on by default so a crash-restart doesn't suddenly allow revenge
    trading we haven't calibrated in paper yet. ``size_multipliers`` must have
    at least ``max_entries_per_symbol`` elements; the validator enforces that.
    """

    enabled: bool = True
    max_entries_per_symbol: int = 3
    size_multipliers: list[float] = Field(default_factory=lambda: [1.0, 1.0, 0.5])
    cooldown_seconds: int = 120  # ~2 one-minute bars; blocks within-bar re-entries
    require_profitable_prior_exit: bool = True

    @model_validator(mode="after")
    def _validate_multiplier_length(self) -> ReEntryConfig:
        """Refuse configs whose multiplier list is shorter than ``max_entries_per_symbol``.

        Silently padding with 0.0 or 1.0 would mask a typo — fail loudly so
        the operator fixes the YAML before the bot places a wrong-sized trade.
        """
        if len(self.size_multipliers) < self.max_entries_per_symbol:
            raise ValueError(
                f"re_entry.size_multipliers has {len(self.size_multipliers)} elements but "
                f"max_entries_per_symbol is {self.max_entries_per_symbol}; provide one "
                "multiplier per allowed entry."
            )
        return self

    @field_validator("size_multipliers")
    @classmethod
    def _validate_multiplier_values(cls, value: list[float]) -> list[float]:
        """Reject negative multipliers; cap > 1.0 at 1.0 would hide a config typo, so leave it."""
        if any(m < 0 for m in value):
            raise ValueError("re_entry.size_multipliers entries must be >= 0.")
        return value


class RehabConfig(BaseModel):
    """Phase 4g — automatic rehab tier system.

    *"When you're in a drawdown, trade smaller. When you're in
    a deep drawdown, trade way smaller. Earn your way back to normal size."*
    Encodes that rule so the bot auto-shrinks risk after a cold streak and
    restores normal caps once the drawdown is recovered by the configured
    fraction (0.50 by default — earn back half before returning to normal).

    Two triggers feed the tier decision:
      * ``*_consecutive_red_days``: raw count of back-to-back negative-PnL
        sessions in the journal.
      * ``*_drawdown_multiple_of_daily_loss``: cumulative drawdown over the
        lookback window, expressed as a multiple of ``max_daily_loss_usd``
        (so a 3.0× trigger with a $300 daily cap fires at $900 underwater).

    The stricter tier wins: DEEP_REHAB always dominates REHAB. Multipliers
    are scale-down only — the validator forbids values > 1.0 so a typo can
    never *inflate* caps. ``rehab.enabled: false`` bypasses the whole engine
    and caps always come from base config (used by test_risk.py's existing
    suite, which predates rehab).
    """

    enabled: bool = True
    rehab_consecutive_red_days: int = 2
    rehab_lookback_days: int = 10
    rehab_drawdown_multiple_of_daily_loss: float = 3.0
    rehab_max_loss_multiplier: float = 0.5
    rehab_max_daily_loss_multiplier: float = 0.5
    rehab_max_trades_per_day: int = 3
    deep_rehab_consecutive_red_days: int = 4
    deep_rehab_drawdown_multiple_of_daily_loss: float = 5.0
    deep_rehab_max_loss_multiplier: float = 0.25
    deep_rehab_max_daily_loss_multiplier: float = 0.25
    deep_rehab_max_trades_per_day: int = 1
    recovery_drawdown_recovered_fraction: float = 0.50

    @model_validator(mode="after")
    def _validate_bounds(self) -> RehabConfig:
        """Enforce scale-down-only multipliers and sensible trade counts.

        Rehab must never *relax* risk, so all four multipliers live in
        ``(0.0, 1.0]``. Trade counts must be >= 1 (a 0-trade tier would
        simply halt — use the halt module for that). The recovery fraction
        must be positive and no greater than 1.0: 1.0 means "fully recover
        the drawdown before exiting rehab"; 0.50 is the documented default.
        """
        for name in (
            "rehab_max_loss_multiplier",
            "rehab_max_daily_loss_multiplier",
            "deep_rehab_max_loss_multiplier",
            "deep_rehab_max_daily_loss_multiplier",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"rehab.{name} must be in (0.0, 1.0]; got {value}. "
                    "Rehab is scale-down only — values >1 would inflate caps."
                )
        for name in ("rehab_max_trades_per_day", "deep_rehab_max_trades_per_day"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(
                    f"rehab.{name} must be >= 1; got {value}. "
                    "Use halt.flag if you need to block entries entirely."
                )
        if not 0.0 < self.recovery_drawdown_recovered_fraction <= 1.0:
            raise ValueError(
                f"rehab.recovery_drawdown_recovered_fraction must be in (0.0, 1.0]; "
                f"got {self.recovery_drawdown_recovered_fraction}."
            )
        return self


class RiskConfig(BaseModel):
    """Hard risk/session guardrails from PLAN §2.5 + Phase 4b design consultation.

    Sizing is per-trade dollar max loss (``max_loss_per_trade_usd``),
    capped by a secondary ``max_position_value_usd`` so penny stocks with tight
    stops don't blow the position footprint. Halts are dollar-based (not
    percent): ``max_daily_loss_usd`` halts new entries on loss,
    ``daily_profit_goal_usd`` halts on goal, and ``giveback_trigger_usd`` /
    ``giveback_pct`` halt when a green day starts bleeding out.

    The Phase 4a ``position_risk_usd`` key was replaced by
    ``max_loss_per_trade_usd``; see the model validator below for the
    migration guard.
    """

    max_loss_per_trade_usd: float = 100.0
    max_position_value_usd: float = (
        15_000.0  # 3× $5k starting equity; under 4:1 margin's $20k ceiling
    )
    max_daily_loss_usd: float = 300.0
    daily_profit_goal_usd: float = 500.0
    giveback_trigger_usd: float = 400.0
    giveback_pct: float = 50.0
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 5
    # Phase 4c — the published quality gates.
    max_stop_width_usd: float = 0.50  # the methodology: "If I risk 50 cents or more… I try to avoid."
    max_pct_of_bar_volume: float = 2.0  # the methodology scanner: order ≪ 2–5% of 1-min volume
    extension_bar_trigger_multiple: float = (
        2.0  # threshold = max_loss × this (≈ the "$200+" spike)
    )
    # Phase 4d — multiple pullback re-entries on the same symbol.
    re_entry: ReEntryConfig = Field(default_factory=ReEntryConfig)
    # Phase 4g — automatic rehab tier (scale-down caps during cold streaks).
    rehab: RehabConfig = Field(default_factory=RehabConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_deprecated_4a_keys(cls, values: Any) -> Any:
        """Fail loudly on the 4a ``position_risk_usd`` key to prevent silent config drift.

        pydantic's default ``extra='ignore'`` would quietly drop the key and the
        bot would run with the 4b default ($100 max loss) — probably fine, but
        the user won't know their YAML is stale. Explicit error > silent drift.
        """
        if isinstance(values, dict) and "position_risk_usd" in values:
            raise ValueError(
                "risk.position_risk_usd is a deprecated Phase 4a key. Replace it with "
                "risk.max_loss_per_trade_usd in config.yaml (see the migration comment "
                "in config.yaml for details)."
            )
        return values


class ExecutionConfig(BaseModel):
    """Execution-layer guardrails + exit discipline (Phase 4i).

    ``rth_only`` forces ``outsideRth=False`` on bracket legs — premarket fills
    are too slippery for these strategies. ``require_paper_mode`` is the hard
    gate that blocks the ``trade`` CLI against a live account; Phase 4b unlocks
    live by flipping this (and only after the risk module lands).

    Phase 4i re-anchors the exit plumbing on the published rules:

    * ``scale_out_multiple`` (default **2.0**) — first-half take-profit in
      R-multiples. the "sell half at a 2:1" rule. Strategies compute the
      scale-out anchor as ``entry + scale_out_multiple × initial_risk``. The
      trade manager reads ``position.scale_out_price`` directly (not the
      implicit +1R of earlier phases) and fires the scale-out when that
      price is tagged.
    * ``runner_target_enabled`` (default **False**) — the methodology does not place
      hard profit ceilings on the runner. When false, the bracket is a
      two-leg parent+STP (no runner LMT), and the post-scale-out stop has
      no OCA partner. When true, the bracket carries a runner LMT at
      ``entry + runner_target_multiple × initial_risk``.
    * ``runner_target_multiple`` — only consumed when the runner is
      enabled. Validated against ``scale_out_multiple`` below: the runner
      ceiling must sit above the scale-out (otherwise the ceiling fills
      before the scale-out even fires).

    Phase 6.14 replaced the ``post_scaleout_trail_enabled`` boolean with
    the ``post_scaleout_stop_mode`` enum. Three modes:

    * ``static_breakeven`` — Phase 4e fallback. Flat STP at
      ``position.avg_price``, no trail.
    * ``adjustable_to_trail`` — Phase 4h. STP starts at breakeven and
      server-side converts to a TRAIL once price tags
      ``scale_out + trail_activation_r_multiple × initial_risk``. The
      tail is protected at breakeven until then; profit-locking only
      engages after the further +N R move.
    * ``immediate_trail`` — Phase 6.14 default. Plants a TRAIL order at
      the scale-out moment with ``trailingAmount = trail_amount_r_multiple
      × initial_risk``. Starting stop position is
      ``scale_out_price - trail_amount`` (≈ +1R from entry when defaults
      hold). The tail is profit-locked immediately at at least ~+1R and
      follows the runner upward via IBKR's server-side trailing logic.
      The intent: "ensure profit no matter what after scale-out."

    ``trail_amount_r_multiple`` is the dollar distance of the trail, in
    R-multiples of the original per-share risk, and applies to both
    trail modes (``adjustable_to_trail`` and ``immediate_trail``).
    ``trail_activation_r_multiple`` is only consumed by
    ``adjustable_to_trail`` (the activation trigger for the conversion).
    """

    rth_only: bool = True
    require_paper_mode: bool = True
    scale_out_multiple: float = 2.0
    runner_target_enabled: bool = False
    runner_target_multiple: float = 3.0
    post_scaleout_stop_mode: Literal[
        "static_breakeven", "adjustable_to_trail", "immediate_trail"
    ] = "immediate_trail"
    trail_activation_r_multiple: float = 1.0
    trail_amount_r_multiple: float = 1.0
    # Phase 4j — BUY STP-LMT parent entries. ``STP_LMT`` is the Phase 4j
    # default: IBKR triggers server-side on the first tick at/above
    # ``signal.entry`` and fills up to ``entry + entry_limit_buffer_usd``.
    # ``LMT`` preserves the Phase 4i marketable-limit behavior as a
    # toggle-back escape hatch. ``entry_limit_buffer_usd`` is the LMT
    # ceiling above the STP trigger; 10 cents sweeps 2-3 typical ask
    # levels without allowing catastrophic overfills on thin names.
    # Phase 6.12 ``MKT`` is the manual hotkey flow: bar-close signal →
    # immediate market BUY at top-of-book ask. Skips the server-side
    # stop trigger entirely (no "did the STP-LMT convert?" ambiguity)
    # but accepts uncapped slippage on thin books. Use only on
    # sufficiently liquid tickers — the Phase 4c ``max_pct_of_bar_volume``
    # guardrail is your defence against a MKT sweeping an illiquid
    # name through multiple price levels.
    entry_order_type: Literal["LMT", "STP_LMT", "MKT"] = "STP_LMT"
    entry_limit_buffer_usd: float = 0.10
    # Phase 7.6: server-side adjustable STP on the *initial* bracket stop.
    # At entry, the full-size protective STP sits at ``signal.stop`` as today.
    # When the market tags ``signal.entry + initial_stop_trigger_r_multiple × R``
    # (default +1R), IBKR auto-converts the STP to a TRAIL with
    # ``initial_stop_trail_r_multiple × R`` trailing distance (default 1.5R).
    # Zero bot-side code runs at the trigger — it's encoded on the order at
    # placement time. When the scale-out LMT later fills, the OCA group
    # cancels this TRAIL and ``_handle_scale_out_lmt_fill`` installs the
    # tighter post-scale trail per ``post_scaleout_stop_mode``.
    initial_stop_adjustable_enabled: bool = True
    initial_stop_trigger_r_multiple: float = 1.0
    initial_stop_trail_r_multiple: float = 1.5
    # Phase 7.8: the "first red candle close" pre-scale exit. When True,
    # any bar that closes red (close < open) AND below the prior close
    # triggers a full-position market-close BEFORE scale-out. Post-scale
    # this is suppressed entirely — the server-side TRAIL is the runner's
    # exit. The pre-scale check is a belt-and-suspenders layer on top of
    # the server-side STP: catches bar-close violations even when the paper
    # STP misses (known IBKR paper issue) and aligns with the discipline
    # of cutting quickly when the initial move rolls over.
    pre_scale_red_candle_exit_enabled: bool = True
    # Phase 8.2: percent-based buffer above ``signal.entry`` for the LMT
    # parent on the ``"LMT"`` entry path. Buffer = ``entry × pct/100``,
    # clamped to ``[floor_usd, cap_usd]``. Floor prevents sub-spread
    # buffers on penny stocks (e.g. $1.50 × 2% = $0.03 << typical
    # spread); cap prevents catastrophic slippage on high-priced
    # tickers during halt reopens or fast spikes. Unfilled LMTs are
    # auto-cancelled by Phase 6.5 logic on the next bar.
    #
    # Only consumed by the ``"LMT"`` entry path. ``"STP_LMT"`` keeps
    # its existing flat ``entry_limit_buffer_usd`` (different mechanic
    # — that buffer is the LMT ceiling above the STP trigger).
    lmt_buffer_pct: float = 2.0
    lmt_buffer_usd_floor: float = 0.15
    lmt_buffer_usd_cap: float = 0.50
    # Phase 10.6: percentage ceiling on the LMT buffer to keep the parent
    # below IBKR's ~9.8% aggressive-LMT cap on low-priced stocks. Without
    # this, the fixed ``lmt_buffer_usd_floor`` produces LMTs that are a
    # huge percentage of price on sub-$2 names (e.g. $0.15 floor on a
    # $1.12 stock → 13% buffer → Error 202). The ceiling caps the buffer
    # at ``entry × max_pct/100``; default 7% leaves ~2.8 percentage
    # points of margin against the IBKR cap, accounting for drift between
    # signal-time price and broker-side market at submission.
    #
    # Symmetric to Phase 10.2 (stop-distance floor): the floor widens
    # too-tight stops, the ceiling narrows too-aggressive entry buffers.
    lmt_buffer_max_pct: float = 7.0

    @model_validator(mode="after")
    def _validate_scale_out_multiple(self) -> ExecutionConfig:
        """Reject sub-1R scale-outs; warn above 3R (the 2:1 R:R rule + wide-R breakdowns).

        A scale-out below 1R books the first half at a loss relative to the
        initial risk — strictly worse than taking the stop. Above 3R is
        legal (wide-R breakouts exist) but departs from the published
        2:1 rule, so we emit a startup warning.
        """
        if self.scale_out_multiple < 1.0:
            raise ValueError(
                f"execution.scale_out_multiple must be >= 1.0 "
                f"(got {self.scale_out_multiple}); sub-1R scale-outs book the first "
                "half at a loss relative to initial risk."
            )
        if self.scale_out_multiple > 3.0:
            structlog.get_logger("bot.config").warning(
                "config.scale_out_multiple_high",
                value=self.scale_out_multiple,
                hint="Scale-out above 3R departs from the 2:1 R:R rule — most trades "
                "won't reach the anchor before reversing.",
            )
        return self

    @model_validator(mode="after")
    def _validate_runner_target_multiple(self) -> ExecutionConfig:
        """When the runner is enabled, require the ceiling to sit above the scale-out.

        With ``runner_target_enabled=False`` (the Phase 4i default) the
        runner multiple is never consumed, so we skip the check entirely —
        operators who leave the legacy 3.0 default alone won't trip on it.
        When the runner is turned on, its multiple must exceed
        ``scale_out_multiple`` or the ceiling fills before scale-out fires,
        which is a configuration bug.
        """
        if not self.runner_target_enabled:
            return self
        if self.runner_target_multiple <= self.scale_out_multiple:
            raise ValueError(
                f"execution.runner_target_multiple ({self.runner_target_multiple}) must be "
                f"greater than execution.scale_out_multiple ({self.scale_out_multiple}) when "
                "runner_target_enabled=true; otherwise the runner LMT fires before scale-out."
            )
        return self

    @model_validator(mode="after")
    def _validate_post_scaleout_trail_params(self) -> ExecutionConfig:
        """Enforce trail-parameter bounds; warn when trail distance is unusually tight.

        ``trail_activation_r_multiple`` may legally be 0.0 (meaning the STP
        converts to a TRAIL immediately at the scale-out price) but negative
        values are nonsense. ``trail_amount_r_multiple`` must be strictly
        positive — a zero-distance trail would follow price tick-for-tick
        and exit on the first down-tick. A trail distance below 0.5× initial_risk
        is unusually tight and tends to whip out on normal pullbacks, so we warn.
        """
        if self.trail_activation_r_multiple < 0.0:
            raise ValueError(
                f"execution.trail_activation_r_multiple must be >= 0.0 "
                f"(got {self.trail_activation_r_multiple}); negative values are nonsensical."
            )
        if self.trail_amount_r_multiple <= 0.0:
            raise ValueError(
                f"execution.trail_amount_r_multiple must be > 0.0 "
                f"(got {self.trail_amount_r_multiple}); a zero-distance trail exits on "
                "the first down-tick."
            )
        if self.trail_amount_r_multiple < 0.5:
            structlog.get_logger("bot.config").warning(
                "config.trail_amount_r_multiple_tight",
                value=self.trail_amount_r_multiple,
                hint="Trail distance below 0.5R tends to whip out on normal pullbacks.",
            )
        return self

    @model_validator(mode="after")
    def _validate_initial_stop_adjustable_params(self) -> ExecutionConfig:
        """Phase 7.6: enforce sane bounds on the initial-stop trigger + trail multiples.

        Trigger must be > 0.0 (a 0-R trigger would convert immediately at
        entry fill — the STP would never provide initial-risk protection).
        Trail distance must be > 0.0 (same zero-distance argument as the
        post-scale trail). We also warn if the trigger is >= scale-out
        multiple, which would mean the initial-stop adjustment never
        fires before the LMT takes over — degenerate but legal.
        """
        if self.initial_stop_trigger_r_multiple <= 0.0:
            raise ValueError(
                f"execution.initial_stop_trigger_r_multiple must be > 0.0 "
                f"(got {self.initial_stop_trigger_r_multiple}); a 0-R trigger would "
                "convert the protective STP to a TRAIL at entry fill, eliminating "
                "initial-risk protection."
            )
        if self.initial_stop_trail_r_multiple <= 0.0:
            raise ValueError(
                f"execution.initial_stop_trail_r_multiple must be > 0.0 "
                f"(got {self.initial_stop_trail_r_multiple}); a zero-distance "
                "trail exits on the first down-tick post-conversion."
            )
        if (
            self.initial_stop_adjustable_enabled
            and self.initial_stop_trigger_r_multiple >= self.scale_out_multiple
        ):
            structlog.get_logger("bot.config").warning(
                "config.initial_stop_trigger_geq_scale_out",
                trigger_r=self.initial_stop_trigger_r_multiple,
                scale_out_r=self.scale_out_multiple,
                hint=(
                    "initial_stop_trigger_r_multiple >= scale_out_multiple means the "
                    "scale-out LMT fires before (or at) the initial-stop conversion, "
                    "so the adjustable fields never take effect in practice."
                ),
            )
        return self

    @model_validator(mode="after")
    def _validate_entry_limit_buffer(self) -> ExecutionConfig:
        """Reject negative entry-limit buffers; warn on unusually loose fills.

        ``entry_limit_buffer_usd`` is the per-share dollars between the STP
        trigger and the LMT ceiling on a BUY STP-LMT entry. Strictly
        negative values would invert the stop/limit relationship and IBKR
        would reject the order at placement. A buffer above $0.50 lets the
        fill sweep 5+ cents of ask depth on a thin book — that's a
        configuration choice, not a bug, but worth flagging at startup.
        """
        if self.entry_limit_buffer_usd < 0.0:
            raise ValueError(
                f"execution.entry_limit_buffer_usd must be >= 0.0 "
                f"(got {self.entry_limit_buffer_usd}); negative buffers invert the "
                "STP/LMT relationship and IBKR will reject the order."
            )
        if self.entry_limit_buffer_usd > 0.50:
            structlog.get_logger("bot.config").warning(
                "config.entry_limit_buffer_usd_loose",
                value=self.entry_limit_buffer_usd,
                hint="Buffer above $0.50 allows meaningful slippage above the trigger.",
            )
        return self

    @model_validator(mode="after")
    def _validate_lmt_buffer_params(self) -> ExecutionConfig:
        """Phase 8.2 — enforce sane bounds on the LMT entry buffer triplet.

        ``lmt_buffer_pct`` must be > 0 (a 0%/negative buffer would make
        the LMT not marketable, defeating the entry intent). The floor
        and cap must both be > 0 with cap > floor (otherwise the clamp
        is degenerate or inverted).
        """
        if self.lmt_buffer_pct <= 0.0:
            raise ValueError(
                f"execution.lmt_buffer_pct must be > 0.0 "
                f"(got {self.lmt_buffer_pct}); a 0% buffer makes the LMT "
                "unable to lift the offer, no fills."
            )
        if self.lmt_buffer_usd_floor <= 0.0:
            raise ValueError(
                f"execution.lmt_buffer_usd_floor must be > 0.0 "
                f"(got {self.lmt_buffer_usd_floor}); the floor exists to keep "
                "the buffer above typical bid-ask spreads on penny stocks."
            )
        if self.lmt_buffer_usd_cap <= self.lmt_buffer_usd_floor:
            raise ValueError(
                f"execution.lmt_buffer_usd_cap ({self.lmt_buffer_usd_cap}) must be "
                f"strictly greater than execution.lmt_buffer_usd_floor "
                f"({self.lmt_buffer_usd_floor}); cap <= floor degenerates the clamp."
            )
        # Phase 10.6: percentage ceiling must be strictly above the
        # percentage floor's mechanic — if ``max_pct <= buffer_pct``,
        # the ceiling would always bind and the floor logic becomes
        # unreachable, defeating the spread-clearing intent.
        if self.lmt_buffer_max_pct <= 0.0:
            raise ValueError(
                f"execution.lmt_buffer_max_pct must be > 0.0 "
                f"(got {self.lmt_buffer_max_pct}); the ceiling is the "
                "primary safety against IBKR's aggressive-LMT cap on "
                "low-priced names."
            )
        if self.lmt_buffer_max_pct <= self.lmt_buffer_pct:
            raise ValueError(
                f"execution.lmt_buffer_max_pct ({self.lmt_buffer_max_pct}) must be "
                f"strictly greater than execution.lmt_buffer_pct "
                f"({self.lmt_buffer_pct}); a ceiling at or below the raw "
                "percentage would always bind, making the floor logic "
                "unreachable."
            )
        return self


class SessionConfig(BaseModel):
    """Timezone and intraday window boundaries.

    Phase 6.1 adds ``loop_operation_timeout_seconds``: a per-IBKR-operation
    ceiling inside the orchestrator's per-iteration evaluation path. Day-2
    paper trading (2026-04-21) wedged silently after market open because a
    blocking await had no bound; one stalled symbol blocked the rest of the
    watchlist for the remainder of the session. The timeout gives each
    symbol's bar snapshot a detection window; a stall logs
    ``orchestrator.loop_stall`` and the loop moves on to the next symbol.
    """

    timezone: str = "America/New_York"
    premarket_scan_start: str = "07:00"
    trading_start: str = "09:30"
    trading_end: str = "11:30"
    flatten_all: str = "15:55"
    loop_operation_timeout_seconds: float = 10.0
    # Phase 6.2: continuous scanner rescan cadence + watchlist size cap. The
    # scanner fires at ``watchlist_rescan_interval_seconds`` intervals from
    # loop start through ``flatten_all``; the hard cap at
    # ``watchlist_max_size`` evicts the oldest non-position symbol when a new
    # gapper arrives and the book is full. Active-position symbols are exempt
    # from eviction.
    watchlist_rescan_interval_seconds: int = 300
    watchlist_max_size: int = 10
    # Phase 6.4: wall-time staleness guard. When the latest bar for a
    # subscribed symbol is older than this many seconds relative to NY
    # wall-clock, strategy evaluation is skipped and a single
    # ``strategy.bar_stale`` event is logged per (symbol, strategy) per
    # stale period. Detects halted symbols whose IBKR stream has stopped
    # producing bars but whose subscription is still live.
    bar_staleness_threshold_seconds: int = 180
    # Phase 10.4 — bar-source switch.
    #   ``ibkr_1min`` (default, pre-10.4 behavior): live bars come from
    #     ``reqHistoricalData(keepUpToDate=True, "1 min")``. Day-7 paper
    #     trading observed ~5,250 ms median latency between bar close
    #     and bar-received event (BIYA 2026-04-30 entry timeline).
    #   ``rtbars_5sec_aggregated``: live bars come from ``reqRealTimeBars(5)``
    #     plus an in-process 5-sec → 1-min aggregator
    #     (``bot.brokerage.bar_aggregator.RollingMinuteAggregator``). Measured
    #     against the same TWS environment, median bar-finalization
    #     latency dropped to ~312 ms (8-paired-minute spike on AAPL,
    #     ``scripts/measure_aggregator_to_submit.py``). Initial backfill
    #     still uses the historical 1-min path; only the live updates
    #     change.
    # Default flipped from ``ibkr_1min`` to ``rtbars_5sec_aggregated`` —
    # the aggregator path is the production source. ``ibkr_1min`` is
    # retained as an escape hatch (e.g., if a future TWS version regresses
    # ``reqRealTimeBars`` reliability).
    bar_source: Literal["ibkr_1min", "rtbars_5sec_aggregated"] = "rtbars_5sec_aggregated"

    @field_validator("loop_operation_timeout_seconds")
    @classmethod
    def _validate_loop_op_timeout(cls, value: float) -> float:
        """Reject non-positive timeouts; a zero/negative ceiling would fail immediately."""
        if value <= 0.0:
            raise ValueError(
                "session.loop_operation_timeout_seconds must be > 0.0 "
                f"(got {value}); a non-positive ceiling would time out every operation."
            )
        return value

    @field_validator("watchlist_rescan_interval_seconds")
    @classmethod
    def _validate_rescan_interval(cls, value: int) -> int:
        """Reject non-positive intervals; zero would rescan every iteration + hammer Finnhub."""
        if value <= 0:
            raise ValueError(
                "session.watchlist_rescan_interval_seconds must be > 0 "
                f"(got {value}); a zero/negative interval would rescan every iteration."
            )
        return value

    @field_validator("watchlist_max_size")
    @classmethod
    def _validate_watchlist_max_size(cls, value: int) -> int:
        """Reject non-positive caps; at least one subscription slot is required."""
        if value <= 0:
            raise ValueError(
                f"session.watchlist_max_size must be > 0 (got {value}); "
                "at least one subscription slot is required."
            )
        return value

    @field_validator("bar_staleness_threshold_seconds")
    @classmethod
    def _validate_bar_staleness_threshold(cls, value: int) -> int:
        """Reject non-positive thresholds; a zero ceiling would mark every bar stale."""
        if value <= 0:
            raise ValueError(
                f"session.bar_staleness_threshold_seconds must be > 0 (got {value}); "
                "a non-positive threshold would mark every bar stale immediately."
            )
        return value


class UniverseConfig(BaseModel):
    """5-Pillar universe filters from PLAN §2.1."""

    price_min: float = 1.0
    price_max: float = 20.0
    float_max: int = 20_000_000
    gap_pct_min: float = 4.0
    rvol_min: float = 5.0
    premarket_vol_min: int = 300_000


class GapAndGoConfig(BaseModel):
    """Gap-and-Go strategy parameters.

    Phase 5.5 adds ``vwap_extension_grace_minutes``: bars stamped within the
    first N minutes of ``session.trading_start`` bypass the 3× ATR
    ``extended_from_vwap`` rejection. Day-1 paper trading showed real gappers (ENVB, JLHL, FCHL) getting rejected at the open because
    distance-from-VWAP scales with gap size while ATR is suppressed by
    premarket bars' low range. the documented methodology treats the
    opening range breakout *as* the gap-and-go entry; VWAP distance is
    context, not a gate in the opening window. A value of 0 disables the
    grace period and restores pre-5.5 behaviour.
    """

    enabled: bool = True
    vwap_extension_grace_minutes: int = 15
    # Phase 6.6 — was 3.0 hardcoded inside ``is_extension_bar_atr``; Day 3
    # paper trading rejected normal continuation setups (ELPW, TORO, ZENA,
    # BTM, ENVB) because 3× ATR is structurally too tight for low-float
    # gappers where ATR is suppressed by thin premarket bars. 5× is the
    # calibrated default for a low-float universe; tests and
    # production paths read from here so the multiple is one config edit.
    extended_from_vwap_atr_multiple: float = 5.0
    # Phase 6.7 — configurable end of the strategy's evaluation window
    # (HH:MM, NY-local). Window start is always 09:30 ET (market open).
    # Default ``10:00`` preserves the the methodology-documented 30-min sweet spot;
    # tests / off-hours experimentation can widen this (e.g. ``"16:00"``)
    # without touching code. Validator enforces HH:MM + end > 09:30.
    window_end: str = "16:00"

    @field_validator("vwap_extension_grace_minutes")
    @classmethod
    def _validate_grace_minutes(cls, value: int) -> int:
        """Reject negative grace periods; 0 is legal (disables bypass)."""
        if value < 0:
            raise ValueError(
                "strategies.gap_and_go.vwap_extension_grace_minutes must be >= 0 "
                f"(got {value}); use 0 to restore pre-5.5 always-apply behaviour."
            )
        return value

    @field_validator("extended_from_vwap_atr_multiple")
    @classmethod
    def _validate_atr_multiple(cls, value: float) -> float:
        """Phase 6.6 — must be positive and ≤ 20× (above that defeats the check)."""
        if value <= 0:
            raise ValueError(
                "strategies.gap_and_go.extended_from_vwap_atr_multiple must be > 0 "
                f"(got {value}); a non-positive multiple would always reject."
            )
        if value > 20.0:
            raise ValueError(
                "strategies.gap_and_go.extended_from_vwap_atr_multiple must be <= 20.0 "
                f"(got {value}); 20× ATR on a low-float gapper is genuinely a climax, "
                "above that the check has no meaningful effect."
            )
        return value

    @field_validator("window_end")
    @classmethod
    def _validate_window_end(cls, value: str) -> str:
        """Phase 6.7 — HH:MM, strictly after 09:30 market open."""
        _parse_strategy_hh_mm("strategies.gap_and_go.window_end", value)
        return value


class MomentumConfig(BaseModel):
    """Momentum / bull-flag strategy parameters."""

    enabled: bool = True
    flag_max_pullback_pct: float = 5.0
    # Phase 6.6 — see GapAndGoConfig docstring for the same field. Momentum
    # has no grace-period bypass (it's an ongoing-intraday pattern, not an
    # opening-range play), so this multiple gates *every* momentum bar.
    extended_from_vwap_atr_multiple: float = 5.0
    # Phase 6.7 — configurable end of the momentum evaluation window
    # (HH:MM, NY-local). Window start is always 09:30 ET (market open).
    # Default ``11:30`` preserves the pre-6.7 hardcoded cutoff; widen for
    # off-hours testing (``"16:00"``) without touching code.
    window_end: str = "16:00"

    @field_validator("extended_from_vwap_atr_multiple")
    @classmethod
    def _validate_atr_multiple(cls, value: float) -> float:
        """Phase 6.6 — must be positive and ≤ 20× (above that defeats the check)."""
        if value <= 0:
            raise ValueError(
                "strategies.momentum.extended_from_vwap_atr_multiple must be > 0 "
                f"(got {value}); a non-positive multiple would always reject."
            )
        if value > 20.0:
            raise ValueError(
                "strategies.momentum.extended_from_vwap_atr_multiple must be <= 20.0 "
                f"(got {value}); 20× ATR on a low-float gapper is genuinely a climax, "
                "above that the check has no meaningful effect."
            )
        return value

    @field_validator("window_end")
    @classmethod
    def _validate_window_end(cls, value: str) -> str:
        """Phase 6.7 — HH:MM, strictly after 09:30 market open."""
        _parse_strategy_hh_mm("strategies.momentum.window_end", value)
        return value


def _parse_strategy_hh_mm(field_name: str, value: str) -> tuple[int, int]:
    """Phase 6.7 — shared HH:MM validator for per-strategy ``window_end``.

    Strategies evaluate bars only inside [09:30, window_end). The validator
    enforces the format and rejects values at-or-before 09:30 (a window
    that ends before it starts would evaluate zero bars).
    """
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"{field_name} must be HH:MM, got {value!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be HH:MM, got {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{field_name} must be a valid HH:MM, got {value!r}")
    if (hour, minute) <= (9, 30):
        raise ValueError(f"{field_name} must be strictly after 09:30 ET market open, got {value!r}")
    return hour, minute


class StopFloorConfig(BaseModel):
    """Phase 10.2 — minimum stop-distance floor on entry stop placement.

    The Phase 7.1 structural stop (lowest 10-RTH-bar low for momentum,
    ``min(VWAP, 3-bar pullback low)`` for gap-and-go) generally works,
    but has a pathological case: tight consolidation breakouts where the
    entry bar pierces the consolidation by a cent or two. 2026-04-30
    ZENA setup filled at $2.18 with a structural stop at $2.17 — a 1¢
    stop that any normal microstructure noise would tag immediately.

    The floor widens the placed stop to
    ``entry − max(min_abs, entry × min_pct)`` whenever the structural
    stop is closer to entry than that distance. The structural value is
    preserved in the ``signal.emitted`` log's ``pullback_low`` field
    for forensics; the placed ``stop`` reflects the floored value
    end-to-end so risk sizing, fill-anchored re-protection, and
    downstream stop watchdogs all see one consistent number.

    Sizing flows through naturally — the existing ``RiskEngine`` divides
    the per-trade risk budget by ``entry − stop`` to compute shares, so
    a wider stop produces a smaller position with no risk-module
    changes. The Phase 4c ``risk.max_stop_width_usd`` ($0.50) gate still
    applies to the floored stop, so a floor that pushes width above the
    rule-based threshold rejects the trade.

    Defaults: 5¢ absolute floor, 2% percentage floor — whichever is
    farther from entry wins. At $2.50 the two are equal; below $2.50
    the absolute floor binds, above $2.50 the percentage floor binds
    (when either binds at all).
    """

    min_abs: float = 0.05
    min_pct: float = 0.02

    @field_validator("min_abs", "min_pct")
    @classmethod
    def _validate_non_negative(cls, value: float) -> float:
        """Reject negative floor values; 0.0 is legal (disables that branch)."""
        if value < 0.0:
            raise ValueError(
                f"strategies.stop_floor.* must be >= 0.0 (got {value}); use 0.0 to "
                "disable that branch (e.g. ``min_pct=0.0`` for fixed-cents-only flooring)."
            )
        return value


class StrategiesConfig(BaseModel):
    """Container for per-strategy configs."""

    gap_and_go: GapAndGoConfig = Field(default_factory=GapAndGoConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    # Phase 6.6 — emit ``strategy.extension_check_passed`` on every passing
    # extension check so an operator can grep ``extension_ratio`` across a
    # session to recalibrate the multiple. OFF by default because a busy
    # 10-symbol × 2-strategy × ~100-bars/session day is ~2000 events of
    # "nothing happened, check passed". Flip ON during calibration windows.
    log_extension_check_passes: bool = False
    # Phase 8.4 — when True, both strategies cap their scale-out target at
    # ``min(entry + N×R, premarket_high − $0.01)`` if a premarket high
    # exists above the entry price. Locks in profit just below well-known
    # intraday resistance instead of trying to push through it. When the
    # gapper has already cleared PMH (entry > PMH) the cap can't bind and
    # the standard 2R target stands. Phase 8.3's fill-anchored protection
    # uses ``position.entry_trigger_price`` directly so the cap doesn't
    # break the post-fill STP/scale-LMT placement math.
    premarket_high_cap_enabled: bool = True
    # Phase 10.2 — minimum stop-distance floor. Shared across both
    # strategies because the pathological-tight-stop case (2026-04-30
    # ZENA) is a property of the breakout/consolidation pattern, not of
    # the per-strategy stop reference. See StopFloorConfig.
    stop_floor: StopFloorConfig = Field(default_factory=StopFloorConfig)


class NameExtensionConfig(BaseModel):
    """Phase 10.5 — catalyst-attribution name-extension controls.

    Phase 9.7 established a precision gate: a catalyst-bearing news item
    is only attributed to a symbol when the ticker appears in the
    headline. That correctly rejects wrap articles that Finnhub mistags
    to an unrelated symbol (2026-04-30 BIYA precedent: a Coca-Cola
    earnings wrap article matched ``"raises guidance"`` for BIYA).

    Phase 10.5 widens recall on the same gate. Real press releases for
    sub-$20 names sometimes name the company directly without the
    cashtag — 2026-05-01 SHPH ("Shuttle Pharmaceutical Enters Definite
    Agreement..."), 2026-04-29 RPGL ("Republic Power Group Acquires
    10% Stake..."), 2026-04-21 ELSE ("Electro-Sensors, Inc. to be
    acquired..."). With name extension, the ticker-not-in-headline
    rejection now falls back to a name-token match against
    :class:`ContractDetails.longName`. Gate 2 (ticker-anchored
    proximity) is unchanged.

    Tokenisation produces a per-symbol list of "name signature" words
    by:
      1. lowercasing ``longName``,
      2. splitting on any non-alphanumeric run (whitespace, hyphen,
         slash — so ``RE/MAX HOLDINGS`` becomes ``["re", "max",
         "holdings"]`` and ``ELECTRO-SENSORS INC`` becomes
         ``["electro", "sensors", "inc"]``),
      3. dropping tokens shorter than ``min_token_len`` and any in
         ``stopwords``.

    The defaults below (5-char minimum, corporate-suffix + generic-
    financial stopwords) reflect the 2026-05-01
    ``scripts/measure_longname_match_rate.py`` analysis: tighter than
    3 chars to keep generic short tokens like "max" out of the
    matcher, broad enough to catch the actual recall wins.
    """

    stopwords: list[str] = Field(
        default_factory=lambda: [
            # Corporate suffixes
            "incorporated",
            "corporation",
            "company",
            "limited",
            "holdings",
            "group",
            "trust",
            "partners",
            # Fund / ETF markers
            "etf",
            "fund",
            "spdr",
            "ishares",
            # Generic financial-context words too common to anchor on
            "international",
            "national",
            "american",
            "global",
            "united",
            "first",
            "common",
            "shares",
            "class",
            # Geographic — too broad
            "america",
            "states",
        ]
    )
    min_token_len: int = 5
    high_rate_threshold: int = 10

    @field_validator("min_token_len")
    @classmethod
    def _validate_min_token_len(cls, value: int) -> int:
        """Reject ``min_token_len < 1`` — would accept empty-string tokens."""
        if value < 1:
            raise ValueError(
                f"catalyst.name_extension.min_token_len must be >= 1 (got {value}); "
                "values < 1 would admit empty-string tokens that match every headline."
            )
        return value

    @field_validator("high_rate_threshold")
    @classmethod
    def _validate_high_rate_threshold(cls, value: int) -> int:
        """Reject non-positive thresholds; ``catalyst.name_extension_high_rate``
        wouldn't fire meaningfully at 0 or below."""
        if value < 1:
            raise ValueError(
                f"catalyst.name_extension.high_rate_threshold must be >= 1 "
                f"(got {value}); 0 or negative would fire on every rescue."
            )
        return value


class CatalystConfig(BaseModel):
    """Phase 10.5 — container for catalyst-classifier sub-configs.

    Currently holds only ``name_extension``; future catalyst tuning knobs
    (e.g. proximity-window override, per-category attribution toggles)
    would land here.
    """

    name_extension: NameExtensionConfig = Field(default_factory=NameExtensionConfig)


class _ExitEventClassConfig(BaseModel):
    """Common shape for an event-class block: a master ``enabled`` flag plus
    free-form per-class fields. Subclasses add the fields specific to that class.

    Spike (exit-advisor): Layer 1 wires the harness to a subset of event
    classes; classes whose ``enabled`` flag is True but whose implementation
    has not yet shipped raise at config-validate time so a stale YAML can
    never silently look like it works.
    """


class _ExitPriceLevelsConfig(_ExitEventClassConfig):
    enabled: bool = True  # layer 2
    hod_lod: bool = True
    prior_day_high_low: bool = True
    prior_day_close: bool = True
    premarket_high_low: bool = False
    round_numbers: bool = False
    gap_fill: bool = True
    gap_threshold_pct: float = 0.01
    """Minimum |today_open - prior_close| / prior_close for the gap_fill
    level to be meaningful. 0.01 = 1% — anything tighter and the gap is
    indistinguishable from normal noise."""


class _ExitMovingAveragesConfig(_ExitEventClassConfig):
    enabled: bool = True  # layer 2
    vwap: bool = True
    ema_9: bool = True
    sma_200: bool = False


class _ExitVolumeConfig(_ExitEventClassConfig):
    enabled: bool = True  # layer 2
    spike_threshold_x_avg: float = 2.0
    dryup_threshold_x_avg: float = 0.4
    rvol_milestones: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])
    baseline_window_bars: int = 20
    """Rolling N-bar within-session baseline for spike/dryup detection.
    Choice of within-session over today-vs-prior-days is deliberate:
    bot's setups inherently trade days where today's volume is abnormally
    high vs prior days, so the relative comparison that matters during
    the trade is "is this bar unusual for this trade right now"."""

    rvol_lookback_days: int = 10
    """Number of prior trading days to average for RVOL milestones. If
    fewer days of session logs are available, RVOL degrades gracefully
    (one-shot RVolDataUnavailable warning, no milestone events)."""


class _ExitBarShapeConfig(_ExitEventClassConfig):
    enabled: bool = True  # layer 2
    shapes: list[str] = Field(
        default_factory=lambda: [
            "doji",
            "hammer",
            "shooting_star",
            "engulfing",
            "inside_bar",
            "outside_bar",
        ]
    )
    wick_threshold_pct: float = 0.6
    consecutive_bars_threshold: int = 3


class _ExitTimeConfig(_ExitEventClassConfig):
    enabled: bool = True
    milestones_minutes_after_open: list[int] = Field(default_factory=lambda: [5, 30, 120])
    time_in_trade_milestones: list[int] = Field(default_factory=lambda: [2, 5, 10, 30])


class _ExitPnLConfig(_ExitEventClassConfig):
    enabled: bool = True
    r_multiples: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0])
    drawdown_pct_from_peak: list[float] = Field(default_factory=lambda: [0.25, 0.5])
    track_mfe: bool = True


class _ExitOrderStateConfig(_ExitEventClassConfig):
    enabled: bool = True
    partial_fills: bool = True
    order_rejections: bool = True


class _ExitMarketContextConfig(_ExitEventClassConfig):
    enabled: bool = False
    leaderboard: bool = False
    broad_market: bool = False
    peers: bool = False


class _ExitNewsConfig(_ExitEventClassConfig):
    enabled: bool = False


class _ExitHaltsConfig(_ExitEventClassConfig):
    enabled: bool = False


class _ExitL2BidOfferPullsConfig(BaseModel):
    enabled: bool = True
    lookback_ms: int = 100


class _ExitL2AbsorptionConfig(BaseModel):
    enabled: bool = True
    refresh_multiplier: float = 3.0


class _ExitL2SpreadConfig(BaseModel):
    enabled: bool = True
    widening_ratio: float = 2.0
    tightening_ratio: float = 0.5
    rolling_window_events: int = 20


class _ExitL2ImbalanceConfig(BaseModel):
    enabled: bool = True
    threshold_ratio: float = 3.0
    levels_to_sum: int = 5


class _ExitL2PrintClustersConfig(BaseModel):
    enabled: bool = True
    window_seconds: float = 10.0
    min_prints: int = 5


class _ExitL2LargePrintsConfig(BaseModel):
    enabled: bool = True
    size_multiplier: float = 5.0
    rolling_window_prints: int = 50


class _ExitL2Config(_ExitEventClassConfig):
    """Layer L2-A: order-book + tick-by-tick prints. Activatable per
    sub-detector. Defaults pulled from the layer L2-A spec; tunable in
    config.yaml. Each sub-detector is a nested model so its threshold
    fields stay co-located with the toggle."""

    enabled: bool = True
    bid_offer_pulls: _ExitL2BidOfferPullsConfig = Field(
        default_factory=_ExitL2BidOfferPullsConfig
    )
    absorption: _ExitL2AbsorptionConfig = Field(default_factory=_ExitL2AbsorptionConfig)
    spread_events: _ExitL2SpreadConfig = Field(default_factory=_ExitL2SpreadConfig)
    imbalance: _ExitL2ImbalanceConfig = Field(default_factory=_ExitL2ImbalanceConfig)
    print_clusters: _ExitL2PrintClustersConfig = Field(
        default_factory=_ExitL2PrintClustersConfig
    )
    large_prints: _ExitL2LargePrintsConfig = Field(default_factory=_ExitL2LargePrintsConfig)


class ExitEventsConfig(BaseModel):
    """Spike (exit-advisor) — taxonomy of events the exit-advisor harness
    consumes during trade replay.

    Layer 1 implements the ``time``, ``pnl``, and ``order_state`` classes.
    Other classes are placeholders for later layers; a config that flips
    one of them on is rejected here so the harness never silently treats
    a no-op class as "passing".
    """

    price_levels: _ExitPriceLevelsConfig = Field(default_factory=_ExitPriceLevelsConfig)
    moving_averages: _ExitMovingAveragesConfig = Field(default_factory=_ExitMovingAveragesConfig)
    volume: _ExitVolumeConfig = Field(default_factory=_ExitVolumeConfig)
    bar_shape: _ExitBarShapeConfig = Field(default_factory=_ExitBarShapeConfig)
    time: _ExitTimeConfig = Field(default_factory=_ExitTimeConfig)
    pnl: _ExitPnLConfig = Field(default_factory=_ExitPnLConfig)
    order_state: _ExitOrderStateConfig = Field(default_factory=_ExitOrderStateConfig)
    market_context: _ExitMarketContextConfig = Field(default_factory=_ExitMarketContextConfig)
    news: _ExitNewsConfig = Field(default_factory=_ExitNewsConfig)
    halts: _ExitHaltsConfig = Field(default_factory=_ExitHaltsConfig)
    l2: _ExitL2Config = Field(default_factory=_ExitL2Config)

    _DEFERRED_CLASSES: tuple[str, ...] = (
        "market_context",
        "news",
        "halts",
    )

    @model_validator(mode="after")
    def _reject_deferred_layers(self) -> ExitEventsConfig:
        """Layer-3-and-beyond gating: refuse a config that enables an
        event class whose implementation has not shipped yet. Layer 2
        moved price_levels / moving_averages / volume / bar_shape out of
        this list."""
        offenders = [
            name
            for name in self._DEFERRED_CLASSES
            if getattr(getattr(self, name), "enabled", False)
        ]
        if offenders:
            raise ValueError(
                "exit_events: classes enabled in config but not implemented in this layer: "
                f"{', '.join(offenders)}. Set enabled: false until the corresponding layer "
                "ships, or remove the override."
            )
        return self


class ExitGatesConfig(BaseModel):
    """Spike (exit-advisor, layer 3) — risk gate framework.

    Hard guardrails (StopProtection, NoReentry, ProtectedPosition,
    NakedPosition, MaxHoldTime) are not configured here; their *behavior*
    is non-negotiable. Only their parameters and the soft gates' tunables
    live in this section. ``max_hold_minutes`` is the threshold the
    MaxHoldTimePolicy + MaxHoldTimeGuardrail share — the value is
    configurable, the rule (force exit at threshold) is not.

    Disabling the whole layer (``enabled: false``) bypasses both the
    gate chain AND the built-in MaxHoldTimePolicy. Use only for tests
    and ablation studies; production should keep this True.
    """

    enabled: bool = True
    max_hold_minutes: int = 60
    confidence_threshold: float = 0.7
    drawdown_acceleration_pct: float = 0.5
    drawdown_reduced_confidence_threshold: float = 0.5
    recency_throttle_seconds: int = 30
    min_r_for_partial: float = 1.0
    min_r_for_stop_tighten: float = 0.5

    @model_validator(mode="after")
    def _validate_ranges(self) -> ExitGatesConfig:
        if self.max_hold_minutes < 1:
            raise ValueError("exit_gates.max_hold_minutes must be >= 1")
        for field_name in (
            "confidence_threshold",
            "drawdown_acceleration_pct",
            "drawdown_reduced_confidence_threshold",
        ):
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"exit_gates.{field_name} must be in [0, 1] (got {value})")
        for field_name in (
            "recency_throttle_seconds",
            "min_r_for_partial",
            "min_r_for_stop_tighten",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"exit_gates.{field_name} must be non-negative (got {value})")
        return self


class TestingConfig(BaseModel):
    """Phase 6.8 — paper-trading-only test/debug switches.

    The fields here enable operator-in-the-loop tooling whose semantics
    would contaminate live trading (manual catalyst injection, future
    deterministic-time override, etc.). The top-level ``testing:`` section
    is off by default; paper configs opt in, and live configs MUST leave
    everything at the defaults. Each field individually also checks the
    gate at runtime so even a stale artifact on disk can't influence the
    live path if the flag slips back to false.
    """

    allow_catalyst_overrides: bool = False
    """Gate for ``bot inject-catalyst`` and scanner-side override application.

    When false (default + required in live configs), both the CLI
    injection command and the scanner's override lookup short-circuit
    before touching the store. When true, the scanner consults
    ``data/test_catalyst_overrides.json`` before the Finnhub fetch and
    applies any active entry for the symbol.
    """

    allow_force_entry: bool = False
    """Phase 6.13 — gate for the ``bot force-entry`` paper-testing CLI.

    ``force-entry`` synthesises a Signal and hands it to the executor
    directly, bypassing scanner + strategy evaluation. Useful for
    validating Phase 6.12 MKT entries, Phase 6.9 tick rounding, and
    post-fill protection-children planting without waiting on a real
    breakout. Double-gated: this flag AND ``account.mode == "paper"``
    are both required. MUST be false in live configs.
    """


class IBKRConfig(BaseModel):
    """IBKR TWS/Gateway connection parameters."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17


class WatchdogConfig(BaseModel):
    """Phase 10.1 — naked-position watchdog.

    Detects (does not auto-remediate) any tracked position whose IBKR-side
    working SELL orders fall short of "at least one protective STP / STP-LMT
    / TRAIL whose total quantity covers the position size". Catches failures
    regardless of root cause: bot state drift, OCA proportional-reduction
    collateral damage on partial stop fills (the 2026-04-30 BIYA scenario),
    operator manual cancels in TWS, or hypothetical IBKR-side cancels.

    ``shadow_mode`` defaults True so the first session post-deploy validates
    the detector against real session data without spamming Telegram on
    edge cases the test suite didn't anticipate. The operator flips it to
    False after one clean session of shadow logs (look for
    ``watchdog.shadow_alert_skipped`` events; if zero unexpected fires, ship
    it). See README/runbook for the procedure.

    ``entry_grace_seconds`` is the window from the watchdog's first
    observation of a position in ``open`` status (Phase 8.3 fill-anchored
    protection-planting completes within ~1 s in normal conditions; 30 s
    leaves margin for slow IBKR ack on thin SCM names).
    """

    enabled: bool = True
    shadow_mode: bool = True
    check_interval_seconds: float = 5.0
    entry_grace_seconds: float = 30.0

    @field_validator("check_interval_seconds")
    @classmethod
    def _validate_check_interval(cls, value: float) -> float:
        """Reject non-positive cadence; a zero/negative interval would re-evaluate every tick."""
        if value <= 0.0:
            raise ValueError(
                f"watchdog.check_interval_seconds must be > 0.0 (got {value}); "
                "use a small positive value (e.g. 1.0) if you want near-every-tick evaluation."
            )
        return value

    @field_validator("entry_grace_seconds")
    @classmethod
    def _validate_entry_grace(cls, value: float) -> float:
        """Reject negative grace; 0.0 is legal (alert immediately on first naked observation)."""
        if value < 0.0:
            raise ValueError(
                f"watchdog.entry_grace_seconds must be >= 0.0 (got {value}); use 0.0 to "
                "disable the post-fill grace and alert on the first naked observation."
            )
        return value


class ExitAdvisorConfig(BaseModel):
    """Phase 11 — exit-advisor hook surface configuration.

    Production main ships with ``enabled: false`` and ``hook_acts:
    false``. With ``enabled=false``, every notify call in the hook
    package short-circuits and the bot's behaviour is identical to
    pre-Phase-11. With ``enabled=true`` but ``hook_acts=false``, the
    hook fires and recommendations are logged but never executed
    against IBKR — the "log-only" diagnostic mode.

    The spike branch flips both ``enabled`` and ``hook_acts`` to true
    in its own config to actually drive exits from its advisor
    implementation.

    ``timeout_seconds`` bounds how long any single advisor call can
    run before the hook wrapper abandons it (worker thread continues
    but the bot moves on). Default 10 s leaves headroom for an LLM
    round-trip without freezing the bar-evaluation loop indefinitely.

    ``log_skipped_events`` controls the high-volume forensic path:
    every event the advisor *didn't* reason about. Default true for
    diagnostics; flip false in busy sessions where L2-event volume
    would otherwise dwarf the JSONL.
    """

    enabled: bool = False
    hook_acts: bool = False
    timeout_seconds: float = 10.0
    log_skipped_events: bool = True

    # --- Live LLM advisor (bot/exit_advisor/advisor) configuration ---
    # All fields below are only consumed when ``enabled=true``. ANTHROPIC_API_KEY
    # is read from the environment at bootstrap (never from this config).
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 8.0
    cost_soft_cap_usd: float = 10.0
    cost_hard_cap_usd: float = 50.0
    event_buffer_time_floor_seconds: float = 30.0
    event_buffer_hard_floor_seconds: float = 10.0
    self_disable_failure_rate: float = 0.5
    self_disable_min_calls: int = 5

    @field_validator("timeout_seconds", "llm_timeout_seconds")
    @classmethod
    def _validate_positive_timeout(cls, value: float) -> float:
        """Reject non-positive timeouts; a zero/negative cap would abandon every call."""
        if value <= 0.0:
            raise ValueError(
                "exit_advisor timeout fields must be > 0.0 "
                f"(got {value}); non-positive timeouts would abandon every advisor call."
            )
        return value

    @field_validator("cost_soft_cap_usd", "cost_hard_cap_usd")
    @classmethod
    def _validate_positive_cap(cls, value: float) -> float:
        """Reject non-positive caps; a zero/negative cap would always be tripped."""
        if value <= 0.0:
            raise ValueError(
                f"exit_advisor cost caps must be > 0.0 USD (got {value})."
            )
        return value

    @field_validator("event_buffer_time_floor_seconds", "event_buffer_hard_floor_seconds")
    @classmethod
    def _validate_non_negative_floor(cls, value: float) -> float:
        """Reject negatives; 0.0 is legal (no floor — every event triggers immediately)."""
        if value < 0.0:
            raise ValueError(
                f"exit_advisor event-buffer floors must be >= 0.0 seconds (got {value})."
            )
        return value

    @field_validator("self_disable_failure_rate")
    @classmethod
    def _validate_failure_rate(cls, value: float) -> float:
        """Failure rate is a probability in (0.0, 1.0]; 0.0 would disable on first failure."""
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"exit_advisor.self_disable_failure_rate must be in (0.0, 1.0] (got {value})."
            )
        return value

    @field_validator("self_disable_min_calls", "llm_max_tokens")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"exit_advisor positive-int field must be > 0 (got {value}).")
        return value

    @model_validator(mode="after")
    def _validate_hook_acts_requires_enabled(self) -> ExitAdvisorConfig:
        """``hook_acts=true`` is meaningless without ``enabled=true``.

        Catch the obvious YAML typo at startup instead of having the
        operator wonder why ``hook_acts: true`` did nothing.
        """
        if self.hook_acts and not self.enabled:
            raise ValueError(
                "exit_advisor.hook_acts=true requires exit_advisor.enabled=true; "
                "the hook can't act on recommendations it never solicits."
            )
        return self

    @model_validator(mode="after")
    def _validate_cost_caps_ordered(self) -> ExitAdvisorConfig:
        """Soft cap must precede hard cap; otherwise the soft warning never fires."""
        if self.cost_soft_cap_usd >= self.cost_hard_cap_usd:
            raise ValueError(
                "exit_advisor.cost_soft_cap_usd "
                f"({self.cost_soft_cap_usd}) must be strictly less than "
                f"cost_hard_cap_usd ({self.cost_hard_cap_usd}); the soft warning "
                "must fire before the hard cap to be useful."
            )
        return self


class DataSourcesSettings(BaseModel):
    """External API credentials + news window tuning (Phase 5.1).

    Phase 5.1 adds two news-window knobs: ``news_lookback_hours`` widens the
    Finnhub fetch window so weekend + Friday catalysts survive into Monday's
    scan, while ``news_max_age_hours_for_classify`` bounds how old a headline
    can be before the classifier ignores it. Widening the fetch alone would
    let a 3-day-old earnings beat flag a stock that's moving today for
    unrelated reasons; the classifier-side filter prevents that without
    narrowing the fetch back.

    Defaults:
      * ``news_lookback_hours = 96`` (4 days) — minimum that always covers
        Friday late + Saturday + Sunday news for a Monday scan.
      * ``news_max_age_hours_for_classify = 72`` (3 days) — items older than
        this are filtered out before keyword matching, so stale green-list
        phrases can't misattribute today's move.
    """

    finnhub_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    news_lookback_hours: int = 96
    news_max_age_hours_for_classify: int = 72

    @model_validator(mode="after")
    def _validate_news_windows(self) -> DataSourcesSettings:
        """Reject non-positive windows and catch the obvious classify > fetch misconfig.

        A classifier window larger than the fetch window is always a bug:
        items beyond the fetch don't exist, so raising the classify cap
        above the fetch has no effect and hides intent. A fetch of 0 would
        mean "never see news"; a classify of 0 would mean "reject all news
        as stale" — both guaranteed-no-catalyst states the operator almost
        certainly didn't mean.
        """
        if self.news_lookback_hours < 1:
            raise ValueError(
                f"data_sources.news_lookback_hours must be >= 1 (got {self.news_lookback_hours})."
            )
        if self.news_max_age_hours_for_classify < 1:
            raise ValueError(
                "data_sources.news_max_age_hours_for_classify must be >= 1 "
                f"(got {self.news_max_age_hours_for_classify})."
            )
        if self.news_max_age_hours_for_classify > self.news_lookback_hours:
            raise ValueError(
                "data_sources.news_max_age_hours_for_classify "
                f"({self.news_max_age_hours_for_classify}) exceeds news_lookback_hours "
                f"({self.news_lookback_hours}); the classifier cannot consider items "
                "that were never fetched."
            )
        return self


class LoggingSettings(BaseModel):
    """Phase 5.1 — structlog + optional JSONL file-handler configuration.

    Previously, ``configure_logging`` hardcoded ``stream=sys.stdout`` and no
    ``logging`` key existed in ``Settings``, so any ``logging:`` block in
    ``config.yaml`` was silently dropped. Phase 5.1 wires a real
    ``LoggingSettings`` plus ``Settings.model_config(extra="forbid")`` so
    unknown top-level YAML keys now fail at startup instead of vanishing.

    Fields:
      * ``level`` — stdlib log level name (``"DEBUG" | "INFO" | ...``).
      * ``json`` — True selects the JSON renderer (the Phase 1 default);
        False drops to the dev ConsoleRenderer for human-friendly output.
      * ``path`` — directory for per-session JSONL files. When None (the
        default), no file handler is attached and behaviour matches the
        pre-5.1 stdout-only setup. When set, a file handler is attached
        at ``{path}/session_{YYYY-MM-DD}.jsonl`` where the date uses the
        ``session.timezone`` (so sessions don't split across UTC midnight).
    """

    level: str = "INFO"
    # Field name in YAML is ``json`` but we store it as ``json_renderer`` to
    # avoid shadowing pydantic's ``BaseModel.json`` method. The alias keeps
    # the YAML surface unchanged.
    json_renderer: bool = Field(default=True, alias="json")
    path: Path | None = None

    model_config = {"populate_by_name": True}

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        """Accept only stdlib-recognised level names; normalise to upper case."""
        normalised = value.upper()
        if normalised not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"logging.level must be DEBUG|INFO|WARNING|ERROR|CRITICAL (got {value!r})."
            )
        return normalised


class Settings(BaseSettings):
    """Top-level settings loaded from ``config.yaml`` with ``.env`` / env-var overrides.

    Precedence (highest first): explicit constructor args, environment variables
    (``BOT_*`` prefix, ``__`` as nesting delimiter), ``.env`` file, ``config.yaml``,
    field defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="BOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        # Phase 5.1: unknown top-level YAML keys must fail at startup. The
        # prior ``extra="ignore"`` silently dropped a ``logging:`` block the
        # operator added to config.yaml for weeks, masking the bug that
        # motivated Phase 5.1's file-logging fix. ``forbid`` surfaces drift
        # the moment it's introduced.
        extra="forbid",
    )

    # Legacy ``.env`` key retained for backwards-compat with existing
    # operator setups. The YAML path is hardwired in ``model_config`` above,
    # so this value is never read — but it would trip ``extra="forbid"`` if
    # declared as unknown. Exposed as a typed no-op instead of silently
    # swallowed so the presence of the key is visible in settings dumps.
    config_file: str | None = None

    # Phase 11 LLM advisor — the Anthropic API key is read by
    # ``bot.exit_advisor.advisor.bootstrap`` directly from ``os.environ``
    # (after a ``load_dotenv()`` call there). It is declared here as a
    # typed no-op so the presence of ``ANTHROPIC_API_KEY`` in ``.env``
    # doesn't trip ``extra="forbid"`` — same pattern as ``config_file``
    # above. Settings.anthropic_api_key is intentionally never consumed
    # by the bot; the canonical reader is the bootstrap function.
    anthropic_api_key: str | None = None

    account: AccountConfig = Field(default_factory=AccountConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    data_sources: DataSourcesSettings = Field(default_factory=DataSourcesSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    catalyst: CatalystConfig = Field(default_factory=CatalystConfig)
    exit_advisor: ExitAdvisorConfig = Field(default_factory=ExitAdvisorConfig)
    exit_events: ExitEventsConfig = Field(default_factory=ExitEventsConfig)
    exit_gates: ExitGatesConfig = Field(default_factory=ExitGatesConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Wire in YAML as a lower-priority source than env vars and ``.env``."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached)."""
    return Settings()


def warn_on_missing_data_source_credentials(settings: Settings | None = None) -> None:
    """Emit a structlog warning (not error) for each missing external-data credential."""
    settings = settings or get_settings()
    log = structlog.get_logger("bot.config")
    if not settings.data_sources.finnhub_api_key:
        log.warning(
            "config.finnhub_api_key_missing",
            hint="Set BOT_DATA_SOURCES__FINNHUB_API_KEY to enable float + news filters.",
        )
    if not settings.data_sources.telegram_bot_token or not settings.data_sources.telegram_chat_id:
        log.warning(
            "config.telegram_credentials_missing",
            hint="Set BOT_DATA_SOURCES__TELEGRAM_BOT_TOKEN and BOT_DATA_SOURCES__TELEGRAM_CHAT_ID "
            "to enable Telegram pushes.",
        )
