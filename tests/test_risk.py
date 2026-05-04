"""Tests for ``bot.risk`` — sizing, gates, halt persistence, PDT advisory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from bot.config import AccountConfig, ExecutionConfig, RiskConfig, Settings
from bot.execution.position_state import Position, PositionStore
from bot.risk import (
    Approved,
    HaltRecord,
    Rejected,
    RiskEngine,
    compute_shares,
    daily_loss_hit,
    delete_halt_flag,
    giveback_hit,
    profit_goal_hit,
    read_halt_flag,
    write_halt_flag,
)
from bot.strategies.base import Signal


def _settings(**overrides: float | int) -> Settings:
    """Build a Settings with RiskConfig overrides for a test."""
    defaults = {
        "max_loss_per_trade_usd": 100.0,
        "max_position_value_usd": 25_000.0,  # kept generous here; margin test relies on it
        "max_daily_loss_usd": 300.0,
        "daily_profit_goal_usd": 500.0,
        "giveback_trigger_usd": 400.0,
        "giveback_pct": 50.0,
        "max_concurrent_positions": 1,
        "max_trades_per_day": 5,
        # Existing pre-Phase-4c tests use $1 stop widths. Keep the cap
        # permissive here so only Phase 4c tests that override it exercise
        # the stop-width gate.
        "max_stop_width_usd": 100.0,
        "max_pct_of_bar_volume": 2.0,
        "extension_bar_trigger_multiple": 2.0,
    }
    defaults.update(overrides)
    base = Settings()
    return base.model_copy(
        update={
            "account": AccountConfig(mode="paper"),
            "execution": ExecutionConfig(rth_only=True, require_paper_mode=True),
            "risk": RiskConfig(**defaults),  # type: ignore[arg-type]
        }
    )


def _signal(
    *,
    symbol: str = "TEST",
    entry: float = 10.0,
    stop: float = 9.0,
    target: float = 13.0,
) -> Signal:
    """3:1 R:R signal with a $1 per-share risk.

    ``target`` is mapped onto both ``scale_out_price`` and
    ``runner_target_price`` so existing call-sites keep the same R:R
    without needing to know about the Phase 4e split.
    """
    return Signal(
        symbol=symbol,
        strategy="gap_and_go",
        entry=entry,
        stop=stop,
        scale_out_price=target,
        runner_target_price=target,
        timestamp=datetime(2026, 4, 16, 9, 31, tzinfo=UTC),
    )


def _summary(
    *,
    available: float = 1_000_000,
    buying_power: float = 2_000_000,
    day_trades_remaining: int = -1,
) -> dict[str, str]:
    """IBKR account-summary stub with generous defaults."""
    return {
        "AvailableFunds": str(available),
        "BuyingPower": str(buying_power),
        "NetLiquidation": str(available),
        "DayTradesRemaining": str(day_trades_remaining),
    }


def _closed_position(symbol: str = "TEST") -> Position:
    """Minimal closed Position for ``on_fill_closed``."""
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
        status="closed",
    )


# ---------- Pure function tests ---------- #


def test_compute_shares_ross_rule_floors() -> None:
    """Entry 10.50, stop 10.00 → per-share risk $0.50; $100 budget → 200 shares."""
    shares = compute_shares(_signal(entry=10.5, stop=10.0), 100.0, 25_000.0)
    assert shares == 200


def test_compute_shares_capped_by_max_position_value() -> None:
    """Entry 5.00 with $100 budget + $0.10 risk/sh = 1000 by risk; cap $2500 = 500."""
    sig = _signal(entry=5.0, stop=4.9)
    shares = compute_shares(sig, 100.0, 2500.0)
    assert shares == 500


def test_compute_shares_zero_on_invalid_risk() -> None:
    """entry <= stop → risk per share ≤ 0 → zero shares (caller rejects)."""
    sig = _signal(entry=10.0, stop=10.0)
    assert compute_shares(sig, 100.0, 25_000.0) == 0


def test_daily_loss_profit_giveback_predicates() -> None:
    """Boundary checks for the three halt predicates."""
    assert daily_loss_hit(-300.0, 300.0) is True
    assert daily_loss_hit(-299.99, 300.0) is False
    assert profit_goal_hit(500.0, 500.0) is True
    assert profit_goal_hit(499.99, 500.0) is False
    # giveback: peak must cross the trigger *and* current must bleed below threshold.
    assert giveback_hit(200.0, 300.0, 400.0, 50.0) is False  # peak below trigger
    assert giveback_hit(200.0, 500.0, 400.0, 50.0) is True  # 50% of 500 = 250; 200 ≤ 250
    assert giveback_hit(300.0, 500.0, 400.0, 50.0) is False  # 300 > 250


# ---------- Gate tests ---------- #


@pytest.mark.asyncio
async def test_check_entry_approves_with_fresh_budget(tmp_path: Path) -> None:
    """Fresh session + valid signal → Approved with correct share count."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    decision = await engine.check_entry(_signal(), store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 100  # $100 / $1 risk per share


@pytest.mark.asyncio
async def test_check_entry_rejects_when_halted(tmp_path: Path) -> None:
    """A halted engine rejects with ``halted`` and no counter increment."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    # Trip the halt via a big simulated loss.
    await engine.on_fill_closed(_closed_position(), -400.0)
    store = PositionStore()
    decision = await engine.check_entry(_signal(), store, _summary())
    assert isinstance(decision, Rejected)
    assert decision.reason == "halted"


@pytest.mark.asyncio
async def test_check_entry_rejects_on_max_trades_per_day(tmp_path: Path) -> None:
    """trades_today >= limit → rejected with ``max_trades_per_day_exceeded``.

    Phase 9.6: ``trades_today`` increments on confirmed fill, not approval.
    Two simulated fills consume the budget; the third entry is rejected.
    """
    engine = RiskEngine(
        settings=_settings(max_trades_per_day=2), halt_flag_path=tmp_path / "halt.flag"
    )
    store = PositionStore()
    await engine.check_entry(_signal(symbol="A"), store, _summary())
    await engine.on_first_fill("A")
    await engine.check_entry(_signal(symbol="B"), store, _summary())
    await engine.on_first_fill("B")
    decision = await engine.check_entry(_signal(symbol="C"), store, _summary())
    assert isinstance(decision, Rejected)
    assert decision.reason == "max_trades_per_day_exceeded"


@pytest.mark.asyncio
async def test_check_entry_rejects_on_max_concurrent_positions(tmp_path: Path) -> None:
    """Store already has N active positions → rejected with ``max_concurrent_positions_exceeded``."""
    engine = RiskEngine(
        settings=_settings(max_concurrent_positions=1), halt_flag_path=tmp_path / "halt.flag"
    )
    store = PositionStore()
    store.open_position(
        symbol="EXISTING",
        strategy="gap_and_go",
        shares=100,
        stop_price=9.0,
        scale_out_price=12.0,
        runner_target_price=13.0,
        parent_order_id=1,
        stop_order_id=2,
        target_order_id=3,
        opened_at=datetime.now(UTC),
    )
    decision = await engine.check_entry(_signal(), store, _summary())
    assert isinstance(decision, Rejected)
    assert decision.reason == "max_concurrent_positions_exceeded"


@pytest.mark.asyncio
async def test_check_entry_rejects_on_margin_awareness(tmp_path: Path) -> None:
    """Position value exceeds AvailableFunds * 0.95 → reject with ``margin_awareness_exceeded``."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    # shares=100, entry=10 → position value = $1000; available funds = $1000 → cap $950.
    decision = await engine.check_entry(_signal(), store, _summary(available=1000.0))
    assert isinstance(decision, Rejected)
    assert decision.reason == "margin_awareness_exceeded"


@pytest.mark.asyncio
async def test_check_entry_emits_pdt_advisory_but_does_not_block(tmp_path: Path) -> None:
    """DayTradesRemaining=0 → ``pdt.advisory`` logged with warning level, entry still approved."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    with capture_logs() as captured:
        decision = await engine.check_entry(_signal(), store, _summary(day_trades_remaining=0))
    assert isinstance(decision, Approved)
    pdt = [e for e in captured if e.get("event") == "pdt.advisory"]
    assert len(pdt) == 1
    assert pdt[0]["day_trades_remaining"] == 0


# ---------- Halt lifecycle ---------- #


@pytest.mark.asyncio
async def test_on_fill_closed_trips_daily_loss_halt(tmp_path: Path) -> None:
    """A loss past the daily cap flips the halt flag + writes ``logs/halt.flag``."""
    path = tmp_path / "halt.flag"
    engine = RiskEngine(settings=_settings(), halt_flag_path=path)
    await engine.on_fill_closed(_closed_position(), -350.0)
    assert engine.is_halted() is True
    assert engine.state.halt_reason == "daily_loss_limit"
    record = read_halt_flag(path)
    assert record is not None
    assert record.reason == "daily_loss_limit"
    assert record.pnl_at_halt == pytest.approx(-350.0)


@pytest.mark.asyncio
async def test_on_fill_closed_trips_giveback_halt(tmp_path: Path) -> None:
    """Peak past trigger + bleed past 50% → ``giveback_limit``."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_fill_closed(_closed_position(), 450.0)  # peak = 450 > trigger 400
    # Don't hit the daily profit goal ($500): peak stays at 450.
    await engine.on_fill_closed(_closed_position(), -250.0)  # current = 200 <= 225 (50% of 450)
    assert engine.is_halted() is True
    assert engine.state.halt_reason == "giveback_limit"


@pytest.mark.asyncio
async def test_apply_halt_flag_if_current_adopts_same_day(tmp_path: Path) -> None:
    """A same-date flag on disk is adopted on startup; engine is halted."""
    path = tmp_path / "halt.flag"
    engine = RiskEngine(settings=_settings(), halt_flag_path=path)
    record = HaltRecord(
        date=engine.state.session_date,
        reason="daily_loss_limit",
        triggered_at=datetime.now(UTC),
        pnl_at_halt=-320.0,
    )
    write_halt_flag(path, record)
    applied = await engine.apply_halt_flag_if_current()
    assert applied is not None
    assert engine.is_halted() is True


@pytest.mark.asyncio
async def test_apply_halt_flag_if_current_cleans_stale(tmp_path: Path) -> None:
    """A flag with a stale date auto-cleans; engine remains un-halted."""
    from datetime import date as date_cls

    path = tmp_path / "halt.flag"
    engine = RiskEngine(settings=_settings(), halt_flag_path=path)
    stale = HaltRecord(
        date=date_cls(2020, 1, 1),
        reason="daily_loss_limit",
        triggered_at=datetime.now(UTC),
        pnl_at_halt=-350.0,
    )
    write_halt_flag(path, stale)
    applied = await engine.apply_halt_flag_if_current()
    assert applied is None
    assert engine.is_halted() is False
    assert not path.exists()


@pytest.mark.asyncio
async def test_reset_for_new_session_clears_counters_not_flag(tmp_path: Path) -> None:
    """reset_for_new_session zeros state but does NOT delete the flag (operator friction)."""
    path = tmp_path / "halt.flag"
    engine = RiskEngine(settings=_settings(), halt_flag_path=path)
    await engine.on_fill_closed(_closed_position(), -400.0)
    assert engine.is_halted()
    await engine.reset_for_new_session()
    assert engine.is_halted() is False
    assert engine.state.realized_pnl_usd == 0.0
    assert path.exists()  # halt file untouched; operator must explicitly reset
    delete_halt_flag(path)
    assert not path.exists()


# ---------- Phase 4c: stop-width + liquidity + position-value caps ---------- #


@pytest.mark.asyncio
async def test_stop_width_cap_rejects_wide_stops(tmp_path: Path) -> None:
    """$1.50 width → rejected with ``stop_too_wide``; $0.40 width → approved."""
    settings = _settings(max_stop_width_usd=0.50)
    engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()

    wide = _signal(entry=10.0, stop=8.5, target=16.0)  # width $1.50
    decision = await engine.check_entry(wide, store, _summary())
    assert isinstance(decision, Rejected)
    assert decision.reason == "stop_too_wide"
    assert decision.detail["stop_width_usd"] == pytest.approx(1.50)
    assert decision.detail["max"] == pytest.approx(0.50)

    tight = _signal(entry=10.0, stop=9.6, target=11.6)  # width $0.40
    decision = await engine.check_entry(tight, store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares > 0


@pytest.mark.asyncio
async def test_liquidity_cap_binds(tmp_path: Path) -> None:
    """50k by risk, 20k by value, 5k by liquidity → final 5000 + log event fires."""
    # risk_per_share = 0.01 → 50k shares by risk on $500 budget; $200k value cap → 20k;
    # liquidity = 250_000 * 2% = 5000. Liquidity is the binding constraint.
    settings = _settings(
        max_loss_per_trade_usd=500.0,
        max_position_value_usd=200_000.0,
        max_stop_width_usd=100.0,
        max_pct_of_bar_volume=2.0,
    )
    engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    sig = Signal(
        symbol="LIQ",
        strategy="gap_and_go",
        entry=10.0,
        stop=9.99,
        scale_out_price=10.20,
        runner_target_price=10.20,
        timestamp=datetime(2026, 4, 16, 9, 31, tzinfo=UTC),
        recent_bar_volume=250_000,
    )
    with capture_logs() as captured:
        decision = await engine.check_entry(sig, store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 5000
    cap_events = [e for e in captured if e.get("event") == "risk.shares_capped_by_liquidity"]
    assert len(cap_events) == 1
    assert cap_events[0]["shares_by_liquidity"] == 5000


@pytest.mark.asyncio
async def test_liquidity_cap_null_volume_skipped(tmp_path: Path) -> None:
    """recent_bar_volume=None → liquidity path skipped; sized on risk + value only."""
    settings = _settings()  # permissive stop-width default
    engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    sig = _signal()  # recent_bar_volume defaults to None
    assert sig.recent_bar_volume is None
    with capture_logs() as captured:
        decision = await engine.check_entry(sig, store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 100  # $100 budget / $1 risk-per-share
    assert not any(e.get("event") == "risk.shares_capped_by_liquidity" for e in captured)


@pytest.mark.asyncio
async def test_liquidity_cap_does_not_fire_when_ample(tmp_path: Path) -> None:
    """Ample bar volume → no cap log; shares still determined by risk."""
    settings = _settings()
    engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    sig = Signal(
        symbol="AMPLE",
        strategy="gap_and_go",
        entry=10.0,
        stop=9.0,
        scale_out_price=13.0,
        runner_target_price=13.0,
        timestamp=datetime(2026, 4, 16, 9, 31, tzinfo=UTC),
        recent_bar_volume=50_000,  # 2% = 1000 shares; risk gives only 100 → no cap
    )
    with capture_logs() as captured:
        decision = await engine.check_entry(sig, store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 100
    assert not any(e.get("event") == "risk.shares_capped_by_liquidity" for e in captured)


@pytest.mark.asyncio
async def test_max_position_value_cap_binds(tmp_path: Path) -> None:
    """With $15k cap + $10 stock, shares_by_value = 1500 binds ahead of 2000 by risk."""
    # 2000 shares by risk needs risk-per-share = 100/2000 = $0.05.
    settings = _settings(max_position_value_usd=15_000.0, max_stop_width_usd=1.0)
    engine = RiskEngine(settings=settings, halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    sig = _signal(entry=10.0, stop=9.95, target=10.25)  # $0.05 risk → 2000 by risk
    decision = await engine.check_entry(sig, store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 1500  # floor(15000 / 10)


# ---------- Phase 4g: rehab-adjusted caps flow through RiskEngine ---------- #


@pytest.mark.asyncio
async def test_check_entry_sizes_against_rehab_adjusted_budget(tmp_path: Path) -> None:
    """A RehabEngine in REHAB tier halves the per-trade budget — 100 shares → 50."""
    from bot.risk.rehab import RehabEngine, RehabState, RehabTier

    settings = _settings()  # base $100 per-trade budget
    rehab = RehabEngine(settings=settings, flag_path=tmp_path / "rehab.flag")
    rehab.save_state(
        RehabState(
            tier=RehabTier.REHAB,
            trigger_reason="consecutive_red_days",
            entered_at=datetime.now(UTC),
            drawdown_at_entry_usd=-200.0,
            consecutive_red_days_at_entry=2,
        )
    )
    engine = RiskEngine(
        settings=settings,
        halt_flag_path=tmp_path / "halt.flag",
        rehab_engine=rehab,
    )
    store = PositionStore()
    # Base: $100 / $1 risk-per-share = 100 shares. REHAB halves the
    # budget to $50 → 50 shares. the rule "trade smaller in a
    # drawdown" enforced at the sizing boundary, not ad-hoc in the UI.
    decision = await engine.check_entry(_signal(), store, _summary())
    assert isinstance(decision, Approved)
    assert decision.shares == 50


# ---------- Phase 9.6: trades_today on fill + broker rejection lockout ---------- #


@pytest.mark.asyncio
async def test_trades_today_does_not_increment_on_signal_approval(tmp_path: Path) -> None:
    """Approval alone must not bump the counter — only confirmed fills do.

    Day 8 RPGL bug: three TWS-rejected placements consumed the daily
    trade budget despite zero fills.
    """
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    store = PositionStore()
    await engine.check_entry(_signal(), store, _summary())
    assert engine.state.trades_today == 0


@pytest.mark.asyncio
async def test_trades_today_increments_on_first_fill(tmp_path: Path) -> None:
    """``on_first_fill`` is the only counter-increment site post-9.6."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_first_fill("ABCD")
    assert engine.state.trades_today == 1


@pytest.mark.asyncio
async def test_trades_today_unaffected_by_broker_rejection(tmp_path: Path) -> None:
    """Broker rejection must NOT increment the daily-trade counter."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_broker_rejection("RPGL", error_code=10349)
    assert engine.state.trades_today == 0


@pytest.mark.asyncio
async def test_broker_rejection_increments_per_symbol_counter(tmp_path: Path) -> None:
    """One broker rejection increments to 1; symbol not yet blocked."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    just_blocked = await engine.on_broker_rejection("RPGL", error_code=10349)
    assert engine.state.broker_rejection_count["RPGL"] == 1
    assert just_blocked is False
    assert engine.is_symbol_blocked("RPGL") is False


@pytest.mark.asyncio
async def test_two_consecutive_rejections_block_symbol(tmp_path: Path) -> None:
    """Threshold (2) crossed → ``is_symbol_blocked`` true and just_blocked returns True once."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    first = await engine.on_broker_rejection("RPGL", error_code=10349)
    second = await engine.on_broker_rejection("RPGL", error_code=10349)
    third = await engine.on_broker_rejection("RPGL", error_code=10349)
    assert first is False
    assert second is True
    assert third is False  # already blocked, no re-emit
    assert engine.is_symbol_blocked("RPGL") is True


@pytest.mark.asyncio
async def test_blocked_symbol_rejected_at_risk_gate(tmp_path: Path) -> None:
    """A locked-out symbol's signals are rejected with ``symbol_blocked_broker_rejections``."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_broker_rejection("RPGL", error_code=10349)
    await engine.on_broker_rejection("RPGL", error_code=10349)
    assert engine.is_symbol_blocked("RPGL") is True

    decision = await engine.check_entry(_signal(symbol="RPGL"), PositionStore(), _summary())
    assert isinstance(decision, Rejected)
    assert decision.reason == "symbol_blocked_broker_rejections"


@pytest.mark.asyncio
async def test_blocked_symbol_does_not_block_others(tmp_path: Path) -> None:
    """RPGL lockout must not affect entry approvals on a different symbol.

    The Day 8 cascade prevented other symbols' signals from firing because
    the bot-wide trade counter was exhausted; under 9.6 the counter only
    moves on real fills, and the lockout is per-symbol.
    """
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_broker_rejection("RPGL", error_code=10349)
    await engine.on_broker_rejection("RPGL", error_code=10349)

    decision = await engine.check_entry(_signal(symbol="OTHER"), PositionStore(), _summary())
    assert isinstance(decision, Approved)


@pytest.mark.asyncio
async def test_first_fill_resets_broker_rejection_counter(tmp_path: Path) -> None:
    """A successful fill clears prior transient rejections so the symbol stays tradable."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    await engine.on_broker_rejection("RPGL", error_code=10349)
    assert engine.state.broker_rejection_count["RPGL"] == 1
    await engine.on_first_fill("RPGL")
    assert engine.state.broker_rejection_count.get("RPGL", 0) == 0
    assert engine.is_symbol_blocked("RPGL") is False


@pytest.mark.asyncio
async def test_broker_rejection_emits_warning_event(tmp_path: Path) -> None:
    """``executor.broker_rejection_detected`` carries symbol, code, count, threshold."""
    engine = RiskEngine(settings=_settings(), halt_flag_path=tmp_path / "halt.flag")
    with capture_logs() as captured:
        await engine.on_broker_rejection("RPGL", error_code=10349)
    rejections = [e for e in captured if e.get("event") == "executor.broker_rejection_detected"]
    assert len(rejections) == 1
    evt = rejections[0]
    assert evt["symbol"] == "RPGL"
    assert evt["error_code"] == 10349
    assert evt["rejection_count"] == 1
    assert evt["threshold"] == 2
    assert evt["blocked"] is False


@pytest.mark.asyncio
async def test_rpgl_cascade_prevented(tmp_path: Path) -> None:
    """Day 8 RPGL replay: 3 broker rejections + later legitimate signal on another symbol.

    Pre-9.6 the bot exhausted ``max_trades_per_day`` on RPGL placements that
    never filled, then rejected ALL subsequent signals (including a clean
    momentum breakout on OTHER) for the rest of the session.

    Post-9.6:
    - ``trades_today`` stays at 0 across all RPGL rejections (no fills).
    - After the 2nd rejection, RPGL is locked out at the risk gate.
    - The 3rd rejection still records but RPGL was already blocked — no
      duplicate watchlist-drop emission.
    - A signal on OTHER passes risk approval normally.
    """
    engine = RiskEngine(
        settings=_settings(max_trades_per_day=3),
        halt_flag_path=tmp_path / "halt.flag",
    )
    store = PositionStore()
    summary = _summary()

    # Three RPGL placements, all broker-rejected without fill.
    await engine.check_entry(_signal(symbol="RPGL"), store, summary)
    await engine.on_broker_rejection("RPGL", error_code=10349)
    await engine.check_entry(_signal(symbol="RPGL"), store, summary)
    just_blocked = await engine.on_broker_rejection("RPGL", error_code=10349)
    assert just_blocked is True

    # 3rd attempt is now blocked at the risk gate before the rejection
    # accounting fires. (In production the bracket also wouldn't be
    # placed; here we just verify the gate.)
    decision = await engine.check_entry(_signal(symbol="RPGL"), store, summary)
    assert isinstance(decision, Rejected)
    assert decision.reason == "symbol_blocked_broker_rejections"

    # Counter never moved.
    assert engine.state.trades_today == 0

    # OTHER still trades normally.
    decision = await engine.check_entry(_signal(symbol="OTHER"), store, summary)
    assert isinstance(decision, Approved)
