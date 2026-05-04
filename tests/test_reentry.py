"""Tests for Phase 4d: same-symbol pullback re-entries.

Covers:

* RiskEngine.check_reentry gate order (disabled / terminal / max / unprofitable /
  cooldown / multiplier).
* Size-multiplier sequence [1.0, 1.0, 0.5] → floor(adjusted_max_loss / risk).
* Distinct symbols are independent.
* PositionStore rebuild from journal-shaped rows.
* Session-start reset zeroes every history.

These tests poke the in-memory state directly instead of running brackets
through IBKR — the RiskEngine gates are pure functions of ``SymbolHistory``
and ``RiskConfig``, so mocking the IBKR layer would only obscure the gate
math.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from bot.config import (
    AccountConfig,
    ExecutionConfig,
    ReEntryConfig,
    RiskConfig,
    Settings,
)
from bot.execution.position_state import PositionStore, SymbolHistory
from bot.risk import Approved, ReEntryAllowed, Rejected, RiskEngine
from bot.strategies.base import Signal

_NOW = datetime(2026, 4, 16, 10, 0, tzinfo=UTC)


def _settings(
    *,
    enabled: bool = True,
    max_entries: int = 3,
    multipliers: list[float] | None = None,
    cooldown_seconds: int = 120,
    require_profitable_prior_exit: bool = True,
    max_loss_per_trade_usd: float = 100.0,
) -> Settings:
    """Build Settings with a ReEntryConfig override for the test."""
    re_entry = ReEntryConfig(
        enabled=enabled,
        max_entries_per_symbol=max_entries,
        size_multipliers=multipliers if multipliers is not None else [1.0, 1.0, 0.5],
        cooldown_seconds=cooldown_seconds,
        require_profitable_prior_exit=require_profitable_prior_exit,
    )
    base = Settings()
    return base.model_copy(
        update={
            "account": AccountConfig(mode="paper"),
            "execution": ExecutionConfig(rth_only=True, require_paper_mode=True),
            "risk": RiskConfig(
                max_loss_per_trade_usd=max_loss_per_trade_usd,
                max_stop_width_usd=100.0,
                re_entry=re_entry,
            ),
        }
    )


def _signal(symbol: str = "AAA", entry: float = 10.0, stop: float = 9.0) -> Signal:
    """3:1 R:R signal with $1 per-share risk."""
    return Signal(
        symbol=symbol,
        strategy="gap_and_go",
        entry=entry,
        stop=stop,
        scale_out_price=entry + 3.0 * (entry - stop),
        runner_target_price=entry + 3.0 * (entry - stop),
        timestamp=_NOW,
    )


def _summary() -> dict[str, str]:
    """Fat-headroom IBKR account summary so margin/buying-power never bind."""
    return {
        "AvailableFunds": "1000000",
        "BuyingPower": "2000000",
        "NetLiquidation": "1000000",
        "DayTradesRemaining": "-1",
    }


@pytest.mark.asyncio
async def test_first_entry_multiplier_is_one() -> None:
    """A symbol with no history → multiplier 1.0, entries_count 0."""
    engine = RiskEngine(settings=_settings())
    history = SymbolHistory(symbol="AAA")
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, ReEntryAllowed)
    assert decision.multiplier == 1.0
    assert decision.entries_count == 0


@pytest.mark.asyncio
async def test_second_and_third_entries_scale_down() -> None:
    """Progressive size reduction follows ``size_multipliers`` indexed by ``entries_count``."""
    engine = RiskEngine(settings=_settings(multipliers=[1.0, 0.75, 0.5]))
    history = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=_NOW - timedelta(seconds=200),
        last_exit_pnl=25.0,
        last_exit_type="target_hit",
    )
    second = await engine.check_reentry(_signal(), history)
    assert isinstance(second, ReEntryAllowed)
    assert second.multiplier == 0.75
    history.entries_count = 2
    third = await engine.check_reentry(_signal(), history)
    assert isinstance(third, ReEntryAllowed)
    assert third.multiplier == 0.5


@pytest.mark.asyncio
async def test_max_reentries_reached_rejects() -> None:
    """``entries_count >= max_entries_per_symbol`` → rejection carries the limit."""
    engine = RiskEngine(settings=_settings(max_entries=3))
    history = SymbolHistory(
        symbol="AAA",
        entries_count=3,
        last_exit_time=_NOW - timedelta(seconds=200),
        last_exit_pnl=25.0,
        last_exit_type="target_hit",
    )
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, Rejected)
    assert decision.reason == "max_reentries_reached"
    assert decision.detail["limit"] == 3
    assert decision.detail["entries_count"] == 3


@pytest.mark.asyncio
async def test_cooldown_active_rejects_before_expiry() -> None:
    """Inside the cooldown window → ``reentry_cooldown_active`` with seconds remaining."""
    engine = RiskEngine(settings=_settings(cooldown_seconds=120))
    history = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=datetime.now(UTC) - timedelta(seconds=30),
        last_exit_pnl=25.0,
        last_exit_type="target_hit",
    )
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, Rejected)
    assert decision.reason == "reentry_cooldown_active"
    assert decision.detail["cooldown_remaining_s"] > 0


@pytest.mark.asyncio
async def test_prior_exit_unprofitable_rejects() -> None:
    """With ``require_profitable_prior_exit`` on, a losing prior exit blocks the re-entry."""
    engine = RiskEngine(settings=_settings())
    history = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=_NOW - timedelta(seconds=200),
        last_exit_pnl=-15.0,
        last_exit_type="stop_hit",
    )
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, Rejected)
    assert decision.reason == "prior_exit_unprofitable"
    assert decision.detail["last_exit_pnl"] == -15.0


@pytest.mark.asyncio
async def test_reentry_disabled_blocks_after_first_entry() -> None:
    """Master switch off → first entry still allowed, second rejected with ``re_entry_disabled``."""
    engine = RiskEngine(settings=_settings(enabled=False))
    fresh = SymbolHistory(symbol="AAA")
    first = await engine.check_reentry(_signal(), fresh)
    assert isinstance(first, ReEntryAllowed)
    assert first.multiplier == 1.0
    seeded = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=_NOW - timedelta(seconds=300),
        last_exit_pnl=40.0,
        last_exit_type="target_hit",
    )
    second = await engine.check_reentry(_signal(), seeded)
    assert isinstance(second, Rejected)
    assert second.reason == "re_entry_disabled"


@pytest.mark.asyncio
async def test_auto_flatten_is_terminal_for_the_session() -> None:
    """A symbol whose last exit was ``auto_flatten`` can never re-enter that session."""
    engine = RiskEngine(settings=_settings())
    history = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=_NOW - timedelta(seconds=400),
        last_exit_pnl=0.0,
        last_exit_type="auto_flatten",
    )
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, Rejected)
    assert decision.reason == "auto_flattened_terminal"


@pytest.mark.asyncio
async def test_different_symbols_have_independent_histories() -> None:
    """AAA hitting the cap does not affect BBB's gate."""
    engine = RiskEngine(settings=_settings(max_entries=3))
    aaa = SymbolHistory(
        symbol="AAA",
        entries_count=3,
        last_exit_time=_NOW - timedelta(seconds=200),
        last_exit_pnl=30.0,
        last_exit_type="target_hit",
    )
    bbb = SymbolHistory(symbol="BBB")
    aaa_decision = await engine.check_reentry(_signal("AAA"), aaa)
    bbb_decision = await engine.check_reentry(_signal("BBB"), bbb)
    assert isinstance(aaa_decision, Rejected)
    assert aaa_decision.reason == "max_reentries_reached"
    assert isinstance(bbb_decision, ReEntryAllowed)
    assert bbb_decision.multiplier == 1.0


@pytest.mark.asyncio
async def test_check_entry_applies_multiplier_to_share_size() -> None:
    """The third entry uses multiplier 0.5 → compute_shares halves the risk budget."""
    settings = _settings(
        multipliers=[1.0, 1.0, 0.5],
        max_loss_per_trade_usd=100.0,
    )
    engine = RiskEngine(settings=settings)
    store = PositionStore()
    history = store.symbol_history("AAA")
    history.entries_count = 2
    history.last_exit_time = _NOW - timedelta(seconds=300)
    history.last_exit_pnl = 25.0
    history.last_exit_type = "target_hit"
    decision = await engine.check_entry(_signal(), store, _summary())
    assert isinstance(decision, Approved)
    # max_loss $100 * 0.5 = $50; risk per share $1 → 50 shares.
    assert decision.shares == 50


@pytest.mark.asyncio
async def test_scale_out_then_trail_prior_exit_classified_as_profitable() -> None:
    """A ``scale_out_then_trail`` exit with positive PnL passes the profitable-prior gate."""
    engine = RiskEngine(settings=_settings())
    history = SymbolHistory(
        symbol="AAA",
        entries_count=1,
        last_exit_time=_NOW - timedelta(seconds=300),
        last_exit_pnl=62.5,
        last_exit_type="scale_out_then_trail",
    )
    decision = await engine.check_reentry(_signal(), history)
    assert isinstance(decision, ReEntryAllowed)
    assert decision.multiplier == 1.0
    assert decision.entries_count == 1


def test_reset_symbol_histories_clears_every_symbol() -> None:
    """Session start drops every ``SymbolHistory`` so the day starts at entries_count=0."""
    store = PositionStore()
    store.symbol_history("AAA").record_entry()
    store.symbol_history("BBB").record_entry()
    assert len(store.list_symbol_histories()) == 2
    store.reset_symbol_histories()
    assert store.list_symbol_histories() == []


def test_rebuild_from_journal_reconstructs_entries_and_last_exit() -> None:
    """Journal rows in chronological order rebuild the store's in-memory histories."""
    store = PositionStore()
    rows = [
        SimpleNamespace(
            symbol="AAA",
            opened_at=_NOW,
            closed_at=_NOW + timedelta(minutes=5),
            pnl=25.0,
            exit_type="target_hit",
        ),
        SimpleNamespace(
            symbol="AAA",
            opened_at=_NOW + timedelta(minutes=10),
            closed_at=None,
            pnl=None,
            exit_type=None,
        ),
        SimpleNamespace(
            symbol="BBB",
            opened_at=_NOW + timedelta(minutes=2),
            closed_at=_NOW + timedelta(minutes=30),
            pnl=0.0,
            exit_type="auto_flatten",
        ),
    ]
    store.rebuild_symbol_histories_from_journal(rows)
    aaa = store.symbol_history("AAA")
    assert aaa.entries_count == 2  # both rows count as entries
    assert aaa.last_exit_type == "target_hit"
    assert aaa.last_exit_pnl == 25.0
    bbb = store.symbol_history("BBB")
    assert bbb.entries_count == 1
    assert bbb.last_exit_type == "auto_flatten"


def test_rebuild_ignores_unknown_exit_type_but_counts_entry() -> None:
    """Bad ``exit_type`` string → entry still counts, but exit metadata stays blank."""
    store = PositionStore()
    rows = [
        SimpleNamespace(
            symbol="AAA",
            opened_at=_NOW,
            closed_at=_NOW + timedelta(minutes=5),
            pnl=10.0,
            exit_type="NOT_A_REAL_TYPE",
        ),
    ]
    store.rebuild_symbol_histories_from_journal(rows)
    aaa = store.symbol_history("AAA")
    assert aaa.entries_count == 1
    assert aaa.last_exit_type is None
    assert aaa.last_exit_pnl is None


def test_rebuild_preserves_pre_scale_red_candle_exit_type() -> None:
    """Phase 7.8 ``pre_scale_red_candle`` exits must round-trip through journal replay.

    Regression for a latent bug surfaced during Phase 11 review: the
    if/elif/else narrowing chain in ``rebuild_symbol_histories_from_journal``
    accepted ``"pre_scale_red_candle"`` past the ``valid_types`` filter (the
    string is in the frozenset) but had no matching elif branch, so the
    rebuild silently downgraded it to ``"auto_flatten"`` via the fallthrough
    ``else``.

    Operational impact of the bug: ``RiskEngine._check_reentry_locked`` is
    the only consumer of ``last_exit_type`` that classifies an exit as
    terminal, and it does so *only* for ``"auto_flatten"`` (rejection
    reason ``auto_flattened_terminal``, "session_ending"). Live exits via
    ``trade_manager._execute_pre_scale_red_candle_exit`` correctly record
    ``"pre_scale_red_candle"``, which proceeds through the normal re-entry
    gates (cooldown, profitable-prior-exit, max-entries-per-symbol). After a
    crash-restart, the rebuilt history would fabricate an
    ``"auto_flatten"`` classification and **block all further entries on
    that symbol for the rest of the session** — silently diverging from the
    live behavior.

    The fix preserves the exit type verbatim so post-restart re-entry
    decisions match what would have happened without the crash.
    """
    store = PositionStore()
    rows = [
        SimpleNamespace(
            symbol="WLDS",
            opened_at=_NOW,
            closed_at=_NOW + timedelta(minutes=4),
            pnl=-3.50,
            exit_type="pre_scale_red_candle",
        ),
    ]
    store.rebuild_symbol_histories_from_journal(rows)
    history = store.symbol_history("WLDS")
    assert history.entries_count == 1
    assert history.last_exit_type == "pre_scale_red_candle", (
        "pre_scale_red_candle must round-trip through rebuild verbatim. "
        "If this fails as 'auto_flatten', the if/elif chain in "
        "rebuild_symbol_histories_from_journal is missing a branch and the "
        "exit is being silently downgraded — see test docstring for impact."
    )
    assert history.last_exit_pnl == pytest.approx(-3.50)
    assert history.last_exit_time == _NOW + timedelta(minutes=4)
