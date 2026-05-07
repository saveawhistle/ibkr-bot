"""Phase 12.5 tests -- IBKR Error 202 (aggressive-LMT cancel) recovery + market-anchored buffer ceiling.

Three sub-suites:

1. ``test_parse_aggressive_limit_ceiling_*`` -- regex parser unit tests
   for the IBKR Error 202 message body. The string format has been
   stable across TWS releases since at least 2020; we still test against
   the actual message captured from the 2026-05-07 FEED incident.
2. ``test_compute_lmt_buffer_breakdown_*`` -- the anchor-aware
   ``min(entry, anchor) × max_pct`` ceiling math (Option 2 of the
   recovery story). Confirms legacy anchor=None numerics are preserved
   and that anchor < entry tightens the ceiling without ever loosening it.
3. ``test_executor_lmt_retry_*`` -- end-to-end retry path: synthetic
   ``ib.errorEvent`` dispatch into the Executor's handler, with
   assertions on the new bracket placement, single-retry guard, the
   below-entry skip, the disabled-flag short-circuit, and the
   non-aggressive 202 (operator cancel) no-op path.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from bot.execution.executor import (
    _compute_lmt_buffer_breakdown,
    _LmtRetryContext,
    _parse_aggressive_limit_ceiling,
)

# ---------- 1. Parser ----------


# The actual message from the 2026-05-07 FEED Error 202 incident.
_REAL_FEED_MESSAGE = (
    "Order Canceled - reason:We cannot accept an order at a limit price at "
    "or more aggressive than 1.6046172. Please submit your order <br>using "
    "a limit price that is closer to the current market price of 1.4614. "
    "or convert your order to an <br>algorithmic Order (IBALGO)"
)


def test_parser_extracts_ceiling_from_real_feed_message() -> None:
    """Sanity: parse the actual message captured in production."""
    assert _parse_aggressive_limit_ceiling(_REAL_FEED_MESSAGE) == pytest.approx(1.6046172)


def test_parser_returns_none_on_unrelated_202_message() -> None:
    """Code 202 with a different reason (operator cancel) returns None -- no retry path triggered."""
    msg = "Order Canceled - reason:Cancelled by user (TWS)"
    assert _parse_aggressive_limit_ceiling(msg) is None


def test_parser_returns_none_on_empty_message() -> None:
    assert _parse_aggressive_limit_ceiling("") is None


def test_parser_handles_integer_ceiling() -> None:
    """The regex permits an integer (no decimal); IBKR sometimes rounds."""
    msg = "limit price at or more aggressive than 7. Please submit a closer order."
    assert _parse_aggressive_limit_ceiling(msg) == pytest.approx(7.0)


def test_parser_handles_lowercase_message() -> None:
    """Defensive: regex is case-insensitive."""
    assert (
        _parse_aggressive_limit_ceiling(_REAL_FEED_MESSAGE.lower())
        == pytest.approx(1.6046172)
    )


# ---------- 2. Anchor-aware buffer ceiling ----------


def test_breakdown_no_anchor_preserves_legacy_numerics() -> None:
    """anchor=None falls back to entry-only ceiling -- pre-12.5 behaviour."""
    bd = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.15,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=None,
    )
    # Floor would otherwise produce 0.20 from the percentage cap; ceiling is
    # min($0.50 dollar cap, $10 × 7% = $0.70) = $0.50; floor binds.
    assert bd.ceiling_value == pytest.approx(0.50)
    assert bd.final == pytest.approx(0.20)  # max(pct_raw=$0.20, floor=$0.15) = $0.20
    assert bd.clamp == "none"


def test_breakdown_anchor_below_entry_tightens_ceiling() -> None:
    """anchor < entry: ceiling computed against anchor, not entry. The FEED case."""
    # FEED parameters: entry $1.51, anchor (prior bar close) $1.48.
    # Without anchor: ceiling = min($0.50, $1.51 × 7% = $0.1057) = $0.1057.
    # With anchor: ceiling = min($0.50, $1.48 × 7% = $0.1036) = $0.1036.
    bd_no_anchor = _compute_lmt_buffer_breakdown(
        entry_price=1.51,
        buffer_pct=2.0,
        buffer_floor_usd=0.15,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=None,
    )
    bd_with_anchor = _compute_lmt_buffer_breakdown(
        entry_price=1.51,
        buffer_pct=2.0,
        buffer_floor_usd=0.15,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=1.48,
    )
    assert bd_no_anchor.ceiling_value == pytest.approx(0.1057, abs=1e-3)
    assert bd_with_anchor.ceiling_value == pytest.approx(0.1036, abs=1e-3)
    assert bd_with_anchor.ceiling_value < bd_no_anchor.ceiling_value


def test_breakdown_anchor_above_entry_does_not_loosen_ceiling() -> None:
    """anchor > entry: should NEVER widen the ceiling -- defensive guard."""
    # If the anchor logic naively used `min(anchor, entry)` we'd be safe;
    # the implementation actually checks `anchor < entry` and falls back
    # to entry otherwise. Both produce the same answer here.
    bd_high_anchor = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.05,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=15.0,  # above entry
    )
    bd_no_anchor = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.05,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=None,
    )
    assert bd_high_anchor.ceiling_value == bd_no_anchor.ceiling_value


def test_breakdown_anchor_zero_or_negative_is_ignored() -> None:
    """Defensive: a zero/negative anchor (bad data) falls back to entry."""
    bd_zero = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.05,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=0.0,
    )
    bd_negative = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.05,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=-3.0,
    )
    bd_baseline = _compute_lmt_buffer_breakdown(
        entry_price=10.0,
        buffer_pct=2.0,
        buffer_floor_usd=0.05,
        buffer_cap_usd=0.50,
        max_pct=7.0,
        anchor_price=None,
    )
    assert bd_zero.ceiling_value == bd_baseline.ceiling_value
    assert bd_negative.ceiling_value == bd_baseline.ceiling_value


# ---------- 3. End-to-end retry handler ----------


def _make_minimal_executor(
    *,
    lmt_aggressive_limit_retry: bool = True,
) -> tuple[Any, MagicMock]:
    """Build an Executor with only the bits the retry handler actually touches.

    No TWS, no real risk engine -- the handler doesn't reach into those
    paths. We synthesize ``_active_trades`` and ``_lmt_retry_contexts``
    directly to drive the unit-under-test.
    """
    from bot.config import (
        AccountConfig,
        ExecutionConfig,
        RiskConfig,
        Settings,
    )
    from bot.execution.executor import Executor
    from bot.execution.position_state import PositionStore
    from bot.persistence.journal import Journal
    from bot.risk.engine import RiskEngine

    ib = MagicMock(name="ib")
    # ``errorEvent`` supports ``+=`` via __iadd__ -> we capture the listener.
    ib.errorEvent = _RecordingEvent()
    ibkr = MagicMock()
    ibkr.ib = ib
    settings = Settings(
        account=AccountConfig(mode="paper"),
        execution=ExecutionConfig(
            entry_order_type="LMT",
            require_paper_mode=True,
            lmt_aggressive_limit_retry=lmt_aggressive_limit_retry,
        ),
        risk=RiskConfig(max_loss_per_trade_usd=100.0),
    )
    risk_engine = RiskEngine(settings=settings)
    store = PositionStore()
    journal = MagicMock(spec=Journal)
    executor = Executor(
        ibkr=cast("Any", ibkr),
        position_store=store,
        journal=journal,
        risk_engine=risk_engine,
        settings=settings,
    )
    return executor, ib


class _RecordingEvent:
    """Minimal stand-in for ib_async's eventkit Event with __iadd__ subscribe."""

    def __init__(self) -> None:
        self.listeners: list[Any] = []

    def __iadd__(self, listener: Any) -> _RecordingEvent:
        self.listeners.append(listener)
        return self

    def fire(self, *args: Any, **kwargs: Any) -> None:
        for listener in self.listeners:
            listener(*args, **kwargs)


def _make_retry_context(
    *,
    symbol: str = "FEED",
    entry: float = 1.51,
    original_lmt: float = 1.62,
    market_anchor_price: float | None = 1.48,
) -> _LmtRetryContext:
    from datetime import UTC, datetime

    return _LmtRetryContext(
        symbol=symbol,
        contract=MagicMock(symbol=symbol),
        entry=entry,
        stop=entry - 0.07,
        target=entry + 0.10,
        shares=100,
        strategy="momentum",
        original_limit_price=original_lmt,
        market_anchor_price=market_anchor_price,
        placement_bar_ts=datetime.now(UTC),
    )


def test_handler_ignores_non_202_codes() -> None:
    """Other IBKR errors (200, 365, etc) must NOT trigger any retry path."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context()
    executor._lmt_retry_contexts[1125] = context

    # Fire a 200-series error against the same reqId.
    ib.errorEvent.fire(1125, 200, "No security definition has been found.")
    assert context.retried is False
    # Context untouched.
    assert 1125 in executor._lmt_retry_contexts


def test_handler_ignores_202_with_unrelated_reason() -> None:
    """Code 202 fires for many reasons; only the aggressive-limit text triggers retry."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context()
    executor._lmt_retry_contexts[1125] = context

    ib.errorEvent.fire(1125, 202, "Order Canceled - reason:Cancelled by user")
    assert context.retried is False
    assert 1125 in executor._lmt_retry_contexts


def test_handler_ignores_202_for_unknown_req_id() -> None:
    """An aggressive-limit 202 for a parent we don't track is a no-op."""
    executor, ib = _make_minimal_executor()
    # No retry context registered.
    ib.errorEvent.fire(9999, 202, _REAL_FEED_MESSAGE)
    assert executor._lmt_retry_contexts == {}


def test_handler_replaces_bracket_at_corrected_lmt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aggressive-limit 202 → handler re-places bracket at IBKR's ceiling minus a tick."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context()
    executor._lmt_retry_contexts[1125] = context

    # Stub the placement so we capture the force_limit_price without
    # touching the real placeOrder / contract qualification chain.
    placed_calls: list[dict[str, Any]] = []

    def _fake_place(**kwargs: Any) -> Any:
        placed_calls.append(kwargs)
        new_bracket = MagicMock()
        new_bracket.parent.order.orderId = 2001
        new_bracket.stop.order.orderId = 2002
        new_bracket.target = None
        return new_bracket

    monkeypatch.setattr(executor, "_place_bracket", _fake_place)
    monkeypatch.setattr(executor, "_wire_fill_handlers", lambda *a, **k: None)

    ib.errorEvent.fire(1125, 202, _REAL_FEED_MESSAGE)

    assert len(placed_calls) == 1
    call = placed_calls[0]
    # IBKR ceiling 1.6046172, minus 1¢ → $1.59 (rounded down to 2 decimals).
    assert call["force_limit_price"] == pytest.approx(1.59)
    assert call["entry"] == pytest.approx(1.51)
    assert call["stop"] == pytest.approx(1.44)
    assert call["shares"] == 100
    # Retry guard set; original key removed; new key registered.
    assert context.retried is True
    assert 1125 not in executor._lmt_retry_contexts
    assert executor._lmt_retry_contexts[2001] is context
    # New bracket installed in active_trades.
    assert executor._active_trades["FEED"].parent.order.orderId == 2001


def test_handler_skips_when_corrected_lmt_below_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If IBKR's ceiling sits below entry, the breakout already faded -- skip retry."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context(entry=1.60, original_lmt=1.72)
    executor._lmt_retry_contexts[1125] = context

    placed_calls: list[Any] = []
    monkeypatch.setattr(
        executor, "_place_bracket", lambda **k: placed_calls.append(k) or MagicMock()
    )
    monkeypatch.setattr(executor, "_wire_fill_handlers", lambda *a, **k: None)

    # Suggested ceiling 1.55, entry 1.60. 1.55 - 0.01 = 1.54 < entry → skip.
    msg = "limit price at or more aggressive than 1.55. closer to current market price"
    ib.errorEvent.fire(1125, 202, msg)
    assert placed_calls == []
    # Context dropped (no point keeping it around).
    assert 1125 not in executor._lmt_retry_contexts


def test_handler_blocks_second_retry_on_same_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the retried bracket also gets a 202, we MUST NOT retry again (loop guard)."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context()
    executor._lmt_retry_contexts[1125] = context

    placed_calls: list[Any] = []

    def _fake_place(**kwargs: Any) -> Any:
        placed_calls.append(kwargs)
        new_bracket = MagicMock()
        new_bracket.parent.order.orderId = 2001
        new_bracket.stop.order.orderId = 2002
        new_bracket.target = None
        return new_bracket

    monkeypatch.setattr(executor, "_place_bracket", _fake_place)
    monkeypatch.setattr(executor, "_wire_fill_handlers", lambda *a, **k: None)

    # First 202 → retried.
    ib.errorEvent.fire(1125, 202, _REAL_FEED_MESSAGE)
    assert len(placed_calls) == 1
    # Second 202 (against the new parent) → blocked.
    ib.errorEvent.fire(2001, 202, _REAL_FEED_MESSAGE)
    assert len(placed_calls) == 1  # NOT incremented
    # Context dropped after the failed second-retry attempt.
    assert 2001 not in executor._lmt_retry_contexts


def test_disabled_flag_skips_subscription() -> None:
    """When the config flag is off, the handler must NOT be subscribed."""
    executor, ib = _make_minimal_executor(lmt_aggressive_limit_retry=False)
    assert ib.errorEvent.listeners == []


def test_handler_does_not_crash_on_internal_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_do_lmt_retry`` raises, the handler logs and swallows -- broker connection unaffected."""
    executor, ib = _make_minimal_executor()
    context = _make_retry_context()
    executor._lmt_retry_contexts[1125] = context

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(executor, "_do_lmt_retry", _boom)

    # Should NOT raise.
    ib.errorEvent.fire(1125, 202, _REAL_FEED_MESSAGE)


# ---------- 4. ScanHit-side: signal carries the anchor ----------


def test_signal_carries_market_anchor_price_default_none() -> None:
    """Pre-12.5 callsites that don't set the field still construct cleanly."""
    from datetime import UTC, datetime

    from bot.strategies.base import Signal

    sig = Signal(
        symbol="X",
        strategy="t",
        entry=10.0,
        stop=9.0,
        scale_out_price=11.0,
        runner_target_price=None,
        timestamp=datetime.now(UTC),
    )
    assert sig.market_anchor_price is None


def test_signal_carries_market_anchor_price_when_set() -> None:
    from datetime import UTC, datetime

    from bot.strategies.base import Signal

    sig = Signal(
        symbol="X",
        strategy="t",
        entry=10.0,
        stop=9.0,
        scale_out_price=11.0,
        runner_target_price=None,
        timestamp=datetime.now(UTC),
        market_anchor_price=9.85,
    )
    assert sig.market_anchor_price == pytest.approx(9.85)
