"""Tests for ``bot.execution.watchdog`` — Phase 10.1 naked-position detection + ack flow.

The watchdog is a detection-only safety floor: it inspects the bot's
position store and IBKR's working-order cache, classifies each tracked
position into PROTECTED / PROTECTED_PENDING / UNDERPROTECTED / NAKED,
and fires Telegram alerts (with an Ack button) for the bad classifications.
Auto-remediation is intentionally out of scope.

These tests exercise every classification branch + the BIYA verbatim
scenario from session_2026-04-30.jsonl, the ack/re-arm rules, the entry-
grace window, the shadow-mode bypass, and the position-state-mismatch
event.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from structlog.testing import capture_logs

from bot.config import Settings, WatchdogConfig
from bot.execution.position_state import Position, PositionStore
from bot.execution.watchdog import Watchdog
from bot.notify import Notifier

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _settings(
    *,
    enabled: bool = True,
    shadow_mode: bool = False,
    check_interval_seconds: float = 0.0,
    entry_grace_seconds: float = 0.0,
) -> Settings:
    """Watchdog-tuned Settings.

    ``check_interval_seconds=0.0`` would be rejected by the validator, so
    callers exercising the throttle pass an explicit positive value. The
    classification tests want zero to mean "evaluate every tick".
    """
    base = Settings()
    return base.model_copy(
        update={
            "watchdog": WatchdogConfig(
                enabled=enabled,
                shadow_mode=shadow_mode,
                check_interval_seconds=max(check_interval_seconds, 0.001),
                entry_grace_seconds=entry_grace_seconds,
            ),
        }
    )


def _build_position(
    *,
    symbol: str = "BIYA",
    shares: int = 41,
    avg_price: float = 2.49,
    status: str = "open",
) -> Position:
    """Construct an ``open`` Position; tests vary shares + symbol."""
    return Position(
        symbol=symbol,
        strategy="gap_and_go",
        shares=shares,
        avg_price=avg_price,
        stop_price=avg_price - 0.20,
        scale_out_price=avg_price + 0.40,
        runner_target_price=None,
        parent_order_id=767,
        stop_order_id=770,
        target_order_id=0,
        opened_at=datetime.now(UTC),
        status=status,  # type: ignore[arg-type]
    )


class _FakeOrder:
    """Minimal ib_async.Order stand-in for watchdog tests."""

    def __init__(
        self,
        *,
        order_id: int,
        action: str,
        order_type: str,
        total_quantity: int,
        client_id: int = 17,
        aux_price: float = 0.0,
        lmt_price: float = 0.0,
    ) -> None:
        self.orderId = order_id
        self.action = action
        self.orderType = order_type
        self.totalQuantity = total_quantity
        self.clientId = client_id
        self.auxPrice = aux_price
        self.lmtPrice = lmt_price


class _FakeOrderStatus:
    def __init__(self, *, status: str = "Submitted", remaining: int | None = None) -> None:
        self.status = status
        self.remaining = remaining


class _FakeContract:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _FakeTrade:
    """ib_async.Trade stand-in — order + contract + orderStatus."""

    def __init__(
        self,
        *,
        symbol: str,
        order: _FakeOrder,
        status: str = "Submitted",
        remaining: int | None = None,
    ) -> None:
        self.contract = _FakeContract(symbol)
        self.order = order
        self.orderStatus = _FakeOrderStatus(status=status, remaining=remaining)


class _FakeIBKRPosition:
    """ib_async account-level Position stand-in for ``ib.positions()``."""

    def __init__(self, symbol: str, qty: float) -> None:
        self.contract = _FakeContract(symbol)
        self.position = qty


class _FakeIB:
    """Minimal ``ib_async.IB`` stand-in supporting positions() + openTrades()."""

    def __init__(
        self,
        *,
        positions: list[_FakeIBKRPosition] | None = None,
        open_trades: list[_FakeTrade] | None = None,
    ) -> None:
        self._positions = positions or []
        self._open_trades = open_trades or []

    def positions(self) -> list[_FakeIBKRPosition]:
        return list(self._positions)

    def openTrades(self) -> list[_FakeTrade]:  # noqa: N802 - mirrors ib_async
        return list(self._open_trades)


def _build_ibkr(
    *,
    positions: list[tuple[str, float]] | None = None,
    open_trades: list[_FakeTrade] | None = None,
) -> MagicMock:
    """Build an IBKRClient mock whose ``ib`` exposes positions() + openTrades()."""
    fake = _FakeIB(
        positions=[_FakeIBKRPosition(s, q) for s, q in (positions or [])],
        open_trades=open_trades or [],
    )
    client = MagicMock()
    client.ib = fake
    return client


def _watchdog(
    *,
    store: PositionStore,
    ibkr: MagicMock,
    notifier: Notifier | None = None,
    settings: Settings | None = None,
) -> Watchdog:
    return Watchdog(
        ibkr=ibkr,
        position_store=store,
        notifier=notifier,
        settings=settings or _settings(),
    )


# ---------------------------------------------------------------------------
# Classification basics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifies_protected_when_stp_covers_full_position() -> None:
    """Working SELL STP @ ≥ position size → PROTECTED, no alert event."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=83))
    stp = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="STP", total_quantity=83, aux_price=2.34),
    )
    ibkr = _build_ibkr(positions=[("ABC", 83)], open_trades=[stp])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected" in events
    assert "watchdog.position_naked" not in events
    assert "watchdog.position_underprotected" not in events


@pytest.mark.asyncio
async def test_classifies_naked_when_no_sell_orders() -> None:
    """Position with zero working SELL orders → NAKED."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_naked" in events
    naked = next(e for e in captured if e["event"] == "watchdog.position_naked")
    assert naked["symbol"] == "ABC"
    assert naked["shares"] == 100
    assert naked["protective_quantity"] == 0


@pytest.mark.asyncio
async def test_classifies_underprotected_when_only_lmt_above_market() -> None:
    """The BIYA verbatim scenario: 41 shares + working SELL LMT only → UNDERPROTECTED.

    LMT-above-market is a take-profit, not protection. Order type alone
    drives the discriminator (the spec is explicit: do not infer from
    price comparison). Any non-protective sell that *is* present means
    UNDERPROTECTED rather than NAKED so the operator sees the half-baked
    state distinctly.
    """
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="BIYA", shares=41, avg_price=2.49))
    take_profit = _FakeTrade(
        symbol="BIYA",
        order=_FakeOrder(order_id=771, action="SELL", order_type="LMT", total_quantity=20, lmt_price=2.77),
    )
    ibkr = _build_ibkr(positions=[("BIYA", 41)], open_trades=[take_profit])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_underprotected" in events
    under = next(e for e in captured if e["event"] == "watchdog.position_underprotected")
    assert under["symbol"] == "BIYA"
    assert under["shares"] == 41
    # The LMT does NOT count toward protective qty even though it sums to 20.
    assert under["protective_quantity"] == 0


@pytest.mark.asyncio
async def test_lmt_above_market_does_not_count_as_protection() -> None:
    """SELL LMT (any price) must not contribute to protective_quantity."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="BIYA", shares=41))
    lmt_only = _FakeTrade(
        symbol="BIYA",
        order=_FakeOrder(order_id=2, action="SELL", order_type="LMT", total_quantity=41, lmt_price=2.77),
    )
    ibkr = _build_ibkr(positions=[("BIYA", 41)], open_trades=[lmt_only])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    # Even though qty matches, classification must be UNDERPROTECTED — the
    # order type is the source of truth, not the quantity match.
    events = [e["event"] for e in captured]
    assert "watchdog.position_underprotected" in events


@pytest.mark.asyncio
async def test_multiple_stops_summing_to_cover_classify_protected() -> None:
    """Two SELL STPs with combined qty ≥ position size → PROTECTED."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    stp_a = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="STP", total_quantity=60, aux_price=2.0),
    )
    stp_b = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=2, action="SELL", order_type="TRAIL", total_quantity=40, aux_price=2.1),
    )
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[stp_a, stp_b])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected" in events


@pytest.mark.asyncio
async def test_trail_order_counts_as_protection() -> None:
    """SELL TRAIL is the Phase 6.14 immediate-trail post-scale stop — protective."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=50))
    trail = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="TRAIL", total_quantity=50, aux_price=10.0),
    )
    ibkr = _build_ibkr(positions=[("ABC", 50)], open_trades=[trail])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected" in events


# ---------------------------------------------------------------------------
# Entry grace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protected_pending_within_entry_grace() -> None:
    """A naked-shaped position inside the entry-grace window classifies PROTECTED_PENDING.

    Models the brief window between executor.parent_filled and Phase 8.3's
    fill-anchored protection planting. The watchdog should hold its fire
    during this window so we don't alert on every clean entry fill.
    """
    settings = _settings(entry_grace_seconds=30.0)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr, settings=settings)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected_pending" in events
    assert "watchdog.position_naked" not in events


@pytest.mark.asyncio
async def test_grace_expires_into_naked() -> None:
    """After entry_grace_seconds elapse, a naked-shaped position transitions to NAKED.

    Drives time forward by mutating the watchdog's per-symbol
    ``first_seen_open_at`` rather than sleeping; the test wants to see the
    transition, not wait the wall clock.
    """
    settings = _settings(entry_grace_seconds=30.0)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr, settings=settings)
    # First tick — within grace, classifies PROTECTED_PENDING.
    await wd.tick()
    # Rewind first-seen so the next tick falls outside the grace window.
    wd._symbols["ABC"].first_seen_open_at = datetime.now(UTC) - timedelta(seconds=120)
    # Reset throttle so the next tick re-evaluates.
    wd._last_ran_at = None
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_naked" in events


# ---------------------------------------------------------------------------
# Position-state mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mismatch_between_bot_and_ibkr_emits_separate_event() -> None:
    """Bot says 100 shares, IBKR says 50 → watchdog.position_state_mismatch fires.

    Mismatch is a distinct concern from naked/underprotected; both can
    fire on the same tick. The mismatch alert must not be conflated with
    the protection-class alerts.
    """
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    stp = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="STP", total_quantity=100, aux_price=2.0),
    )
    ibkr = _build_ibkr(positions=[("ABC", 50)], open_trades=[stp])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_state_mismatch" in events
    mismatch = next(e for e in captured if e["event"] == "watchdog.position_state_mismatch")
    assert mismatch["symbol"] == "ABC"
    assert mismatch["bot_shares"] == 100
    assert mismatch["ibkr_shares"] == 50


@pytest.mark.asyncio
async def test_mismatch_position_missing_from_ibkr() -> None:
    """Bot has the position, IBKR doesn't — mismatch event fires with ibkr_shares=None."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    stp = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="STP", total_quantity=100, aux_price=2.0),
    )
    ibkr = _build_ibkr(positions=[], open_trades=[stp])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    mismatches = [e for e in captured if e["event"] == "watchdog.position_state_mismatch"]
    assert mismatches, "expected position_state_mismatch event"
    assert mismatches[0]["ibkr_shares"] is None


# ---------------------------------------------------------------------------
# Suppression / re-arm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alert_only_fires_once_per_arming_cycle() -> None:
    """Two ticks of the same naked state → alert_sent only once.

    Without this guarantee a 5-second-cadence watchdog would Telegram-spam
    every cycle while the operator was reading the first message.
    """
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured1:
        await wd.tick()
    wd._last_ran_at = None
    with capture_logs() as captured2:
        await wd.tick()
    sent_count = sum(1 for e in captured1 + captured2 if e["event"] == "watchdog.alert_sent")
    suppressed_count = sum(
        1 for e in captured1 + captured2 if e["event"] == "watchdog.alert_suppressed"
    )
    # alert_no_notifier fires when no notifier wired (this test path) — it
    # behaves like alert_sent for the once-per-cycle guarantee. Sum the
    # two emission paths and the per-tick suppression.
    no_notifier_count = sum(
        1 for e in captured1 + captured2 if e["event"] == "watchdog.alert_no_notifier"
    )
    assert sent_count + no_notifier_count == 1, "first tick fires the alert"
    assert suppressed_count >= 1, "second tick suppresses with prior_alert_pending_ack"


@pytest.mark.asyncio
async def test_ack_suppresses_further_alerts_for_same_classification() -> None:
    """Operator taps Ack → watchdog suppresses alerts for same (symbol, classification)."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    notifier = Notifier(settings=_settings(), bot=MagicMock())
    notifier._chat_id = "test"  # bypass missing-creds gate
    notifier._token = "test"
    wd = _watchdog(store=store, ibkr=ibkr, notifier=notifier, settings=_settings())
    # First tick fires alert
    await wd.tick()
    # Operator acks
    notifier.mark_alert_acked("watchdog:ABC:NAKED")
    wd._last_ran_at = None
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.alert_acked" in events
    assert "watchdog.alert_sent" not in events


@pytest.mark.asyncio
async def test_position_size_change_rearms_suppression() -> None:
    """Any change to position size clears suppressions and the underlying ack."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    notifier = Notifier(settings=_settings(), bot=MagicMock())
    notifier._chat_id = "test"
    notifier._token = "test"
    wd = _watchdog(store=store, ibkr=ibkr, notifier=notifier, settings=_settings())
    await wd.tick()
    notifier.mark_alert_acked("watchdog:ABC:NAKED")
    # Size change: simulate a partial fill dropping position to 41.
    store.insert_reconciled(_build_position(symbol="ABC", shares=41))
    wd._last_ran_at = None
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    # Re-arm log fires; the ack is cleared so a fresh alert can go out.
    assert "watchdog.suppressions_cleared" in evt_names(events)
    cleared_evt = next(e for e in captured if e["event"] == "watchdog.suppressions_cleared")
    assert cleared_evt["reason"] == "position_size_changed"
    assert not notifier.is_alert_acked("watchdog:ABC:NAKED")


def evt_names(events: list[str]) -> set[str]:
    """Convenience — set of event names from a list of structlog log dicts."""
    return set(events)


@pytest.mark.asyncio
async def test_protected_transition_clears_prior_suppressions() -> None:
    """Once protection comes back online, the watchdog re-arms automatically."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr_naked = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    notifier = Notifier(settings=_settings(), bot=MagicMock())
    notifier._chat_id = "test"
    notifier._token = "test"
    wd = _watchdog(store=store, ibkr=ibkr_naked, notifier=notifier, settings=_settings())
    await wd.tick()
    notifier.mark_alert_acked("watchdog:ABC:NAKED")
    # Operator manually places a stop in TWS — IBKR view now has it.
    stp = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=99, action="SELL", order_type="STP", total_quantity=100, aux_price=2.0),
    )
    wd._ibkr.ib._open_trades = [stp]  # type: ignore[attr-defined]
    wd._last_ran_at = None
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected" in events
    # The auto-resolve clears prior suppressions and the ack.
    assert not notifier.is_alert_acked("watchdog:ABC:NAKED")


@pytest.mark.asyncio
async def test_trading_day_rollover_clears_suppressions() -> None:
    """When the NY-local date changes between ticks, all suppressions clear."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    notifier = Notifier(settings=_settings(), bot=MagicMock())
    notifier._chat_id = "test"
    notifier._token = "test"
    wd = _watchdog(store=store, ibkr=ibkr, notifier=notifier, settings=_settings())
    await wd.tick()
    notifier.mark_alert_acked("watchdog:ABC:NAKED")
    assert wd._last_trading_day is not None
    # Force the watchdog to think the previous tick was on a different day.
    wd._last_trading_day = wd._last_trading_day - timedelta(days=1)
    wd._last_ran_at = None
    with capture_logs() as captured:
        await wd.tick()
    cleared = [e for e in captured if e["event"] == "watchdog.suppressions_cleared"]
    assert any(c.get("reason") == "trading_day_rollover" for c in cleared)
    assert not notifier.is_alert_acked("watchdog:ABC:NAKED")


# ---------------------------------------------------------------------------
# Shadow mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_mode_emits_skipped_event_and_does_not_send() -> None:
    """Shadow mode runs detection + emits ``watchdog.shadow_alert_skipped``; no Telegram."""
    settings = _settings(shadow_mode=True)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    notifier = MagicMock(spec=Notifier)
    notifier.is_alert_acked = MagicMock(return_value=False)
    notifier.clear_alert_ack = MagicMock()
    notifier.send_alert_with_ack = MagicMock()
    wd = _watchdog(store=store, ibkr=ibkr, notifier=notifier, settings=settings)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_naked" in events  # detection still active
    assert "watchdog.shadow_alert_skipped" in events
    assert "watchdog.alert_sent" not in events
    notifier.send_alert_with_ack.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_alert_skipped_carries_ack_id_for_review() -> None:
    """The shadow log row contains the ack_id that *would* have been sent — operator triage aid."""
    settings = _settings(shadow_mode=True)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="BIYA", shares=41))
    take_profit = _FakeTrade(
        symbol="BIYA",
        order=_FakeOrder(order_id=771, action="SELL", order_type="LMT", total_quantity=20, lmt_price=2.77),
    )
    ibkr = _build_ibkr(positions=[("BIYA", 41)], open_trades=[take_profit])
    wd = _watchdog(store=store, ibkr=ibkr, settings=settings)
    with capture_logs() as captured:
        await wd.tick()
    skipped = next(e for e in captured if e["event"] == "watchdog.shadow_alert_skipped")
    assert skipped["ack_id"] == "watchdog:BIYA:UNDERPROTECTED"
    assert skipped["classification"] == "UNDERPROTECTED"


# ---------------------------------------------------------------------------
# Throttle + disabled flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_interval_throttles_back_to_back_ticks() -> None:
    """Two ticks inside ``check_interval_seconds`` of each other → second is a no-op."""
    settings = _settings(check_interval_seconds=10.0)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr, settings=settings)
    await wd.tick()
    # Second tick within the 10 s window — must be a no-op (no fresh
    # naked event emitted).
    with capture_logs() as captured:
        await wd.tick()
    assert all(e["event"] != "watchdog.position_naked" for e in captured)


@pytest.mark.asyncio
async def test_disabled_flag_short_circuits() -> None:
    """``watchdog.enabled=False`` makes tick() a no-op — no events at all."""
    settings = _settings(enabled=False)
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr, settings=settings)
    with capture_logs() as captured:
        await wd.tick()
    watchdog_events = [e for e in captured if e["event"].startswith("watchdog.")]
    assert watchdog_events == []


# ---------------------------------------------------------------------------
# Status filtering — only ``open`` positions are evaluated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_entry_position_not_evaluated() -> None:
    """``pending_entry`` positions don't have shares on the wire yet — skipped."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100, status="pending_entry"))
    ibkr = _build_ibkr(positions=[], open_trades=[])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    watchdog_events = [e for e in captured if e["event"].startswith("watchdog.")]
    assert watchdog_events == []


@pytest.mark.asyncio
async def test_partially_filled_remaining_quantity_used_for_protection_count() -> None:
    """The ``orderStatus.remaining`` field is the resting wire-side qty, not totalQuantity.

    This is the BIYA scenario shape: STP totalQuantity=83 but remaining=41
    after a 42-share partial fill. The watchdog must use 41, not 83, when
    summing protective qty against the now-shrunk position.
    """
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="BIYA", shares=41))
    stp_partial = _FakeTrade(
        symbol="BIYA",
        order=_FakeOrder(order_id=770, action="SELL", order_type="STP", total_quantity=83, aux_price=2.34),
        status="PreSubmitted",
        remaining=41,
    )
    ibkr = _build_ibkr(positions=[("BIYA", 41)], open_trades=[stp_partial])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_protected" in events  # 41 remaining covers 41 shares


@pytest.mark.asyncio
async def test_inactive_orders_are_filtered_out() -> None:
    """Cancelled / Filled / PendingCancel orders must not count toward protective qty."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    cancelled = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(order_id=1, action="SELL", order_type="STP", total_quantity=100, aux_price=2.0),
        status="Cancelled",
    )
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[cancelled])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_naked" in events


@pytest.mark.asyncio
async def test_orders_from_other_clients_ignored() -> None:
    """Orders with a different ``clientId`` (manual TWS orders) must not count."""
    store = PositionStore()
    store.insert_reconciled(_build_position(symbol="ABC", shares=100))
    foreign_stp = _FakeTrade(
        symbol="ABC",
        order=_FakeOrder(
            order_id=999,
            action="SELL",
            order_type="STP",
            total_quantity=100,
            aux_price=2.0,
            client_id=42,  # not the bot's client_id
        ),
    )
    ibkr = _build_ibkr(positions=[("ABC", 100)], open_trades=[foreign_stp])
    wd = _watchdog(store=store, ibkr=ibkr)
    with capture_logs() as captured:
        await wd.tick()
    events = [e["event"] for e in captured]
    assert "watchdog.position_naked" in events
