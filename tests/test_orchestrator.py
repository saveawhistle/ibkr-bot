"""Tests for ``bot.orchestrator.run_strategy_loop`` with fake MarketData + canned strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest

from bot.brokerage.market_data import BarStream, MarketData
from bot.orchestrator import run_strategy_loop
from bot.scanning.scanner import ScanHit
from bot.signal_bus import SignalBus
from bot.strategies.base import Signal, Strategy

if TYPE_CHECKING:
    from bot.execution.position_state import PositionStore


def _hit(symbol: str) -> ScanHit:
    """Build a minimal ScanHit — only ``symbol`` matters for the orchestrator."""
    return ScanHit(
        symbol=symbol,
        price=None,
        change_pct=None,
        volume=None,
        float_shares=4_000_000,
        catalyst="earnings_beat",
    )


class _FakeMarketData:
    """Pretend to be ``MarketData`` — subscribe returns a stream with fabricated bars.

    Phase 6.2: also mirrors subscribe/unsubscribe against a real
    ``SubscriptionRegistry`` under ``self.registry`` so rescan-consistency
    tests can assert the registry tracks the currently-subscribed watchlist.
    Symbols without an explicit frame get a default single-row frame — rescan
    tests routinely add new tickers the fixture didn't pre-fabricate.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        """Hold per-symbol fabricated bar DataFrames + fresh SubscriptionRegistry."""
        from bot.brokerage.ibkr_client import SubscriptionRegistry

        self.frames = frames
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.registry = SubscriptionRegistry()
        self._symbol_req_ids: dict[str, int] = {}
        self._next_req_id = 1000
        # Phase 7.3: capture the on_new_bar callback the orchestrator wires
        # onto each subscription so event-driven-path tests can fire it
        # manually without having to simulate IBKR's updateEvent.
        self.on_new_bar_cbs: dict[str, Any] = {}

    async def subscribe_bars(
        self,
        symbol: str,
        bar_size: str = "1 min",
        on_new_bar: Any = None,
    ) -> BarStream:
        """Return a BarStream wrapping the fabricated frame + register a fake req_id.

        Phase 7.3: ``on_new_bar`` accepted so the real subscribe contract is
        honoured; fake doesn't fire it (tests drive the poll path directly).
        """
        from bot.brokerage.ibkr_client import ActiveSubscription

        self.subscribed.append(symbol)
        if symbol not in self.frames:
            self.frames[symbol] = pd.DataFrame({"close": [10.0]})
        self._next_req_id += 1
        req_id = self._next_req_id
        self._symbol_req_ids[symbol] = req_id
        await self.registry.register(
            ActiveSubscription(req_id=req_id, kind="historical", symbol=symbol)
        )
        if on_new_bar is not None:
            self.on_new_bar_cbs[symbol] = on_new_bar
        return BarStream(
            symbol=symbol,
            bars=self.frames[symbol],
            _bar_list=None,  # type: ignore[arg-type]
            on_new_bar=on_new_bar,
        )

    async def subscribe_bars_5sec_aggregated(
        self,
        symbol: str,
        on_new_bar: Any = None,
    ) -> BarStream:
        """Phase 10.4 — orchestrator dispatches to this when ``bar_source`` is
        ``rtbars_5sec_aggregated``. The fake delegates to ``subscribe_bars``
        because these tests don't differentiate the bar-source paths;
        bar-source-specific behavior is covered by ``test_market_data_5sec_path``.
        """
        return await self.subscribe_bars(symbol, on_new_bar=on_new_bar)

    async def unsubscribe(self, symbol: str) -> None:
        """Record the unsubscription and unregister the matching req_id."""
        self.unsubscribed.append(symbol)
        req_id = self._symbol_req_ids.pop(symbol, None)
        if req_id is not None:
            await self.registry.unregister(req_id)

    async def close(self) -> None:
        """Match the real MarketData surface."""


@dataclass
class _CannedStrategy(Strategy):
    """Emit a fixed signal the first time evaluate is called for a given symbol."""

    def __init__(self) -> None:
        super().__init__()
        self.fired: set[str] = set()

    name: str = "canned"

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        """Fire once per symbol with a 3:1 reward/risk signal."""
        if symbol in self.fired:
            return None
        self.fired.add(symbol)
        return Signal(
            symbol=symbol,
            strategy=self.name,
            entry=10.0,
            stop=9.0,
            scale_out_price=12.0,
            runner_target_price=13.0,
            timestamp=datetime(2026, 4, 16, 9, 31, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_evaluate_on_closed_bar_only_slices_trailing_in_progress_bar() -> None:
    """Phase 7.4: by default, strategies receive bars[:-1] — the just-closed bar at iloc[-1].

    Simulates IBKR's ``keepUpToDate=True`` shape: the trailing bar is the
    freshly-started next-minute bar (live-updating, near-zero data). The
    just-closed bar sits at ``bars[-2]``. Asserts the strategy sees the
    closed bar as ``iloc[-1]``, not the in-progress one.
    """

    class _InspectingStrategy(Strategy):
        name: str = "inspector"

        def __init__(self) -> None:
            super().__init__()
            self.seen_last_rows: list[dict[str, float]] = []

        def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
            row = bars.iloc[-1]
            self.seen_last_rows.append({"close": float(row["close"])})
            return None

    # Two-row frame: closed bar close=$10.0, in-progress trailing bar close=$99.0.
    # Use fresh timestamps so bar-staleness doesn't short-circuit evaluation.
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    now = pd.Timestamp.now(tz=ny)
    idx = pd.DatetimeIndex([now - pd.Timedelta(seconds=60), now])
    frames = {"AAA": pd.DataFrame({"close": [10.0, 99.0]}, index=idx)}
    inspector = _InspectingStrategy()
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()

    await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[inspector],
        duration_minutes=0.01,
        poll_interval=0.05,
    )

    assert inspector.seen_last_rows, "strategy should have been called at least once"
    # With the Phase 7.4 slice, every call sees the closed bar's close ($10.0),
    # never the in-progress trailing bar's $99.0.
    assert all(r["close"] == pytest.approx(10.0) for r in inspector.seen_last_rows), (
        f"strategy saw in-progress bar: {inspector.seen_last_rows}"
    )


@pytest.mark.asyncio
async def test_evaluate_on_closed_bar_only_false_preserves_legacy_behaviour() -> None:
    """Phase 7.4: opt-out (``False``) passes the whole frame through — tests/backtest mode."""

    class _InspectingStrategy(Strategy):
        name: str = "inspector"

        def __init__(self) -> None:
            super().__init__()
            self.seen_last_rows: list[dict[str, float]] = []

        def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
            row = bars.iloc[-1]
            self.seen_last_rows.append({"close": float(row["close"])})
            return None

    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    now = pd.Timestamp.now(tz=ny)
    idx = pd.DatetimeIndex([now - pd.Timedelta(seconds=60), now])
    frames = {"AAA": pd.DataFrame({"close": [10.0, 99.0]}, index=idx)}
    inspector = _InspectingStrategy()
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()

    await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[inspector],
        duration_minutes=0.01,
        poll_interval=0.05,
        evaluate_on_closed_bar_only=False,
    )

    assert any(r["close"] == pytest.approx(99.0) for r in inspector.seen_last_rows), (
        f"opt-out mode should surface the trailing row: {inspector.seen_last_rows}"
    )


@pytest.mark.asyncio
async def test_on_new_bar_event_drives_evaluation_without_polling() -> None:
    """Phase 7.3: invoking the per-symbol on_new_bar callback publishes a signal.

    Runs the loop with a long poll_interval so the poll path cannot reach
    the evaluation during the test. Fires the event-driven callback
    directly; the signal must land on the bus before the loop exits. Proves
    the event-driven path is actually wired (regressions would fall back
    to pollwait and starve the bus).
    """
    import asyncio

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    fake_md = _FakeMarketData(frames)
    market_data = cast("MarketData", fake_md)
    bus = SignalBus()

    # Long duration + long poll_interval → if the poll path ran first we'd
    # see the signal regardless; to isolate the event path we fire the
    # callback ourselves and exit via duration_minutes before the next
    # poll tick.
    loop_task = asyncio.create_task(
        run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.02,  # ~1.2s
            poll_interval=10.0,  # intentionally longer than duration
        )
    )
    # Yield until the subscribe_bars call has registered the callback.
    for _ in range(50):
        await asyncio.sleep(0)
        if "AAA" in fake_md.on_new_bar_cbs:
            break
    assert "AAA" in fake_md.on_new_bar_cbs, "orchestrator did not wire on_new_bar"

    # Fire the event-driven callback and let it run.
    await fake_md.on_new_bar_cbs["AAA"]()

    # Signal must be on the bus — event-driven path succeeded.
    assert bus.qsize() == 1

    result = await loop_task
    assert {s.symbol for s in result.signals} == {"AAA"}


@pytest.mark.asyncio
async def test_loop_publishes_signals_and_unsubscribes() -> None:
    """Two watchlist symbols → two signals on the bus, both unsubscribed at shutdown."""
    frames = {
        "AAA": pd.DataFrame({"close": [10.0]}),
        "BBB": pd.DataFrame({"close": [10.0]}),
    }
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA"), _hit("BBB")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.01,  # ~0.6s run, enough for one poll pass
        poll_interval=0.05,
    )
    assert {s.symbol for s in result.signals} == {"AAA", "BBB"}
    assert bus.qsize() == 2
    fake = cast("_FakeMarketData", market_data)
    assert set(fake.unsubscribed) == {"AAA", "BBB"}


@pytest.mark.asyncio
async def test_empty_watchlist_returns_empty_result() -> None:
    """No watchlist → no subscriptions, no signals."""
    market_data = cast("MarketData", _FakeMarketData({}))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.005,
    )
    assert result.signals == []


# ---------- Phase 4g: rehab check runs inside the strategy loop ---------- #


@pytest.mark.asyncio
async def test_loop_drives_rehab_check_when_interval_elapses() -> None:
    """A wired ``rehab_engine`` gets ``check_transitions`` called each tick.

    Proves the loop's rehab cadence wiring — not the tier logic itself,
    which is exhaustively covered in ``test_rehab.py``. A zero interval
    means every poll triggers a check, so by run end we expect ≥1 call.
    """
    from bot.risk.rehab import RehabEngine, RehabTransition

    class _SpyRehabEngine:
        """Record each ``check_transitions`` call; return None (no transition)."""

        def __init__(self) -> None:
            self.calls = 0

        async def check_transitions(self) -> RehabTransition | None:
            self.calls += 1
            return None

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    spy = _SpyRehabEngine()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        rehab_engine=cast("RehabEngine", spy),
        rehab_check_interval_seconds=0.0,
        duration_minutes=0.01,
        poll_interval=0.05,
    )
    assert spy.calls >= 1
    # Loop shouldn't be disrupted by rehab wiring — normal signal path still works.
    assert {s.symbol for s in result.signals} == {"AAA"}


# ---------- Phase 5.1: deadline derived from session.flatten_all ---------- #


@pytest.mark.asyncio
async def test_loop_derives_deadline_from_flatten_all_when_duration_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``duration_minutes=None`` derives deadline from ``session.flatten_all``."""
    from bot import orchestrator as orch_module

    captured: dict[str, float] = {}
    real_derive = orch_module._derive_duration_seconds_to_flatten

    def spy(settings: object, now_ny: datetime, *, safety_buffer_seconds: float = 60.0) -> float:
        seconds = real_derive(settings, now_ny, safety_buffer_seconds=safety_buffer_seconds)  # type: ignore[arg-type]
        captured["seconds"] = seconds
        # clamp to a short test-friendly duration so the loop actually exits
        return 0.5

    monkeypatch.setattr(orch_module, "_derive_duration_seconds_to_flatten", spy)

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=None,  # derive from flatten_all
        poll_interval=0.05,
    )
    assert "seconds" in captured, "expected derive helper to be called"
    assert {s.symbol for s in result.signals} == {"AAA"}


@pytest.mark.asyncio
async def test_loop_exits_immediately_when_flatten_all_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When derived duration ≤ 0, loop exits immediately without subscribing bars."""
    from bot import orchestrator as orch_module

    monkeypatch.setattr(
        orch_module,
        "_derive_duration_seconds_to_flatten",
        lambda *_args, **_kwargs: -1.0,
    )

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=None,
        poll_interval=0.05,
    )
    fake = cast("_FakeMarketData", market_data)
    assert result.signals == []
    assert fake.subscribed == [], "expected no bar subscriptions when flatten already passed"


@pytest.mark.asyncio
async def test_loop_respects_explicit_duration_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``duration_minutes`` wins; derive helper is never called."""
    from bot import orchestrator as orch_module

    def _boom(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("derive helper must not be called when duration is explicit")

    monkeypatch.setattr(orch_module, "_derive_duration_seconds_to_flatten", _boom)

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.01,
        poll_interval=0.05,
    )
    assert {s.symbol for s in result.signals} == {"AAA"}


def test_derive_duration_to_flatten_future() -> None:
    """Flatten time in the future → positive seconds minus buffer."""
    from zoneinfo import ZoneInfo

    from bot.config import SessionConfig, Settings
    from bot.orchestrator import _derive_duration_seconds_to_flatten

    settings = Settings(session=SessionConfig(flatten_all="15:55"))
    tz = ZoneInfo(settings.session.timezone)
    now_ny = datetime(2026, 4, 20, 10, 0, tzinfo=tz)  # 10:00 ET
    seconds = _derive_duration_seconds_to_flatten(settings, now_ny)
    # 15:55 - 10:00 = 5h55m = 21300s; minus 60s buffer = 21240s
    assert seconds == pytest.approx(21240.0)


def test_derive_duration_to_flatten_past_returns_negative() -> None:
    """Flatten time already passed today → negative sentinel."""
    from zoneinfo import ZoneInfo

    from bot.config import SessionConfig, Settings
    from bot.orchestrator import _derive_duration_seconds_to_flatten

    settings = Settings(session=SessionConfig(flatten_all="15:55"))
    tz = ZoneInfo(settings.session.timezone)
    now_ny = datetime(2026, 4, 20, 16, 30, tzinfo=tz)  # 16:30 ET, after flatten
    seconds = _derive_duration_seconds_to_flatten(settings, now_ny)
    assert seconds < 0


# ---------- Phase 5.4: shutdown_event early-exit ---------- #


@pytest.mark.asyncio
async def test_shutdown_event_exits_loop_before_deadline() -> None:
    """Setting ``shutdown_event`` mid-loop must break out before the long duration elapses."""
    import asyncio

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    shutdown_event = asyncio.Event()

    async def trigger_shutdown() -> None:
        await asyncio.sleep(0.1)
        shutdown_event.set()

    trigger = asyncio.create_task(trigger_shutdown())
    # duration=10 min would normally block tests forever; shutdown_event must
    # override it. Budget under 2s as a sanity check on responsiveness.
    start = asyncio.get_event_loop().time()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=10,
        poll_interval=0.05,
        shutdown_event=shutdown_event,
    )
    elapsed = asyncio.get_event_loop().time() - start
    await trigger
    assert elapsed < 2.0, f"shutdown_event did not break loop (took {elapsed:.2f}s)"
    # The signal still fired at least once before shutdown arrived.
    assert result.signals[0].symbol == "AAA"


# ---------- Phase 6.1: per-symbol IBKR-operation timeout + stall logging ---------- #


def _fast_timeout_settings(timeout_seconds: float = 0.1) -> object:
    """Build a Settings with a sub-second loop-operation timeout for stall tests."""
    from bot.config import SessionConfig, Settings

    return Settings(session=SessionConfig(loop_operation_timeout_seconds=timeout_seconds))


@pytest.mark.asyncio
async def test_loop_stall_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A symbol whose bar-fetch exceeds the timeout logs ``orchestrator.loop_stall`` and continues."""
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    async def _stalling_fetch(stream: BarStream) -> pd.DataFrame:
        await asyncio.sleep(1.0)  # dwarf the 0.1s timeout
        return stream.bars

    monkeypatch.setattr(orch_module, "_fetch_bars", _stalling_fetch)
    # Force structlog to route events through the capture_logs processor swap.
    structlog.reset_defaults()

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    with capture_logs() as captured:
        result = await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.01,
            poll_interval=0.05,
            settings=_fast_timeout_settings(0.1),  # type: ignore[arg-type]
        )
    stalls = [e for e in captured if e.get("event") == "orchestrator.loop_stall"]
    assert stalls, "expected at least one orchestrator.loop_stall event"
    first = stalls[0]
    assert first["symbol"] == "AAA"
    assert first["operation"] == "bar_fetch"
    assert first["timeout_seconds"] == pytest.approx(0.1)
    assert first["iteration_index"] >= 1
    # The loop still exited cleanly; nothing reached the bus because bars never arrived.
    assert result.signals == []


@pytest.mark.asyncio
async def test_loop_stall_skips_affected_symbol_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """One stalled symbol must not block evaluation of the other two."""
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    async def _selective_stall(stream: BarStream) -> pd.DataFrame:
        if stream.symbol == "HANG":
            await asyncio.sleep(1.0)
        return stream.bars

    monkeypatch.setattr(orch_module, "_fetch_bars", _selective_stall)
    structlog.reset_defaults()

    frames = {
        "AAA": pd.DataFrame({"close": [10.0]}),
        "HANG": pd.DataFrame({"close": [10.0]}),
        "BBB": pd.DataFrame({"close": [10.0]}),
    }
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    with capture_logs() as captured:
        result = await run_strategy_loop(
            watchlist=[_hit("AAA"), _hit("HANG"), _hit("BBB")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.01,
            poll_interval=0.05,
            settings=_fast_timeout_settings(0.1),  # type: ignore[arg-type]
        )
    # AAA and BBB must have gotten signals; HANG must have been stalled.
    assert {s.symbol for s in result.signals} == {"AAA", "BBB"}
    stall_symbols = {e["symbol"] for e in captured if e.get("event") == "orchestrator.loop_stall"}
    assert stall_symbols == {"HANG"}


@pytest.mark.asyncio
async def test_persistent_stall_upgrades_to_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """After 3 consecutive stalls, the event upgrades to ``orchestrator.persistent_stall``."""
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    async def _always_stall(stream: BarStream) -> pd.DataFrame:
        await asyncio.sleep(1.0)
        return stream.bars

    monkeypatch.setattr(orch_module, "_fetch_bars", _always_stall)
    structlog.reset_defaults()

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            # ~1.5s total budget, 0.05s timeout → ~30 iterations worth of stalls.
            duration_minutes=0.025,
            poll_interval=0.001,
            settings=_fast_timeout_settings(0.05),  # type: ignore[arg-type]
        )
    stall_events = [e for e in captured if e.get("event") == "orchestrator.loop_stall"]
    persistent_events = [e for e in captured if e.get("event") == "orchestrator.persistent_stall"]
    # First two consecutive stalls log INFO; the 3rd and beyond log WARNING.
    assert len(stall_events) == 2, (
        f"expected exactly 2 loop_stall events before escalation, got {len(stall_events)}"
    )
    assert len(persistent_events) >= 1, "expected at least one persistent_stall event"
    first_persistent = persistent_events[0]
    assert first_persistent["log_level"] == "warning"
    assert first_persistent["consecutive_stall_count"] == 3
    assert first_persistent["symbol"] == "AAA"
    assert first_persistent["operation"] == "bar_fetch"


@pytest.mark.asyncio
async def test_loop_resumes_after_stall_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stall counter resets on the first successful fetch; no stall events on recovery iteration."""
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    call_count = {"n": 0}

    async def _stall_once(stream: BarStream) -> pd.DataFrame:
        call_count["n"] += 1
        if call_count["n"] == 1:
            await asyncio.sleep(1.0)
        return stream.bars

    monkeypatch.setattr(orch_module, "_fetch_bars", _stall_once)
    structlog.reset_defaults()

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    with capture_logs() as captured:
        result = await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.02,
            poll_interval=0.05,
            settings=_fast_timeout_settings(0.1),  # type: ignore[arg-type]
        )
    stalls = [e for e in captured if e.get("event") == "orchestrator.loop_stall"]
    persistent = [e for e in captured if e.get("event") == "orchestrator.persistent_stall"]
    # Exactly one stall (iteration 1); iteration 2 succeeded → no upgrade, counter reset.
    assert len(stalls) == 1
    assert persistent == []
    # And the signal fired normally on the recovery iteration.
    assert {s.symbol for s in result.signals} == {"AAA"}


@pytest.mark.asyncio
async def test_configurable_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A smaller ``loop_operation_timeout_seconds`` produces faster stall detection."""
    import asyncio

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    async def _hang_briefly(stream: BarStream) -> pd.DataFrame:
        # 0.3s — longer than the 0.05s timeout, shorter than what a 1.0s timeout
        # would catch within the test budget. Lets us assert the small timeout
        # fires while proving the large one would not.
        await asyncio.sleep(0.3)
        return stream.bars

    monkeypatch.setattr(orch_module, "_fetch_bars", _hang_briefly)
    structlog.reset_defaults()

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()

    start = asyncio.get_event_loop().time()
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.003,  # ~0.18s budget, enough for 1-2 stall fires
            poll_interval=0.05,
            settings=_fast_timeout_settings(0.05),  # type: ignore[arg-type]
        )
    elapsed = asyncio.get_event_loop().time() - start
    stalls = [e for e in captured if e.get("event") == "orchestrator.loop_stall"]
    # At the tight 0.05s timeout a stall must fire (the hang is 0.3s).
    assert stalls, "expected stall events at 0.05s timeout vs 0.3s hang"
    assert stalls[0]["timeout_seconds"] == pytest.approx(0.05)
    # Sanity: we didn't wait the full 0.3s per iteration — the timeout kicked in.
    assert elapsed < 1.0


# ---------- Phase 6.2: continuous scanner rescan + watchlist diff ---------- #


class _FakeScanner:
    """Async stub returning pre-programmed ScanHit lists across successive calls.

    ``results`` is a list of batches; each ``scan_top_gappers`` call returns
    the next batch. When the list is exhausted, every further call returns
    the last batch — mirrors a stable scan state after the interesting
    transitions are done so the loop can idle without raising.
    """

    def __init__(self, results: list[list[ScanHit]]) -> None:
        self.results = results
        self.calls = 0

    async def scan_top_gappers(self) -> list[ScanHit]:
        batch = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return batch


class _FakePositionStore:
    """Minimal PositionStore stand-in exposing only ``has_active``."""

    def __init__(self, active: set[str] | None = None) -> None:
        self.active = active or set()

    def has_active(self, symbol: str) -> bool:
        return symbol in self.active


def _rescan_settings(interval_seconds: float = 0.1, max_size: int = 10) -> object:
    """Build a Settings object with small rescan interval + configurable cap."""
    from bot.config import SessionConfig, Settings

    # The config validator requires int seconds; tests use int values.
    return Settings(
        session=SessionConfig(
            watchlist_rescan_interval_seconds=max(int(interval_seconds), 1),
            watchlist_max_size=max_size,
        )
    )


@pytest.mark.asyncio
async def test_rescan_fires_on_interval() -> None:
    """After the rescan interval elapses, scanner.scan_top_gappers is called again."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    # Second batch keeps AAA and adds BBB so the diff observes a real add.
    scanner = _FakeScanner(
        [
            [_hit("AAA"), _hit("BBB")],  # first rescan: add BBB
        ]
    )
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
        )
    assert scanner.calls >= 1, "scanner.scan_top_gappers should be called at least once"
    rescanned = [e for e in captured if e.get("event") == "orchestrator.watchlist_rescanned"]
    assert rescanned, "expected orchestrator.watchlist_rescanned event"
    evt = rescanned[0]
    assert evt["scan_count"] == 2
    assert evt["current_watchlist_size"] == 2


@pytest.mark.asyncio
async def test_rescan_adds_new_qualifying_symbols() -> None:
    """A rescan returning 5 originals + 2 new symbols subscribes the 2 new."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    new = ["FFF", "GGG"]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + new}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    scanner = _FakeScanner([[_hit(s) for s in initial + new]])
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
        )
    added_events = {
        e["symbol"] for e in captured if e.get("event") == "orchestrator.watchlist_symbol_added"
    }
    fake = cast("_FakeMarketData", market_data)
    assert set(new).issubset(added_events)
    assert set(new).issubset(set(fake.subscribed))


# ---------- Phase 9.3: empty initial scan continues to main loop ---------- #


@pytest.mark.asyncio
async def test_empty_initial_scan_continues_when_scanner_wired() -> None:
    """Day 7 (2026-04-28) defect: empty initial scan exited before reaching the loop.

    With ``scanner`` wired the empty initial watchlist must not short-circuit
    — the rescan tick is the only path to mid-session candidates. Test
    asserts ``orchestrator.empty_initial_scan`` fires (warning, not info) and
    that scanner.scan_top_gappers gets called at least once during the run.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    market_data = cast("MarketData", _FakeMarketData({}))
    bus = SignalBus()
    scanner = _FakeScanner([[]])  # rescan also empty — no candidates anywhere
    with capture_logs() as captured:
        result = await run_strategy_loop(
            watchlist=[],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
        )
    # Loop reached completion (no early exit) — no signals because no symbols.
    assert result.signals == []
    # Old "orchestrator.empty_watchlist" info event must NOT fire on the
    # scanner-wired path; the new warning must.
    empty_initial = [e for e in captured if e.get("event") == "orchestrator.empty_initial_scan"]
    assert empty_initial, "expected orchestrator.empty_initial_scan warning"
    legacy = [e for e in captured if e.get("event") == "orchestrator.empty_watchlist"]
    assert legacy == [], "legacy empty_watchlist info must not fire on scanner path"
    assert scanner.calls >= 1, "rescan tick must fire so candidates can emerge mid-session"


@pytest.mark.asyncio
async def test_empty_initial_scan_picks_up_rescan_candidate() -> None:
    """Empty initial scan + rescan returns a symbol → that symbol gets subscribed.

    This is the load-bearing test for the Phase 9.3 fix: under the old
    behaviour the loop never started, so the rescan-add path could not run.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    frames = {"LATE": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    # First scanner call returns empty (matches the empty initial state); the
    # rescan tick within the loop returns a candidate.
    scanner = _FakeScanner([[_hit("LATE")]])
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
        )
    added = {
        e["symbol"] for e in captured if e.get("event") == "orchestrator.watchlist_symbol_added"
    }
    assert "LATE" in added, "rescan must subscribe candidates that emerge after empty initial"
    fake = cast("_FakeMarketData", market_data)
    assert "LATE" in set(fake.subscribed)


@pytest.mark.asyncio
async def test_empty_watchlist_still_short_circuits_when_no_scanner() -> None:
    """Without a scanner there is no path to candidates — keep the early return.

    Preserves the pre-9.3 behaviour for the ``--dry-run-signal`` CLI path
    and unit tests that call ``run_strategy_loop`` without a scanner. An
    idle spin until the duration expires would just burn CPU.
    """
    market_data = cast("MarketData", _FakeMarketData({}))
    bus = SignalBus()
    result = await run_strategy_loop(
        watchlist=[],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.005,
    )
    assert result.signals == []


@pytest.mark.asyncio
async def test_rescan_drops_symbols_not_in_new_scan() -> None:
    """Symbols absent from the new scan are unsubscribed (no active position)."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    surviving = ["AAA", "BBB", "CCC"]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    scanner = _FakeScanner([[_hit(s) for s in surviving]])
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
            position_store=_FakePositionStore(),  # type: ignore[arg-type]
        )
    # Phase 6.4: drops now always carry reason="not_in_scan" (declarative
    # reconciliation has no other drop reason in the diff surface).
    dropped_events = {
        e["symbol"]
        for e in captured
        if e.get("event") == "orchestrator.watchlist_symbol_dropped"
        and e.get("reason") == "not_in_scan"
    }
    assert dropped_events == {"DDD", "EEE"}


@pytest.mark.asyncio
async def test_rescan_preserves_active_position_symbol() -> None:
    """A symbol with an active position is never dropped, even when missing from new scan."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial = ["BIYA", "AAA"]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    # Rescan omits BIYA but position_store says BIYA is active.
    scanner = _FakeScanner([[_hit("AAA")]])
    store = _FakePositionStore(active={"BIYA"})
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
            position_store=store,  # type: ignore[arg-type]
        )
    # End-state unsubscribed list is not informative — finally block sweeps every
    # stream at loop exit. Verify mid-session protection via captured log events.
    kept = [e for e in captured if e.get("event") == "orchestrator.watchlist_kept_for_position"]
    assert any(e["symbol"] == "BIYA" for e in kept)
    dropped = [
        e
        for e in captured
        if e.get("event") == "orchestrator.watchlist_symbol_dropped" and e.get("symbol") == "BIYA"
    ]
    assert not dropped, f"BIYA should not have been dropped mid-session: {dropped}"


@pytest.mark.asyncio
async def test_rescan_respects_size_cap() -> None:
    """Phase 6.4: scanner-rank-top-N wins. Symbols beyond the cap fall out with ``not_in_scan``.

    Scenario: initial watchlist is at cap (10 symbols). The new scan
    promotes 2 new symbols to the top and ranks the original 10 below,
    for a total scan of 12. With ``max_size=10`` the top-10 survivor
    set is ``{NEW1, NEW2, S00..S07}`` — ``S08`` and ``S09`` fall outside
    and are unsubscribed with ``reason="not_in_scan"``. The Phase 6.2
    ``size_cap`` reason no longer exists; this test verifies the
    declarative diff enforces the cap via top-N truncation, not
    post-hoc eviction.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial = [f"S{i:02d}" for i in range(10)]  # S00..S09 — at cap of 10
    new = ["NEW1", "NEW2"]
    # Rank new symbols at the top so the cap truncates the last two
    # currents (S08, S09) out of the target set.
    rescan_symbols = new + initial
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + new}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    scanner = _FakeScanner([[_hit(s) for s in rescan_symbols]])
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
            position_store=_FakePositionStore(),  # type: ignore[arg-type]
        )
    # Phase 6.4: the ``size_cap`` reason path is gone. Verify no drop events
    # carry that legacy reason.
    legacy_size_cap = [
        e
        for e in captured
        if e.get("event") == "orchestrator.watchlist_symbol_dropped"
        and e.get("reason") == "size_cap"
    ]
    assert legacy_size_cap == [], f"Phase 6.4 removed the size_cap reason; got {legacy_size_cap}"
    dropped_not_in_scan = {
        e["symbol"]
        for e in captured
        if e.get("event") == "orchestrator.watchlist_symbol_dropped"
        and e.get("reason") == "not_in_scan"
    }
    # S08, S09 fell outside the top-10; NEW1, NEW2 took slots.
    assert {"S08", "S09"}.issubset(dropped_not_in_scan)
    added_events = {
        e["symbol"] for e in captured if e.get("event") == "orchestrator.watchlist_symbol_added"
    }
    assert set(new).issubset(added_events)


@pytest.mark.asyncio
async def test_rescan_full_with_all_positions_rejects_new() -> None:
    """Cap full of active-position symbols → new scan hits rejected with WARNING.

    Phase 6.4 semantics: to provoke the "no eviction possible" state, the
    scan must bring **new** symbols — otherwise the desired-set ∪ positions
    never exceeds the cap. Scan returns 15 NEW* symbols that are not
    currently subscribed; the positions fill every slot; the warning fires
    with ``new_scan_hits_rejected`` counting the scan (no per-symbol
    ``symbol`` field, because the rejection is against the whole scan).
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial = [f"P{i:02d}" for i in range(10)]  # all active positions
    new = [f"NEW{i}" for i in range(15)]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + new}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    # Scan returns only the new symbols — none of the current positions are
    # in the scan, so the target would be 20 symbols were it not capped.
    scanner = _FakeScanner([[_hit(s) for s in new]])
    store = _FakePositionStore(active=set(initial))
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
            position_store=store,  # type: ignore[arg-type]
        )
    fake = cast("_FakeMarketData", market_data)
    rejections = [
        e for e in captured if e.get("event") == "orchestrator.watchlist_full_no_eviction_possible"
    ]
    assert rejections, "expected at least one watchlist_full_no_eviction_possible event"
    assert rejections[0]["log_level"] == "warning"
    assert rejections[0]["new_scan_hits_rejected"] == 15
    # No NEW* symbol got subscribed mid-session.
    assert not any(s in fake.subscribed for s in new)
    # And no active-position symbol was dropped (end-state unsubscribed is
    # drained by the shutdown sweep; assert against the mid-session log events).
    mid_session_drops = {
        e["symbol"] for e in captured if e.get("event") == "orchestrator.watchlist_symbol_dropped"
    }
    assert mid_session_drops.isdisjoint(set(initial)), (
        f"active-position symbols must not be dropped mid-session, got {mid_session_drops & set(initial)}"
    )


@pytest.mark.asyncio
async def test_rescan_registry_consistency() -> None:
    """After 10 rescans with varied adds/drops, registry size == current watchlist size."""
    import structlog

    structlog.reset_defaults()

    # Each batch varies membership so the diff has real work to do per tick.
    batches = [
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
        ["B", "C", "D", "E"],
        ["C", "D", "E", "F"],
        ["D", "E", "F"],
        ["D", "E", "F", "G", "H"],
        ["E", "F", "G"],
        ["F", "G", "H", "I"],
        ["G", "H", "I", "J"],
        ["H", "I", "J"],
    ]
    frames: dict[str, pd.DataFrame] = {}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    scanner = _FakeScanner([[_hit(s) for s in b] for b in batches])
    await run_strategy_loop(
        watchlist=[_hit(s) for s in batches[0]],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.1,
        poll_interval=0.02,
        settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
        scanner=scanner,  # type: ignore[arg-type]
        position_store=_FakePositionStore(),  # type: ignore[arg-type]
    )
    fake = cast("_FakeMarketData", market_data)
    # All subscriptions swept at loop_complete — registry must be empty.
    assert len(fake.registry) == 0, (
        f"registry not drained after loop finish: {len(fake.registry)} remain"
    )
    # And the scanner was called at least twice to exercise the diff path.
    assert scanner.calls >= 2


@pytest.mark.asyncio
async def test_rescan_does_not_block_bar_evaluation() -> None:
    """A slow in-flight rescan must not prevent existing symbols from being evaluated."""
    import asyncio as _asyncio

    class _SlowScanner:
        """First call blocks for 1s; subsequent calls return fast."""

        def __init__(self) -> None:
            self.calls = 0

        async def scan_top_gappers(self) -> list[ScanHit]:
            self.calls += 1
            if self.calls == 1:
                await _asyncio.sleep(1.0)
            return [_hit("AAA")]

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    slow = _SlowScanner()
    result = await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[_CannedStrategy()],
        duration_minutes=0.01,  # ~0.6s — shorter than the 1s scanner block
        poll_interval=0.05,
        settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
        scanner=slow,  # type: ignore[arg-type]
    )
    # AAA must have emitted a signal despite the still-in-flight scanner task.
    assert {s.symbol for s in result.signals} == {"AAA"}


@pytest.mark.asyncio
async def test_rescan_failure_is_logged_and_loop_continues() -> None:
    """Scanner exceptions must log ``orchestrator.rescan_failed`` without crashing the loop."""
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    class _ExplodingScanner:
        calls = 0

        async def scan_top_gappers(self) -> list[ScanHit]:
            type(self).calls += 1
            raise RuntimeError("finnhub 429")

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    with capture_logs() as captured:
        result = await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=_ExplodingScanner(),  # type: ignore[arg-type]
        )
    failures = [e for e in captured if e.get("event") == "orchestrator.rescan_failed"]
    assert failures, "expected orchestrator.rescan_failed event on scanner exception"
    assert failures[0]["log_level"] == "warning"
    # Loop survived — AAA still signalled.
    assert {s.symbol for s in result.signals} == {"AAA"}


@pytest.mark.asyncio
async def test_watchlist_excludes_float_unknown_after_rescan() -> None:
    """Phase 6.3 regression: the scanner pre-filters float_unknown symbols, so
    the orchestrator never sees them — watchlist adds exactly the known-float
    count and the SubscriptionRegistry matches. This test simulates that
    contract by feeding the fake scanner only the 2 survivors (KNOWN1, KNOWN2)
    out of a notional 4-symbol raw gap list where UNK1/UNK2 were dropped at
    scanner.dropped_float_unknown upstream.
    """
    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    initial: list[str] = []
    # Scanner returns only known-float survivors; UNK1/UNK2 would've been
    # dropped inside IBKRScanner._apply_float_filter in real flow.
    survivors = [_hit("KNOWN1"), _hit("KNOWN2")]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in ["KNOWN1", "KNOWN2"]}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    scanner = _FakeScanner([survivors])
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit(s) for s in initial] or [_hit("SEED")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[_CannedStrategy()],
            duration_minutes=0.05,
            poll_interval=0.05,
            settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
            scanner=scanner,  # type: ignore[arg-type]
            position_store=_FakePositionStore(),  # type: ignore[arg-type]
        )
    added = {
        e["symbol"] for e in captured if e.get("event") == "orchestrator.watchlist_symbol_added"
    }
    assert added == {"KNOWN1", "KNOWN2"}, (
        f"expected only KNOWN1/KNOWN2 to be added post-rescan, got {added}"
    )
    assert "UNK1" not in added and "UNK2" not in added


# ---------- Phase 6.4: declarative-reconciliation diff ---------- #


async def _seed_streams(market_data: _FakeMarketData, symbols: list[str]) -> dict[str, BarStream]:
    """Subscribe each symbol via the fake so ``streams`` mirrors a live watchlist.

    Used by the direct-diff tests to build the initial state without
    standing up a full ``run_strategy_loop``. Resets the fake's
    subscribe/unsubscribe logs after the seed so assertions measure only
    the diff call's activity.
    """
    streams: dict[str, BarStream] = {}
    for s in symbols:
        streams[s] = await market_data.subscribe_bars(s)
    market_data.subscribed.clear()
    market_data.unsubscribed.clear()
    return streams


@pytest.mark.asyncio
async def test_diff_does_not_thrash_when_scan_exceeds_cap() -> None:
    """New declarative diff: 18-symbol scan + 10-cap → 8 subs + 8 unsubs, no thrash.

    Under Phase 6.2's procedural logic, a scan with new symbols ranked
    above existing ones would subscribe each new symbol then evict the
    oldest non-position in a cascade — a symbol at rank 3 could be
    subscribed then evicted later in the same diff call when a rank-11
    symbol was processed (cap-thrash).

    The declarative form computes ``target = top-N scan ∪ positions``
    first and diffs once. Here: scan is [NEW0..NEW7, A..J] (top 10 is
    NEW0..NEW7 + A + B), so ``to_add = NEW0..NEW7`` and
    ``to_drop = C..J`` — exactly 8 each, and no symbol appears in both.
    """
    from bot.orchestrator import _apply_watchlist_diff

    initial = [chr(ord("A") + i) for i in range(10)]  # A..J
    new = [f"NEW{i}" for i in range(8)]
    scan_order = new + initial  # 18 symbols; top-10 = NEW0..NEW7 + A + B

    frames = {s: pd.DataFrame({"close": [10.0]}) for s in new + initial}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    hits = [_hit(s) for s in scan_order]
    added, removed = await _apply_watchlist_diff(
        hits,
        streams=streams,
        market_data=cast("MarketData", fake),
        position_store=None,
        max_size=10,
    )
    assert set(added) == set(new), f"expected top-8 new adds, got {added}"
    assert len(added) == 8
    assert set(removed) == set(initial[2:]), f"expected C..J drops, got {removed}"
    assert len(removed) == 8
    # Strict no-thrash guarantee: no symbol both subscribed and unsubscribed.
    assert not (set(added) & set(removed))
    assert not (set(fake.subscribed) & set(fake.unsubscribed))
    # Final state: 10-symbol cap respected, exactly NEW0..NEW7 + A + B.
    assert set(streams.keys()) == set(new) | {"A", "B"}
    assert len(streams) == 10


@pytest.mark.asyncio
async def test_diff_preserves_active_positions_over_scanner_rank() -> None:
    """Active positions always survive, even when absent from the scan.

    watchlist={A..J}, position on A, scan returns {K..U} (11 symbols,
    A not in scan). Target = {K..T} ∪ {A} exceeds cap; positions win the
    tie. Fill remaining 9 slots from top-9 of scan. T and U are rejected.
    """
    from bot.orchestrator import _apply_watchlist_diff

    initial = [chr(ord("A") + i) for i in range(10)]  # A..J
    scan_order = [chr(ord("K") + i) for i in range(11)]  # K..U

    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + scan_order}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    position_store = _FakePositionStore(active={"A"})
    hits = [_hit(s) for s in scan_order]
    added, removed = await _apply_watchlist_diff(
        hits,
        streams=streams,
        market_data=cast("MarketData", fake),
        position_store=cast("PositionStore", position_store),
        max_size=10,
    )
    # A kept (position-protected); B..J dropped; K..S (9 scan keepers) added.
    assert "A" in streams
    assert set(added) == {chr(ord("K") + i) for i in range(9)}  # K..S
    assert set(removed) == {chr(ord("B") + i) for i in range(9)}  # B..J
    # T and U are beyond the cap — never subscribed.
    assert "T" not in streams
    assert "U" not in streams
    assert len(streams) == 10


@pytest.mark.asyncio
async def test_diff_rejects_new_when_all_slots_are_positions() -> None:
    """All 10 slots have active positions → zero subs, zero unsubs, WARNING fires."""
    import structlog
    from structlog.testing import capture_logs

    from bot.orchestrator import _apply_watchlist_diff

    structlog.reset_defaults()

    initial = [f"P{i:02d}" for i in range(10)]
    scan_order = [f"NEW{i}" for i in range(15)]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + scan_order}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    store = _FakePositionStore(active=set(initial))
    hits = [_hit(s) for s in scan_order]
    with capture_logs() as captured:
        added, removed = await _apply_watchlist_diff(
            hits,
            streams=streams,
            market_data=cast("MarketData", fake),
            position_store=cast("PositionStore", store),
            max_size=10,
        )
    assert added == []
    assert removed == []
    assert fake.subscribed == []
    assert fake.unsubscribed == []
    warnings = [
        e for e in captured if e.get("event") == "orchestrator.watchlist_full_no_eviction_possible"
    ]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["new_scan_hits_rejected"] == 15
    # All positions still present.
    assert set(streams.keys()) == set(initial)


@pytest.mark.asyncio
async def test_diff_top_10_wins() -> None:
    """cap=5, current={A..E}, scan={F..O} → 5 subs (F..J), 5 unsubs (A..E)."""
    from bot.orchestrator import _apply_watchlist_diff

    initial = ["A", "B", "C", "D", "E"]
    scan_order = ["F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + scan_order}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    hits = [_hit(s) for s in scan_order]
    added, removed = await _apply_watchlist_diff(
        hits,
        streams=streams,
        market_data=cast("MarketData", fake),
        position_store=None,
        max_size=5,
    )
    assert set(added) == {"F", "G", "H", "I", "J"}
    assert len(added) == 5
    assert set(removed) == set(initial)
    assert len(removed) == 5
    assert set(streams.keys()) == {"F", "G", "H", "I", "J"}


# ---------- Phase 6.4: last-evaluated-bar cursor ---------- #


class _CountingStrategy(Strategy):
    """Record every call to ``evaluate`` with the latest bar's timestamp; never fire."""

    name: str = "counting"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.seen_bars: list[object] = []

    def evaluate(self, symbol: str, bars: pd.DataFrame) -> Signal | None:
        self.calls += 1
        self.seen_bars.append(bars.index[-1] if len(bars.index) > 0 else None)
        return None


def _ny_frame(bar_ts: pd.Timestamp, close: float = 10.0) -> pd.DataFrame:
    """Build a single-bar DataFrame with a tz-aware DatetimeIndex."""
    return pd.DataFrame({"close": [close]}, index=pd.DatetimeIndex([bar_ts]))


@pytest.mark.asyncio
async def test_cursor_skips_already_evaluated_bar() -> None:
    """Same bar across iterations → strategy.evaluate called exactly once."""
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    bar_ts = pd.Timestamp.now(tz=ny)  # fresh bar
    bars = _ny_frame(bar_ts)

    frames = {"AAA": bars}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    strat = _CountingStrategy()
    await run_strategy_loop(
        watchlist=[_hit("AAA")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[strat],
        duration_minutes=0.02,  # ~1.2s; plenty of iterations at 0.01s poll
        poll_interval=0.01,
    )
    # Exactly one evaluation across all iterations — cursor de-dup works.
    assert strat.calls == 1, f"expected 1 call, got {strat.calls}"


@pytest.mark.asyncio
async def test_cursor_persists_across_resubscribe() -> None:
    """Drop+resubscribe cycle with identical backfilled bar → no re-evaluation.

    The cursor is keyed on ``(symbol, strategy_name)`` (not stream
    identity), so an unsubscribe does not clear it. When the symbol
    returns via a rescan with the same latest bar, the cursor check
    ``latest_bar_ts <= last_seen`` is True and evaluation is skipped.

    Drives the cycle via a scanner that returns ``[]`` on first call
    (dropping X) and ``[X]`` on subsequent calls (resubscribing). Duration
    is tuned to stay under the 100-iteration cursor sweep boundary so the
    cursor entry is not cleared by the periodic sweep mid-test.
    """
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    bar_ts = pd.Timestamp.now(tz=ny)
    bars = _ny_frame(bar_ts)

    frames = {"X": bars}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    strat = _CountingStrategy()
    # Batch 0 drops X; batch 1 (replayed thereafter) restores X.
    scanner = _FakeScanner([[], [_hit("X")]])
    await run_strategy_loop(
        watchlist=[_hit("X")],
        market_data=market_data,
        signal_bus=bus,
        strategies=[strat],
        duration_minutes=0.08,  # ~4.8s, enough for drop+readd
        poll_interval=0.05,
        settings=_rescan_settings(interval_seconds=1, max_size=10),  # type: ignore[arg-type]
        scanner=scanner,  # type: ignore[arg-type]
        position_store=_FakePositionStore(),  # type: ignore[arg-type]
    )
    fake = cast("_FakeMarketData", market_data)
    # Sanity: X was resubscribed (cycle actually happened).
    assert fake.subscribed.count("X") >= 1
    assert "X" in fake.unsubscribed
    # Cursor persisted across the cycle — single evaluation total.
    assert strat.calls == 1, f"cursor did not survive resubscribe: {strat.calls} calls"


# ---------- Phase 6.4: wall-time staleness guard ---------- #


def _stale_settings(threshold_seconds: int = 180) -> object:
    """Build a Settings with a custom ``bar_staleness_threshold_seconds``."""
    from bot.config import SessionConfig, Settings

    return Settings(
        session=SessionConfig(
            bar_staleness_threshold_seconds=threshold_seconds,
            watchlist_rescan_interval_seconds=60,
        )
    )


@pytest.mark.asyncio
async def test_bar_staleness_skips_evaluation() -> None:
    """5-min-old bar against 180s threshold → no evaluation, one ``strategy.bar_stale`` event."""
    from zoneinfo import ZoneInfo

    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ny = ZoneInfo("America/New_York")
    stale_ts = pd.Timestamp.now(tz=ny) - pd.Timedelta(minutes=5)
    bars = _ny_frame(stale_ts)

    frames = {"AAA": bars}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    strat = _CountingStrategy()
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[strat],
            duration_minutes=0.02,
            poll_interval=0.01,
            settings=_stale_settings(180),  # type: ignore[arg-type]
        )
    stale_events = [e for e in captured if e.get("event") == "strategy.bar_stale"]
    assert len(stale_events) == 1
    evt = stale_events[0]
    assert evt["symbol"] == "AAA"
    assert evt["strategy"] == "counting"
    assert evt["threshold_seconds"] == 180
    assert evt["staleness_seconds"] >= 180
    # latest_bar_ts is an ISO-formatted string in the log surface.
    assert isinstance(evt["latest_bar_ts"], str)
    # Strategy never saw the stale bar.
    assert strat.calls == 0


@pytest.mark.asyncio
async def test_bar_staleness_logs_once_per_stale_period() -> None:
    """Stale bar across many iterations → exactly one ``strategy.bar_stale`` event."""
    from zoneinfo import ZoneInfo

    import structlog
    from structlog.testing import capture_logs

    structlog.reset_defaults()

    ny = ZoneInfo("America/New_York")
    stale_ts = pd.Timestamp.now(tz=ny) - pd.Timedelta(seconds=30)
    bars = _ny_frame(stale_ts)

    frames = {"AAA": bars}
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    strat = _CountingStrategy()
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[strat],
            # ~3s window at 0.01s poll = ~300 iterations.
            duration_minutes=0.05,
            poll_interval=0.01,
            settings=_stale_settings(threshold_seconds=10),  # type: ignore[arg-type]
        )
    stale_events = [e for e in captured if e.get("event") == "strategy.bar_stale"]
    # One log on the first stale encounter; cursor de-dup keeps subsequent
    # iterations silent because the bar timestamp never advances.
    assert len(stale_events) == 1, (
        f"expected exactly 1 stale event across ~300 iterations, got {len(stale_events)}"
    )
    assert strat.calls == 0


@pytest.mark.asyncio
async def test_bar_staleness_resumes_on_new_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale → fresh bar transition: strategy evaluates the fresh bar normally."""
    from zoneinfo import ZoneInfo

    import structlog
    from structlog.testing import capture_logs

    from bot import orchestrator as orch_module

    structlog.reset_defaults()

    ny = ZoneInfo("America/New_York")

    call_n = {"i": 0}

    async def _swapping_fetch(stream: BarStream) -> pd.DataFrame:
        call_n["i"] += 1
        if call_n["i"] <= 2:
            ts = pd.Timestamp.now(tz=ny) - pd.Timedelta(seconds=30)  # stale
        else:
            ts = pd.Timestamp.now(tz=ny)  # fresh
        return _ny_frame(ts)

    monkeypatch.setattr(orch_module, "_fetch_bars", _swapping_fetch)

    frames = {"AAA": pd.DataFrame({"close": [10.0]})}  # unused; monkeypatched
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    strat = _CountingStrategy()
    with capture_logs() as captured:
        await run_strategy_loop(
            watchlist=[_hit("AAA")],
            market_data=market_data,
            signal_bus=bus,
            strategies=[strat],
            duration_minutes=0.03,
            poll_interval=0.01,
            settings=_stale_settings(threshold_seconds=10),  # type: ignore[arg-type]
        )
    stale_events = [e for e in captured if e.get("event") == "strategy.bar_stale"]
    # At least one stale log (from the early stale bars) and at least one
    # successful evaluation (from the fresh bars that followed).
    assert len(stale_events) >= 1
    assert strat.calls >= 1, "expected strategy to evaluate at least one fresh bar"


# ---------- Phase 12: on_symbol_dropped hook ---------- #


@pytest.mark.asyncio
async def test_apply_watchlist_diff_invokes_on_symbol_dropped_hook() -> None:
    """Phase 12: dropped symbols flow through ``on_symbol_dropped`` once each.

    The orchestrator wires the LLM catalyst classifier's
    ``on_watchlist_removal`` here so re-entered tickers re-evaluate
    fresh next time they appear in a scan.
    """
    from bot.orchestrator import _apply_watchlist_diff

    initial = ["A", "B", "C"]
    new_scan = ["B", "C", "D"]  # A dropped, D added
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + new_scan}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    dropped: list[str] = []
    added, removed = await _apply_watchlist_diff(
        [_hit(s) for s in new_scan],
        streams=streams,
        market_data=cast("MarketData", fake),
        position_store=None,
        max_size=10,
        on_symbol_dropped=dropped.append,
    )
    assert removed == ["A"]
    assert dropped == ["A"], "the dropped symbol must reach the hook exactly once"
    assert added == ["D"]


@pytest.mark.asyncio
async def test_apply_watchlist_diff_hook_failure_does_not_break_diff() -> None:
    """A misbehaving hook must not abort the watchlist diff."""
    import structlog
    from structlog.testing import capture_logs

    from bot.orchestrator import _apply_watchlist_diff

    structlog.reset_defaults()

    def _broken_hook(symbol: str) -> None:
        raise RuntimeError(f"hook bug for {symbol}")

    initial = ["A", "B"]
    new_scan = ["C"]
    frames = {s: pd.DataFrame({"close": [10.0]}) for s in initial + new_scan}
    fake = _FakeMarketData(frames)
    streams = await _seed_streams(fake, initial)

    with capture_logs() as captured:
        added, removed = await _apply_watchlist_diff(
            [_hit(s) for s in new_scan],
            streams=streams,
            market_data=cast("MarketData", fake),
            position_store=None,
            max_size=10,
            on_symbol_dropped=_broken_hook,
        )
    assert set(removed) == {"A", "B"}
    assert added == ["C"]
    events = [e.get("event") for e in captured]
    assert "orchestrator.on_symbol_dropped_failed" in events


# ---------- Phase 4g.1: rehab.enabled=false must short-circuit startup check ---------- #


@pytest.mark.asyncio
async def test_session_start_rehab_check_skips_load_when_disabled(
    tmp_path: object,
) -> None:
    """``rehab.enabled=false`` must skip flag-file load + startup notification.

    Pre-fix: the orchestrator unconditionally called ``load_state()`` and
    notified the operator that REHAB/DEEP_REHAB was active even though
    ``apply_to_caps`` correctly bypassed the loaded state. Operator saw
    "Rehab tier active" with scaled-down caps that weren't actually
    applied — pure observability/perception bug.
    """
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock

    from structlog.testing import capture_logs

    from bot.config import RehabConfig, RiskConfig, Settings
    from bot.orchestrator import Orchestrator
    from bot.risk.rehab import RehabEngine, RehabRecord, RehabTier, write_rehab_flag

    # Persist a non-NORMAL flag so we can prove it's NOT being loaded.
    flag_path = Path(cast("Any", tmp_path)) / "rehab.flag"
    write_rehab_flag(
        flag_path,
        RehabRecord(
            tier=RehabTier.DEEP_REHAB,
            trigger_reason="consecutive_red_days",
            entered_at=datetime.now(UTC),
            drawdown_at_entry_usd=-100.0,
            consecutive_red_days_at_entry=4,
        ),
    )

    settings = Settings(risk=RiskConfig(rehab=RehabConfig(enabled=False)))
    rehab_engine = RehabEngine(settings=settings, journal=None, flag_path=flag_path)
    orchestrator = Orchestrator(
        executor=cast("Any", MagicMock()),
        store=cast("Any", MagicMock()),
        settings=settings,
        auto_flatten=cast("Any", MagicMock()),
        rehab_engine=rehab_engine,
        notifier=cast("Any", MagicMock(send_rehab_tier_change=AsyncMock())),
    )

    with capture_logs() as captured:
        result = await orchestrator._session_start_rehab_check()

    assert result is None
    # The flag file's DEEP_REHAB tier must NOT be loaded into memory.
    assert rehab_engine.state.tier is RehabTier.NORMAL
    # No rehab.state_loaded event in the JSONL when disabled.
    events = [e.get("event") for e in captured]
    assert "rehab.state_loaded" not in events


@pytest.mark.asyncio
async def test_session_start_rehab_check_loads_flag_when_enabled(
    tmp_path: object,
) -> None:
    """Sanity: when enabled, the persisted flag IS read and ``rehab.state_loaded`` fires."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock

    from structlog.testing import capture_logs

    from bot.config import RehabConfig, RiskConfig, Settings
    from bot.orchestrator import Orchestrator
    from bot.risk.rehab import RehabEngine, RehabRecord, RehabTier, write_rehab_flag

    flag_path = Path(cast("Any", tmp_path)) / "rehab.flag"
    write_rehab_flag(
        flag_path,
        RehabRecord(
            tier=RehabTier.REHAB,
            trigger_reason="consecutive_red_days",
            entered_at=datetime.now(UTC),
            drawdown_at_entry_usd=-50.0,
            consecutive_red_days_at_entry=2,
        ),
    )

    settings = Settings(risk=RiskConfig(rehab=RehabConfig(enabled=True)))
    rehab_engine = RehabEngine(settings=settings, journal=None, flag_path=flag_path)
    orchestrator = Orchestrator(
        executor=cast("Any", MagicMock()),
        store=cast("Any", MagicMock()),
        settings=settings,
        auto_flatten=cast("Any", MagicMock()),
        rehab_engine=rehab_engine,
        notifier=cast("Any", MagicMock(send_rehab_tier_change=AsyncMock())),
    )

    with capture_logs() as captured:
        result = await orchestrator._session_start_rehab_check()

    # Flag was read off disk regardless of any post-load tier transition.
    events = [e.get("event") for e in captured]
    assert "rehab.state_loaded" in events
    # Result is whatever tier the engine settled on after check_transitions
    # (with no journal entries the empty-stats path may downgrade to NORMAL);
    # the contract under test is simply "non-None when enabled, signalling
    # the CLI to print the rehab summary".
    assert result is not None
