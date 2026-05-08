"""Phase 12.4 tests -- per-strategy catalyst admission + recent-window RVOL suppression.

Three suites:

1. ``test_eligible_strategies_for_*``: per-strategy admission (the unit
   tests for the (catalyst_confirmed, strategy.catalyst_required) truth
   table laid out in the spec).
2. ``test_recent_window_rvol_*``: signal suppression at the strategy
   level (both gap-and-go and momentum exercise the suppression path,
   with structured log events asserted).
3. ``test_scanner_admission_*``: end-to-end scanner flow exercising the
   technical-only path (catalyst classifier returns ``qualifies=False``;
   ticker still admitted to momentum, dropped from gap-and-go).

Phase 12.4 tests opt in to the recent-rvol gate via the
``recent_rvol_enabled`` marker (the conftest autouse fixture
default-disables it for all other tests).
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import structlog

from bot.brokerage.ibkr_client import SubscriptionRegistry
from bot.config import Settings, UniverseConfig
from bot.scanning.finnhub_client import NewsItem
from bot.scanning.float_source import FloatSource
from bot.scanning.scanner import IBKRScanner, ScanHit
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy
from bot.strategies.volume import check_recent_window_rvol

# ---------- Fixture helpers ----------


def _ny_ts(hour: int, minute: int) -> pd.Timestamp:
    """Build a NY-tz timestamp for today at HH:MM."""
    today = datetime.now(UTC).date()
    naive = pd.Timestamp(datetime.combine(today, time(hour, minute)))
    return naive.tz_localize("America/New_York")


def _rising_bar_frame(
    *,
    bar_count: int,
    last_close: float,
    last_volume: float,
    prior_volume: float = 1000.0,
    starting_low: float = 1.00,
) -> pd.DataFrame:
    """Build a bars DataFrame with a clean rising HOD-breaking pattern.

    The breakout bar (last) closes at ``last_close`` with volume
    ``last_volume``; prior bars sit just below the breakout close on
    flat ``prior_volume`` to keep the recent-window average stable.

    Used for momentum / gap_and_go signal-emission tests where we want
    pattern checks to pass and only exercise the volume gate.
    """
    bars = []
    base_close = last_close - 0.05  # prior bars sit just under breakout close
    base_low = starting_low
    base_high = base_close + 0.01
    for _i in range(bar_count - 1):
        # Rising series so HOD breaks cleanly on the last bar; volume flat.
        bars.append(
            {
                "open": base_close,
                "high": base_high,
                "low": base_low,
                "close": base_close,
                "volume": prior_volume,
            }
        )
    bars.append(
        {
            "open": base_close,
            "high": last_close + 0.05,
            "low": base_close - 0.01,
            "close": last_close,
            "volume": last_volume,
        }
    )
    # Phase 12.6: anchor fixtures at 10:00 ET so they land inside both
    # gap-and-go's default opening window (09:30-10:00 -- gap_and_go
    # tests pass via the explicit ``vwap_extension_grace_minutes=60``
    # bypass, not the timestamp anchor) AND momentum's default
    # window_start of 10:00. Pre-12.6 fixtures stamped at 09:30
    # silently dropped under momentum's new ``_within_window``.
    timestamps = [_ny_ts(10, 0) + timedelta(minutes=i) for i in range(bar_count)]
    df = pd.DataFrame(bars, index=pd.DatetimeIndex(timestamps))
    return df


# ============================================================
# Suite 1: per-strategy admission unit tests
# ============================================================


def _hit_with_admission(
    symbol: str = "ATRA", *, catalyst_confirmed: bool, manual: bool = False
) -> ScanHit:
    """ScanHit fixture for admission-truth-table tests."""
    return ScanHit(
        symbol=symbol,
        price=10.0,
        change_pct=5.0,
        volume=500_000,
        float_shares=4_000_000,
        catalyst="clinical_data" if catalyst_confirmed else None,
        catalyst_confirmed=catalyst_confirmed,
        manual=manual,
    )


def _scanner_with_strategy_settings(
    *, gap_required: bool, momentum_required: bool
) -> IBKRScanner:
    """Build a scanner whose only purpose is to exercise the admission helpers."""
    base = Settings(universe=UniverseConfig(rvol_min=0.0))
    settings = base.model_copy(
        update={
            "strategies": base.strategies.model_copy(
                update={
                    "gap_and_go": base.strategies.gap_and_go.model_copy(
                        update={"catalyst_required": gap_required}
                    ),
                    "momentum": base.strategies.momentum.model_copy(
                        update={"catalyst_required": momentum_required}
                    ),
                }
            ),
        }
    )
    ibkr = MagicMock()
    ibkr.subscriptions = SubscriptionRegistry()
    finnhub = MagicMock()
    return IBKRScanner(ibkr=ibkr, finnhub=finnhub, settings=settings)


def test_admission_confirmed_admitted_to_both_strategies() -> None:
    """catalyst_confirmed=True with both strategies enabled → admitted to both."""
    scanner = _scanner_with_strategy_settings(gap_required=True, momentum_required=False)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=True)
    )
    assert sorted(eligible) == ["gap_and_go", "momentum"]


def test_admission_unconfirmed_admitted_only_to_relaxed_strategies() -> None:
    """catalyst_confirmed=False → admitted only to strategies with catalyst_required=False."""
    scanner = _scanner_with_strategy_settings(gap_required=True, momentum_required=False)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=False)
    )
    assert eligible == ["momentum"]


def test_admission_unconfirmed_admitted_to_neither_when_both_strict() -> None:
    """Both strategies catalyst_required=True + unconfirmed catalyst → eligible nowhere."""
    scanner = _scanner_with_strategy_settings(gap_required=True, momentum_required=True)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=False)
    )
    assert eligible == []


def test_admission_unconfirmed_admitted_to_both_when_both_relaxed() -> None:
    """Both strategies catalyst_required=False → unconfirmed catalyst still admitted to both."""
    scanner = _scanner_with_strategy_settings(gap_required=False, momentum_required=False)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=False)
    )
    assert sorted(eligible) == ["gap_and_go", "momentum"]


def test_admission_manual_hit_admitted_to_both_regardless_of_catalyst() -> None:
    """Manual watchlist hit bypasses every catalyst gate -- operator override."""
    scanner = _scanner_with_strategy_settings(gap_required=True, momentum_required=True)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=False, manual=True)
    )
    assert sorted(eligible) == ["gap_and_go", "momentum"]


def test_admission_disabled_strategy_never_appears_in_eligible() -> None:
    """A disabled strategy is not in the eligible list even if its catalyst rule would admit."""
    base = Settings(universe=UniverseConfig(rvol_min=0.0))
    settings = base.model_copy(
        update={
            "strategies": base.strategies.model_copy(
                update={
                    "gap_and_go": base.strategies.gap_and_go.model_copy(
                        update={"enabled": False, "catalyst_required": False}
                    ),
                    "momentum": base.strategies.momentum.model_copy(
                        update={"enabled": True, "catalyst_required": False}
                    ),
                }
            ),
        }
    )
    ibkr = MagicMock()
    ibkr.subscriptions = SubscriptionRegistry()
    scanner = IBKRScanner(ibkr=ibkr, finnhub=MagicMock(), settings=settings)
    eligible = scanner._eligible_strategies_for(
        _hit_with_admission(catalyst_confirmed=True)
    )
    assert eligible == ["momentum"]


# ============================================================
# Suite 2: recent-window RVOL suppression at strategy level
# ============================================================


@pytest.mark.recent_rvol_enabled
def test_check_recent_window_rvol_passes_when_breakout_volume_high() -> None:
    """Candidate volume >> prior-window-average → returns None (no suppression)."""
    bars = _rising_bar_frame(
        bar_count=21,  # 20 prior + 1 candidate
        last_close=2.50,
        last_volume=10_000,
        prior_volume=1_000,
    )
    result = check_recent_window_rvol(
        bars=bars,
        window_bars=20,
        threshold=2.0,
        symbol="ATRA",
        strategy="gap_and_go",
        bar_time=bars.index[-1],
    )
    assert result is None


@pytest.mark.recent_rvol_enabled
def test_check_recent_window_rvol_suppresses_when_volume_below_threshold() -> None:
    """Candidate volume < threshold × average → returns 'low_recent_rvol' + logs event."""
    bars = _rising_bar_frame(
        bar_count=21,
        last_close=2.50,
        last_volume=1_500,  # 1.5x prior_volume; below 2.0x threshold
        prior_volume=1_000,
    )
    with structlog.testing.capture_logs() as logs:
        result = check_recent_window_rvol(
            bars=bars,
            window_bars=20,
            threshold=2.0,
            symbol="WEAK",
            strategy="momentum",
            bar_time=bars.index[-1],
        )
    assert result == "low_recent_rvol"
    suppression = [
        e for e in logs if e.get("event") == "strategy.signal_suppressed_recent_rvol"
    ]
    assert len(suppression) == 1
    record = suppression[0]
    assert record["symbol"] == "WEAK"
    assert record["strategy"] == "momentum"
    assert record["threshold"] == 2.0
    assert record["window_bars"] == 20
    assert record["candidate_volume"] == pytest.approx(1500.0)
    assert record["window_average"] == pytest.approx(1000.0)
    assert record["rvol"] == pytest.approx(1.5)


@pytest.mark.recent_rvol_enabled
def test_check_recent_window_rvol_suppresses_when_window_not_populated() -> None:
    """Fewer than window_bars+1 bars → returns 'window_not_populated' + logs event."""
    bars = _rising_bar_frame(
        bar_count=10,  # fewer than 20+1 needed
        last_close=2.50,
        last_volume=10_000,
        prior_volume=1_000,
    )
    with structlog.testing.capture_logs() as logs:
        result = check_recent_window_rvol(
            bars=bars,
            window_bars=20,
            threshold=2.0,
            symbol="EARLY",
            strategy="gap_and_go",
            bar_time=bars.index[-1],
        )
    assert result == "window_not_populated"
    not_pop = [
        e for e in logs
        if e.get("event") == "strategy.signal_suppressed_window_not_populated"
    ]
    assert len(not_pop) == 1
    assert not_pop[0]["bars_available"] == 10
    assert not_pop[0]["window_required"] == 21


@pytest.mark.recent_rvol_enabled
def test_gap_and_go_emits_when_breakout_volume_high() -> None:
    """End-to-end strategy: high recent-rvol → signal fires."""
    bars = _rising_bar_frame(
        bar_count=21,
        last_close=2.50,
        last_volume=10_000,
        prior_volume=1_000,
    )
    strategy = GapAndGoStrategy(
        vwap_extension_grace_minutes=60,  # bypass extension check for test simplicity
        recent_rvol_min=2.0,
        recent_rvol_window_bars=20,
        window_end=time(16, 0),
        stop_floor_min_abs=0.0,
        stop_floor_min_pct=0.0,
    )
    sig = strategy.evaluate("ATRA", bars)
    assert sig is not None
    assert sig.symbol == "ATRA"


@pytest.mark.recent_rvol_enabled
def test_gap_and_go_suppresses_when_breakout_volume_low() -> None:
    """End-to-end strategy: low recent-rvol → no signal, suppression event logged."""
    bars = _rising_bar_frame(
        bar_count=21,
        last_close=2.50,
        last_volume=1_500,  # 1.5x prior, below 2.0 threshold
        prior_volume=1_000,
    )
    strategy = GapAndGoStrategy(
        vwap_extension_grace_minutes=60,
        recent_rvol_min=2.0,
        recent_rvol_window_bars=20,
        window_end=time(16, 0),
        stop_floor_min_abs=0.0,
        stop_floor_min_pct=0.0,
    )
    with structlog.testing.capture_logs() as logs:
        sig = strategy.evaluate("WEAK", bars)
    assert sig is None
    events = [e.get("event") for e in logs]
    assert "strategy.signal_suppressed_recent_rvol" in events
    # No signal.emitted log (suppressed before that point).
    assert "signal.emitted" not in events


@pytest.mark.recent_rvol_enabled
def test_momentum_emits_when_breakout_volume_high() -> None:
    """End-to-end strategy: momentum fires on high recent-rvol breakout."""
    bars = _rising_bar_frame(
        bar_count=21,
        last_close=2.50,
        last_volume=10_000,
        prior_volume=1_000,
    )
    strategy = MomentumStrategy(
        flag_max_pullback_pct=50.0,  # very lax pullback so flag check passes
        extended_from_vwap_atr_multiple=20.0,
        recent_rvol_min=2.0,
        recent_rvol_window_bars=20,
        window_end=time(16, 0),
        stop_floor_min_abs=0.0,
        stop_floor_min_pct=0.0,
    )
    sig = strategy.evaluate("ATRA", bars)
    assert sig is not None


@pytest.mark.recent_rvol_enabled
def test_momentum_suppresses_when_breakout_volume_low() -> None:
    """End-to-end strategy: momentum suppresses on weak breakout volume."""
    bars = _rising_bar_frame(
        bar_count=21,
        last_close=2.50,
        last_volume=1_500,
        prior_volume=1_000,
    )
    strategy = MomentumStrategy(
        flag_max_pullback_pct=50.0,
        extended_from_vwap_atr_multiple=20.0,
        recent_rvol_min=2.0,
        recent_rvol_window_bars=20,
        window_end=time(16, 0),
        stop_floor_min_abs=0.0,
        stop_floor_min_pct=0.0,
    )
    with structlog.testing.capture_logs() as logs:
        sig = strategy.evaluate("WEAK", bars)
    assert sig is None
    suppressed = [
        e for e in logs if e.get("event") == "strategy.signal_suppressed_recent_rvol"
    ]
    assert len(suppressed) == 1
    assert suppressed[0]["strategy"] == "momentum"


# ============================================================
# Suite 3: end-to-end scanner flow with technical-only path
# ============================================================


def _fake_scan_row(symbol: str) -> SimpleNamespace:
    """Duck-typed ib_async scan row stand-in."""
    return SimpleNamespace(
        contractDetails=SimpleNamespace(contract=SimpleNamespace(symbol=symbol))
    )


def _mock_ibkr_with_symbols(symbols: list[str]) -> MagicMock:
    """IBKRClient mock returning a scan with the given symbols, no historical bars."""
    ibkr = MagicMock(name="IBKRClient")
    ibkr.ib = MagicMock(name="IB")
    ibkr.ib.reqScannerDataAsync = AsyncMock(
        return_value=[_fake_scan_row(s) for s in symbols]
    )
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
    ibkr.ib.cancelScannerSubscription = MagicMock()
    ibkr.ib.cancelHistoricalData = MagicMock()
    ibkr.subscriptions = SubscriptionRegistry()
    return ibkr


def _mock_finnhub() -> MagicMock:
    """FinnhubClient stub. company_news returns empty (technical-only path)."""
    finnhub = MagicMock(name="FinnhubClient")
    finnhub.company_news = AsyncMock(return_value=[])
    finnhub.company_profile = AsyncMock(return_value=None)
    return finnhub


def _float_source(yf_map: dict[str, int]) -> FloatSource:
    def fetch(symbol: str) -> int | None:
        return yf_map.get(symbol)

    return FloatSource(finnhub=None, yfinance_fetcher=fetch)


def _diff_settings(*, gap_required: bool, momentum_required: bool) -> Settings:
    """Settings tuned for end-to-end scanner tests in the differentiation suite.

    Pins keyword classifier on (so we don't need anthropic), rvol pillar
    off (synthetic frames don't carry avg_daily_volume), per-strategy
    catalyst_required tunable, and ``allow_catalyst_overrides=False``
    so a real ``data/test_catalyst_overrides.json`` on the operator's
    machine can't bleed in.
    """
    base = Settings(universe=UniverseConfig(rvol_min=0.0, float_max=20_000_000))
    return base.model_copy(
        update={
            "catalyst_classifier": base.catalyst_classifier.model_copy(
                update={
                    "llm": base.catalyst_classifier.llm.model_copy(update={"enabled": False}),
                    "keyword": base.catalyst_classifier.keyword.model_copy(
                        update={"enabled": True}
                    ),
                }
            ),
            "testing": base.testing.model_copy(update={"allow_catalyst_overrides": False}),
            "strategies": base.strategies.model_copy(
                update={
                    "gap_and_go": base.strategies.gap_and_go.model_copy(
                        update={"catalyst_required": gap_required}
                    ),
                    "momentum": base.strategies.momentum.model_copy(
                        update={"catalyst_required": momentum_required}
                    ),
                }
            ),
        }
    )


@pytest.mark.asyncio
async def test_scanner_admits_unconfirmed_to_momentum_only_when_split() -> None:
    """Unconfirmed-catalyst ticker stays on watchlist for momentum, dropped from gap-and-go."""
    settings = _diff_settings(gap_required=True, momentum_required=False)
    ibkr = _mock_ibkr_with_symbols(["TECHX"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(),  # no news → keyword classifier returns no catalyst
        settings=settings,
        float_source=_float_source({"TECHX": 5_000_000}),
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    # Hit survives because momentum admits unconfirmed.
    assert [h.symbol for h in hits] == ["TECHX"]
    assert hits[0].catalyst_confirmed is False
    admitted = [e for e in logs if e.get("event") == "scanner.watchlist_admitted"]
    assert len(admitted) == 1
    assert admitted[0]["symbol"] == "TECHX"
    assert admitted[0]["catalyst_confirmed"] is False
    assert admitted[0]["eligible_strategies"] == ["momentum"]


@pytest.mark.asyncio
async def test_scanner_drops_unconfirmed_when_no_strategy_admits() -> None:
    """Unconfirmed-catalyst ticker dropped when both strategies require catalyst."""
    settings = _diff_settings(gap_required=True, momentum_required=True)
    ibkr = _mock_ibkr_with_symbols(["TECHX"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(),
        settings=settings,
        float_source=_float_source({"TECHX": 5_000_000}),
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    assert hits == []
    drops = [e for e in logs if e.get("event") == "scanner.dropped_no_catalyst"]
    assert len(drops) == 1
    drop = drops[0]
    assert drop["symbol"] == "TECHX"
    assert drop["catalyst_confirmed"] is False
    assert drop["any_strategy_admits_unconfirmed"] is False


@pytest.mark.asyncio
async def test_scanner_admits_confirmed_catalyst_to_both_strategies() -> None:
    """Confirmed-catalyst ticker admitted to gap-and-go AND momentum."""
    settings = _diff_settings(gap_required=True, momentum_required=False)
    ibkr = _mock_ibkr_with_symbols(["BIGRX"])
    finnhub = MagicMock()
    finnhub.company_news = AsyncMock(
        return_value=[
            NewsItem(
                headline="BIGRX tops estimates",
                source="test",
                url="https://example.com/x",
                datetime=datetime.now(UTC),
                summary="",
                category="company",
            )
        ]
    )
    finnhub.company_profile = AsyncMock(return_value=None)
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=finnhub,
        settings=settings,
        float_source=_float_source({"BIGRX": 5_000_000}),
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    assert [h.symbol for h in hits] == ["BIGRX"]
    assert hits[0].catalyst_confirmed is True
    admitted = [e for e in logs if e.get("event") == "scanner.watchlist_admitted"]
    assert sorted(admitted[0]["eligible_strategies"]) == ["gap_and_go", "momentum"]


@pytest.mark.asyncio
async def test_scanner_daily_rvol_filter_still_drops_low_rvol() -> None:
    """Phase 12.1 rvol pillar stays load-bearing -- unrelated to Phase 12.4 admission."""
    base = Settings(universe=UniverseConfig(rvol_min=5.0, float_max=20_000_000))
    settings = base.model_copy(
        update={
            "catalyst_classifier": base.catalyst_classifier.model_copy(
                update={
                    "llm": base.catalyst_classifier.llm.model_copy(update={"enabled": False}),
                    "keyword": base.catalyst_classifier.keyword.model_copy(
                        update={"enabled": True}
                    ),
                }
            ),
            "testing": base.testing.model_copy(update={"allow_catalyst_overrides": False}),
            "strategies": base.strategies.model_copy(
                update={
                    "gap_and_go": base.strategies.gap_and_go.model_copy(
                        update={"catalyst_required": True}
                    ),
                    "momentum": base.strategies.momentum.model_copy(
                        update={"catalyst_required": False}
                    ),
                }
            ),
        }
    )
    # Yfinance fetcher returns float + avg_volume; today's volume on the
    # synthetic IBKR bar is too low for rvol >= 5.
    yf_map: dict[str, tuple[int | None, int | None]] = {
        "LOWRX": (5_000_000, 1_000_000),  # 1M avg daily vol
    }

    def fetch(symbol: str) -> tuple[int | None, int | None]:
        return yf_map.get(symbol, (None, None))

    fs = FloatSource(finnhub=None, yfinance_fetcher=fetch)
    ibkr = _mock_ibkr_with_symbols(["LOWRX"])

    def _bar(close: float, volume: float) -> SimpleNamespace:
        return SimpleNamespace(close=close, volume=volume)

    # Today's volume = 200k vs avg 1M → rvol 0.2, below threshold 5.0.
    ibkr.ib.reqHistoricalDataAsync = AsyncMock(
        return_value=[_bar(2.0, 100_000), _bar(2.5, 200_000)]
    )
    finnhub = MagicMock()
    finnhub.company_news = AsyncMock(
        return_value=[
            NewsItem(
                headline="LOWRX tops estimates",
                source="test",
                url="https://example.com/x",
                datetime=datetime.now(UTC),
                summary="",
                category="company",
            )
        ]
    )
    finnhub.company_profile = AsyncMock(return_value=None)
    scanner = IBKRScanner(
        ibkr=ibkr, finnhub=finnhub, settings=settings, float_source=fs
    )
    with structlog.testing.capture_logs() as logs:
        hits = await scanner.scan_top_gappers()
    # Daily-rvol filter dropped the ticker before catalyst evaluation.
    assert hits == []
    rvol_drops = [e for e in logs if e.get("event") == "scanner.dropped_low_rvol"]
    assert len(rvol_drops) == 1
    assert rvol_drops[0]["symbol"] == "LOWRX"
    # Watchlist-admitted event NEVER fires (ticker dropped before).
    assert not any(e.get("event") == "scanner.watchlist_admitted" for e in logs)


@pytest.mark.asyncio
async def test_admission_log_records_catalyst_confirmed_and_eligible_list() -> None:
    """The scanner.watchlist_admitted event must carry the full forensic context."""
    settings = _diff_settings(gap_required=True, momentum_required=False)
    ibkr = _mock_ibkr_with_symbols(["PLAINX"])
    scanner = IBKRScanner(
        ibkr=ibkr,
        finnhub=_mock_finnhub(),
        settings=settings,
        float_source=_float_source({"PLAINX": 5_000_000}),
    )
    with structlog.testing.capture_logs() as logs:
        await scanner.scan_top_gappers()
    admitted = [e for e in logs if e.get("event") == "scanner.watchlist_admitted"]
    assert len(admitted) == 1
    record = admitted[0]
    assert "symbol" in record
    assert "catalyst_confirmed" in record
    assert "eligible_strategies" in record
    assert isinstance(record["eligible_strategies"], list)


# ============================================================
# Suite 4: dispatcher-level admission integration
# ============================================================


@pytest.mark.asyncio
async def test_dispatcher_skips_strict_strategy_for_unconfirmed_symbol() -> None:
    """Orchestrator.run_strategy_loop: a catalyst_required strategy never sees an unconfirmed symbol."""
    from typing import cast

    from bot.brokerage.market_data import MarketData
    from bot.orchestrator import run_strategy_loop
    from bot.signal_bus import SignalBus
    from bot.strategies.base import Signal, Strategy

    seen: dict[str, list[str]] = {}

    class _RecordingStrategy(Strategy):
        def __init__(self, name: str, catalyst_required: bool) -> None:
            super().__init__()
            self.name = name
            self.catalyst_required = catalyst_required

        def evaluate(self, symbol: str, _bars: pd.DataFrame) -> Signal | None:
            seen.setdefault(self.name, []).append(symbol)
            return None

    strict = _RecordingStrategy("strict_strategy", catalyst_required=True)
    relaxed = _RecordingStrategy("relaxed_strategy", catalyst_required=False)

    # Reuse the FakeMarketData from the orchestrator tests.
    from tests.test_orchestrator import _FakeMarketData

    frames = {
        "TECHX": pd.DataFrame({"close": [10.0]}),
        "BIGRX": pd.DataFrame({"close": [10.0]}),
    }
    market_data = cast("MarketData", _FakeMarketData(frames))
    bus = SignalBus()
    watchlist = [
        ScanHit(  # unconfirmed → strict skips, relaxed evaluates
            symbol="TECHX",
            price=None,
            change_pct=None,
            volume=None,
            float_shares=None,
            catalyst=None,
            catalyst_confirmed=False,
        ),
        ScanHit(  # confirmed → both evaluate
            symbol="BIGRX",
            price=None,
            change_pct=None,
            volume=None,
            float_shares=None,
            catalyst="clinical_data",
            catalyst_confirmed=True,
        ),
    ]
    await run_strategy_loop(
        watchlist=watchlist,
        market_data=market_data,
        signal_bus=bus,
        strategies=[strict, relaxed],
        duration_minutes=0.02,
        poll_interval=0.01,
    )
    assert "TECHX" not in seen.get("strict_strategy", [])
    assert "BIGRX" in seen.get("strict_strategy", [])
    assert "TECHX" in seen.get("relaxed_strategy", [])
    assert "BIGRX" in seen.get("relaxed_strategy", [])
