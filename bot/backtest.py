"""Historical replay tool — run Phase 3 strategies bar-by-bar against a past date.

Phase 3.5 single-purpose: "given a ticker and a date, show me every signal the
strategies would have fired." Not a backtester — no P&L, no equity curve, no
optimization. Just a sanity check that ``GapAndGoStrategy`` and
``MomentumStrategy`` actually fire on known historical setups before we add
real order execution in Phase 4.

Phase 3.6 addition: the replay also captures ``signal.rejected`` structlog
events (emitted by strategies and the R:R gate) and surfaces them on
``ReplayResult.rejections``. That makes zero-signal runs diagnostic instead of
opaque — you can see which filter killed each candidate.

Two load-bearing limitations to keep in mind when reading the output:

1. **Catalyst gate is replay-side, not strategy-side.** The live bot's catalyst
   check lives in the Phase 2 scanner (Finnhub news classified against the green/black list). Historical news replay is out of scope here, so by
   default we skip Gap-and-Go entirely — that strategy is catalyst-gated by
   design and running it without one produces misleading signals. Pass
   ``--catalyst`` to force a ``manual_override`` catalyst so both strategies
   participate; this is a replay convenience, not a live-bot behaviour.

2. **Signal list is a lower bound** on what the live bot would have produced:
   in real time the orchestrator re-evaluates on every bar update, whereas
   replay iterates once per closed bar. Transient setups that fire and
   invalidate within a minute can be missed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from ib_async import Contract
from structlog.testing import capture_logs

from bot.brokerage.ibkr_client import ActiveSubscription, IBKRClient, ref_req_id
from bot.brokerage.market_data import MarketData
from bot.scanning.scanner import ScanHit
from bot.strategies.base import (
    REJECTION_EVENT,
    RejectedCandidate,
    Signal,
    Strategy,
)
from bot.strategies.gap_and_go import GapAndGoStrategy
from bot.strategies.momentum import MomentumStrategy

_log = structlog.get_logger("bot.backtest")

_NY = ZoneInfo("America/New_York")
_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]

STRATEGY_GAP_AND_GO = "gap_and_go"
STRATEGY_MOMENTUM = "momentum"
STRATEGY_BOTH = "both"
_STRATEGY_CHOICES = (STRATEGY_GAP_AND_GO, STRATEGY_MOMENTUM, STRATEGY_BOTH)

# Keys the rejection parser treats as canonical fields; everything else on the
# event dict becomes part of RejectedCandidate.context.
_CANONICAL_REJECTION_KEYS = frozenset(
    {"event", "log_level", "timestamp", "symbol", "strategy", "bar_time", "stage", "reason"}
)


class ReplayError(RuntimeError):
    """Raised when the replay cannot proceed (stale date, unqualifiable symbol, etc.)."""


@dataclass
class ReplayResult:
    """Return payload of ``Replayer.replay`` — signals + rejections + fabricated context."""

    signals: list[Signal]
    context: ScanHit
    rejections: list[RejectedCandidate] = field(default_factory=list)


class Replayer:
    """Bar-by-bar historical replay of the Phase 3 strategies against one symbol / date."""

    def __init__(
        self,
        ibkr: IBKRClient,
        market_data: MarketData,
        gap_and_go: GapAndGoStrategy,
        momentum: MomentumStrategy,
    ) -> None:
        """Wire the IBKR client, MarketData, and both strategy instances."""
        self._ibkr = ibkr
        self._market_data = market_data
        self._gap_and_go = gap_and_go
        self._momentum = momentum

    async def replay(
        self,
        symbol: str,
        target_date: date,
        *,
        strategy_selection: str = STRATEGY_BOTH,
        force_catalyst: bool = False,
    ) -> ReplayResult:
        """Fetch 2 days of 1-min bars, iterate target-day bars, emit every signal + rejection."""
        if strategy_selection not in _STRATEGY_CHOICES:
            raise ValueError(
                f"strategy_selection must be one of {_STRATEGY_CHOICES}, got {strategy_selection!r}"
            )
        contract = await self._ibkr.qualify_stock(symbol)
        bars = await self._fetch_bars(contract, target_date)
        context = self._build_context(symbol, bars, force_catalyst)

        if bars.empty:
            _log.warning("backtest.no_bars", symbol=symbol, date=target_date.isoformat())
            return ReplayResult(signals=[], context=context, rejections=[])

        target_mask = [d == target_date for d in _index_dates(bars)]
        target_positions = [i for i, keep in enumerate(target_mask) if keep]
        if not target_positions:
            _log.warning("backtest.no_target_day_bars", symbol=symbol, date=target_date.isoformat())
            return ReplayResult(signals=[], context=context, rejections=[])

        collected: list[Signal] = []
        seen: set[tuple[str, str]] = set()

        # capture_logs temporarily swaps the structlog processor chain so we
        # can harvest every ``signal.rejected`` event without plumbing a
        # collector through strategy signatures. Side-effect: other structlog
        # events inside this block don't render to stdout — keep the block
        # tight and emit the summary lines outside it.
        with capture_logs() as captured:
            gap_and_go_selected = strategy_selection in (STRATEGY_GAP_AND_GO, STRATEGY_BOTH)
            if gap_and_go_selected and not force_catalyst:
                # One synthetic rejection representing the scanner-level
                # catalyst gate that would have suppressed Gap-and-Go live.
                self._emit_catalyst_rejection(symbol, target_date, strategy_selection)

            strategies = self._select_strategies(strategy_selection, force_catalyst)
            if strategies:
                for i in target_positions:
                    prefix = bars.iloc[: i + 1]
                    for strategy in strategies:
                        signal = strategy.evaluate(symbol, prefix)
                        if signal is None:
                            continue
                        # Phase 8.1: post-emission R:R filter removed (see
                        # orchestrator + Strategy base docstring). Scale-out
                        # is constructed at a fixed multiple of initial risk,
                        # so R:R is pinned by definition and the old check
                        # was tautological.
                        key = (strategy.name, signal.timestamp.isoformat())
                        if key in seen:
                            continue
                        seen.add(key)
                        collected.append(signal)

        rejections = _parse_rejections(captured)
        collected.sort(key=lambda s: s.timestamp)
        rejections.sort(key=lambda r: r.bar_time)
        _log.info(
            "backtest.replay_complete",
            symbol=symbol,
            date=target_date.isoformat(),
            bar_count=len(bars),
            target_bar_count=len(target_positions),
            signal_count=len(collected),
            rejection_count=len(rejections),
            force_catalyst=force_catalyst,
        )
        return ReplayResult(signals=collected, context=context, rejections=rejections)

    async def _fetch_bars(self, contract: Contract, target_date: date) -> pd.DataFrame:
        """Pull 2 days of 1-min bars ending at target date 23:59:59 ET — premarket included."""
        end_dt = f"{target_date.strftime('%Y%m%d')} 23:59:59 US/Eastern"
        symbol = getattr(contract, "symbol", None)
        req_id: int | None = None
        try:
            bar_list = await self._ibkr.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr="2 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
                keepUpToDate=False,
            )
            # Register the returned bar list so a Ctrl-C during backtest (which
            # triggers cancel_all_subscriptions via disconnect) can release the
            # one-shot historical request on the TWS side before the socket
            # closes. Unregister on the normal return path below.
            req_id = ref_req_id(bar_list)
            await self._ibkr.subscriptions.register(
                ActiveSubscription(
                    req_id=req_id,
                    kind="historical",
                    symbol=symbol,
                    ref=bar_list,
                )
            )
        except Exception as exc:  # noqa: BLE001 - IBKR raises many shapes
            message = str(exc)
            lowered = message.lower()
            if "historical data" in lowered or "no data" in lowered or "pacing" in lowered:
                raise ReplayError(
                    "1-min historical bars are only available for ~6 months; "
                    "pick a more recent date."
                ) from exc
            raise ReplayError(f"IBKR historical data request failed: {message}") from exc
        try:
            return _bars_to_frame(bar_list)
        finally:
            if req_id is not None:
                await self._ibkr.subscriptions.unregister(req_id)

    def _build_context(
        self,
        symbol: str,
        bars: pd.DataFrame,
        force_catalyst: bool,
    ) -> ScanHit:
        """Fabricate a minimal ScanHit matching what a live scanner would have produced."""
        open_price, change_pct, cumulative_volume = self._session_open_metrics(bars)
        return ScanHit(
            symbol=symbol,
            price=open_price,
            change_pct=change_pct,
            volume=cumulative_volume,
            float_shares=None,
            catalyst="manual_override" if force_catalyst else None,
            float_source=None,
            news_items=[],
            reasons=[] if force_catalyst else ["no_catalyst"],
        )

    @staticmethod
    def _session_open_metrics(
        bars: pd.DataFrame,
    ) -> tuple[float | None, float | None, int | None]:
        """Compute (open, change_pct, cumulative_volume_to_open) for the target day's 09:30 bar."""
        if bars.empty:
            return None, None, None
        dates = _index_dates(bars)
        target = max(set(dates))  # latest session in the frame == target date
        target_mask = [d == target for d in dates]
        target_rows = bars.loc[target_mask]
        if target_rows.empty:
            return None, None, None
        open_price = float(target_rows["open"].iloc[0])
        prior_mask = [d != target for d in dates]
        prior_rows = bars.loc[prior_mask]
        change_pct: float | None = None
        if not prior_rows.empty:
            prior_close = float(prior_rows["close"].iloc[-1])
            if prior_close > 0:
                change_pct = (open_price - prior_close) / prior_close * 100.0
        cumulative_volume = int(target_rows["volume"].iloc[0])
        return open_price, change_pct, cumulative_volume

    def _select_strategies(self, selection: str, force_catalyst: bool) -> list[Strategy]:
        """Resolve the strategy-selection string to concrete strategy instances.

        Applies the replay-side catalyst gate: when ``force_catalyst=False``,
        Gap-and-Go is suppressed because the live bot routes catalyst through
        the Phase 2 scanner, not the strategy itself. The corresponding
        ``signal.rejected`` event is emitted by ``_emit_catalyst_rejection``.
        """
        picked: list[Strategy] = []
        if selection in (STRATEGY_GAP_AND_GO, STRATEGY_BOTH) and force_catalyst:
            picked.append(self._gap_and_go)
        if selection in (STRATEGY_MOMENTUM, STRATEGY_BOTH):
            picked.append(self._momentum)
        return picked

    def _emit_catalyst_rejection(self, symbol: str, target_date: date, selection: str) -> None:
        """Log a single setup/missing_catalyst rejection for the skipped Gap-and-Go run."""
        bar_time = datetime.combine(target_date, time(9, 30), tzinfo=_NY)
        _log.info(
            REJECTION_EVENT,
            symbol=symbol,
            strategy=self._gap_and_go.name,
            bar_time=bar_time.isoformat(),
            stage="setup",
            reason="missing_catalyst",
            selection=selection,
        )


def _parse_rejections(captured: Sequence[Mapping[str, Any]]) -> list[RejectedCandidate]:
    """Turn raw structlog event dicts into RejectedCandidate records."""
    out: list[RejectedCandidate] = []
    for event in captured:
        if event.get("event") != REJECTION_EVENT:
            continue
        bar_time = _parse_bar_time(event.get("bar_time"))
        ctx = {k: v for k, v in event.items() if k not in _CANONICAL_REJECTION_KEYS}
        out.append(
            RejectedCandidate(
                symbol=str(event.get("symbol", "")),
                strategy=str(event.get("strategy", "")),
                bar_time=bar_time,
                stage=str(event.get("stage", "")),
                reason=str(event.get("reason", "")),
                context=ctx,
            )
        )
    return out


def _parse_bar_time(value: Any) -> datetime:
    """Rehydrate an ISO-8601 timestamp from a captured rejection event."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    # Fallback so downstream sort/render doesn't explode on a malformed record.
    return datetime(1970, 1, 1, tzinfo=_NY)


def _index_dates(bars: pd.DataFrame) -> list[date]:
    """Return the calendar date (NY-local) for every bar in ``bars``."""
    if bars.empty:
        return []
    index = bars.index
    # ``index`` is the NY-tz DatetimeIndex produced by ``_bars_to_frame``.
    return [ts.date() for ts in index]


def _bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    """Convert an ``ib_async`` BarDataList into a tz-aware (NY) DataFrame.

    Duplicates the logic from ``bot.brokerage.market_data`` to avoid importing a private
    helper; keeps the replay module free of cross-module private coupling.
    """
    if not bars:
        return pd.DataFrame(columns=_BAR_COLUMNS).astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "vwap": float,
            }
        )
    rows = [
        {
            "date": bar.date,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.average),
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows).set_index("date")
    idx = pd.to_datetime(frame.index, utc=True)
    frame.index = idx.tz_convert(_NY)
    frame.index.name = "date"
    return frame[_BAR_COLUMNS]
