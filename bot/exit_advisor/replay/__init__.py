"""Offline trade-replay harness + historical-bar cache loaders.

Used by the validation pipeline + the cross-trade comparison runners
to re-feed closed trades through the gate/policy stack and emit
forensic comparison reports.
"""

from bot.exit_advisor.replay.bar_history import BarHistory
from bot.exit_advisor.replay.cache_loader import CacheCorruptError, HistoricalBarCache
from bot.exit_advisor.replay.harness import ReplayResult, TradeReplayHarness
from bot.exit_advisor.replay.replay_source import (
    DEFAULT_CACHE_DIR,
    Bar,
    TradeReplayData,
    load_prior_n_day_volume_curve,
    load_trade_replay_data,
    session_log_path,
)
from bot.exit_advisor.replay.trade_discovery import (
    ClosedTradeRef,
    discover_closed_trades,
    read_manifest,
    write_manifest,
)

__all__ = [
    # bar_history
    "BarHistory",
    # cache_loader
    "CacheCorruptError",
    "HistoricalBarCache",
    # harness
    "ReplayResult",
    "TradeReplayHarness",
    # replay_source
    "Bar",
    "DEFAULT_CACHE_DIR",
    "TradeReplayData",
    "load_prior_n_day_volume_curve",
    "load_trade_replay_data",
    "session_log_path",
    # trade_discovery
    "ClosedTradeRef",
    "discover_closed_trades",
    "read_manifest",
    "write_manifest",
]
