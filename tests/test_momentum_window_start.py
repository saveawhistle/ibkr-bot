"""Phase 12.6 tests -- configurable momentum window_start (sequencing with gap-and-go).

Three suites:

1. ``test_config_*``: pydantic schema validation of ``window_start``.
2. ``test_strategy_*``: MomentumStrategy honours its instance window_start.
3. ``test_orchestrator_default_strategies_wire_window_start``: build_default_strategies
   passes the configured value through.
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from bot.config import MomentumConfig, Settings
from bot.strategies.momentum import MomentumStrategy

# ---------- 1. Config schema ----------


def test_config_default_window_start_is_10_00() -> None:
    """Default sequence: gap-and-go owns 09:30-10:00, momentum picks up at 10:00."""
    assert MomentumConfig().window_start == "10:00"


def test_config_accepts_09_30_to_restore_concurrent_evaluation() -> None:
    """Operator can set window_start back to 09:30 to opt out of sequencing."""
    cfg = MomentumConfig(window_start="09:30")
    assert cfg.window_start == "09:30"


def test_config_rejects_start_before_market_open() -> None:
    """No premarket evaluation -- window_start must be >= 09:30 ET."""
    with pytest.raises(ValueError, match=r"window_start must be >= 09:30"):
        MomentumConfig(window_start="09:00")


def test_config_rejects_start_at_or_after_end() -> None:
    """A window with start >= end evaluates zero bars."""
    with pytest.raises(ValueError, match=r"window_start.*must be.*before window_end"):
        MomentumConfig(window_start="12:00", window_end="11:30")
    with pytest.raises(ValueError, match=r"window_start.*must be.*before window_end"):
        MomentumConfig(window_start="11:30", window_end="11:30")


def test_config_rejects_malformed_window_start() -> None:
    with pytest.raises(ValueError, match=r"window_start must be HH:MM"):
        MomentumConfig(window_start="not-a-time")
    with pytest.raises(ValueError, match=r"valid HH:MM"):
        MomentumConfig(window_start="25:00")


# ---------- 2. Strategy honours instance window_start ----------


def _build_minimal_breakout_frame(
    *,
    timestamps: list[str],
) -> pd.DataFrame:
    """Build a 10-bar bull-flag-into-HOD-break fixture at the supplied timestamps."""
    return pd.DataFrame(
        {
            "open": [10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.55],
            "high": [10.3, 10.5, 10.5, 10.4, 10.4, 10.35, 10.35, 10.32, 10.4, 10.6],
            "low": [10.0, 10.3, 10.3, 10.3, 10.25, 10.25, 10.3, 10.3, 10.3, 10.35],
            "close": [10.2, 10.45, 10.5, 10.4, 10.3, 10.3, 10.32, 10.32, 10.35, 10.6],
            "volume": [1000, 1100, 1200, 1100, 1100, 1100, 1100, 1100, 1100, 5000],
        },
        index=pd.to_datetime(timestamps),
    )


@pytest.mark.momentum_default_window_start
def test_strategy_default_window_start_silently_drops_pre_10_bars() -> None:
    """Default ``window_start=10:00`` means a 09:30 breakout silently no-ops.

    Opts out of the conftest's legacy 09:30 pin via the marker -- this
    test specifically verifies the production 10:00 default.
    """
    bars = _build_minimal_breakout_frame(
        timestamps=[f"2026-04-16 09:{30 + i:02d}" for i in range(10)],
    )
    strategy = MomentumStrategy()  # defaults: window_start=10:00, window_end=11:30
    assert strategy.evaluate("EARLY", bars) is None


@pytest.mark.momentum_default_window_start
def test_strategy_default_window_start_admits_post_10_bars() -> None:
    """A breakout stamped at 10:30 lands inside the default 10:00-11:30 window."""
    bars = _build_minimal_breakout_frame(
        timestamps=[f"2026-04-16 10:{30 + i:02d}" for i in range(10)],
    )
    strategy = MomentumStrategy()
    sig = strategy.evaluate("LATE", bars)
    assert sig is not None
    assert sig.symbol == "LATE"


def test_strategy_explicit_09_30_window_start_admits_opening_bars() -> None:
    """``window_start=09:30`` restores pre-12.6 behaviour for operator override."""
    bars = _build_minimal_breakout_frame(
        timestamps=[f"2026-04-16 09:{30 + i:02d}" for i in range(10)],
    )
    strategy = MomentumStrategy(window_start=time(9, 30))
    sig = strategy.evaluate("EARLY", bars)
    assert sig is not None


def test_strategy_window_start_after_window_end_is_an_invalid_runtime_state() -> None:
    """Defensive: a strategy with start >= end evaluates nothing.

    Config validators block this combination at the boundary, but if a
    test bypasses pydantic and constructs the strategy directly, the
    short-circuit just emits no signals (no exception). Equivalent to
    the legacy hardcoded-end-before-start case.
    """
    bars = _build_minimal_breakout_frame(
        timestamps=[f"2026-04-16 10:{30 + i:02d}" for i in range(10)],
    )
    strategy = MomentumStrategy(window_start=time(11, 0), window_end=time(10, 0))
    assert strategy.evaluate("BAD", bars) is None


# ---------- 3. build_default_strategies wires window_start ----------


def test_build_default_strategies_passes_window_start_to_momentum() -> None:
    """Orchestrator's default-strategy builder reads from MomentumConfig."""
    from bot.orchestrator import build_default_strategies

    base = Settings()
    settings = base.model_copy(
        update={
            "strategies": base.strategies.model_copy(
                update={
                    "momentum": base.strategies.momentum.model_copy(
                        update={"window_start": "11:00"}
                    ),
                }
            ),
        }
    )
    strategies = build_default_strategies(settings)
    momentum_instances = [s for s in strategies if s.name == "momentum"]
    assert len(momentum_instances) == 1
    assert momentum_instances[0].window_start == time(11, 0)
