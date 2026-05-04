"""Tests for ``bot suggest-caps`` — advisory analytics over the journal.

The command itself is driven by Typer's ``CliRunner``; the heuristic
helpers live in ``bot.cli`` as private (``_``-prefixed) functions so
tests import them directly. The command must never write config —
verified by asserting ``config.yaml`` stays untouched across a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bot.cli import (
    _compute_suggest_caps_stats,
    _suggested_caps,
    app,
)


@dataclass
class _FakeTrade:
    """Minimal stand-in for ``journal.TradeRecord`` — only the fields the helpers read."""

    symbol: str
    pnl: float | None
    closed_at: datetime | None


def _trade(pnl: float, day: int = 1, *, symbol: str = "AAA") -> _FakeTrade:
    """Build a closed trade with the given PnL on 2026-04-``day`` at 10:00 ET."""
    return _FakeTrade(
        symbol=symbol,
        pnl=pnl,
        closed_at=datetime(2026, 4, day, 14, 0, tzinfo=UTC),  # 10:00 ET
    )


def test_compute_stats_empty_trades_returns_zero() -> None:
    """Filtering already happened in _suggest_caps; the helper handles a non-empty
    list but a degenerate list (e.g. only still-open rows) yields zeros."""
    stats = _compute_suggest_caps_stats([], timezone="America/New_York")
    assert stats.total_trades == 0
    assert stats.avg_win_usd == 0.0
    assert stats.avg_loss_usd == 0.0
    assert stats.sessions == 0


def test_compute_stats_separates_winners_and_losers() -> None:
    """Winners → positive PnL; losers → negative. Zero PnL rows contribute to neither."""
    trades = [_trade(100.0, day=1), _trade(-50.0, day=2), _trade(75.0, day=3)]
    stats = _compute_suggest_caps_stats(trades, timezone="America/New_York")
    assert stats.winners == 2
    assert stats.losers == 1
    assert stats.avg_win_usd == pytest.approx(87.5)
    assert stats.avg_loss_usd == pytest.approx(-50.0)


def test_compute_stats_aggregates_by_session_day() -> None:
    """Multiple trades same day collapse to one session entry."""
    trades = [
        _trade(50.0, day=1),
        _trade(-10.0, day=1),
        _trade(-100.0, day=2),
    ]
    stats = _compute_suggest_caps_stats(trades, timezone="America/New_York")
    assert stats.sessions == 2
    assert stats.worst_day_usd == pytest.approx(-100.0)
    assert stats.best_day_usd == pytest.approx(40.0)


def test_suggested_caps_rounds_to_25_and_floors() -> None:
    """Suggestions round up to the nearest $25 with floor guards.

    20 losers at -$40 avg → per-trade = $40 * 1.25 = $50, rounded up to $50.
    Worst day -$100 → daily = max(100*1.2=120, 3*50=150) → $150.
    Best day +$80 → goal = max(80, 2*150=300) → $300.
    """
    trades = [_trade(-40.0, day=i + 1) for i in range(3)]
    trades.append(_trade(80.0, day=10))
    stats = _compute_suggest_caps_stats(trades, timezone="America/New_York")
    suggestions = _suggested_caps(stats)
    assert suggestions.max_loss_per_trade_usd == pytest.approx(50.0)
    assert suggestions.max_daily_loss_usd == pytest.approx(150.0)
    assert suggestions.daily_profit_goal_usd == pytest.approx(300.0)


def test_suggested_caps_trades_per_day_clamps_to_sensible_range() -> None:
    """Thin data shouldn't suggest 1-trade days; rich data shouldn't suggest 50."""
    # One trade only → clamp to 3 (our minimum).
    stats_thin = _compute_suggest_caps_stats([_trade(10.0, day=1)], timezone="UTC")
    thin = _suggested_caps(stats_thin)
    assert thin.max_trades_per_day >= 3
    # Many trades/day → clamp at 10.
    stats_rich = _compute_suggest_caps_stats(
        [_trade(1.0, day=1) for _ in range(100)], timezone="UTC"
    )
    rich = _suggested_caps(stats_rich)
    assert rich.max_trades_per_day <= 10


def test_suggest_caps_cli_never_writes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: invoking the CLI must not mutate ``config.yaml``.

    The advisory nature of ``suggest-caps`` is a critical guardrail. A
    future refactor that accidentally wires in a ``yaml.dump`` call
    would silently override the operator's chosen caps — this test
    catches that class of regression.
    """
    from typer.testing import CliRunner

    # Give the journal a fresh path so the test doesn't depend on real data.
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["suggest-caps", "--lookback-days", "30"])
    # Empty journal → friendly "nothing to suggest" message + clean exit.
    assert result.exit_code == 0
    assert "nothing to suggest" in result.stdout.lower()
    # Assert no config file was written to the cwd.
    assert not (tmp_path / "config.yaml").exists()
