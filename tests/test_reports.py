"""Tests for ``bot.reports.commission_summary`` — Phase 4k aggregate view."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from bot.persistence.journal import TradeRecord
from bot.reports import commission_summary


def _trade(
    *,
    pnl: float | None,
    entry: float | None = None,
    scale: float | None = None,
    exit_: float | None = None,
) -> TradeRecord:
    """Minimal TradeRecord-shaped stub — commission_summary only touches these fields."""
    return cast(
        "TradeRecord",
        cast(
            "Any",
            SimpleNamespace(
                pnl=pnl,
                entry_commission=entry,
                scale_commission=scale,
                exit_commission=exit_,
            ),
        ),
    )


def test_empty_trade_list_returns_zeros() -> None:
    """No trades → zero counters, ratios are None (no divisor)."""
    summary = commission_summary([])
    assert summary.trades_counted == 0
    assert summary.total_commission == 0.0
    assert summary.commission_pct_of_gross is None
    assert summary.scale_out_commission_share is None


def test_pct_of_gross_computed_on_positive_gross() -> None:
    """On a profitable window, ratio is total_commission / total_gross_pnl."""
    summary = commission_summary(
        [
            _trade(pnl=100.0, entry=1.0, exit_=1.0),
            _trade(pnl=50.0, entry=1.0, exit_=1.0),
        ]
    )
    assert summary.total_gross_pnl == pytest.approx(150.0)
    assert summary.total_commission == pytest.approx(4.0)
    assert summary.commission_pct_of_gross == pytest.approx(4.0 / 150.0)
    assert summary.net_pnl == pytest.approx(146.0)


def test_pct_of_gross_is_none_on_losing_window() -> None:
    """Negative gross → ratio is None (a % of a loss is misleading)."""
    summary = commission_summary([_trade(pnl=-30.0, entry=1.0, exit_=1.0)])
    assert summary.total_gross_pnl == pytest.approx(-30.0)
    assert summary.commission_pct_of_gross is None


def test_scale_share_tracks_scale_commission_fraction() -> None:
    """``scale_out_commission_share`` = scale_commission / total_commission."""
    summary = commission_summary([_trade(pnl=100.0, entry=1.0, scale=1.0, exit_=2.0)])
    assert summary.total_commission == pytest.approx(4.0)
    assert summary.scale_out_commission_share == pytest.approx(0.25)


def test_null_commissions_count_as_zero_but_trade_still_counted() -> None:
    """Legacy (pre-4k) rows with NULL commissions don't break the aggregate."""
    summary = commission_summary([_trade(pnl=50.0, entry=None, scale=None, exit_=None)])
    assert summary.trades_counted == 1
    assert summary.trades_with_commission_data == 0
    assert summary.total_commission == 0.0
    assert summary.net_pnl == pytest.approx(50.0)


def test_avg_commission_divides_by_all_trades_not_only_those_with_data() -> None:
    """Avg commission uses the full trade count — mirrors the real per-trade cost."""
    summary = commission_summary(
        [
            _trade(pnl=100.0, entry=1.0, exit_=1.0),
            _trade(pnl=50.0, entry=None, exit_=None),  # legacy / no data
        ]
    )
    assert summary.trades_counted == 2
    assert summary.trades_with_commission_data == 1
    # Total commission $2.00, across 2 trades = $1.00 average.
    assert summary.avg_commission_per_trade == pytest.approx(1.00)


def test_open_trades_are_excluded_by_caller_but_handled_safely() -> None:
    """commission_summary tolerates pnl=None — caller decides whether to filter."""
    summary = commission_summary([_trade(pnl=None, entry=1.0, exit_=1.0)])
    assert summary.total_gross_pnl == 0.0
    assert summary.total_commission == pytest.approx(2.0)
    # Negative net is the right answer: commissions were paid, no realized gross yet.
    assert summary.net_pnl == pytest.approx(-2.0)
