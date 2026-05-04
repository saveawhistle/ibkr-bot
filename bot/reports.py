"""Phase 4k — post-session analytics derived from the trade journal.

Read-only aggregators the CLI surfaces via ``ibkr-bot commissions`` (and
future ``ibkr-bot stats``). Pure functions over ``list[TradeRecord]`` so
tests can feed in fixtures without spinning up SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.persistence.journal import TradeRecord


@dataclass(frozen=True)
class CommissionSummary:
    """Aggregate commission view across a closed-trade window.

    ``commission_pct_of_gross`` is computed only when ``total_gross_pnl``
    is positive — a losing window has no "% of profit" to take a commission
    out of, and reporting a ratio against near-zero (or negative) gross is
    misleading. In that case the field is ``None`` and the CLI prints a dash.
    """

    trades_counted: int
    trades_with_commission_data: int
    total_gross_pnl: float
    total_commission: float
    total_entry_commission: float
    total_scale_commission: float
    total_exit_commission: float
    net_pnl: float
    avg_commission_per_trade: float
    commission_pct_of_gross: float | None
    scale_out_commission_share: float | None


def commission_summary(trades: list[TradeRecord]) -> CommissionSummary:
    """Roll up per-leg commissions + gross/net PnL across ``trades``.

    ``trades`` should already be filtered to the date window of interest
    (``recent_trades`` / ``trades_for_session`` handle that). Open trades
    (``pnl is None``) and trades whose STP-LMT parent never triggered
    (``entry_never_triggered``, ``pnl == 0`` with NULL commissions) are
    still counted — their zero contribution keeps the averages honest.
    """
    trades_counted = len(trades)
    entry_total = sum((t.entry_commission or 0.0) for t in trades)
    scale_total = sum((t.scale_commission or 0.0) for t in trades)
    exit_total = sum((t.exit_commission or 0.0) for t in trades)
    total_commission = entry_total + scale_total + exit_total

    total_gross = sum((t.pnl or 0.0) for t in trades)
    with_data = sum(
        1 for t in trades if (t.entry_commission or t.scale_commission or t.exit_commission)
    )
    avg_per_trade = total_commission / trades_counted if trades_counted else 0.0
    pct_of_gross = (total_commission / total_gross) if total_gross > 0.0 else None
    scale_share = (scale_total / total_commission) if total_commission > 0.0 else None
    net = total_gross - total_commission

    return CommissionSummary(
        trades_counted=trades_counted,
        trades_with_commission_data=with_data,
        total_gross_pnl=total_gross,
        total_commission=total_commission,
        total_entry_commission=entry_total,
        total_scale_commission=scale_total,
        total_exit_commission=exit_total,
        net_pnl=net,
        avg_commission_per_trade=avg_per_trade,
        commission_pct_of_gross=pct_of_gross,
        scale_out_commission_share=scale_share,
    )
