"""Read-only analysis of 2026-05-08 momentum losses (AEHL, TRAW, AIIO).

Pulls bars from logs/session_2026-05-08.jsonl, computes the
impulse/consolidation/breakout/context metrics for each trade, and
matches each trade against the five hypothesis patterns.

Output is JSON-printable so the writer can lift values directly into
the markdown report at reports/momentum_failures_2026_05_08.md.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


@dataclass
class Bar:
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_red(self) -> bool:
        return self.close < self.open


SESSION_LOG = Path("logs/session_2026-05-08.jsonl")
TARGETS = ("AEHL", "TRAW", "AIIO")

# Trade entry context (from the journal + session.log).
TRADE_CTX = {
    "AEHL": {
        "entry_bar": "2026-05-08T09:32:00-04:00",
        "entry_price": 1.05,
        "stop": 0.93,
        "scale_out": 1.29,
        "shares": 199,
        "exit_price": 0.99,
        "exit_bar": "2026-05-08T09:34:00-04:00",  # exit at 09:34:11
        "vwap_at_entry": 0.9215,  # from signal.emitted
    },
    "TRAW": {
        "entry_bar": "2026-05-08T10:34:00-04:00",
        "entry_price": 2.33,
        "stop": 2.19,
        "scale_out": 2.62,
        "shares": 165,
        "exit_price": 2.26,
        "exit_bar": "2026-05-08T10:37:00-04:00",
        "vwap_at_entry": 2.2292,
    },
    "AIIO": {
        "entry_bar": "2026-05-08T11:23:00-04:00",
        "entry_price": 1.15,
        "stop": 1.02,
        "scale_out": 1.40,
        "shares": 192,
        "exit_price": 1.09,
        "exit_bar": "2026-05-08T11:27:00-04:00",
        "vwap_at_entry": 0.9545,
    },
}


def load_bars() -> tuple[dict[str, list[Bar]], dict[str, list[dict]]]:
    """Return (bars_by_sym, agg_by_sym).

    Primary source: ``market_data.bar_received`` (full OHLCV).
    Fallback metadata: ``bar_aggregator.minute_finalized`` (close + volume only)
    used when a breakout bar is missing from bar_received (AEHL halt/sparse case).
    """
    bars_out: dict[str, list[Bar]] = {s: [] for s in TARGETS}
    agg_out: dict[str, list[dict]] = {s: [] for s in TARGETS}
    with SESSION_LOG.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = row.get("event")
            sym = row.get("symbol")
            if sym not in TARGETS:
                continue
            if evt == "market_data.bar_received":
                bars_out[sym].append(
                    Bar(
                        ts=row["bar_time"],
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
            elif evt == "bar_aggregator.minute_finalized":
                agg_out[sym].append(row)
    for sym in bars_out:
        bars_out[sym].sort(key=lambda b: b.ts)
    return bars_out, agg_out


def bar_index(bars: list[Bar], ts: str) -> int:
    for i, b in enumerate(bars):
        if b.ts == ts:
            return i
    raise ValueError(f"bar at {ts} not found")


def safe_pct(num: float, den: float) -> float:
    return (num / den * 100.0) if den else 0.0


def safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def analyze_trade(
    symbol: str, bars: list[Bar], aggs: list[dict]
) -> dict[str, Any]:
    ctx = TRADE_CTX[symbol]
    data_warnings: list[str] = []
    # Try to find the breakout bar in bar_received; if missing, splice from
    # bar_aggregator.minute_finalized (AEHL halt-sparse case).
    try:
        entry_idx = bar_index(bars, ctx["entry_bar"])
    except ValueError:
        # Breakout bar absent from bar_received. Find the closest available
        # bar AFTER the entry (likely the next minute) and use that as the
        # post-entry context anchor; synthesize the breakout bar from the
        # minute_finalized event + the journal's actual fill price.
        data_warnings.append(
            f"breakout bar at {ctx['entry_bar']} not found in market_data.bar_received "
            "(halt or sparse trading); using bar_aggregator.minute_finalized for "
            "close+volume and journal fill price as proxy for OHLC"
        )
        # Find the matching aggregator event (timestamps are UTC there).
        # Convert "2026-05-08T09:32:00-04:00" -> "2026-05-08T13:32:00+00:00".
        from datetime import datetime as _dt

        et_dt = _dt.fromisoformat(ctx["entry_bar"])
        utc_iso = et_dt.astimezone(__import__("datetime").timezone.utc).isoformat()
        agg = next((a for a in aggs if a.get("minute_start") == utc_iso), None)
        synthesized_close = float(agg["close"]) if agg else ctx["entry_price"]
        synthesized_volume = float(agg["volume"]) if agg else 0.0
        contributing = agg.get("bars_contributing") if agg else None
        if contributing is not None and contributing < 12:
            data_warnings.append(
                f"breakout bar contributing={contributing}/12 5-sec aggregations "
                "(partial bar, signal fired before full minute close)"
            )
        # Insert a synthesized Bar at the right position
        synthesized_bar = Bar(
            ts=ctx["entry_bar"],
            open=synthesized_close,  # unknown; use close as proxy
            high=max(synthesized_close, ctx["entry_price"]),  # at minimum the LMT fill
            low=synthesized_close,  # unknown
            close=synthesized_close,
            volume=synthesized_volume,
        )
        # Insert in sorted order
        insert_at = next(
            (i for i, b in enumerate(bars) if b.ts > ctx["entry_bar"]), len(bars)
        )
        bars = bars[:insert_at] + [synthesized_bar] + bars[insert_at:]
        entry_idx = bar_index(bars, ctx["entry_bar"])
    try:
        exit_idx = bar_index(bars, ctx["exit_bar"])
    except ValueError:
        exit_idx = entry_idx + 3  # best-effort post window
        data_warnings.append(f"exit bar at {ctx['exit_bar']} not found; using entry_idx+3")

    # Bull-flag window: 10 bars ending at the breakout (entry) bar.
    # impulse = bars[-10:-7] (first 3 of window)
    # consolidation = bars[-7:-1] (middle 6)
    # breakout = bars[-1]
    if entry_idx < 9:
        # AEHL case: bar_received gap means the 10-bar window doesn't fill.
        # Use whatever pre-entry bars exist and document the limitation.
        data_warnings.append(
            f"only {entry_idx} bars available before entry in bar_received "
            "(pre-12.4 strategy needs >=10); analysis uses partial window"
        )

    window = bars[max(0, entry_idx - 9) : entry_idx + 1]  # 10 bars ending at entry
    if len(window) < 10:
        data_warnings.append(
            f"only {len(window)} bars in 10-bar window; impulse/consolidation "
            "split degraded"
        )
    # Impulse = first 3 of window; consolidation = middle bars; breakout = last
    impulse = window[: min(3, len(window) - 1)]
    consolidation = window[len(impulse) : -1] if len(window) > len(impulse) + 1 else []
    breakout = window[-1]
    assert breakout.ts == ctx["entry_bar"]

    # ----- Impulse window -----
    impulse_first_open = impulse[0].open
    impulse_high = max(b.high for b in impulse)
    impulse_last_close = impulse[-1].close
    impulse_pct_move = safe_pct(impulse_high - impulse_first_open, impulse_first_open)
    impulse_total_vol = sum(b.volume for b in impulse)
    impulse_avg_vol = impulse_total_vol / len(impulse)
    if impulse_last_close > impulse_first_open * 1.005:
        impulse_slope = "positive"
    elif impulse_last_close < impulse_first_open * 0.995:
        impulse_slope = "negative"
    else:
        impulse_slope = "flat"

    # ----- Consolidation window -----
    if not consolidation:
        cons_high = cons_low = cons_min_close = cons_max_close = 0.0
        cons_total_vol = cons_avg_vol = 0.0
        red_count = green_count = flat_count = 0
        closes_stdev = 0.0
        range_tightness = 0.0
        pullback_pct = 0.0
        vol_ratio_cons_to_impulse = 0.0
        distribution_bars = []
        lower_wick_bars = []
        data_warnings.append("no consolidation bars available; H2/H3 not assessable")
    else:
        cons_high = max(b.high for b in consolidation)
        cons_low = min(b.low for b in consolidation)
        cons_min_close = min(b.close for b in consolidation)
        cons_max_close = max(b.close for b in consolidation)
        range_tightness = safe_pct(cons_high - cons_low, impulse_high)
        pullback_pct = safe_pct(impulse_high - cons_min_close, impulse_high)
        red_count = sum(1 for b in consolidation if b.is_red)
        green_count = sum(1 for b in consolidation if b.close > b.open)
        flat_count = len(consolidation) - red_count - green_count
        closes_stdev = statistics.pstdev([b.close for b in consolidation])
        cons_total_vol = sum(b.volume for b in consolidation)
        cons_avg_vol = cons_total_vol / len(consolidation)
        vol_ratio_cons_to_impulse = safe_div(cons_avg_vol, impulse_avg_vol)
        distribution_bars = [
            b for b in consolidation
            if b.is_red and b.volume > 1.5 * cons_avg_vol
        ]
        lower_wick_bars = [
            b for b in consolidation
            if (min(b.open, b.close) - b.low) > 0.4 * (b.high - b.low) and (b.high - b.low) > 0
        ]

    # ----- Breakout candle -----
    bo_range = breakout.high - breakout.low
    bo_body = abs(breakout.close - breakout.open)
    bo_range_pct = safe_pct(bo_range, breakout.open)
    bo_body_pct = safe_pct(bo_body, breakout.open)
    bo_close_position = safe_div(breakout.close - breakout.low, bo_range)
    bo_vol_vs_cons_avg = safe_div(breakout.volume, cons_avg_vol)
    bo_vol_vs_impulse_avg = safe_div(breakout.volume, impulse_avg_vol)
    # 20-bar recent-rvol average (matches recent_rvol_window_bars=20)
    rvol_window = bars[max(0, entry_idx - 20) : entry_idx]
    rvol_window_size = len(rvol_window)
    rvol_avg_vol = (
        sum(b.volume for b in rvol_window) / rvol_window_size if rvol_window_size > 0 else 0.0
    )
    bo_vol_vs_rvol_window = safe_div(breakout.volume, rvol_avg_vol)
    bo_direction = "green" if breakout.close > breakout.open else "red"

    # ----- Context -----
    # Day open: first bar of the calendar day (>= 09:30 ET).
    rth_bars = [b for b in bars if b.ts.split("T")[1][:5] >= "09:30" and b.ts.startswith("2026-05-08")]
    day_open = rth_bars[0].open if rth_bars else None
    extension_from_open = (
        safe_pct(ctx["entry_price"] - day_open, day_open) if day_open else None
    )
    # Premarket high: any bar before 09:30 ET on the same day.
    pm_bars = [b for b in bars if b.ts.startswith("2026-05-08") and b.ts.split("T")[1][:5] < "09:30"]
    pm_high = max((b.high for b in pm_bars), default=None)
    # Day-prior-high attempts: count bars BEFORE the breakout that printed
    # within 0.5% of the entry price (potential prior HOD attempts).
    pre_breakout_rth = [b for b in rth_bars if b.ts < ctx["entry_bar"]]
    near_hod_attempts = sum(
        1 for b in pre_breakout_rth if b.high >= ctx["entry_price"] * 0.995
    )
    vwap_at_entry = ctx["vwap_at_entry"]
    pct_above_vwap = safe_pct(ctx["entry_price"] - vwap_at_entry, vwap_at_entry)

    # VWAP slope over last 5 bars (approximate via close trend).
    last5 = window[-5:]
    closes_last5 = [b.close for b in last5]
    if closes_last5[-1] > closes_last5[0] * 1.005:
        close_trend = "rising"
    elif closes_last5[-1] < closes_last5[0] * 0.995:
        close_trend = "falling"
    else:
        close_trend = "flat"

    # ----- Post-entry behaviour (3 bars after entry, or up to exit) -----
    post = bars[entry_idx + 1 : min(entry_idx + 4, exit_idx + 1)]
    post_max_high = max((b.high for b in post), default=breakout.high)
    new_high_after_entry = max(0.0, post_max_high - breakout.high)
    risk_per_share = ctx["entry_price"] - ctx["stop"]
    peak_r = (
        safe_div(post_max_high - ctx["entry_price"], risk_per_share) if risk_per_share > 0 else 0.0
    )
    first_red_idx = next((i for i, b in enumerate(post) if b.is_red), None)

    # ----- Hypothesis matches -----
    matches: dict[str, dict[str, Any]] = {}

    matches["H1_weak_or_absent_impulse"] = {
        "match": False,
        "reasons": [],
    }
    if impulse_pct_move < 1.5:
        matches["H1_weak_or_absent_impulse"]["match"] = True
        matches["H1_weak_or_absent_impulse"]["reasons"].append(
            f"impulse_pct_move={impulse_pct_move:.2f}% < 1.5%"
        )
    if impulse_slope != "positive":
        matches["H1_weak_or_absent_impulse"]["match"] = True
        matches["H1_weak_or_absent_impulse"]["reasons"].append(
            f"impulse_slope={impulse_slope}"
        )

    matches["H2_sloppy_consolidation"] = {"match": False, "reasons": []}
    if range_tightness > 4.0:
        matches["H2_sloppy_consolidation"]["match"] = True
        matches["H2_sloppy_consolidation"]["reasons"].append(
            f"range_tightness={range_tightness:.2f}% > 4.0%"
        )
    if red_count >= green_count:
        matches["H2_sloppy_consolidation"]["match"] = True
        matches["H2_sloppy_consolidation"]["reasons"].append(
            f"red_bars={red_count} >= green_bars={green_count} (one-sided pressure)"
        )
    if closes_stdev > 0 and impulse_high > 0:
        stdev_to_high_pct = closes_stdev / impulse_high * 100
        if stdev_to_high_pct > 1.5:
            matches["H2_sloppy_consolidation"]["match"] = True
            matches["H2_sloppy_consolidation"]["reasons"].append(
                f"closes_stdev={closes_stdev:.4f} ({stdev_to_high_pct:.2f}% of impulse high)"
            )
    if len(lower_wick_bars) >= 2:
        matches["H2_sloppy_consolidation"]["match"] = True
        matches["H2_sloppy_consolidation"]["reasons"].append(
            f"{len(lower_wick_bars)} bars with long lower wicks (rejection at lows)"
        )

    matches["H3_no_volume_contraction"] = {"match": False, "reasons": []}
    if vol_ratio_cons_to_impulse > 0.8:
        matches["H3_no_volume_contraction"]["match"] = True
        matches["H3_no_volume_contraction"]["reasons"].append(
            f"vol_ratio_cons/impulse={vol_ratio_cons_to_impulse:.2f} > 0.8 "
            "(volume did not contract)"
        )
    if vol_ratio_cons_to_impulse > 1.0:
        matches["H3_no_volume_contraction"]["reasons"].append(
            f"vol_ratio={vol_ratio_cons_to_impulse:.2f} > 1.0 (volume actually increased)"
        )
    if distribution_bars:
        matches["H3_no_volume_contraction"]["match"] = True
        matches["H3_no_volume_contraction"]["reasons"].append(
            f"{len(distribution_bars)} red consolidation bars with >1.5x avg volume "
            "(distribution signal)"
        )

    matches["H4_late_entry_extension"] = {"match": False, "reasons": []}
    if extension_from_open is not None and extension_from_open > 25.0:
        matches["H4_late_entry_extension"]["match"] = True
        matches["H4_late_entry_extension"]["reasons"].append(
            f"extension_from_open={extension_from_open:.2f}% > 25%"
        )
    if near_hod_attempts >= 3:
        matches["H4_late_entry_extension"]["match"] = True
        matches["H4_late_entry_extension"]["reasons"].append(
            f"{near_hod_attempts} prior HOD attempts (this is the {near_hod_attempts + 1}th)"
        )
    if pct_above_vwap > 5.0:
        matches["H4_late_entry_extension"]["match"] = True
        matches["H4_late_entry_extension"]["reasons"].append(
            f"entry {pct_above_vwap:.2f}% above VWAP > 5%"
        )

    matches["H5_weak_breakout_candle"] = {"match": False, "reasons": []}
    body_to_range_ratio = safe_div(bo_body, bo_range) if bo_range > 0 else 0.0
    if body_to_range_ratio < 0.5:
        matches["H5_weak_breakout_candle"]["match"] = True
        matches["H5_weak_breakout_candle"]["reasons"].append(
            f"body/range={body_to_range_ratio:.2f} < 0.5 (more wick than body)"
        )
    if bo_close_position < 0.6:
        matches["H5_weak_breakout_candle"]["match"] = True
        matches["H5_weak_breakout_candle"]["reasons"].append(
            f"close_position_in_range={bo_close_position:.2f} < 0.6 "
            "(closed below upper third)"
        )
    # Average recent range across the 10-bar window
    avg_window_range = statistics.mean([b.high - b.low for b in window[:-1]])
    if bo_range < 0.7 * avg_window_range:
        matches["H5_weak_breakout_candle"]["match"] = True
        matches["H5_weak_breakout_candle"]["reasons"].append(
            f"breakout range {bo_range:.4f} < 0.7x avg recent range {avg_window_range:.4f}"
        )
    if bo_vol_vs_cons_avg < 1.5:
        matches["H5_weak_breakout_candle"]["match"] = True
        matches["H5_weak_breakout_candle"]["reasons"].append(
            f"breakout volume {bo_vol_vs_cons_avg:.2f}x consolidation avg < 1.5x"
        )

    return {
        "symbol": symbol,
        "data_warnings": data_warnings,
        "entry_bar": ctx["entry_bar"],
        "entry_price": ctx["entry_price"],
        "exit_price": ctx["exit_price"],
        "stop": ctx["stop"],
        "scale_out": ctx["scale_out"],
        "shares": ctx["shares"],
        "impulse": {
            "bars": [
                f"{b.ts.split('T')[1][:5]} O={b.open} H={b.high} L={b.low} C={b.close} V={int(b.volume)}"
                for b in impulse
            ],
            "first_open": impulse_first_open,
            "high": impulse_high,
            "last_close": impulse_last_close,
            "pct_move": round(impulse_pct_move, 3),
            "slope": impulse_slope,
            "total_volume": int(impulse_total_vol),
            "avg_volume": int(impulse_avg_vol),
        },
        "consolidation": {
            "bars": [
                f"{b.ts.split('T')[1][:5]} O={b.open} H={b.high} L={b.low} C={b.close} V={int(b.volume)}"
                for b in consolidation
            ],
            "high": cons_high,
            "low": cons_low,
            "min_close": cons_min_close,
            "max_close": cons_max_close,
            "range_tightness_pct": round(range_tightness, 3),
            "pullback_pct": round(pullback_pct, 3),
            "red_bars": red_count,
            "green_bars": green_count,
            "flat_bars": flat_count,
            "closes_stdev": round(closes_stdev, 4),
            "total_volume": int(cons_total_vol),
            "avg_volume": int(cons_avg_vol),
            "vol_ratio_to_impulse": round(vol_ratio_cons_to_impulse, 3),
            "distribution_bar_count": len(distribution_bars),
            "lower_wick_bar_count": len(lower_wick_bars),
        },
        "breakout": {
            "open": breakout.open,
            "high": breakout.high,
            "low": breakout.low,
            "close": breakout.close,
            "volume": int(breakout.volume),
            "range_pct": round(bo_range_pct, 3),
            "body_pct": round(bo_body_pct, 3),
            "body_to_range_ratio": round(body_to_range_ratio, 3),
            "close_position_in_range": round(bo_close_position, 3),
            "vol_vs_cons_avg": round(bo_vol_vs_cons_avg, 3),
            "vol_vs_impulse_avg": round(bo_vol_vs_impulse_avg, 3),
            "vol_vs_rvol_window": round(bo_vol_vs_rvol_window, 3),
            "rvol_window_size": rvol_window_size,
            "direction": bo_direction,
        },
        "context": {
            "day_open": day_open,
            "extension_from_open_pct": round(extension_from_open, 3) if extension_from_open is not None else None,
            "premarket_high": pm_high,
            "near_hod_attempt_count": near_hod_attempts,
            "vwap_at_entry": vwap_at_entry,
            "pct_above_vwap": round(pct_above_vwap, 3),
            "close_trend_last5": close_trend,
        },
        "post_entry": {
            "bars": [
                f"{b.ts.split('T')[1][:5]} O={b.open} H={b.high} L={b.low} C={b.close} V={int(b.volume)}"
                for b in post
            ],
            "post_max_high": post_max_high,
            "new_high_after_entry": round(new_high_after_entry, 4),
            "peak_r_achieved": round(peak_r, 3),
            "first_red_bar_index": first_red_idx,
        },
        "hypothesis_matches": matches,
    }


def main() -> None:
    bars_by_sym, agg_by_sym = load_bars()
    results = []
    for sym in TARGETS:
        bars = bars_by_sym[sym]
        if not bars:
            results.append({"symbol": sym, "error": "no bars"})
            continue
        results.append(analyze_trade(sym, bars, agg_by_sym[sym]))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
