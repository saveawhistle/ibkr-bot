"""Closed-trade aggregate report — analytical bridge from layer 4 to layer 5.

Reads the manifest of closed trades + each trade's per-trade JSONL
detail file (produced by ``run_policy_comparison --all-trades``),
classifies trades by failure mode, computes per-policy aggregates,
and emits a Markdown report plus a JSONL detail file.

Usage:
    python scripts/run_aggregate_report.py

Output: ``reports/exit_advisor/aggregates/closed_trades_aggregate_{YYYY-MM-DD}.md``
+ ``.jsonl`` (date is the local date the script runs, so each
invocation produces a snapshot the operator can compare across runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from bot.exit_advisor.analysis.aggregation import (
    PolicyAggregateMetrics,
    TradeOutcome,
    aggregate_results,
    aggregate_subset_by_trades,
)
from bot.exit_advisor.analysis.failure_modes import (
    FailureMode,
    TradeClassification,
    classify_trade,
)
from bot.exit_advisor.replay.replay_source import (
    _iter_structured_events,
    load_trade_replay_data,
)
from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef, read_manifest

DEFAULT_MANIFEST = Path("reports/exit_advisor/closed_trades_manifest.jsonl")
DEFAULT_COMPARISONS_DIR = Path("reports/exit_advisor/comparisons/")
DEFAULT_AGGREGATES_DIR = Path("reports/exit_advisor/aggregates/")

log = logging.getLogger(__name__)


def _detail_path(symbol: str, trade_date: date, comparisons_dir: Path) -> Path:
    return comparisons_dir / f"{symbol}_{trade_date.isoformat()}_comparison.jsonl"


def _load_per_trade_outcomes(
    refs: list[ClosedTradeRef], comparisons_dir: Path
) -> tuple[list[TradeOutcome], list[ClosedTradeRef]]:
    """For each trade, read its comparison JSONL into TradeOutcome rows.

    Returns (outcomes, missing_refs). A missing detail file results in
    a WARNING and exclusion from aggregation; this is the documented
    graceful-degradation path.
    """
    outcomes: list[TradeOutcome] = []
    missing: list[ClosedTradeRef] = []
    for ref in refs:
        path = _detail_path(ref.symbol, ref.trade_date, comparisons_dir)
        if not path.exists():
            log.warning(
                "no per-trade detail file for %s %s — excluded from aggregate",
                ref.symbol,
                ref.trade_date,
            )
            missing.append(ref)
            continue
        # Parse all rows; locate Actual + Oracle for this trade so each
        # outcome row carries the comparison anchors.
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        # Use gates_enabled=True rows to compute the per-trade Actual/Oracle anchors
        # (we still emit one TradeOutcome per gate-mode below).
        actual_pnl_by_mode: dict[bool, float] = {}
        oracle_pnl_by_mode: dict[bool, float] = {}
        for row in rows:
            if row["policy_name"] == "ActualPolicy":
                actual_pnl_by_mode[row["gates_enabled"]] = row["effective_pnl"]
            elif row["policy_name"] == "OracleExitPolicy":
                oracle_pnl_by_mode[row["gates_enabled"]] = row["effective_pnl"]
        for row in rows:
            outcomes.append(
                TradeOutcome(
                    trade_ref=ref,
                    policy_name=row["policy_name"],
                    policy_params=row["policy_params"],
                    gates_enabled=row["gates_enabled"],
                    final_pnl=row["effective_pnl"],
                    fired=row["decisions_proposed"] > 0,
                    decisions_proposed=row["decisions_proposed"],
                    decisions_accepted=row["decisions_accepted"],
                    decisions_rejected=row["decisions_rejected"],
                    open_at_end_of_replay=row["final_position_size"] > 0,
                    actual_pnl=actual_pnl_by_mode.get(row["gates_enabled"], 0.0),
                    oracle_pnl=oracle_pnl_by_mode.get(row["gates_enabled"], 0.0),
                )
            )
    return outcomes, missing


def _classify_all(
    refs: list[ClosedTradeRef], outcomes_by_trade: dict[str, list[TradeOutcome]]
) -> list[TradeClassification]:
    """Classify each trade. Pulls the inputs the classifier needs from
    the trade's replay_data (real cache) and the recorded session log."""
    classifications: list[TradeClassification] = []
    for ref in refs:
        if ref.trade_id not in outcomes_by_trade:
            continue
        try:
            rd = load_trade_replay_data(ref.symbol, ref.trade_date)
        except Exception as exc:  # noqa: BLE001 - skip with reason
            log.warning("could not load replay data for %s %s: %s", ref.symbol, ref.trade_date, exc)
            continue

        # Pull anchor values from recorded events.
        if rd.fill_event:
            entry_price = float(rd.fill_event["fill_price"])
            position_size = int(rd.fill_event["filled_shares"])
        else:
            entry_price = float(rd.bracket_event["entry_price"])
            position_size = int(rd.bracket_event["shares"])
        initial_stop = float(rd.bracket_event["stop_price"])
        actual_pnl = float(rd.recorded_pnl)
        actual_exit_price = float(rd.recorded_exit_price)

        # Peak price during trade — max(bar.high) over the trade window.
        peak_price = entry_price
        for bar in rd.bars:
            if bar.high > peak_price:
                peak_price = bar.high

        # Scale-out detection: a position.scaled_out event (or
        # trade_manager.scale_out) for this symbol between bracket and
        # close means the runner half was active.
        scale_out_was_hit = _detect_scale_out(ref)

        # Oracle anchor for this trade: pull from the gates-on outcome.
        oracle_pnl = 0.0
        for o in outcomes_by_trade[ref.trade_id]:
            if o.policy_name == "OracleExitPolicy" and o.gates_enabled:
                oracle_pnl = o.final_pnl
                break

        duration_min = (ref.exit_timestamp - ref.entry_timestamp).total_seconds() / 60.0
        bar_count = len(rd.bars)

        classifications.append(
            classify_trade(
                trade_ref=ref,
                actual_pnl=actual_pnl,
                oracle_pnl=oracle_pnl,
                initial_stop=initial_stop,
                entry_price=entry_price,
                peak_price_during_trade=peak_price,
                actual_exit_price=actual_exit_price,
                trade_duration_minutes=duration_min,
                bar_count_post_protection=bar_count,
                scale_out_was_hit=scale_out_was_hit,
                position_size=position_size,
            )
        )
    return classifications


def _detect_scale_out(ref: ClosedTradeRef) -> bool:
    for evt in _iter_structured_events(ref.session_log_path):
        if evt.get("symbol") != ref.symbol:
            continue
        ts_raw = evt.get("timestamp")
        if not ts_raw:
            continue
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts < ref.entry_timestamp or ts > ref.exit_timestamp:
            continue
        if evt.get("event") in (
            "position.scaled_out",
            "trade_manager.scale_out",
            "executor.scale_out_lmt_filled",
        ):
            return True
    return False


def _format_pnl(value: float) -> str:
    if value >= 0:
        return f"+${value:.2f}"
    return f"-${abs(value):.2f}"


def _policy_label(name: str, params: dict[str, Any]) -> str:
    if not params:
        return name
    parts = [f"{k}={v}" for k, v in params.items()]
    return f"{name}({', '.join(parts)})"


def _build_report(
    refs: list[ClosedTradeRef],
    missing: list[ClosedTradeRef],
    outcomes: list[TradeOutcome],
    classifications: list[TradeClassification],
    aggregates: dict[str, PolicyAggregateMetrics],
    iso_date: str,
) -> str:
    out: list[str] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    n_total = len(refs)
    n_replayed = len(refs) - len(missing)

    out.append(f"# Closed-Trade Aggregate Comparison Report — {iso_date}\n\n")
    out.append(f"Generated: {now}\n\n")
    out.append(f"- Trades discovered: {n_total}\n")
    out.append(f"- Trades successfully replayed: {n_replayed}\n")
    if missing:
        names = ", ".join(f"{r.symbol} {r.trade_date}" for r in missing)
        out.append(f"- Trades excluded ({len(missing)}): missing detail files — {names}\n")
    out.append("\n")

    # Dataset summary.
    classified_refs = [c.trade_ref for c in classifications]
    if classified_refs:
        earliest = min(r.trade_date for r in classified_refs)
        latest = max(r.trade_date for r in classified_refs)
        symbols = sorted({r.symbol for r in classified_refs})
        # Anchors come from gates-on Actual + Oracle rows for each trade.
        actual_total = _sum_anchor(outcomes, "ActualPolicy")
        oracle_total = _sum_anchor(outcomes, "OracleExitPolicy")
        out.append("## Dataset summary\n\n")
        out.append(f"- Date range: {earliest} to {latest}\n")
        out.append(f"- Symbols: {', '.join(symbols)}\n")
        out.append(f"- Total ActualPolicy P&L: {_format_pnl(actual_total)}\n")
        out.append(f"- Total OracleExitPolicy P&L (theoretical max): {_format_pnl(oracle_total)}\n")
        out.append(f"- Theoretical room for improvement: {_format_pnl(oracle_total - actual_total)}\n\n")

    # Failure mode distribution.
    out.append("## Failure mode distribution\n\n")
    out.append("| Mode | Count | % | Total Actual | Total Oracle | Total room |\n")
    out.append("|------|------:|--:|-------------:|-------------:|-----------:|\n")
    by_mode = _group_by_mode(classifications, outcomes)
    for mode in FailureMode:
        entries = by_mode.get(mode, [])
        n = len(entries)
        pct = (n / len(classifications) * 100) if classifications else 0.0
        actual = sum(e["actual"] for e in entries)
        oracle = sum(e["oracle"] for e in entries)
        out.append(
            f"| {mode.value} | {n} | {pct:.0f}% | {_format_pnl(actual)} | "
            f"{_format_pnl(oracle)} | {_format_pnl(oracle - actual)} |\n"
        )
    out.append("\n")

    # Per-policy aggregate (gates active).
    out.append("## Per-policy aggregate (gates active)\n\n")
    out.append(_build_aggregate_table(aggregates, gates_enabled=True))
    out.append("\n")

    out.append("## Per-policy aggregate (gates disabled)\n\n")
    out.append(_build_aggregate_table(aggregates, gates_enabled=False))
    out.append("\n")

    # Per-mode rankings.
    out.append("## Mode-specific policy ranking\n\n")
    for mode in FailureMode:
        entries = by_mode.get(mode, [])
        out.append(f"### {mode.value} subset (N = {len(entries)})\n\n")
        if len(entries) < 3:
            out.append("Insufficient data for statistical comparison.\n\n")
            continue
        trade_ids = {e["trade_id"] for e in entries}
        sub_aggs = aggregate_subset_by_trades(outcomes, trade_ids)
        rankable = [
            a for a in sub_aggs.values()
            if a.gates_enabled
            and a.policy_name not in ("ActualPolicy", "OracleExitPolicy")
        ]
        if not rankable:
            out.append("No rankable mechanical policies fired on this subset.\n\n")
            continue
        ranked = sorted(rankable, key=lambda a: a.delta_vs_actual, reverse=True)
        best = ranked[0]
        worst = ranked[-1]
        out.append(
            f"- Best: `{_policy_label(best.policy_name, best.policy_params)}` — "
            f"{_format_pnl(best.delta_vs_actual)} vs Actual\n"
        )
        out.append(
            f"- Worst: `{_policy_label(worst.policy_name, worst.policy_params)}` — "
            f"{_format_pnl(worst.delta_vs_actual)} vs Actual\n\n"
        )

    # Gate impact.
    out.append("## Gate impact summary\n\n")
    gate_rejections = _count_gate_rejections(outcomes)
    if gate_rejections:
        out.append("| Gate | Total rejections |\n|------|----------------:|\n")
        for gate, count in sorted(gate_rejections.items(), key=lambda kv: -kv[1]):
            out.append(f"| `{gate}` | {count} |\n")
        out.append("\n")
    else:
        out.append("No gate rejections across the dataset.\n\n")

    # Notable findings + layer-5 recommendations.
    out.append("## Notable findings\n\n")
    out.extend(_notable_findings(classifications, by_mode, aggregates, outcomes))
    out.append("\n")

    out.append("## Recommendations for layer 5\n\n")
    out.extend(_layer_5_recommendations(classifications, by_mode, aggregates, outcomes))
    out.append("\n")

    return "".join(out)


def _sum_anchor(outcomes: list[TradeOutcome], anchor_policy: str) -> float:
    """Sum the gates-on anchor P&L for a given policy across one row
    per trade (avoids double-counting if a policy appears under
    multiple gate modes)."""
    seen: set[str] = set()
    total = 0.0
    for o in outcomes:
        if o.policy_name != anchor_policy or not o.gates_enabled:
            continue
        if o.trade_ref.trade_id in seen:
            continue
        seen.add(o.trade_ref.trade_id)
        total += o.final_pnl
    return total


def _group_by_mode(
    classifications: list[TradeClassification], outcomes: list[TradeOutcome]
) -> dict[FailureMode, list[dict[str, Any]]]:
    """Group classifications + their per-trade Actual/Oracle anchors by mode."""
    by_mode: dict[FailureMode, list[dict[str, Any]]] = defaultdict(list)
    actuals: dict[str, float] = {}
    oracles: dict[str, float] = {}
    for o in outcomes:
        if not o.gates_enabled:
            continue
        if o.policy_name == "ActualPolicy":
            actuals[o.trade_ref.trade_id] = o.final_pnl
        elif o.policy_name == "OracleExitPolicy":
            oracles[o.trade_ref.trade_id] = o.final_pnl
    for c in classifications:
        tid = c.trade_ref.trade_id
        by_mode[c.mode].append(
            {
                "trade_id": tid,
                "trade_ref": c.trade_ref,
                "actual": actuals.get(tid, 0.0),
                "oracle": oracles.get(tid, 0.0),
                "reasoning": c.reasoning,
            }
        )
    return by_mode


def _build_aggregate_table(
    aggregates: dict[str, PolicyAggregateMetrics], gates_enabled: bool
) -> str:
    rows = [
        a for a in aggregates.values()
        if a.gates_enabled is gates_enabled
        and a.policy_name not in ("ActualPolicy", "OracleExitPolicy")
    ]
    rows.sort(key=lambda a: a.delta_vs_actual, reverse=True)
    out = [
        "| Policy(params) | Trades fired | Mean P&L when fired | Total P&L | "
        "Δ vs Actual | Δ vs Oracle |\n",
        "|----------------|-------------:|--------------------:|----------:|"
        "------------:|------------:|\n",
    ]
    for a in rows:
        out.append(
            f"| {_policy_label(a.policy_name, a.policy_params)} | "
            f"{a.trades_fired}/{a.trades_total} | "
            f"{_format_pnl(a.mean_pnl_when_fired) if a.trades_fired else '—'} | "
            f"{_format_pnl(a.total_pnl)} | "
            f"{_format_pnl(a.delta_vs_actual)} | "
            f"{_format_pnl(a.delta_vs_oracle)} |\n"
        )
    return "".join(out)


def _count_gate_rejections(outcomes: list[TradeOutcome]) -> dict[str, int]:
    """The TradeOutcome doesn't carry gate-name detail; we summarize at
    the dataset level by aggregating ``decisions_rejected`` totals
    grouped by policy. A more detailed gate-by-gate breakdown would
    require re-reading the per-trade JSONL — kept simple here."""
    rejections_by_policy: Counter[str] = Counter()
    for o in outcomes:
        if not o.gates_enabled or o.decisions_rejected == 0:
            continue
        rejections_by_policy[o.policy_name] += o.decisions_rejected
    return dict(rejections_by_policy)


def _notable_findings(
    classifications: list[TradeClassification],
    by_mode: dict[FailureMode, list[dict[str, Any]]],
    aggregates: dict[str, PolicyAggregateMetrics],
    outcomes: list[TradeOutcome],
) -> list[str]:
    """Auto-generated findings — favors plain statements over guesses
    so the operator can spot what's surprising at a glance."""
    out: list[str] = []
    n = len(classifications)
    if n == 0:
        out.append("- No trades classified; aggregate report has no findings.\n")
        return out

    mode_counts = Counter(c.mode for c in classifications)
    dominant_mode, dominant_n = mode_counts.most_common(1)[0]
    pct = dominant_n / n * 100
    out.append(
        f"- **{dominant_mode.value}** is the dominant mode "
        f"({dominant_n}/{n} = {pct:.0f}% of dataset).\n"
    )

    # Total room: oracle - actual.
    actual_total = _sum_anchor(outcomes, "ActualPolicy")
    oracle_total = _sum_anchor(outcomes, "OracleExitPolicy")
    room = oracle_total - actual_total
    out.append(
        f"- Theoretical maximum improvement (Oracle - Actual) across the "
        f"dataset: **{_format_pnl(room)}**.\n"
    )

    # Best mechanical with gates on.
    rankable = [
        a for a in aggregates.values()
        if a.gates_enabled
        and a.policy_name not in ("ActualPolicy", "OracleExitPolicy")
    ]
    if rankable:
        best = max(rankable, key=lambda a: a.delta_vs_actual)
        out.append(
            f"- Best mechanical with gates active: "
            f"`{_policy_label(best.policy_name, best.policy_params)}` — "
            f"total Δ vs Actual {_format_pnl(best.delta_vs_actual)} "
            f"(fired on {best.trades_fired}/{best.trades_total} trades).\n"
        )
        # Surface "no policy improves on Actual" if true.
        positive = [a for a in rankable if a.delta_vs_actual > 0.005]
        if not positive:
            out.append(
                "- **No mechanical policy improved on ActualPolicy on net.** "
                "All sweep configurations either matched Actual (when they didn't fire) "
                "or under-performed (when they did). The dataset is dominated by "
                "trades where mechanical exits don't engage.\n"
            )

    # Mode-specific commentary.
    if mode_counts.get(FailureMode.DEGENERATE, 0) >= n * 0.3:
        out.append(
            f"- **{mode_counts[FailureMode.DEGENERATE]} trades classified DEGENERATE** "
            "(sub-5-minute or sub-5-bar). These tell us little about exit policy "
            "quality — they reflect entry-side timing or stop-out luck.\n"
        )
    if mode_counts.get(FailureMode.MODE_2_RUNNER_EXHAUSTION, 0) == 0:
        out.append(
            "- **No MODE_2_RUNNER_EXHAUSTION trades in the dataset.** Either "
            "the bot's trail is already managing runners well, or no trade "
            "reached scale-out with material runner gain — likely the latter "
            "given the dataset size.\n"
        )

    return out


def _layer_5_recommendations(
    classifications: list[TradeClassification],
    by_mode: dict[FailureMode, list[dict[str, Any]]],
    aggregates: dict[str, PolicyAggregateMetrics],
    outcomes: list[TradeOutcome],
) -> list[str]:
    out: list[str] = []
    n = len(classifications)
    if n == 0:
        out.append("- Insufficient data for layer-5 recommendations.\n")
        return out

    mode1_n = len(by_mode.get(FailureMode.MODE_1_FLAGGING_BREAKOUT, []))
    mode2_n = len(by_mode.get(FailureMode.MODE_2_RUNNER_EXHAUSTION, []))
    degen_n = len(by_mode.get(FailureMode.DEGENERATE, []))

    if mode1_n >= 3:
        out.append(
            f"- MODE_1 ({mode1_n} trades): mechanical FixedRTakeProfit / StallExit "
            "policies have already been characterized; the agent's value here is "
            "incremental discrimination (when to take 1R vs hold for more).\n"
        )
    elif mode1_n > 0:
        out.append(
            f"- MODE_1 ({mode1_n} trades): too few for statistical ranking. "
            "Layer 5 should defer MODE_1-specific design until more data lands.\n"
        )

    if mode2_n >= 3:
        out.append(
            f"- MODE_2 ({mode2_n} trades): mechanical trails struggled — agent's "
            "value on runner management is likely the highest-leverage application.\n"
        )
    elif mode2_n > 0:
        out.append(
            f"- MODE_2 ({mode2_n} trades): too few for statistical ranking.\n"
        )
    else:
        out.append(
            "- MODE_2: zero trades in this dataset. Either the bot rarely reaches "
            "scale-out, or runner gains are quickly captured. Layer 5's design "
            "shouldn't prioritize MODE_2 until it shows up.\n"
        )

    if degen_n >= n * 0.3:
        out.append(
            f"- DEGENERATE ({degen_n}/{n}): a substantial fraction of trades are "
            "too short for exit-side policy work. Improvement here likely lives "
            "upstream (entry timing, pre-protection sequence) rather than in any "
            "exit advisor — mechanical OR agent-driven.\n"
        )

    return out


def write_classification_jsonl(
    classifications: list[TradeClassification], out_path: Path
) -> None:
    """One JSON object per classification, with the trade ref + bucket + reasoning."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in classifications:
            obj = {
                "symbol": c.trade_ref.symbol,
                "trade_date": c.trade_ref.trade_date.isoformat(),
                "trade_id": c.trade_ref.trade_id,
                "mode": c.mode.value,
                "reasoning": c.reasoning,
            }
            fh.write(json.dumps(obj) + "\n")


def run_aggregate_report(
    manifest_path: Path = DEFAULT_MANIFEST,
    comparisons_dir: Path = DEFAULT_COMPARISONS_DIR,
    output_dir: Path = DEFAULT_AGGREGATES_DIR,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifest not found: {manifest_path}. "
            "Run: python scripts/discover_closed_trades.py"
        )
    refs = read_manifest(manifest_path)
    if not refs:
        raise RuntimeError("Manifest is empty — no trades to aggregate.")

    outcomes, missing = _load_per_trade_outcomes(refs, comparisons_dir)
    outcomes_by_trade: dict[str, list[TradeOutcome]] = defaultdict(list)
    for o in outcomes:
        outcomes_by_trade[o.trade_ref.trade_id].append(o)

    classifications = _classify_all(refs, outcomes_by_trade)
    aggregates = aggregate_results(outcomes)

    iso_date = date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"closed_trades_aggregate_{iso_date}.md"
    jsonl_path = output_dir / f"closed_trades_aggregate_{iso_date}.jsonl"

    md_path.write_text(
        _build_report(refs, missing, outcomes, classifications, aggregates, iso_date),
        encoding="utf-8",
    )
    # JSONL detail: classifications + per-policy aggregates as separate lines
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for c in classifications:
            fh.write(
                json.dumps(
                    {
                        "kind": "classification",
                        "symbol": c.trade_ref.symbol,
                        "trade_date": c.trade_ref.trade_date.isoformat(),
                        "trade_id": c.trade_ref.trade_id,
                        "mode": c.mode.value,
                        "reasoning": c.reasoning,
                    }
                )
                + "\n"
            )
        for key, agg in aggregates.items():
            agg_dict = asdict(agg)
            agg_dict["kind"] = "aggregate"
            agg_dict["group_key"] = key
            fh.write(json.dumps(agg_dict, default=str) + "\n")

    return md_path, jsonl_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--comparisons-dir", default=str(DEFAULT_COMPARISONS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_AGGREGATES_DIR))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        md_path, jsonl_path = run_aggregate_report(
            manifest_path=Path(args.manifest),
            comparisons_dir=Path(args.comparisons_dir),
            output_dir=Path(args.output_dir),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {md_path}")
    print(f"wrote {jsonl_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
