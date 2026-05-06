"""Mechanical-policy comparison runner.

Takes a (symbol, date) pair and runs every configured policy against
the trade with parameter sweeps, both with gates active and gates
disabled. Produces a Markdown report (primary analytical artifact)
plus a JSONL detail file (one row per policy run).

Usage:
    python scripts/run_policy_comparison.py \\
        --symbol ZENA --date 2026-04-30

Real cache must be populated first:
    python scripts/fetch_historical_bars.py \\
        --symbol {symbol} --date {date}

The runner refuses to proceed without a populated cache. Synthetic
fixtures are for unit tests; the comparison report is meant to
calibrate against real data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from bot.config import ExitGatesConfig, Settings
from bot.exit_advisor.decision.policy import (
    ActualPolicy,
    ExitPolicy,
    FixedRTakeProfit,
    MechanicalTrailPolicy,
    OracleExitPolicy,
    StallExitPolicy,
)
from bot.exit_advisor.replay.cache_loader import HistoricalBarCache
from bot.exit_advisor.replay.harness import ReplayResult, TradeReplayHarness
from bot.exit_advisor.replay.replay_source import (
    DEFAULT_CACHE_DIR,
    TradeReplayData,
    load_trade_replay_data,
)

DEFAULT_PARAMETER_SWEEPS: dict[str, list[dict[str, Any]]] = {
    "MechanicalTrailPolicy": [
        {"trail_abs": 0.05},
        {"trail_abs": 0.10},
        {"trail_abs": 0.15},
        {"trail_abs": 0.20},
        {"trail_pct": 0.02},
        {"trail_pct": 0.04},
        {"trail_pct": 0.06},
    ],
    "FixedRTakeProfit": [
        {"target_r": 0.5},
        {"target_r": 1.0},
        {"target_r": 1.5},
        {"target_r": 2.0},
        {"target_r": 3.0},
    ],
    "StallExitPolicy": [
        {"target_r": 1.0, "max_minutes": 5},
        {"target_r": 1.0, "max_minutes": 10},
        {"target_r": 1.5, "max_minutes": 5},
        {"target_r": 1.5, "max_minutes": 10},
        {"target_r": 2.0, "max_minutes": 10},
    ],
}


POLICY_FACTORIES: dict[str, Any] = {
    "MechanicalTrailPolicy": MechanicalTrailPolicy,
    "FixedRTakeProfit": FixedRTakeProfit,
    "StallExitPolicy": StallExitPolicy,
}


@dataclass
class PolicyRunResult:
    """One row in the comparison output. ``final_pnl`` is the harness's
    realized P&L; ``mark_to_market_pnl`` is what the operator would have
    seen if the position had been closed at the recorded exit price
    (used when a policy never produced an exit). ``effective_pnl`` is
    the comparison-friendly value: realized when an exit fired, MTM
    otherwise — clearly labeled in the report so "no exit" doesn't get
    silently treated as $0 P&L.
    """

    policy_name: str
    policy_params: dict[str, Any]
    gates_enabled: bool
    final_pnl: float
    mark_to_market_pnl: float
    effective_pnl: float
    exit_price: float | None
    exit_timestamp: datetime | None
    exit_reason: str
    final_position_size: int
    decisions_proposed: int
    decisions_accepted: int
    decisions_rejected: int
    gate_rejections_by_gate: dict[str, int]
    gate_rejection_reasons: list[dict[str, Any]]
    events_observed: int
    bars_consumed: int


def _build_advisor_policies(
    replay_data: TradeReplayData,
) -> list[tuple[str, dict[str, Any], ExitPolicy]]:
    """Construct the full policy list for the sweep.

    Returns ``(name, params_dict, policy_instance)`` triples so the
    runner can serialize the params separately from re-instantiating
    the policy.
    """
    out: list[tuple[str, dict[str, Any], ExitPolicy]] = []
    out.append(("ActualPolicy", {}, ActualPolicy(replay_data)))
    out.append(("OracleExitPolicy", {}, OracleExitPolicy(replay_data)))
    for cls_name, sweeps in DEFAULT_PARAMETER_SWEEPS.items():
        cls = POLICY_FACTORIES[cls_name]
        for params in sweeps:
            out.append((cls_name, dict(params), cls(**params)))
    return out


def _run_one(
    replay_data: TradeReplayData,
    policy_name: str,
    policy_params: dict[str, Any],
    policy: ExitPolicy,
    events_cfg: Any,
    gates_cfg: ExitGatesConfig,
    gates_enabled: bool,
) -> PolicyRunResult:
    """Run the harness once with the given policy + gate config and
    flatten the result into a comparison row."""
    effective_gates = gates_cfg if gates_enabled else ExitGatesConfig(enabled=False)
    harness = TradeReplayHarness(
        replay_data,
        policy,
        events_cfg,
        gates_config=effective_gates,
    )
    result = harness.run()
    return _result_to_row(
        replay_data,
        policy_name,
        policy_params,
        gates_enabled,
        result,
    )


def _result_to_row(
    replay_data: TradeReplayData,
    policy_name: str,
    policy_params: dict[str, Any],
    gates_enabled: bool,
    result: ReplayResult,
) -> PolicyRunResult:
    # Gate rejections summary.
    by_gate: dict[str, int] = {}
    reasons: list[dict[str, Any]] = []
    for rej in result.gate_rejections:
        by_gate[rej.gate_name] = by_gate.get(rej.gate_name, 0) + 1
        reasons.append(
            {
                "gate": rej.gate_name,
                "reason": rej.rejection_reason,
                **{k: v for k, v in rej.rejection_detail.items() if not isinstance(v, dict)},
            }
        )

    # Mark-to-market for "no exit" cases — treats the open position as if
    # the operator had closed at the recorded exit price.
    if result.final_position_size > 0:
        # Position size at end. The harness's TradeState wasn't preserved
        # in ReplayResult (intentional — replay-only state), but we can
        # reconstruct the unrealized P&L from the trade's entry/exit prices.
        if replay_data.fill_event:
            entry = float(replay_data.fill_event["fill_price"])
            shares = int(replay_data.fill_event["filled_shares"])
        else:
            entry = float(replay_data.bracket_event["entry_price"])
            shares = int(replay_data.bracket_event["shares"])
        mtm = (replay_data.recorded_exit_price - entry) * shares
        exit_reason = "open_at_end_of_replay"
    else:
        mtm = result.final_pnl
        exit_reason = result.notes[-1] if result.notes else "exit_unknown"

    effective = result.final_pnl if result.final_position_size == 0 else mtm

    return PolicyRunResult(
        policy_name=policy_name,
        policy_params=policy_params,
        gates_enabled=gates_enabled,
        final_pnl=result.final_pnl,
        mark_to_market_pnl=mtm,
        effective_pnl=effective,
        exit_price=result.exit_price,
        exit_timestamp=result.exit_timestamp,
        exit_reason=exit_reason,
        final_position_size=result.final_position_size,
        decisions_proposed=result.decisions_proposed,
        decisions_accepted=result.decisions_accepted,
        decisions_rejected=result.decisions_rejected,
        gate_rejections_by_gate=by_gate,
        gate_rejection_reasons=reasons,
        events_observed=len(result.events_emitted),
        bars_consumed=result.bars_consumed,
    )


def _format_pnl(value: float) -> str:
    if value >= 0:
        return f"+${value:.2f}"
    return f"-${abs(value):.2f}"


def _policy_label(name: str, params: dict[str, Any]) -> str:
    if not params:
        return name
    parts = [f"{k}={v}" for k, v in params.items()]
    return f"{name}({', '.join(parts)})"


def _format_summary_table(
    rows: list[PolicyRunResult], baseline_pnl: float, oracle_pnl: float
) -> str:
    """Build the gates-on/gates-off summary table block."""
    header = (
        "| Policy | Effective P&L | vs Actual | vs Oracle | Exit | "
        "Decisions (P/A/R) | Open at end? |\n"
        "| --- | ---: | ---: | ---: | --- | ---: | :---: |\n"
    )
    lines = [header]
    for row in rows:
        label = _policy_label(row.policy_name, row.policy_params)
        vs_actual = _format_pnl(row.effective_pnl - baseline_pnl)
        vs_oracle = _format_pnl(row.effective_pnl - oracle_pnl)
        exit_str = f"{row.exit_price:.4f}" if row.exit_price is not None else "—"
        open_marker = "✓" if row.final_position_size > 0 else ""
        lines.append(
            f"| {label} | {_format_pnl(row.effective_pnl)} | "
            f"{vs_actual} | {vs_oracle} | {exit_str} | "
            f"{row.decisions_proposed}/{row.decisions_accepted}/{row.decisions_rejected} | "
            f"{open_marker} |\n"
        )
    return "".join(lines)


def _build_markdown_report(
    symbol: str,
    trade_date: date,
    replay_data: TradeReplayData,
    rows_gates_on: list[PolicyRunResult],
    rows_gates_off: list[PolicyRunResult],
) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    fill_price = float(replay_data.fill_event.get("fill_price") or 0)
    fill_shares = int(replay_data.fill_event.get("filled_shares") or 0)

    actual_row = next(r for r in rows_gates_on if r.policy_name == "ActualPolicy")
    oracle_row = next(r for r in rows_gates_on if r.policy_name == "OracleExitPolicy")
    baseline = actual_row.effective_pnl
    oracle_pnl = oracle_row.effective_pnl

    # Best mechanical (excluding ActualPolicy + OracleExitPolicy).
    mechanicals_on = [
        r for r in rows_gates_on if r.policy_name not in ("ActualPolicy", "OracleExitPolicy")
    ]
    mechanicals_off = [
        r for r in rows_gates_off if r.policy_name not in ("ActualPolicy", "OracleExitPolicy")
    ]
    best_on = max(mechanicals_on, key=lambda r: r.effective_pnl) if mechanicals_on else None
    best_off = max(mechanicals_off, key=lambda r: r.effective_pnl) if mechanicals_off else None

    # Net gates impact: sum of (off - on) effective_pnl across mechanicals.
    on_by_key = {(r.policy_name, _params_key(r.policy_params)): r for r in mechanicals_on}
    gates_impact_total = 0.0
    for r_off in mechanicals_off:
        key = (r_off.policy_name, _params_key(r_off.policy_params))
        if key in on_by_key:
            gates_impact_total += r_off.effective_pnl - on_by_key[key].effective_pnl

    out: list[str] = []
    out.append(f"# Policy Comparison Report — {symbol} {trade_date}\n\n")
    out.append(f"Generated: {now}\n\n")
    out.append(
        f"Trade: {symbol}, entry ${fill_price:.4f} × {fill_shares} sh at "
        f"{replay_data.fill_event.get('timestamp', '?')}, "
        f"recorded exit ${replay_data.recorded_exit_price:.4f}, "
        f"recorded P&L {_format_pnl(replay_data.recorded_pnl)}.\n\n"
    )
    out.append(
        f"Pre-trade backfill: {len(replay_data.pre_trade_bars)} bars  |  "
        f"Trade-window: {len(replay_data.bars)} bars  |  "
        f"Prior-day cache: {replay_data.prior_day_cache_state}\n\n"
    )

    out.append("## Summary table (gates active)\n\n")
    out.append(_format_summary_table(rows_gates_on, baseline, oracle_pnl))
    out.append("\n")

    out.append("## Summary table (gates disabled)\n\n")
    out.append(_format_summary_table(rows_gates_off, baseline, oracle_pnl))
    out.append("\n")

    out.append("## Key findings\n\n")
    out.append(f"- **Oracle ceiling (theoretical max):** {_format_pnl(oracle_pnl)}\n")
    out.append(f"- **ActualPolicy baseline:** {_format_pnl(baseline)}\n")
    if best_on is not None:
        out.append(
            "- **Best mechanical with gates active:** "
            f"`{_policy_label(best_on.policy_name, best_on.policy_params)}` → "
            f"{_format_pnl(best_on.effective_pnl)} "
            f"(vs Actual: {_format_pnl(best_on.effective_pnl - baseline)})\n"
        )
    if best_off is not None:
        out.append(
            "- **Best mechanical without gates:** "
            f"`{_policy_label(best_off.policy_name, best_off.policy_params)}` → "
            f"{_format_pnl(best_off.effective_pnl)} "
            f"(vs Actual: {_format_pnl(best_off.effective_pnl - baseline)})\n"
        )
    out.append(
        f"- **Gates net impact across mechanicals:** {_format_pnl(gates_impact_total)} "
        "(positive = gates protected on net; negative = gates over-constrained)\n"
    )
    out.append("\n")

    # Notable gate rejections.
    rejections_block = _build_rejections_block(rows_gates_on)
    if rejections_block:
        out.append("## Notable gate rejections\n\n")
        out.append(rejections_block)
        out.append("\n")

    # Failure-mode analysis.
    out.append("## Failure mode analysis\n\n")
    out.append("### Failure mode 1 — breakouts that don't reach 2:1\n\n")
    fm1_candidates = [
        r for r in rows_gates_on if r.policy_name in ("FixedRTakeProfit", "StallExitPolicy")
    ]
    if fm1_candidates:
        best_fm1 = max(fm1_candidates, key=lambda r: r.effective_pnl)
        out.append(
            f"- ActualPolicy outcome: {_format_pnl(baseline)}\n"
            f"- Best fixed-R / stall policy: "
            f"`{_policy_label(best_fm1.policy_name, best_fm1.policy_params)}` → "
            f"{_format_pnl(best_fm1.effective_pnl)} "
            f"(Δ {_format_pnl(best_fm1.effective_pnl - baseline)})\n\n"
        )
    out.append("### Failure mode 2 — runner exhaustion after scale-out\n\n")
    out.append(
        "- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), "
        "so failure mode 2 is not exercised by this comparison. Multi-trade aggregation "
        "or a different sample trade is needed to evaluate runner-management policies.\n\n"
    )

    out.append("## Per-policy detail\n\n")
    for row in rows_gates_on:
        label = _policy_label(row.policy_name, row.policy_params)
        out.append(f"### {label}  (gates on)\n\n")
        out.append(f"- Effective P&L: {_format_pnl(row.effective_pnl)}\n")
        out.append(f"- Final realized P&L: {_format_pnl(row.final_pnl)}\n")
        if row.final_position_size > 0:
            out.append(
                f"- Position open at end of replay ({row.final_position_size} sh); "
                "effective P&L is mark-to-market at recorded exit price.\n"
            )
        out.append(
            f"- Decisions: {row.decisions_proposed} proposed / "
            f"{row.decisions_accepted} accepted / {row.decisions_rejected} rejected\n"
        )
        if row.gate_rejections_by_gate:
            out.append("- Gate rejections by gate:\n")
            for gate, n in row.gate_rejections_by_gate.items():
                out.append(f"  - `{gate}`: {n}\n")
        out.append(f"- Events observed: {row.events_observed}\n")
        out.append(f"- Bars consumed: {row.bars_consumed}\n\n")

    return "".join(out)


def _params_key(params: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _build_rejections_block(rows: list[PolicyRunResult]) -> str:
    """Markdown block listing (policy, params, gate, count) for any
    policy run with non-trivial rejections (>= 3)."""
    notable: list[str] = []
    for row in rows:
        for gate, count in row.gate_rejections_by_gate.items():
            if count >= 3:
                notable.append(
                    f"- `{_policy_label(row.policy_name, row.policy_params)}`: "
                    f"`{gate}` rejected {count} decisions"
                )
    return "\n".join(notable) + "\n" if notable else ""


def _row_to_jsonl(row: PolicyRunResult, symbol: str, trade_date: date) -> str:
    obj = {
        "symbol": symbol,
        "trade_date": trade_date.isoformat(),
        "policy_name": row.policy_name,
        "policy_params": row.policy_params,
        "gates_enabled": row.gates_enabled,
        "final_pnl": row.final_pnl,
        "mark_to_market_pnl": row.mark_to_market_pnl,
        "effective_pnl": row.effective_pnl,
        "exit_price": row.exit_price,
        "exit_timestamp": row.exit_timestamp.isoformat() if row.exit_timestamp else None,
        "exit_reason": row.exit_reason,
        "final_position_size": row.final_position_size,
        "decisions_proposed": row.decisions_proposed,
        "decisions_accepted": row.decisions_accepted,
        "decisions_rejected": row.decisions_rejected,
        "gate_rejections_by_gate": row.gate_rejections_by_gate,
        "gate_rejection_reasons": row.gate_rejection_reasons,
        "events_observed": row.events_observed,
        "bars_consumed": row.bars_consumed,
    }
    return json.dumps(obj, default=str)


def run_comparison(
    symbol: str,
    trade_date: date,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Top-level entry. Returns (markdown_path, jsonl_path)."""
    cache_path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    cache = HistoricalBarCache(cache_dir=cache_path)
    if not cache.is_available(symbol, trade_date):
        raise RuntimeError(
            f"Cache not populated for ({symbol}, {trade_date}). "
            f"Run: python scripts/fetch_historical_bars.py "
            f"--symbol {symbol} --date {trade_date.isoformat()}"
        )

    replay_data = load_trade_replay_data(symbol, trade_date, cache_dir=cache_path)
    settings = Settings()

    rows_on: list[PolicyRunResult] = []
    rows_off: list[PolicyRunResult] = []
    for name, params, policy_instance in _build_advisor_policies(replay_data):
        # Re-instantiate per gate-mode so per-policy state (e.g. the
        # ``_exit_emitted`` latch) doesn't leak between runs.
        policy_on = _reinstantiate(name, params, replay_data)
        policy_off = _reinstantiate(name, params, replay_data)
        rows_on.append(
            _run_one(
                replay_data,
                name,
                params,
                policy_on,
                settings.exit_events,
                settings.exit_gates,
                gates_enabled=True,
            )
        )
        rows_off.append(
            _run_one(
                replay_data,
                name,
                params,
                policy_off,
                settings.exit_events,
                settings.exit_gates,
                gates_enabled=False,
            )
        )
        # Avoid unused-variable warning for the original instance.
        del policy_instance

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{symbol}_{trade_date.isoformat()}_comparison.md"
    jsonl_path = output_dir / f"{symbol}_{trade_date.isoformat()}_comparison.jsonl"

    md_path.write_text(
        _build_markdown_report(symbol, trade_date, replay_data, rows_on, rows_off),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows_on + rows_off:
            fh.write(_row_to_jsonl(row, symbol, trade_date) + "\n")

    return md_path, jsonl_path


def _reinstantiate(name: str, params: dict[str, Any], replay_data: TradeReplayData) -> ExitPolicy:
    if name == "ActualPolicy":
        return ActualPolicy(replay_data)
    if name == "OracleExitPolicy":
        return OracleExitPolicy(replay_data)
    return POLICY_FACTORIES[name](**params)  # type: ignore[no-any-return]


DEFAULT_MANIFEST = Path("reports/exit_advisor/closed_trades_manifest.jsonl")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--symbol", help="Single-trade mode")
    parser.add_argument("--date", help="Trading date (YYYY-MM-DD) for --symbol mode")
    parser.add_argument(
        "--all-trades",
        action="store_true",
        help="Batch mode: every closed trade in the manifest",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Manifest used by --all-trades",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/exit_advisor/comparisons/",
        help="Where to write the .md + .jsonl outputs",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override historical-bar cache dir (default: real cache)",
    )
    return parser.parse_args(argv)


def _run_all_trades(args: argparse.Namespace) -> int:
    """Batch mode. Iterate the manifest; for each trade, run a comparison
    if its cache is populated; skip with a warning otherwise. Never fail
    the whole batch on a single missing cache or replay error."""
    from bot.exit_advisor.replay.trade_discovery import read_manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"error: manifest not found at {manifest_path}. "
            "Run: python scripts/discover_closed_trades.py",
            file=sys.stderr,
        )
        return 1

    refs = read_manifest(manifest_path)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    output_dir = Path(args.output_dir)

    processed = 0
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    for ref in refs:
        label = f"{ref.symbol} {ref.trade_date}"
        try:
            md_path, _ = run_comparison(
                symbol=ref.symbol,
                trade_date=ref.trade_date,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )
            processed += 1
            # ASCII arrow — Windows cp1252 console can't encode the unicode one.
            print(f"  ok   {label} -> {md_path.name}")
        except RuntimeError as exc:
            # Cache missing — graceful skip per spec.
            skipped.append(label)
            print(f"  skip {label}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((label, repr(exc)))
            print(f"  FAIL {label}: {exc}")

    print()
    print(f"processed={processed} skipped={len(skipped)} failed={len(failed)}")
    if skipped:
        print(f"skipped trades (cache missing): {', '.join(skipped)}")
    if failed:
        for label, err in failed:
            print(f"failed: {label} -- {err}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.all_trades:
        return _run_all_trades(args)

    if not (args.symbol and args.date):
        print("Specify either --all-trades or both --symbol and --date.", file=sys.stderr)
        return 2

    try:
        trade_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"Bad --date: {args.date}", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    try:
        md_path, jsonl_path = run_comparison(
            symbol=args.symbol,
            trade_date=trade_date,
            output_dir=Path(args.output_dir),
            cache_dir=cache_dir,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {md_path}")
    print(f"wrote {jsonl_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
