"""Comparison-runner integration tests using the synthetic ZENA fixture.

Per the validation step's discipline: tests don't run against real
cache state. They use the committed synthetic fixtures so the runner's
structure can be exercised reproducibly across machines.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from bot.exit_advisor.replay.replay_source import load_trade_replay_data
from scripts.run_policy_comparison import (
    DEFAULT_PARAMETER_SWEEPS,
    _build_advisor_policies,
    run_comparison,
)

ZENA_DATE = date(2026, 4, 30)
FIXTURES = Path(__file__).parent / "fixtures" / "exit_advisor_zena"


def test_default_parameter_sweep_shape() -> None:
    """Pin the v1 parameter sweep so a refactor that drops a sweep
    entry surfaces in tests rather than silently shrinking the report."""
    assert "MechanicalTrailPolicy" in DEFAULT_PARAMETER_SWEEPS
    assert "FixedRTakeProfit" in DEFAULT_PARAMETER_SWEEPS
    assert "StallExitPolicy" in DEFAULT_PARAMETER_SWEEPS
    assert len(DEFAULT_PARAMETER_SWEEPS["MechanicalTrailPolicy"]) == 7
    assert len(DEFAULT_PARAMETER_SWEEPS["FixedRTakeProfit"]) == 5
    assert len(DEFAULT_PARAMETER_SWEEPS["StallExitPolicy"]) == 5


def test_build_advisor_policies_includes_actual_and_oracle() -> None:
    """Every comparison run must include the ActualPolicy baseline and
    the OracleExitPolicy ceiling — the report's vs-Actual / vs-Oracle
    columns depend on these being present."""
    rd = load_trade_replay_data("ZENA", ZENA_DATE, cache_dir=FIXTURES)
    triples = _build_advisor_policies(rd)
    names = [name for name, _, _ in triples]
    assert "ActualPolicy" in names
    assert "OracleExitPolicy" in names
    # Total = 1 (Actual) + 1 (Oracle) + 7 trail + 5 fixed-R + 5 stall = 19
    assert len(triples) == 19


def test_runner_refuses_when_cache_missing(tmp_path: Path) -> None:
    """Refusing-to-proceed has a clear remediation message — operator
    needs to know which fetch command to run."""
    # tmp_path is empty, so no cache file exists for any (symbol, date).
    with pytest.raises(RuntimeError, match="Cache not populated"):
        run_comparison(
            symbol="ZENA",
            trade_date=ZENA_DATE,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path,
        )


def test_runner_produces_md_and_jsonl(tmp_path: Path) -> None:
    """End-to-end: against the synthetic fixture, produce both output
    files. Don't assert specific P&L numbers — those depend on the
    fixture's random seed and would couple the test to the data."""
    md_path, jsonl_path = run_comparison(
        symbol="ZENA",
        trade_date=ZENA_DATE,
        output_dir=tmp_path / "reports",
        cache_dir=FIXTURES,
    )
    assert md_path.exists()
    assert jsonl_path.exists()

    md = md_path.read_text(encoding="utf-8")
    assert "# Policy Comparison Report" in md
    assert "## Summary table (gates active)" in md
    assert "## Summary table (gates disabled)" in md
    assert "## Key findings" in md
    assert "## Failure mode analysis" in md
    assert "## Per-policy detail" in md


def test_runner_jsonl_one_row_per_policy_run(tmp_path: Path) -> None:
    """Every sweep × gate-mode produces one JSONL row. With 19 policies
    and gates-on/off, the JSONL has 38 lines."""
    _, jsonl_path = run_comparison(
        symbol="ZENA",
        trade_date=ZENA_DATE,
        output_dir=tmp_path / "reports",
        cache_dir=FIXTURES,
    )
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 38
    # Each line is valid JSON with the expected shape.
    for line in lines:
        obj = json.loads(line)
        assert "policy_name" in obj
        assert "gates_enabled" in obj
        assert "effective_pnl" in obj
        assert "decisions_proposed" in obj


def test_runner_runs_each_policy_twice(tmp_path: Path) -> None:
    """Once with gates on, once with gates off, for every policy."""
    _, jsonl_path = run_comparison(
        symbol="ZENA",
        trade_date=ZENA_DATE,
        output_dir=tmp_path / "reports",
        cache_dir=FIXTURES,
    )
    rows = [json.loads(ln) for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln]
    by_pair: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in rows:
        key = (r["policy_name"], json.dumps(r["policy_params"], sort_keys=True))
        by_pair.setdefault(key, []).append(r)
    for key, runs in by_pair.items():
        assert len(runs) == 2, f"{key} ran {len(runs)} times, expected 2"
        modes = {r["gates_enabled"] for r in runs}
        assert modes == {True, False}, f"{key} missing one gate-mode"


def test_runner_actual_policy_pnl_matches_recorded(tmp_path: Path) -> None:
    """Regression: the comparison runner is just a wrapper around the
    harness. ActualPolicy must still reproduce the recorded -$2.38."""
    _, jsonl_path = run_comparison(
        symbol="ZENA",
        trade_date=ZENA_DATE,
        output_dir=tmp_path / "reports",
        cache_dir=FIXTURES,
    )
    rows = [json.loads(ln) for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln]
    actual_rows = [r for r in rows if r["policy_name"] == "ActualPolicy"]
    assert len(actual_rows) == 2  # gates on + off
    for r in actual_rows:
        assert abs(r["effective_pnl"] - (-2.38)) < 0.01
