"""Aggregate-report runner tests with mocked file I/O. Verifies that
missing detail files trigger graceful skips and that the date-stamped
filenames follow the expected pattern."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from bot.exit_advisor.replay.trade_discovery import ClosedTradeRef, write_manifest
from scripts import run_aggregate_report as runner


def _make_ref(symbol: str, ts: str) -> ClosedTradeRef:
    from datetime import datetime

    entry = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return ClosedTradeRef(
        symbol=symbol,
        trade_date=entry.date(),
        trade_id=ts,
        entry_timestamp=entry,
        exit_timestamp=entry,
        session_log_path=Path("logs/x.jsonl"),
    )


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Manifest not found"):
        runner.run_aggregate_report(
            manifest_path=tmp_path / "no_manifest.jsonl",
            comparisons_dir=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_empty_manifest_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        runner.run_aggregate_report(
            manifest_path=manifest,
            comparisons_dir=tmp_path,
            output_dir=tmp_path / "out",
        )


def test_skips_trades_without_detail_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    refs = [_make_ref("AAPL", "2026-04-30T15:00:00Z")]
    write_manifest(refs, manifest)
    # Comparisons dir is empty — no detail files.
    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        md_path, jsonl_path = runner.run_aggregate_report(
            manifest_path=manifest,
            comparisons_dir=tmp_path / "comparisons",  # nonexistent
            output_dir=out_dir,
        )
    assert md_path.exists()
    assert jsonl_path.exists()
    assert any("no per-trade detail file" in m.getMessage() for m in caplog.records)


def test_date_stamped_output_filenames(tmp_path: Path) -> None:
    """The output filenames embed today's local date — operator can
    eyeball the timestamp without opening the file."""
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([_make_ref("AAPL", "2026-04-30T15:00:00Z")], manifest)
    out_dir = tmp_path / "aggregates"
    md_path, jsonl_path = runner.run_aggregate_report(
        manifest_path=manifest,
        comparisons_dir=tmp_path / "comparisons",
        output_dir=out_dir,
    )
    iso = date.today().isoformat()
    assert md_path.name == f"closed_trades_aggregate_{iso}.md"
    assert jsonl_path.name == f"closed_trades_aggregate_{iso}.jsonl"


def test_classification_jsonl_writer_round_trip(tmp_path: Path) -> None:
    """Classifications round-trip via the JSONL writer."""
    from bot.exit_advisor.analysis.failure_modes import FailureMode, TradeClassification

    classifications = [
        TradeClassification(
            trade_ref=_make_ref("AAPL", "2026-04-30T15:00:00Z"),
            mode=FailureMode.MODE_1_FLAGGING_BREAKOUT,
            reasoning="peak 1.5R, actual 0.2R",
        )
    ]
    out = tmp_path / "classifications.jsonl"
    runner.write_classification_jsonl(classifications, out)
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln]
    assert len(lines) == 1
    assert lines[0]["mode"] == "mode_1_flagging_breakout"
    assert lines[0]["symbol"] == "AAPL"
