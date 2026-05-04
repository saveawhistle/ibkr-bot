"""CLI: discover closed trades from session logs and write a manifest.

Usage:
    python scripts/discover_closed_trades.py

Reads from ``logs/`` (configurable via ``--logs-dir``) and writes
the manifest to ``reports/exit_advisor/closed_trades_manifest.jsonl``.
Re-runnable; overwrites the previous manifest.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from bot.exit_advisor.replay.trade_discovery import discover_closed_trades, write_manifest

DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_MANIFEST = Path("reports/exit_advisor/closed_trades_manifest.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refs = discover_closed_trades(Path(args.logs_dir))
    write_manifest(refs, Path(args.manifest))

    if not refs:
        print("No closed trades discovered.")
        return 0

    by_symbol = Counter(r.symbol for r in refs)
    earliest = min(r.trade_date for r in refs)
    latest = max(r.trade_date for r in refs)

    print(f"Total closed trades: {len(refs)}")
    print(f"Date range: {earliest} to {latest}")
    print("By symbol:")
    for sym, n in by_symbol.most_common():
        print(f"  {sym}: {n}")
    print(f"Manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
