"""Operator-managed manual watchlist entries that bypass the scanner gates.

Sister module to ``bot.scanning.catalyst_overrides``. Where overrides attach
a synthetic catalyst to a symbol that the IBKR scanner DOES return,
manual watchlist entries inject the symbol into the watchlist directly --
the IBKR ``TOP_PERC_GAIN`` snapshot doesn't have to include it. Use case:
the operator has decided independently of the gappers list that ATRA is
worth watching today (FDA news, earnings whisper, sympathy play, etc.)
and wants live bars + strategy evaluation on it.

Manual entries skip every scanner pillar: price / gap / premarket-vol /
float / rvol / catalyst classification. The risk engine still gates any
signal the strategies fire, so this is an *observability + opportunity*
escape hatch, not a way around the safety net.

Non-negotiable safety properties (mirror catalyst_overrides):

* **Load is gated at the call site.** Pure load/save primitives here;
  callers (CLI + scanner) check ``settings.testing.allow_catalyst_overrides``
  BEFORE invoking. Defence in depth: a stale file on disk can't affect a
  live run if the flag is off.
* **Auto-expiration is per-entry.** ``ManualWatchlistEntry.is_active(now)``
  returns False past ``expires_at``. Filtered lazily on read -- no cron.
* **No unbounded growth.** ``upsert_entry`` replaces by symbol, so
  re-adding the same ticker overwrites rather than accumulates.

On-disk format is a JSON array at ``data/manual_watchlist.json`` (the
``data/`` directory is gitignored so operator state stays out of the
repo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

_log = structlog.get_logger("bot.scanning.manual_watchlist")

DEFAULT_STORE_PATH = Path("data/manual_watchlist.json")

MANUAL_CATALYST_SENTINEL = "manual_watchlist"
"""Catalyst tag attached to every ScanHit derived from a manual entry.

Distinct from any classifier-emitted category so post-session forensics
(grep on the JSONL for ``catalyst="manual_watchlist"``) clearly separate
operator-injected hits from organically-classified ones.
"""


@dataclass(frozen=True)
class ManualWatchlistEntry:
    """A single operator-injected watchlist symbol with explicit expiration."""

    symbol: str
    expires_at: datetime
    note: str | None
    added_at: datetime
    added_by: str  # "cli" today; reserved for future ("api", "test", etc.)

    def is_active(self, now: datetime) -> bool:
        """Strict less-than: ``now == expires_at`` counts as expired."""
        return now < self.expires_at

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to the on-disk JSON shape (ISO-8601 timestamps)."""
        return {
            "symbol": self.symbol,
            "expires_at": self.expires_at.isoformat(),
            "note": self.note,
            "added_at": self.added_at.isoformat(),
            "added_by": self.added_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ManualWatchlistEntry:
        """Parse one JSON entry; raises ``KeyError``/``ValueError`` on malformed data."""
        expires_raw = data["expires_at"]
        added_raw = data["added_at"]
        if not isinstance(expires_raw, str) or not isinstance(added_raw, str):
            raise ValueError("expires_at and added_at must be ISO-8601 strings")
        note = data.get("note")
        return cls(
            symbol=str(data["symbol"]),
            expires_at=datetime.fromisoformat(expires_raw),
            note=str(note) if note is not None else None,
            added_at=datetime.fromisoformat(added_raw),
            added_by=str(data.get("added_by", "cli")),
        )


def load_entries(path: Path = DEFAULT_STORE_PATH) -> list[ManualWatchlistEntry]:
    """Return every entry currently on disk, or [] if absent/malformed.

    A malformed file logs a warning and returns an empty list rather than
    raising -- the scanner must never crash because the operator edited
    the JSON by hand and slipped a comma.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("manual_watchlist.load_failed", path=str(path), error=str(exc))
        return []
    if not isinstance(raw, list):
        _log.warning("manual_watchlist.unexpected_format", path=str(path))
        return []
    result: list[ManualWatchlistEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            result.append(ManualWatchlistEntry.from_dict(entry))
        except (KeyError, ValueError) as exc:
            _log.warning("manual_watchlist.parse_failed", error=str(exc))
    return result


def save_entries(
    entries: list[ManualWatchlistEntry], path: Path = DEFAULT_STORE_PATH
) -> None:
    """Serialize entries to disk, creating ``path.parent`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.to_dict() for e in entries]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_entry(
    entry: ManualWatchlistEntry, path: Path = DEFAULT_STORE_PATH
) -> list[ManualWatchlistEntry]:
    """Add or replace an entry matching by symbol; returns the new list.

    Replace-by-symbol (not append) because re-adding an existing ticker
    typically means "update the note / extend the expiry"; keeping both
    entries would create ambiguity on read.
    """
    existing = load_entries(path)
    filtered = [e for e in existing if e.symbol != entry.symbol]
    filtered.append(entry)
    save_entries(filtered, path)
    return filtered


def remove_entry(symbol: str, path: Path = DEFAULT_STORE_PATH) -> bool:
    """Remove the entry for ``symbol``; returns True if anything was removed.

    Idempotent -- removing an absent symbol returns False and writes no
    file (avoids gratuitous mtime churn).
    """
    existing = load_entries(path)
    filtered = [e for e in existing if e.symbol != symbol]
    if len(filtered) == len(existing):
        return False
    save_entries(filtered, path)
    return True


def load_active_entries(
    *,
    now: datetime,
    path: Path = DEFAULT_STORE_PATH,
) -> list[ManualWatchlistEntry]:
    """Return every currently-active (non-expired) entry.

    Used by the scanner each scan tick to merge manual entries into the
    final watchlist. Ordering matches insertion order on disk -- the
    scanner re-sorts according to its own placement policy.
    """
    return [e for e in load_entries(path) if e.is_active(now)]


__all__ = [
    "DEFAULT_STORE_PATH",
    "MANUAL_CATALYST_SENTINEL",
    "ManualWatchlistEntry",
    "load_active_entries",
    "load_entries",
    "remove_entry",
    "save_entries",
    "upsert_entry",
]
