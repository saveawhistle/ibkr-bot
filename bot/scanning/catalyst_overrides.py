"""Phase 6.8 — file-backed manual catalyst overrides for paper trading.

When the Finnhub-keyword classifier misses a real catalyst (weekend news
outside the lookback, unusual phrasing, sector/policy spillover that
doesn't match any green-bucket keyword), the operator injects a catalyst
manually via ``bot inject-catalyst`` and the scanner picks it up on the
next scan without re-running Finnhub for that symbol.

Non-negotiable safety properties:

* **Load is gated at the call site.** This module provides pure
  load/save/upsert primitives; callers (the CLI and the scanner) are
  responsible for checking ``settings.testing.allow_catalyst_overrides``
  BEFORE calling any function here. Defence in depth: even a stale file
  on disk can't affect a live run if the flag is off.
* **Auto-expiration is per-entry.** ``CatalystOverride.is_active(now)``
  returns False past ``expires_at``. ``find_active_override`` skips
  expired entries lazily — no cron or manual cleanup needed.
* **No unbounded growth.** ``upsert_override`` replaces by symbol, so
  re-injecting the same ticker overwrites rather than accumulates.
  Expired entries persist until the next upsert for the same symbol
  (harmless — they're ignored by reads) or a manual file delete.

The on-disk format is a JSON array of entries at
``data/test_catalyst_overrides.json``; ``data/`` is gitignored so operator
state never contaminates the repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

_log = structlog.get_logger("bot.scanning.catalyst_overrides")

DEFAULT_STORE_PATH = Path("data/test_catalyst_overrides.json")


@dataclass(frozen=True)
class CatalystOverride:
    """A single operator-injected catalyst for one symbol, with explicit expiration."""

    symbol: str
    category: str
    expires_at: datetime
    note: str | None
    injected_at: datetime
    injected_by: str  # "cli" today; reserved for future ("api", "test", etc.)

    def is_active(self, now: datetime) -> bool:
        """Strict less-than: ``now == expires_at`` counts as expired."""
        return now < self.expires_at

    def to_dict(self) -> dict[str, str | None]:
        """Serialize to the on-disk JSON shape (ISO-8601 timestamps)."""
        return {
            "symbol": self.symbol,
            "category": self.category,
            "expires_at": self.expires_at.isoformat(),
            "note": self.note,
            "injected_at": self.injected_at.isoformat(),
            "injected_by": self.injected_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CatalystOverride:
        """Parse one JSON entry; raises ``KeyError``/``ValueError`` on malformed data."""
        expires_raw = data["expires_at"]
        injected_raw = data["injected_at"]
        if not isinstance(expires_raw, str) or not isinstance(injected_raw, str):
            raise ValueError("expires_at and injected_at must be ISO-8601 strings")
        note = data.get("note")
        return cls(
            symbol=str(data["symbol"]),
            category=str(data["category"]),
            expires_at=datetime.fromisoformat(expires_raw),
            note=str(note) if note is not None else None,
            injected_at=datetime.fromisoformat(injected_raw),
            injected_by=str(data.get("injected_by", "cli")),
        )


def load_overrides(path: Path = DEFAULT_STORE_PATH) -> list[CatalystOverride]:
    """Return every override currently on disk, or [] if the file is absent/malformed.

    A malformed file logs a warning and returns an empty list rather than
    raising — the scanner must never crash because the operator edited
    the JSON by hand and slipped a comma.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("catalyst_overrides.load_failed", path=str(path), error=str(exc))
        return []
    if not isinstance(raw, list):
        _log.warning("catalyst_overrides.unexpected_format", path=str(path))
        return []
    result: list[CatalystOverride] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            result.append(CatalystOverride.from_dict(entry))
        except (KeyError, ValueError) as exc:
            _log.warning("catalyst_overrides.parse_failed", error=str(exc))
    return result


def save_overrides(overrides: list[CatalystOverride], path: Path = DEFAULT_STORE_PATH) -> None:
    """Serialize overrides to disk, creating ``path.parent`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [o.to_dict() for o in overrides]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_override(
    override: CatalystOverride, path: Path = DEFAULT_STORE_PATH
) -> list[CatalystOverride]:
    """Add or replace an override matching by symbol; returns the new list.

    Replace-by-symbol (not append) because an operator re-injecting an
    existing ticker typically means "update the note / extend the expiry";
    keeping both entries would create ambiguity on read.
    """
    existing = load_overrides(path)
    filtered = [o for o in existing if o.symbol != override.symbol]
    filtered.append(override)
    save_overrides(filtered, path)
    return filtered


def find_active_override(
    symbol: str,
    *,
    now: datetime,
    path: Path = DEFAULT_STORE_PATH,
) -> CatalystOverride | None:
    """Return the active override for ``symbol`` or None. Expired entries are skipped."""
    for override in load_overrides(path):
        if override.symbol == symbol and override.is_active(now):
            return override
    return None


def load_active_overrides_map(
    *,
    now: datetime,
    path: Path = DEFAULT_STORE_PATH,
) -> dict[str, CatalystOverride]:
    """Return ``{symbol: override}`` for every currently-active entry.

    Used by the scanner to partition the symbol list into "has override"
    (skip Finnhub fetch) and "needs normal classification" on each scan.
    """
    return {o.symbol: o for o in load_overrides(path) if o.is_active(now)}


__all__ = [
    "DEFAULT_STORE_PATH",
    "CatalystOverride",
    "find_active_override",
    "load_active_overrides_map",
    "load_overrides",
    "save_overrides",
    "upsert_override",
]
