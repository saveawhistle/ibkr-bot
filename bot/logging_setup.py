"""Phase 5.1 — structlog configuration + optional per-session JSONL file handler.

Originally lived inside ``bot.brokerage.ibkr_client`` with ``stream=sys.stdout``
hardcoded and no way to route structured logs to disk. Phase 5.1 moves
the setup out (it has nothing to do with IBKR) and reads the new
``LoggingSettings`` so an operator can enable file logging via
``config.yaml`` (or ``BOT_LOGGING__PATH=...``) without touching code.

One handler is always attached for ``sys.stdout``. When
``settings.logging.path`` is set, an additional ``FileHandler`` is
attached at ``{path}/session_{YYYY-MM-DD}.jsonl`` where the date uses
``session.timezone`` (so a single session that spans UTC midnight does
not split across two files).

``configure_logging`` is idempotent — subsequent calls are no-ops so the
CLI can call it from every command without worrying about duplicate
handlers on restart-in-the-same-process cases (mainly tests).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from bot.config import Settings, get_settings

_LOG_CONFIGURED = False


def configure_logging(settings: Settings | None = None) -> None:
    """Initialise structlog + optional session JSONL FileHandler (idempotent)."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    resolved = settings or get_settings()
    cfg = resolved.logging
    level = getattr(logging, cfg.level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # Wipe existing handlers so repeated calls in the same process (tests
    # re-using a warmed interpreter) don't leak duplicates. The idempotent
    # flag would normally short-circuit this, but pytest fixtures that
    # explicitly reset ``_LOG_CONFIGURED`` rely on the cleanup.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    plain_formatter = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(plain_formatter)
    root.addHandler(stdout_handler)

    if cfg.path is not None:
        path = Path(cfg.path)
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / _session_log_filename(resolved)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(plain_formatter)
        root.addHandler(file_handler)

    renderer = (
        structlog.processors.JSONRenderer()
        if cfg.json_renderer
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Keep the first-call logger cache OFF. Caching pins the processor
        # chain onto the cached BoundLogger, which means later test code
        # using ``structlog.testing.capture_logs`` sees an empty buffer
        # because the cached logger still routes through the production
        # processors. The perf cost of rebuilding a BoundLogger per call
        # is trivial; test reliability is not.
        cache_logger_on_first_use=False,
    )
    _LOG_CONFIGURED = True


def resolve_session_log_path(settings: Settings | None = None) -> Path | None:
    """Return the JSONL file path the next ``configure_logging`` call will write to.

    Returns ``None`` when file logging is disabled (``logging.path`` unset).
    The ``status`` CLI uses this to show operators where to expect the
    current session's file before starting the loop.
    """
    resolved = settings or get_settings()
    if resolved.logging.path is None:
        return None
    return Path(resolved.logging.path) / _session_log_filename(resolved)


def _session_log_filename(settings: Settings) -> str:
    """Build ``session_{YYYY-MM-DD}.jsonl`` using the configured session timezone."""
    tz = ZoneInfo(settings.session.timezone)
    session_date = datetime.now(tz).date().isoformat()
    return f"session_{session_date}.jsonl"
