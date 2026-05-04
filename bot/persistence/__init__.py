"""SQLite-backed persistence (write-only historical record, not primary state).

The ``PositionStore`` (in :mod:`bot.execution.position_state`) is the
authoritative source of truth for "does the bot have a position on
``X``?". The journal here is the append-only history layer used by
post-session reports and the rehab tier's lookback queries.
"""
