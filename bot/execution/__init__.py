"""Signal → bracket order → fill → close pipeline.

The four modules here form one tightly-coupled state machine:

* :mod:`bot.execution.executor` — converts ``Signal`` objects into
  IBKR bracket orders, wires fill handlers, and drives the entry
  side of the position lifecycle.
* :mod:`bot.execution.trade_manager` — bar-close exit logic
  (scale-out, trailing stop, pre-scale red-candle exit) on the
  remaining tail.
* :mod:`bot.execution.position_state` — in-memory ``PositionStore``
  (the executor's source of truth) plus the one-way state machine
  ``pending_entry → open → closing → closed``.
* :mod:`bot.execution.watchdog` — naked-position detector that
  cross-checks bot state against IBKR-side working orders.
"""
