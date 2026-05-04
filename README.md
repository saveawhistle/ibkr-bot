# ibkr-bot

An automated day-trading bot for Interactive Brokers that encodes the **Gap and Go** and **Momentum (Bull Flag / HOD Breakout)** strategies. It scans the premarket universe through the 5-Pillar filters, enters via atomic bracket orders during the 9:30–11:30 ET window where the strategy's edge is strongest, and enforces non-negotiable daily risk rules ($-based per-trade and daily loss caps, profit-goal halt, give-back guard, automatic rehab tier on cold streaks, flat by 15:55 ET). See [PLAN.md](PLAN.md) for the strategy spec and the original architecture.

## Project status

**Paper trading.** The core production loop (Phase 4b risk + Phase 4i exit discipline) is fully exercised; layered enhancements through Phase 10.6 are landed and active; Phase 10.1 watchdog and Phase 11 exit advisor ship in **shadow / disabled** mode pending operator-controlled activation. No real money has run through this code.

What's in the main loop today:
- 5-Pillar premarket scanner with Finnhub catalyst classification and float lookup
- `GapAndGoStrategy` and `MomentumStrategy` operating on 1-min bars
- $-based risk engine (per-trade cap, daily loss halt, profit goal, give-back, max trades/day)
- Phase 4d multi-pullback re-entries (cooldown + profitable-prior-exit gate)
- Phase 4g automatic **rehab tier** scaling caps to 50% / 25% on consecutive red days or cumulative drawdown, with hysteresis recovery
- Phase 4j **BUY STP-LMT** server-side tick triggers (also LMT and MKT entry paths)
- Phase 8.2 + 10.6 LMT buffer with floor / dollar cap / percentage ceiling chain
- Phase 10.2 minimum stop-distance floor
- Phase 10.3 explicit `DAY` TIF on every leg
- 2-leg bracket (parent + breakeven STP + partial LMT scale-out at 2R), Phase 6.14 immediate-trail runner
- Phase 7.8 first-red-candle pre-scale exit (code path live, runtime-disabled by default — see [config.yaml](config.yaml))
- 15:55 ET auto-flatten and graceful reconciliation on restart
- SQLite trade journal, structlog JSONL session logs, optional Telegram push for watchlist + alerts

In shadow / disabled state, ready for activation:
- **Phase 10.1 watchdog** — naked-position detector, ships in `shadow_mode: true` (logs would-have-fired alerts; does not send Telegram)
- **Phase 11 exit advisor** — full hook surface, detector layers (time / PnL / order-state / price-levels / moving-averages / volume / bar-shape / NASDAQ TotalView L2), gate framework, and offline replay harness; production main has `exit_advisor.enabled: false`

## Prerequisites

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Interactive Brokers **TWS** or **IB Gateway** running locally with the API enabled
  (Global Config → API → Settings → *Enable ActiveX and Socket Clients*)
- An IBKR **paper trading** account logged into TWS/Gateway
- Market-data subscriptions sufficient for the scanner + real-time quotes
- Optional: Finnhub API key (catalyst news + float), Telegram bot token (notifications), NASDAQ TotalView entitlement (L2 detectors in the exit advisor)

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — IBKR connection, plus optional Finnhub + Telegram credentials
```

Defaults target **TWS paper** on `127.0.0.1:7497` with `client_id=17`. For IB Gateway paper, change the port to `4002`.

## Quickstart — connectivity check

With TWS/Gateway running and the paper account logged in:

```bash
uv run python -m bot ping
```

Expected output: a JSON log line on connect, a formatted table with `NetLiquidation`, `TotalCashValue`, `BuyingPower`, and `DayTradesRemaining`, and a clean disconnect.

## Architecture

```
bot/
├── cli.py              # typer entrypoint (15+ commands)
├── orchestrator.py     # session state machine — premarket → open → trading → flatten
├── config.py           # pydantic-settings (config.yaml + .env)
├── logging_setup.py    # structlog → JSONL session files
├── signal_bus.py       # async pub/sub between strategies and executor (dedupes Gap > Momentum on same symbol/bar)
├── notify.py           # Telegram (watchlist push + watchdog ack flow)
├── reports.py          # post-session commission / PnL aggregation
├── indicators.py       # VWAP, EMA, HOD, extension checks (pure)
├── backtest.py         # historical bar replay (signal-only, no PnL)
│
├── brokerage/          # IBKR connection + market data
│   ├── ibkr_client.py  # ib_async wrapper, reconnection, account summary
│   ├── market_data.py  # historical bars, real-time bar subscriptions
│   └── bar_aggregator.py # 5-sec → 1-min in-process aggregation
│
├── scanning/           # premarket universe → ranked watchlist
│   ├── scanner.py      # IBKR TOP_PERC_GAIN + 5-Pillar post-filter
│   ├── catalyst.py     # Finnhub news → category classifier (green/black list)
│   ├── catalyst_overrides.py  # operator-injected catalysts (paper testing)
│   ├── finnhub_client.py
│   └── float_source.py # Finnhub primary, yfinance fallback
│
├── strategies/         # signal generators (no order placement)
│   ├── gap_and_go.py
│   └── momentum.py
│
├── risk/
│   ├── engine.py       # sizing, halts, re-entry gate, margin awareness, halt-flag persistence
│   └── rehab.py        # auto tier detection (REHAB / DEEP_REHAB) + recovery hysteresis
│
├── execution/
│   ├── executor.py     # Signal → bracket order; reconciliation on startup
│   ├── trade_manager.py# bar-close exit logic: scale-out, trailing, pre-scale red-candle
│   ├── position_state.py # in-memory PositionStore state machine
│   └── watchdog.py     # Phase 10.1 naked-position detection (shadow mode)
│
├── persistence/
│   └── journal.py      # SQLite append-only trade record (commissions, fills, exits)
│
└── exit_advisor/       # Phase 11 (disabled in production main)
    ├── core/           # event taxonomy, types
    ├── detectors/      # bar-shape, MA, price-level, volume + L2 (pulls/absorption/spread/imbalance/prints)
    ├── decision/       # gate chain, policies, MaxHoldTime
    ├── hook/           # registry + applier (executes recommendations when hook_acts=true)
    ├── market/         # L2 stream + book state tracker
    ├── replay/         # offline harness with historical-bar cache
    └── analysis/       # multi-trade aggregation
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `ping` | Connect, print account summary, disconnect (Phase 1 connectivity check) |
| `scan` | Run the morning 5-Pillar scanner once and print/Telegram the ranked watchlist |
| `watch` | Subscribe to live 1-min bars for the watchlist and print signals (no orders) |
| `trade` | Full production loop — scan + subscribe + bracket execution + risk gates + auto-flatten |
| `backtest` | Replay one strategy against one ticker on a past date (see below) |
| `status` | Reconcile with IBKR; print risk state, halt flag, and active positions (read-only) |
| `positions` | Read-only IBKR-reconciled position table |
| `flatten` | Manual kill-switch — cancel bracket legs and market-close every active symbol |
| `reset-halt` | Delete `logs/halt.flag` so entries are allowed again (requires `--yes` to skip confirm) |
| `rehab-status` | Print the active rehab tier, effective scaled caps, and recovery target |
| `suggest-caps` | Analyze the journal and print advisory risk-cap suggestions (never writes config) |
| `commissions` | Per-leg commission rollup over a journal window |
| `inject-catalyst` | Phase 6.8 paper-testing hook — manually inject a catalyst category for a symbol |
| `force-entry` | Phase 6.13 paper-testing hook — fabricate a signal and push it through the executor |

Run any command with `--help` for its flags. Most commands need TWS/Gateway running; the journal-only commands (`commissions`, `suggest-caps`, `rehab-status`) and the catalyst-override CLI are read/write against local state only.

The `inject-catalyst` and `force-entry` commands are double-gated: `testing.allow_catalyst_overrides` / `testing.allow_force_entry` in [config.yaml](config.yaml) **and** `account.mode == paper`. Live configs must keep both flags off.

## Risk model

The risk engine ([bot/risk/engine.py](bot/risk/engine.py)) is the only path to a fill. Every signal passes through `compute_shares` → halt checks → re-entry checks → margin gate before the executor sees it. The `risk:` block in [config.yaml](config.yaml) is the source of truth; the defaults below are tuned for a small paper account and **must be raised deliberately** before scaling.

Default caps (small-account profile):
- `max_loss_per_trade_usd: 24` — share count = floor(this / (entry − stop))
- `max_position_value_usd: 2400` — hard ceiling on shares × entry
- `max_daily_loss_usd: 72` — halt new entries; in-flight brackets run to their own stops
- `daily_profit_goal_usd: 120` — halt on goal ("walk away when you hit your number")
- `giveback_trigger_usd: 400` / `giveback_pct: 50` — only arms once peak PnL clears the floor
- `max_concurrent_positions: 1`, `max_trades_per_day: 5`
- `max_stop_width_usd: 0.50` — the "if I risk 50¢ I avoid the trade" rule
- `max_pct_of_bar_volume: 2.0` — your order ≪ 2–5% of 1-min bar volume

Re-entries (Phase 4d, `risk.re_entry`): up to 3 per symbol per day, with size multipliers `[1.0, 1.0, 0.5]`, a 120 s cooldown between exits, and a profitable-prior-exit requirement (no revenge trades).

**Halt flag.** When `max_daily_loss_usd`, `daily_profit_goal_usd`, or the give-back guard fires, the engine writes `logs/halt.flag` and refuses new entries. The flag is durable across restarts; clear it with `uv run python -m bot reset-halt --yes` (or wait for the next session — the orchestrator clears stale flags on session rollover).

**PDT.** FINRA abolished the pattern-day-trader designation in the April 2026 rule amendment, so PDT is now **advisory-only**: `DayTradesRemaining` is logged as a `pdt.advisory` event but never blocks. The real intraday gate is the 95% AvailableFunds margin check inside the engine.

### Rehab tier

The rehab engine ([bot/risk/rehab.py](bot/risk/rehab.py)) automatically downsizes after a cold streak — the methodology rule "when you're in a drawdown, trade smaller." Two trigger paths:

| Tier | Trigger | Per-trade cap | Daily loss cap | Trades/day |
| --- | --- | --- | --- | --- |
| BASE | (default) | 100% | 100% | 5 |
| REHAB | 2 consecutive red days **or** cumulative drawdown ≥ 3× daily-loss over a 10-day window | 50% | 50% | 3 |
| DEEP_REHAB | 4 consecutive reds **or** drawdown ≥ 5× daily-loss | 25% | 25% | 1 |

Recovery is hysteretic: the bot must earn back 50% of the drawdown before tiers step back down. Tier state is persisted in `logs/rehab.flag` and survives restarts. Inspect with `uv run python -m bot rehab-status`.

## Order execution

Every entry is an **atomic bracket** placed via `Order.transmit=False` on all legs except the last (per IBKR docs to avoid dangling orders). The bracket has 2–3 legs:

1. **Parent** — `STP_LMT` (Phase 4j default, server-side tick trigger), `LMT` (Phase 8.2 marketable-limit with the buffer chain), or `MKT` (Phase 6.12 manual-hotkey path; uncapped slippage). Selected by `execution.entry_order_type`.
2. **Stop-loss** — full-position STP at the signal's stop level.
3. **Scale-out** — partial-position LMT at `entry + scale_out_multiple × initial_risk` (default 2R).

After the scale-out fills, the trade manager reshapes the remaining stop. Three modes via `execution.post_scaleout_stop_mode`:
- **`immediate_trail`** (Phase 6.14 default) — STP cancelled, IBKR `TRAIL` planted at scale-out price; locks in ≥1R immediately.
- **`adjustable_to_trail`** — flat STP at breakeven until price reaches `scale_out + trail_activation_r_multiple × R`, then converted to TRAIL.
- **`static_breakeven`** — flat STP at entry, no trail.

Runner exits (when no scale-out fires, or after one): close below 9-EMA on a 1-min bar, or an extension-bar spike ≥ `max_loss_per_trade_usd × extension_bar_trigger_multiple`. The pre-scale "first red candle close" rule (Phase 7.8) is wired in but **runtime-disabled by default** after a 2026-04-24 doji false-fire on WLDS — see the comment block in [config.yaml](config.yaml) before re-enabling.

The LMT entry buffer is a clamped chain (Phase 8.2 + 10.6): `entry × lmt_buffer_pct/100`, then floored at `lmt_buffer_usd_floor`, capped at `lmt_buffer_usd_cap`, and finally percentage-ceilinged at `lmt_buffer_max_pct` to stay below IBKR's ~9.8% aggressive-LMT rejection threshold on low-priced stocks.

**Reconciliation on restart.** The executor reads IBKR's open positions and working orders on startup and rehydrates the in-memory `PositionStore`. A crash mid-trade does not produce double opens.

## Phase 10.1 watchdog — flipping shadow_mode off

The naked-position watchdog polls IBKR's working orders every 5 s and classifies each tracked position into `PROTECTED` / `PROTECTED_PENDING` / `UNDERPROTECTED` / `NAKED`. It ships in **shadow mode**: transitions and would-have-alerted events go to the session JSONL, but Telegram is suppressed. After one clean session of shadow logs:

```bash
# Inspect what would have been sent during the session
grep '"event": "watchdog.shadow_alert_skipped"' logs/session_*.jsonl
```

If the result set is empty or exclusively legitimate (real protection gaps, not false positives in the entry-fill grace window), flip the flag in [config.yaml](config.yaml):

```yaml
watchdog:
  shadow_mode: false
```

Subsequent sessions deliver alerts to Telegram with a single "Ack" button. Tapping Ack suppresses re-fires for that `(symbol, classification)` until the position transitions to PROTECTED, the position size changes (partial fill / scale-out re-arms), the trading day rolls over, or the bot restarts.

## Phase 11 exit advisor (disabled in production main)

The exit advisor is a hook-based decision framework that observes every bar close, runs detector layers, and (when enabled) emits `tighten_stop` / `partial_exit` / `full_exit` recommendations. The implementation is complete and tested; production main has `exit_advisor.enabled: false`. The integration is the spike branch.

Detector layers, each independently toggled in `exit_events:` ([config.yaml](config.yaml)):

| Layer | Detectors |
| --- | --- |
| **L1** | time-in-trade milestones, R-multiple PnL milestones, partial-fill / rejection events |
| **L2** | price levels (HOD/LOD, prior-day OHLC, gap fill), moving averages (VWAP, 9-EMA), volume (RVOL milestones, spikes, dry-up), bar shapes (doji, hammer, engulfing, inside/outside) |
| **L2-A** | NASDAQ TotalView L2 — bid/offer pulls, absorption, spread widening/tightening, depth imbalance, print clusters, large prints |
| **L3** | risk-gate framework — hard guardrails (StopProtection, NoReentry, ProtectedPosition, NakedPosition, MaxHoldTime) plus soft gates with confidence thresholding and recency throttling |
| **L4–5** | market-context, news, halts — placeholders, not yet implemented |

Two activation switches: `exit_advisor.enabled` (master) and `exit_advisor.hook_acts` (when false, recommendations are log-only diagnostic). Offline evaluation of historical trades runs through `bot/exit_advisor/replay/harness.py` against the [data/historical_bars/](data/historical_bars/) cache (operator-populated via [scripts/fetch_historical_bars.py](scripts/fetch_historical_bars.py)).

## Backtesting

`ibkr-bot backtest` is a historical replay of the strategies against one ticker on one past date — *not* a full backtester. There is no P&L, no equity curve, no optimization; the command exists solely to confirm that `GapAndGoStrategy` and `MomentumStrategy` fire on known setups.

```bash
uv run python -m bot backtest AMC 2021-06-02 --catalyst
```

**Catalyst caveat.** The live bot's catalyst gate lives in the Phase 2 scanner (Finnhub news, the green/black list), and historical news replay is out of scope here. By default `backtest` runs Momentum only and suppresses Gap-and-Go, since replaying it without a catalyst would be misleading. Pass `--catalyst` to force a synthetic `manual_override` catalyst and run both strategies — use this only when you *know* the target day had a legitimate catalyst. IBKR 1-min historical bars are only available for ~6 months; older dates will be rejected. Signals and rejections are written to `logs/backtest_<symbol>_<date>.jsonl` and `logs/backtest_rejections_<symbol>_<date>.jsonl` for later inspection.

For the offline exit-advisor harness (different code path), see [scripts/run_aggregate_report.py](scripts/run_aggregate_report.py) and [scripts/run_policy_comparison.py](scripts/run_policy_comparison.py).

## Logging, persistence, and reports

- **Session logs** — `logs/session_<YYYY-MM-DD>.jsonl` (NY date), structlog JSON. Single source of truth for replay and post-mortems.
- **Trade journal** — `logs/trades.db` (SQLite). Append-only `TradeRecord` rows with per-leg fills and commissions; consumed by `commissions`, `suggest-caps`, and the rehab engine's drawdown lookback.
- **Halt flag** — `logs/halt.flag`. Empty marker file; presence blocks new entries.
- **Rehab flag** — `logs/rehab.flag`. JSON snapshot of current tier + recovery target.
- **Reports** — `reports/exit_advisor/` holds the closed-trade manifest, multi-trade aggregates, and policy A/B comparisons produced by the harness scripts.
- **Historical bar cache** — `data/historical_bars/` (gitignored except `.gitkeep`). Operator-populated via `scripts/fetch_historical_bars.py --all-trades`; consumed by the exit-advisor replay harness.

After a paper-trading window, check commission cost structure:

```bash
uv run python -m bot commissions --lookback-days 14
```

Prints gross/net PnL, avg commission per trade, commission as a % of gross profit, and the scale-out leg's share of total commission. On paper the absolute dollars are simulated by IBKR, but the ratios are structural if your paper commission tier matches the live tier you'll trade.

## Development

```bash
uv run ruff check --fix
uv run ruff format
uv run mypy bot/
uv run pytest                  # unit tests; integration tests auto-skip if TWS isn't reachable
uv run pytest -m integration   # run only the integration tests (requires paper TWS)
```

Strict mypy is enabled (`disallow_untyped_defs`, `warn_return_any`). The test suite has ~80 files spanning risk, execution, scanner, strategies, all exit-advisor detector layers (heavy L2 coverage), and infrastructure (config, CLI, orchestrator, signal bus, journal, market data).

## ⚠️ Disclaimer

This is a coding project. Automated day trading is extremely risky and most day traders lose money — the published methodology says so in every post. The bot encodes a strategy; it does not guarantee the strategy works for you, in the current market regime, or at all. Paper trade extensively, start live with tiny size, and never risk money you can't afford to lose. This is not financial advice.
