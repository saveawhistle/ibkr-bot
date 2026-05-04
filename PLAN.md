# Automated Day Trader — Project Plan

## 1. Vision

Build a fully automated Python trading bot that executes the **Gap and Go** and **Momentum (Bull Flag / HOD Breakout)** strategies against an Interactive Brokers Pro margin account via the TWS API. The bot runs during premarket and the first two trading hours (the window where the strategy's edge is strongest), manages its own risk with bracket orders, and enforces hard daily stop-loss / profit-goal rules.

**Explicit non-goals for v1:**
- No reversal, short-selling, or afternoon trading strategies
- No options or futures
- No ML / predictive models — this is rule-based execution of a known discretionary strategy
- No custom GUI (CLI + logs + optional web dashboard later)

---

## 2. Strategy Rules (Encoded from the Published Methodology)

### 2.1 Universe & Hard Filters (the "5 Pillars")

A ticker is only tradable if **all** of these pass:

| Filter | Threshold | Source |
|---|---|---|
| Price | `$1 ≤ price ≤ $20` (sweet spot `$5–$10`) | the published "5 Pillars" |
| % change (gap) | `≥ 10%` preferred, `≥ 4%` minimum | "Gaps over 4% are good for Gap and Go, under 4% tend to fill" |
| Relative volume (RVOL) | `≥ 5×` vs 30/60-day avg | 5 Pillars |
| Float | `≤ 20M` preferred, `≤ 10M` ideal, sub-5M is "rocket fuel" | 5 Pillars |
| News catalyst | Must have one (earnings beat, FDA, contract, partnership, etc.) | 5 Pillars |
| Premarket volume (liquidity guardrail) | `≥ 300k` shares | the minimum for fills |
| Spread | `≤ 1%` of price (or ≤ $0.05 under $10) | Liquidity guardrail |

### 2.2 Strategy 1 — Gap and Go

**Setup:** Stock gaps up ≥4% on news with premarket volume; rides momentum at the open.

**Entry triggers (any one):**
- Break of premarket high on the 1-minute chart with volume confirmation
- First pullback after the open forms a bull flag → entry on breakout of flag high
- Stock holds above VWAP after open

**Trade window:** 9:30 – 11:30 ET (the "sweet spot" is actually 9:30–10:00)

### 2.3 Strategy 2 — Momentum (Bull Flag / HOD Breakout)

**Setup:** Strong stock on HOD scanner, making a new high of day with volume; pulls back on low volume (the "flag"); breaks out on high volume.

**Entry triggers:**
- Bull flag: pullback of 2–5 red candles on 1-min chart, breakout of flag high with RVOL spike
- HOD breakout: new high of day with volume ≥ 2× recent avg
- Price holding above 9-EMA on 1-min chart

**Trade window:** 9:30 – 11:30 ET on 1-min chart; after 11:30 switch to 5-min or stop trading.

### 2.4 Exits (shared across both strategies)

- **Stop loss:** Below the low of the pullback / consolidation, sized so the trade has a **minimum 2:1 profit:loss ratio**
- **Scale out #1:** Sell 50% at first profit target (hit +$X when risking $X/2 = 2:1)
- **Move stop to breakeven** on remaining 50% after scale-out
- **Scale out #2:** Exit remainder on (a) first red candle to close on 1-min, (b) extension bar spike, or (c) break of 9-EMA

### 2.5 Risk & Session Rules (hard-coded, non-negotiable)

- **Per-trade risk:** Configurable, default **1% of account equity**
- **Max daily loss:** Configurable, default **3% of account equity** → bot halts for the day and flattens
- **Daily profit goal:** Configurable → on hit, bot halts for the day (the methodology: "When I hit my goal, I pack up and walk away")
- **Give-back rule:** If up ≥ threshold and give back 50% of day's gains → halt
- **Max concurrent positions:** 1 (the methodology recommends one name at a time)
- **Max trades per day:** 5 (prevents revenge trading)
- **PDT awareness:** Enforce pattern-day-trader rules — require account equity > $25,000, count day trades in rolling 5-day window
- **All positions flat by 15:55 ET** no matter what

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    orchestrator (main loop)                 │
│  ─ scheduler (premarket → open → trading → flatten → EOD)  │
└──────────┬──────────────────────────────────────────────────┘
           │
   ┌───────┼───────┬────────────┬───────────┬──────────────┐
   ▼       ▼       ▼            ▼           ▼              ▼
┌──────┐ ┌─────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│ IBKR │ │News │ │ Scanner  │ │Strategy│ │   Risk   │ │Executor │
│Client│ │Feed │ │(5 Pillar)│ │ Engine │ │  Engine  │ │(Orders) │
└──────┘ └─────┘ └──────────┘ └────────┘ └──────────┘ └─────────┘
   │        │         │           │           │            │
   └────────┴─────────┴───────────┴───────────┴────────────┘
                         │
                         ▼
                   ┌──────────┐
                   │  State   │  (SQLite: trades, positions, P&L, logs)
                   └──────────┘
```

### Module breakdown

| Module | Responsibility |
|---|---|
| `ibkr_client.py` | Wraps `ib_async`. Manages TWS/Gateway connection, reconnection, contract qualification, order placement. |
| `market_data.py` | Real-time quotes, historical bars (1-min for strategy, 5-min fallback), VWAP calc, 9-EMA calc. |
| `scanner.py` | Runs IBKR `reqScannerSubscription` with `TOP_PERC_GAIN` + custom filter tags (`usdMarketCapAbove`, `changePercAbove`, `priceAbove/Below`, `volumeAbove`). Post-filters for float and news. |
| `news.py` | Pulls headlines via IBKR `reqNewsArticle` / `reqHistoricalNews` (Briefing, DJ, Reuters) to confirm catalyst. |
| `strategies/gap_and_go.py` | Detects Gap & Go setups, emits `Signal(entry, stop, target, size)`. |
| `strategies/momentum.py` | Detects bull flag + HOD breakout setups on 1-min bars. |
| `risk.py` | Position sizing (1% rule), daily loss/goal tracking, PDT counter, kill-switch. |
| `executor.py` | Converts `Signal` → IBKR **bracket order** (parent limit + stop-loss + profit-taker). Handles fills, partial exits, trailing moves. |
| `state.py` | SQLite persistence. Trade journal, daily stats, PDT day-trade ledger. |
| `config.py` | Pydantic settings. Reads `.env` + `config.yaml`. |
| `orchestrator.py` | Event loop. Routes scanner hits → strategies → risk → executor. Handles session transitions. |
| `cli.py` | `python -m bot run|paper|backtest|status|flatten` |
| `tests/` | Unit tests with mocked IBKR, integration tests against paper account. |

### Key technical decisions

- **`ib_async` over native `ibapi`** — actively maintained successor to `ib_insync` (original creator passed in 2024), asyncio-native, much cleaner API, same binary protocol under the hood. Not officially supported by IBKR but widely used.
- **IB Gateway, not TWS**, for production (lighter weight, no GUI, auto-restart friendly). TWS for dev so you can see orders.
- **Paper account first.** IBKR paper accounts mirror the real API. Zero real money until every rule has run for 10+ sessions clean.
- **Bracket orders for every entry** — parent + stop + profit-taker atomically via `Order.transmit=False` on parent and takeProfit, `True` only on the final leg (required per IBKR docs to avoid dangling orders).
- **SQLite, not Postgres** — single-user local app, keep it simple.
- **APScheduler** for the session state machine (premarket scan → open → trading → flatten → EOD report).
- **structlog** for JSON-structured logs (critical for debugging a live trading bot).

---

## 4. Development Phases

### Phase 0 — Prerequisites (you, not Claude)
- [ ] Enable API in TWS: Global Config → API → Settings → "Enable ActiveX and Socket Clients"
- [ ] Open a paper trading account (Account menu → paper trading)
- [ ] Subscribe to required market data (US Securities Snapshot / Bundle — needed for scanner + real-time quotes)
- [ ] Install IB Gateway (for prod) and TWS (for dev)
- [ ] Confirm account equity > $25k for PDT compliance

### Phase 1 — Skeleton + IBKR Connectivity (Claude Code, day 1)
- Project structure, dependencies, config loader
- `ibkr_client.py` connects to paper TWS, qualifies a contract, prints account summary
- `tests/test_ibkr_connection.py` passes

### Phase 2 — Scanner + News (day 2)
- Premarket scanner returns ranked list filtered through the 5 Pillars
- News module confirms catalyst
- `python -m bot scan` prints the morning watchlist

### Phase 3 — Strategy engines (days 3–4)
- Bull flag detector on 1-min bars, unit tested against historical bars for known patterns
- Gap-and-Go setup detector
- Both emit `Signal` objects but **do not place orders yet**

### Phase 4 — Risk + Executor (day 5)
- Position sizer, PDT tracker, daily halt
- Bracket order placement against paper account
- End-to-end: scanner → signal → risk check → bracket order → fill → exit

### Phase 5 — Orchestrator + Session Management (day 6)
- APScheduler state machine
- Auto-flatten at 15:55 ET
- Daily report generation

### Phase 6 — Paper Trading (2+ weeks, real time)
- Run live in paper account every morning
- Review every trade vs. what the methodology would have done
- Tune thresholds

### Phase 7 — Live (only after clean paper record)
- Start with minimum position size (1 share, or $100 max risk)
- Scale gradually

---

## 5. Tech Stack

```
python         >= 3.11
ib_async       >= 2.0     # IBKR API wrapper
pandas         >= 2.0     # bar data manipulation
numpy          >= 1.26
pydantic       >= 2.0     # config + signal validation
pydantic-settings          # .env loading
apscheduler    >= 3.10    # session state machine
structlog      >= 24.0    # structured logging
sqlalchemy     >= 2.0     # SQLite ORM
pytest         >= 8.0     # testing
pytest-asyncio             # async tests
ruff                       # lint + format
mypy                       # type checking
python-dotenv
```

---

## 6. Repo Layout

```
ibkr-bot/
├── README.md
├── PLAN.md                 # this file
├── pyproject.toml
├── .env.example
├── config.yaml
├── bot/
│   ├── __init__.py
│   ├── __main__.py         # python -m bot
│   ├── cli.py
│   ├── config.py
│   ├── ibkr_client.py
│   ├── market_data.py
│   ├── scanner.py
│   ├── news.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py         # Strategy ABC + Signal dataclass
│   │   ├── gap_and_go.py
│   │   └── momentum.py
│   ├── risk.py
│   ├── executor.py
│   ├── state.py
│   ├── orchestrator.py
│   └── indicators.py       # VWAP, EMA, ATR
├── tests/
│   ├── conftest.py
│   ├── test_scanner.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   ├── test_executor.py
│   └── fixtures/
│       └── bars/           # historical 1-min bars for backtesting setups
└── logs/                   # gitignored
```

---

## 7. Configuration (config.yaml example)

```yaml
account:
  mode: paper              # paper | live

risk:
  per_trade_pct: 1.0
  daily_loss_pct: 3.0
  daily_profit_goal_usd: 500
  giveback_pct: 50
  max_positions: 1
  max_trades_per_day: 5

session:
  timezone: America/New_York
  premarket_scan_start: "07:00"
  trading_start: "09:30"
  trading_end: "11:30"
  flatten_all: "15:55"

universe:
  price_min: 1.0
  price_max: 20.0
  float_max: 20_000_000
  gap_pct_min: 4.0
  rvol_min: 5.0
  premarket_vol_min: 300_000

strategies:
  gap_and_go:
    enabled: true
  momentum:
    enabled: true
    flag_max_pullback_pct: 5.0

ibkr:
  host: 127.0.0.1
  port: 7497               # 7497 TWS paper, 7496 TWS live, 4002 GW paper, 4001 GW live
  client_id: 17
```

---

## 8. Risks & Open Questions

**Known gotchas:**
- IBKR scanner API is limited to 10 active scans and 50 results per scan code. Should be plenty.
- IBKR scanner does **not** include float directly — will need to cross-reference via `reqFundamentalData` or a secondary source (Finviz scrape is a common workaround; user should confirm comfort with this).
- Scanner refresh rate is throttled server-side; cannot be increased. Fine for a strategy that trades at the open.
- News API requires separate IBKR subscriptions (Briefing Trader, DJ News) — ~$10–20/month each.
- `ib_async` is community-maintained; IBKR will not provide support if something breaks.
- Simulated stop orders (IBKR doesn't always route stops to the exchange) can be subject to slippage on gaps. Consider stop-limit with a wider limit for production.

**Questions to resolve before/during build:**
1. Float data source — IBKR fundamentals, or third-party (Finviz, Polygon, etc.)?
2. Do you want the bot to place **market** or **limit** entry orders? the methodology enters with marketable limits.
3. Should the bot send trade notifications (Discord/Telegram/SMS) on fills?
4. Backtesting — build a simple event-driven backtester against IBKR historical 1-min bars, or defer to a dedicated library (vectorbt, backtrader)?

---

## 9. Success Criteria for v1

- ✅ Connects to IBKR paper account and stays connected across a full session
- ✅ Morning scan produces the same watchlist a human applying the 5 Pillars would produce (verified by spot-check against a known reference watchlist for 5 consecutive sessions)
- ✅ All entries use bracket orders; no naked positions ever
- ✅ Hard risk rules (daily loss, PDT, max positions) cannot be violated even with a code bug — enforced at the executor level
- ✅ Every trade is logged with full context (scanner snapshot, signal reason, R:R, outcome)
- ✅ 2+ weeks of clean paper trading before a single real dollar

---

## ⚠️ Disclaimer

This is a coding project. Automated day trading is extremely risky and most day traders lose money — the published methodology says so in every post. The bot encodes a strategy; it does not guarantee the strategy works for you, in the current market regime, or at all. Paper trade extensively, start live with tiny size, and never risk money you can't afford to lose. This plan is not financial advice.
