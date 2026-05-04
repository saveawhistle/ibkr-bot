# Closed-Trade Aggregate Comparison Report — 2026-05-02

Generated: 2026-05-02T23:00:16+00:00

- Trades discovered: 6
- Trades successfully replayed: 5
- Trades excluded (1): missing detail files — SBLX 2026-04-28

## Dataset summary

- Date range: 2026-04-24 to 2026-05-01
- Symbols: ATLX, RMAX, WKHS, WLDS, ZENA
- Total ActualPolicy P&L: -$17.60
- Total OracleExitPolicy P&L (theoretical max): +$1.54
- Theoretical room for improvement: +$19.14

## Failure mode distribution

| Mode | Count | % | Total Actual | Total Oracle | Total room |
|------|------:|--:|-------------:|-------------:|-----------:|
| mode_1_flagging_breakout | 0 | 0% | +$0.00 | +$0.00 | +$0.00 |
| mode_2_runner_exhaustion | 0 | 0% | +$0.00 | +$0.00 | +$0.00 |
| degenerate | 4 | 80% | -$7.68 | -$5.90 | +$1.78 |
| stop_out | 1 | 20% | -$9.92 | +$7.44 | +$17.36 |
| successful_runner | 0 | 0% | +$0.00 | +$0.00 | +$0.00 |
| unclassified | 0 | 0% | +$0.00 | +$0.00 | +$0.00 |

## Per-policy aggregate (gates active)

| Policy(params) | Trades fired | Mean P&L when fired | Total P&L | Δ vs Actual | Δ vs Oracle |
|----------------|-------------:|--------------------:|----------:|------------:|------------:|
| StallExitPolicy(target_r=1.0, max_minutes=5) | 2/5 | +$0.00 | +$11.16 | +$28.76 | +$9.62 |
| StallExitPolicy(target_r=1.5, max_minutes=5) | 2/5 | +$0.00 | +$11.16 | +$28.76 | +$9.62 |
| FixedRTakeProfit(target_r=0.5) | 1/5 | +$4.96 | -$2.72 | +$14.88 | -$4.26 |
| MechanicalTrailPolicy(trail_abs=0.05) | 3/5 | -$13.51 | -$5.20 | +$12.40 | -$6.74 |
| StallExitPolicy(target_r=1.0, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| StallExitPolicy(target_r=1.5, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| StallExitPolicy(target_r=2.0, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| MechanicalTrailPolicy(trail_pct=0.02) | 3/5 | -$14.28 | -$7.51 | +$10.09 | -$9.05 |
| MechanicalTrailPolicy(trail_abs=0.1) | 3/5 | -$15.58 | -$11.40 | +$6.20 | -$12.94 |
| MechanicalTrailPolicy(trail_abs=0.15) | 2/5 | -$17.05 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_abs=0.2) | 1/5 | -$24.18 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_pct=0.04) | 1/5 | -$9.92 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_pct=0.06) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=1.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=1.5) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=2.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=3.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |

## Per-policy aggregate (gates disabled)

| Policy(params) | Trades fired | Mean P&L when fired | Total P&L | Δ vs Actual | Δ vs Oracle |
|----------------|-------------:|--------------------:|----------:|------------:|------------:|
| StallExitPolicy(target_r=1.0, max_minutes=5) | 2/5 | +$0.00 | +$11.16 | +$28.76 | +$9.62 |
| StallExitPolicy(target_r=1.5, max_minutes=5) | 2/5 | +$0.00 | +$11.16 | +$28.76 | +$9.62 |
| FixedRTakeProfit(target_r=0.5) | 1/5 | +$4.96 | -$2.72 | +$14.88 | -$4.26 |
| MechanicalTrailPolicy(trail_abs=0.05) | 3/5 | -$13.51 | -$5.20 | +$12.40 | -$6.74 |
| StallExitPolicy(target_r=1.0, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| StallExitPolicy(target_r=1.5, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| StallExitPolicy(target_r=2.0, max_minutes=10) | 1/5 | +$1.24 | -$6.44 | +$11.16 | -$7.98 |
| MechanicalTrailPolicy(trail_pct=0.02) | 3/5 | -$14.28 | -$7.51 | +$10.09 | -$9.05 |
| MechanicalTrailPolicy(trail_abs=0.1) | 3/5 | -$15.58 | -$11.40 | +$6.20 | -$12.94 |
| MechanicalTrailPolicy(trail_abs=0.15) | 2/5 | -$17.05 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_abs=0.2) | 1/5 | -$24.18 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_pct=0.04) | 1/5 | -$9.92 | -$17.60 | +$0.00 | -$19.14 |
| MechanicalTrailPolicy(trail_pct=0.06) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=1.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=1.5) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=2.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |
| FixedRTakeProfit(target_r=3.0) | 0/5 | — | -$17.60 | +$0.00 | -$19.14 |

## Mode-specific policy ranking

### mode_1_flagging_breakout subset (N = 0)

Insufficient data for statistical comparison.

### mode_2_runner_exhaustion subset (N = 0)

Insufficient data for statistical comparison.

### degenerate subset (N = 4)

- Best: `StallExitPolicy(target_r=1.0, max_minutes=5)` — +$18.84 vs Actual
- Worst: `StallExitPolicy(target_r=2.0, max_minutes=10)` — +$0.00 vs Actual

### stop_out subset (N = 1)

Insufficient data for statistical comparison.

### successful_runner subset (N = 0)

Insufficient data for statistical comparison.

### unclassified subset (N = 0)

Insufficient data for statistical comparison.

## Gate impact summary

| Gate | Total rejections |
|------|----------------:|
| `MechanicalTrailPolicy` | 52 |

## Notable findings

- **degenerate** is the dominant mode (4/5 = 80% of dataset).
- Theoretical maximum improvement (Oracle - Actual) across the dataset: **+$19.14**.
- Best mechanical with gates active: `StallExitPolicy(target_r=1.0, max_minutes=5)` — total Δ vs Actual +$28.76 (fired on 2/5 trades).
- **4 trades classified DEGENERATE** (sub-5-minute or sub-5-bar). These tell us little about exit policy quality — they reflect entry-side timing or stop-out luck.
- **No MODE_2_RUNNER_EXHAUSTION trades in the dataset.** Either the bot's trail is already managing runners well, or no trade reached scale-out with material runner gain — likely the latter given the dataset size.

## Recommendations for layer 5

- MODE_2: zero trades in this dataset. Either the bot rarely reaches scale-out, or runner gains are quickly captured. Layer 5's design shouldn't prioritize MODE_2 until it shows up.
- DEGENERATE (4/5): a substantial fraction of trades are too short for exit-side policy work. Improvement here likely lives upstream (entry timing, pre-protection sequence) rather than in any exit advisor — mechanical OR agent-driven.

