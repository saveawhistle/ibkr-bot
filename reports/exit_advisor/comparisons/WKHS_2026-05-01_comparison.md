# Policy Comparison Report — WKHS 2026-05-01

Generated: 2026-05-02T23:00:07+00:00

Trade: WKHS, entry $3.3600 × 124 sh at 2026-05-01T17:20:02.415320Z, recorded exit $3.2800, recorded P&L -$9.92.

Pre-trade backfill: 230 bars  |  Trade-window: 57 bars  |  Prior-day cache: hit

## Summary table (gates active)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$9.92 | +$0.00 | -$17.36 | 3.2800 | 1/1/0 |  |
| OracleExitPolicy | +$7.44 | +$17.36 | +$0.00 | 3.4200 | 1/1/0 |  |
| MechanicalTrailPolicy(trail_abs=0.05) | +$2.48 | +$12.40 | -$4.96 | 3.3800 | 26/4/22 |  |
| MechanicalTrailPolicy(trail_abs=0.1) | -$3.72 | +$6.20 | -$11.16 | 3.3300 | 4/4/0 |  |
| MechanicalTrailPolicy(trail_abs=0.15) | -$9.92 | +$0.00 | -$17.36 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | +$0.17 | +$10.09 | -$7.27 | 3.3614 | 26/4/22 |  |
| MechanicalTrailPolicy(trail_pct=0.04) | -$9.92 | +$0.00 | -$17.36 | — | 2/2/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | +$4.96 | +$14.88 | -$2.48 | 3.4000 | 1/1/0 |  |
| FixedRTakeProfit(target_r=1.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$0.00 | +$9.92 | -$7.44 | 3.3600 | 1/1/0 |  |
| StallExitPolicy(target_r=1.0, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$0.00 | +$9.92 | -$7.44 | 3.3600 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |
| StallExitPolicy(target_r=2.0, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |

## Summary table (gates disabled)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$9.92 | +$0.00 | -$17.36 | 3.2800 | 1/1/0 |  |
| OracleExitPolicy | +$7.44 | +$17.36 | +$0.00 | 3.4200 | 1/1/0 |  |
| MechanicalTrailPolicy(trail_abs=0.05) | +$2.48 | +$12.40 | -$4.96 | 3.3800 | 6/6/0 |  |
| MechanicalTrailPolicy(trail_abs=0.1) | -$3.72 | +$6.20 | -$11.16 | 3.3300 | 4/4/0 |  |
| MechanicalTrailPolicy(trail_abs=0.15) | -$9.92 | +$0.00 | -$17.36 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | +$0.17 | +$10.09 | -$7.27 | 3.3614 | 6/6/0 |  |
| MechanicalTrailPolicy(trail_pct=0.04) | -$9.92 | +$0.00 | -$17.36 | — | 2/2/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | +$4.96 | +$14.88 | -$2.48 | 3.4000 | 1/1/0 |  |
| FixedRTakeProfit(target_r=1.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$9.92 | +$0.00 | -$17.36 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$0.00 | +$9.92 | -$7.44 | 3.3600 | 1/1/0 |  |
| StallExitPolicy(target_r=1.0, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$0.00 | +$9.92 | -$7.44 | 3.3600 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |
| StallExitPolicy(target_r=2.0, max_minutes=10) | +$1.24 | +$11.16 | -$6.20 | 3.3700 | 1/1/0 |  |

## Key findings

- **Oracle ceiling (theoretical max):** +$7.44
- **ActualPolicy baseline:** -$9.92
- **Best mechanical with gates active:** `FixedRTakeProfit(target_r=0.5)` → +$4.96 (vs Actual: +$14.88)
- **Best mechanical without gates:** `FixedRTakeProfit(target_r=0.5)` → +$4.96 (vs Actual: +$14.88)
- **Gates net impact across mechanicals:** +$0.00 (positive = gates protected on net; negative = gates over-constrained)

## Notable gate rejections

- `MechanicalTrailPolicy(trail_abs=0.05)`: `min_r_for_stop_tighten` rejected 22 decisions
- `MechanicalTrailPolicy(trail_pct=0.02)`: `min_r_for_stop_tighten` rejected 22 decisions

## Failure mode analysis

### Failure mode 1 — breakouts that don't reach 2:1

- ActualPolicy outcome: -$9.92
- Best fixed-R / stall policy: `FixedRTakeProfit(target_r=0.5)` → +$4.96 (Δ +$14.88)

### Failure mode 2 — runner exhaustion after scale-out

- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), so failure mode 2 is not exercised by this comparison. Multi-trade aggregation or a different sample trade is needed to evaluate runner-management policies.

## Per-policy detail

### ActualPolicy  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: -$9.92
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 343
- Bars consumed: 57

### OracleExitPolicy  (gates on)

- Effective P&L: +$7.44
- Final realized P&L: +$7.44
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 302
- Bars consumed: 20

### MechanicalTrailPolicy(trail_abs=0.05)  (gates on)

- Effective P&L: +$2.48
- Final realized P&L: +$2.48
- Decisions: 26 proposed / 4 accepted / 22 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 22
- Events observed: 367
- Bars consumed: 29

### MechanicalTrailPolicy(trail_abs=0.1)  (gates on)

- Effective P&L: -$3.72
- Final realized P&L: -$3.72
- Decisions: 4 proposed / 4 accepted / 0 rejected
- Events observed: 334
- Bars consumed: 46

### MechanicalTrailPolicy(trail_abs=0.15)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 343
- Bars consumed: 57

### MechanicalTrailPolicy(trail_abs=0.2)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### MechanicalTrailPolicy(trail_pct=0.02)  (gates on)

- Effective P&L: +$0.17
- Final realized P&L: +$0.17
- Decisions: 26 proposed / 4 accepted / 22 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 22
- Events observed: 378
- Bars consumed: 46

### MechanicalTrailPolicy(trail_pct=0.04)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 2 proposed / 2 accepted / 0 rejected
- Events observed: 344
- Bars consumed: 57

### MechanicalTrailPolicy(trail_pct=0.06)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### FixedRTakeProfit(target_r=0.5)  (gates on)

- Effective P&L: +$4.96
- Final realized P&L: +$4.96
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 290
- Bars consumed: 12

### FixedRTakeProfit(target_r=1.0)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### FixedRTakeProfit(target_r=1.5)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### FixedRTakeProfit(target_r=2.0)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### FixedRTakeProfit(target_r=3.0)  (gates on)

- Effective P&L: -$9.92
- Final realized P&L: +$0.00
- Position open at end of replay (124 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 342
- Bars consumed: 57

### StallExitPolicy(target_r=1.0, max_minutes=5)  (gates on)

- Effective P&L: +$0.00
- Final realized P&L: +$0.00
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 276
- Bars consumed: 6

### StallExitPolicy(target_r=1.0, max_minutes=10)  (gates on)

- Effective P&L: +$1.24
- Final realized P&L: +$1.24
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 286
- Bars consumed: 11

### StallExitPolicy(target_r=1.5, max_minutes=5)  (gates on)

- Effective P&L: +$0.00
- Final realized P&L: +$0.00
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 276
- Bars consumed: 6

### StallExitPolicy(target_r=1.5, max_minutes=10)  (gates on)

- Effective P&L: +$1.24
- Final realized P&L: +$1.24
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 286
- Bars consumed: 11

### StallExitPolicy(target_r=2.0, max_minutes=10)  (gates on)

- Effective P&L: +$1.24
- Final realized P&L: +$1.24
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 286
- Bars consumed: 11

