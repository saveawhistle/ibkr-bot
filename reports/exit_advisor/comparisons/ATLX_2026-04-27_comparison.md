# Policy Comparison Report — ATLX 2026-04-27

Generated: 2026-05-02T23:00:06+00:00

Trade: ATLX, entry $4.8700 × 314 sh at 2026-04-27T14:53:38.247569Z, recorded exit $4.8100, recorded P&L -$18.84.

Pre-trade backfill: 83 bars  |  Trade-window: 0 bars  |  Prior-day cache: hit

## Summary table (gates active)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$18.84 | +$0.00 | +$0.00 | 4.8100 | 1/1/0 |  |
| OracleExitPolicy | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | -$18.84 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$18.84 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$18.84 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$0.00 | +$18.84 | +$18.84 | 4.8700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$0.00 | +$18.84 | +$18.84 | 4.8700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Summary table (gates disabled)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$18.84 | +$0.00 | +$0.00 | 4.8100 | 1/1/0 |  |
| OracleExitPolicy | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | -$18.84 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$18.84 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$18.84 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$0.00 | +$18.84 | +$18.84 | 4.8700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$0.00 | +$18.84 | +$18.84 | 4.8700 | 1/1/0 |  |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$18.84 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Key findings

- **Oracle ceiling (theoretical max):** -$18.84
- **ActualPolicy baseline:** -$18.84
- **Best mechanical with gates active:** `StallExitPolicy(target_r=1.0, max_minutes=5)` → +$0.00 (vs Actual: +$18.84)
- **Best mechanical without gates:** `StallExitPolicy(target_r=1.0, max_minutes=5)` → +$0.00 (vs Actual: +$18.84)
- **Gates net impact across mechanicals:** +$0.00 (positive = gates protected on net; negative = gates over-constrained)

## Failure mode analysis

### Failure mode 1 — breakouts that don't reach 2:1

- ActualPolicy outcome: -$18.84
- Best fixed-R / stall policy: `StallExitPolicy(target_r=1.0, max_minutes=5)` → +$0.00 (Δ +$18.84)

### Failure mode 2 — runner exhaustion after scale-out

- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), so failure mode 2 is not exercised by this comparison. Multi-trade aggregation or a different sample trade is needed to evaluate runner-management policies.

## Per-policy detail

### ActualPolicy  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: -$18.84
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 120
- Bars consumed: 0

### OracleExitPolicy  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.05)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 121
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.1)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 121
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.15)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.2)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.02)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 121
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.04)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.06)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### FixedRTakeProfit(target_r=0.5)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.0)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.5)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### FixedRTakeProfit(target_r=2.0)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### FixedRTakeProfit(target_r=3.0)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=5)  (gates on)

- Effective P&L: +$0.00
- Final realized P&L: +$0.00
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 120
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=10)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=5)  (gates on)

- Effective P&L: +$0.00
- Final realized P&L: +$0.00
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 120
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=10)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

### StallExitPolicy(target_r=2.0, max_minutes=10)  (gates on)

- Effective P&L: -$18.84
- Final realized P&L: +$0.00
- Position open at end of replay (314 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 119
- Bars consumed: 0

