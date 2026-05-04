# Policy Comparison Report — RMAX 2026-04-27

Generated: 2026-05-02T23:00:06+00:00

Trade: RMAX, entry $9.9000 × 93 sh at 2026-04-27T13:35:06.320577Z, recorded exit $9.6400, recorded P&L -$24.18.

Pre-trade backfill: 5 bars  |  Trade-window: 0 bars  |  Prior-day cache: hit

## Summary table (gates active)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$24.18 | +$0.00 | +$0.00 | 9.6400 | 1/1/0 |  |
| OracleExitPolicy | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | -$24.18 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$24.18 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$24.18 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$24.18 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$24.18 | +$0.00 | +$0.00 | — | 1/0/1 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Summary table (gates disabled)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$24.18 | +$0.00 | +$0.00 | 9.6400 | 1/1/0 |  |
| OracleExitPolicy | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | -$24.18 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$24.18 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$24.18 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$24.18 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$24.18 | +$0.00 | +$0.00 | — | 1/1/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$24.18 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Key findings

- **Oracle ceiling (theoretical max):** -$24.18
- **ActualPolicy baseline:** -$24.18
- **Best mechanical with gates active:** `MechanicalTrailPolicy(trail_abs=0.05)` → -$24.18 (vs Actual: +$0.00)
- **Best mechanical without gates:** `MechanicalTrailPolicy(trail_abs=0.05)` → -$24.18 (vs Actual: +$0.00)
- **Gates net impact across mechanicals:** +$0.00 (positive = gates protected on net; negative = gates over-constrained)

## Failure mode analysis

### Failure mode 1 — breakouts that don't reach 2:1

- ActualPolicy outcome: -$24.18
- Best fixed-R / stall policy: `FixedRTakeProfit(target_r=0.5)` → -$24.18 (Δ +$0.00)

### Failure mode 2 — runner exhaustion after scale-out

- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), so failure mode 2 is not exercised by this comparison. Multi-trade aggregation or a different sample trade is needed to evaluate runner-management policies.

## Per-policy detail

### ActualPolicy  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: -$24.18
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 12
- Bars consumed: 0

### OracleExitPolicy  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.05)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 13
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.1)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 13
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.15)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 13
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.2)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 13
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.02)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 1 proposed / 0 accepted / 1 rejected
- Gate rejections by gate:
  - `min_r_for_stop_tighten`: 1
- Events observed: 13
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.04)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.06)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### FixedRTakeProfit(target_r=0.5)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.0)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.5)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### FixedRTakeProfit(target_r=2.0)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### FixedRTakeProfit(target_r=3.0)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=5)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=10)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=5)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=10)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

### StallExitPolicy(target_r=2.0, max_minutes=10)  (gates on)

- Effective P&L: -$24.18
- Final realized P&L: +$0.00
- Position open at end of replay (93 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 11
- Bars consumed: 0

