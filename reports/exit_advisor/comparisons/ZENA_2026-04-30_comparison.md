# Policy Comparison Report — ZENA 2026-04-30

Generated: 2026-05-02T23:00:06+00:00

Trade: ZENA, entry $2.1800 × 119 sh at 2026-04-30T15:02:57.479086Z, recorded exit $2.1600, recorded P&L -$2.38.

Pre-trade backfill: 92 bars  |  Trade-window: 1 bars  |  Prior-day cache: hit

## Summary table (gates active)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$2.38 | +$0.00 | -$1.78 | 2.1600 | 1/1/0 |  |
| OracleExitPolicy | -$0.60 | +$1.78 | +$0.00 | 2.1750 | 1/1/0 |  |
| MechanicalTrailPolicy(trail_abs=0.05) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |

## Summary table (gates disabled)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | -$2.38 | +$0.00 | -$1.78 | 2.1600 | 1/1/0 |  |
| OracleExitPolicy | -$0.60 | +$1.78 | +$0.00 | 2.1750 | 1/1/0 |  |
| MechanicalTrailPolicy(trail_abs=0.05) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | -$2.38 | +$0.00 | -$1.78 | — | 0/0/0 | ✓ |

## Key findings

- **Oracle ceiling (theoretical max):** -$0.60
- **ActualPolicy baseline:** -$2.38
- **Best mechanical with gates active:** `MechanicalTrailPolicy(trail_abs=0.05)` → -$2.38 (vs Actual: +$0.00)
- **Best mechanical without gates:** `MechanicalTrailPolicy(trail_abs=0.05)` → -$2.38 (vs Actual: +$0.00)
- **Gates net impact across mechanicals:** +$0.00 (positive = gates protected on net; negative = gates over-constrained)

## Failure mode analysis

### Failure mode 1 — breakouts that don't reach 2:1

- ActualPolicy outcome: -$2.38
- Best fixed-R / stall policy: `FixedRTakeProfit(target_r=0.5)` → -$2.38 (Δ +$0.00)

### Failure mode 2 — runner exhaustion after scale-out

- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), so failure mode 2 is not exercised by this comparison. Multi-trade aggregation or a different sample trade is needed to evaluate runner-management policies.

## Per-policy detail

### ActualPolicy  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: -$2.38
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 112
- Bars consumed: 1

### OracleExitPolicy  (gates on)

- Effective P&L: -$0.60
- Final realized P&L: -$0.60
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_abs=0.05)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_abs=0.1)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_abs=0.15)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_abs=0.2)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_pct=0.02)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_pct=0.04)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### MechanicalTrailPolicy(trail_pct=0.06)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### FixedRTakeProfit(target_r=0.5)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### FixedRTakeProfit(target_r=1.0)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### FixedRTakeProfit(target_r=1.5)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### FixedRTakeProfit(target_r=2.0)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### FixedRTakeProfit(target_r=3.0)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### StallExitPolicy(target_r=1.0, max_minutes=5)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### StallExitPolicy(target_r=1.0, max_minutes=10)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### StallExitPolicy(target_r=1.5, max_minutes=5)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### StallExitPolicy(target_r=1.5, max_minutes=10)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

### StallExitPolicy(target_r=2.0, max_minutes=10)  (gates on)

- Effective P&L: -$2.38
- Final realized P&L: +$0.00
- Position open at end of replay (119 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 111
- Bars consumed: 1

