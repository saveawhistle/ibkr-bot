# Policy Comparison Report — WLDS 2026-04-24

Generated: 2026-05-02T23:00:06+00:00

Trade: WLDS, entry $1.1500 × 410 sh at 2026-04-24T15:56:08.160515Z, recorded exit $1.2420, recorded P&L +$37.72.

Pre-trade backfill: 146 bars  |  Trade-window: 0 bars  |  Prior-day cache: hit

## Summary table (gates active)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | +$37.72 | +$0.00 | +$0.00 | 1.2420 | 1/1/0 |  |
| OracleExitPolicy | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Summary table (gates disabled)

| Policy | Effective P&L | vs Actual | vs Oracle | Exit | Decisions (P/A/R) | Open at end? |
| --- | ---: | ---: | ---: | --- | ---: | :---: |
| ActualPolicy | +$37.72 | +$0.00 | +$0.00 | 1.2420 | 1/1/0 |  |
| OracleExitPolicy | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.05) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.1) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.15) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_abs=0.2) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.02) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.04) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| MechanicalTrailPolicy(trail_pct=0.06) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=0.5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=1.5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=2.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| FixedRTakeProfit(target_r=3.0) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.0, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=5) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=1.5, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |
| StallExitPolicy(target_r=2.0, max_minutes=10) | +$37.72 | +$0.00 | +$0.00 | — | 0/0/0 | ✓ |

## Key findings

- **Oracle ceiling (theoretical max):** +$37.72
- **ActualPolicy baseline:** +$37.72
- **Best mechanical with gates active:** `MechanicalTrailPolicy(trail_abs=0.05)` → +$37.72 (vs Actual: +$0.00)
- **Best mechanical without gates:** `MechanicalTrailPolicy(trail_abs=0.05)` → +$37.72 (vs Actual: +$0.00)
- **Gates net impact across mechanicals:** +$0.00 (positive = gates protected on net; negative = gates over-constrained)

## Failure mode analysis

### Failure mode 1 — breakouts that don't reach 2:1

- ActualPolicy outcome: +$37.72
- Best fixed-R / stall policy: `FixedRTakeProfit(target_r=0.5)` → +$37.72 (Δ +$0.00)

### Failure mode 2 — runner exhaustion after scale-out

- ZENA's 2026-04-30 trade did not reach scale-out (recorded exit was a stop), so failure mode 2 is not exercised by this comparison. Multi-trade aggregation or a different sample trade is needed to evaluate runner-management policies.

## Per-policy detail

### ActualPolicy  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$37.72
- Decisions: 1 proposed / 1 accepted / 0 rejected
- Events observed: 243
- Bars consumed: 0

### OracleExitPolicy  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.05)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.1)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.15)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_abs=0.2)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.02)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.04)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### MechanicalTrailPolicy(trail_pct=0.06)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### FixedRTakeProfit(target_r=0.5)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.0)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### FixedRTakeProfit(target_r=1.5)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### FixedRTakeProfit(target_r=2.0)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### FixedRTakeProfit(target_r=3.0)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=5)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### StallExitPolicy(target_r=1.0, max_minutes=10)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=5)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### StallExitPolicy(target_r=1.5, max_minutes=10)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

### StallExitPolicy(target_r=2.0, max_minutes=10)  (gates on)

- Effective P&L: +$37.72
- Final realized P&L: +$0.00
- Position open at end of replay (410 sh); effective P&L is mark-to-market at recorded exit price.
- Decisions: 0 proposed / 0 accepted / 0 rejected
- Events observed: 242
- Bars consumed: 0

