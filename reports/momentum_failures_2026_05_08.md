# 2026-05-08 momentum failure analysis

Read-only post-mortem of the three losing momentum trades from 2026-05-08
(AEHL, TRAW, AIIO). Goal: characterize the failure modes with bar-level
evidence so a follow-up PR can add a targeted entry gate based on data,
not speculation.

**Scope**: trade entries; bull-flag heuristic; bar-level OHLCV around
entry. **Out of scope**: PnL, commissions, advisor exit reasoning
(already covered in the daily summary).

---

## TL;DR

All three trades shared **two unanimous structural failures** that the
current `is_bull_flag` heuristic does not check for:

1. **No real impulse.** Each setup's first 3 bars (the "impulse window")
   showed flat or negative price movement. The heuristic uses
   `max(high)` and doesn't require an upward move; a sideways drift
   that pokes a wick still counts.
2. **Loose consolidation range.** All three "consolidation" windows
   spanned **> 4.8% of the impulse high**. The heuristic only checks the
   minimum close vs the impulse high (the pullback %); a noisy,
   wide-range chop with the right minimum-close value passes the same
   as a clean drift.

H3 (no volume contraction) and H4 (extended-from-VWAP entry) matched 2
of 3. H5 (weak breakout candle) matched 2 of 3.

**Recommended primary gate**: combined impulse-strength + consolidation-
tightness check. Both fire unanimously; either alone would have
rejected all three trades. See [Recommended gates](#recommended-gates).

---

## Data sources & gaps

* Source: `logs/session_2026-05-08.jsonl` --- specifically
  `market_data.bar_received` events for OHLCV and
  `bar_aggregator.minute_finalized` events for backfilling sparse
  windows.
* No cached historical bars for these tickers in
  `data/historical_bars/`; would have required an IBKR `reqHistoricalData`
  pull that's out of scope for a read-only analysis.

**AEHL data caveat.** AEHL's `bar_received` log shows an 11-minute gap
from 09:21 ET to 09:33 ET (presumably halt or sparse-trading window
between premarket and RTH open). The 09:32 breakout bar is only present
in `bar_aggregator.minute_finalized` with `contributing=4 of 12` 5-sec
aggregations -- the signal fired ~50 s into the minute on a partial bar.
The breakout-bar OHLC for AEHL is synthesized: `close` and `volume`
come from the aggregator event; `high` is set to the journal's actual
fill price ($1.05); `open`/`low` default to `close` ($1.00). All AEHL
breakout-candle metrics (H5) carry this caveat.

The pre-entry window for AEHL was also sparse: 8 `bar_received` bars
before entry vs the 10 needed for a full bull-flag window. The strategy
itself saw 11 bars (counting partial premarket aggregations not present
in `bar_received`), which is why `is_bull_flag` ran at all. The analysis
uses what's available and treats the consolidation as 5 bars instead of 6.

The **time discontinuity** between AEHL's consolidation (ending 09:21)
and breakout (09:32) is itself a finding -- the bull-flag heuristic
doesn't enforce time-continuity of the consolidation bars. An 11-minute
gap before the breakout means the "flag" is a stale snapshot of where
price sat before the halt.

---

## Per-trade analysis

### AEHL --- entry 09:32:51 ET @ $1.05, exit 09:34:11 ET @ $0.99

**Bars (10-bar window ending at the breakout):**

| Time (ET) | Open | High | Low | Close | Volume | Notes |
|---|---:|---:|---:|---:|---:|---|
| 08:53 | 1.0400 | 1.0400 | 0.9600 | 0.9984 | 1,685,832 | impulse |
| 08:54 | 1.0000 | 1.0000 | 0.9702 | 0.9901 | 568,854 | impulse |
| 08:55 | 0.9920 | 1.0100 | 0.9600 | 0.9889 | 797,098 | impulse |
| 08:56 | 0.9884 | 1.0000 | 0.9859 | 0.9917 | 410,575 | consolidation |
| 09:18 | 0.9897 | 0.9918 | 0.9799 | 0.9888 | 369,797 | consolidation |
| 09:19 | 0.9899 | 1.0300 | 0.9873 | 1.0098 | 699,444 | consolidation |
| 09:20 | 1.0100 | 1.0100 | 0.9850 | 0.9956 | 237,853 | consolidation |
| 09:21 | 0.9960 | 1.0100 | 0.9820 | 0.9931 | 244,637 | consolidation |
| **09:32** | 1.00 | 1.05 | 1.00 | 1.00 | 1,729,468 | **breakout (synthesized; partial bar 4/12)** |
| 09:33 | 1.0000 | 1.0000 | 0.9651 | 0.9900 | 1,725,886 | post-entry (red, immediately) |
| 09:34 | 0.9899 | 0.9925 | 0.9450 | 0.9532 | 1,491,522 | exit fired here |

**Metrics:**

| Group | Metric | Value | Note |
|---|---|---|---|
| Impulse | first_open / last_close | $1.04 / $0.9889 | drifted *down* |
| Impulse | high (max of first 3 bars) | $1.04 | from the very first bar's open wick |
| Impulse | % move | **0.00%** | first bar's high == first bar's open |
| Impulse | slope | **negative** | last_close < first_open |
| Consolidation | range tightness | **4.82%** of impulse high | loose chop |
| Consolidation | pullback % | 4.92% | just under the 5% bull-flag threshold |
| Consolidation | red / green / flat | 3 / 2 / 0 | one-sided |
| Consolidation | closes stdev | 0.0073 | low-but-noisy |
| Consolidation | vol ratio (cons/impulse) | 0.39 | volume *did* contract |
| Consolidation | distribution bars | 0 | no clear distribution |
| Consolidation | lower-wick bars | **2** | rejection at lows |
| Breakout | OHLC | (1.00, 1.05, 1.00, 1.00) | partial-bar; synthesized |
| Breakout | body / range | **0.00** | no body (close == open == low) |
| Breakout | vol vs cons avg | 4.4× | volume surge real |
| Context | day open | $1.00 | RTH opened at 09:32 (post-halt resume) |
| Context | extension from open | 5.0% | minimal |
| Context | premarket high | $1.04 | breakout cleared PMH by 1¢ |
| Context | % above VWAP | **13.94%** | extended |
| Post-entry | peak R after entry | **−0.42** | never traded above entry |
| Post-entry | first red bar after entry | bar 0 | next bar is red |

**Hypothesis matches:**

* **H1 weak/absent impulse**: ✓ (impulse % = 0.00%, slope negative)
* **H2 sloppy consolidation**: ✓ (range tightness 4.82%, 3R/2G one-sided, 2 lower-wick rejection bars)
* H3 no volume contraction: no (vol ratio 0.39, no distribution bars detected)
* **H4 late entry / extended**: ✓ (entry 13.94% above VWAP)
* **H5 weak breakout candle**: ✓ (body/range = 0.00 -- but this is artifact of the synthesized OHLC)

The H5 "weak breakout candle" finding is contaminated by the partial-bar
synthesis. The real signal here was that AEHL was **structurally degenerate
at entry time**: 11-minute trading gap right before the breakout, RTH had
just opened on a halted name, the strategy's "flag" was actually 8 minutes
of pre-halt premarket bars stitched to a fresh halt-resume candle.

### TRAW --- entry 10:34:59 ET @ $2.33, exit 10:37:09 ET @ $2.26

**Bars:**

| Time (ET) | Open | High | Low | Close | Volume | Notes |
|---|---:|---:|---:|---:|---:|---|
| 10:25 | 2.2000 | 2.2200 | 2.1800 | 2.1899 | 91,133 | impulse |
| 10:26 | 2.1900 | 2.2000 | 2.1800 | 2.2000 | 15,262 | impulse |
| 10:27 | 2.2000 | 2.2100 | 2.1900 | 2.2000 | 61,378 | impulse |
| 10:28 | 2.2000 | 2.2099 | 2.1900 | 2.1922 | 34,252 | consolidation |
| 10:29 | 2.1939 | 2.2400 | 2.1939 | 2.2250 | 132,172 | consolidation |
| 10:30 | 2.2300 | 2.2600 | 2.2100 | 2.2552 | 161,452 | consolidation |
| 10:31 | 2.2556 | 2.2800 | 2.2400 | 2.2500 | 362,494 | consolidation (large red on volume) |
| 10:32 | 2.2501 | 2.3000 | 2.2300 | 2.2600 | 301,540 | consolidation |
| 10:33 | 2.2650 | 2.3000 | 2.2550 | 2.2901 | 299,931 | consolidation |
| **10:34** | 2.3000 | 2.3500 | 2.2800 | 2.3250 | **689,364** | **breakout (green, body/range = 0.36)** |
| 10:35 | 2.3200 | 2.3999 | 2.3200 | 2.3750 | 682,964 | post-entry (peaked at +0.50R) |
| 10:36 | 2.3800 | 2.3800 | 2.2350 | 2.2600 | 895,093 | bearish reversal |
| 10:37 | 2.2650 | 2.2900 | 1.9300 | 2.0500 | 1,273,469 | exit fired here |

**Metrics:**

| Group | Metric | Value | Note |
|---|---|---|---|
| Impulse | first_open / last_close | $2.20 / $2.20 | flat |
| Impulse | high | $2.22 | a 1¢ wick |
| Impulse | % move | **0.91%** | weak |
| Impulse | slope | **flat** | |
| Consolidation | range tightness | **4.95%** of impulse high | loose |
| Consolidation | pullback % | 1.25% | shallow (the heuristic's only check) |
| Consolidation | red / green / flat | 2 / 4 / 0 | green-leaning (good) |
| Consolidation | closes stdev | 0.0305 | wide drift |
| Consolidation | vol ratio (cons/impulse) | **3.85×** | volume **increased** during the "flag" |
| Consolidation | distribution bars | 1 | red-on-elevated-volume |
| Breakout | OHLC | (2.30, 2.35, 2.28, 2.325) | real bar |
| Breakout | body / range | **0.36** | more wick than body |
| Breakout | close position in range | 0.64 | upper third (good) |
| Breakout | vol vs cons avg | 3.2× | strong |
| Context | day open | $2.05 | |
| Context | extension from open | 13.66% | normal |
| Context | premarket high | $2.12 | breakout well above PMH |
| Context | % above VWAP | 4.52% | just under H4 threshold |
| Post-entry | peak R after entry | **+0.50** | reached half of scale-out |
| Post-entry | first red bar after entry | bar 1 | next bar after high-water |

**Hypothesis matches:**

* **H1 weak/absent impulse**: ✓ (impulse % = 0.91%, slope flat)
* **H2 sloppy consolidation**: ✓ (range tightness 4.95%)
* **H3 no volume contraction**: ✓ (vol ratio **3.85×** -- volume nearly quadrupled during "consolidation" vs impulse; one distribution bar)
* H4 late entry / extended: no (4.52% above VWAP, just under threshold)
* **H5 weak breakout candle**: ✓ (body/range = 0.36, more wick than body)

The TRAW pattern looked the most like a real flag of the three (clean
green-leaning consolidation, breakout cleared PMH, peaked at +0.5R) but
the volume signature was wrong: instead of seller-exhaustion volume
during the flag, **volume nearly quadrupled during the "consolidation"
relative to the "impulse"**. That's the opposite of what a true bull
flag looks like and matches a distribution-into-strength pattern.

### AIIO --- entry 11:24:02 ET @ $1.15, exit 11:27:10 ET @ $1.09

**Bars:**

| Time (ET) | Open | High | Low | Close | Volume | Notes |
|---|---:|---:|---:|---:|---:|---|
| 11:14 | 1.0980 | 1.1000 | 1.0300 | 1.0601 | 1,451,302 | impulse |
| 11:15 | 1.0697 | 1.0700 | 1.0200 | 1.0700 | 1,464,202 | impulse |
| 11:16 | 1.0700 | 1.0800 | 1.0500 | 1.0750 | 800,744 | impulse |
| 11:17 | 1.0800 | 1.1100 | 1.0500 | 1.0702 | 1,481,436 | consolidation (poked HOD then closed at lows) |
| 11:18 | 1.0734 | 1.0800 | 1.0500 | 1.0500 | 669,486 | consolidation |
| 11:19 | 1.0550 | 1.0600 | 1.0500 | 1.0500 | 282,465 | consolidation |
| 11:20 | 1.0550 | 1.1000 | 1.0500 | 1.0800 | 929,139 | consolidation |
| 11:21 | 1.0790 | 1.0900 | 1.0600 | 1.0850 | 544,042 | consolidation |
| 11:22 | 1.0801 | 1.1050 | 1.0800 | 1.1000 | 1,014,254 | consolidation |
| **11:23** | 1.1000 | 1.1700 | 1.0950 | 1.1450 | **2,929,292** | **breakout** |
| 11:24 | 1.1487 | 1.1800 | 1.1400 | 1.1597 | 2,242,352 | post-entry (peaked at +0.23R) |
| 11:25 | 1.1501 | 1.1800 | 1.1500 | 1.1697 | 1,145,822 | post-entry |
| 11:26 | 1.1600 | 1.1800 | 1.0800 | 1.0900 | 2,321,646 | exit fired here -- 9% intrabar reversal |

**Metrics:**

| Group | Metric | Value | Note |
|---|---|---|---|
| Impulse | first_open / last_close | $1.098 / $1.075 | drifted down |
| Impulse | high | $1.10 | from the very first bar |
| Impulse | % move | **0.18%** | essentially nothing |
| Impulse | slope | **negative** | |
| Consolidation | range tightness | **5.45%** of impulse high | sloppy |
| Consolidation | pullback % | 4.55% | just inside the 5% threshold |
| Consolidation | red / green / flat | 3 / 3 / 0 | balanced (one-sided per H2 rule) |
| Consolidation | closes stdev | 0.018 | 1.65% of impulse high (wide) |
| Consolidation | vol ratio (cons/impulse) | 0.66 | volume contracted |
| Consolidation | distribution bars | 1 | red-on-elevated-volume |
| Breakout | OHLC | (1.10, 1.17, 1.095, 1.145) | real bar, strong shape |
| Breakout | body / range | 0.60 | acceptable |
| Breakout | close position in range | 0.67 | upper third |
| Breakout | vol vs cons avg | 3.6× | strong volume surge |
| Context | day open | $1.14 | |
| Context | extension from open | 0.88% | minimal -- mid-day move |
| Context | premarket high | (no PM bars in log) | unknown |
| Context | % above VWAP | **20.48%** | very extended |
| Post-entry | peak R after entry | **+0.23** | weak follow-through |
| Post-entry | first red bar after entry | bar 2 | held up briefly then reversed |

**Hypothesis matches:**

* **H1 weak/absent impulse**: ✓ (impulse % = 0.18%, slope negative)
* **H2 sloppy consolidation**: ✓ (range tightness 5.45%, balanced R/G, high stdev)
* **H3 no volume contraction**: ✓ (one distribution bar at 11:17 -- close at the lows on 1.48M volume after poking HOD; volume ratio passed but the distribution signal fired)
* **H4 late entry / extended**: ✓ (entry 20.48% above VWAP)
* H5 weak breakout candle: no (clean green breakout candle)

AIIO had the cleanest breakout candle of the three, but the underlying
setup was the worst: no impulse at all (price drifted DOWN through the
"impulse" window), the consolidation was a 5.45%-wide chop, and the
entry sat 20% above VWAP. The 11:17 bar shows the warning sign --
poked HOD at $1.11 and immediately closed at $1.07 on 1.48M volume.

---

## Cross-trade pattern matrix

| Hypothesis | AEHL | TRAW | AIIO | Match count |
|---|:-:|:-:|:-:|:-:|
| **H1 weak/absent impulse** | ✓ | ✓ | ✓ | **3/3** |
| **H2 sloppy consolidation** | ✓ | ✓ | ✓ | **3/3** |
| H3 no volume contraction | — | ✓ | ✓ | 2/3 |
| H4 late entry / extended | ✓ | — | ✓ | 2/3 |
| H5 weak breakout candle | ✓ (caveat) | ✓ | — | 2/3 |

H1 and H2 are unanimous. H3, H4, H5 each fire on 2 of 3.

---

## Dominant failure mode

**The bull-flag heuristic accepted three setups that didn't have an
impulse and weren't tight consolidations.** The current
`is_bull_flag` check (in [bot/indicators.py:181-201](../bot/indicators.py#L181-L201)) only validates two things:

1. The minimum close in the consolidation region sits within
   `max_pullback_pct` (5% by default) of the impulse high.
2. There are at least `lookback` (10) bars to slice.

It does NOT validate:

- That the impulse window represents an actual upward move (a flat or
  down-drifting first 3 bars passes if their `max(high)` is high
  enough).
- That the consolidation range is tight (a 5% pullback PLUS a 5%
  range-spread can pass while looking visually like chop).
- That volume contracted during the consolidation (the canonical
  bull-flag signature).

All three failed trades exhibited the failure that #1 + #2 couldn't
catch. The trades looked like "price-moves-sideways-then-pops" rather
than "price-impulses-up-then-tightens-then-breaks". The exit advisor's
0.82-confidence reversal calls on the bar after entry are the natural
consequence: there was no buying conviction underneath, so a single
distribution candle was sufficient evidence to flip the recommendation.

---

## Recommended gates

Two coordinated additions to `is_bull_flag` (or a new
`is_clean_bull_flag` wrapper that supplements the existing call). Both
gates are unanimous-match with respect to the three trades; either
alone would have rejected all three.

### Gate 1 (primary): impulse strength

Require the impulse window to actually be an impulse:

```python
impulse_pct_move = (impulse_high - impulse_first_open) / impulse_first_open * 100
impulse_slope_positive = impulse_last_close > impulse_first_open * (1 + min_slope_pct / 100)
require: impulse_pct_move >= min_impulse_pct
require: impulse_slope_positive
```

**Suggested thresholds**:

* `min_impulse_pct: 1.5` (% move from first bar's open to impulse high)
* `min_slope_pct: 0.5` (% rise from first bar's open to last bar's close)

The 1.5% threshold is calibrated against the three failures (0.00%,
0.91%, 0.18%) -- all three would fail. A real morning impulse on a
low-float gapper typically prints 5%+ in the first 3 minutes; 1.5% is
the floor.

Pre-existing momentum tests would need either updated impulse data in
their fixtures or an explicit `min_impulse_pct=0.0` opt-out (mirrors
the Phase 12.4 test-disable pattern for `recent_rvol_min`).

### Gate 2 (secondary): consolidation tightness

Require the consolidation range to be tight, not just the pullback:

```python
range_tightness_pct = (cons_high - cons_low) / impulse_high * 100
require: range_tightness_pct <= max_consolidation_range_pct
```

**Suggested threshold**: `max_consolidation_range_pct: 4.0` (% of
impulse high).

All three failed trades exceeded this (4.82%, 4.95%, 5.45%). A real
flag's consolidation typically spans 1-3% of the impulse high; 4% is
generous.

### Gate 3 (tertiary, for consideration): volume contraction

This one is more nuanced and didn't unanimously match -- AEHL's volume
did contract -- but TRAW's 3.85× volume RATIO and the distribution-bar
detection on TRAW + AIIO are real signals. Adding this requires more
calibration data than three trades; flagging for follow-up.

A possible shape:

```python
vol_ratio_cons_to_impulse = cons_avg_volume / impulse_avg_volume
forbid: vol_ratio_cons_to_impulse > max_consolidation_volume_ratio
```

Suggested threshold (provisional): `0.8`. TRAW would clearly fail
(3.85). AEHL would pass (0.39). AIIO would pass (0.66). Combined with
distribution-bar detection (any red consolidation bar with volume >
1.5× consolidation average), TRAW and AIIO would both fail.

---

## Implementation notes (if/when proceeding)

If Gates 1 + 2 are implemented as additions to `is_bull_flag` rather
than as a separate `is_clean_bull_flag` wrapper:

* Existing momentum tests use synthetic 10-bar fixtures (e.g.,
  `tests/test_strategies_momentum.py::test_emits_signal_on_bull_flag_hod_break`)
  that may not satisfy the new thresholds. Likely option: a conftest
  autouse fixture that pins the new thresholds to permissive values
  (mirrors the Phase 12.4 `recent_rvol_enabled` pattern), with the new
  gate tests opting in via marker.
* Strategy ABC's `_reject` already supports the `setup` stage; the
  new gate emits `signal.rejected` with reason `weak_impulse` /
  `loose_consolidation` for forensic grep-ability.
* No schema-breaking changes to `Signal` or `ScanHit`; this is purely
  internal to the strategy's evaluation path.
* All four gates (ruff, mypy, pytest, integration) and existing
  Phase 12.4 + 12.6 invariants must continue to pass.

---

## Caveats

* **n = 3.** Three trades is a tiny sample. A pattern that's unanimous
  here might be coincidence. Before tightening the live gate,
  recommend backtesting Gates 1 + 2 against a wider window (e.g.,
  `--lookback-days 30` of momentum signals from the journal + replay
  harness) to measure how many historical winning trades would also be
  rejected. A gate that improves precision but cuts recall in half
  may not be a net win.
* **AEHL's analysis is partial-data.** The 11-minute pre-entry gap and
  the synthesized breakout-bar OHLC mean H5 (weak breakout candle) for
  AEHL is unreliable. H1 + H2 + H4 are unaffected.
* **The exit advisor's intervention saved meaningful loss.** Without
  the advisor's early-exit on these three, the structural-stop
  drawdowns would have been roughly 2× larger (-$67.96 vs -$35.01;
  see daily summary). The proposed gates would prevent these trades
  entirely -- a stricter form of the same risk reduction the advisor
  is providing today, applied at admission rather than after entry.
* **The recommendation is "add a gate", not "tune the existing 5%
  pullback threshold"**. The 5% pullback check did its job (all three
  trades had pullbacks under 5%); the heuristic is missing dimensions,
  not mis-calibrated on its current dimension.

---

## Reproducibility

The script that produced these metrics:
[`scripts/analyze_2026_05_08_momentum_failures.py`](../scripts/analyze_2026_05_08_momentum_failures.py).

Run with `uv run python scripts/analyze_2026_05_08_momentum_failures.py`
to regenerate the per-trade JSON output. The script is read-only --
no journal mutation, no IBKR calls, only `logs/session_2026-05-08.jsonl`
inputs.
