"""XGN 2026-05-11 hold simulation.

Simulates what would have happened if the exit advisor had NOT exited at 09:49
and the trade was managed by the original stop + immediate-trail bracket.
"""

ENTRY       = 3.43
STOP        = 3.30
SCALE_OUT   = 3.69
SHARES      = 184
TRAIL_AMT   = round(ENTRY - STOP, 4)   # 1.0R = $0.13
ACTUAL_EXIT = 3.47
ACTUAL_PNL  = 7.36

bars = [
    ("09:47", 3.32, 3.45, 3.31, 3.42),
    ("09:48", 3.42, 3.54, 3.42, 3.50),
    ("09:49", 3.49, 3.49, 3.42, 3.47),   # LLM exited here
    ("09:50", 3.45, 3.52, 3.38, 3.49),
    ("09:51", 3.51, 3.57, 3.51, 3.52),
    ("09:52", 3.52, 3.60, 3.51, 3.59),
    ("09:53", 3.59, 3.59, 3.55, 3.59),
    ("09:54", 3.58, 3.59, 3.48, 3.49),
    ("09:55", 3.50, 3.52, 3.48, 3.51),
    ("09:56", 3.51, 3.55, 3.47, 3.47),
    ("09:57", 3.49, 3.63, 3.49, 3.61),
    ("09:58", 3.60, 3.62, 3.51, 3.58),
    ("09:59", 3.58, 3.59, 3.57, 3.59),
    ("10:00", 3.58, 3.60, 3.57, 3.60),
    ("10:01", 3.59, 3.80, 3.59, 3.77),   # scale-out fires here
    ("10:02", 3.77, 3.77, 3.661, 3.70),  # trail likely triggers here
    ("10:03", 3.70, 3.71, 3.670, 3.71),
    ("10:04", 3.70, 3.75, 3.692, 3.72),
    ("10:05", 3.72, 3.76, 3.710, 3.74),
    ("10:06", 3.75, 3.75, 3.671, 3.70),
    ("10:10", 3.69, 3.71, 3.640, 3.67),
    ("10:11", 3.67, 3.67, 3.480, 3.53),  # major flush
]

runner_shares = SHARES // 2   # 92
scaled_out    = False
scale_out_pnl = 0.0
runner_pnl    = 0.0
peak          = ENTRY
trail_active  = False
trail_stop    = None
exit_bar      = None

print(f"Entry: ${ENTRY}  Stop: ${STOP}  Scale-out: ${SCALE_OUT}  "
      f"Trail: ${TRAIL_AMT}  Shares: {SHARES}")
print(f"Risk/share: ${TRAIL_AMT:.2f}   1R = ${SHARES * TRAIL_AMT:.2f}")
print()
print(f"{'Bar':<6} {'O':>6} {'H':>6} {'L':>6} {'C':>6}  Note")
print("-" * 80)

for i, (bar_time, o, h, l, c) in enumerate(bars):
    if i == 0:
        print(f"{bar_time:<6} {o:>6.2f} {h:>6.2f} {l:>6.2f} {c:>6.2f}"
              f"  [entry trigger bar; fill ~${ENTRY}]")
        continue

    note = ""

    # --- Pre-scale-out stop check ---
    if not scaled_out and l <= STOP:
        runner_pnl = (STOP - ENTRY) * SHARES
        note = f"STOP @${STOP} fired -- loss ${runner_pnl:.2f}"
        print(f"{bar_time:<6} {o:>6.2f} {h:>6.2f} {l:>6.2f} {c:>6.2f}  {note}")
        exit_bar = bar_time
        break

    # --- Scale-out check ---
    if not scaled_out and h >= SCALE_OUT:
        scaled_out    = True
        scale_out_pnl = runner_shares * (SCALE_OUT - ENTRY)
        trail_active  = True
        if h > peak:
            peak = h
        trail_stop = peak - TRAIL_AMT
        note += (f"SCALE-OUT 92sh @${SCALE_OUT} (+${scale_out_pnl:.2f}) "
                 f"| peak={peak:.2f} trail_stop={trail_stop:.2f}")

    # --- Trail management (post-scale-out) ---
    if trail_active and not exit_bar:
        if h > peak:
            peak       = h
            trail_stop = peak - TRAIL_AMT
        if not note:
            note = f"peak={peak:.2f} trail_stop={trail_stop:.2f}"

        if l <= trail_stop:
            fill       = trail_stop
            runner_pnl = runner_shares * (fill - ENTRY)
            note      += (f" | TRAIL HIT; runner 92sh @${fill:.2f} "
                          f"(+${runner_pnl:.2f})")
            print(f"{bar_time:<6} {o:>6.2f} {h:>6.2f} {l:>6.2f} {c:>6.2f}  {note}")
            exit_bar = bar_time
            break

    # --- Inline annotations ---
    annotations = {
        "09:49": "<-- LLM exited at $3.47 (actual)",
        "09:50": "<-- lowest low after LLM exit ($3.38); stop still at $3.30",
    }
    if bar_time in annotations:
        note = (note + "  " if note else "") + annotations[bar_time]

    print(f"{bar_time:<6} {o:>6.2f} {h:>6.2f} {l:>6.2f} {c:>6.2f}  {note}")

# --- Summary ---
print()
print("=" * 80)
total = scale_out_pnl + runner_pnl
print(f"  Scale-out P&L  : ${scale_out_pnl:>7.2f}  (92 sh x ${SCALE_OUT - ENTRY:.2f})")
print(f"  Runner P&L     : ${runner_pnl:>7.2f}  "
      f"(92 sh x ${runner_pnl/runner_shares:.2f})")
print(f"  HOLD total     : ${total:>7.2f}")
print(f"  Actual (LLM)   : ${ACTUAL_PNL:>7.2f}  (184 sh x ${ACTUAL_EXIT - ENTRY:.2f})")
print(f"  Left on table  : ${total - ACTUAL_PNL:>7.2f}")
print()
pre_exit_lows = [l for t, o, h, l, c in bars[1:15]]
print(f"  Min low 09:48-10:01 : ${min(pre_exit_lows):.2f}  "
      f"(cushion above stop: ${min(pre_exit_lows) - STOP:.2f})")
print(f"  Stop at $3.30 was NEVER in danger during the consolidation.")
