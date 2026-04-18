"""
analyze_pnl.py  (Round 2)
--------------------------
Decomposes PnL from a prosperity4btx backtest log.

For ASH:
  - Spread capture rate (PnL per tick)
  - Position distribution (neutral vs extreme)
  - Time spent far from fair value (= best opportunity for a MM)
  - Estimated adverse selection: ticks where we were on wrong side of mean-reversion

For PEPPER:
  - Theoretical max PnL (80 units × full trend move from tick 0)
  - Lag loss: PnL lost because position < 80 during buildup
  - Position build time
  - Trend capture efficiency (actual / theoretical)

Usage:
    python analyze_pnl.py <log_file>
    python analyze_pnl.py ../../backtests/2026-04-18_run.log
"""

import os
import sys

PEPPER = "INTARIAN_PEPPER_ROOT"
ASH    = "ASH_COATED_OSMIUM"

PEPPER_POSITION_LIMIT = 80
ASH_POSITION_LIMIT    = 80
ASH_FAIR              = 10_000.0


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log(path):
    rows = []
    in_activities = False
    header_seen   = False

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == "Activities log:":
                in_activities = True
                continue
            if not in_activities:
                continue
            if line.startswith("Trade History:"):
                break
            if not header_seen:
                header_seen = True
                continue
            parts = line.split(";")
            if len(parts) < 17:
                continue
            try:
                rows.append({
                    "day":     int(parts[0]),
                    "ts":      int(parts[1]),
                    "product": parts[2],
                    "bid1":    float(parts[3])  if parts[3]  else None,
                    "bvol1":   int(parts[4])    if parts[4]  else None,
                    "ask1":    float(parts[9])  if parts[9]  else None,
                    "avol1":   int(parts[10])   if parts[10] else None,
                    "mid":     float(parts[15]) if parts[15] else None,
                    "pnl":     float(parts[16]) if parts[16] else 0.0,
                })
            except (ValueError, IndexError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Position reconstruction
# ---------------------------------------------------------------------------

def reconstruct_position(rows, product, day):
    subset = [r for r in rows if r["product"] == product and r["day"] == day]
    subset.sort(key=lambda r: r["ts"])
    pos = 0.0
    result = []
    for i in range(len(subset) - 1):
        curr = subset[i]
        nxt  = subset[i + 1]
        mid_chg = ((nxt["mid"] or 0) - (curr["mid"] or 0)) if (curr["mid"] and nxt["mid"]) else 0
        pnl_chg = nxt["pnl"] - curr["pnl"]
        if abs(mid_chg) > 0.1:
            pos = max(-80, min(80, round(pnl_chg / mid_chg)))
        result.append((curr["ts"], curr["mid"], curr["pnl"], pos))
    if subset:
        result.append((subset[-1]["ts"], subset[-1]["mid"], subset[-1]["pnl"], pos))
    return result


# ---------------------------------------------------------------------------
# ASH analysis
# ---------------------------------------------------------------------------

def analyze_ash(rows, day):
    subset = [r for r in rows if r["product"] == ASH and r["day"] == day]
    if not subset:
        print(f"  No ASH data for day {day}")
        return
    subset.sort(key=lambda r: r["ts"])

    actual_pnl = subset[-1]["pnl"]
    pos_series = reconstruct_position(rows, ASH, day)
    positions  = [p for _, _, _, p in pos_series]
    total_ticks = len(positions)

    # Position distribution
    buckets = {"neutral [0,10)": 0, "moderate [10,30)": 0,
               "elevated [30,60)": 0, "extreme [60,80]": 0}
    for p in positions:
        ap = abs(p)
        if ap < 10:   buckets["neutral [0,10)"] += 1
        elif ap < 30: buckets["moderate [10,30)"] += 1
        elif ap < 60: buckets["elevated [30,60)"] += 1
        else:         buckets["extreme [60,80]"] += 1

    mids = [r["mid"] for r in subset if r["mid"] is not None]
    mid_min = min(mids) if mids else 0
    mid_max = max(mids) if mids else 0

    ticks_far      = sum(1 for m in mids if abs(m - ASH_FAIR) > 5)
    ticks_very_far = sum(1 for m in mids if abs(m - ASH_FAIR) > 15)

    spreads = [r["ask1"] - r["bid1"] for r in subset
               if r["bid1"] is not None and r["ask1"] is not None]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    # Adverse selection: ticks where position sign disagrees with mean-reversion direction
    adverse_ticks = 0
    for ts, mid, pnl, pos in pos_series:
        if mid is None:
            continue
        deviation = mid - ASH_FAIR
        # Adverse: long when price is above fair (reverter will push it down on us)
        #          short when price is below fair (reverter will push it up against us)
        if (pos > 5 and deviation > 5) or (pos < -5 and deviation < -5):
            adverse_ticks += 1

    print(f"\n  ASH  Day {day}:")
    print(f"    Actual PnL         = {actual_pnl:>10,.0f}")
    print(f"    PnL / tick         = {actual_pnl/total_ticks:.3f}")
    print(f"    Mid range          = {mid_min:.0f} – {mid_max:.0f}  (span {mid_max-mid_min:.0f})")
    print(f"    Avg market spread  = {avg_spread:.2f} ticks")
    print(f"    Position distribution:")
    for bucket, count in buckets.items():
        pct = 100 * count / total_ticks if total_ticks else 0
        print(f"      {bucket:<30}: {count:>5}/{total_ticks}  ({pct:.0f}%)")
    print(f"    Ticks mid > ±5  from 10000 : {ticks_far}/{total_ticks} ({100*ticks_far/max(1,total_ticks):.0f}%)")
    print(f"    Ticks mid > ±15 from 10000 : {ticks_very_far}/{total_ticks} ({100*ticks_very_far/max(1,total_ticks):.0f}%)")
    print(f"    Adverse-selection ticks    : {adverse_ticks}/{total_ticks} ({100*adverse_ticks/max(1,total_ticks):.0f}%)")


# ---------------------------------------------------------------------------
# PEPPER analysis
# ---------------------------------------------------------------------------

def analyze_pepper(rows, day):
    subset = [r for r in rows if r["product"] == PEPPER and r["day"] == day]
    if not subset:
        print(f"  No PEPPER data for day {day}")
        return
    subset.sort(key=lambda r: r["ts"])

    actual_pnl  = subset[-1]["pnl"]
    first_mid   = subset[0]["mid"] or 0
    last_mid    = subset[-1]["mid"] or 0
    theoretical = PEPPER_POSITION_LIMIT * (last_mid - first_mid)

    pos_series  = reconstruct_position(rows, PEPPER, day)
    positions   = [p for _, _, _, p in pos_series]
    avg_pos     = sum(positions) / len(positions) if positions else 0

    # Lag loss: each tick where position < 80 and mid moves up, we miss that gain
    lag_loss = 0.0
    for i in range(len(pos_series) - 1):
        _, mid_curr, _, pos_est  = pos_series[i]
        _, mid_next, _, _        = pos_series[i + 1]
        if mid_curr is None or mid_next is None:
            continue
        mid_chg   = mid_next - mid_curr
        shortfall = PEPPER_POSITION_LIMIT - pos_est
        if shortfall > 0 and mid_chg > 0:
            lag_loss += shortfall * mid_chg

    ticks_full  = sum(1 for p in positions if p >= 79)
    total_ticks = len(positions)

    # Build time: first tick reaching full position
    build_tick = next((ts for ts, _, _, p in pos_series if p >= 79), None)

    efficiency = actual_pnl / theoretical if theoretical != 0 else float("nan")

    print(f"\n  PEPPER  Day {day}:")
    print(f"    Actual PnL         = {actual_pnl:>10,.0f}")
    print(f"    Theoretical max    = {theoretical:>10,.0f}  (80 × {last_mid-first_mid:+.1f})")
    print(f"    Efficiency         = {efficiency:>10.1%}")
    print(f"    Gap vs theoretical = {actual_pnl - theoretical:>10,.0f}")
    print(f"    Lag loss (est)     = {-lag_loss:>10,.0f}")
    print(f"    First mid / Last   = {first_mid:.1f} / {last_mid:.1f}")
    print(f"    Build time         = tick {build_tick if build_tick is not None else 'never'}")
    print(f"    Avg position       = {avg_pos:.1f}/80  ({100*avg_pos/80:.0f}%)")
    print(f"    Ticks at full pos  = {ticks_full}/{total_ticks}  ({100*ticks_full/max(1,total_ticks):.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "backtests")
        if not os.path.isdir(log_dir):
            print("No log file specified and no backtests/ directory found.")
            sys.exit(1)
        logs = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
        if not logs:
            print("No .log files in backtests/. Run the backtester first.")
            sys.exit(1)
        log_path = os.path.join(log_dir, logs[-1])
        print(f"Using most recent log: {log_path}")
    else:
        log_path = sys.argv[1]

    print(f"\nParsing {log_path}...")
    rows = parse_log(log_path)
    print(f"Loaded {len(rows)} activity rows.")

    days = sorted(set(r["day"] for r in rows))
    print(f"Days: {days}")

    for day in days:
        ash_pnl    = next((r["pnl"] for r in reversed(rows)
                           if r["product"] == ASH    and r["day"] == day), 0)
        pepper_pnl = next((r["pnl"] for r in reversed(rows)
                           if r["product"] == PEPPER and r["day"] == day), 0)
        print(f"\n{'='*60}")
        print(f"DAY {day}  |  ASH={ash_pnl:,.0f}  PEPPER={pepper_pnl:,.0f}  "
              f"TOTAL={ash_pnl+pepper_pnl:,.0f}")
        print(f"{'='*60}")
        analyze_ash(rows, day)
        analyze_pepper(rows, day)

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL DAYS")
    print(f"{'='*60}")
    for product in [ASH, PEPPER]:
        pnls  = [next((r["pnl"] for r in reversed(rows)
                       if r["product"] == product and r["day"] == day), 0) for day in days]
        total = sum(pnls)
        print(f"  {product}: {' + '.join(f'{p:,.0f}' for p in pnls)} = {total:,.0f}")


if __name__ == "__main__":
    main()
