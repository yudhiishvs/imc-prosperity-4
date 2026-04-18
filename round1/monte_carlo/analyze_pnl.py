"""
analyze_pnl.py
--------------
Decomposes PnL from a prosperity4btx backtest log.

For PEPPER:
  - Theoretical max PnL (80 units × trend from tick 0)
  - Lag loss: PnL lost because position < 80 during buildup
  - Entry premium: total overpay above fair value on aggressive buys
  - Actual PnL vs theoretical gap

For ASH:
  - Spread capture per trade
  - Position time distribution (neutral vs extreme)
  - Emergency flatten events (position crosses KILL_SWITCH or EMERGENCY_THRESHOLD)
  - Estimated adverse selection cost

Usage:
    python analyze_pnl.py <log_file>
    python analyze_pnl.py ../backtests/2026-04-15_21-07-48.log
"""

import sys
import csv
import io
import os

PEPPER = "INTARIAN_PEPPER_ROOT"
ASH    = "ASH_COATED_OSMIUM"
PEPPER_SLOPE = 0.001        # price per timestamp tick
PEPPER_POSITION_LIMIT = 80
ASH_POSITION_LIMIT    = 80
ASH_FAIR = 10_000.0


def parse_log(path: str):
    """
    Parse prosperity4btx log file.
    Returns list of dicts with keys:
      day, timestamp, product, bid1, bvol1, ask1, avol1, mid, pnl
    Only parses the Activities log section.
    """
    rows = []
    in_activities = False
    header_seen = False

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
                continue  # skip header row

            parts = line.split(";")
            if len(parts) < 17:
                continue
            try:
                row = {
                    "day":     int(parts[0]),
                    "ts":      int(parts[1]),
                    "product": parts[2],
                    "bid1":    float(parts[3]) if parts[3] else None,
                    "bvol1":   int(parts[4])   if parts[4] else None,
                    "ask1":    float(parts[9])  if parts[9] else None,
                    "avol1":   int(parts[10])   if parts[10] else None,
                    "mid":     float(parts[15]) if parts[15] else None,
                    "pnl":     float(parts[16]) if parts[16] else 0.0,
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue
    return rows


def pnl_delta_per_tick(rows, product, day):
    """Returns list of (ts, pnl_delta, mid, position_implied) for one product/day."""
    subset = [r for r in rows if r["product"] == product and r["day"] == day]
    subset.sort(key=lambda r: r["ts"])
    result = []
    prev_pnl = 0.0
    for r in subset:
        delta = r["pnl"] - prev_pnl
        prev_pnl = r["pnl"]
        result.append((r["ts"], delta, r["mid"], r["pnl"]))
    return result


def reconstruct_position(rows, product, day):
    """
    Reconstruct approximate position from PnL changes and mid price changes.
    PnL[t+1] - PnL[t] = position[t] * (mid[t+1] - mid[t])  (mark-to-market)
    => position[t] ≈ (PnL[t+1] - PnL[t]) / (mid[t+1] - mid[t])  when mid changes

    Where mid doesn't change, position is unknown from mark-to-market alone.
    We use the last known position as carry-forward.
    Returns list of (ts, mid, pnl, position_estimate).
    """
    subset = [r for r in rows if r["product"] == product and r["day"] == day]
    subset.sort(key=lambda r: r["ts"])

    positions = []
    pos = 0.0
    for i in range(len(subset) - 1):
        curr = subset[i]
        nxt  = subset[i + 1]
        mid_chg = (nxt["mid"] or 0) - (curr["mid"] or 0) if (curr["mid"] and nxt["mid"]) else 0
        pnl_chg = nxt["pnl"] - curr["pnl"]

        if abs(mid_chg) > 0.1:
            pos = pnl_chg / mid_chg
            pos = max(-80, min(80, round(pos)))  # clip to position limits
        # else keep previous position estimate

        positions.append((curr["ts"], curr["mid"], curr["pnl"], pos))

    if subset:
        positions.append((subset[-1]["ts"], subset[-1]["mid"], subset[-1]["pnl"], pos))
    return positions


def analyze_pepper(rows, day):
    """
    PEPPER PnL decomposition for one day.
    """
    subset = [r for r in rows if r["product"] == PEPPER and r["day"] == day]
    if not subset:
        print(f"  No PEPPER data for day {day}")
        return
    subset.sort(key=lambda r: r["ts"])

    # Final actual PnL
    actual_pnl = subset[-1]["pnl"]

    # Reconstruct position
    pos_series = reconstruct_position(rows, PEPPER, day)

    # Theoretical PnL: 80 units held from tick 0 (mark-to-market on mid)
    first_mid = subset[0]["mid"] or 0
    last_mid  = subset[-1]["mid"] or 0
    theoretical_pnl = PEPPER_POSITION_LIMIT * (last_mid - first_mid)

    # Lag loss: each tick where position < 80, we lose (80 - pos) * mid_change
    lag_loss = 0.0
    for i in range(len(pos_series) - 1):
        ts_curr, mid_curr, pnl_curr, pos_est = pos_series[i]
        ts_next, mid_next, pnl_next, _       = pos_series[i + 1]
        if mid_curr is None or mid_next is None:
            continue
        mid_chg = mid_next - mid_curr
        shortfall = PEPPER_POSITION_LIMIT - pos_est
        if shortfall > 0 and mid_chg > 0:
            lag_loss += shortfall * mid_chg

    # Position build time: first tick where position reaches 80
    build_tick = None
    for ts, mid, pnl, pos in pos_series:
        if pos >= 79:
            build_tick = ts
            break

    # Position stats
    positions = [p for _, _, _, p in pos_series]
    avg_pos = sum(positions) / len(positions) if positions else 0

    # Ticks at full position
    ticks_full = sum(1 for p in positions if p >= 79)
    ticks_total = len(positions)

    print(f"\n  PEPPER Day {day}:")
    print(f"    Actual PnL         = {actual_pnl:>10,.0f}")
    print(f"    Theoretical max    = {theoretical_pnl:>10,.0f}  (80 units × {last_mid-first_mid:+.1f} price move)")
    print(f"    Gap vs theoretical = {actual_pnl - theoretical_pnl:>10,.0f}")
    print(f"    Lag loss (est)     = {-lag_loss:>10,.0f}  (ticks underweight × upward moves)")
    print(f"    First mid          = {first_mid:.1f}  |  Last mid = {last_mid:.1f}")
    print(f"    Build time         = tick {build_tick if build_tick else 'never'}")
    print(f"    Avg position       = {avg_pos:.1f} / 80  ({100*avg_pos/80:.0f}%)")
    print(f"    Ticks at full pos  = {ticks_full}/{ticks_total}  ({100*ticks_full/ticks_total:.0f}%)")


def analyze_ash(rows, day):
    """
    ASH PnL decomposition for one day.
    """
    subset = [r for r in rows if r["product"] == ASH and r["day"] == day]
    if not subset:
        print(f"  No ASH data for day {day}")
        return
    subset.sort(key=lambda r: r["ts"])

    actual_pnl = subset[-1]["pnl"]

    # Position reconstruction
    pos_series = reconstruct_position(rows, ASH, day)
    positions = [p for _, _, _, p in pos_series]

    # Position distribution buckets
    buckets = {
        "neutral [0,10)": 0,
        "moderate [10,30)": 0,
        "elevated [30,60)": 0,
        "extreme [60,80]": 0,
    }
    for p in positions:
        ap = abs(p)
        if ap < 10:   buckets["neutral [0,10)"] += 1
        elif ap < 30: buckets["moderate [10,30)"] += 1
        elif ap < 60: buckets["elevated [30,60)"] += 1
        else:         buckets["extreme [60,80]"] += 1

    total_ticks = len(positions)

    # Price range
    mids = [r["mid"] for r in subset if r["mid"] is not None]
    mid_min = min(mids) if mids else 0
    mid_max = max(mids) if mids else 0
    mid_range = mid_max - mid_min

    # Time spent far from fair value (where edge is best)
    ticks_far = sum(1 for m in mids if abs(m - ASH_FAIR) > 5)
    ticks_very_far = sum(1 for m in mids if abs(m - ASH_FAIR) > 15)

    # PnL per tick (rate of capture)
    pnl_per_tick = actual_pnl / total_ticks if total_ticks > 0 else 0

    # Spread around mid
    spreads = []
    for r in subset:
        if r["bid1"] is not None and r["ask1"] is not None:
            spreads.append(r["ask1"] - r["bid1"])
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    print(f"\n  ASH Day {day}:")
    print(f"    Actual PnL         = {actual_pnl:>10,.0f}")
    print(f"    PnL per tick       = {pnl_per_tick:>10.2f}")
    print(f"    Mid range          = {mid_min:.0f} – {mid_max:.0f}  (span {mid_range:.0f})")
    print(f"    Avg market spread  = {avg_spread:.1f} ticks")
    print(f"    Price position:")
    for bucket, count in buckets.items():
        print(f"      {bucket:30s}: {count:5d}/{total_ticks}  ({100*count/total_ticks:.0f}%)")
    print(f"    Ticks mid > ±5 from 10000  : {ticks_far}/{total_ticks} ({100*ticks_far/total_ticks:.0f}%)")
    print(f"    Ticks mid > ±15 from 10000 : {ticks_very_far}/{total_ticks} ({100*ticks_very_far/total_ticks:.0f}%)")


def main():
    if len(sys.argv) < 2:
        # Use the most recent log
        log_dir = os.path.join(os.path.dirname(__file__), "..", "backtests")
        logs = sorted(os.listdir(log_dir))
        if not logs:
            print("No log files found. Run the backtester first.")
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

    # Per-day breakdown
    for day in days:
        ash_pnl    = next((r["pnl"] for r in reversed(rows) if r["product"] == ASH    and r["day"] == day), 0)
        pepper_pnl = next((r["pnl"] for r in reversed(rows) if r["product"] == PEPPER and r["day"] == day), 0)
        print(f"\n{'='*60}")
        print(f"DAY {day}  |  ASH={ash_pnl:,.0f}  PEPPER={pepper_pnl:,.0f}  TOTAL={ash_pnl+pepper_pnl:,.0f}")
        print(f"{'='*60}")
        analyze_pepper(rows, day)
        analyze_ash(rows, day)

    # Summary across all days
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL DAYS")
    print(f"{'='*60}")
    for product in [PEPPER, ASH]:
        pnls = []
        for day in days:
            pnl = next((r["pnl"] for r in reversed(rows) if r["product"] == product and r["day"] == day), 0)
            pnls.append(pnl)
        total = sum(pnls)
        print(f"  {product}: {' + '.join(f'{p:,.0f}' for p in pnls)} = {total:,.0f}")


if __name__ == "__main__":
    main()
