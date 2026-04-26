#!/usr/bin/env python3
"""
IMC Prosperity 4 - Round 3 Trading Log Analysis
Log: 446005
Total P&L: -55112.5
"""

import json
import csv
import io
from collections import defaultdict

# ─── Load JSON ────────────────────────────────────────────────────────────────
LOG_PATH = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/446005/446005.json"

with open(LOG_PATH) as f:
    data = json.load(f)

print("=" * 70)
print("IMC PROSPERITY 4 — ROUND 3 TRADING LOG ANALYSIS (446005)")
print("=" * 70)
print(f"  Round  : {data['round']}")
print(f"  Status : {data['status']}")
print(f"  Profit : {data['profit']:,.2f}")
print()

# ─── Parse activitiesLog ──────────────────────────────────────────────────────
activities_raw = data["activitiesLog"]
reader = csv.DictReader(io.StringIO(activities_raw), delimiter=";")

# Per-product time-series: { product: [(timestamp, mid_price, pnl), ...] }
product_series = defaultdict(list)

for row in reader:
    day = int(row["day"])
    ts  = int(row["timestamp"])
    product = row["product"]
    try:
        pnl = float(row["profit_and_loss"]) if row["profit_and_loss"] else 0.0
    except ValueError:
        pnl = 0.0
    try:
        mid = float(row["mid_price"]) if row["mid_price"] else None
    except ValueError:
        mid = None

    # Canonical timestamp across days
    global_ts = day * 1_000_000 + ts
    product_series[product].append((global_ts, mid, pnl))

# Sort each product's series by timestamp
for prod in product_series:
    product_series[prod].sort(key=lambda x: x[0])

all_products = sorted(product_series.keys())

# ─── Per-product summary ──────────────────────────────────────────────────────
print("─" * 70)
print("PER-PRODUCT P&L SUMMARY")
print("─" * 70)
print(f"{'Product':<25} {'Final P&L':>12} {'Max P&L':>12} {'Min P&L':>12} {'# Ticks':>8}")
print(f"{'─'*25} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

product_final_pnl = {}
product_max_pnl   = {}
product_min_pnl   = {}

winners = []
losers  = []

for prod in all_products:
    series = product_series[prod]
    pnls   = [x[2] for x in series]
    final  = pnls[-1]
    maxp   = max(pnls)
    minp   = min(pnls)

    product_final_pnl[prod] = final
    product_max_pnl[prod]   = maxp
    product_min_pnl[prod]   = minp

    marker = "  +" if final > 0 else ("  -" if final < 0 else "   ")
    print(f"{prod:<25} {final:>12,.1f} {maxp:>12,.1f} {minp:>12,.1f} {len(series):>8}{marker}")

    if final > 0:
        winners.append((prod, final))
    elif final < 0:
        losers.append((prod, final))

print()
total_product_pnl = sum(product_final_pnl.values())
print(f"  Sum of per-product P&Ls : {total_product_pnl:,.2f}")
print(f"  Reported total profit   : {data['profit']:,.2f}")
print()

# ─── Product groups ───────────────────────────────────────────────────────────
print("─" * 70)
print("PRODUCT GROUP ANALYSIS")
print("─" * 70)

# Define groupings
groups = {
    "HYDROGEL_PACK": [p for p in all_products if p == "HYDROGEL_PACK"],
    "VELVETFRUIT_EXTRACT": [p for p in all_products if p == "VELVETFRUIT_EXTRACT"],
    "VEV deep ITM (4000-4500)": [p for p in all_products if p.startswith("VEV_") and
                                   any(p == f"VEV_{x}" for x in ["4000","4100","4200","4300","4400","4500"])],
    "VEV ATM scalp (5000-5500)": [p for p in all_products if p.startswith("VEV_") and
                                    any(p == f"VEV_{x}" for x in ["5000","5100","5200","5300","5400","5500"])],
    "VEV passive short (6000-6500)": [p for p in all_products if p.startswith("VEV_") and
                                       any(p == f"VEV_{x}" for x in ["6000","6100","6200","6300","6400","6500"])],
    "OTHER": [],
}

# Anything not caught goes to OTHER
assigned = set()
for g, members in groups.items():
    assigned.update(members)
groups["OTHER"] = [p for p in all_products if p not in assigned]

for group_name, members in groups.items():
    if not members:
        continue
    group_final = sum(product_final_pnl.get(m, 0) for m in members)
    group_max   = sum(product_max_pnl.get(m, 0)   for m in members)
    group_min   = sum(product_min_pnl.get(m, 0)   for m in members)

    print(f"\nGroup: {group_name}")
    print(f"  Members : {', '.join(members) if members else 'none'}")
    print(f"  Final P&L (sum) : {group_final:>12,.1f}")
    print(f"  Max P&L  (sum)  : {group_max:>12,.1f}")
    print(f"  Min P&L  (sum)  : {group_min:>12,.1f}")

# ─── Parse graphLog ───────────────────────────────────────────────────────────
print()
print("─" * 70)
print("PORTFOLIO P&L CURVE ANALYSIS (graphLog)")
print("─" * 70)

graph_raw = data["graphLog"]
graph_reader = csv.DictReader(io.StringIO(graph_raw), delimiter=";")
graph_series = []
for row in graph_reader:
    try:
        ts  = int(row["timestamp"])
        val = float(row["value"])
        graph_series.append((ts, val))
    except (ValueError, KeyError):
        pass

graph_series.sort(key=lambda x: x[0])

if graph_series:
    ts_vals = [x[1] for x in graph_series]
    overall_max   = max(ts_vals)
    overall_min   = min(ts_vals)
    overall_final = ts_vals[-1]

    # Find peak and trough timestamps
    peak_ts    = graph_series[ts_vals.index(overall_max)][0]
    trough_ts  = graph_series[ts_vals.index(overall_min)][0]

    print(f"  Data points   : {len(graph_series)}")
    print(f"  Final value   : {overall_final:>12,.2f}")
    print(f"  Peak value    : {overall_max:>12,.2f}  @ ts={peak_ts:,}")
    print(f"  Trough value  : {overall_min:>12,.2f}  @ ts={trough_ts:,}")
    print()

    # Day 0 / Day 1 / Day 2 breakdown
    # Timestamps reset per day; graphLog timestamps seem continuous
    # We'll use breakpoints at 1_000_000 and 2_000_000
    def find_pnl_at_ts_boundary(series, boundary):
        """Return last value <= boundary."""
        val = None
        for ts, v in series:
            if ts <= boundary:
                val = v
            else:
                break
        return val

    # Build epoch-relative segments (each day 0..999_999 ticks)
    # graphLog doesn't include day col - need to infer from ts resets
    # Strategy: large negative jumps in ts indicate new day
    day_boundaries = [0]
    prev_ts = graph_series[0][0]
    for ts, _ in graph_series[1:]:
        if ts < prev_ts - 500_000:   # day rolled over
            day_boundaries.append(ts)
        prev_ts = ts

    print(f"  Detected day boundaries at timestamps: {day_boundaries}")

    # Segment into days
    day_segs = defaultdict(list)
    current_day = 0
    for i, (ts, val) in enumerate(graph_series):
        if current_day + 1 < len(day_boundaries) and ts >= day_boundaries[current_day + 1]:
            current_day += 1
        day_segs[current_day].append((ts, val))

    for day_idx, seg in sorted(day_segs.items()):
        seg_vals = [v for _, v in seg]
        day_start_pnl = seg_vals[0]
        day_end_pnl   = seg_vals[-1]
        day_max       = max(seg_vals)
        day_min       = min(seg_vals)
        print(f"  Day {day_idx}: start={day_start_pnl:>10,.2f}  end={day_end_pnl:>10,.2f}  "
              f"max={day_max:>10,.2f}  min={day_min:>10,.2f}  "
              f"delta={day_end_pnl - day_start_pnl:>10,.2f}")

    # Find largest P&L drops (drawdowns)
    print()
    print("  Largest intra-period drawdowns:")
    running_peak = ts_vals[0]
    worst_drawdowns = []
    for i, (ts, val) in enumerate(graph_series):
        if val > running_peak:
            running_peak = val
        dd = val - running_peak
        worst_drawdowns.append((dd, ts, val, running_peak))

    worst_drawdowns.sort(key=lambda x: x[0])  # most negative first
    for dd, ts, val, peak in worst_drawdowns[:5]:
        print(f"    ts={ts:>8,}  drawdown={dd:>10,.2f}  (pnl={val:,.2f} from peak={peak:,.2f})")

    # Find the sharpest single-step drops
    print()
    print("  Sharpest single-step P&L drops (consecutive ticks):")
    drops = []
    for i in range(1, len(graph_series)):
        delta = graph_series[i][1] - graph_series[i-1][1]
        drops.append((delta, graph_series[i-1][0], graph_series[i][0], graph_series[i-1][1], graph_series[i][1]))
    drops.sort(key=lambda x: x[0])
    for delta, ts1, ts2, v1, v2 in drops[:5]:
        print(f"    ts={ts1:>8,}→{ts2:>8,}  drop={delta:>10,.2f}  ({v1:,.2f}→{v2:,.2f})")

# ─── Parse positions ──────────────────────────────────────────────────────────
print()
print("─" * 70)
print("FINAL POSITIONS")
print("─" * 70)

positions = data["positions"]
pos_map = {p["symbol"]: p["quantity"] for p in positions}

non_zero = [(sym, qty) for sym, qty in pos_map.items() if qty != 0]
zero     = [(sym, qty) for sym, qty in pos_map.items() if qty == 0]

print(f"  Total symbols tracked : {len(positions)}")
print(f"  Non-zero positions    : {len(non_zero)}")
print()

if non_zero:
    print("  NON-ZERO POSITIONS (potential stuck inventory):")
    print(f"  {'Symbol':<25} {'Qty':>12} {'Final P&L':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    for sym, qty in sorted(non_zero, key=lambda x: abs(x[1]), reverse=True):
        fpnl = product_final_pnl.get(sym, "N/A")
        fpnl_str = f"{fpnl:,.1f}" if isinstance(fpnl, float) else fpnl
        print(f"  {sym:<25} {qty:>12,} {fpnl_str:>12}")
else:
    print("  All positions flat at end of run.")

if zero:
    zero_syms = ", ".join(s for s, _ in zero)
    print(f"\n  Flat positions: {zero_syms}")

# ─── Winners vs Losers ────────────────────────────────────────────────────────
print()
print("─" * 70)
print("WINNERS vs LOSERS")
print("─" * 70)

winners_sorted = sorted([(p, v) for p, v in product_final_pnl.items() if v > 0], key=lambda x: -x[1])
losers_sorted  = sorted([(p, v) for p, v in product_final_pnl.items() if v < 0], key=lambda x: x[1])
flat_prods     = [(p, v) for p, v in product_final_pnl.items() if v == 0]

print(f"\n  Winners ({len(winners_sorted)}):")
for prod, val in winners_sorted:
    print(f"    {prod:<25}  +{val:>10,.1f}")

print(f"\n  Losers ({len(losers_sorted)}):")
for prod, val in losers_sorted:
    print(f"    {prod:<25}   {val:>10,.1f}")

if flat_prods:
    print(f"\n  Flat/Zero P&L ({len(flat_prods)}): {', '.join(p for p, _ in flat_prods)}")

# ─── Strategy failure diagnosis ───────────────────────────────────────────────
print()
print("─" * 70)
print("STRATEGY FAILURE DIAGNOSIS")
print("─" * 70)

# Check VEV options coverage
vev_products = [p for p in all_products if p.startswith("VEV_")]
vev_strikes  = sorted(int(p.split("_")[1]) for p in vev_products)

print(f"\n  VEV strikes traded: {vev_strikes}")

# Check for deep ITM options (these have high intrinsic value; holding them is capital-tied)
itm_vev = [p for p in vev_products if int(p.split("_")[1]) <= 4500]
atm_vev = [p for p in vev_products if 5000 <= int(p.split("_")[1]) <= 5500]
otm_vev = [p for p in vev_products if int(p.split("_")[1]) >= 6000]

print(f"  ITM (≤4500): {itm_vev}")
print(f"  ATM (5000-5500): {atm_vev}")
print(f"  OTM (≥6000): {otm_vev}")

print()
print("  P&L by VEV strike:")
for p in sorted(vev_products, key=lambda x: int(x.split("_")[1])):
    fpnl = product_final_pnl.get(p, 0)
    maxp = product_max_pnl.get(p, 0)
    minp = product_min_pnl.get(p, 0)
    pos  = pos_map.get(p, 0)
    bar  = "+" if fpnl > 0 else ("-" if fpnl < 0 else " ")
    print(f"    {p:<12}  P&L={fpnl:>10,.1f}  max={maxp:>10,.1f}  min={minp:>10,.1f}  pos={pos:>5}  {bar}")

# Check for XIRECS (seashells / base currency) impact
print()
xirecs_qty = pos_map.get("XIRECS", 0)
if xirecs_qty != 0:
    print(f"  XIRECS (base currency) position: {xirecs_qty:,}")
    print(f"  NOTE: Large XIRECS position ({xirecs_qty:,}) indicates capital tied up / not settled.")

# Check stuck inventory diagnosis
print()
print("  Inventory risk assessment:")
for sym, qty in non_zero:
    if sym == "XIRECS":
        continue
    fpnl = product_final_pnl.get(sym, 0)
    is_options = sym.startswith("VEV_")
    strike = int(sym.split("_")[1]) if is_options else None

    if abs(qty) > 0:
        print(f"    {sym} qty={qty:>6}: ", end="")
        if sym == "HYDROGEL_PACK":
            print(f"STUCK LONG {qty} units, final P&L {fpnl:,.1f} — likely bought high, couldn't exit")
        elif sym == "VELVETFRUIT_EXTRACT":
            print(f"STUCK LONG {qty} units, final P&L {fpnl:,.1f} — likely bought high, couldn't exit")
        elif is_options and qty > 0:
            print(f"LONG {qty} VEV options at strike {strike}, P&L {fpnl:,.1f}")
        elif is_options and qty < 0:
            print(f"SHORT {abs(qty)} VEV options at strike {strike}, P&L {fpnl:,.1f}")
        else:
            print(f"qty={qty}, P&L={fpnl:,.1f}")

# ─── Overall conclusions ──────────────────────────────────────────────────────
print()
print("─" * 70)
print("OVERALL CONCLUSIONS")
print("─" * 70)

biggest_loser  = min(product_final_pnl.items(), key=lambda x: x[1]) if product_final_pnl else None
biggest_winner = max(product_final_pnl.items(), key=lambda x: x[1]) if product_final_pnl else None

if biggest_loser:
    print(f"\n  Biggest loser   : {biggest_loser[0]} with P&L {biggest_loser[1]:,.1f}")
if biggest_winner:
    print(f"  Biggest winner  : {biggest_winner[0]} with P&L {biggest_winner[1]:,.1f}")

total_winners_pnl = sum(v for v in product_final_pnl.values() if v > 0)
total_losers_pnl  = sum(v for v in product_final_pnl.values() if v < 0)
print(f"\n  Total from winners : {total_winners_pnl:>12,.1f}")
print(f"  Total from losers  : {total_losers_pnl:>12,.1f}")
print(f"  Net (product sum)  : {total_winners_pnl + total_losers_pnl:>12,.1f}")

print()
# Heuristic conclusions
if biggest_loser and abs(biggest_loser[1]) > 10000:
    print(f"  [CRITICAL] {biggest_loser[0]} is responsible for a dominant share of losses.")
    print(f"             This suggests a systematic strategy failure, not noise.")

# Check if we're stuck with positions at end
stuck = [(sym, qty) for sym, qty in non_zero if sym not in ("XIRECS",) and abs(qty) > 5]
if stuck:
    print()
    print("  [WARNING] Stuck non-trivial positions at end-of-round:")
    for sym, qty in stuck:
        print(f"    {sym}: qty={qty}")
    print("    These positions were NOT exited — mark-to-market losses may be unrealized but real.")

# VEV net exposure
vev_pnl_total = sum(product_final_pnl.get(p, 0) for p in vev_products)
print()
print(f"  VEV options total P&L : {vev_pnl_total:>12,.1f}")
print(f"  HYDROGEL_PACK P&L     : {product_final_pnl.get('HYDROGEL_PACK', 0):>12,.1f}")
print(f"  VELVETFRUIT_EXTRACT   : {product_final_pnl.get('VELVETFRUIT_EXTRACT', 0):>12,.1f}")

print()
print("─" * 70)
print("END OF ANALYSIS")
print("─" * 70)
