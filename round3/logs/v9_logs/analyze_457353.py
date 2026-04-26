#!/usr/bin/env python3
"""Analysis script for IMC Prosperity 4 Round 3 backtest log 457353.json"""

import json
import csv
import io
from collections import defaultdict

LOG_PATH = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/v9_logs/457353.json"

print("=" * 80)
print("LOADING LOG FILE")
print("=" * 80)
with open(LOG_PATH) as f:
    data = json.load(f)

print(f"Top-level keys: {list(data.keys())}")

# ─── 1. OVERALL PNL FROM activitiesLog ────────────────────────────────────────
print("\n" + "=" * 80)
print("1. OVERALL PnL BREAKDOWN (activitiesLog)")
print("=" * 80)

activities_raw = data.get("activitiesLog", "")
if activities_raw:
    reader = csv.DictReader(io.StringIO(activities_raw), delimiter=";")
    rows = list(reader)
    print(f"Total activity rows: {len(rows)}")
    if rows:
        print(f"Columns: {list(rows[0].keys())}")

    # Group rows by product, track PnL over time
    product_pnl = defaultdict(list)
    for row in rows:
        prod = row.get("product", "").strip()
        pnl_str = row.get("profit_and_loss", "0").strip()
        ts = int(row.get("timestamp", 0))
        try:
            pnl = float(pnl_str)
        except ValueError:
            pnl = 0.0
        product_pnl[prod].append((ts, pnl))

    print("\nPnL by product (final value at last timestamp):")
    total_final = 0.0
    for prod, vals in sorted(product_pnl.items()):
        vals_sorted = sorted(vals, key=lambda x: x[0])
        first_ts, first_pnl = vals_sorted[0]
        mid_idx = len(vals_sorted) // 2
        mid_ts, mid_pnl = vals_sorted[mid_idx]
        last_ts, last_pnl = vals_sorted[-1]
        total_final += last_pnl
        print(f"  {prod:30s}  first={first_pnl:10.1f}  mid={mid_pnl:10.1f}  last={last_pnl:10.1f}  (n={len(vals_sorted)})")
    print(f"\n  TOTAL (sum of last values): {total_final:.1f}")
else:
    print("No activitiesLog found")

# ─── 2. OWN TRADES (tradeHistory) ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. OWN TRADES (tradeHistory)")
print("=" * 80)

trade_history = data.get("tradeHistory", [])
print(f"Total trades in tradeHistory: {len(trade_history)}")
if trade_history:
    sample = trade_history[0]
    print(f"Sample trade keys: {list(sample.keys())}")
    print(f"Sample trade: {sample}")

# Filter our trades (SUBMISSION is buyer or seller)
our_trades = [t for t in trade_history if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
print(f"\nOur trades (SUBMISSION as buyer or seller): {len(our_trades)}")

# Per-product buy/sell stats
buy_data = defaultdict(list)   # prod -> list of prices
sell_data = defaultdict(list)

for t in our_trades:
    sym = t.get("symbol", t.get("product", "?"))
    price = float(t.get("price", 0))
    qty = int(t.get("quantity", 0))
    if t.get("buyer") == "SUBMISSION":
        for _ in range(abs(qty)):
            buy_data[sym].append(price)
    else:
        for _ in range(abs(qty)):
            sell_data[sym].append(price)

all_syms = set(list(buy_data.keys()) + list(sell_data.keys()))
print("\nPer-product trade summary:")
print(f"  {'Product':30s}  {'#Buys':>7}  {'AvgBuy':>9}  {'#Sells':>7}  {'AvgSell':>9}  {'Net':>7}")
for sym in sorted(all_syms):
    buys = buy_data[sym]
    sells = sell_data[sym]
    nb = len(buys)
    ns = len(sells)
    ab = sum(buys)/nb if nb else 0
    as_ = sum(sells)/ns if ns else 0
    net = ns - nb
    print(f"  {sym:30s}  {nb:7d}  {ab:9.2f}  {ns:7d}  {as_:9.2f}  {net:+7d}")

# ─── 3. ALGORITHM PRINT OUTPUT (logs) ─────────────────────────────────────────
print("\n" + "=" * 80)
print("3. ALGORITHM PRINT OUTPUT (logs field)")
print("=" * 80)

logs = data.get("logs", [])
print(f"Total log entries: {len(logs)}")

# Flatten all algo output lines
all_lines = []
for entry in logs:
    if isinstance(entry, list) and len(entry) >= 3:
        algo_out = entry[2]
        if isinstance(algo_out, str):
            for line in algo_out.split("\n"):
                line = line.strip()
                if line:
                    all_lines.append((entry[0], line))

print(f"Total non-empty output lines: {len(all_lines)}")

# Count by tag
tag_counts = defaultdict(int)
tags = ["[INIT]", "[DAY]", "[TICK]", "[HG TAKE]", "[OPT]", "[CAP]", "[HEDGE]", "[FLOOR]",
        "[ERROR]", "[WARN]", "[VEV]", "[VELFRUIT]"]
for ts, line in all_lines:
    for tag in tags:
        if tag in line:
            tag_counts[tag] += 1

print("\nTag counts:")
for tag in tags:
    if tag_counts[tag] > 0:
        print(f"  {tag:15s}: {tag_counts[tag]}")

# HG TAKE BUY vs SELL
hg_take_buy = [line for _, line in all_lines if "[HG TAKE]" in line and "BUY" in line.upper()]
hg_take_sell = [line for _, line in all_lines if "[HG TAKE]" in line and "SELL" in line.upper()]
print(f"\n[HG TAKE] BUY  lines: {len(hg_take_buy)}")
print(f"[HG TAKE] SELL lines: {len(hg_take_sell)}")
if hg_take_buy:
    print(f"  Example BUY:  {hg_take_buy[0]}")
if hg_take_sell:
    print(f"  Example SELL: {hg_take_sell[0]}")

# OPT lines with non-zero bid/ask sizes
opt_lines = [(ts, line) for ts, line in all_lines if "[OPT]" in line]
print(f"\n[OPT] total lines: {len(opt_lines)}")
# Show sample OPT lines
for ts, line in opt_lines[:10]:
    print(f"  ts={ts}: {line}")

# CAP lines
cap_lines = [(ts, line) for ts, line in all_lines if "[CAP]" in line]
print(f"\n[CAP] total lines: {len(cap_lines)}")
for ts, line in cap_lines[:10]:
    print(f"  ts={ts}: {line}")

# HEDGE lines
hedge_lines = [(ts, line) for ts, line in all_lines if "[HEDGE]" in line]
print(f"\n[HEDGE] total lines: {len(hedge_lines)}")
for ts, line in hedge_lines[:10]:
    print(f"  ts={ts}: {line}")

# FLOOR lines
floor_lines = [(ts, line) for ts, line in all_lines if "[FLOOR]" in line]
print(f"\n[FLOOR] total lines: {len(floor_lines)}")
for ts, line in floor_lines[:5]:
    print(f"  ts={ts}: {line}")

# Error patterns
error_lines = [line for _, line in all_lines if "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower()]
print(f"\nError/exception lines: {len(error_lines)}")
for line in error_lines[:10]:
    print(f"  {line}")

# INIT / DAY lines
init_lines = [(ts, line) for ts, line in all_lines if "[INIT]" in line]
day_lines  = [(ts, line) for ts, line in all_lines if "[DAY]" in line]
print(f"\n[INIT] lines: {len(init_lines)}")
for ts, line in init_lines[:5]:
    print(f"  ts={ts}: {line}")
print(f"\n[DAY] lines: {len(day_lines)}")
for ts, line in day_lines[:5]:
    print(f"  ts={ts}: {line}")

# ─── 4. OPTIONS FILLS ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. OPTIONS FILLS (VEV_* in tradeHistory)")
print("=" * 80)

vev_trades = [t for t in our_trades if "VEV" in t.get("symbol", t.get("product", ""))]
print(f"Total VEV_* own trades: {len(vev_trades)}")
if vev_trades:
    for t in vev_trades[:20]:
        print(f"  {t}")

# Compare to market_trades
market_trades = data.get("marketTrades", data.get("market_trades", []))
print(f"\nMarket trades total: {len(market_trades)}")
vev_market = [t for t in market_trades if "VEV" in t.get("symbol", t.get("product", ""))]
print(f"VEV_* market trades: {len(vev_market)}")
if vev_market:
    for t in vev_market[:10]:
        print(f"  {t}")

# ─── 5. POSITION TRAJECTORY (HG and VF) ───────────────────────────────────────
print("\n" + "=" * 80)
print("5. POSITION TRAJECTORY (PnL over time)")
print("=" * 80)

for prod_key in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
    if prod_key in product_pnl:
        vals = sorted(product_pnl[prod_key], key=lambda x: x[0])
        n = len(vals)
        checkpoints = [0, n//4, n//2, 3*n//4, n-1]
        print(f"\n{prod_key} PnL trajectory:")
        for idx in checkpoints:
            ts, pnl = vals[idx]
            print(f"  ts={ts:7d}  pnl={pnl:10.1f}")

# ─── 6. ORDER BOOK PATTERNS FOR VEV OPTIONS ───────────────────────────────────
print("\n" + "=" * 80)
print("6. ORDER BOOK PATTERNS FOR VEV OPTIONS")
print("=" * 80)

vev_rows = [row for row in rows if "VEV" in row.get("product", "")]
print(f"VEV_* activitiesLog rows: {len(vev_rows)}")

if vev_rows:
    print("\nSample VEV rows (first 5):")
    for row in vev_rows[:5]:
        print(f"  {dict(row)}")

    # Compute spreads
    spreads = []
    for row in vev_rows:
        try:
            ask1 = float(row.get("ask_price_1", 0) or 0)
            bid1 = float(row.get("bid_price_1", 0) or 0)
            if ask1 > 0 and bid1 > 0:
                spreads.append((row["timestamp"], row["product"], ask1 - bid1, bid1, ask1))
        except (ValueError, TypeError):
            pass
    print(f"\nVEV rows with valid bid/ask: {len(spreads)}")
    if spreads:
        spread_vals = [s[2] for s in spreads]
        print(f"Spread stats: min={min(spread_vals):.1f}  max={max(spread_vals):.1f}  avg={sum(spread_vals)/len(spread_vals):.2f}")
        print(f"Spreads >= 3: {sum(1 for s in spread_vals if s >= 3)} ({100*sum(1 for s in spread_vals if s >= 3)/len(spread_vals):.1f}%)")
        print("\nSample spreads (first 10):")
        for ts, prod, spread, bid, ask in spreads[:10]:
            print(f"  ts={ts}  {prod:25s}  bid={bid:.1f}  ask={ask:.1f}  spread={spread:.1f}")

# Unique VEV products
vev_products = sorted(set(row["product"] for row in vev_rows))
print(f"\nUnique VEV products: {vev_products}")

# ─── 7. KEY FINDINGS ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("7. KEY FINDINGS SUMMARY")
print("=" * 80)

# Overall biggest losers
print("\nProducts by final PnL (sorted):")
final_pnls = [(prod, sorted(vals, key=lambda x: x[0])[-1][1]) for prod, vals in product_pnl.items()]
final_pnls.sort(key=lambda x: x[1])
for prod, pnl in final_pnls:
    print(f"  {prod:35s}  {pnl:10.1f}")

# HG TAKE stats
print(f"\nHG TAKE activity: {len(hg_take_buy)} buys, {len(hg_take_sell)} sells (total {len(hg_take_buy)+len(hg_take_sell)})")

# Any OPT placements that fired
vev_buy_qty = sum(buy_data.get(sym, []) for sym in all_syms if "VEV" in sym if False) or sum(len(buy_data[sym]) for sym in all_syms if "VEV" in sym)
vev_sell_qty = sum(len(sell_data[sym]) for sym in all_syms if "VEV" in sym)
print(f"VEV options fills: {vev_buy_qty} buys, {vev_sell_qty} sells")

# Final summary note
print("\nHydrogel PnL:", sorted(product_pnl.get("HYDROGEL_PACK", [(0, 0)]), key=lambda x: x[0])[-1][1] if "HYDROGEL_PACK" in product_pnl else "N/A")
print("VelvetFruit PnL:", sorted(product_pnl.get("VELVETFRUIT_EXTRACT", [(0, 0)]), key=lambda x: x[0])[-1][1] if "VELVETFRUIT_EXTRACT" in product_pnl else "N/A")

# Sample some OPT lines to check non-zero sizes
print("\nAll unique [OPT] line patterns (first 20 unique):")
opt_unique = list(dict.fromkeys(line for _, line in opt_lines))
for line in opt_unique[:20]:
    print(f"  {line}")

# FLOOR / CAP details
print("\nAll [FLOOR] lines:")
for ts, line in floor_lines:
    print(f"  ts={ts}: {line}")

print("\nAll [CAP] lines (first 20):")
for ts, line in cap_lines[:20]:
    print(f"  ts={ts}: {line}")

print("\nDone.")
