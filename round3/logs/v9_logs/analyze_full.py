#!/usr/bin/env python3
"""Full analysis of 457353.log for IMC Prosperity 4 Round 3"""

import json, csv, io
from collections import defaultdict

LOG_PATH = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/v9_logs/457353.log"

print("=" * 80)
print("LOADING LOG FILE (.log)")
print("=" * 80)
with open(LOG_PATH) as f:
    d = json.load(f)
print(f"Keys: {list(d.keys())}")

# ─── Parse activitiesLog ──────────────────────────────────────────────────────
activities_raw = d["activitiesLog"]
reader = csv.DictReader(io.StringIO(activities_raw), delimiter=";")
rows = list(reader)
print(f"\nactivitiesLog rows: {len(rows)}")

product_pnl = defaultdict(list)
for row in rows:
    prod = row["product"].strip()
    try:
        pnl = float(row["profit_and_loss"])
    except (ValueError, KeyError):
        pnl = 0.0
    ts = int(row["timestamp"])
    product_pnl[prod].append((ts, pnl))

# ─── Parse tradeHistory ───────────────────────────────────────────────────────
trade_history = d.get("tradeHistory", [])
print(f"tradeHistory trades: {len(trade_history)}")
if trade_history:
    print(f"Sample: {trade_history[0]}")

# Separate our trades from market trades
our_trades = [t for t in trade_history if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
market_only = [t for t in trade_history if t.get("buyer") != "SUBMISSION" and t.get("seller") != "SUBMISSION"]
print(f"  Our trades (SUBMISSION): {len(our_trades)}")
print(f"  Market-only trades: {len(market_only)}")

# ─── Parse logs (lambdaLog) ───────────────────────────────────────────────────
logs_entries = d.get("logs", [])
print(f"\nlogs entries: {len(logs_entries)}")

all_algo_lines = []   # (timestamp, line)
trader_states = []    # (timestamp, state_dict)

for entry in logs_entries:
    try:
        ll = json.loads(entry["lambdaLog"])
    except (json.JSONDecodeError, KeyError):
        continue

    # ll is a list of tick-level data, each tick is:
    # item[0] = [ts, trader_data_str, listings, ...]
    # item[4] = algo print output string
    # Actually from inspection: ll is a list where each element corresponds to one tick
    # structure: [ts, trader_data_before, listings, orders_placed, new_trader_data, algo_output]
    # Let's be flexible

    for item in ll:
        if not isinstance(item, list):
            continue
        # Find the algo output string (last string element that has [TICK] or similar)
        ts_val = None
        algo_out = None
        trader_data = None
        for elem in item:
            if isinstance(elem, int) and ts_val is None:
                ts_val = elem
            elif isinstance(elem, str):
                if "[TICK]" in elem or "[HG" in elem or "[OPT]" in elem or "[INIT]" in elem or "[DAY]" in elem:
                    algo_out = elem
                else:
                    # Try to parse as JSON state
                    try:
                        state = json.loads(elem)
                        if isinstance(state, dict):
                            trader_data = (ts_val, state)
                    except (json.JSONDecodeError, TypeError):
                        pass
        if algo_out and ts_val is not None:
            for line in algo_out.split("\n"):
                line = line.strip()
                if line:
                    all_algo_lines.append((ts_val, line))
        if trader_data:
            trader_states.append(trader_data)

print(f"Algo output lines extracted: {len(all_algo_lines)}")

# ─── 1. OVERALL PnL BREAKDOWN ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. OVERALL PnL BREAKDOWN")
print("=" * 80)

final_pnls = {}
for prod, vals in product_pnl.items():
    vals_s = sorted(vals, key=lambda x: x[0])
    final_pnls[prod] = vals_s[-1][1]

print("\nProduct                           Final PnL")
total = 0.0
for prod, pnl in sorted(final_pnls.items(), key=lambda x: x[1]):
    print(f"  {prod:35s}  {pnl:+10.1f}")
    total += pnl
print(f"\n  {'TOTAL':35s}  {total:+10.1f}")

# ─── 2. OWN TRADES ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. OWN TRADES")
print("=" * 80)

buy_data = defaultdict(list)
sell_data = defaultdict(list)

for t in our_trades:
    sym = t.get("symbol", "?")
    price = float(t.get("price", 0))
    qty = int(t.get("quantity", 0))
    if t.get("buyer") == "SUBMISSION":
        buy_data[sym].extend([price] * abs(qty))
    else:
        sell_data[sym].extend([price] * abs(qty))

all_syms = sorted(set(list(buy_data.keys()) + list(sell_data.keys())))
if all_syms:
    print(f"\n  {'Product':30s}  {'#Buys':>7}  {'AvgBuy':>9}  {'#Sells':>7}  {'AvgSell':>9}  {'Spread':>8}")
    for sym in all_syms:
        buys = buy_data[sym]
        sells = sell_data[sym]
        nb = len(buys)
        ns = len(sells)
        ab = sum(buys)/nb if nb else 0
        as_ = sum(sells)/ns if ns else 0
        spread = as_ - ab if nb and ns else 0
        print(f"  {sym:30s}  {nb:7d}  {ab:9.2f}  {ns:7d}  {as_:9.2f}  {spread:+8.2f}")
else:
    print("  No own trades found (SUBMISSION not as buyer/seller in tradeHistory)")
    print("  Note: this likely means the format uses empty buyer/seller for our trades")
    # Show all unique buyer/seller combos
    bs_combos = defaultdict(int)
    for t in trade_history:
        b = t.get("buyer", "")
        s = t.get("seller", "")
        bs_combos[(b, s)] += 1
    print("\n  Buyer/Seller combinations in tradeHistory:")
    for (b, s), cnt in sorted(bs_combos.items(), key=lambda x: -x[1]):
        print(f"    buyer={b!r:20s}  seller={s!r:20s}  count={cnt}")

# ─── 3. ALGORITHM PRINT OUTPUT ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. ALGORITHM PRINT OUTPUT")
print("=" * 80)

# Count tags
tags = ["[INIT]", "[DAY]", "[TICK]", "[HG TAKE]", "[OPT]", "[CAP]", "[HEDGE]", "[FLOOR]",
        "[HG]", "[VF]", "[ERROR]", "[WARN]"]
tag_counts = defaultdict(int)
for _, line in all_algo_lines:
    for tag in tags:
        if tag in line:
            tag_counts[tag] += 1

print("\nTag counts:")
for tag in tags:
    cnt = tag_counts[tag]
    if cnt > 0:
        print(f"  {tag:15s}: {cnt:5d}")

# Show first few of each tag type
print("\nSample lines per tag:")
for tag in ["[TICK]", "[HG]", "[HG TAKE]", "[OPT]", "[CAP]", "[HEDGE]", "[FLOOR]", "[DAY]", "[INIT]"]:
    tag_lines = [(ts, l) for ts, l in all_algo_lines if tag in l]
    if tag_lines:
        print(f"\n  {tag} ({len(tag_lines)} total) - first 5:")
        for ts, l in tag_lines[:5]:
            print(f"    ts={ts:6d}: {l}")

# HG TAKE BUY vs SELL
hg_take = [(ts, l) for ts, l in all_algo_lines if "[HG TAKE]" in l]
hg_take_buy = [(ts, l) for ts, l in hg_take if "BUY" in l.upper()]
hg_take_sell = [(ts, l) for ts, l in hg_take if "SELL" in l.upper()]
print(f"\n[HG TAKE] total={len(hg_take)}  BUY={len(hg_take_buy)}  SELL={len(hg_take_sell)}")
if hg_take_buy:
    print(f"  Example BUY:  {hg_take_buy[0][1]}")
if hg_take_sell:
    print(f"  Example SELL: {hg_take_sell[0][1]}")

# OPT lines - parse for non-zero sizes
opt_lines = [(ts, l) for ts, l in all_algo_lines if "[OPT]" in l]
print(f"\n[OPT] total lines: {len(opt_lines)}")
# Parse: [OPT] VEV_5000 pos=0 bid=266×15 ask=271×15 iv=0.01374
opt_with_bids = []
opt_with_no_bid = []
for ts, line in opt_lines:
    # Check for bid/ask sizes > 0
    import re
    bid_match = re.search(r'bid=(\d+)×(\d+)', line)
    ask_match = re.search(r'ask=(\d+)×(\d+)', line)
    has_bid = bid_match and int(bid_match.group(2)) > 0
    has_ask = ask_match and int(ask_match.group(2)) > 0
    if has_bid or has_ask:
        opt_with_bids.append((ts, line))
    else:
        opt_with_no_bid.append((ts, line))
print(f"[OPT] with non-zero bid/ask sizes: {len(opt_with_bids)}")
print(f"[OPT] with no bid/ask sizes: {len(opt_with_no_bid)}")
if opt_with_bids:
    print("  Sample OPT with sizes:")
    for ts, l in opt_with_bids[:5]:
        print(f"    ts={ts}: {l}")
if opt_with_no_bid:
    print("  Sample OPT without sizes:")
    for ts, l in opt_with_no_bid[:5]:
        print(f"    ts={ts}: {l}")

# Error patterns
error_lines = [(ts, l) for ts, l in all_algo_lines if any(e in l.lower() for e in ["error", "exception", "traceback", "none", "nan"])]
print(f"\nPotential error lines: {len(error_lines)}")
for ts, l in error_lines[:10]:
    print(f"  ts={ts}: {l}")

# ─── 4. OPTIONS FILLS ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. OPTIONS FILLS (VEV_* trades)")
print("=" * 80)

vev_trades = [t for t in trade_history if "VEV" in t.get("symbol", "")]
print(f"All VEV_* trades in tradeHistory: {len(vev_trades)}")

# Our VEV trades
our_vev = [t for t in vev_trades if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
print(f"Our VEV_* trades: {len(our_vev)}")

# Market VEV trades (not ours)
mkt_vev = [t for t in vev_trades if t.get("buyer") != "SUBMISSION" and t.get("seller") != "SUBMISSION"]
print(f"Market-only VEV_* trades: {len(mkt_vev)}")

if our_vev:
    print("\nOur VEV fills:")
    for t in our_vev[:30]:
        side = "BUY" if t.get("buyer") == "SUBMISSION" else "SELL"
        print(f"  ts={t['timestamp']:6d}  {t['symbol']:12s}  {side}  price={t['price']:.1f}  qty={t['quantity']}")

# Per-symbol VEV trade counts
vev_sym_counts = defaultdict(lambda: {"mkt_trades": 0, "our_buys": 0, "our_sells": 0})
for t in vev_trades:
    sym = t["symbol"]
    if t.get("buyer") == "SUBMISSION":
        vev_sym_counts[sym]["our_buys"] += t["quantity"]
    elif t.get("seller") == "SUBMISSION":
        vev_sym_counts[sym]["our_sells"] += t["quantity"]
    else:
        vev_sym_counts[sym]["mkt_trades"] += 1

print("\nVEV_* trade summary:")
print(f"  {'Symbol':15s}  {'MktTrades':>10}  {'OurBuyQty':>10}  {'OurSellQty':>11}")
for sym in sorted(vev_sym_counts.keys()):
    c = vev_sym_counts[sym]
    print(f"  {sym:15s}  {c['mkt_trades']:10d}  {c['our_buys']:10d}  {c['our_sells']:11d}")

# ─── 5. POSITION TRAJECTORY ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. POSITION TRAJECTORY (PnL over time)")
print("=" * 80)

for prod_key in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_5000", "VEV_5100", "VEV_5200"]:
    if prod_key in product_pnl:
        vals = sorted(product_pnl[prod_key], key=lambda x: x[0])
        n = len(vals)
        idxs = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
        print(f"\n{prod_key}:")
        for i in idxs:
            ts, pnl = vals[i]
            print(f"  ts={ts:7d}  pnl={pnl:+10.2f}")

# ─── 6. ORDER BOOK PATTERNS FOR VEV OPTIONS ───────────────────────────────────
print("\n" + "=" * 80)
print("6. ORDER BOOK PATTERNS FOR VEV OPTIONS")
print("=" * 80)

vev_rows = [row for row in rows if "VEV" in row.get("product", "")]
print(f"VEV_* activity rows: {len(vev_rows)}")

import re
spreads_by_product = defaultdict(list)
for row in vev_rows:
    prod = row["product"]
    try:
        ask1 = float(row.get("ask_price_1") or 0)
        bid1 = float(row.get("bid_price_1") or 0)
        if ask1 > 0 and bid1 > 0:
            spreads_by_product[prod].append(ask1 - bid1)
    except (ValueError, TypeError):
        pass

print(f"\n{'Product':15s}  {'N':>5}  {'Min':>6}  {'Avg':>6}  {'Max':>6}  {'>=3 ticks':>10}")
for prod in sorted(spreads_by_product.keys()):
    sv = spreads_by_product[prod]
    n = len(sv)
    mn = min(sv)
    avg = sum(sv)/n
    mx = max(sv)
    ge3 = sum(1 for s in sv if s >= 3)
    print(f"  {prod:15s}  {n:5d}  {mn:6.1f}  {avg:6.2f}  {mx:6.1f}  {ge3:5d} ({100*ge3/n:.0f}%)")

# Sample specific timestamps
print("\nSample order books at ts=0:")
t0_rows = [row for row in vev_rows if row["timestamp"] == "0"]
for row in sorted(t0_rows, key=lambda x: x["product"]):
    prod = row["product"]
    bid1 = row.get("bid_price_1", "")
    bv1 = row.get("bid_volume_1", "")
    ask1 = row.get("ask_price_1", "")
    av1 = row.get("ask_volume_1", "")
    spread = float(ask1) - float(bid1) if ask1 and bid1 else "?"
    print(f"  {prod:12s}  bid={bid1}x{bv1}  ask={ask1}x{av1}  spread={spread}")

# ─── 7. HG TRADE ANALYSIS ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("7. HYDROGEL PACK TRADE ANALYSIS")
print("=" * 80)

hg_trades = [t for t in trade_history if t.get("symbol") == "HYDROGEL_PACK"]
hg_our = [t for t in hg_trades if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
print(f"HG total trades: {len(hg_trades)}")
print(f"HG our trades: {len(hg_our)}")

if hg_our:
    print("\nOur HG fills (all):")
    hg_buy_prices = []
    hg_sell_prices = []
    for t in hg_our[:50]:
        side = "BUY" if t.get("buyer") == "SUBMISSION" else "SELL"
        price = float(t["price"])
        qty = int(t["quantity"])
        if side == "BUY":
            hg_buy_prices.extend([price]*qty)
        else:
            hg_sell_prices.extend([price]*qty)
        print(f"  ts={t['timestamp']:6d}  {side}  price={price:.1f}  qty={qty}")
    if hg_buy_prices:
        print(f"\n  Avg buy price: {sum(hg_buy_prices)/len(hg_buy_prices):.2f}  (total qty={len(hg_buy_prices)})")
    if hg_sell_prices:
        print(f"  Avg sell price: {sum(hg_sell_prices)/len(hg_sell_prices):.2f}  (total qty={len(hg_sell_prices)})")
else:
    # Show market HG trades to understand price action
    print("\nSample HG market trades (price action):")
    for t in hg_trades[:20]:
        print(f"  ts={t['timestamp']:6d}  buyer={t.get('buyer',''):15s}  seller={t.get('seller',''):15s}  price={t['price']:.1f}  qty={t['quantity']}")

# HG PnL analysis from activitiesLog
hg_vals = sorted(product_pnl.get("HYDROGEL_PACK", []), key=lambda x: x[0])
print(f"\nHG PnL: start={hg_vals[0][1]:.1f}  end={hg_vals[-1][1]:.1f}")
# Find biggest single-step drop
if len(hg_vals) > 1:
    steps = [(hg_vals[i+1][0], hg_vals[i+1][1] - hg_vals[i][1]) for i in range(len(hg_vals)-1)]
    biggest_drops = sorted(steps, key=lambda x: x[1])[:10]
    biggest_gains = sorted(steps, key=lambda x: -x[1])[:5]
    print("\nBiggest HG PnL drops (single step):")
    for ts, delta in biggest_drops:
        print(f"  ts={ts:7d}  delta={delta:+10.2f}")
    print("\nBiggest HG PnL gains (single step):")
    for ts, delta in biggest_gains:
        print(f"  ts={ts:7d}  delta={delta:+10.2f}")

# ─── 8. KEY FINDINGS ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. KEY FINDINGS")
print("=" * 80)

# HG position from algo state
print("\nTrader state samples (hg_ema and day):")
for ts, state in trader_states[:5]:
    print(f"  ts={ts}: {state}")
if trader_states:
    # Show last few
    print("  ...")
    for ts, state in trader_states[-3:]:
        print(f"  ts={ts}: {state}")

# Final positions
print(f"\nFinal positions (from .json): see above analysis")
print(f"Algo trades total (from tradeHistory): {len(trade_history)}")
print(f"  HG trades: {len(hg_trades)}")
print(f"  VEV trades: {len(vev_trades)}")
vf_trades = [t for t in trade_history if t.get("symbol") == "VELVETFRUIT_EXTRACT"]
print(f"  VF trades: {len(vf_trades)}")

# VEV fills summary
total_vev_our_buys = sum(vev_sym_counts[s]["our_buys"] for s in vev_sym_counts)
total_vev_our_sells = sum(vev_sym_counts[s]["our_sells"] for s in vev_sym_counts)
print(f"\n  Our VEV option fills: {total_vev_our_buys} buy qty, {total_vev_our_sells} sell qty")

# HG take lines breakdown
print(f"\n[HG TAKE] breakdown: {len(hg_take_buy)} BUY events, {len(hg_take_sell)} SELL events")
if hg_take:
    print("  All [HG TAKE] lines:")
    for ts, l in hg_take:
        print(f"    ts={ts}: {l}")

# CAP / FLOOR details
cap_lines_all = [(ts, l) for ts, l in all_algo_lines if "[CAP]" in l]
floor_lines_all = [(ts, l) for ts, l in all_algo_lines if "[FLOOR]" in l]
hedge_lines_all = [(ts, l) for ts, l in all_algo_lines if "[HEDGE]" in l]
print(f"\n[CAP] lines: {len(cap_lines_all)}")
for ts, l in cap_lines_all[:10]:
    print(f"  ts={ts}: {l}")
print(f"\n[FLOOR] lines: {len(floor_lines_all)}")
for ts, l in floor_lines_all[:10]:
    print(f"  ts={ts}: {l}")
print(f"\n[HEDGE] lines: {len(hedge_lines_all)}")
for ts, l in hedge_lines_all[:10]:
    print(f"  ts={ts}: {l}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total PnL: {total:.1f}")
print(f"HYDROGEL_PACK: {final_pnls.get('HYDROGEL_PACK', 0):.1f}  ← MAIN LOSER")
print(f"VELVETFRUIT_EXTRACT: {final_pnls.get('VELVETFRUIT_EXTRACT', 0):.1f}")
vev_total = sum(final_pnls.get(p, 0) for p in final_pnls if "VEV" in p)
print(f"VEV options total: {vev_total:.1f}")
print(f"Our trade count: {len(our_trades)} (HG={len(hg_our)}, VEV={len(our_vev)})")

# Show OPT quoting activity
print(f"\nOPT quoting (market-making):")
# Parse OPT lines for what we were quoting
opt_product_quotes = defaultdict(lambda: {"ticks_quoted": 0, "spread_sufficient": 0})
import re
for ts, line in opt_lines:
    m = re.match(r'\[OPT\] (\S+) pos=([+-]?\d+) bid=(\S+)×(\d+) ask=(\S+)×(\d+)', line)
    if m:
        prod = m.group(1)
        pos = int(m.group(2))
        bid_sz = int(m.group(4))
        ask_sz = int(m.group(6))
        if bid_sz > 0 or ask_sz > 0:
            opt_product_quotes[prod]["ticks_quoted"] += 1
        # check market spread from activitiesLog
    else:
        # Try simpler pattern
        mm = re.search(r'\[OPT\] (\S+)', line)
        if mm:
            prod = mm.group(1)
            opt_product_quotes[prod]  # just register

for prod in sorted(opt_product_quotes.keys()):
    q = opt_product_quotes[prod]
    print(f"  {prod:15s}  ticks_with_nonzero_quote={q['ticks_quoted']}")

print("\nDone.")
