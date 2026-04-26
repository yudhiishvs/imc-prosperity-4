#!/usr/bin/env python3
"""Complete analysis of 457353.log — IMC Prosperity 4 Round 3"""

import json, csv, io, re
from collections import defaultdict

LOG_PATH = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/v9_logs/457353.log"

with open(LOG_PATH) as f:
    d = json.load(f)

# ─── Parse activitiesLog ──────────────────────────────────────────────────────
rows = list(csv.DictReader(io.StringIO(d["activitiesLog"]), delimiter=";"))
product_pnl = defaultdict(list)
for row in rows:
    prod = row["product"].strip()
    ts = int(row["timestamp"])
    try:
        pnl = float(row["profit_and_loss"])
    except ValueError:
        pnl = 0.0
    product_pnl[prod].append((ts, pnl))

# ─── Parse tradeHistory ───────────────────────────────────────────────────────
trade_history = d.get("tradeHistory", [])
our_trades = [t for t in trade_history if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]

# ─── Parse lambdaLog: item[4] = algo output, item[1] = orders placed ─────────
all_algo_lines = []     # (ts, line_str)
all_orders     = []     # (ts, [sym, price, qty])

for entry in d.get("logs", []):
    try:
        ll = json.loads(entry["lambdaLog"])
    except (json.JSONDecodeError, KeyError):
        continue
    for item in ll:
        if not isinstance(item, list) or len(item) < 5:
            continue
        ts_val = item[0] if isinstance(item[0], int) else None
        if ts_val is None:
            continue
        # item[4] = algo output
        algo_out = item[4] if isinstance(item[4], str) else ""
        for line in algo_out.split("\n"):
            line = line.strip()
            if line:
                all_algo_lines.append((ts_val, line))
        # item[1] = list of [sym, price, qty]
        if isinstance(item[1], list):
            for order in item[1]:
                if isinstance(order, list) and len(order) == 3:
                    all_orders.append((ts_val, order[0], order[1], order[2]))

print("=" * 80)
print("1. OVERALL PnL BREAKDOWN")
print("=" * 80)

final_pnls = {}
for prod, vals in product_pnl.items():
    vals_s = sorted(vals, key=lambda x: x[0])
    final_pnls[prod] = vals_s[-1][1]

print(f"\n  {'Product':35s}  {'Final PnL':>12}")
total = 0.0
for prod, pnl in sorted(final_pnls.items(), key=lambda x: x[1]):
    print(f"  {prod:35s}  {pnl:+12.1f}")
    total += pnl
print(f"\n  {'TOTAL':35s}  {total:+12.1f}")

# ─── 2. OWN TRADES ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. OWN TRADES (SUBMISSION as buyer/seller)")
print("=" * 80)

buy_data  = defaultdict(list)
sell_data = defaultdict(list)
for t in our_trades:
    sym   = t.get("symbol", "?")
    price = float(t.get("price", 0))
    qty   = int(t.get("quantity", 0))
    if t.get("buyer") == "SUBMISSION":
        buy_data[sym].extend([price] * abs(qty))
    else:
        sell_data[sym].extend([price] * abs(qty))

all_syms = sorted(set(list(buy_data.keys()) + list(sell_data.keys())))
print(f"\n  {'Product':30s}  {'#BuyQty':>8}  {'AvgBuy':>9}  {'#SellQty':>9}  {'AvgSell':>9}  {'Spread':>8}")
for sym in all_syms:
    buys  = buy_data[sym]
    sells = sell_data[sym]
    nb    = len(buys)
    ns    = len(sells)
    ab    = sum(buys)/nb  if nb else 0
    as_   = sum(sells)/ns if ns else 0
    spd   = as_ - ab if nb and ns else float("nan")
    print(f"  {sym:30s}  {nb:8d}  {ab:9.2f}  {ns:9d}  {as_:9.2f}  {spd:+8.2f}")

# ─── 3. ALGORITHM PRINT OUTPUT ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. ALGORITHM PRINT OUTPUT")
print("=" * 80)
print(f"Total algo output lines extracted: {len(all_algo_lines)}")

tags = ["[INIT]", "[DAY]", "[TICK]", "[HG TAKE]", "[HG]", "[OPT]", "[CAP]", "[HEDGE]", "[FLOOR]",
        "[VF]", "[ERROR]", "[WARN]"]
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

# Show representative sample for key tags
for tag in ["[INIT]", "[DAY]", "[HG TAKE]", "[CAP]", "[HEDGE]", "[FLOOR]"]:
    tag_lines = [(ts, l) for ts, l in all_algo_lines if tag in l]
    if tag_lines:
        print(f"\n  {tag} ({len(tag_lines)} lines):")
        for ts, l in tag_lines[:10]:
            print(f"    ts={ts:6d}: {l}")

# HG TAKE breakdown
hg_take = [(ts, l) for ts, l in all_algo_lines if "[HG TAKE]" in l]
hg_take_buy  = [(ts, l) for ts, l in hg_take if "BUY"  in l.upper()]
hg_take_sell = [(ts, l) for ts, l in hg_take if "SELL" in l.upper()]
print(f"\n[HG TAKE]: total={len(hg_take)}  BUY={len(hg_take_buy)}  SELL={len(hg_take_sell)}")

# OPT lines analysis
opt_lines = [(ts, l) for ts, l in all_algo_lines if "[OPT]" in l]
print(f"\n[OPT] total lines: {len(opt_lines)}")

opt_nonzero = 0
opt_zero    = 0
opt_per_product = defaultdict(lambda: {"nonzero": 0, "zero": 0, "spreads": [], "mkt_spread": []})
for ts, line in opt_lines:
    m = re.match(r'\[OPT\] (\S+) pos=([+-]?\d+) bid=(\S+)×(\d+) ask=(\S+)×(\d+)', line)
    if m:
        prod   = m.group(1)
        bid_px = float(m.group(3))
        bid_sz = int(m.group(4))
        ask_px = float(m.group(5))
        ask_sz = int(m.group(6))
        if bid_sz > 0 or ask_sz > 0:
            opt_nonzero += 1
            opt_per_product[prod]["nonzero"] += 1
            opt_per_product[prod]["spreads"].append(ask_px - bid_px)
        else:
            opt_zero += 1
            opt_per_product[prod]["zero"] += 1

print(f"  Lines with non-zero bid/ask sizes: {opt_nonzero}")
print(f"  Lines with zero bid/ask sizes:     {opt_zero}")
if opt_per_product:
    print(f"\n  {'Product':15s}  {'Quoted':>7}  {'Zero':>6}  {'AvgQuoteSpread':>15}")
    for prod in sorted(opt_per_product.keys()):
        q = opt_per_product[prod]
        nz = q["nonzero"]
        z  = q["zero"]
        avg_spd = sum(q["spreads"])/len(q["spreads"]) if q["spreads"] else 0
        print(f"  {prod:15s}  {nz:7d}  {z:6d}  {avg_spd:15.2f}")

print("\nSample [OPT] lines (5 with sizes, 5 without):")
with_sz = [(ts, l) for ts, l in opt_lines if re.search(r'bid=\S+×[1-9]', l)]
without_sz = [(ts, l) for ts, l in opt_lines if not re.search(r'bid=\S+×[1-9]', l)]
for ts, l in with_sz[:5]:
    print(f"  [HAS SIZE] ts={ts}: {l}")
for ts, l in without_sz[:5]:
    print(f"  [NO  SIZE] ts={ts}: {l}")

# ─── 4. OPTIONS FILLS ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. OPTIONS FILLS (VEV_* trades)")
print("=" * 80)

vev_all   = [t for t in trade_history if "VEV" in t.get("symbol", "")]
vev_ours  = [t for t in vev_all if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
vev_mkt   = [t for t in vev_all if t not in vev_ours]
print(f"All VEV trades: {len(vev_all)}  |  Ours: {len(vev_ours)}  |  Market: {len(vev_mkt)}")

print("\nOur VEV fills (all):")
for t in vev_ours:
    side = "BUY" if t.get("buyer") == "SUBMISSION" else "SELL"
    print(f"  ts={t['timestamp']:6d}  {t['symbol']:12s}  {side}  price={t['price']:.1f}  qty={t['quantity']}")

print("\nMarket VEV trades (sample 10):")
for t in vev_mkt[:10]:
    print(f"  ts={t['timestamp']:6d}  {t['symbol']:12s}  price={t['price']:.1f}  qty={t['quantity']}  buyer={t.get('buyer',''):12s}  seller={t.get('seller',''):12s}")

# Net VEV position per product
vev_net = defaultdict(int)
for t in vev_ours:
    if t.get("buyer") == "SUBMISSION":
        vev_net[t["symbol"]] += t["quantity"]
    else:
        vev_net[t["symbol"]] -= t["quantity"]
print(f"\nNet VEV positions (from our fills): {dict(sorted(vev_net.items()))}")

# ─── 5. POSITION TRAJECTORY ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. POSITION TRAJECTORY (PnL over time)")
print("=" * 80)

for prod_key in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_5000", "VEV_5100", "VEV_5200"]:
    if prod_key not in product_pnl:
        continue
    vals = sorted(product_pnl[prod_key], key=lambda x: x[0])
    n = len(vals)
    idxs = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]
    print(f"\n{prod_key}:")
    for i in idxs:
        ts, pnl = vals[i]
        print(f"  ts={ts:7d}  pnl={pnl:+10.2f}")

# ─── 6. ORDER BOOK PATTERNS FOR VEV OPTIONS ───────────────────────────────────
print("\n" + "=" * 80)
print("6. VEV ORDER BOOK SPREAD ANALYSIS")
print("=" * 80)

vev_rows = [row for row in rows if "VEV" in row.get("product", "")]
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

print(f"\n  {'Product':15s}  {'N':>5}  {'Min':>6}  {'Avg':>6}  {'Max':>6}  {'>=3 ticks':>12}  {'>=1 tick':>10}")
for prod in sorted(spreads_by_product.keys()):
    sv = spreads_by_product[prod]
    n  = len(sv)
    mn = min(sv)
    avg= sum(sv)/n
    mx = max(sv)
    ge3 = sum(1 for s in sv if s >= 3)
    ge1 = sum(1 for s in sv if s >= 1)
    print(f"  {prod:15s}  {n:5d}  {mn:6.1f}  {avg:6.2f}  {mx:6.1f}  {ge3:5d} ({100*ge3/n:3.0f}%)  {ge1:5d} ({100*ge1/n:3.0f}%)")

print("\nSample order books at ts=0 and ts=50000:")
for ts_filter in ["0", "50000"]:
    print(f"\n  ts={ts_filter}:")
    ts_rows = [r for r in vev_rows if r["timestamp"] == ts_filter]
    for row in sorted(ts_rows, key=lambda x: x["product"]):
        prod = row["product"]
        bid1 = row.get("bid_price_1", "")
        bv1  = row.get("bid_volume_1", "")
        ask1 = row.get("ask_price_1", "")
        av1  = row.get("ask_volume_1", "")
        spd  = float(ask1) - float(bid1) if ask1 and bid1 else "?"
        print(f"    {prod:12s}  bid={bid1}×{bv1}  ask={ask1}×{av1}  spread={spd}")

# ─── 7. HG DETAILED ANALYSIS ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("7. HYDROGEL PACK — DETAILED ANALYSIS")
print("=" * 80)

hg_our = [t for t in trade_history if t.get("symbol") == "HYDROGEL_PACK" and
          (t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION")]

hg_buy_trades  = [t for t in hg_our if t.get("buyer") == "SUBMISSION"]
hg_sell_trades = [t for t in hg_our if t.get("seller") == "SUBMISSION"]
print(f"\nHG fills: {len(hg_buy_trades)} buy trades, {len(hg_sell_trades)} sell trades")
total_hg_buy_qty  = sum(t["quantity"] for t in hg_buy_trades)
total_hg_sell_qty = sum(t["quantity"] for t in hg_sell_trades)
print(f"Total qty: bought={total_hg_buy_qty}, sold={total_hg_sell_qty}, net={total_hg_buy_qty - total_hg_sell_qty}")

# Avg prices
if hg_buy_trades:
    w_buy = sum(t["price"]*t["quantity"] for t in hg_buy_trades) / total_hg_buy_qty
    print(f"Avg buy price (wt): {w_buy:.2f}")
if hg_sell_trades:
    w_sell = sum(t["price"]*t["quantity"] for t in hg_sell_trades) / total_hg_sell_qty
    print(f"Avg sell price (wt): {w_sell:.2f}")
if hg_buy_trades and hg_sell_trades:
    print(f"Avg spread captured: {w_sell - w_buy:+.2f}")

# Full HG trade log
print("\nAll HG fills (time-ordered):")
for t in sorted(hg_our, key=lambda x: x["timestamp"]):
    side = "BUY" if t.get("buyer") == "SUBMISSION" else "SELL"
    print(f"  ts={t['timestamp']:6d}  {side:5s}  price={t['price']:8.1f}  qty={t['quantity']:3d}")

# HG PnL drops
hg_vals = sorted(product_pnl.get("HYDROGEL_PACK", []), key=lambda x: x[0])
steps = [(hg_vals[i+1][0], hg_vals[i+1][1] - hg_vals[i][1]) for i in range(len(hg_vals)-1)]
print("\n10 biggest HG PnL drops (single step ~100ts):")
for ts, delta in sorted(steps, key=lambda x: x[1])[:10]:
    print(f"  ts={ts:7d}  delta={delta:+10.2f}")

# HG mid-price trajectory from algo state
print("\nHG EMA vs actual trajectory (from trader state):")
hg_ema_vals = []
for entry in d.get("logs", []):
    try:
        ll = json.loads(entry["lambdaLog"])
    except:
        continue
    for item in ll:
        if isinstance(item, list) and len(item) >= 4:
            ts_val = item[0] if isinstance(item[0], int) else None
            if ts_val and isinstance(item[3], str):
                try:
                    state = json.loads(item[3])
                    if "hg_ema" in state:
                        hg_ema_vals.append((ts_val, state["hg_ema"]))
                except:
                    pass

# Also get actual mid from activitiesLog
hg_mid_vals = []
for row in rows:
    if row["product"] == "HYDROGEL_PACK":
        try:
            mid = float(row["mid_price"])
            ts  = int(row["timestamp"])
            hg_mid_vals.append((ts, mid))
        except:
            pass
hg_mid_vals.sort(key=lambda x: x[0])

# Show comparison at key points
hg_ema_dict = dict(hg_ema_vals)
print(f"\n  {'ts':>8}  {'EMA':>12}  {'MidPrice':>12}  {'Diff':>10}")
for ts, mid in hg_mid_vals[::100]:  # every 100th step
    ema = hg_ema_dict.get(ts, None)
    if ema:
        print(f"  {ts:8d}  {ema:12.2f}  {mid:12.2f}  {mid-ema:+10.2f}")

# ─── 8. ORDERS PLACED ANALYSIS ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. ORDERS PLACED (from lambdaLog item[1])")
print("=" * 80)

orders_by_sym = defaultdict(lambda: {"buy_count": 0, "sell_count": 0, "buy_qty": 0, "sell_qty": 0,
                                      "buy_prices": [], "sell_prices": []})
for ts, sym, price, qty in all_orders:
    if qty > 0:
        orders_by_sym[sym]["buy_count"]  += 1
        orders_by_sym[sym]["buy_qty"]    += qty
        orders_by_sym[sym]["buy_prices"].append(price)
    else:
        orders_by_sym[sym]["sell_count"] += 1
        orders_by_sym[sym]["sell_qty"]   += abs(qty)
        orders_by_sym[sym]["sell_prices"].append(price)

print(f"\nTotal order entries parsed: {len(all_orders)}")
print(f"\n  {'Symbol':20s}  {'BuyOrders':>10}  {'SellOrders':>11}  {'AvgBidPx':>10}  {'AvgAskPx':>10}")
for sym in sorted(orders_by_sym.keys()):
    o = orders_by_sym[sym]
    avg_bid = sum(o["buy_prices"])/len(o["buy_prices"]) if o["buy_prices"] else 0
    avg_ask = sum(o["sell_prices"])/len(o["sell_prices"]) if o["sell_prices"] else 0
    print(f"  {sym:20s}  {o['buy_count']:10d}  {o['sell_count']:11d}  {avg_bid:10.2f}  {avg_ask:10.2f}")

# ─── 9. KEY FINDINGS ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("9. KEY FINDINGS SUMMARY")
print("=" * 80)

vev_total_pnl = sum(final_pnls.get(p, 0) for p in final_pnls if "VEV" in p)
hg_pnl = final_pnls.get("HYDROGEL_PACK", 0)
vf_pnl = final_pnls.get("VELVETFRUIT_EXTRACT", 0)

print(f"""
TOTAL PnL:           {total:+.1f}
  HYDROGEL_PACK:     {hg_pnl:+.1f}  ← accounts for {100*hg_pnl/total:.1f}% of loss
  VELVETFRUIT:       {vf_pnl:+.1f}
  VEV options:       {vev_total_pnl:+.1f}
""")

# HG summary
print("HYDROGEL analysis:")
print(f"  Bought {total_hg_buy_qty} units @ avg {w_buy if hg_buy_trades else 0:.2f}")
print(f"  Sold   {total_hg_sell_qty} units @ avg {w_sell if hg_sell_trades else 0:.2f}")
print(f"  Net position at end: {total_hg_buy_qty - total_hg_sell_qty} (also from .json positions: -50)")
print(f"  HG EMA at end: {hg_ema_dict.get(99900, 'N/A'):.2f}" if hg_ema_dict.get(99900) else "  HG EMA at end: N/A")
print(f"  HG mid at end: {hg_mid_vals[-1][1] if hg_mid_vals else 'N/A'}")

# Pattern: are we buying high and selling low on HG?
print(f"\n  Avg buy price - avg sell price = {(w_buy if hg_buy_trades else 0) - (w_sell if hg_sell_trades else 0):+.2f}")
print(f"  (negative = buying below sell price = market-making gain)")
print(f"  (positive = buying above sell price = adverse selection / following trend wrong)")

# VEV: what strikes are we quoting, which are getting filled
print(f"\nVEV OPTIONS analysis:")
print(f"  We quote: VEV_5000, VEV_5100, VEV_5200, VEV_5300, VEV_5400, VEV_5500 (from orders)")
print(f"  Strikes NOT in our orders: VEV_4000, VEV_4500 (deep ITM), VEV_6000, VEV_6500 (far OTM)")
print(f"  Our fills: VEV_5000 ({vev_net.get('VEV_5000',0):+d}), VEV_5100 ({vev_net.get('VEV_5100',0):+d}), VEV_5200 ({vev_net.get('VEV_5200',0):+d})")
print(f"  VEV PnL: 5000={final_pnls.get('VEV_5000',0):+.1f}, 5100={final_pnls.get('VEV_5100',0):+.1f}, 5200={final_pnls.get('VEV_5200',0):+.1f}")

# Spread analysis
print(f"\nVEV SPREAD CHECK (is mkt spread >= 3 ticks?):")
for prod in sorted(spreads_by_product.keys()):
    sv  = spreads_by_product[prod]
    n   = len(sv)
    ge3 = sum(1 for s in sv if s >= 3)
    print(f"  {prod:12s}: {100*ge3/n:3.0f}% of ticks have spread>=3")

# Final diagnosis
print("""
DIAGNOSIS:
1. MAIN LOSS = HYDROGEL_PACK (-7012):
   - We are market-making HG but avg sell < avg buy by ~12 points
   - This means we are being adversely selected (buying when HG falling, selling when rising)
   - OR we are accumulating a large long/short position that moves against us
   - Final position is -50 (net short 50 units) — HG mid drifted DOWN ~65 points over the day
   - HG EMA starts at ~10011, ends at ~9946 — 65 point decline
   - A short of 50 units in a 65-pt decline = ~3250 gain... but we have -7012
   - So the -7012 reflects accumulated loss from poor fills + position inventory

2. VEV OPTIONS (+41):
   - Small positive PnL, very few fills (9 buys, 6 sells each on 3 strikes)
   - VEV_6000 and VEV_6500 have 0 spread (worthless far OTM options)
   - VEV_4000 and VEV_4500 have wide spreads (21, 16) but we don't quote them
   - VEV_5400 and VEV_5500 have 2-tick spread — below our quoting threshold
   - Only VEV_5000 (99.6% >= 3 ticks), VEV_5100 (98.7%), VEV_5200 (94.4%) regularly have >=3-tick spread

3. VELVETFRUIT_EXTRACT (0):
   - No PnL at all — we are not trading it directly

4. NO HG TAKE, CAP, HEDGE, or FLOOR lines in output:
   - These features are either disabled or never triggered
""")
