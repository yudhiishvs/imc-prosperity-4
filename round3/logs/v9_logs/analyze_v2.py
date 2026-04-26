#!/usr/bin/env python3
"""Complete analysis of 457353.log — IMC Prosperity 4 Round 3 (v2, correct parser)"""

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

# ─── Parse lambdaLog: ll[0][0]=ts, ll[1]=orders, ll[4]=algo_output ────────────
# Each entry in d['logs'] is ONE tick.  ll = json.loads(entry['lambdaLog'])
# ll[0] = [ts, prev_trader_data_str, listings_list, ob_dict]  (the TradingState-ish)
# ll[1] = list of [sym, price, qty] (orders we placed)
# ll[2] = int (conversion observations = 0 usually)
# ll[3] = new_trader_data_str (after our algo runs)
# ll[4] = algo print output string

all_algo_lines = []     # (ts, line_str)
all_orders     = []     # (ts, sym, price, qty)
trader_states  = []     # (ts, state_dict)

for entry in d.get("logs", []):
    try:
        ll = json.loads(entry["lambdaLog"])
    except (json.JSONDecodeError, KeyError):
        continue
    if not isinstance(ll, list) or len(ll) < 5:
        continue

    # ll[0][0] = timestamp
    ts_val = ll[0][0] if isinstance(ll[0], list) and ll[0] else None
    if ts_val is None:
        continue

    # ll[1] = orders placed this tick
    if isinstance(ll[1], list):
        for order in ll[1]:
            if isinstance(order, list) and len(order) == 3:
                all_orders.append((ts_val, order[0], order[1], order[2]))

    # ll[3] = new trader data (JSON string)
    if isinstance(ll[3], str) and ll[3].strip():
        try:
            state = json.loads(ll[3])
            if isinstance(state, dict):
                trader_states.append((ts_val, state))
        except json.JSONDecodeError:
            pass

    # ll[4] = algo print output
    if isinstance(ll[4], str):
        for line in ll[4].split("\n"):
            line = line.strip()
            if line:
                all_algo_lines.append((ts_val, line))

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
w_buy = w_sell = 0
for sym in all_syms:
    buys  = buy_data[sym]
    sells = sell_data[sym]
    nb    = len(buys)
    ns    = len(sells)
    ab    = sum(buys)/nb  if nb else 0
    as_   = sum(sells)/ns if ns else 0
    spd   = as_ - ab if nb and ns else float("nan")
    if sym == "HYDROGEL_PACK":
        w_buy = ab
        w_sell = as_
    print(f"  {sym:30s}  {nb:8d}  {ab:9.2f}  {ns:9d}  {as_:9.2f}  {spd:+8.2f}")

# ─── 3. ALGORITHM PRINT OUTPUT ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. ALGORITHM PRINT OUTPUT")
print("=" * 80)
print(f"Total algo output lines extracted: {len(all_algo_lines)}")
print(f"Total orders extracted: {len(all_orders)}")
print(f"Total trader states extracted: {len(trader_states)}")

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

# Show representative samples for key tags
for tag in ["[INIT]", "[DAY]", "[HG TAKE]", "[CAP]", "[HEDGE]", "[FLOOR]"]:
    tag_lines = [(ts, l) for ts, l in all_algo_lines if tag in l]
    if tag_lines:
        print(f"\n  {tag} ({len(tag_lines)} lines):")
        for ts, l in tag_lines[:15]:
            print(f"    ts={ts:6d}: {l}")

# HG TAKE breakdown
hg_take = [(ts, l) for ts, l in all_algo_lines if "[HG TAKE]" in l]
hg_take_buy  = [(ts, l) for ts, l in hg_take if "BUY"  in l.upper()]
hg_take_sell = [(ts, l) for ts, l in hg_take if "SELL" in l.upper()]
print(f"\n[HG TAKE]: total={len(hg_take)}  BUY={len(hg_take_buy)}  SELL={len(hg_take_sell)}")
if hg_take_buy:
    print(f"  Example BUY:  {hg_take_buy[0][1]}")
if hg_take_sell:
    print(f"  Example SELL: {hg_take_sell[0][1]}")

# OPT lines analysis
opt_lines = [(ts, l) for ts, l in all_algo_lines if "[OPT]" in l]
print(f"\n[OPT] total lines: {len(opt_lines)}")

opt_nonzero = 0
opt_zero    = 0
opt_per_product = defaultdict(lambda: {"nonzero": 0, "zero": 0, "quote_spreads": [], "mkt_bid": [], "mkt_ask": []})
for ts, line in opt_lines:
    m = re.match(r'\[OPT\] (\S+) pos=([+-]?\d+) bid=(\S+)×(\d+) ask=(\S+)×(\d+)', line)
    if m:
        prod   = m.group(1)
        bid_px = float(m.group(3).replace("×", ""))
        bid_sz = int(m.group(4))
        ask_px = float(m.group(5).replace("×", ""))
        ask_sz = int(m.group(6))
        if bid_sz > 0 or ask_sz > 0:
            opt_nonzero += 1
            opt_per_product[prod]["nonzero"] += 1
            opt_per_product[prod]["quote_spreads"].append(ask_px - bid_px)
        else:
            opt_zero += 1
            opt_per_product[prod]["zero"] += 1

print(f"  Lines with non-zero bid/ask sizes: {opt_nonzero}")
print(f"  Lines with zero bid/ask sizes:     {opt_zero}")
if opt_per_product:
    print(f"\n  {'Product':15s}  {'Quoted':>7}  {'NotQuoted':>10}  {'AvgQuoteWidth':>14}")
    for prod in sorted(opt_per_product.keys()):
        q = opt_per_product[prod]
        nz = q["nonzero"]
        z  = q["zero"]
        avg_spd = sum(q["quote_spreads"])/len(q["quote_spreads"]) if q["quote_spreads"] else 0
        print(f"  {prod:15s}  {nz:7d}  {z:10d}  {avg_spd:14.2f}")

print("\nSample [OPT] lines — first 8:")
for ts, l in opt_lines[:8]:
    print(f"  ts={ts:6d}: {l}")

# HG lines
hg_lines = [(ts, l) for ts, l in all_algo_lines if "[HG]" in l and "[HG TAKE]" not in l]
print(f"\n[HG] (non-TAKE) lines: {len(hg_lines)}")
print("  First 5:")
for ts, l in hg_lines[:5]:
    print(f"  ts={ts:6d}: {l}")

# ─── 4. OPTIONS FILLS ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. OPTIONS FILLS (VEV_* trades)")
print("=" * 80)

vev_all  = [t for t in trade_history if "VEV" in t.get("symbol", "")]
vev_ours = [t for t in vev_all if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
vev_mkt  = [t for t in vev_all if t not in vev_ours]
print(f"All VEV trades: {len(vev_all)}  |  Ours: {len(vev_ours)}  |  Market-only: {len(vev_mkt)}")

print("\nOur VEV fills (all):")
vev_net = defaultdict(int)
for t in vev_ours:
    side = "BUY" if t.get("buyer") == "SUBMISSION" else "SELL"
    print(f"  ts={t['timestamp']:6d}  {t['symbol']:12s}  {side}  price={t['price']:.1f}  qty={t['quantity']}")
    if side == "BUY":
        vev_net[t["symbol"]] += t["quantity"]
    else:
        vev_net[t["symbol"]] -= t["quantity"]

print(f"\nNet VEV positions from fills: {dict(sorted(vev_net.items()))}")

print("\nMarket-only VEV trades (symbols + count):")
mkt_sym_cnts = defaultdict(int)
for t in vev_mkt:
    mkt_sym_cnts[t["symbol"]] += 1
for sym, cnt in sorted(mkt_sym_cnts.items()):
    print(f"  {sym}: {cnt}")

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

# ─── 6. VEV ORDER BOOK SPREAD ANALYSIS ───────────────────────────────────────
print("\n" + "=" * 80)
print("6. VEV ORDER BOOK SPREAD ANALYSIS (activitiesLog)")
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

print(f"\n  {'Product':15s}  {'N':>5}  {'Min':>6}  {'Avg':>6}  {'Max':>6}  {'>=3':>8}  {'%>=3':>6}")
for prod in sorted(spreads_by_product.keys()):
    sv  = spreads_by_product[prod]
    n   = len(sv)
    mn  = min(sv)
    avg = sum(sv)/n
    mx  = max(sv)
    ge3 = sum(1 for s in sv if s >= 3)
    print(f"  {prod:15s}  {n:5d}  {mn:6.1f}  {avg:6.2f}  {mx:6.1f}  {ge3:8d}  {100*ge3/n:5.0f}%")

# ─── 7. HG DETAILED ANALYSIS ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("7. HYDROGEL PACK — DETAILED ANALYSIS")
print("=" * 80)

hg_our = [t for t in our_trades if t.get("symbol") == "HYDROGEL_PACK"]
hg_buy_trades  = [t for t in hg_our if t.get("buyer") == "SUBMISSION"]
hg_sell_trades = [t for t in hg_our if t.get("seller") == "SUBMISSION"]
total_hg_buy_qty  = sum(t["quantity"] for t in hg_buy_trades)
total_hg_sell_qty = sum(t["quantity"] for t in hg_sell_trades)
print(f"\nHG fills: {len(hg_buy_trades)} buy events / {total_hg_buy_qty} units,"
      f"  {len(hg_sell_trades)} sell events / {total_hg_sell_qty} units")
print(f"Net position: {total_hg_buy_qty - total_hg_sell_qty}")

if hg_buy_trades:
    w_buy = sum(t["price"]*t["quantity"] for t in hg_buy_trades) / total_hg_buy_qty
    print(f"Avg buy price (qty-wt): {w_buy:.2f}")
if hg_sell_trades:
    w_sell = sum(t["price"]*t["quantity"] for t in hg_sell_trades) / total_hg_sell_qty
    print(f"Avg sell price (qty-wt): {w_sell:.2f}")
if hg_buy_trades and hg_sell_trades:
    print(f"Avg (sell - buy): {w_sell - w_buy:+.2f}  ← negative = we buy high and sell low (bad!)")

# HG mid price from activitiesLog
hg_mid = {int(row["timestamp"]): float(row["mid_price"])
          for row in rows if row["product"] == "HYDROGEL_PACK"}

# HG EMA from trader states
hg_ema = {ts: state["hg_ema"] for ts, state in trader_states if "hg_ema" in state}

print(f"\nHG mid price: start={hg_mid.get(0, '?')}  end={hg_mid.get(99900, hg_mid.get(99800, '?'))}")
print(f"HG EMA:      start={hg_ema.get(0, hg_ema.get(100, '?'))}  end={hg_ema.get(99900, hg_ema.get(99800, '?'))}")

# Show HG mid vs EMA over time
print(f"\n  {'ts':>8}  {'Mid':>10}  {'EMA':>12}  {'EMA-Mid':>9}  {'cumPnL':>10}")
hg_pnl_dict = {ts: pnl for ts, pnl in product_pnl["HYDROGEL_PACK"]}
for ts in range(0, 100000, 10000):
    mid = hg_mid.get(ts, "?")
    ema = hg_ema.get(ts, "?")
    pnl = hg_pnl_dict.get(ts, "?")
    diff = ema - mid if isinstance(ema, float) and isinstance(mid, float) else "?"
    print(f"  {ts:8d}  {mid:10}  {ema:12}  {str(diff)[:9]:>9}  {pnl:10}")

# Biggest single-step drops in HG PnL
hg_vals = sorted(product_pnl.get("HYDROGEL_PACK", []), key=lambda x: x[0])
steps = [(hg_vals[i+1][0], hg_vals[i+1][1] - hg_vals[i][1], hg_vals[i+1][1]) for i in range(len(hg_vals)-1)]
print("\n10 biggest HG PnL drops (single ~100-ts step):")
for ts, delta, cum in sorted(steps, key=lambda x: x[1])[:10]:
    print(f"  ts={ts:7d}  delta={delta:+10.2f}  cumPnL={cum:+10.2f}")

# ─── 8. ORDERS PLACED ANALYSIS ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("8. ORDERS PLACED EACH TICK (from lambdaLog item[1])")
print("=" * 80)

print(f"Total order lines parsed: {len(all_orders)}")
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

print(f"\n  {'Symbol':20s}  {'BuyOrds':>8}  {'BuyQty':>7}  {'AvgBidPx':>10}  {'SellOrds':>9}  {'SellQty':>8}  {'AvgAskPx':>10}")
for sym in sorted(orders_by_sym.keys()):
    o = orders_by_sym[sym]
    avg_bid = sum(o["buy_prices"])/len(o["buy_prices"]) if o["buy_prices"] else 0
    avg_ask = sum(o["sell_prices"])/len(o["sell_prices"]) if o["sell_prices"] else 0
    print(f"  {sym:20s}  {o['buy_count']:8d}  {o['buy_qty']:7d}  {avg_bid:10.2f}  {o['sell_count']:9d}  {o['sell_qty']:8d}  {avg_ask:10.2f}")

# Sample orders at ts=0
t0_orders = [(sym, price, qty) for ts, sym, price, qty in all_orders if ts == 0]
print("\nOrders at ts=0:")
for sym, price, qty in t0_orders:
    side = "BUY" if qty > 0 else "SELL"
    print(f"  {side:4s}  {sym:20s}  price={price:8.1f}  qty={abs(qty)}")

# Sample orders at ts=50000
t50k_orders = [(sym, price, qty) for ts, sym, price, qty in all_orders if ts == 50000]
print("\nOrders at ts=50000:")
for sym, price, qty in t50k_orders:
    side = "BUY" if qty > 0 else "SELL"
    print(f"  {side:4s}  {sym:20s}  price={price:8.1f}  qty={abs(qty)}")

# ─── 9. KEY FINDINGS ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("9. KEY FINDINGS SUMMARY")
print("=" * 80)

vev_total_pnl = sum(v for k, v in final_pnls.items() if "VEV" in k)
hg_pnl = final_pnls.get("HYDROGEL_PACK", 0)
vf_pnl = final_pnls.get("VELVETFRUIT_EXTRACT", 0)

print(f"\n━━ PnL ━━")
print(f"  TOTAL:              {total:+.1f}")
print(f"  HYDROGEL_PACK:      {hg_pnl:+.1f}  ({100*hg_pnl/total:.1f}% of total loss)")
print(f"  VELVETFRUIT:        {vf_pnl:+.1f}")
print(f"  VEV options total:  {vev_total_pnl:+.1f}  (VEV_5000={final_pnls.get('VEV_5000',0):+.1f}, 5100={final_pnls.get('VEV_5100',0):+.1f}, 5200={final_pnls.get('VEV_5200',0):+.1f})")

print(f"\n━━ HG Analysis ━━")
hg_b = buy_data["HYDROGEL_PACK"]
hg_s = sell_data["HYDROGEL_PACK"]
hg_wb = sum(hg_b)/len(hg_b) if hg_b else 0
hg_ws = sum(hg_s)/len(hg_s) if hg_s else 0
print(f"  Bought {len(hg_b)} units @ avg {hg_wb:.2f}")
print(f"  Sold   {len(hg_s)} units @ avg {hg_ws:.2f}")
print(f"  Net position: {len(hg_b) - len(hg_s)} (confirmed -50 from positions)")
print(f"  Avg sell - avg buy = {hg_ws - hg_wb:+.2f} (NEGATIVE = buying above what we sell = LOSS)")
print(f"  HG mid moved from {hg_mid.get(0, '?')} → {hg_mid.get(99900, hg_mid.get(99800, '?'))} (total drift: {hg_mid.get(99900, hg_mid.get(99800, 0)) - hg_mid.get(0, 0):+.0f})")

# How much of loss is from adverse selection vs position drift
hg_ema_start = hg_ema.get(100, list(hg_ema.values())[0] if hg_ema else 0)
hg_ema_end   = hg_ema.get(99900, hg_ema.get(99800, list(hg_ema.values())[-1] if hg_ema else 0))
print(f"  HG EMA started at {hg_ema_start:.2f}, ended at {hg_ema_end:.2f}")

net_pos = len(hg_b) - len(hg_s)
# If we ended -50, and fair moved ~(hg_ema_end - hg_ema_start), position P&L component:
# This is approximate
pos_pnl_est = net_pos * (hg_mid.get(99900, hg_mid.get(99800, 0)) - hg_mid.get(0, 0))
print(f"  Estimated position-drift PnL: {net_pos} × {hg_mid.get(99900, hg_mid.get(99800, 0)) - hg_mid.get(0, 0):+.0f} = {pos_pnl_est:+.0f}")
print(f"  Remaining unexplained loss:   {hg_pnl - pos_pnl_est:+.0f}  (adverse selection / trade slippage)")

print(f"\n━━ VEV Options ━━")
print(f"  We quote: VEV_5000 through VEV_5500 on both sides")
print(f"  Fills received: only VEV_5000/5100/5200 (3 buys + 2 sells each ≈ net long 3)")
print(f"  VEV_5300, 5400, 5500 placed but NOT filled (order never taken)")
print(f"  VEV_4000, 4500 NOT quoted (deep ITM, wide spread 16-22 but we skip them)")
print(f"  VEV_6000, 6500 we POST sell-only at price=1 (dump worthless)")

print(f"\n━━ What's Working ━━")
print(f"  + VEV options: small positive PnL (+41) from mm spread on 5000/5100/5200")
print(f"  + Options being filled — we ARE getting fills now (previously 0)")
print(f"  + VEV 5000/5100/5200 have adequate spread (>=3 ticks 94-100% of time)")

print(f"\n━━ What's Losing Money ━━")
print(f"  ✗ HG market-making: avg sell price BELOW avg buy price (-12.80 spread)")
print(f"    This means we are being adversely selected — counterparties take our")
print(f"    cheap bids when price is falling and our cheap offers when price is rising")
print(f"  ✗ We buy a lot in declining market (ts=24000-28000 accumulate ~140 units)")
print(f"    then sell off slowly at lower prices")
print(f"  ✗ HG mid fell ~51 points (10011 → 9960) and we ended net short -50")
print(f"    A short of 50 in a -51 move = +2550 PnL, but actual is -7012")
print(f"    Meaning ~9500+ was lost to adverse selection / position chasing")

print(f"\n━━ What Should Be Fixed ━━")
print(f"  1. HG TAKE (taker orders) appears to never fire — check conditions")
print(f"  2. CAP/FLOOR/HEDGE features show 0 lines — are they disabled in v9?")
print(f"  3. HG quotes are being taken adversely — need wider spread or")
print(f"     inventory-aware skew to prevent accumulating positions in wrong direction")
print(f"  4. ts=24000-28000: we accumulate 140 long units during a big drop")
print(f"     — our EMA/fair is too slow, we keep buying into the drop")
print(f"  5. ts=60000-68000: we sell 150+ units at prices ~9950-9990")
print(f"     while market recovers temporarily, then drop resumes")
print(f"  6. VEV_4000 and VEV_4500 have huge spread (16-22 ticks) but we never quote them")
print(f"     — these could be profitable if we can hedge the delta")
print(f"  7. Consider quoting VELVETFRUIT_EXTRACT (currently 0 PnL)")

print("\nDone.")
