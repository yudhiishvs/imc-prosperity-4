"""
analyze_osmium_deep.py
Deep fill analysis for 177226 Osmium strategy.

Usage:
    python3 vedant/analyze_osmium_deep.py best_strat_logs/177226.log
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PRODUCT = "ASH_COATED_OSMIUM"
LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("best_strat_logs/177226.log")

EMA_ALPHA   = 0.2     # as used in 177226.py
BASE_SIZE   = 40
KILL_THRESH = 70


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_activities(raw_csv):
    rows = raw_csv.strip().split("\n")
    header = rows[0].split(";")
    records = []
    for row in rows[1:]:
        parts = row.split(";")
        if len(parts) < len(header):
            continue
        r = dict(zip(header, parts))
        if r.get("product", "").strip() != PRODUCT:
            continue
        def fv(k): return float(r[k]) if r.get(k, "").strip() else None
        records.append({
            "ts":    int(r["timestamp"]),
            "bid1":  fv("bid_price_1"),   "bid1v": fv("bid_volume_1"),
            "bid2":  fv("bid_price_2"),   "bid2v": fv("bid_volume_2"),
            "ask1":  fv("ask_price_1"),   "ask1v": fv("ask_volume_1"),
            "ask2":  fv("ask_price_2"),   "ask2v": fv("ask_volume_2"),
            "mid":   fv("mid_price"),
            "pnl":   fv("profit_and_loss"),
        })
    return records


def parse_trades(trade_history):
    buys, sells = [], []
    for t in trade_history:
        if t.get("symbol") != PRODUCT:
            continue
        rec = {"ts": t["timestamp"], "price": float(t["price"]), "qty": int(t["quantity"])}
        if t.get("buyer") == "SUBMISSION":
            buys.append(rec)
        elif t.get("seller") == "SUBMISSION":
            sells.append(rec)
    return buys, sells


def rebuild_position(buys, sells, timestamps):
    events = sorted(
        [(t["ts"], +t["qty"]) for t in buys] +
        [(t["ts"], -t["qty"]) for t in sells]
    )
    pos = 0; idx = 0; pos_by_ts = {}
    for ts in timestamps:
        while idx < len(events) and events[idx][0] <= ts:
            pos += events[idx][1]; idx += 1
        pos_by_ts[ts] = pos
    return pos_by_ts


def compute_ema(acts, alpha):
    ema_by_ts = {}
    ema = None
    for r in acts:
        if r["mid"] is None:
            ema_by_ts[r["ts"]] = ema
            continue
        ema = r["mid"] if ema is None else alpha * r["mid"] + (1 - alpha) * ema
        ema_by_ts[r["ts"]] = ema
    return ema_by_ts


# ─── ANALYSIS SECTIONS ────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def analyze_fill_vs_mid(buys, sells, acts):
    """Classify fills into: passive maker (we were limit-quoted, got hit)
    vs active taker (we crossed the spread to hit someone else).
    Proxy: if our fill price == market bid (for sells) or ask (for buys), we're a MAKER.
    """
    section("FILL TYPE ANALYSIS: Maker vs Taker")

    mid_by_ts  = {r["ts"]: r["mid"]  for r in acts if r["mid"]  is not None}
    bid_by_ts  = {r["ts"]: r["bid1"] for r in acts if r["bid1"] is not None}
    ask_by_ts  = {r["ts"]: r["ask1"] for r in acts if r["ask1"] is not None}

    buy_maker = buy_taker = 0
    buy_maker_vol = buy_taker_vol = 0
    for t in buys:
        ask = ask_by_ts.get(t["ts"])
        if ask is None:
            continue
        if t["price"] <= ask:   # we posted a buy limit and it got hit at or below ask
            buy_maker += 1;     buy_maker_vol += t["qty"]
        else:
            buy_taker += 1;     buy_taker_vol += t["qty"]

    sell_maker = sell_taker = 0
    sell_maker_vol = sell_taker_vol = 0
    for t in sells:
        bid = bid_by_ts.get(t["ts"])
        if bid is None:
            continue
        if t["price"] >= bid:   # we posted a sell limit and it got hit at or above bid
            sell_maker += 1;    sell_maker_vol += t["qty"]
        else:
            sell_taker += 1;    sell_taker_vol += t["qty"]

    total_vol = buy_maker_vol + buy_taker_vol + sell_maker_vol + sell_taker_vol
    print(f"  BUY fills:    maker={buy_maker_vol:,}u ({100*buy_maker_vol/max(1,total_vol):.1f}%)  "
          f"taker={buy_taker_vol:,}u ({100*buy_taker_vol/max(1,total_vol):.1f}%)")
    print(f"  SELL fills:   maker={sell_maker_vol:,}u ({100*sell_maker_vol/max(1,total_vol):.1f}%)  "
          f"taker={sell_taker_vol:,}u ({100*sell_taker_vol/max(1,total_vol):.1f}%)")

    maker_pct = 100 * (buy_maker_vol + sell_maker_vol) / max(1, total_vol)
    print(f"\n  Overall MAKER fill rate: {maker_pct:.1f}%")
    if maker_pct < 50:
        print("  ❌ Less than half our volume is passive — we are losing spread on most fills.")
    else:
        print("  ✅ Majority of volume is passive maker flow — spread captured properly.")


def analyze_oim(acts, buys, sells):
    """Order Imbalance analysis: does OIM at time of fill predict adverse price moves?"""
    section("ORDER BOOK IMBALANCE (OIM) AT FILL TIMES")

    oim_by_ts = {}
    for r in acts:
        b, a = r["bid1v"], r["ask1v"]
        if b is not None and a is not None and b + a > 0:
            oim_by_ts[r["ts"]] = (b - a) / (b + a)

    # OIM at buy fills
    buy_oims = [oim_by_ts[t["ts"]] for t in buys if t["ts"] in oim_by_ts]
    sell_oims = [oim_by_ts[t["ts"]] for t in sells if t["ts"] in oim_by_ts]

    if buy_oims:
        avg_buy_oim = sum(buy_oims) / len(buy_oims)
        print(f"  Avg OIM at BUY fills:  {avg_buy_oim:+.3f}  "
              f"({'bid-heavy ✅' if avg_buy_oim > 0 else 'ask-heavy ❌ = adverse selection'})")
    if sell_oims:
        avg_sell_oim = sum(sell_oims) / len(sell_oims)
        print(f"  Avg OIM at SELL fills: {avg_sell_oim:+.3f}  "
              f"({'ask-heavy ✅' if avg_sell_oim < 0 else 'bid-heavy ❌ = adverse selection'})")

    # OIM regime breakdown: how often was OIM strongly against us?
    adv_buys  = sum(1 for o in buy_oims  if o < -0.3)
    adv_sells = sum(1 for o in sell_oims if o > +0.3)
    print(f"\n  BUY fills with OIM < -0.3 (selling pressure):  {adv_buys}/{len(buy_oims)} ({100*adv_buys/max(1,len(buy_oims)):.0f}%)")
    print(f"  SELL fills with OIM > +0.3 (buying pressure):  {adv_sells}/{len(sell_oims)} ({100*adv_sells/max(1,len(sell_oims)):.0f}%)")

    # OIM vs subsequent 3-tick price move
    mids = {r["ts"]: r["mid"] for r in acts if r["mid"] is not None}
    ts_list = sorted(mids.keys())
    ts_idx = {ts: i for i, ts in enumerate(ts_list)}

    oim_predictive = []
    for r in acts:
        ts = r["ts"]
        if ts not in oim_by_ts or ts not in ts_idx:
            continue
        i = ts_idx[ts]
        if i + 3 >= len(ts_list):
            continue
        future_mid = mids[ts_list[i + 3]]
        current_mid = mids.get(ts)
        if current_mid is None:
            continue
        oim = oim_by_ts[ts]
        move = future_mid - current_mid
        oim_predictive.append((oim, move))

    if oim_predictive:
        # Bin OIM and check average future 3-tick move
        bins = [(-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1.0)]
        print("\n  OIM → 3-tick price move predictiveness:")
        print(f"  {'OIM Bucket':<18} {'N':>6} {'Avg Move':>10}")
        print(f"  {'-'*18} {'-'*6} {'-'*10}")
        for lo, hi in bins:
            subset = [m for o, m in oim_predictive if lo <= o < hi]
            if subset:
                avg = sum(subset) / len(subset)
                print(f"  [{lo:+.1f}, {hi:+.1f})         {len(subset):>6}  {avg:>+10.3f}")


def analyze_spread_capture_by_level(buys, sells, acts):
    """Classify fills as inner (penny jump) vs outer (2 ticks back) to see
    whether the outer quote is actually contributing."""
    section("FILL LEVEL ATTRIBUTION (Inner vs Outer Quote)")

    bid_by_ts = {r["ts"]: r["bid1"] for r in acts if r["bid1"] is not None}
    ask_by_ts = {r["ts"]: r["ask1"] for r in acts if r["ask1"] is not None}

    inner_buy = outer_buy = other_buy = 0
    inner_buy_v = outer_buy_v = other_buy_v = 0
    for t in buys:
        bb = bid_by_ts.get(t["ts"])
        ba = ask_by_ts.get(t["ts"])
        if bb is None or ba is None:
            other_buy += 1; other_buy_v += t["qty"]; continue
        inner = bb + 1      # penny jump
        outer = bb - 1      # 2 ticks behind inner (inner-2)
        diff = abs(t["price"] - inner)
        if diff <= 1:
            inner_buy += 1; inner_buy_v += t["qty"]
        elif abs(t["price"] - outer) <= 1:
            outer_buy += 1; outer_buy_v += t["qty"]
        elif t["price"] <= ba:   # mispricing take
            other_buy += 1; other_buy_v += t["qty"]
        else:
            other_buy += 1; other_buy_v += t["qty"]

    inner_sell = outer_sell = other_sell = 0
    inner_sell_v = outer_sell_v = other_sell_v = 0
    for t in sells:
        bb = bid_by_ts.get(t["ts"])
        ba = ask_by_ts.get(t["ts"])
        if bb is None or ba is None:
            other_sell += 1; other_sell_v += t["qty"]; continue
        inner = ba - 1
        outer = ba + 1
        diff = abs(t["price"] - inner)
        if diff <= 1:
            inner_sell += 1; inner_sell_v += t["qty"]
        elif abs(t["price"] - outer) <= 1:
            outer_sell += 1; outer_sell_v += t["qty"]
        else:
            other_sell += 1; other_sell_v += t["qty"]

    total_v = inner_buy_v + outer_buy_v + other_buy_v + inner_sell_v + outer_sell_v + other_sell_v
    def pct(v): return f"{100*v/max(1,total_v):.1f}%"
    print(f"  BUY fills:   inner={inner_buy_v:>4}u ({pct(inner_buy_v)})  outer={outer_buy_v:>4}u ({pct(outer_buy_v)})  other={other_buy_v:>4}u ({pct(other_buy_v)})")
    print(f"  SELL fills:  inner={inner_sell_v:>4}u ({pct(inner_sell_v)})  outer={outer_sell_v:>4}u ({pct(outer_sell_v)})  other={other_sell_v:>4}u ({pct(other_sell_v)})")

    outer_total = outer_buy_v + outer_sell_v
    outer_pct = 100 * outer_total / max(1, total_v)
    print(f"\n  Outer quote contribution: {outer_total}u = {outer_pct:.1f}% of all volume")
    if outer_pct > 10:
        print("  ✅ Outer quotes are actively filling — the two-tier structure IS earning its keep.")
    else:
        print("  ⚠ Outer quotes contribute very little — consider whether removing them loses much.")

    other_total = other_buy_v + other_sell_v
    print(f"  Mispricing takes (other): {other_total}u = {100*other_total/max(1,total_v):.1f}% of all volume")


def analyze_fill_price_vs_fair(buys, sells, ema_by_ts):
    """How far below EMA do we buy, and how far above do we sell?"""
    section("FILL PRICE vs EMA FAIR VALUE — Edge Per Fill")

    buy_edges = []
    for t in buys:
        e = ema_by_ts.get(t["ts"])
        if e: buy_edges.append(e - t["price"])  # positive = we bought below fair ✅

    sell_edges = []
    for t in sells:
        e = ema_by_ts.get(t["ts"])
        if e: sell_edges.append(t["price"] - e)  # positive = we sold above fair ✅

    if buy_edges:
        avg_be = sum(buy_edges) / len(buy_edges)
        neg = sum(1 for e in buy_edges if e < 0)
        print(f"  Avg edge on BUY  (EMA - fill): {avg_be:+.2f} ticks  ({neg}/{len(buy_edges)} fills were ABOVE EMA ❌)")
    if sell_edges:
        avg_se = sum(sell_edges) / len(sell_edges)
        neg = sum(1 for e in sell_edges if e < 0)
        print(f"  Avg edge on SELL (fill - EMA): {avg_se:+.2f} ticks  ({neg}/{len(sell_edges)} fills were BELOW EMA ❌)")

    if buy_edges and sell_edges:
        round_trip = (sum(buy_edges)/len(buy_edges)) + (sum(sell_edges)/len(sell_edges))
        print(f"\n  Round-trip edge vs EMA: {round_trip:+.2f} ticks per buy+sell pair")
        if round_trip < 5:
            print("  ⚠ Round-trip edge is thin — strategy may be quoting too close to EMA.")
        else:
            print("  ✅ Healthy round-trip edge — spread capture working relative to fair.")


def analyze_inventory_transitions(buys, sells, acts, pos_by_ts):
    """Look at what happens to fills when position is near the kill threshold."""
    section("FILLS NEAR INVENTORY LIMIT (|pos| > 60)")

    fills = sorted(
        [{"ts": t["ts"], "price": t["price"], "qty": t["qty"], "side": "buy"} for t in buys] +
        [{"ts": t["ts"], "price": t["price"], "qty": t["qty"], "side": "sell"} for t in sells],
        key=lambda x: x["ts"]
    )

    mid_by_ts = {r["ts"]: r["mid"] for r in acts if r["mid"] is not None}

    high_invent_fills = []
    for f in fills:
        pos = pos_by_ts.get(f["ts"], 0)
        if abs(pos) > 60:
            mid = mid_by_ts.get(f["ts"])
            edge = None
            if mid is not None:
                edge = (mid - f["price"]) if f["side"] == "buy" else (f["price"] - mid)
            high_invent_fills.append({**f, "pos": pos, "edge": edge})

    print(f"  Fills with |pos| > 60: {len(high_invent_fills)}")
    if not high_invent_fills:
        print("  ✅ No fills near limit — kill switch and inventory gate working cleanly.")
        return

    wrong_side = [f for f in high_invent_fills
                  if (f["side"] == "buy" and f["pos"] > 60) or
                     (f["side"] == "sell" and f["pos"] < -60)]
    print(f"  Fills that DEEPEN an already extreme position: {len(wrong_side)} ❌")

    edges = [f["edge"] for f in high_invent_fills if f["edge"] is not None]
    if edges:
        avg_e = sum(edges) / len(edges)
        print(f"  Avg edge at high inventory fills: {avg_e:+.2f}  "
              f"({'decent fill' if avg_e > 2 else '⚠ thin edge near limit'})")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    data = load(LOG_PATH)
    acts = parse_activities(data["activitiesLog"])
    buys, sells = parse_trades(data["tradeHistory"])
    timestamps = sorted(set(r["ts"] for r in acts))
    pos_by_ts = rebuild_position(buys, sells, timestamps)
    ema_by_ts = compute_ema(acts, EMA_ALPHA)

    print(f"\n{'='*64}")
    print(f"  DEEP OSMIUM ANALYSIS  |  {LOG_PATH.name}")
    print(f"  {len(buys)} buy fills, {len(sells)} sell fills across {len(acts)} ticks")
    print(f"{'='*64}")

    analyze_fill_vs_mid(buys, sells, acts)
    analyze_oim(acts, buys, sells)
    analyze_spread_capture_by_level(buys, sells, acts)
    analyze_fill_price_vs_fair(buys, sells, ema_by_ts)
    analyze_inventory_transitions(buys, sells, acts, pos_by_ts)

if __name__ == "__main__":
    main()
