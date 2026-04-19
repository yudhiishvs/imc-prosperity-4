"""
analyze_272299_volume.py — Understand why Osmium fill volume is so low.
Focus: where are our quotes landing vs where fills happen?
"""

import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "272299" / "272299.log"

def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_activities(raw_csv, product):
    rows = raw_csv.strip().split("\n")
    header = rows[0].split(";")
    records = []
    for row in rows[1:]:
        parts = row.split(";")
        if len(parts) < len(header):
            continue
        r = dict(zip(header, parts))
        if r.get("product", "").strip() != product:
            continue
        def fv(k):
            v = r.get(k, "").strip()
            return float(v) if v else None
        records.append({
            "day": int(r.get("day", 0)),
            "ts": int(r.get("timestamp", 0)),
            "bid1": fv("bid_price_1"), "bid1v": fv("bid_volume_1"),
            "bid2": fv("bid_price_2"), "bid2v": fv("bid_volume_2"),
            "ask1": fv("ask_price_1"), "ask1v": fv("ask_volume_1"),
            "ask2": fv("ask_price_2"), "ask2v": fv("ask_volume_2"),
            "mid": fv("mid_price"),
            "pnl": fv("profit_and_loss"),
        })
    return records

def parse_all_trades(trade_history, product):
    """Parse ALL trades (not just ours) to understand total market volume."""
    our_buys, our_sells, other_trades = [], [], []
    for t in trade_history:
        if t.get("symbol") != product:
            continue
        rec = {"ts": t["timestamp"], "price": float(t["price"]), "qty": int(t["quantity"]),
               "buyer": t.get("buyer", ""), "seller": t.get("seller", "")}
        if t.get("buyer") == "SUBMISSION":
            our_buys.append(rec)
        elif t.get("seller") == "SUBMISSION":
            our_sells.append(rec)
        else:
            other_trades.append(rec)
    return our_buys, our_sells, other_trades

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def pct(n, d):
    return f"{100*n/max(1,d):.1f}%"

def main():
    data = load_log(LOG_PATH)
    osm_acts = parse_activities(data["activitiesLog"], "ASH_COATED_OSMIUM")
    our_buys, our_sells, other_trades = parse_all_trades(data["tradeHistory"], "ASH_COATED_OSMIUM")

    total_ts = len(set(r["ts"] for r in osm_acts))

    # ── 1. Total market volume vs our volume ──
    section("1. MARKET VOLUME vs OUR VOLUME")
    total_market_vol = sum(t["qty"] for t in our_buys) + sum(t["qty"] for t in our_sells) + sum(t["qty"] for t in other_trades)
    our_vol = sum(t["qty"] for t in our_buys) + sum(t["qty"] for t in our_sells)
    other_vol = sum(t["qty"] for t in other_trades)

    print(f"  Total market volume:  {total_market_vol:>8,} units")
    print(f"  Our volume:           {our_vol:>8,} units ({pct(our_vol, total_market_vol)})")
    print(f"  Other volume:         {other_vol:>8,} units ({pct(other_vol, total_market_vol)})")
    print(f"  Our trades:           {len(our_buys) + len(our_sells)}")
    print(f"  Other trades:         {len(other_trades)}")
    print(f"  Fill rate:            {(len(our_buys) + len(our_sells)) / total_ts:.3f} fills/tick")

    # ── 2. Where do ALL trades happen? ──
    section("2. TRADE PRICE DISTRIBUTION (all market participants)")
    all_trades = our_buys + our_sells + other_trades
    price_dist = defaultdict(int)
    for t in all_trades:
        price_dist[int(t["price"])] += t["qty"]

    # Show distribution around 10k
    print(f"  {'Price':>8} {'Volume':>8} {'Bar':>40}")
    total = sum(price_dist.values())
    for p in sorted(price_dist):
        if 9985 <= p <= 10015:
            bar = "█" * min(40, price_dist[p] * 40 // max(1, max(price_dist.values())))
            print(f"  {p:>8} {price_dist[p]:>8} {bar}")

    # ── 3. Where do OUR fills happen vs fair value? ──
    section("3. OUR FILL PRICES vs 10k")
    our_price_dist = defaultdict(lambda: {"buy": 0, "sell": 0})
    for t in our_buys:
        our_price_dist[int(t["price"])]["buy"] += t["qty"]
    for t in our_sells:
        our_price_dist[int(t["price"])]["sell"] += t["qty"]

    print(f"  {'Price':>8} {'Buy Vol':>8} {'Sell Vol':>8} {'Verdict':>10}")
    for p in sorted(our_price_dist):
        buy = our_price_dist[p]["buy"]
        sell = our_price_dist[p]["sell"]
        if p < 10000:
            verdict = "✅ good buy" if buy > 0 else ("❌ bad sell" if sell > 0 else "")
        elif p > 10000:
            verdict = "✅ good sell" if sell > 0 else ("❌ bad buy" if buy > 0 else "")
        else:
            verdict = "neutral"
        print(f"  {p:>8} {buy:>8} {sell:>8} {verdict}")

    # ── 4. Where does our quote land vs where the best bid/ask is? ──
    section("4. QUOTE PLACEMENT ANALYSIS — Where we SHOULD be quoting")
    two_sided = [r for r in osm_acts if r["bid1"] is not None and r["ask1"] is not None]

    # With penny_jump=1: our bid = min(bb+1, 10000), our ask = max(ba-1, 10000)
    our_bid_prices = []
    our_ask_prices = []
    spreads_quoting = []
    for r in two_sided:
        bb, ba = r["bid1"], r["ask1"]
        spread = ba - bb
        # Current strategy.py penny-jump logic:
        our_bid = min(int(bb) + 1, 10000)
        our_ask = max(int(ba) - 1, 10000)
        our_spread = our_ask - our_bid
        our_bid_prices.append(our_bid)
        our_ask_prices.append(our_ask)
        spreads_quoting.append(our_spread)

    avg_our_spread = sum(spreads_quoting) / len(spreads_quoting) if spreads_quoting else 0
    at_10k_bid = sum(1 for b in our_bid_prices if b == 10000)
    at_10k_ask = sum(1 for a in our_ask_prices if a == 10000)
    below_10k_bid = sum(1 for b in our_bid_prices if b < 10000)
    above_10k_ask = sum(1 for a in our_ask_prices if a > 10000)

    print(f"  Two-sided ticks: {len(two_sided)}")
    print(f"  Avg spread we quote: {avg_our_spread:.1f} ticks")
    print(f"")
    print(f"  Our BID placement:")
    print(f"    At 10,000:      {at_10k_bid} ({pct(at_10k_bid, len(two_sided))}) ← capped at fair")
    print(f"    Below 10,000:   {below_10k_bid} ({pct(below_10k_bid, len(two_sided))})")
    print(f"  Our ASK placement:")
    print(f"    At 10,000:      {at_10k_ask} ({pct(at_10k_ask, len(two_sided))}) ← capped at fair")
    print(f"    Above 10,000:   {above_10k_ask} ({pct(above_10k_ask, len(two_sided))})")

    # ── 5. How often is mid above/below 10k? ──
    section("5. MID PRICE vs 10k — Direction Bias")
    above = sum(1 for r in two_sided if r["mid"] > 10000)
    below = sum(1 for r in two_sided if r["mid"] < 10000)
    at = sum(1 for r in two_sided if r["mid"] == 10000)
    print(f"  Mid > 10k: {above} ({pct(above, len(two_sided))})")
    print(f"  Mid = 10k: {at} ({pct(at, len(two_sided))})")
    print(f"  Mid < 10k: {below} ({pct(below, len(two_sided))})")

    # Distribution of mid
    mid_buckets = defaultdict(int)
    for r in two_sided:
        mid = r["mid"]
        bucket = int(round(mid - 10000))
        mid_buckets[bucket] += 1
    print(f"\n  Mid price offset from 10k:")
    for b in sorted(mid_buckets):
        if abs(b) <= 15:
            bar = "█" * min(40, mid_buckets[b] * 40 // max(1, max(mid_buckets.values())))
            print(f"    {b:+4d}: {bar:40s} {mid_buckets[b]:>5}")

    # ── 6. Theoretical max PnL from perfect spread capture ──
    section("6. THEORETICAL PnL — Perfect Spread Capture")
    # If we caught every tick at penny-jump prices with max volume
    all_spreads = [r["ask1"] - r["bid1"] for r in two_sided]
    avg_spread = sum(all_spreads) / len(all_spreads)
    print(f"  Avg market spread: {avg_spread:.1f} ticks")
    print(f"  If we penny-jump (capture spread - 2): {avg_spread - 2:.1f} ticks per round trip")
    print(f"")
    for fill_rate in [0.05, 0.1, 0.2, 0.3, 0.5]:
        rt_per_tick = fill_rate
        volume = rt_per_tick * len(two_sided)
        pnl = volume * (avg_spread - 2)
        print(f"  At {fill_rate:.0%} fill rate: {volume:.0f} round trips → {pnl:,.0f} PnL")

    # ── 7. What volume of existing orders below/above 10k can we take? ──
    section("7. TAKEABLE ORDER FLOW — Free Money Below/Above 10k")
    below_10k_volume = 0
    above_10k_volume = 0
    ticks_with_asks_below = 0
    ticks_with_bids_above = 0

    for r in osm_acts:
        # Asks below 10k = free buy opportunities
        if r["ask1"] is not None and r["ask1"] < 10000:
            vol = r["ask1v"] or 0
            below_10k_volume += vol
            ticks_with_asks_below += 1
            if r["ask2"] is not None and r["ask2"] < 10000:
                below_10k_volume += (r["ask2v"] or 0)

        # Bids above 10k = free sell opportunities
        if r["bid1"] is not None and r["bid1"] > 10000:
            vol = r["bid1v"] or 0
            above_10k_volume += vol
            ticks_with_bids_above += 1
            if r["bid2"] is not None and r["bid2"] > 10000:
                above_10k_volume += (r["bid2v"] or 0)

    print(f"  Ticks with asks below 10k: {ticks_with_asks_below} ({pct(ticks_with_asks_below, total_ts)})")
    print(f"  Total takeable ask volume below 10k: {below_10k_volume:.0f} units")
    print(f"  Ticks with bids above 10k: {ticks_with_bids_above} ({pct(ticks_with_bids_above, total_ts)})")
    print(f"  Total takeable bid volume above 10k: {above_10k_volume:.0f} units")
    print(f"  Total 'free money' volume: {below_10k_volume + above_10k_volume:.0f} units")

    # What PnL does that represent?
    take_buy_pnl = 0
    for r in osm_acts:
        if r["ask1"] is not None and r["ask1"] < 10000:
            take_buy_pnl += (10000 - r["ask1"]) * (r["ask1v"] or 0)
            if r["ask2"] is not None and r["ask2"] < 10000:
                take_buy_pnl += (10000 - r["ask2"]) * (r["ask2v"] or 0)

    take_sell_pnl = 0
    for r in osm_acts:
        if r["bid1"] is not None and r["bid1"] > 10000:
            take_sell_pnl += (r["bid1"] - 10000) * (r["bid1v"] or 0)
            if r["bid2"] is not None and r["bid2"] > 10000:
                take_sell_pnl += (r["bid2"] - 10000) * (r["bid2v"] or 0)

    print(f"\n  PnL from taking all asks below 10k: {take_buy_pnl:,.0f}")
    print(f"  PnL from taking all bids above 10k: {take_sell_pnl:,.0f}")
    print(f"  Total from taking alone:            {take_buy_pnl + take_sell_pnl:,.0f}")

    # ── 8. What's the observed fill rate per tick by region? ──
    section("8. FILL TIMING — How clustered are our fills?")
    fill_ts = defaultdict(int)
    for t in our_buys + our_sells:
        fill_ts[t["ts"]] += t["qty"]

    ticks_with_fills = len(fill_ts)
    print(f"  Ticks with at least 1 fill: {ticks_with_fills} ({pct(ticks_with_fills, total_ts)})")
    if fill_ts:
        avg_fill_per_active = sum(fill_ts.values()) / ticks_with_fills
        max_fill = max(fill_ts.values())
        print(f"  Avg volume per active tick: {avg_fill_per_active:.1f}")
        print(f"  Max volume in single tick:  {max_fill}")

    # ── 9. Other bots' trade volume distribution ──
    section("9. OTHER BOTS' ACTIVITY")
    other_price_dist = defaultdict(int)
    for t in other_trades:
        other_price_dist[int(t["price"])] += t["qty"]

    print(f"  Other-bot trades: {len(other_trades)} ({sum(t['qty'] for t in other_trades)} units)")
    print(f"\n  Price distribution (around 10k):")
    for p in sorted(other_price_dist):
        if 9985 <= p <= 10015:
            vol = other_price_dist[p]
            bar = "█" * min(40, vol * 40 // max(1, max(other_price_dist.values())))
            print(f"    {p:>6}: {bar:40s} {vol:>6}")


if __name__ == "__main__":
    main()
