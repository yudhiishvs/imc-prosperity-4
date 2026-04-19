"""
analyze_272299.py — Deep post-mortem of the 272299 live submission (iteration 2).

Strategy: EMA(0.1122) + A-S reservation price + momentum fade + 2-tier quoting.
Goal: Understand why Osmium assumptions broke OOS while Pepper Root assumptions held.

Usage:
    python3 vedant/analyze_272299.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "272299" / "272299.log"

OSMIUM_EMA_ALPHA = 0.1122
OSMIUM_INVENTORY_SKEW = 0.0408
OSMIUM_MOMENTUM_QUOTE_SHIFT = 6


# ─── PARSING ──────────────────────────────────────────────────────────────────

def load_log(path: Path):
    print(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_activities(raw_csv: str, product: str):
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
            "bid3": fv("bid_price_3"), "bid3v": fv("bid_volume_3"),
            "ask1": fv("ask_price_1"), "ask1v": fv("ask_volume_1"),
            "ask2": fv("ask_price_2"), "ask2v": fv("ask_volume_2"),
            "ask3": fv("ask_price_3"), "ask3v": fv("ask_volume_3"),
            "mid": fv("mid_price"),
            "pnl": fv("profit_and_loss"),
        })
    return records


def parse_trades(trade_history: list, product: str):
    buys, sells = [], []
    for t in trade_history:
        if t.get("symbol") != product:
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
    pos = 0
    idx = 0
    pos_by_ts = {}
    for ts in timestamps:
        while idx < len(events) and events[idx][0] <= ts:
            pos += events[idx][1]
            idx += 1
        pos_by_ts[ts] = pos
    return pos_by_ts


def compute_ema(acts, alpha):
    ema_by_ts = {}
    ema = None
    for r in acts:
        mid = r["mid"]
        # Skip degenerate mids (only-bid or only-ask producing unreliable mid)
        if mid is None or (r["bid1"] is None and r["ask1"] is not None) or (r["ask1"] is None and r["bid1"] is not None):
            ema_by_ts[r["ts"]] = ema
            continue
        ema = mid if ema is None else alpha * mid + (1 - alpha) * ema
        ema_by_ts[r["ts"]] = ema
    return ema_by_ts


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def pct(num, den):
    return f"{100*num/max(1,den):.1f}%"


# ─── ANALYSES ─────────────────────────────────────────────────────────────────

def analyze_pnl_trajectory(osm_acts, pep_acts):
    section("1. PnL TRAJECTORY — Osmium vs Pepper Root")

    for name, acts in [("OSMIUM", osm_acts), ("PEPPER", pep_acts)]:
        pnls = [(r["ts"], r["pnl"]) for r in acts if r["pnl"] is not None]
        if not pnls:
            print(f"  {name}: No PnL data.")
            continue

        n = len(pnls)
        qs = [pnls[:n//4], pnls[n//4:n//2], pnls[n//2:3*n//4], pnls[3*n//4:]]
        labels = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]

        def delta(q):
            return q[-1][1] - q[0][1] if len(q) >= 2 else 0

        deltas = [delta(q) for q in qs]
        total = pnls[-1][1] - pnls[0][1]

        print(f"\n  {name}:")
        print(f"  {'Quarter':<15} {'PnL Change':>12}")
        print(f"  {'-'*15} {'-'*12}")
        for label, d in zip(labels, deltas):
            icon = "✅" if d > 0 else "❌"
            print(f"  {label:<15} {d:>+12,.0f}  {icon}")
        print(f"  {'TOTAL':<15} {total:>+12,.0f}")
        print(f"  Final PnL:     {pnls[-1][1]:>12,.0f}")


def analyze_book_quality(osm_acts, pep_acts):
    """Understand how often we have bad book data (no bid, no ask, one-sided)."""
    section("2. ORDER BOOK QUALITY — One-Sided Books & Degenerate Mids")

    for name, acts in [("OSMIUM", osm_acts), ("PEPPER", pep_acts)]:
        total = len(acts)
        both_sides = sum(1 for r in acts if r["bid1"] is not None and r["ask1"] is not None)
        bid_only = sum(1 for r in acts if r["bid1"] is not None and r["ask1"] is None)
        ask_only = sum(1 for r in acts if r["bid1"] is None and r["ask1"] is not None)
        neither = sum(1 for r in acts if r["bid1"] is None and r["ask1"] is None)

        print(f"\n  {name} ({total} ticks):")
        print(f"    Both sides:   {both_sides:>6} ({pct(both_sides, total)})")
        print(f"    Bid only:     {bid_only:>6} ({pct(bid_only, total)}) ← mid = bid, NO asks visible")
        print(f"    Ask only:     {ask_only:>6} ({pct(ask_only, total)}) ← mid = ask, NO bids visible")
        print(f"    Neither:      {neither:>6} ({pct(neither, total)})")

        if bid_only + ask_only > total * 0.1:
            print(f"    ⚠ {pct(bid_only + ask_only, total)} of ticks have one-sided books!")
            print(f"      Mid prices on these ticks are unreliable.")
            print(f"      EMA tracking mid on one-sided ticks will DRIFT from true fair value!")


def analyze_spread_regime(osm_acts, pep_acts):
    section("3. MARKET MICROSTRUCTURE — Spread & Volatility (two-sided only)")

    for name, acts in [("OSMIUM", osm_acts), ("PEPPER", pep_acts)]:
        # Only use ticks where BOTH bid and ask exist
        valid = [r for r in acts if r["bid1"] is not None and r["ask1"] is not None]
        spreads = [r["ask1"] - r["bid1"] for r in valid]
        mids = [r["mid"] for r in valid]

        if not spreads or not mids:
            print(f"  {name}: Not enough two-sided data")
            continue

        # Filter out absurd moves for avg calc (consecutive two-sided mids)
        moves = [abs(mids[i] - mids[i-1]) for i in range(1, len(mids)) if mids[i-1] > 1000]
        avg_move = sum(moves) / len(moves) if moves else 0

        # Mid price range (excluding degenerate)
        valid_mids = [m for m in mids if m > 1000]
        print(f"\n  {name} ({len(valid)} two-sided ticks):")
        print(f"    Mid range:        [{min(valid_mids):.0f}, {max(valid_mids):.0f}]")
        print(f"    Avg spread:       {sum(spreads)/len(spreads):.2f} ticks")
        print(f"    Spread range:     [{min(spreads):.0f}, {max(spreads):.0f}]")
        print(f"    Avg |tick move|:  {avg_move:.3f} ticks  (two-sided consecutive only)")

        # Spread distribution
        sp_buckets = defaultdict(int)
        for s in spreads:
            sp_buckets[int(s)] += 1
        print(f"    Spread distribution:")
        for b in sorted(sp_buckets):
            bar = "█" * min(40, sp_buckets[b] * 40 // len(spreads))
            print(f"      sp={b:>3}: {bar:40s} {sp_buckets[b]:>5} ({pct(sp_buckets[b], len(spreads))})")


def analyze_ema_vs_static(osm_acts):
    section("4. EMA(0.1122) vs STATIC 10,000 — Fair Value Quality")

    # Only use ticks with BOTH bid and ask for reliable mid
    valid = [r for r in osm_acts if r["bid1"] is not None and r["ask1"] is not None and r["mid"] is not None]
    if len(valid) < 10:
        print("  Not enough two-sided data.")
        return

    ema = valid[0]["mid"]
    ema_lags = []
    static_lags = []
    ema_vals = [ema]

    for r in valid[1:]:
        mid = r["mid"]
        ema = OSMIUM_EMA_ALPHA * mid + (1 - OSMIUM_EMA_ALPHA) * ema
        ema_lags.append(abs(ema - mid))
        static_lags.append(abs(10000 - mid))
        ema_vals.append(ema)

    avg_ema = sum(ema_lags) / len(ema_lags)
    avg_static = sum(static_lags) / len(static_lags)
    p90_ema = sorted(ema_lags)[int(0.9 * len(ema_lags))]
    p90_static = sorted(static_lags)[int(0.9 * len(static_lags))]

    print(f"  Using {len(valid)} two-sided-book ticks only.")
    print(f"  {'Metric':<25} {'EMA(0.1122)':>12} {'Static 10k':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Avg |fair - mid|':<25} {avg_ema:>12.2f} {avg_static:>12.2f}")
    print(f"  {'P90 |fair - mid|':<25} {p90_ema:>12.2f} {p90_static:>12.2f}")
    print(f"  {'Max |fair - mid|':<25} {max(ema_lags):>12.2f} {max(static_lags):>12.2f}")

    ema_closer = sum(1 for e, s in zip(ema_lags, static_lags) if e < s)
    static_closer = sum(1 for e, s in zip(ema_lags, static_lags) if s < e)
    print(f"\n  EMA closer to mid:    {ema_closer:>6} ({pct(ema_closer, len(ema_lags))})")
    print(f"  Static closer to mid: {static_closer:>6} ({pct(static_closer, len(ema_lags))})")

    # BUT: the real question is whether EMA drifts from the TRUE fair (10k)
    ema_drift = [abs(e - 10000) for e in ema_vals]
    avg_drift = sum(ema_drift) / len(ema_drift)
    max_drift = max(ema_drift)
    print(f"\n  EMA drift from true fair (10,000):")
    print(f"    Avg |EMA - 10k|:  {avg_drift:.2f}")
    print(f"    Max |EMA - 10k|:  {max_drift:.2f}")

    # Distribution of EMA vs 10k
    drift_buckets = defaultdict(int)
    for d in ema_drift:
        drift_buckets[int(d) // 2 * 2] += 1  # 2-tick buckets
    print(f"\n    EMA drift distribution (2-tick buckets):")
    for b in sorted(drift_buckets)[:15]:
        bar = "█" * min(40, drift_buckets[b] * 40 // len(ema_drift))
        print(f"      [{b:>3}..{b+1:>3}]: {bar:40s} {drift_buckets[b]:>5} ({pct(drift_buckets[b], len(ema_drift))})")

    # ONE-SIDED BOOK CONTAMINATION: how often does a one-sided tick inject noise into EMA?
    one_sided_count = sum(1 for r in osm_acts if (r["bid1"] is None) != (r["ask1"] is None))
    total = len(osm_acts)
    print(f"\n  One-sided book ticks (EMA contamination risk):")
    print(f"    {one_sided_count} / {total} = {pct(one_sided_count, total)}")
    if one_sided_count > total * 0.05:
        print(f"    ❌ SIGNIFICANT: The EMA is consuming degenerate mid prices on {pct(one_sided_count, total)}")
        print(f"       of ticks. This is the primary mechanism for EMA drift from 10k.")


def analyze_inventory(acts, pos_by_ts, product_name):
    section(f"5. INVENTORY — {product_name}")

    all_pos = [pos_by_ts.get(r["ts"], 0) for r in acts]
    if not all_pos:
        print("  No position data.")
        return

    print(f"  Total ticks: {len(all_pos):,}")
    print(f"  Position range: [{min(all_pos)}, {max(all_pos)}]")
    print(f"  Avg position: {sum(all_pos)/len(all_pos):.1f}")

    bins = [(-80,-60), (-60,-40), (-40,-20), (-20,0), (0,20), (20,40), (40,60), (60,80)]
    print(f"\n  Bucket distribution:")
    for lo, hi in bins:
        c = sum(1 for p in all_pos if lo <= p < hi)
        bar = "█" * min(40, c * 40 // len(all_pos))
        print(f"    [{lo:+4d},{hi:+4d}): {bar:40s} {c:>5} ({pct(c, len(all_pos))})")

    # For Pepper: time at 80
    if product_name == "PEPPER":
        at_80 = sum(1 for p in all_pos if p >= 80)
        print(f"\n  Time at position 80: {at_80} ({pct(at_80, len(all_pos))})")


def analyze_fill_quality(buys, sells, acts, product_name):
    section(f"6. FILL QUALITY — {product_name}")

    total_buy_vol = sum(t["qty"] for t in buys)
    total_sell_vol = sum(t["qty"] for t in sells)
    if total_buy_vol == 0 and total_sell_vol == 0:
        print("  No fills.")
        return

    avg_buy = sum(t["price"]*t["qty"] for t in buys) / max(1, total_buy_vol)
    avg_sell = sum(t["price"]*t["qty"] for t in sells) / max(1, total_sell_vol)
    spread_capture = avg_sell - avg_buy

    print(f"  Total BUY fills:   {len(buys):>5} ({total_buy_vol:,} units)")
    print(f"  Total SELL fills:  {len(sells):>5} ({total_sell_vol:,} units)")
    print(f"  Net flow (buy-sell): {total_buy_vol - total_sell_vol:+,} units")
    print(f"  Avg buy price:   {avg_buy:.2f}")
    print(f"  Avg sell price:  {avg_sell:.2f}")
    print(f"  Spread capture:  {spread_capture:+.2f}")

    # For osmium: compare fills vs 10,000
    if "OSMIUM" in product_name:
        true_fair = 10000
        buy_edge = sum((true_fair - t["price"]) * t["qty"] for t in buys)
        sell_edge = sum((t["price"] - true_fair) * t["qty"] for t in sells)
        print(f"\n  Total edge vs 10k (buys):  {buy_edge:+,.0f}")
        print(f"  Total edge vs 10k (sells): {sell_edge:+,.0f}")
        print(f"  Net edge vs 10k:           {buy_edge + sell_edge:+,.0f}")


def analyze_osmium_ema_contamination(osm_acts):
    """The core mechanism: EMA tracking degenerate mids pulls fair value away from 10k,
    causing all downstream taking and quoting to use a stale/wrong fair value."""
    section("7. OSMIUM — THE EMA CONTAMINATION MECHANISM")

    # Rebuild what the strategy's EMA actually saw (including degenerate mids)
    ema = None
    ema_trace = []

    for r in osm_acts:
        mid = r["mid"]
        ts = r["ts"]
        has_bid = r["bid1"] is not None
        has_ask = r["ask1"] is not None

        if mid is not None:
            if ema is None:
                ema = mid
            else:
                ema = OSMIUM_EMA_ALPHA * mid + (1 - OSMIUM_EMA_ALPHA) * ema

        ema_trace.append({
            "ts": ts,
            "mid": mid,
            "ema": ema,
            "has_bid": has_bid,
            "has_ask": has_ask,
            "two_sided": has_bid and has_ask,
            "ema_vs_10k": abs(ema - 10000) if ema else None,
        })

    # When the EMA is far from 10k, is it because of one-sided book contamination?
    big_drift_events = [t for t in ema_trace if t["ema_vs_10k"] is not None and t["ema_vs_10k"] > 5]
    big_drift_after_one_sided = [t for t in big_drift_events if not t["two_sided"]]

    print(f"  Ticks where |EMA - 10k| > 5:  {len(big_drift_events)}")
    print(f"    ...of which had one-sided book: {len(big_drift_after_one_sided)} ({pct(len(big_drift_after_one_sided), len(big_drift_events))})")

    # Sequence analysis: does a cluster of one-sided ticks precede EMA drift?
    print(f"\n  EXAMPLE: First 5 large EMA drift events:")
    count = 0
    for t in ema_trace:
        if t["ema_vs_10k"] is not None and t["ema_vs_10k"] > 5:
            book_type = "TWO-SIDED" if t["two_sided"] else ("BID-ONLY" if t["has_bid"] else "ASK-ONLY")
            print(f"    t={t['ts']:>7}  mid={t['mid']:>10.1f}  EMA={t['ema']:>10.2f}  |drift|={t['ema_vs_10k']:.2f}  book={book_type}")
            count += 1
            if count >= 5:
                break

    # Impact: how much does the EMA-induced mispricing cost us?
    # When EMA > 10005 and we buy (taking asks up to EMA), we overpay
    # When EMA < 9995 and we sell (hitting bids down to EMA), we undersell
    print(f"\n  EMA drift regime analysis:")
    normal = sum(1 for t in ema_trace if t["ema_vs_10k"] is not None and t["ema_vs_10k"] <= 2)
    mild = sum(1 for t in ema_trace if t["ema_vs_10k"] is not None and 2 < t["ema_vs_10k"] <= 5)
    moderate = sum(1 for t in ema_trace if t["ema_vs_10k"] is not None and 5 < t["ema_vs_10k"] <= 10)
    severe = sum(1 for t in ema_trace if t["ema_vs_10k"] is not None and t["ema_vs_10k"] > 10)
    total = normal + mild + moderate + severe
    print(f"    |drift| ≤ 2:   {normal:>5} ({pct(normal, total)})  ← EMA basically correct")
    print(f"    |drift| 2-5:   {mild:>5} ({pct(mild, total)})  ← mild skew")
    print(f"    |drift| 5-10:  {moderate:>5} ({pct(moderate, total)})  ← taking thresholds shifted")
    print(f"    |drift| > 10:  {severe:>5} ({pct(severe, total)})  ← actively harmful")


def analyze_osmium_fill_breakdown(osm_buys, osm_sells, osm_acts, osm_pos):
    """Classify each Osmium fill by what triggered it and whether it was profitable vs 10k."""
    section("8. OSMIUM — Fill vs 10,000 Analysis")

    # Rebuild EMA for each timestamp
    ema_by_ts = compute_ema(osm_acts, OSMIUM_EMA_ALPHA)

    total_buys = len(osm_buys)
    total_sells = len(osm_sells)

    # Buys
    good_buys = [t for t in osm_buys if t["price"] < 10000]
    neutral_buys = [t for t in osm_buys if t["price"] == 10000]
    bad_buys = [t for t in osm_buys if t["price"] > 10000]

    good_buy_vol = sum(t["qty"] for t in good_buys)
    neutral_buy_vol = sum(t["qty"] for t in neutral_buys)
    bad_buy_vol = sum(t["qty"] for t in bad_buys)

    print(f"  BUYS ({total_buys} fills, {sum(t['qty'] for t in osm_buys)} units):")
    print(f"    Below 10k ✅:  {len(good_buys):>4} fills, {good_buy_vol:>5} vol")
    print(f"    At 10k:        {len(neutral_buys):>4} fills, {neutral_buy_vol:>5} vol")
    print(f"    Above 10k ❌:  {len(bad_buys):>4} fills, {bad_buy_vol:>5} vol")

    if bad_buys:
        worst_buy = max(bad_buys, key=lambda t: t["price"])
        avg_bad_premium = sum(t["price"] - 10000 for t in bad_buys) / len(bad_buys)
        total_bad_cost = sum((t["price"] - 10000) * t["qty"] for t in bad_buys)
        print(f"    Avg overpay on bad buys: {avg_bad_premium:+.1f} ticks")
        print(f"    Worst buy: {worst_buy['price']:.0f} at t={worst_buy['ts']}")
        print(f"    Total cost of buys above 10k: {total_bad_cost:.0f} XIRECs")

    # Sells
    good_sells = [t for t in osm_sells if t["price"] > 10000]
    neutral_sells = [t for t in osm_sells if t["price"] == 10000]
    bad_sells = [t for t in osm_sells if t["price"] < 10000]

    good_sell_vol = sum(t["qty"] for t in good_sells)
    neutral_sell_vol = sum(t["qty"] for t in neutral_sells)
    bad_sell_vol = sum(t["qty"] for t in bad_sells)

    print(f"\n  SELLS ({total_sells} fills, {sum(t['qty'] for t in osm_sells)} units):")
    print(f"    Above 10k ✅:  {len(good_sells):>4} fills, {good_sell_vol:>5} vol")
    print(f"    At 10k:        {len(neutral_sells):>4} fills, {neutral_sell_vol:>5} vol")
    print(f"    Below 10k ❌:  {len(bad_sells):>4} fills, {bad_sell_vol:>5} vol")

    if bad_sells:
        worst_sell = min(bad_sells, key=lambda t: t["price"])
        avg_bad_discount = sum(10000 - t["price"] for t in bad_sells) / len(bad_sells)
        total_bad_cost = sum((10000 - t["price"]) * t["qty"] for t in bad_sells)
        print(f"    Avg undersell on bad sells: {avg_bad_discount:+.1f} ticks")
        print(f"    Worst sell: {worst_sell['price']:.0f} at t={worst_sell['ts']}")
        print(f"    Total cost of sells below 10k: {total_bad_cost:.0f} XIRECs")

    # Net alpha from fills
    buy_pnl = sum((10000 - t["price"]) * t["qty"] for t in osm_buys)
    sell_pnl = sum((t["price"] - 10000) * t["qty"] for t in osm_sells)
    print(f"\n  Net edge vs 10k from all fills:")
    print(f"    Buys:  {buy_pnl:+,.0f} XIRECs")
    print(f"    Sells: {sell_pnl:+,.0f} XIRECs")
    print(f"    Total: {buy_pnl + sell_pnl:+,.0f} XIRECs")

    # WHY are we buying above 10k? Check what EMA was at those moments
    if bad_buys:
        print(f"\n  WHY did we buy above 10k? EMA at those fills:")
        for t in bad_buys[:10]:
            e = ema_by_ts.get(t["ts"])
            pos = osm_pos.get(t["ts"], 0)
            ema_str = f"{e:.1f}" if e else "None"
            print(f"    t={t['ts']:>7} price={t['price']:.0f} EMA={ema_str} pos={pos:+d}")


def analyze_momentum_fade(osm_acts):
    """Analyze the momentum fade signal using only clean two-sided mids."""
    section("9. OSMIUM — Momentum Fade Signal (Two-Sided Mids Only)")

    # Use only ticks with both bid and ask for reliable mid
    clean = [(r["ts"], r["mid"]) for r in osm_acts 
             if r["bid1"] is not None and r["ask1"] is not None and r["mid"] is not None]

    if len(clean) < 20:
        print("  Not enough clean data.")
        return

    correct = wrong = 0
    fade_returns = {1: [], 3: [], 5: [], 10: []}

    for i in range(1, len(clean)):
        _, mid = clean[i]
        _, prev_mid = clean[i-1]
        change = mid - prev_mid
        if change == 0:
            continue

        fade_dir = -1 if change > 0 else 1

        for h in fade_returns:
            if i + h < len(clean):
                future = clean[i+h][1]
                fade_returns[h].append((future - mid) * fade_dir)

        if i + 1 < len(clean):
            next_move = clean[i+1][1] - mid
            if next_move * fade_dir > 0:
                correct += 1
            elif next_move * fade_dir < 0:
                wrong += 1

    total = correct + wrong
    print(f"  Clean two-sided mid ticks: {len(clean)}")
    print(f"  Non-zero change signals:   {total}")
    print(f"  Fade correct (1-tick):     {correct} ({pct(correct, total)})")
    print(f"  Fade wrong (1-tick):       {wrong} ({pct(wrong, total)})")

    for h, results in sorted(fade_returns.items()):
        if results:
            avg = sum(results) / len(results)
            pos_frac = sum(1 for r in results if r > 0) / len(results)
            print(f"\n  Fade {h}-tick: avg={avg:+.3f}  correct={100*pos_frac:.1f}%  n={len(results)}")
            if avg < 0:
                print(f"    ❌ Momentum PERSISTS at {h}-tick horizon. Fading hurts.")
            elif avg > 0.5:
                print(f"    ✅ Strong mean reversion at {h}-tick. Fading adds alpha.")
            else:
                print(f"    ≈ Marginal mean reversion at {h}-tick.")

    # The quote_shift of ±6 ticks is very aggressive. Is it justified?
    big_moves = [abs(clean[i][1] - clean[i-1][1]) for i in range(1, len(clean))]
    avg_move = sum(big_moves) / len(big_moves) if big_moves else 0
    print(f"\n  Avg |tick-to-tick move|:     {avg_move:.2f} ticks")
    print(f"  Config MOMENTUM_QUOTE_SHIFT: {OSMIUM_MOMENTUM_QUOTE_SHIFT} ticks")
    if OSMIUM_MOMENTUM_QUOTE_SHIFT > avg_move * 3:
        print(f"  ⚠ QUOTE SHIFT is {OSMIUM_MOMENTUM_QUOTE_SHIFT/avg_move:.1f}x the avg move — WAY too aggressive!")
        print(f"    This shifts quotes 6 ticks on every non-zero move, but most moves are <{avg_move:.0f} ticks.")


def analyze_pepper_trend_confirmation(pep_acts, pep_buys, pep_sells, pep_pos):
    section("10. PEPPER ROOT — Trend Assumption Validation")

    # Use only two-sided mids
    valid = [(r["ts"], r["mid"]) for r in pep_acts 
             if r["bid1"] is not None and r["ask1"] is not None and r["mid"] is not None]
    if len(valid) < 100:
        print("  Not enough data.")
        return

    first_ts, first_mid = valid[0]
    last_ts, last_mid = valid[-1]
    actual_slope = (last_mid - first_mid) / max(1, last_ts - first_ts)

    print(f"  First mid: {first_mid:.1f} at t={first_ts}")
    print(f"  Last mid:  {last_mid:.1f} at t={last_ts}")
    print(f"  Expected slope: 0.001000")
    print(f"  Actual slope:   {actual_slope:.6f}")
    print(f"  Slope ratio:    {actual_slope/0.001:.3f}x")

    # Accumulation speed
    positions = [(r["ts"], pep_pos.get(r["ts"], 0)) for r in pep_acts]
    first_80 = next((ts for ts, p in positions if p >= 80), None)
    if first_80:
        print(f"\n  Reached 80 at t={first_80} ({first_80/100:.0f} ticks in)")
    else:
        print(f"\n  ⚠ Never reached 80.")

    at_80 = sum(1 for _, p in positions if p >= 80)
    print(f"  Time at 80: {at_80} ({pct(at_80, len(positions))})")
    print(f"  Trend carry @ 80 × {len(positions)} ticks = {80 * 0.001 * len(positions):.0f} XIRECs theoretical max")

    # Final PnL
    pep_pnls = [(r["ts"], r["pnl"]) for r in pep_acts if r["pnl"] is not None]
    if pep_pnls:
        print(f"  Actual Pepper PnL: {pep_pnls[-1][1]:,.0f} XIRECs")
        theoretical = 80 * 0.001 * len(positions)
        print(f"  Capture efficiency: {pep_pnls[-1][1] / theoretical * 100:.1f}% of theoretical max")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    data = load_log(LOG_PATH)

    print(f"\n{'='*70}")
    print(f"  SUBMISSION: {data.get('submissionId', 'unknown')}")
    print(f"  272299 LIVE ROUND POST-MORTEM (Iteration 2)")
    print(f"{'='*70}")

    osm_acts = parse_activities(data["activitiesLog"], "ASH_COATED_OSMIUM")
    pep_acts = parse_activities(data["activitiesLog"], "INTARIAN_PEPPER_ROOT")

    osm_buys, osm_sells = parse_trades(data["tradeHistory"], "ASH_COATED_OSMIUM")
    pep_buys, pep_sells = parse_trades(data["tradeHistory"], "INTARIAN_PEPPER_ROOT")

    print(f"\n  Osmium: {len(osm_acts):,} ticks, {len(osm_buys)} buy fills, {len(osm_sells)} sell fills")
    print(f"  Pepper: {len(pep_acts):,} ticks, {len(pep_buys)} buy fills, {len(pep_sells)} sell fills")

    all_ts = sorted(set(r["ts"] for r in osm_acts + pep_acts))
    osm_pos = rebuild_position(osm_buys, osm_sells, all_ts)
    pep_pos = rebuild_position(pep_buys, pep_sells, all_ts)

    analyze_pnl_trajectory(osm_acts, pep_acts)
    analyze_book_quality(osm_acts, pep_acts)
    analyze_spread_regime(osm_acts, pep_acts)
    analyze_ema_vs_static(osm_acts)
    analyze_inventory(osm_acts, osm_pos, "OSMIUM")
    analyze_inventory(pep_acts, pep_pos, "PEPPER")
    analyze_fill_quality(osm_buys, osm_sells, osm_acts, "OSMIUM")
    analyze_fill_quality(pep_buys, pep_sells, pep_acts, "PEPPER")
    analyze_osmium_ema_contamination(osm_acts)
    analyze_osmium_fill_breakdown(osm_buys, osm_sells, osm_acts, osm_pos)
    analyze_momentum_fade(osm_acts)
    analyze_pepper_trend_confirmation(pep_acts, pep_buys, pep_sells, pep_pos)


if __name__ == "__main__":
    main()
