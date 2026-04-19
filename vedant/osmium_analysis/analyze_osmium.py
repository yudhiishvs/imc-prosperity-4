"""
analyze_osmium.py - Post-mortem analysis of a live round submission log.

Usage:
    python3 vedant/analyze_osmium.py 272299/272299.log

Data sources (from JSON log format):
  - activitiesLog: CSV with per-tick order book, mid price, PnL
  - tradeHistory:  list of actual fills (buyer/seller = "SUBMISSION" = us)
"""

import json
import sys
import re
from collections import defaultdict
from pathlib import Path

PRODUCT = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("272299/272299.log")


# ─── LOAD & PARSE ─────────────────────────────────────────────────────────────

def load_data(path: Path):
    print(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_activities(raw_csv: str, product: str):
    """Parse the activitiesLog CSV for a specific product."""
    rows = raw_csv.strip().split("\n")
    header = rows[0].split(";")
    
    records = []
    for row in rows[1:]:
        parts = row.split(";")
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        if rec.get("product", "").strip() != product:
            continue
        try:
            records.append({
                "day":       int(rec.get("day", 0)),
                "timestamp": int(rec.get("timestamp", 0)),
                "bid1":      float(rec["bid_price_1"]) if rec["bid_price_1"].strip() else None,
                "bid1v":     float(rec["bid_volume_1"]) if rec["bid_volume_1"].strip() else None,
                "ask1":      float(rec["ask_price_1"]) if rec["ask_price_1"].strip() else None,
                "ask1v":     float(rec["ask_volume_1"]) if rec["ask_volume_1"].strip() else None,
                "mid":       float(rec["mid_price"]) if rec["mid_price"].strip() else None,
                "pnl":       float(rec["profit_and_loss"]) if rec["profit_and_loss"].strip() else None,
            })
        except (KeyError, ValueError):
            continue
    return records


def parse_trades(trade_history: list, product: str):
    """Extract our fills from tradeHistory."""
    our_buys = []
    our_sells = []
    for t in trade_history:
        if t.get("symbol") != product:
            continue
        price = float(t.get("price", 0))
        qty = int(t.get("quantity", 0))
        if t.get("buyer") == "SUBMISSION":
            our_buys.append({"ts": t["timestamp"], "price": price, "qty": qty})
        elif t.get("seller") == "SUBMISSION":
            our_sells.append({"ts": t["timestamp"], "price": price, "qty": qty})
    return our_buys, our_sells


# ─── RECONSTRUCT POSITION ─────────────────────────────────────────────────────

def reconstruct_positions(our_buys, our_sells, tick_timestamps):
    """Walk through fills in time order to rebuild position at each timestamp."""
    events = sorted(
        [(t["ts"], +t["qty"]) for t in our_buys] +
        [(t["ts"], -t["qty"]) for t in our_sells]
    )
    
    pos = 0
    ev_idx = 0
    pos_by_ts = {}
    for ts in tick_timestamps:
        while ev_idx < len(events) and events[ev_idx][0] <= ts:
            pos += events[ev_idx][1]
            ev_idx += 1
        pos_by_ts[ts] = pos
    return pos_by_ts


# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

def analyze_inventory(acts, pos_by_ts):
    print("\n" + "="*62)
    print("  [1] INVENTORY DRIFT ANALYSIS")
    print("="*62)

    all_pos = [pos_by_ts.get(r["timestamp"], 0) for r in acts]
    if not all_pos:
        print("  No position data.")
        return

    stuck_long  = sum(1 for p in all_pos if p >= 40)
    stuck_short = sum(1 for p in all_pos if p <= -40)
    near_flat   = sum(1 for p in all_pos if abs(p) <= 5)
    total = len(all_pos)

    print(f"  Total ticks: {total:,}")
    print(f"  |pos| >= 40 (STUCK LONG):  {stuck_long:>6,}  ({100*stuck_long/total:.1f}%)")
    print(f"  |pos| <= -40 (STUCK SHORT): {stuck_short:>6,}  ({100*stuck_short/total:.1f}%)")
    print(f"  |pos| <= 5  (NEAR FLAT):   {near_flat:>6,}  ({100*near_flat/total:.1f}%)")

    # Streak analysis
    max_ls = max_ss = cur_ls = cur_ss = 0
    for p in all_pos:
        if p >= 40:   cur_ls += 1; cur_ss = 0
        elif p <= -40: cur_ss += 1; cur_ls = 0
        else:          cur_ls = cur_ss = 0
        max_ls = max(max_ls, cur_ls)
        max_ss = max(max_ss, cur_ss)

    print(f"\n  Longest STUCK LONG streak:  {max_ls:,} ticks")
    print(f"  Longest STUCK SHORT streak: {max_ss:,} ticks")
    if max_ls > 200 or max_ss > 200:
        print("  ⚠ DEATH SPIRAL: Strategy accumulated and could NOT unwind.")
    else:
        print("  ✅ No runaway inventory streaks.")

    # Distribution histogram
    buckets = defaultdict(int)
    for p in all_pos:
        buckets[(p // 10) * 10] += 1
    print("\n  Position distribution (per 10-unit bucket):")
    for b in sorted(buckets):
        bar = "█" * min(40, buckets[b] * 40 // total)
        print(f"    [{b:+4d}..{b+9:+4d}]: {bar:40s} {buckets[b]:,}")


def analyze_spread_captures(our_buys, our_sells, acts):
    print("\n" + "="*62)
    print("  [2] TRADE FILL QUALITY (Spread Capture)")
    print("="*62)

    total_buy_vol  = sum(t["qty"] for t in our_buys)
    total_sell_vol = sum(t["qty"] for t in our_sells)
    avg_buy  = sum(t["price"]*t["qty"] for t in our_buys)  / max(1, total_buy_vol)
    avg_sell = sum(t["price"]*t["qty"] for t in our_sells) / max(1, total_sell_vol)
    net_capture = avg_sell - avg_buy

    print(f"  Total BUY fills:  {len(our_buys):>5,}  ({total_buy_vol:,} units)")
    print(f"  Total SELL fills: {len(our_sells):>5,}  ({total_sell_vol:,} units)")
    print(f"  Net flow (buys-sells): {total_buy_vol - total_sell_vol:+,} units")
    print(f"\n  Avg buy price:  {avg_buy:.2f}")
    print(f"  Avg sell price: {avg_sell:.2f}")
    print(f"  Net spread capture per unit: {net_capture:+.2f} ticks")

    if net_capture < 0:
        print("  ❌ CRITICAL: We bought high and sold low — acting as liquidity TAKER.")
    elif net_capture < 1.0:
        print("  ⚠ Spread capture is very thin. Near break-even.")
    else:
        print("  ✅ Positive spread capture — market making working.")

    # Also compare our prices vs the market mid at each fill
    mid_by_ts = {r["timestamp"]: r["mid"] for r in acts if r["mid"] is not None}
    buy_vs_mid  = []
    sell_vs_mid = []
    for t in our_buys:
        m = mid_by_ts.get(t["ts"])
        if m: buy_vs_mid.append(t["price"] - m)
    for t in our_sells:
        m = mid_by_ts.get(t["ts"])
        if m: sell_vs_mid.append(t["price"] - m)

    if buy_vs_mid:
        avg_bv = sum(buy_vs_mid) / len(buy_vs_mid)
        print(f"\n  Avg BUY vs mid:  {avg_bv:+.2f}  (negative = we paid BELOW mid ✅, positive = above mid ❌)")
    if sell_vs_mid:
        avg_sv = sum(sell_vs_mid) / len(sell_vs_mid)
        print(f"  Avg SELL vs mid: {avg_sv:+.2f}  (positive = we sold ABOVE mid ✅, negative = below mid ❌)")


def analyze_pnl(acts):
    print("\n" + "="*62)
    print("  [3] PnL TRAJECTORY")
    print("="*62)

    pnls = [(r["timestamp"], r["pnl"]) for r in acts if r["pnl"] is not None]
    if not pnls:
        print("  No PnL data.")
        return

    n = len(pnls)
    qs = [pnls[:n//4], pnls[n//4:n//2], pnls[n//2:3*n//4], pnls[3*n//4:]]

    def delta(q):
        if len(q) < 2: return 0
        return q[-1][1] - q[0][1]

    labels = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
    deltas = [delta(q) for q in qs]

    print(f"  {'Quarter':<15} {'PnL Change':>12}")
    print(f"  {'-'*15} {'-'*12}")
    for label, d in zip(labels, deltas):
        icon = "✅" if d > 0 else "❌"
        print(f"  {label:<15} {d:>+12,.0f}  {icon}")

    total_delta = pnls[-1][1] - pnls[0][1]
    print(f"\n  Total PnL change: {total_delta:+,.0f}")
    print(f"  Final PnL:        {pnls[-1][1]:,.0f}")

    if deltas[3] < 0 and deltas[0] > 0:
        print("\n  ⚠ OVERFIT SIGNAL: Profitable early but lost money in Q4.")
        print("    The live market diverged from your backtest patterns.")
    elif all(d < 0 for d in deltas):
        print("\n  ❌ CONSISTENT LOSSES every quarter — structural strategy issue.")
    elif deltas[3] > max(deltas[0], deltas[1], deltas[2]):
        print("\n  ✅ Strategy IMPROVED over the round — good generalization.")


def analyze_ema_lag(acts, alpha=0.1122):
    print("\n" + "="*62)
    print(f"  [4] EMA LAG ANALYSIS (alpha={alpha})")
    print("="*62)

    mids = [r["mid"] for r in acts if r["mid"] is not None]
    if len(mids) < 10:
        print("  Not enough mid price data.")
        return

    ema = mids[0]
    lags = []
    for m in mids[1:]:
        ema = alpha * m + (1 - alpha) * ema
        lags.append(abs(ema - m))

    avg_lag = sum(lags) / len(lags)
    max_lag = max(lags)
    p90_lag = sorted(lags)[int(0.9 * len(lags))]

    print(f"  Avg |EMA - mid|:  {avg_lag:.2f} ticks")
    print(f"  90th pct lag:     {p90_lag:.2f} ticks")
    print(f"  Max lag:          {max_lag:.2f} ticks")

    spread_sample = [
        (r["ask1"] - r["bid1"])
        for r in acts
        if r["bid1"] is not None and r["ask1"] is not None
    ]
    avg_spread = sum(spread_sample) / len(spread_sample) if spread_sample else 5.0

    print(f"\n  Avg market spread: {avg_spread:.2f} ticks")
    print(f"  EMA lag / spread:  {avg_lag / avg_spread:.2f}x")

    if avg_lag > avg_spread:
        print("  ❌ EMA lags by MORE than 1 full spread — fair value signal is stale.")
        print("    Consider INCREASING OSMIUM_EMA_ALPHA to make the signal react faster.")
    elif avg_lag > avg_spread * 0.5:
        print("  ⚠ EMA lag is significant (>0.5 spreads). Might be missing fast moves.")
    else:
        print("  ✅ EMA lag is within reason relative to spread.")


def analyze_spread_regime(acts):
    print("\n" + "="*62)
    print("  [5] MARKET REGIME vs BACKTEST ASSUMPTIONS")
    print("="*62)

    spreads = [
        r["ask1"] - r["bid1"]
        for r in acts
        if r["bid1"] is not None and r["ask1"] is not None
    ]
    mids = [r["mid"] for r in acts if r["mid"] is not None]

    if not spreads or not mids:
        print("  Not enough data.")
        return

    avg_spread = sum(spreads) / len(spreads)
    min_spread = min(spreads)
    max_spread = max(spreads)

    # Volatility: std of tick-to-tick mid returns
    returns = [abs(mids[i] - mids[i-1]) for i in range(1, len(mids))]
    avg_move = sum(returns) / len(returns) if returns else 0

    print(f"  Avg bid-ask spread:   {avg_spread:.2f} ticks")
    print(f"  Spread range:         [{min_spread:.0f}, {max_spread:.0f}] ticks")
    print(f"  Avg tick-to-tick move: {avg_move:.3f} ticks")
    print(f"  Spread: {avg_spread:.2f}  |  Avg move: {avg_move:.3f}")

    if avg_spread <= 2:
        print("\n  ⚠ Very tight spread environment. High sensitivity to quote offset params.")
        print("    At 1-tick spreads, OUTER_QUOTE_OFFSET > 2 is effectively ghost quoting.")
    if avg_move > avg_spread:
        print("  ⚠ Market moves faster than the spread — momentum fade is CRITICAL.")
        print("    MOMENTUM_AGRESS_SCALE being very high may have caused runaway positions.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    data = load_data(LOG_PATH)

    print(f"\n{'='*62}")
    print(f"  SUBMISSION: {data.get('submissionId', 'unknown')}")
    print(f"  PRODUCT:    {PRODUCT}")
    print(f"{'='*62}")

    acts = parse_activities(data["activitiesLog"], PRODUCT)
    our_buys, our_sells = parse_trades(data["tradeHistory"], PRODUCT)

    print(f"\n  Activity rows:  {len(acts):,}")
    print(f"  Our BUY fills:  {len(our_buys):,}")
    print(f"  Our SELL fills: {len(our_sells):,}")

    tick_timestamps = sorted(set(r["timestamp"] for r in acts))
    pos_by_ts = reconstruct_positions(our_buys, our_sells, tick_timestamps)

    analyze_inventory(acts, pos_by_ts)
    analyze_spread_captures(our_buys, our_sells, acts)
    analyze_pnl(acts)
    analyze_ema_lag(acts, alpha=0.1122)
    analyze_spread_regime(acts)

    print("\n" + "="*62)
    print("  SUMMARY OF HYPOTHESES")
    print("="*62)
    print("""
  Review the flags above. Common causes of out-of-sample failure:

  ① INVENTORY DEATH SPIRAL
    If stuck_long/stuck_short % is high, the strategy accumulated a
    directional position it couldn't unwind. Root cause: EMA chasing
    price (alpha too low), or inventory skew too weak to force sells.

  ② LIQUIDITY TAKER (negative spread capture)
    Negative avg_sell - avg_buy means aggressive taking is firing at
    bad prices. TAKE_UNWIND_WIDTH or SYMMETRIC_ZONE being too large
    causes us to hit bids/lift asks at unfavorable moments.

  ③ EMA LAG > 1 SPREAD
    If EMA lags by more than one spread, every quote is stale. This
    is the core overfit mechanism: the backtests may have had specific
    price paths where a slow EMA worked, but it fails in general.

  ④ GHOST QUOTING (outer offset > spread)
    If avg_spread ~ 2 and OUTER_QUOTE_OFFSET = 15, the outer level
    never fills and all volume goes through a single inner level.

  ⑤ MOMENTUM SCALE INSTABILITY
    AGRESS_SCALE ~9.8 means a 1-tick move triggers 9.8x volume on one
    side. In a noisy live market, this blows up positions rapidly.
""")

if __name__ == "__main__":
    main()
