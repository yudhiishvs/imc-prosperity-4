"""
Analyze bot-check submission logs to determine fill levels during empty-book events.

Submissions:
  287781: Main strategy (OSMIUM_EMPTY_BID=1333, OSMIUM_EMPTY_ASK=14689)
  287874: Bot-check probe, offsets [100, 10_000] — wide range
  287911: Bot-check probe, offsets [0, 100] — tight near fair value
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

repo_root = Path(__file__).resolve().parent.parent.parent

submissions = {
    "287781": {"label": "Main Strategy (EMPTY_BID=1333, EMPTY_ASK=14689)", "type": "main"},
    "287874": {"label": "Bot Check: Wide Probe [100, 10000]", "type": "probe"},
    "287911": {"label": "Bot Check: Tight Probe [0, 100]", "type": "probe"},
}

FAIR = 10_000

for sub_id, meta in submissions.items():
    log_path = repo_root / sub_id / f"{sub_id}.log"
    if not log_path.exists():
        print(f"Skipping {sub_id}: log not found at {log_path}")
        continue

    print(f"\n{'='*70}")
    print(f"  SUBMISSION {sub_id}: {meta['label']}")
    print(f"{'='*70}")

    raw = log_path.read_text()

    # The log is a single JSON object. Parse it.
    data = json.loads(raw)

    # Extract trades from the sandbox results
    # The structure contains trade entries with buyer/seller = "SUBMISSION"
    all_trades = []

    # Try different JSON structures
    if isinstance(data, dict):
        # Could be nested under various keys
        for key in data:
            val = data[key]
            if isinstance(val, list):
                all_trades.extend(val)
    elif isinstance(data, list):
        all_trades = data

    # Filter to SUBMISSION osmium trades only
    osmium_buys = []   # We bought (SUBMISSION is buyer)
    osmium_sells = []  # We sold (SUBMISSION is seller)

    for t in all_trades:
        if not isinstance(t, dict):
            continue
        sym = t.get("symbol", "")
        if sym != "ASH_COATED_OSMIUM":
            continue

        buyer = t.get("buyer", "")
        seller = t.get("seller", "")
        price = t.get("price", 0)
        qty = t.get("quantity", 0)
        ts = t.get("timestamp", 0)

        if buyer == "SUBMISSION":
            osmium_buys.append({"price": price, "qty": qty, "ts": ts})
        elif seller == "SUBMISSION":
            osmium_sells.append({"price": price, "qty": qty, "ts": ts})

    print(f"\n  Total OSMIUM trades: {len(osmium_buys) + len(osmium_sells)}")
    print(f"    Buys (we bought):  {len(osmium_buys)}")
    print(f"    Sells (we sold):   {len(osmium_sells)}")

    # ── Buy Analysis (prices we got filled at when buying) ──
    if osmium_buys:
        buy_prices = [t["price"] for t in osmium_buys]
        buy_below_fair = [p for p in buy_prices if p < FAIR]
        buy_at_fair = [p for p in buy_prices if p == FAIR]
        buy_above_fair = [p for p in buy_prices if p > FAIR]

        # Extreme buys (far from fair) — these are the probe fills
        extreme_buys = [t for t in osmium_buys if t["price"] < FAIR - 5]

        print(f"\n  ── BUY fills ──")
        print(f"    Min price:    {min(buy_prices):>10.0f}  (offset from fair: {FAIR - min(buy_prices):>+10.0f})")
        print(f"    Max price:    {max(buy_prices):>10.0f}")
        print(f"    Mean price:   {sum(buy_prices)/len(buy_prices):>10.1f}")
        print(f"    Below fair:   {len(buy_below_fair):>6}  (min={min(buy_below_fair):.0f})" if buy_below_fair else "    Below fair:        0")
        print(f"    At fair:      {len(buy_at_fair):>6}")
        print(f"    Above fair:   {len(buy_above_fair):>6}")

        if extreme_buys:
            extreme_prices = sorted(set(t["price"] for t in extreme_buys))
            total_extreme_qty = sum(t["qty"] for t in extreme_buys)
            print(f"\n    EXTREME BUY fills (>5 ticks below fair):")
            print(f"      Count:       {len(extreme_buys)}")
            print(f"      Total qty:   {total_extreme_qty}")
            print(f"      Price range: {min(extreme_prices):.0f} — {max(extreme_prices):.0f}")
            # Distribution
            price_hist = defaultdict(int)
            qty_hist = defaultdict(int)
            for t in extreme_buys:
                price_hist[int(t["price"])] += 1
                qty_hist[int(t["price"])] += t["qty"]
            print(f"      Fill distribution (price → count × qty):")
            for p in sorted(price_hist.keys()):
                offset_from_fair = FAIR - p
                print(f"        {p:>8.0f}  (fair-{offset_from_fair:>5})  ×{price_hist[p]:>3} fills, {qty_hist[p]:>5} units")

    # ── Sell Analysis ──
    if osmium_sells:
        sell_prices = [t["price"] for t in osmium_sells]
        sell_below_fair = [p for p in sell_prices if p < FAIR]
        sell_at_fair = [p for p in sell_prices if p == FAIR]
        sell_above_fair = [p for p in sell_prices if p > FAIR]

        extreme_sells = [t for t in osmium_sells if t["price"] > FAIR + 5]

        print(f"\n  ── SELL fills ──")
        print(f"    Min price:    {min(sell_prices):>10.0f}")
        print(f"    Max price:    {max(sell_prices):>10.0f}  (offset from fair: {max(sell_prices) - FAIR:>+10.0f})")
        print(f"    Mean price:   {sum(sell_prices)/len(sell_prices):>10.1f}")
        print(f"    Below fair:   {len(sell_below_fair):>6}")
        print(f"    At fair:      {len(sell_at_fair):>6}")
        print(f"    Above fair:   {len(sell_above_fair):>6}  (max={max(sell_above_fair):.0f})" if sell_above_fair else "    Above fair:        0")

        if extreme_sells:
            extreme_prices = sorted(set(t["price"] for t in extreme_sells))
            total_extreme_qty = sum(t["qty"] for t in extreme_sells)
            print(f"\n    EXTREME SELL fills (>5 ticks above fair):")
            print(f"      Count:       {len(extreme_sells)}")
            print(f"      Total qty:   {total_extreme_qty}")
            print(f"      Price range: {min(extreme_prices):.0f} — {max(extreme_prices):.0f}")
            price_hist = defaultdict(int)
            qty_hist = defaultdict(int)
            for t in extreme_sells:
                price_hist[int(t["price"])] += 1
                qty_hist[int(t["price"])] += t["qty"]
            print(f"      Fill distribution (price → count × qty):")
            for p in sorted(price_hist.keys()):
                offset_from_fair = p - FAIR
                print(f"        {p:>8.0f}  (fair+{offset_from_fair:>5})  ×{price_hist[p]:>3} fills, {qty_hist[p]:>5} units")

    # ── PnL from extreme fills only ──
    if osmium_buys or osmium_sells:
        extreme_buy_pnl = sum((FAIR - t["price"]) * t["qty"] for t in osmium_buys if t["price"] < FAIR - 5)
        extreme_sell_pnl = sum((t["price"] - FAIR) * t["qty"] for t in osmium_sells if t["price"] > FAIR + 5)
        print(f"\n  ── Extreme Fill PnL (if unwound at fair) ──")
        print(f"    From extreme buys:  {extreme_buy_pnl:>+10,.0f}")
        print(f"    From extreme sells: {extreme_sell_pnl:>+10,.0f}")
        print(f"    Total extreme PnL:  {extreme_buy_pnl + extreme_sell_pnl:>+10,.0f}")

print(f"\n{'='*70}")
print("  Analysis complete.")
print(f"{'='*70}")
