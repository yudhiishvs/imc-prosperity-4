"""
ash_l2_timing.py
----------------
Checks whether L2 large volume on ASH appears at the same tick as trades
(i.e., the whale consumes both L1 and L2 simultaneously) or whether the
L2 volume is a pre-existing resting backstop order.

Approach:
1. For each tick, record L1/L2 bid and ask volumes.
2. Find ticks where L2 volume is "large" (above a threshold, e.g. top quartile).
3. Check: does a trade occur at the same timestamp?
4. Check: does the L2 volume *appear for the first time* at that tick (new order)
   or was it already present 1-2 ticks earlier (resting backstop)?
5. Cross-reference with ASH price breaks to see if large-L2 ticks coincide with
   whale break ticks.

Output:
- Summary table: pre-existing vs new-appearing L2 volume, with/without trade
- Print aligned examples so you can eyeball the sequence
"""

import os
import pandas as pd
import numpy as np

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DAYS      = [-1, 0, 1]
PRODUCT   = "ASH_COATED_OSMIUM"
L2_THRESH_PERCENTILE = 75   # "large" = above this percentile of all L2 vols
BREAK_THRESH         = 6    # |Δmid| >= this → price break tick


def load_data():
    prices_list, trades_list = [], []
    for day in DAYS:
        p = pd.read_csv(
            os.path.join(DATA_DIR, f"prices_round_2_day_{day}.csv"), sep=";"
        )
        t = pd.read_csv(
            os.path.join(DATA_DIR, f"trades_round_2_day_{day}.csv"), sep=";"
        )
        p["day"] = day
        t["day"] = day
        prices_list.append(p)
        trades_list.append(t)
    prices = pd.concat(prices_list, ignore_index=True)
    trades = pd.concat(trades_list, ignore_index=True)

    ash_p = prices[prices["product"] == PRODUCT].copy().sort_values(["day", "timestamp"])
    ash_t = trades[trades["symbol"] == PRODUCT].copy().sort_values(["day", "timestamp"])
    return ash_p, ash_t


def analyze(ash_p: pd.DataFrame, ash_t: pd.DataFrame):
    # ── Build per-tick L2 volume (bid2 + ask2) ──────────────────────────────
    ash_p = ash_p.copy()
    ash_p["bid_vol_2"] = pd.to_numeric(ash_p["bid_volume_2"], errors="coerce").fillna(0)
    ash_p["ask_vol_2"] = pd.to_numeric(ash_p["ask_volume_2"], errors="coerce").fillna(0)
    ash_p["l2_total"]  = ash_p["bid_vol_2"] + ash_p["ask_vol_2"]

    # Previous-tick L2 (shifted within each day group)
    ash_p = ash_p.sort_values(["day", "timestamp"])
    ash_p["l2_prev"] = ash_p.groupby("day")["l2_total"].shift(1).fillna(0)

    # "L2 appeared this tick" = current L2 > 0 and previous L2 == 0
    ash_p["l2_new"]      = (ash_p["l2_total"] > 0) & (ash_p["l2_prev"] == 0)
    # "L2 was already there" = current L2 > 0 and previous L2 > 0
    ash_p["l2_resting"]  = (ash_p["l2_total"] > 0) & (ash_p["l2_prev"] > 0)

    # ── Price breaks ────────────────────────────────────────────────────────
    ash_p["mid_price"] = pd.to_numeric(ash_p["mid_price"], errors="coerce")
    ash_p["delta_mid"] = ash_p.groupby("day")["mid_price"].diff().fillna(0)
    ash_p["is_break"]  = ash_p["delta_mid"].abs() >= BREAK_THRESH

    # ── Large L2 threshold ──────────────────────────────────────────────────
    nonzero_l2 = ash_p.loc[ash_p["l2_total"] > 0, "l2_total"]
    l2_thresh  = np.percentile(nonzero_l2, L2_THRESH_PERCENTILE)
    ash_p["l2_large"] = ash_p["l2_total"] >= l2_thresh

    print(f"L2 volume threshold (p{L2_THRESH_PERCENTILE} of non-zero): {l2_thresh:.0f}")
    print(f"Total ticks: {len(ash_p)}")
    print(f"Ticks with any L2:   {(ash_p['l2_total'] > 0).sum()}")
    print(f"Ticks with large L2: {ash_p['l2_large'].sum()}")
    print()

    # ── Build trade presence per (day, timestamp) ────────────────────────────
    trade_ticks = ash_t.groupby(["day", "timestamp"]).size().reset_index(name="n_trades")
    ash_p = ash_p.merge(trade_ticks, on=["day", "timestamp"], how="left")
    ash_p["has_trade"] = ash_p["n_trades"].notna() & (ash_p["n_trades"] > 0)

    # ── Summary: for large-L2 ticks, was L2 new or resting? + trade? ────────
    large = ash_p[ash_p["l2_large"]]
    total = len(large)

    cats = {
        "new L2 + trade":      ((large["l2_new"])      & (large["has_trade"])).sum(),
        "new L2 + no trade":   ((large["l2_new"])      & (~large["has_trade"])).sum(),
        "resting L2 + trade":  ((large["l2_resting"])  & (large["has_trade"])).sum(),
        "resting L2 + no trade": ((large["l2_resting"]) & (~large["has_trade"])).sum(),
    }

    print("=== Large-L2 ticks breakdown ===")
    for label, count in cats.items():
        print(f"  {label:35s}: {count:5d}  ({100*count/total:.1f}%)")
    print(f"  {'TOTAL':35s}: {total:5d}")
    print()

    # ── Subset: large-L2 ticks that also have a price break ─────────────────
    large_break = large[large["is_break"]]
    total_b = len(large_break)
    if total_b:
        cats_b = {
            "new L2 + trade":        ((large_break["l2_new"])     & (large_break["has_trade"])).sum(),
            "new L2 + no trade":     ((large_break["l2_new"])     & (~large_break["has_trade"])).sum(),
            "resting L2 + trade":    ((large_break["l2_resting"]) & (large_break["has_trade"])).sum(),
            "resting L2 + no trade": ((large_break["l2_resting"]) & (~large_break["has_trade"])).sum(),
        }
        print("=== Large-L2 + price-break ticks breakdown ===")
        for label, count in cats_b.items():
            print(f"  {label:35s}: {count:5d}  ({100*count/total_b:.1f}%)")
        print(f"  {'TOTAL':35s}: {total_b:5d}")
        print()

    # ── Show example windows: 3 ticks before/after a large-L2 trade tick ────
    example_ticks = large[large["has_trade"] & large["l2_new"]].head(5)
    if len(example_ticks):
        print("=== Example windows: new large L2 appeared at same tick as trade ===")
    else:
        example_ticks = large[large["has_trade"] & large["l2_resting"]].head(5)
        print("=== Example windows: resting large L2 at same tick as trade ===")

    for _, row in example_ticks.iterrows():
        d, ts = int(row["day"]), int(row["timestamp"])
        window = ash_p[(ash_p["day"] == d) & (ash_p["timestamp"].between(ts - 300, ts + 300))]
        print(f"\n  Day {d}, ts {ts}  (delta_mid={row['delta_mid']:+.1f}, l2_total={row['l2_total']:.0f})")
        print(f"  {'ts':>8}  {'bid1':>6}  {'bv1':>4}  {'bid2':>6}  {'bv2':>4}  "
              f"{'ask1':>6}  {'av1':>4}  {'ask2':>6}  {'av2':>4}  {'trade?':>7}  {'break?':>7}")
        for _, wr in window.iterrows():
            trade_flag = f"{int(wr['n_trades']) if pd.notna(wr['n_trades']) else 0:>7}"
            break_flag = "  *" if wr["is_break"] else "   "
            def _iv(v, default=0):
                try: return int(v) if pd.notna(v) else default
                except: return default
            print(f"  {int(wr['timestamp']):>8}  "
                  f"{str(wr.get('bid_price_1','')):>6}  {_iv(wr.get('bid_volume_1')):>4}  "
                  f"{str(wr.get('bid_price_2','')):>6}  {_iv(wr.get('bid_vol_2')):>4}  "
                  f"{str(wr.get('ask_price_1','')):>6}  {_iv(wr.get('ask_volume_1')):>4}  "
                  f"{str(wr.get('ask_price_2','')):>6}  {_iv(wr.get('ask_vol_2')):>4}  "
                  f"{trade_flag}  {break_flag}")

    # ── Timing: how many ticks before a trade does large L2 appear? ─────────
    # For each trade tick, look back up to 10 ticks to find when L2 first appeared
    print("\n=== L2 lead time before trade (how many ticks L2 was resting before trade hit) ===")
    trade_large = large[large["has_trade"]].copy()
    lead_times = []
    for _, row in trade_large.iterrows():
        d, ts = int(row["day"]), int(row["timestamp"])
        # Walk backward to find when this L2 level first appeared
        preceding = ash_p[
            (ash_p["day"] == d) & (ash_p["timestamp"] < ts)
        ].tail(10)
        # Find the last tick where L2 was 0 (before the current L2 block started)
        was_zero = preceding[preceding["l2_total"] == 0]
        if len(was_zero) == 0:
            lead = len(preceding)  # L2 was there for all 10 preceding ticks
        else:
            last_zero_idx = was_zero.index[-1]
            ticks_after_zero = (preceding.index > last_zero_idx).sum()
            lead = ticks_after_zero
        lead_times.append(lead)

    if lead_times:
        lt = np.array(lead_times)
        print(f"  N trade ticks with large L2: {len(lt)}")
        print(f"  Lead-time distribution (ticks):")
        for cutoff in [0, 1, 2, 3, 5, 10]:
            print(f"    == {cutoff} ticks: {(lt == cutoff).sum():4d}  ({100*(lt == cutoff).mean():.1f}%)")
        print(f"  Mean lead time: {lt.mean():.2f} ticks | Median: {np.median(lt):.1f}")


if __name__ == "__main__":
    print(f"Loading ASH data for days {DAYS}...")
    ash_p, ash_t = load_data()
    analyze(ash_p, ash_t)
