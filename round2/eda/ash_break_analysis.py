"""
ash_break_analysis.py
---------------------
Detects "break" events in the ASH mid-price and classifies whether each
break is driven by:

  WHALE  — one or a few large trades dominating the volume
  SWARM  — many small trades arriving in a tight window

A break is defined as a mid-price move whose absolute value exceeds
BREAK_THRESHOLD ticks in a single 100-ts step.

For each break we collect all trades within ±WINDOW_TS timestamps,
then compute:
  n_trades       : number of trades in the window
  max_qty        : single largest trade
  total_qty      : sum of all trade quantities
  dominant_share : max_qty / total_qty  (1.0 = pure whale, <0.5 = swarm)
  time_span      : last_trade_ts - first_trade_ts in the window

Classification:
  WHALE  if dominant_share >= WHALE_THRESHOLD
  SWARM  otherwise

Outputs
-------
  - Console table of every break event
  - Summary stats (whale vs swarm counts, avg trade profile per class)
  - Plot: trade qty distribution at breaks vs quiet periods
  - Plot: timeline of breaks coloured by Whale/Swarm
  - Saved to eda_output/

Usage:
    python ash_break_analysis.py
    python ash_break_analysis.py --threshold 8 --window 500
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")
OUT_DIR  = os.path.join(_HERE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS    = [-1, 0, 1]
PRODUCT = "ASH_COATED_OSMIUM"

# ── Defaults (overridable via CLI) ────────────────────────────────────────────
BREAK_THRESHOLD = 6      # ticks: mid-price move per step to count as a break
WINDOW_TS       = 400    # ± timestamps around break to collect trades
WHALE_THRESHOLD = 0.60   # dominant_share ≥ this → WHALE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices(days=DAYS):
    frames = []
    for day in days:
        path = os.path.join(DATA_DIR, f"prices_round_2_day_{day}.csv")
        df   = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    return df[df["mid_price"] > 0].copy()


def load_trades(days=DAYS):
    frames = []
    for day in days:
        path = os.path.join(DATA_DIR, f"trades_round_2_day_{day}.csv")
        df   = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"symbol": "product"})
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    return df


# ---------------------------------------------------------------------------
# Break detection
# ---------------------------------------------------------------------------

def detect_breaks(prices, threshold):
    """
    Returns a DataFrame of break events for ASH with columns:
      day, timestamp, global_ts, mid_price, prev_mid, move
    """
    ash = prices[prices["product"] == PRODUCT].sort_values("global_ts").copy()
    ash["prev_mid"] = ash["mid_price"].shift(1)
    ash["move"]     = ash["mid_price"] - ash["prev_mid"]
    breaks = ash[ash["move"].abs() >= threshold].copy()
    return breaks.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Break classification
# ---------------------------------------------------------------------------

def classify_breaks(breaks, trades, window_ts):
    """
    For each break, gather trades in [global_ts - window_ts, global_ts + window_ts]
    and classify as WHALE or SWARM.

    Returns a list of dicts with full stats per break.
    """
    ash_trades = trades[trades["product"] == PRODUCT].copy()
    records = []

    for _, brk in breaks.iterrows():
        t0 = brk["global_ts"]
        lo = t0 - window_ts
        hi = t0 + window_ts

        window = ash_trades[(ash_trades["global_ts"] >= lo) &
                            (ash_trades["global_ts"] <= hi)]

        if window.empty:
            records.append({
                "day":            int(brk["day"]),
                "timestamp":      int(brk["timestamp"]),
                "global_ts":      t0,
                "move":           brk["move"],
                "mid_before":     brk["prev_mid"],
                "mid_after":      brk["mid_price"],
                "n_trades":       0,
                "max_qty":        0,
                "total_qty":      0,
                "dominant_share": np.nan,
                "time_span":      0,
                "label":          "NO_TRADES",
            })
            continue

        n_trades       = len(window)
        max_qty        = window["quantity"].max()
        total_qty      = window["quantity"].sum()
        dominant_share = max_qty / total_qty if total_qty > 0 else np.nan
        time_span      = window["global_ts"].max() - window["global_ts"].min()

        label = "WHALE" if dominant_share >= WHALE_THRESHOLD else "SWARM"

        records.append({
            "day":            int(brk["day"]),
            "timestamp":      int(brk["timestamp"]),
            "global_ts":      t0,
            "move":           brk["move"],
            "mid_before":     brk["prev_mid"],
            "mid_after":      brk["mid_price"],
            "n_trades":       n_trades,
            "max_qty":        int(max_qty),
            "total_qty":      int(total_qty),
            "dominant_share": dominant_share,
            "time_span":      int(time_span),
            "label":          label,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_report(classified, threshold, window_ts):
    print("=" * 80)
    print(f"ASH MID-PRICE BREAK ANALYSIS")
    print(f"  Break threshold : |move| >= {threshold} ticks per step")
    print(f"  Trade window    : ±{window_ts} timestamps around break")
    print(f"  Whale threshold : dominant_share >= {WHALE_THRESHOLD:.0%}")
    print("=" * 80)

    total   = len(classified)
    whales  = (classified["label"] == "WHALE").sum()
    swarms  = (classified["label"] == "SWARM").sum()
    no_trd  = (classified["label"] == "NO_TRADES").sum()

    print(f"\nTotal breaks detected : {total}")
    print(f"  WHALE             : {whales}  ({100*whales/max(1,total):.0f}%)")
    print(f"  SWARM             : {swarms}  ({100*swarms/max(1,total):.0f}%)")
    print(f"  No trades nearby  : {no_trd}  ({100*no_trd/max(1,total):.0f}%)")

    for label in ["WHALE", "SWARM"]:
        sub = classified[classified["label"] == label]
        if sub.empty:
            continue
        print(f"\n── {label} profile ({len(sub)} events) ──────────────────────")
        print(f"  Avg |move|          : {sub['move'].abs().mean():.2f} ticks")
        print(f"  Avg n_trades        : {sub['n_trades'].mean():.1f}")
        print(f"  Avg max_qty         : {sub['max_qty'].mean():.1f}")
        print(f"  Avg total_qty       : {sub['total_qty'].mean():.1f}")
        print(f"  Avg dominant_share  : {sub['dominant_share'].mean():.2%}")
        print(f"  Avg time_span (ts)  : {sub['time_span'].mean():.0f}")
        print(f"  Move distribution   : "
              f"min={sub['move'].min():.1f}  "
              f"mean={sub['move'].mean():.1f}  "
              f"max={sub['move'].max():.1f}")

    print(f"\n── Per-day breakdown ─────────────────────────────────────────")
    for day in DAYS:
        sub = classified[classified["day"] == day]
        if sub.empty:
            print(f"  Day {day:+d}: no breaks")
            continue
        w = (sub["label"] == "WHALE").sum()
        s = (sub["label"] == "SWARM").sum()
        avg_move = sub["move"].abs().mean()
        print(f"  Day {day:+d}: {len(sub):>3} breaks  "
              f"WHALE={w:>2}  SWARM={s:>2}  "
              f"avg|move|={avg_move:.2f}  "
              f"avg_trades={sub['n_trades'].mean():.1f}")

    print(f"\n── Full break event table ────────────────────────────────────")
    print(f"  {'Day':>4}  {'TS':>7}  {'Move':>6}  {'N':>4}  "
          f"{'MaxQty':>6}  {'TotQty':>6}  {'Dom%':>6}  Label")
    print("  " + "-" * 62)
    for _, row in classified.iterrows():
        if row["label"] == "NO_TRADES":
            print(f"  {int(row['day']):>4}  {int(row['timestamp']):>7}  "
                  f"{row['move']:>+6.1f}  {'—':>4}  {'—':>6}  {'—':>6}  {'—':>6}  NO_TRADES")
        else:
            print(f"  {int(row['day']):>4}  {int(row['timestamp']):>7}  "
                  f"{row['move']:>+6.1f}  {int(row['n_trades']):>4}  "
                  f"{int(row['max_qty']):>6}  {int(row['total_qty']):>6}  "
                  f"{row['dominant_share']:>6.1%}  {row['label']}")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_break_timeline(prices, classified, save=True):
    """Timeline of ASH mid-price with break events coloured by Whale/Swarm."""
    fig, axes = plt.subplots(len(DAYS), 1, figsize=(18, 10))
    colors = {"WHALE": "#c62828", "SWARM": "#1565c0", "NO_TRADES": "#9e9e9e"}

    for ax, day in zip(axes, DAYS):
        p = prices[(prices["product"] == PRODUCT) & (prices["day"] == day)]
        ax.plot(p["timestamp"], p["mid_price"],
                lw=0.6, color="black", alpha=0.7, label="Mid price")

        for _, row in classified[classified["day"] == day].iterrows():
            if row["label"] == "NO_TRADES":
                continue
            ax.axvline(row["timestamp"], color=colors[row["label"]],
                       alpha=0.6, lw=1.2,
                       label=row["label"])
            ax.annotate(
                f"{row['move']:+.0f}",
                xy=(row["timestamp"], row["mid_after"]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=6, color=colors[row["label"]],
                ha="center",
            )

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
        ax.set_title(f"ASH  Day {day:+d} — Mid Price with Break Events", fontsize=9)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Mid Price")
        ax.grid(True, alpha=0.25)

    fig.suptitle("ASH Break Events: WHALE (red) vs SWARM (blue)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "C_ash_break_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → C_ash_break_timeline.png")
    else:
        plt.show()
    return fig


def plot_trade_profiles(classified, trades, save=True):
    """
    Two panels:
      Left:  trade quantity distribution at WHALE vs SWARM breaks
      Right: dominant_share distribution (what fraction of volume one trade owns)
    """
    ash_trades = trades[trades["product"] == PRODUCT].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: qty distribution at whale vs swarm breaks vs quiet
    ax = axes[0]
    whale_ts = set()
    swarm_ts = set()
    for _, row in classified.iterrows():
        window = ash_trades[
            (ash_trades["global_ts"] >= row["global_ts"] - WINDOW_TS) &
            (ash_trades["global_ts"] <= row["global_ts"] + WINDOW_TS)
        ]
        if row["label"] == "WHALE":
            whale_ts.update(window.index.tolist())
        elif row["label"] == "SWARM":
            swarm_ts.update(window.index.tolist())

    break_idx = whale_ts | swarm_ts
    quiet_idx = set(ash_trades.index) - break_idx

    whale_qty = ash_trades.loc[list(whale_ts), "quantity"] if whale_ts else pd.Series([], dtype=float)
    swarm_qty = ash_trades.loc[list(swarm_ts), "quantity"] if swarm_ts else pd.Series([], dtype=float)
    quiet_qty = ash_trades.loc[list(quiet_idx), "quantity"] if quiet_idx else pd.Series([], dtype=float)

    bins = range(0, int(ash_trades["quantity"].max()) + 2, 1)
    if len(whale_qty):
        ax.hist(whale_qty, bins=bins, alpha=0.6, color="#c62828",
                density=True, label=f"WHALE breaks (n={len(whale_qty)})")
    if len(swarm_qty):
        ax.hist(swarm_qty, bins=bins, alpha=0.6, color="#1565c0",
                density=True, label=f"SWARM breaks (n={len(swarm_qty)})")
    if len(quiet_qty):
        ax.hist(quiet_qty, bins=bins, alpha=0.4, color="#4caf50",
                density=True, label=f"Quiet (n={len(quiet_qty)})")
    ax.set_title("Trade Qty Distribution\nat Break Events vs Quiet", fontsize=10)
    ax.set_xlabel("Trade Quantity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: dominant_share histogram
    ax = axes[1]
    has_trades = classified[classified["label"].isin(["WHALE", "SWARM"])]
    ax.hist(has_trades["dominant_share"], bins=20, color="#5c35c1",
            alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(WHALE_THRESHOLD, color="red", lw=1.5, ls="--",
               label=f"Whale threshold ({WHALE_THRESHOLD:.0%})")
    ax.set_title("Dominant Share Distribution\n(max_qty / total_qty at break)", fontsize=10)
    ax.set_xlabel("Dominant Share")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: n_trades at break vs move size scatter
    ax = axes[2]
    has_trades = classified[classified["n_trades"] > 0].copy()
    whale_sub = has_trades[has_trades["label"] == "WHALE"]
    swarm_sub = has_trades[has_trades["label"] == "SWARM"]
    ax.scatter(whale_sub["move"].abs(), whale_sub["n_trades"],
               color="#c62828", alpha=0.7, s=40, label="WHALE", edgecolors="none")
    ax.scatter(swarm_sub["move"].abs(), swarm_sub["n_trades"],
               color="#1565c0", alpha=0.7, s=40, label="SWARM", edgecolors="none")
    ax.set_title("Break Size vs # Trades in Window\n(bigger move = whale or swarm?)", fontsize=10)
    ax.set_xlabel("|Move| (ticks)")
    ax.set_ylabel("# Trades in ±window")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("ASH Break Classification — Trade Profiles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(OUT_DIR, "C_ash_break_profiles.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → C_ash_break_profiles.png")
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(threshold=BREAK_THRESHOLD, window_ts=WINDOW_TS, save=True):
    print("Loading data...")
    prices = load_prices()
    trades = load_trades()

    ash_prices = prices[prices["product"] == PRODUCT]
    ash_trades = trades[trades["product"] == PRODUCT]
    print(f"  ASH price ticks : {len(ash_prices):,}")
    print(f"  ASH trades      : {len(ash_trades):,}")

    print(f"\nDetecting breaks (|move| >= {threshold} ticks)...")
    breaks = detect_breaks(prices, threshold)
    print(f"  Found {len(breaks)} break events")

    print("Classifying breaks (Whale vs Swarm)...")
    classified = classify_breaks(breaks, trades, window_ts)

    print_report(classified, threshold, window_ts)

    print("Generating plots...")
    plot_break_timeline(prices, classified, save=save)
    plot_trade_profiles(classified, trades, save=save)

    # Save results table
    out_csv = os.path.join(OUT_DIR, "C_ash_breaks.csv")
    classified.to_csv(out_csv, index=False)
    print(f"  saved → C_ash_breaks.csv")

    return classified


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=BREAK_THRESHOLD,
                        help="Min |mid-price move| per step to count as a break")
    parser.add_argument("--window",    type=int,   default=WINDOW_TS,
                        help="±timestamps around break to collect trades")
    parser.add_argument("--whale",     type=float, default=WHALE_THRESHOLD,
                        help="dominant_share >= this → WHALE")
    parser.add_argument("--no-save",   action="store_true",
                        help="Show plots interactively instead of saving")
    args = parser.parse_args()

    WHALE_THRESHOLD = args.whale
    run(threshold=args.threshold, window_ts=args.window, save=not args.no_save)
