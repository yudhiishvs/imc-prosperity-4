"""
log_analysis.py  (Round 2)
--------------------------
Parses prosperity4 submission logs, plots actual trade fills, PnL trajectories,
and identifies missed opportunities — situations where price moved in our favour
but we weren't positioned to capture it.

Log format: JSON with keys
  activitiesLog  — semicolon-delimited CSV (day;ts;product;bid/ask levels;mid;pnl)
  graphLog       — timestamp;value CSV (total cumulative PnL)
  profit         — final total PnL (float)
  positions      — [{"symbol", "quantity"}, ...] end state

Usage:
    uv run python log_analysis.py                  # all logs in logs/
    uv run python log_analysis.py --log logs/307686/307686.json

Output:
    plots/  — PNG plots saved here
    Printed analysis: PnL stats, whale detection, missed trade windows
"""

import argparse
import io
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = Path(__file__).parent
LOGS_DIR   = SCRIPT_DIR / "logs"
PLOTS_DIR  = SCRIPT_DIR / "log_plots"
PLOTS_DIR.mkdir(exist_ok=True)

ASH    = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

# Known constants
ASH_FAIR    = 10_000
PEPPER_SLOPE = 0.1001      # ticks per timestamp, anchored at first tick

# Spread threshold for "wide" events
ASH_WIDE_SPREAD = 14       # median ~16, flag ticks > this
BREAK_THRESH    = 6        # |Δmid| ≥ this → price break (whale / swarm)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_log(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_activities(log: dict) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(log["activitiesLog"]), sep=";")
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df


def split_products(df: pd.DataFrame):
    ash = df[df["product"] == ASH].copy().reset_index(drop=True)
    pep = df[df["product"] == PEPPER].copy().reset_index(drop=True)
    return ash, pep


# ─────────────────────────────────────────────────────────────────────────────
# Derived features
# ─────────────────────────────────────────────────────────────────────────────

def add_derived(df: pd.DataFrame, fair_const: Optional[float] = None) -> pd.DataFrame:
    df = df.copy()
    df["spread"]    = df["ask_price_1"] - df["bid_price_1"]
    df["delta_mid"] = df["mid_price"].diff().fillna(0)
    df["is_break"]  = df["delta_mid"].abs() >= BREAK_THRESH
    df["delta_pnl"] = df["profit_and_loss"].diff().fillna(0)

    if fair_const is not None:
        df["fair"]      = fair_const
        df["dev_mid"]   = df["mid_price"] - fair_const
    else:
        # PEPPER: infer trend from first mid + slope
        first_mid = df["mid_price"].iloc[0]
        first_ts  = df["timestamp"].iloc[0]
        df["fair"]    = first_mid + PEPPER_SLOPE * (df["timestamp"] - first_ts)
        df["dev_mid"] = df["mid_price"] - df["fair"]

    # Infer position from PnL changes — rough approximation:
    # Δpnl ≈ position * Δmid  ⟹  pos ≈ Δpnl / Δmid (when Δmid ≠ 0)
    mask = df["delta_mid"].abs() > 0.5
    df["pos_est"] = np.nan
    df.loc[mask, "pos_est"] = (df.loc[mask, "delta_pnl"] /
                               df.loc[mask, "delta_mid"]).round()
    df["pos_est"] = df["pos_est"].ffill().fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Missed trade detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_missed_buys(df: pd.DataFrame, pos_limit: int = 80) -> pd.DataFrame:
    """
    Ticks where:
      - Price was below fair (good buy opportunity)
      - Estimated position was below pos_limit
      - No positive PnL captured on the following tick (Δpnl ≤ 0)
    """
    missed = df[
        (df["dev_mid"] < -2) &
        (df["pos_est"] < pos_limit * 0.9) &
        (df["delta_pnl"].shift(-1).fillna(0) <= 0)
    ].copy()
    return missed


def detect_missed_sells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ticks where:
      - Price was above fair (good sell opportunity for ASH)
      - Position was estimated positive
      - No positive PnL captured on the following tick
    """
    missed = df[
        (df["dev_mid"] > 2) &
        (df["pos_est"] > 5) &
        (df["delta_pnl"].shift(-1).fillna(0) <= 0)
    ].copy()
    return missed


def detect_whale_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price breaks that coincide with PnL jumps — likely a whale filling our passive quotes.
    """
    hits = df[df["is_break"] & (df["delta_pnl"].abs() > 50)].copy()
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(product: str, df: pd.DataFrame, final_pnl: float):
    print(f"\n{'─'*60}")
    print(f"  {product}")
    print(f"{'─'*60}")
    print(f"  Final PnL:       {final_pnl:>10,.2f}")
    print(f"  Ticks:           {len(df):>10,}")
    print(f"  Mid price range: {df['mid_price'].min():.1f} – {df['mid_price'].max():.1f}")
    print(f"  Spread (med):    {df['spread'].median():.1f}  (mean {df['spread'].mean():.1f})")
    print(f"  Price breaks:    {df['is_break'].sum():>10,}")

    breaks = df[df["is_break"]]
    if len(breaks) > 0:
        up   = (breaks["delta_mid"] > 0).sum()
        down = (breaks["delta_mid"] < 0).sum()
        print(f"    Up breaks:     {up:>8,}  ({100*up/len(breaks):.0f}%)")
        print(f"    Down breaks:   {down:>8,}  ({100*down/len(breaks):.0f}%)")

    print(f"  Est. pos range:  {df['pos_est'].min():.0f} – {df['pos_est'].max():.0f}")

    # PnL per-break vs per-non-break
    pb    = df.loc[df["is_break"],     "delta_pnl"]
    nonpb = df.loc[~df["is_break"],    "delta_pnl"]
    print(f"  Avg Δpnl/tick (break):     {pb.mean():>8.2f}")
    print(f"  Avg Δpnl/tick (non-break): {nonpb.mean():>8.2f}")


def print_missed(product: str, missed_buys: pd.DataFrame, missed_sells: pd.DataFrame):
    print(f"\n  {product} — Missed opportunities:")
    print(f"    Below-fair ticks not captured: {len(missed_buys):>5}")
    print(f"    Above-fair ticks not captured: {len(missed_sells):>5}")
    if len(missed_buys) > 0:
        print(f"    Avg deviation (missed buys):   {missed_buys['dev_mid'].mean():>8.2f} ticks")
    if len(missed_sells) > 0:
        print(f"    Avg deviation (missed sells):  {missed_sells['dev_mid'].mean():>8.2f} ticks")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pnl_trajectory(ash: pd.DataFrame, pep: pd.DataFrame,
                        log_id: str, total_pnl: float):
    """Cumulative PnL over time per product."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, df, label, color in [
        (axes[0], ash, ASH,    "steelblue"),
        (axes[1], pep, PEPPER, "darkorange"),
    ]:
        ax.plot(df["timestamp"], df["profit_and_loss"], color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_title(f"{label} — Cumulative PnL  (final: {df['profit_and_loss'].iloc[-1]:,.0f})")
        ax.set_ylabel("PnL")
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Log {log_id} — Total PnL {total_pnl:,.0f}", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / f"{log_id}_pnl.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_mid_and_fair(df: pd.DataFrame, product_label: str, log_id: str):
    """Mid price vs fair value, breaks highlighted, missed trades annotated."""
    short = "ASH" if "OSMIUM" in product_label else "PEPPER"
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    ax_mid, ax_dev, ax_pnl = axes

    # ── Price + fair ──────────────────────────────────────────────────────────
    ax_mid.plot(df["timestamp"], df["mid_price"], color="black",   lw=1, label="mid")
    ax_mid.plot(df["timestamp"], df["fair"],      color="crimson", lw=1, ls="--", label="fair")

    # Shade bid/ask band
    ax_mid.fill_between(df["timestamp"], df["bid_price_1"], df["ask_price_1"],
                        alpha=0.1, color="steelblue", label="bid-ask band")

    # Mark break ticks
    breaks = df[df["is_break"]]
    ax_mid.scatter(breaks["timestamp"], breaks["mid_price"],
                   color="red", s=20, zorder=5, label=f"break (|Δmid|≥{BREAK_THRESH})")

    ax_mid.set_title(f"{product_label} — Mid price vs Fair  ({log_id})")
    ax_mid.set_ylabel("Price")
    ax_mid.legend(fontsize=8, loc="upper left")
    ax_mid.grid(True, alpha=0.3)

    # ── Deviation from fair ───────────────────────────────────────────────────
    ax_dev.plot(df["timestamp"], df["dev_mid"], color="purple", lw=0.8)
    ax_dev.axhline(0, color="black", lw=0.5)
    ax_dev.fill_between(df["timestamp"], df["dev_mid"], 0,
                        where=df["dev_mid"] > 0, alpha=0.3, color="green", label="above fair")
    ax_dev.fill_between(df["timestamp"], df["dev_mid"], 0,
                        where=df["dev_mid"] < 0, alpha=0.3, color="red", label="below fair")
    ax_dev.set_ylabel("dev from fair")
    ax_dev.legend(fontsize=7, loc="upper left")
    ax_dev.grid(True, alpha=0.3)

    # ── Per-tick PnL delta ─────────────────────────────────────────────────────
    delta = df["delta_pnl"]
    ax_pnl.bar(df["timestamp"], delta,
               color=["green" if v > 0 else "red" for v in delta],
               width=90, alpha=0.7)
    ax_pnl.axhline(0, color="black", lw=0.5)
    ax_pnl.set_ylabel("Δ PnL / tick")
    ax_pnl.set_xlabel("Timestamp")
    ax_pnl.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"{log_id}_{short}_price.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_spread_and_position(df: pd.DataFrame, product_label: str, log_id: str):
    """Spread over time + estimated position."""
    short = "ASH" if "OSMIUM" in product_label else "PEPPER"
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax_sp, ax_pos = axes

    # Spread
    ax_sp.plot(df["timestamp"], df["spread"], color="teal", lw=0.8)
    ax_sp.axhline(df["spread"].median(), color="red", lw=1, ls="--",
                  label=f"median={df['spread'].median():.1f}")
    ax_sp.set_ylabel("Spread (ticks)")
    ax_sp.set_title(f"{product_label} — Spread & Estimated Position  ({log_id})")
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.3)

    # Position
    ax_pos.plot(df["timestamp"], df["pos_est"], color="navy", lw=1.2)
    ax_pos.axhline(0,  color="black", lw=0.5, ls="--")
    ax_pos.axhline(80, color="green", lw=0.8, ls=":", label="limit=80")
    ax_pos.axhline(-80, color="green", lw=0.8, ls=":")
    ax_pos.fill_between(df["timestamp"], df["pos_est"], 0,
                        where=df["pos_est"] > 0, alpha=0.2, color="green")
    ax_pos.fill_between(df["timestamp"], df["pos_est"], 0,
                        where=df["pos_est"] < 0, alpha=0.2, color="red")
    ax_pos.set_ylabel("Est. Position")
    ax_pos.set_xlabel("Timestamp")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"{log_id}_{short}_spread_pos.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def plot_whale_hits(ash: pd.DataFrame, log_id: str):
    """ASH break ticks aligned with PnL delta — whale hit visualization."""
    whales = detect_whale_hits(ash)
    if len(whales) == 0:
        print("  No whale hits detected (|Δpnl| > 50 at break ticks).")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax_mid, ax_dpnl = axes

    ax_mid.plot(ash["timestamp"], ash["mid_price"], color="black", lw=0.8, label="mid")
    ax_mid.plot(ash["timestamp"], ash["fair"],      color="crimson", lw=0.8, ls="--", label="fair")
    ax_mid.scatter(whales["timestamp"], whales["mid_price"],
                   color="orange", s=60, zorder=5, label="whale hit")
    ax_mid.set_title(f"ASH — Whale Hits (n={len(whales)})  [{log_id}]")
    ax_mid.set_ylabel("Price")
    ax_mid.legend(fontsize=8)
    ax_mid.grid(True, alpha=0.3)

    colors = ["green" if v > 0 else "red" for v in ash["delta_pnl"]]
    ax_dpnl.bar(ash["timestamp"], ash["delta_pnl"], color=colors, width=90, alpha=0.7)
    for _, row in whales.iterrows():
        ax_dpnl.axvline(row["timestamp"], color="orange", lw=1.5, alpha=0.8)
    ax_dpnl.axhline(0, color="black", lw=0.5)
    ax_dpnl.set_ylabel("Δ PnL / tick")
    ax_dpnl.set_xlabel("Timestamp")
    ax_dpnl.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"{log_id}_ASH_whales.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")

    # Print whale summary
    print(f"\n  Whale hit details ({len(whales)} hits):")
    print(f"  {'ts':>8}  {'Δmid':>7}  {'Δpnl':>10}  {'direction'}")
    for _, row in whales.iterrows():
        direction = "UP  " if row["delta_mid"] > 0 else "DOWN"
        print(f"  {int(row['timestamp']):>8}  {row['delta_mid']:>+7.1f}  "
              f"{row['delta_pnl']:>+10.2f}  {direction}")


def plot_pepper_build(pep: pd.DataFrame, log_id: str):
    """PEPPER position build-up vs fair value capture."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax2 = ax.twinx()
    ax.plot(pep["timestamp"], pep["mid_price"], color="darkorange", lw=1.2, label="mid price")
    ax.plot(pep["timestamp"], pep["fair"],      color="red",        lw=1, ls="--", label="fair trend")
    ax2.plot(pep["timestamp"], pep["pos_est"],  color="navy", lw=1.2, alpha=0.7, label="est. position")
    ax2.axhline(80, color="green", lw=0.8, ls=":", label="limit 80")

    ax.set_ylabel("Price", color="darkorange")
    ax2.set_ylabel("Est. Position", color="navy")
    ax.set_xlabel("Timestamp")
    ax.set_title(f"PEPPER — Price vs Position Build  [{log_id}]")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"{log_id}_PEPPER_build.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def analyze_log(path: Path):
    log_id = path.stem
    print(f"\n{'═'*60}")
    print(f"  Log: {path}")
    print(f"{'═'*60}")

    log      = load_log(path)
    df       = parse_activities(log)
    ash, pep = split_products(df)

    total_pnl = log.get("profit", 0.0)
    print(f"  Total PnL: {total_pnl:,.2f}")
    print(f"  Final positions: {log.get('positions', [])}")

    ash = add_derived(ash, fair_const=ASH_FAIR)
    pep = add_derived(pep, fair_const=None)

    ash_final_pnl = ash["profit_and_loss"].iloc[-1]
    pep_final_pnl = pep["profit_and_loss"].iloc[-1]

    print_stats(ASH,    ash, ash_final_pnl)
    print_stats(PEPPER, pep, pep_final_pnl)

    # Missed trade analysis
    ash_miss_buy  = detect_missed_buys(ash)
    ash_miss_sell = detect_missed_sells(ash)
    pep_miss_buy  = detect_missed_buys(pep, pos_limit=80)
    print_missed(ASH,    ash_miss_buy, ash_miss_sell)
    print_missed(PEPPER, pep_miss_buy, pd.DataFrame())

    # Mid price discrepancy analysis
    print(f"\n  Mid price discrepancy analysis (ASH):")
    print(f"    Ticks where mid ≠ fair (10000): {(ash['mid_price'] != ASH_FAIR).sum()}")
    disc = ash[ash["mid_price"] != ASH_FAIR]
    if len(disc) > 0:
        print(f"    Avg deviation:  {disc['dev_mid'].mean():>8.2f}")
        print(f"    Std deviation:  {disc['dev_mid'].std():>8.2f}")
        print(f"    Max pos dev:    {disc['dev_mid'].max():>8.2f}")
        print(f"    Max neg dev:    {disc['dev_mid'].min():>8.2f}")
        # Classify: break ticks vs non-break
        disc_break    = disc[disc["is_break"]]
        disc_nonbreak = disc[~disc["is_break"]]
        print(f"    Devs at break ticks:     {len(disc_break)} "
              f"  (avg={disc_break['dev_mid'].mean():.2f})")
        print(f"    Devs at non-break ticks: {len(disc_nonbreak)} "
              f"  (avg={disc_nonbreak['dev_mid'].mean():.2f})")
        print(f"    → Most mid discrepancy is {'at breaks (whale/swarm)' if len(disc_break) > len(disc_nonbreak)/3 else 'between breaks (quote reshuffling)'}")

    # Plots
    print(f"\n  Generating plots → {PLOTS_DIR}/")
    plot_pnl_trajectory(ash, pep, log_id, total_pnl)
    plot_mid_and_fair(ash, ASH,    log_id)
    plot_mid_and_fair(pep, PEPPER, log_id)
    plot_spread_and_position(ash, ASH,    log_id)
    plot_spread_and_position(pep, PEPPER, log_id)
    plot_whale_hits(ash, log_id)
    plot_pepper_build(pep, log_id)

    return {
        "log_id": log_id,
        "total_pnl": total_pnl,
        "ash_pnl": ash_final_pnl,
        "pepper_pnl": pep_final_pnl,
        "ash_breaks": int(ash["is_break"].sum()),
        "ash_missed_buys": len(ash_miss_buy),
        "ash_missed_sells": len(ash_miss_sell),
        "pepper_missed_buys": len(pep_miss_buy),
    }


def compare_logs(results: list):
    if len(results) < 2:
        return
    print(f"\n{'═'*60}")
    print("  Comparison across logs")
    print(f"{'═'*60}")
    print(f"  {'log_id':>10}  {'total_pnl':>12}  {'ash_pnl':>10}  "
          f"{'pepper_pnl':>12}  {'breaks':>7}  {'miss_buy':>9}  {'miss_sell':>10}")
    for r in results:
        print(f"  {r['log_id']:>10}  {r['total_pnl']:>12,.0f}  {r['ash_pnl']:>10,.0f}  "
              f"{r['pepper_pnl']:>12,.0f}  {r['ash_breaks']:>7}  "
              f"{r['ash_missed_buys']:>9}  {r['ash_missed_sells']:>10}")


def main():
    parser = argparse.ArgumentParser(description="Round 2 log analysis")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to a specific .json log (default: all in logs/)")
    args = parser.parse_args()

    if args.log:
        paths = [Path(args.log)]
    else:
        paths = sorted(LOGS_DIR.rglob("*.json"))
        if not paths:
            print(f"No .json logs found in {LOGS_DIR}/ — pass --log <path>")
            return

    print(f"Analyzing {len(paths)} log(s)...")
    results = [analyze_log(p) for p in paths]
    compare_logs(results)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
