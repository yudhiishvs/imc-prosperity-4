"""
ASH COATED OSMIUM Market-Making Analysis — IMC Prosperity 4
============================================================
Reads all day data from data/round_1/ and produces:
  1. Mid-price histogram & frequency table
  2. Spread analysis (distribution, stats)
  3. Bid/ask size descriptive statistics (1-var)
  4. Autocorrelation of mid-price at lags 1, 2, 4, 9, 16, 25
  5. Order-book depth profile (volume at each price level)
  6. Trade-price distribution (from trades CSVs)
  7. Optimal quote-placement analysis
  8. Mean-reversion half-life estimate
  9. Bid/Ask price-level frequency
 10. Time-series plot
"""

import os, sys, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ── paths ──────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "round_1"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "osmium_analysis"
FIGS   = OUT_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

PRICE_FILES = sorted(DATA.glob("prices_round_1_day_*.csv"))
TRADE_FILES = sorted(DATA.glob("trades_round_1_day_*.csv"))

# ── styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "font.family":      "monospace",
    "font.size":        10,
})
ACCENT  = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"
ACCENT4 = "#d2a8ff"


# =====================================================================
#  DATA LOADING
# =====================================================================
def load_prices() -> pd.DataFrame:
    """Load all price snapshot CSVs, filter to ASH_COATED_OSMIUM."""
    frames = []
    for fp in PRICE_FILES:
        try:
            df = pd.read_csv(fp, sep=";")
            frames.append(df)
        except Exception as e:
            print(f"Error reading {fp}: {e}")
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    df_filtered = all_df[all_df["product"] == "ASH_COATED_OSMIUM"].copy()
    df_filtered.sort_values(["day", "timestamp"], inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    # coerce numerics
    for c in ["bid_price_1","bid_volume_1","bid_price_2","bid_volume_2",
              "bid_price_3","bid_volume_3",
              "ask_price_1","ask_volume_1","ask_price_2","ask_volume_2",
              "ask_price_3","ask_volume_3","mid_price","profit_and_loss"]:
        if c in df_filtered.columns:
            df_filtered[c] = pd.to_numeric(df_filtered[c], errors="coerce")
    return df_filtered


def load_trades() -> pd.DataFrame:
    """Load all trade CSVs, filter to ASH_COATED_OSMIUM."""
    frames = []
    for fp in TRADE_FILES:
        try:
            df = pd.read_csv(fp, sep=";")
            frames.append(df)
        except Exception as e:
            print(f"Error reading {fp}: {e}")
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    df_filtered = all_df[all_df["symbol"] == "ASH_COATED_OSMIUM"].copy()
    df_filtered["price"] = pd.to_numeric(df_filtered["price"], errors="coerce")
    df_filtered["quantity"] = pd.to_numeric(df_filtered["quantity"], errors="coerce")
    df_filtered.sort_values(["timestamp"], inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered


# =====================================================================
#  ANALYSIS FUNCTIONS
# =====================================================================
def print_header(title: str):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def one_var_stats(series: pd.Series, label: str):
    """Print a 1-variable statistics summary."""
    s = series.dropna()
    print(f"\n  ── {label} (n={len(s)}) ──")
    if len(s) == 0:
        return
    print(f"    Mean        : {s.mean():.4f}")
    print(f"    Median      : {s.median():.4f}")
    print(f"    Std Dev     : {s.std():.4f}")
    print(f"    Min         : {s.min():.4f}")
    print(f"    Max         : {s.max():.4f}")


# ── 1. Mid-price histogram & frequency ────────────────────────────────
def analysis_mid_price(df: pd.DataFrame):
    print_header("1. MID-PRICE FREQUENCY TABLE")
    mid = df["mid_price"].dropna()
    freq = mid.value_counts().sort_index()

    one_var_stats(mid, "Mid-Price")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(freq.index.astype(str), freq.values, color=ACCENT, edgecolor="#0d1117", linewidth=0.5)
    ax.set_xlabel("Mid-Price")
    ax.set_ylabel("Frequency")
    ax.set_title("ASH_COATED_OSMIUM — Mid-Price Frequency Distribution", fontsize=13, fontweight="bold")
    if len(freq) > 40:
        ax.set_xticks([])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "01_mid_price_histogram.png", dpi=150)
    plt.close(fig)
    print(f"\n  [saved] {FIGS / '01_mid_price_histogram.png'}")


# ── 2. Spread analysis ────────────────────────────────────────────────
def analysis_spread(df: pd.DataFrame):
    print_header("2. SPREAD ANALYSIS")
    spread = df["ask_price_1"] - df["bid_price_1"]
    spread = spread.dropna()
    one_var_stats(spread, "Best Bid–Ask Spread")

    freq = spread.value_counts().sort_index()
    print("\n  Spread | Count  | Pct")
    print("  -------|--------|--------")
    for s, cnt in freq.items():
        print(f"  {s:>6.0f} | {cnt:>6d} | {cnt / len(spread) * 100:>5.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(freq.index.astype(int).astype(str), freq.values, color=ACCENT2, edgecolor="#0d1117")
    ax.set_xlabel("Spread (ask₁ − bid₁)")
    ax.set_ylabel("Frequency")
    ax.set_title("ASH_COATED_OSMIUM — Bid-Ask Spread Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "02_spread_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  [saved] {FIGS / '02_spread_distribution.png'}")


# ── 3. Bid / Ask size statistics ──────────────────────────────────────
def analysis_bid_ask_sizes(df: pd.DataFrame):
    print_header("3. BID / ASK SIZE — 1-Var Statistics")
    one_var_stats(df["bid_volume_1"], "Bid Volume Level 1")
    one_var_stats(df["ask_volume_1"], "Ask Volume Level 1")
    one_var_stats(df["bid_volume_2"], "Bid Volume Level 2")
    one_var_stats(df["ask_volume_2"], "Ask Volume Level 2")


# ── 4. Autocorrelation ────────────────────────────────────────────────
def analysis_autocorrelation(df: pd.DataFrame):
    print_header("4. AUTOCORRELATION OF MID-PRICE")
    lags = [1, 2, 4, 9, 16, 25]
    mid = df["mid_price"].dropna().values
    returns = np.diff(mid)

    print("\n  A) Raw Mid-Price Autocorrelation")
    ac_mid_vals = []
    for lag in lags:
        if lag < len(mid):
            ac = np.corrcoef(mid[:-lag], mid[lag:])[0, 1]
        else:
            ac = np.nan
        ac_mid_vals.append(ac)
        print(f"  {lag:>3d} | {ac:>+.6f}")

    print("\n  B) Returns (Δmid) Autocorrelation")
    ac_ret_vals = []
    for lag in lags:
        if lag < len(returns):
            ac = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
        else:
            ac = np.nan
        ac_ret_vals.append(ac)
        print(f"  {lag:>3d} | {ac:>+.6f}")


# ── 5. Order-book depth profile ───────────────────────────────────────
def analysis_depth_profile(df: pd.DataFrame):
    print_header("5. ORDER-BOOK DEPTH PROFILE (Price-Level Heatmap)")

    bid_prices = {}
    ask_prices = {}
    for _, row in df.iterrows():
        for lvl in [1, 2, 3]:
            bp = row.get(f"bid_price_{lvl}")
            bv = row.get(f"bid_volume_{lvl}")
            if pd.notna(bp) and pd.notna(bv):
                bp = int(bp)
                bid_prices[bp] = bid_prices.get(bp, 0) + bv

            ap = row.get(f"ask_price_{lvl}")
            av = row.get(f"ask_volume_{lvl}")
            if pd.notna(ap) and pd.notna(av):
                ap = int(ap)
                ask_prices[ap] = ask_prices.get(ap, 0) + av

    print("\n  Top 5 BID levels by aggregated volume:")
    for p in sorted(bid_prices.keys(), key=lambda k: bid_prices[k], reverse=True)[:5]:
        print(f"  {p:>6d} | {bid_prices[p]:>10.0f}")

    print("\n  Top 5 ASK levels by aggregated volume:")
    for p in sorted(ask_prices.keys(), key=lambda k: ask_prices[k], reverse=True)[:5]:
        print(f"  {p:>6d} | {ask_prices[p]:>10.0f}")


# ── 6. Trade-price distribution ───────────────────────────────────────
def analysis_trades(trades: pd.DataFrame):
    print_header("6. TRADE-PRICE DISTRIBUTION")
    if trades.empty:
        print("  No ASH_COATED_OSMIUM trades found.")
        return

    vwap = (trades["price"] * trades["quantity"]).sum() / trades["quantity"].sum()
    print(f"\n  VWAP: {vwap:.4f}")
    print(f"  Total volume traded: {trades['quantity'].sum():.0f}")
    print(f"  Total trades: {len(trades)}")


# ── 7. Optimal quote placement analysis ──────────────────────────────
def analysis_optimal_quoting(df: pd.DataFrame, trades: pd.DataFrame):
    print_header("7. OPTIMAL QUOTE-PLACEMENT ANALYSIS")

    fair = round(df["mid_price"].mean())
    print(f"  Assuming proxy FAIR VALUE = {fair}")
    mid = df["mid_price"].dropna()

    at_fair = (mid == fair).sum()
    print(f"  Mid-price == {fair}:  {at_fair}/{len(mid)} ({at_fair/len(mid)*100:.1f}%)")

    print("\n  Quote-Offset Fill-Rate Simulation (vs bot order book)")
    print("  Offset | Bid Fill-Rate | Ask Fill-Rate")
    print("  -------|---------------|---------------")

    offsets = list(range(1, 15))
    for offset in offsets:
        bid_price = fair - offset
        bid_fills = df[df["ask_price_1"] <= bid_price]
        bid_fill_rate = len(bid_fills) / len(df) * 100

        ask_price = fair + offset
        ask_fills = df[df["bid_price_1"] >= ask_price]
        ask_fill_rate = len(ask_fills) / len(df) * 100

        print(f"  {offset:>6d} | {bid_fill_rate:>12.2f}% | {ask_fill_rate:>12.2f}%")


# ── 8. Mean-reversion half-life ───────────────────────────────────────
def analysis_mean_reversion(df: pd.DataFrame):
    print_header("8. MEAN-REVERSION HALF-LIFE (OLS on Δmid ~ mid_lag)")
    mid = df["mid_price"].dropna().values.astype(float)
    fair = mid.mean()
    deviation = mid - fair

    if len(deviation) < 3:
        return

    y = np.diff(deviation)  # Δdeviation
    x = deviation[:-1]      # lagged deviation

    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    print(f"\n  OLS: Δ(mid−fair) = {intercept:.6f} + {slope:.6f} × (mid−fair)_lag")
    
    if slope < 0:
        half_life = -np.log(2) / np.log(1 + slope)
        print(f"  Half-life: {half_life:.2f} ticks")
    else:
        print("  Slope >= 0 → no mean-reversion detected (possibly trending or pure noise)")


# ── 10. Time-series plot ──────────────────────────────────────────────
def analysis_timeseries(df: pd.DataFrame):
    print_header("10. MID-PRICE TIME SERIES")
    day1 = df[df["day"] == df["day"].iloc[0]].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Mid-price
    axes[0].plot(day1["timestamp"], day1["mid_price"], lw=0.8, color=ACCENT)
    axes[0].set_ylabel("Mid-Price")
    axes[0].set_title("ASH_COATED_OSMIUM — Price & Book Dynamics (Day 1)", fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.2)

    # Spread
    spread = day1["ask_price_1"] - day1["bid_price_1"]
    axes[1].plot(day1["timestamp"], spread, lw=0.8, color=ACCENT3)
    axes[1].set_ylabel("Spread")
    axes[1].set_title("Bid-Ask Spread", fontsize=11)
    axes[1].grid(alpha=0.2)

    # Bid/Ask L1 volume
    axes[2].fill_between(day1["timestamp"], day1["bid_volume_1"], alpha=0.5, color=ACCENT3, label="Bid Vol L1")
    axes[2].fill_between(day1["timestamp"], -day1["ask_volume_1"], alpha=0.5, color=ACCENT2, label="Ask Vol L1")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Volume")
    axes[2].set_title("Level-1 Volume", fontsize=11)
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGS / "10_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"\n  [saved] {FIGS / '10_timeseries.png'}")


# =====================================================================
#  MAIN
# =====================================================================
def main():
    print("\n" + "█" * 72)
    print("  ASH_COATED_OSMIUM MARKET-MAKING ANALYSIS")
    print("█" * 72)

    df = load_prices()
    trades = load_trades()
    print(f"\n  Loaded {len(df)} price snapshots across {df['day'].nunique()} day(s)")
    print(f"  Loaded {len(trades)} trades")

    analysis_mid_price(df)
    analysis_spread(df)
    analysis_bid_ask_sizes(df)
    analysis_autocorrelation(df)
    analysis_depth_profile(df)
    analysis_trades(trades)
    analysis_optimal_quoting(df, trades)
    analysis_mean_reversion(df)
    analysis_timeseries(df)

if __name__ == "__main__":
    main()
