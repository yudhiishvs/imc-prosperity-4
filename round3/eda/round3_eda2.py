"""
round3_eda2.py — IMC Prosperity 4 Round 3  (second-pass EDA)
=============================================================
Focus areas driven by Orin's clues:
  A. IV smile structure, moneyness mapping, deviation detection
  B. Size-15 "insider" order-book bot — identification, timing, predictive edge
  C. Reference-point / consensus analysis for VF Extract fair value
  D. Sizing guidance: deviation magnitude → position conviction map

Outputs (eda_output/):
  23_iv_moneyness_scatter.png   IV vs log-moneyness, parabola fit, deviation highlights
  24_iv_smile_residuals.png     Residual δIV per strike per day — who is mis-priced
  25_insider_size15_hydrogel.png  Size-15 orders in HG book + price timeseries
  26_insider_size15_vf.png      Size-15 orders in VF book + VF price
  27_insider_predictive.png     Size-15 bid/ask side vs next-100-tick VF return
  28_reference_point.png        VF Extract rolling fair value + cluster analysis
  29_deviation_sizing.png       |δIV| → suggested position size (conviction map)
  30_insider_option_book.png    Size-15 presence in VEV_4000 / VEV_6500 order books

All plots are saved to round3/eda/eda_output/
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from scipy.stats import norm, linregress
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "ROUND_3"
OUT_DIR  = Path(__file__).parent / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [0, 1, 2]
TTE_START = {0: 7.0, 1: 6.0, 2: 5.0}   # historical data TTE mapping

STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
STRIKE_SYMS  = {k: f"VEV_{k}" for k in STRIKES}
VF_PRODUCT   = "VELVETFRUIT_EXTRACT"
HG_PRODUCT   = "HYDROGEL_PACK"
PLOT_IDX = 23   # starting plot number

# ── Colour palette ─────────────────────────────────────────────────────────────
DAY_COLORS   = ["#2196F3", "#FF9800", "#4CAF50"]
INSIDER_COLOR = "#E91E63"   # hot pink = insider
REGULAR_COLOR = "#B0BEC5"

# ── Helpers ────────────────────────────────────────────────────────────────────

def bs_call(S, K, T, sigma, r=0.0):
    """European call.  T in years."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

def bs_iv(C, S, K, T, r=0.0):
    """Implied vol via Brent.  Returns NaN on failure."""
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-3 or T <= 1e-8 or S <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, s, r) - C, 1e-6, 10.0, xtol=1e-7, maxiter=200)
    except Exception:
        return np.nan

def load_prices():
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"prices_round_3_day_{d}.csv", sep=";")
        df["data_day"] = d
        df["global_ts"] = d * 1_000_000 + df["timestamp"]
        df["TTE"] = TTE_START[d] - df["timestamp"] / 1_000_000
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_trades():
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA_DIR / f"trades_round_3_day_{d}.csv", sep=";")
        df["data_day"] = d
        df["global_ts"] = d * 1_000_000 + df["timestamp"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

print("Loading data …")
prices = load_prices()
trades = load_trades()

# ── Underlying VF Extract price ─────────────────────────────────────────────────
px_vf  = prices[prices["product"] == VF_PRODUCT].copy()
px_hg  = prices[prices["product"] == HG_PRODUCT].copy()

# ── IV computation ──────────────────────────────────────────────────────────────
print("Computing IV …")

px_vev = prices[prices["product"].str.startswith("VEV_")].copy()
px_vev["strike"] = px_vev["product"].str.replace("VEV_", "").astype(int)

# Merge underlying S onto each VEV row by (data_day, timestamp)
vf_ts = px_vf[["data_day", "timestamp", "mid_price"]].rename(columns={"mid_price": "S"})
px_vev = px_vev.merge(vf_ts, on=["data_day", "timestamp"], how="left")

def safe_iv(row):
    S = row["S"]
    K = row["strike"]
    T = max(row["TTE"], 1e-4) / 252.0
    C = row["mid_price"]
    return bs_iv(C, S, K, T)

# Sample for speed (every 10th tick)
sample = px_vev.iloc[::10].copy()
sample["iv"] = sample.apply(safe_iv, axis=1)
sample["log_moneyness"] = np.log(sample["S"] / sample["strike"])

iv_data = sample.dropna(subset=["iv"]).copy()
iv_data = iv_data[(iv_data["iv"] > 0.01) & (iv_data["iv"] < 2.0)]

print(f"  {len(iv_data)} valid IV observations")

# ── Parabola fit per day ────────────────────────────────────────────────────────
def fit_parabola(df):
    x = df["log_moneyness"].values
    y = df["iv"].values
    coeffs = np.polyfit(x, y, 2)
    return coeffs

fit_results = {}
for d in DAYS:
    sub = iv_data[iv_data["data_day"] == d]
    if len(sub) > 20:
        fit_results[d] = fit_parabola(sub)

# Residuals: IV - smile_fit(log_moneyness)
def smile_residual(row, coeffs):
    x = row["log_moneyness"]
    fitted = np.polyval(coeffs, x)
    return row["iv"] - fitted

for d in DAYS:
    if d in fit_results:
        mask = iv_data["data_day"] == d
        iv_data.loc[mask, "iv_fitted"] = np.polyval(fit_results[d], iv_data.loc[mask, "log_moneyness"])
        iv_data.loc[mask, "iv_resid"] = iv_data.loc[mask, "iv"] - iv_data.loc[mask, "iv_fitted"]

# Per-strike median residual
strike_resid = iv_data.groupby(["data_day", "strike"])["iv_resid"].agg(["median", "std"]).reset_index()
strike_resid.columns = ["data_day", "strike", "median_resid", "std_resid"]

# ── Size-15 insider detection ───────────────────────────────────────────────────
print("Detecting size-15 insider …")

# For each product, find ticks where bid_volume_1 == 15 OR ask_volume_1 == 15
def get_insider_ticks(product):
    sub = prices[prices["product"] == product].copy()
    sub["b1v"] = pd.to_numeric(sub.get("bid_volume_1", np.nan), errors="coerce")
    sub["a1v"] = pd.to_numeric(sub.get("ask_volume_1", np.nan), errors="coerce")
    sub["b1p"] = pd.to_numeric(sub.get("bid_price_1", np.nan), errors="coerce")
    sub["a1p"] = pd.to_numeric(sub.get("ask_price_1", np.nan), errors="coerce")
    sub["insider_bid"] = sub["b1v"] == 15
    sub["insider_ask"] = sub["a1v"] == 15
    sub["insider"]     = sub["insider_bid"] | sub["insider_ask"]
    return sub

hg_insider = get_insider_ticks(HG_PRODUCT)
vf_insider = get_insider_ticks(VF_PRODUCT)
vev4000_insider = get_insider_ticks("VEV_4000")
vev6500_insider = get_insider_ticks("VEV_6500")

# When insider is on BID side: they want to buy → bullish signal
# When insider is on ASK side: they want to sell → bearish signal
# "Insider" = they are adding size-15 at L1 bid or ask

print(f"  HG insider ticks: {hg_insider['insider'].sum()} / {len(hg_insider)}")
print(f"  VF insider ticks: {vf_insider['insider'].sum()} / {len(vf_insider)}")

# ── Predictive analysis: insider bid/ask vs future return ──────────────────────
print("Running predictive analysis …")

LOOKAHEAD = 100   # ticks

def compute_future_returns(df, lookahead=100):
    """For each tick, compute mid_price change over next `lookahead` ticks."""
    df = df.sort_values(["data_day", "timestamp"]).reset_index(drop=True)
    df["future_mid"] = df["mid_price"].shift(-lookahead)
    df["future_ret"]  = df["future_mid"] - df["mid_price"]
    return df

vf_for_pred = compute_future_returns(vf_insider.copy())

# Insider on bid (bullish) vs on ask (bearish) vs neither
vf_for_pred["signal"] = "none"
vf_for_pred.loc[vf_for_pred["insider_bid"], "signal"] = "insider_bid"
vf_for_pred.loc[vf_for_pred["insider_ask"], "signal"] = "insider_ask"

pred_stats = vf_for_pred.dropna(subset=["future_ret"]).groupby("signal")["future_ret"].agg(
    ["mean", "median", "std", "count"]
).reset_index()
print("  Predictive stats (VF future return by signal):")
print(pred_stats.to_string(index=False))

# ── Reference point analysis ────────────────────────────────────────────────────
print("Reference point analysis …")

# Rolling mean of VF Extract trades
vf_trades = trades[trades["symbol"] == VF_PRODUCT].copy()
vf_trades = vf_trades.sort_values("global_ts")
vf_trades["rolling_mean"] = vf_trades["price"].expanding().mean()

# VF Extract mid price rolling stats
px_vf_sorted = px_vf.sort_values("global_ts")
px_vf_sorted["roll_mean_200"]  = px_vf_sorted["mid_price"].rolling(200, min_periods=50).mean()
px_vf_sorted["roll_mean_1000"] = px_vf_sorted["mid_price"].rolling(1000, min_periods=200).mean()
px_vf_sorted["deviation_from_mean"] = px_vf_sorted["mid_price"] - px_vf_sorted["roll_mean_200"]

# Cluster: distribution of VF prices around means
vf_dev_all = px_vf_sorted["deviation_from_mean"].dropna()

print(f"  VF deviation from 200-MA: mean={vf_dev_all.mean():.2f}, std={vf_dev_all.std():.2f}")

# ── HYDROGEL insider price-level analysis ──────────────────────────────────────
# What price does the size-15 insider sit at when they appear?
hg_insider_bid = hg_insider[hg_insider["insider_bid"]]
hg_insider_ask = hg_insider[hg_insider["insider_ask"]]

print(f"\n  HG insider BID prices: {hg_insider_bid['b1p'].describe().to_dict()}")
print(f"  HG insider ASK prices: {hg_insider_ask['a1p'].describe().to_dict()}")

# ── VF insider: what price is size-15 on? ─────────────────────────────────────
vf_insider_bid = vf_insider[vf_insider["insider_bid"]]
vf_insider_ask = vf_insider[vf_insider["insider_ask"]]

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating plots …")

# ── Plot 23: IV vs Log-Moneyness ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle("Plot 23 — IV vs Log-Moneyness with Parabola Smile Fit", fontsize=14, fontweight="bold")

lm_grid = np.linspace(-0.35, 0.35, 200)
for i, d in enumerate(DAYS):
    ax = axes[i]
    sub = iv_data[iv_data["data_day"] == d]

    # Scatter by strike (colour)
    for k in STRIKES:
        kd = sub[sub["strike"] == k]
        if len(kd) == 0:
            continue
        ax.scatter(kd["log_moneyness"], kd["iv"], s=8, alpha=0.4, label=str(k))

    # Parabola fit
    if d in fit_results:
        fitted = np.polyval(fit_results[d], lm_grid)
        ax.plot(lm_grid, fitted, "k--", lw=2, label="parabola fit")
        # 2σ bands
        resid_std = iv_data.loc[iv_data["data_day"] == d, "iv_resid"].std()
        ax.fill_between(lm_grid, fitted - 2*resid_std, fitted + 2*resid_std,
                        alpha=0.12, color="gray", label="±2σ band")

    ax.set_xlabel("log(S/K) = log-moneyness")
    ax.set_title(f"Day {d} (TTE~{TTE_START[d]-0.5:.1f}d)")
    ax.set_ylim(0.0, 0.9)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.8, ls=":")

axes[0].set_ylabel("Implied Volatility (σ)")
axes[2].legend(title="Strike", fontsize=7, loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig(OUT_DIR / "23_iv_moneyness_scatter.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 23_iv_moneyness_scatter.png")

# ── Plot 24: IV Smile Residuals ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Plot 24 — IV Residuals from Parabola Smile (δIV = market IV − fitted IV)\n"
             "Positive = market overprices volatility → SELL; Negative = underprice → BUY",
             fontsize=12, fontweight="bold")

for i, d in enumerate(DAYS):
    ax = axes[i]
    sub = strike_resid[strike_resid["data_day"] == d]

    bar_colors = ["#EF5350" if v > 0 else "#42A5F5" for v in sub["median_resid"]]
    bars = ax.bar(sub["strike"], sub["median_resid"], color=bar_colors,
                  edgecolor="black", alpha=0.8, width=40)
    ax.errorbar(sub["strike"], sub["median_resid"], yerr=sub["std_resid"],
                fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_title(f"Day {d}")
    ax.set_xlabel("Strike")
    ax.set_xticks(sub["strike"])
    ax.set_xticklabels(sub["strike"], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate direction
    for _, row in sub.iterrows():
        direction = "SELL↑" if row["median_resid"] > 0.01 else ("BUY↓" if row["median_resid"] < -0.01 else "")
        if direction:
            ax.annotate(direction, (row["strike"], row["median_resid"]),
                       xytext=(0, 8 if row["median_resid"] > 0 else -12),
                       textcoords="offset points", ha="center", fontsize=7, color="black")

axes[0].set_ylabel("Median δIV (market − fitted)")
plt.tight_layout()
plt.savefig(OUT_DIR / "24_iv_smile_residuals.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 24_iv_smile_residuals.png")

# ── Plot 25: Size-15 Insider in HYDROGEL ──────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle("Plot 25 — Size-15 'Insider' Trader in HYDROGEL_PACK\n"
             "Pink dots = ticks where size-15 order appears at best bid or ask",
             fontsize=13, fontweight="bold")

for i, d in enumerate(DAYS):
    ax = axes[i]
    hg_d = hg_insider[hg_insider["data_day"] == d]

    ts = hg_d["timestamp"].values
    mid = hg_d["mid_price"].values

    ax.plot(ts, mid, color="#1565C0", lw=0.8, alpha=0.7, label="Mid price")

    # Insider bid (bullish): they want to buy
    ins_bid = hg_d[hg_d["insider_bid"]]
    ax.scatter(ins_bid["timestamp"], ins_bid["b1p"], s=15, color=INSIDER_COLOR,
               alpha=0.6, zorder=5, label="Insider BID (size-15 at best bid)", marker="^")

    # Insider ask (bearish): they want to sell
    ins_ask = hg_d[hg_d["insider_ask"]]
    ax.scatter(ins_ask["timestamp"], ins_ask["a1p"], s=15, color="#FF6F00",
               alpha=0.6, zorder=5, label="Insider ASK (size-15 at best ask)", marker="v")

    ax.set_title(f"Day {d} — HG insider: {hg_d['insider'].sum()}/{len(hg_d)} ticks "
                 f"({hg_d['insider'].mean()*100:.0f}%)")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestamp")
plt.tight_layout()
plt.savefig(OUT_DIR / "25_insider_size15_hydrogel.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 25_insider_size15_hydrogel.png")

# ── Plot 26: Size-15 Insider in VELVETFRUIT_EXTRACT ───────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle("Plot 26 — Size-15 'Insider' Trader in VELVETFRUIT_EXTRACT\n"
             "When insider bids (wants to buy), does price subsequently rise?",
             fontsize=13, fontweight="bold")

for i, d in enumerate(DAYS):
    ax = axes[i]
    vf_d = vf_insider[vf_insider["data_day"] == d]

    ax.plot(vf_d["timestamp"], vf_d["mid_price"], color="#1565C0", lw=0.8, alpha=0.7)

    # Insider bid ticks
    ins_bid = vf_d[vf_d["insider_bid"]]
    ax.scatter(ins_bid["timestamp"], ins_bid["b1p"], s=20, color=INSIDER_COLOR,
               alpha=0.7, zorder=5, marker="^", label=f"Insider BID (n={len(ins_bid)})")

    # Insider ask ticks
    ins_ask = vf_d[vf_d["insider_ask"]]
    ax.scatter(ins_ask["timestamp"], ins_ask["a1p"], s=20, color="#FF6F00",
               alpha=0.7, zorder=5, marker="v", label=f"Insider ASK (n={len(ins_ask)})")

    # Overlay VF trades
    vf_tr_d = vf_trades[vf_trades["data_day"] == d]
    ax.scatter(vf_tr_d["timestamp"], vf_tr_d["price"], s=12, color="#4CAF50",
               alpha=0.5, zorder=3, marker="x", label="Market trades")

    ax.set_title(f"Day {d}")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestamp")
plt.tight_layout()
plt.savefig(OUT_DIR / "26_insider_size15_vf.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 26_insider_size15_vf.png")

# ── Plot 27: Insider Predictive Power ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)
fig.suptitle("Plot 27 — Insider Size-15 Predictive Edge\n"
             "Does the insider's quote side predict short-term price direction?",
             fontsize=13, fontweight="bold")

# Box plot: future return by signal
ax1 = fig.add_subplot(gs[0, 0])
signal_groups = vf_for_pred.dropna(subset=["future_ret"]).groupby("signal")["future_ret"]
box_data = [signal_groups.get_group(g).values for g in ["insider_bid", "insider_ask", "none"]
            if g in signal_groups.groups]
box_labels = [g for g in ["insider_bid", "insider_ask", "none"] if g in signal_groups.groups]
bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True,
                 flierprops=dict(marker=".", markersize=2))
colors_box = [INSIDER_COLOR, "#FF6F00", "#B0BEC5"][:len(box_data)]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.axhline(0, color="black", lw=1.2, ls="--")
ax1.set_title(f"VF Future Return ({LOOKAHEAD}-tick)\nby Insider Signal")
ax1.set_ylabel("Price change")
ax1.grid(True, alpha=0.3, axis="y")

# Means table
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
pred_table = vf_for_pred.dropna(subset=["future_ret"]).groupby("signal")["future_ret"].agg(
    ["mean", "median", "std", "count"]
).round(3).reset_index()
table_data = [pred_table.columns.tolist()] + pred_table.values.tolist()
table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax2.set_title("Summary Statistics")

# Cumulative: insider bid - follow the signal
ax3 = fig.add_subplot(gs[1, :])
vf_sorted = vf_for_pred.sort_values(["data_day", "timestamp"]).copy()
# Simulate: when insider bids, buy 1 unit; when insider asks, sell 1 unit
vf_sorted["sim_position"] = 0
vf_sorted.loc[vf_sorted["signal"] == "insider_bid", "sim_position"] = 1
vf_sorted.loc[vf_sorted["signal"] == "insider_ask", "sim_position"] = -1
vf_sorted["sim_pnl_delta"] = vf_sorted["sim_position"] * vf_sorted["future_ret"].fillna(0)
vf_sorted["cum_pnl"] = vf_sorted["sim_pnl_delta"].cumsum()

ax3.plot(vf_sorted["global_ts"], vf_sorted["cum_pnl"], color=INSIDER_COLOR, lw=1.5)
ax3.axhline(0, color="black", lw=1, ls="--")
ax3.fill_between(vf_sorted["global_ts"], 0, vf_sorted["cum_pnl"],
                 where=vf_sorted["cum_pnl"] > 0, alpha=0.3, color="#4CAF50", label="Profitable")
ax3.fill_between(vf_sorted["global_ts"], 0, vf_sorted["cum_pnl"],
                 where=vf_sorted["cum_pnl"] < 0, alpha=0.3, color="#EF5350", label="Loss")
ax3.set_title(f"Cumulative PnL if following insider signal (1 unit × {LOOKAHEAD}-tick return)")
ax3.set_xlabel("Global timestamp")
ax3.set_ylabel("Cumulative return")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Add day separators
for d in [1, 2]:
    ax3.axvline(d * 1_000_000, color="gray", ls=":", lw=1, label=f"Day {d}")

plt.tight_layout()
plt.savefig(OUT_DIR / "27_insider_predictive.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 27_insider_predictive.png")

# ── Plot 28: Reference Point Analysis ─────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle("Plot 28 — VF Extract Reference Point & Consensus Analysis\n"
             "'Means to an End': where do traders cluster? What is the implied benchmark?",
             fontsize=13, fontweight="bold")

for i, d in enumerate(DAYS):
    # Left: price + rolling means + insider
    ax_left = axes[i, 0]
    vf_d  = px_vf_sorted[px_vf_sorted["data_day"] == d]

    ax_left.plot(vf_d["timestamp"], vf_d["mid_price"], color="#1565C0", lw=0.8, alpha=0.7, label="Mid")
    ax_left.plot(vf_d["timestamp"], vf_d["roll_mean_200"].ffill(),
                 color="#FF5722", lw=1.5, ls="--", label="200-MA")
    ax_left.plot(vf_d["timestamp"], vf_d["roll_mean_1000"].ffill(),
                 color="#9C27B0", lw=1.5, ls="-.", label="1000-MA")

    # VF trades
    vf_tr_d = vf_trades[vf_trades["data_day"] == d]
    trade_sizes = vf_tr_d["quantity"].values
    ax_left.scatter(vf_tr_d["timestamp"], vf_tr_d["price"],
                    s=trade_sizes * 3, color="#4CAF50", alpha=0.6, zorder=5,
                    label="Trades (size∝area)")

    # Highlight size-15 trades
    size15_trades = vf_tr_d[vf_tr_d["quantity"] == 15]
    if len(size15_trades) > 0:
        ax_left.scatter(size15_trades["timestamp"], size15_trades["price"],
                       s=100, color=INSIDER_COLOR, zorder=8, marker="*",
                       label=f"Size-15 trades (n={len(size15_trades)})")

    ax_left.set_title(f"Day {d} — VF Extract Price & Reference Lines")
    ax_left.set_ylabel("Price")
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    # Right: distribution of deviations from 200-MA
    ax_right = axes[i, 1]
    dev = vf_d["deviation_from_mean"].dropna()
    ax_right.hist(dev, bins=50, color="#42A5F5", edgecolor="none", alpha=0.8, density=True)
    ax_right.axvline(0, color="black", lw=1.5, ls="--", label="Mean=0")
    ax_right.axvline(dev.mean(), color="#EF5350", lw=1.5, ls="-", label=f"μ={dev.mean():.1f}")
    ax_right.axvline(dev.median(), color="#4CAF50", lw=1.5, ls="-.", label=f"med={dev.median():.1f}")
    # Shade ±1 std
    ax_right.axvspan(-dev.std(), dev.std(), alpha=0.15, color="gray", label=f"±1σ ({dev.std():.1f})")
    ax_right.set_xlabel("VF deviation from 200-MA")
    ax_right.set_ylabel("Density")
    ax_right.set_title(f"Day {d} — Deviation Distribution")
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Timestamp")
axes[-1, 1].set_xlabel("Deviation (ticks)")
plt.tight_layout()
plt.savefig(OUT_DIR / "28_reference_point.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 28_reference_point.png")

# ── Plot 29: Deviation-Conviction Sizing Map ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Plot 29 — |δIV| → Position Conviction Map\n"
             "Larger absolute residual from smile = stronger mis-pricing = more conviction",
             fontsize=13, fontweight="bold")

for i, d in enumerate(DAYS):
    ax = axes[i]
    sub = strike_resid[strike_resid["data_day"] == d].copy()
    sub = sub.sort_values("strike")
    sub["abs_resid"] = sub["median_resid"].abs()
    sub["direction"] = sub["median_resid"].apply(lambda x: "SELL (overpriced)" if x > 0 else "BUY (underpriced)")
    sub["norm_size"] = (sub["abs_resid"] / sub["abs_resid"].max() * 100).round(0)

    bar_colors = ["#EF5350" if v > 0 else "#42A5F5" for v in sub["median_resid"]]
    bars = ax.barh(sub["strike"].astype(str), sub["abs_resid"],
                   color=bar_colors, edgecolor="black", alpha=0.85)

    # Annotate suggested size
    for _, row in sub.iterrows():
        ax.annotate(f"size~{row['norm_size']:.0f}u\n({row['direction'][:4]})",
                   (row["abs_resid"], str(int(row["strike"]))),
                   xytext=(3, 0), textcoords="offset points",
                   va="center", fontsize=7.5, color="black")

    ax.set_xlabel("|δIV| (absolute residual from smile)")
    ax.set_title(f"Day {d}")
    ax.grid(True, alpha=0.3, axis="x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#EF5350", label="SELL (overpriced IV)"),
                        Patch(color="#42A5F5", label="BUY (underpriced IV)")], fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "29_deviation_sizing.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 29_deviation_sizing.png")

# ── Plot 30: Insider in Option Order Books ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Plot 30 — Size-15 Insider in Option Order Books (VEV_4000 and VEV_6500)\n"
             "Deep ITM and deep OTM — where does the insider show up relative to BS price?",
             fontsize=13, fontweight="bold")

# Redo as 2×3 grid: rows=VEV_4000/VEV_6500, cols=day0/day1/day2
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
fig.suptitle("Plot 30 — Size-15 Insider in Option Order Books\n"
             "Top: VEV_4000 (deep ITM) | Bottom: VEV_6500 (deep OTM)",
             fontsize=13, fontweight="bold")

for row_idx, (sym, insider_df) in enumerate(zip(["VEV_4000", "VEV_6500"],
                                                  [vev4000_insider, vev6500_insider])):
    K = int(sym.replace("VEV_", ""))
    for col_idx, d in enumerate(DAYS):
        ax = axes[row_idx, col_idx]
        sub = insider_df[insider_df["data_day"] == d]

        ax.plot(sub["timestamp"], sub["mid_price"], color="#455A64", lw=0.7, alpha=0.8, label="Mid")

        # BS theoretical price on top (using VF mid price)
        vf_d = px_vf[px_vf["data_day"] == d][["timestamp", "mid_price"]].rename(columns={"mid_price": "S"})
        merged = sub.merge(vf_d, on="timestamp", how="left")
        tte_vals = TTE_START[d] - merged["timestamp"] / 1_000_000
        # Use median IV from EDA
        sigma_map = {4000: 0.336, 4500: 0.209, 5000: 0.218, 5100: 0.217,
                     5200: 0.220, 5300: 0.224, 5400: 0.208, 5500: 0.226,
                     6000: 0.361, 6500: 0.546}
        sigma = sigma_map.get(K, 0.25)

        bs_prices = []
        for _, r in merged.iloc[::20].iterrows():
            S = r["S"]
            T = max(TTE_START[d] - r["timestamp"] / 1_000_000, 1e-4) / 252.0
            bp = bs_call(S, K, T, sigma)
            bs_prices.append((r["timestamp"], bp))

        if bs_prices:
            bp_ts, bp_vals = zip(*bs_prices)
            ax.plot(list(bp_ts), list(bp_vals), color="#F57F17", lw=1.2, ls="--",
                    alpha=0.8, label=f"BS (σ={sigma})")

        # Insider presence
        ins = sub[sub["insider"]]
        ax.scatter(ins["timestamp"], ins["mid_price"], s=12, color=INSIDER_COLOR,
                   alpha=0.6, zorder=5, label=f"Size-15 ({len(ins)} ticks)")

        ax.set_title(f"{sym} | Day {d} ({TTE_START[d]}d TTE)")
        ax.set_xlabel("Timestamp")
        if col_idx == 0:
            ax.set_ylabel(f"Price")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "30_insider_option_book.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved 30_insider_option_book.png")

# ═══════════════════════════════════════════════════════════════════════════════
# TEXTUAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SUMMARY — IV DEVIATION MAP (trade signals from smile fit)")
print("="*70)
print(strike_resid.to_string(index=False))

print("\n" + "="*70)
print("SUMMARY — INSIDER SIZE-15 PRESENCE (%% of ticks)")
print("="*70)
for d in DAYS:
    hg_pct = (hg_insider[hg_insider["data_day"]==d]["insider"].mean()*100)
    vf_pct = (vf_insider[vf_insider["data_day"]==d]["insider"].mean()*100)
    v4_pct = (vev4000_insider[vev4000_insider["data_day"]==d]["insider"].mean()*100)
    v6_pct = (vev6500_insider[vev6500_insider["data_day"]==d]["insider"].mean()*100)
    print(f"Day {d}: HG={hg_pct:.0f}%  VF={vf_pct:.0f}%  VEV_4000={v4_pct:.0f}%  VEV_6500={v6_pct:.0f}%")

print("\n" + "="*70)
print("SUMMARY — INSIDER PREDICTIVE POWER (VF future return, 100-tick horizon)")
print("="*70)
print(pred_stats.to_string(index=False))

# Regression: insider bid/ask dummy vs future return
insider_bid_dummy = (vf_for_pred["signal"] == "insider_bid").astype(int)
insider_ask_dummy = (vf_for_pred["signal"] == "insider_ask").astype(int)
net_signal = insider_bid_dummy - insider_ask_dummy
valid = vf_for_pred.dropna(subset=["future_ret"])
slope, intercept, r, p, se = linregress(net_signal[valid.index], valid["future_ret"])
print(f"\n  Regression (net_signal -> future_ret): slope={slope:.4f} r²={r**2:.4f} p={p:.4f}")
print(f"  → size-15 net direction explains {r**2*100:.1f}% of variance in 100-tick VF return")

print("\n" + "="*70)
print("SUMMARY — VF REFERENCE POINT STATISTICS")
print("="*70)
vf_overall = px_vf_sorted["deviation_from_mean"].dropna()
print(f"  Mean deviation from 200-MA: {vf_overall.mean():.2f}")
print(f"  Std of deviations: {vf_overall.std():.2f}")
print(f"  Pct within ±5 ticks: {(vf_overall.abs() < 5).mean()*100:.1f}%")
print(f"  Pct within ±10 ticks: {(vf_overall.abs() < 10).mean()*100:.1f}%")

print("\nAll plots saved to:", OUT_DIR)
print("Done.")
