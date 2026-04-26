"""
Prosperity 4 — Round 3 EDA
Products:
  HYDROGEL_PACK          — regular product (mean-reversion / trend candidate)
  VELVETFRUIT_EXTRACT    — underlying asset for options
  VEV_4000 … VEV_6500   — Velvetfruit Extract Vouchers (call options)
                           TTE = 7 Solvenarian days at data-day 0;
                           each data-day counts as 1 day of TTE.
Days: 0, 1, 2  (TTE: 7, 6, 5)

Run:
    python round3_eda.py
Outputs:
    eda_output/01_*.png … 22_*.png
    eda_output/eda_summary.log
    eda_output/FINDINGS.md
"""

import os, sys, warnings, json, textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── Paths & constants ──────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data", "ROUND_3")
OUT_DIR  = os.path.join(_HERE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS       = [0, 1, 2]
# TTE at the START of each data-day (7 solvenarian days from day 0)
TTE_START  = {0: 7.0, 1: 6.0, 2: 5.0}

STRIKES    = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_NAMES  = [f"VEV_{k}" for k in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL   = "HYDROGEL_PACK"

COLORS_DAY  = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
COLORS_VEV  = {
    "VEV_4000": "#d62728", "VEV_4500": "#9467bd", "VEV_5000": "#8c564b",
    "VEV_5100": "#e377c2", "VEV_5200": "#7f7f7f", "VEV_5300": "#bcbd22",
    "VEV_5400": "#17becf", "VEV_5500": "#aec7e8", "VEV_6000": "#ffbb78",
    "VEV_6500": "#98df8a",
}

LOG_LINES = []

def log(msg: str):
    print(msg)
    LOG_LINES.append(msg)

def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")
    log(f"  [saved] {name}")

# ── Black-Scholes helpers ──────────────────────────────────────────────────────
RISK_FREE = 0.0  # assume zero interest in Solvenarian market

def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """European call price via Black-Scholes. T in Solvenarian days."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    # Scale T: treat 1 day = 1/252 year for volatility normalisation
    T_yr = T / 252.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
    d2 = d1 - sigma * np.sqrt(T_yr)
    return S * norm.cdf(d1) - K * np.exp(-r * T_yr) * norm.cdf(d2)

def bs_call_iv(C_market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    """Implied vol via Brent root-finding. Returns NaN if not solvable."""
    intrinsic = max(S - K, 0.0)
    if C_market < intrinsic - 0.5 or T <= 0 or S <= 0:
        return np.nan
    # Clamp market price to valid range
    C_market = max(C_market, intrinsic + 1e-6)
    def obj(sigma):
        return bs_call_price(S, K, T, sigma, r) - C_market
    try:
        lo_val = obj(1e-6)
        hi_val = obj(10.0)
        if np.sign(lo_val) == np.sign(hi_val):
            return np.nan
        return brentq(obj, 1e-6, 10.0, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan

# ── Data loading ──────────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    frames = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_3_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        df["data_day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["data_day"] * 1_000_000 + df["timestamp"]
    df = df[df["mid_price"] > 0].copy()
    # TTE for each row (continuous: decreases through each day)
    df["TTE"] = df["data_day"].map(TTE_START) - df["timestamp"] / 1_000_000
    return df

def load_trades() -> pd.DataFrame:
    frames = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"trades_round_3_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = df.columns.str.strip()
        df["data_day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["data_day"] * 1_000_000 + df["timestamp"]
    df["TTE"] = df["data_day"].map(TTE_START) - df["timestamp"] / 1_000_000
    return df

log("=" * 70)
log("ROUND 3 EDA  —  IMC Prosperity 4")
log(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)
log("\n[1] Loading data …")

prices = load_prices()
trades = load_trades()

log(f"  Prices: {len(prices):,} rows | Products: {sorted(prices['product'].unique())}")
log(f"  Trades: {len(trades):,} rows | Symbols:  {sorted(trades['symbol'].unique())}")

# Convenience sub-sets
px_hg  = prices[prices["product"] == HYDROGEL].copy()
px_vf  = prices[prices["product"] == UNDERLYING].copy()
px_vev = prices[prices["product"].isin(VEV_NAMES)].copy()
tr_vf  = trades[trades["symbol"] == UNDERLYING].copy()
tr_hg  = trades[trades["symbol"] == HYDROGEL].copy()
tr_vev = trades[trades["symbol"].isin(VEV_NAMES)].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — Overview: mid-price timeseries for all products
# ══════════════════════════════════════════════════════════════════════════════
log("\n[2] Plot 01 — Mid-price overview …")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig.suptitle("Round 3 — Mid-price Overview (all days)", fontsize=14, fontweight="bold")

for day, ax in zip(DAYS, axes):
    sub = prices[prices["data_day"] == day]
    for prod, grp in sub.groupby("product"):
        color = COLORS_VEV.get(prod, ("steelblue" if prod == HYDROGEL else "crimson"))
        lw = 0.8 if prod in VEV_NAMES else 1.8
        ax.plot(grp["timestamp"], grp["mid_price"], label=prod, lw=lw, color=color)
    ax.set_title(f"Day {day}  (TTE {TTE_START[day]} → {TTE_START[day]-1})", fontsize=10)
    ax.set_ylabel("Mid Price")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Timestamp")
fig.tight_layout()
savefig("01_midprice_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — HYDROGEL_PACK detailed analysis
# ══════════════════════════════════════════════════════════════════════════════
log("[3] Plot 02 — HYDROGEL_PACK analysis …")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("HYDROGEL_PACK Analysis", fontsize=13, fontweight="bold")

# 2a: timeseries all days
ax = axes[0, 0]
for day in DAYS:
    sub = px_hg[px_hg["data_day"] == day]
    ax.plot(sub["global_ts"], sub["mid_price"], color=COLORS_DAY[day], label=f"Day {day}", lw=1.2)
ax.set_title("Mid-price (all days)")
ax.set_xlabel("Global timestamp")
ax.set_ylabel("Price")
ax.legend()
ax.grid(alpha=0.3)

# 2b: distribution
ax = axes[0, 1]
for day in DAYS:
    sub = px_hg[px_hg["data_day"] == day]["mid_price"]
    ax.hist(sub, bins=60, alpha=0.5, label=f"Day {day}", color=COLORS_DAY[day], density=True)
ax.set_title("Mid-price distribution")
ax.set_xlabel("Price")
ax.set_ylabel("Density")
ax.legend()
ax.grid(alpha=0.3)

# 2c: bid-ask spread
px_hg["spread"] = px_hg["ask_price_1"] - px_hg["bid_price_1"]
ax = axes[1, 0]
for day in DAYS:
    sub = px_hg[px_hg["data_day"] == day]
    ax.plot(sub["timestamp"], sub["spread"], color=COLORS_DAY[day], label=f"Day {day}", lw=0.8)
ax.set_title("Bid-ask spread")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Spread")
ax.legend()
ax.grid(alpha=0.3)

# 2d: returns distribution
px_hg_sorted = px_hg.sort_values("global_ts")
hg_returns   = px_hg_sorted.groupby("data_day")["mid_price"].pct_change().dropna() * 100
ax = axes[1, 1]
ax.hist(hg_returns, bins=80, color="steelblue", density=True, alpha=0.7)
xr = np.linspace(hg_returns.min(), hg_returns.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, hg_returns.mean(), hg_returns.std()), "r-", lw=2, label="Normal fit")
ax.set_title("HYDROGEL returns distribution")
ax.set_xlabel("Return (%)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("02_hydrogel_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — VF Extract: price timeseries + trade overlay
# ══════════════════════════════════════════════════════════════════════════════
log("[4] Plot 03 — VF Extract timeseries …")

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
fig.suptitle("VELVETFRUIT_EXTRACT — Mid-price & Trade Overlay", fontsize=13, fontweight="bold")

for i, day in enumerate(DAYS):
    ax = axes[i]
    sub_px = px_vf[px_vf["data_day"] == day]
    sub_tr = tr_vf[tr_vf["data_day"] == day]
    ax.plot(sub_px["timestamp"], sub_px["mid_price"], color="steelblue", lw=1.2, label="Mid price")
    if len(sub_tr):
        ax.scatter(sub_tr["timestamp"], sub_tr["price"], color="crimson", s=18,
                   zorder=5, alpha=0.7, label=f"Trades ({len(sub_tr)})")
    ax.set_title(f"Day {day}  |  TTE {TTE_START[day]}")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Timestamp")
fig.tight_layout()
savefig("03_vf_extract_timeseries.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — VF Extract: returns, autocorrelation, rolling std
# ══════════════════════════════════════════════════════════════════════════════
log("[5] Plot 04 — VF Extract returns …")

px_vf_sorted = px_vf.sort_values("global_ts").reset_index(drop=True)
px_vf_sorted["log_ret"] = np.log(px_vf_sorted["mid_price"]).diff()
vf_rets = px_vf_sorted["log_ret"].dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("VELVETFRUIT_EXTRACT — Returns Analysis", fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(px_vf_sorted["global_ts"].iloc[1:], vf_rets.values, color="steelblue", lw=0.5, alpha=0.8)
ax.set_title("Log-returns over time")
ax.set_xlabel("Global timestamp")
ax.set_ylabel("Log return")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.hist(vf_rets, bins=120, density=True, color="steelblue", alpha=0.7)
xr = np.linspace(vf_rets.min(), vf_rets.max(), 300)
ax.plot(xr, stats.norm.pdf(xr, vf_rets.mean(), vf_rets.std()), "r-", lw=2, label="Normal")
ax.set_title("Return distribution")
ax.set_xlabel("Log return")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
lags = range(1, 41)
acf_vals = [vf_rets.autocorr(lag=k) for k in lags]
ax.bar(list(lags), acf_vals, color="steelblue", alpha=0.8)
ci = 1.96 / np.sqrt(len(vf_rets))
ax.axhline(ci, color="red", ls="--", lw=1.2, label="±95% CI")
ax.axhline(-ci, color="red", ls="--", lw=1.2)
ax.set_title("ACF of log-returns (lags 1–40)")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
acf_sq = [vf_rets.pow(2).autocorr(lag=k) for k in lags]
ax.bar(list(lags), acf_sq, color="orange", alpha=0.8)
ax.axhline(ci, color="red", ls="--", lw=1.2, label="±95% CI")
ax.axhline(-ci, color="red", ls="--", lw=1.2)
ax.set_title("ACF of squared returns (vol clustering)")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("04_vf_extract_returns.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 05 — Rolling volatility
# ══════════════════════════════════════════════════════════════════════════════
log("[6] Plot 05 — Rolling volatility …")

WINDOW = 200  # ticks
px_vf_sorted["roll_vol"] = vf_rets.rolling(WINDOW).std() * np.sqrt(252 * 10_000 / WINDOW)

fig, axes = plt.subplots(2, 1, figsize=(15, 8))
fig.suptitle("VELVETFRUIT_EXTRACT — Rolling Annualised Volatility", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(px_vf_sorted["global_ts"], px_vf_sorted["mid_price"], color="steelblue", lw=1)
ax.set_ylabel("Mid price")
ax.set_title("Price")
ax.grid(alpha=0.3)
for day in DAYS:
    ax.axvline(day * 1_000_000, color="grey", ls="--", lw=0.8)

ax = axes[1]
ax.plot(px_vf_sorted["global_ts"], px_vf_sorted["roll_vol"], color="crimson", lw=1)
ax.set_ylabel("Annualised vol")
ax.set_xlabel("Global timestamp")
ax.set_title(f"Rolling std × √(252 × 10k / {WINDOW}) — rolling {WINDOW}-tick window")
ax.grid(alpha=0.3)
for day in DAYS:
    ax.axvline(day * 1_000_000, color="grey", ls="--", lw=0.8)

fig.tight_layout()
savefig("05_rolling_volatility.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 06 — Option mid-price timeseries
# ══════════════════════════════════════════════════════════════════════════════
log("[7] Plot 06 — Option mid-price timeseries …")

fig, axes = plt.subplots(2, 1, figsize=(16, 11))
fig.suptitle("VEV Option Mid-prices across Days", fontsize=13, fontweight="bold")

# Panel a: all strikes, all days (log scale)
ax = axes[0]
for vev in VEV_NAMES:
    sub = px_vev[px_vev["product"] == vev]
    if sub.empty:
        continue
    ax.plot(sub["global_ts"], sub["mid_price"], label=vev, lw=0.9, color=COLORS_VEV[vev])
ax.set_yscale("log")
ax.set_title("All strikes (log scale) — all 3 days")
ax.set_ylabel("Option mid-price (log)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
for day in DAYS:
    ax.axvline(day * 1_000_000, color="grey", ls="--", lw=0.8)

# Panel b: zoomed on near-ATM strikes
ax = axes[1]
atm_strikes = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
for vev in atm_strikes:
    sub = px_vev[px_vev["product"] == vev]
    ax.plot(sub["global_ts"], sub["mid_price"], label=vev, lw=1, color=COLORS_VEV[vev])
ax.set_title("Near-ATM strikes — linear scale")
ax.set_ylabel("Option mid-price")
ax.set_xlabel("Global timestamp")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
for day in DAYS:
    ax.axvline(day * 1_000_000, color="grey", ls="--", lw=0.8)

fig.tight_layout()
savefig("06_option_price_timeseries.png")

# ══════════════════════════════════════════════════════════════════════════════
# Compute intrinsic value and time value (needs underlying aligned per tick)
# ══════════════════════════════════════════════════════════════════════════════
log("[8] Computing intrinsic & time value …")

# Pivot VF Extract mid price indexed by (data_day, timestamp)
vf_idx = px_vf.set_index(["data_day", "timestamp"])["mid_price"]

def get_underlying(day, ts):
    try:
        return vf_idx.loc[(day, ts)]
    except KeyError:
        return np.nan

# Build enriched options frame
px_vev = px_vev.copy()
px_vev["strike"] = px_vev["product"].str.replace("VEV_", "").astype(float)

# Vectorised nearest-timestamp join using merge_asof per day
under_parts = []
for day in DAYS:
    sub_vev = px_vev[px_vev["data_day"] == day].copy()
    sub_vf  = px_vf[px_vf["data_day"] == day][["timestamp", "mid_price"]].rename(
        columns={"mid_price": "S"}).sort_values("timestamp")
    sub_vev = sub_vev.sort_values("timestamp")
    merged = pd.merge_asof(sub_vev, sub_vf, on="timestamp", direction="nearest")
    under_parts.append(merged)

px_vev_e = pd.concat(under_parts, ignore_index=True)
px_vev_e["intrinsic"]  = np.maximum(px_vev_e["S"] - px_vev_e["strike"], 0.0)
px_vev_e["time_value"] = px_vev_e["mid_price"] - px_vev_e["intrinsic"]
# clamp very slight negatives from rounding
px_vev_e["time_value"] = px_vev_e["time_value"].clip(lower=-5)

# Compute BS IV for each row
log("  Computing implied vols (this may take ~30 s) …")
def safe_iv(row):
    return bs_call_iv(row["mid_price"], row["S"], row["strike"], row["TTE"])

# Subsample to speed up: every 10th tick per product
iv_sample = px_vev_e.iloc[::10].copy()
iv_sample["IV"] = iv_sample.apply(safe_iv, axis=1)
iv_sample["moneyness"] = iv_sample["S"] / iv_sample["strike"]
log(f"  IV sample size: {len(iv_sample):,} rows")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 07 — Intrinsic vs time value decomposition
# ══════════════════════════════════════════════════════════════════════════════
log("[9] Plot 07 — Intrinsic vs time value …")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Option Price Decomposition: Intrinsic vs Time Value", fontsize=13, fontweight="bold")

for i, day in enumerate(DAYS):
    ax = axes[i]
    sub = px_vev_e[px_vev_e["data_day"] == day].groupby("strike").agg(
        mid_mean=("mid_price", "mean"),
        intrinsic_mean=("intrinsic", "mean"),
        tv_mean=("time_value", "mean"),
    ).reset_index()
    ax.bar(sub["strike"].astype(str), sub["intrinsic_mean"], label="Intrinsic", color="steelblue")
    ax.bar(sub["strike"].astype(str), sub["tv_mean"], bottom=sub["intrinsic_mean"],
           label="Time value", color="orange")
    ax.set_title(f"Day {day}  (TTE ≈ {TTE_START[day]})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option price")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
savefig("07_intrinsic_time_value.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 08 — Moneyness analysis
# ══════════════════════════════════════════════════════════════════════════════
log("[10] Plot 08 — Moneyness analysis …")

# Per-snapshot moneyness = S/K
px_vev_e["moneyness"] = px_vev_e["S"] / px_vev_e["strike"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Moneyness Analysis (S/K)", fontsize=13, fontweight="bold")

ax = axes[0]
for vev in VEV_NAMES:
    sub = px_vev_e[px_vev_e["product"] == vev]
    ax.plot(sub["global_ts"], sub["moneyness"], label=vev, lw=0.8, color=COLORS_VEV[vev])
ax.axhline(1.0, color="black", ls="--", lw=1.5, label="ATM (S=K)")
ax.set_title("Moneyness (S/K) over time — all strikes")
ax.set_xlabel("Global timestamp")
ax.set_ylabel("S / K")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
# At-the-money time value: how much time value for options near ATM
atm_mask = (px_vev_e["moneyness"] > 0.95) & (px_vev_e["moneyness"] < 1.05)
sub_atm = px_vev_e[atm_mask]
for day in DAYS:
    d = sub_atm[sub_atm["data_day"] == day]
    if len(d):
        ax.scatter(d["strike"], d["time_value"], s=8, alpha=0.4,
                   label=f"Day {day}", color=COLORS_DAY[day])
ax.set_title("Time value for near-ATM options (|S/K - 1| < 5%)")
ax.set_xlabel("Strike")
ax.set_ylabel("Time value")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("08_moneyness_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 09 — Implied volatility smile (per day)
# ══════════════════════════════════════════════════════════════════════════════
log("[11] Plot 09 — IV smile …")

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("Implied Volatility Smile by Day", fontsize=13, fontweight="bold")

iv_by_day_strike = {}
for i, day in enumerate(DAYS):
    ax = axes[i]
    sub = iv_sample[(iv_sample["data_day"] == day) & iv_sample["IV"].notna()]
    if sub.empty:
        ax.set_title(f"Day {day} — no IV data")
        continue
    med_iv = sub.groupby("strike")["IV"].median().reset_index()
    q25    = sub.groupby("strike")["IV"].quantile(0.25).reset_index().rename(columns={"IV": "q25"})
    q75    = sub.groupby("strike")["IV"].quantile(0.75).reset_index().rename(columns={"IV": "q75"})
    iv_band = med_iv.merge(q25).merge(q75)
    iv_by_day_strike[day] = med_iv

    ax.plot(iv_band["strike"], iv_band["IV"], "o-", color=COLORS_DAY[day], lw=1.8, label="Median IV")
    ax.fill_between(iv_band["strike"], iv_band["q25"], iv_band["q75"],
                    alpha=0.25, color=COLORS_DAY[day], label="IQR")
    ax.set_title(f"Day {day}  (TTE ≈ {TTE_START[day]})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.tight_layout()
savefig("09_iv_smile.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10 — IV surface (heat-map: strike × global_ts)
# ══════════════════════════════════════════════════════════════════════════════
log("[12] Plot 10 — IV surface …")

iv_pivot = iv_sample[iv_sample["IV"].notna() & (iv_sample["IV"] < 5)].copy()
# Bin timestamps for x-axis
iv_pivot["ts_bin"] = (iv_pivot["global_ts"] // 50_000) * 50_000
iv_surf = iv_pivot.groupby(["ts_bin", "strike"])["IV"].median().unstack("strike")

fig, ax = plt.subplots(figsize=(16, 7))
fig.suptitle("Implied Volatility Surface (Strike × Time)", fontsize=13, fontweight="bold")
im = ax.pcolormesh(iv_surf.index, iv_surf.columns, iv_surf.T, cmap="RdYlGn_r", shading="auto",
                   vmin=0.1, vmax=min(iv_surf.values.max(), 3.0))
plt.colorbar(im, ax=ax, label="Implied Volatility")
ax.set_xlabel("Global timestamp")
ax.set_ylabel("Strike")
for day in DAYS:
    ax.axvline(day * 1_000_000, color="white", ls="--", lw=1.2)
fig.tight_layout()
savefig("10_iv_surface.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 11 — IV evolution as TTE decreases
# ══════════════════════════════════════════════════════════════════════════════
log("[13] Plot 11 — IV vs TTE …")

# ATM options only (moneyness 0.97-1.03)
atm_iv = iv_sample[
    (iv_sample["moneyness"] > 0.97) & (iv_sample["moneyness"] < 1.03) &
    iv_sample["IV"].notna() & (iv_sample["IV"] < 5)
].copy()

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("ATM Implied Volatility vs Time-To-Expiry (TTE)", fontsize=13, fontweight="bold")

for vev in VEV_NAMES:
    sub = atm_iv[atm_iv["product"] == vev]
    if len(sub) < 3:
        continue
    ax.scatter(sub["TTE"], sub["IV"], s=10, alpha=0.4, label=vev, color=COLORS_VEV[vev])

ax.set_xlabel("TTE (Solvenarian days)")
ax.set_ylabel("Implied Volatility")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
savefig("11_iv_vs_tte.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 12 — Bid-ask spread analysis for all products
# ══════════════════════════════════════════════════════════════════════════════
log("[14] Plot 12 — Spread analysis …")

prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
spread_by_prod = prices.groupby("product")["spread"].describe()[["mean", "50%", "std", "max"]]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Bid-Ask Spread Analysis", fontsize=13, fontweight="bold")

ax = axes[0]
prods_sorted = spread_by_prod.sort_values("mean").index
means = spread_by_prod.loc[prods_sorted, "mean"]
meds  = spread_by_prod.loc[prods_sorted, "50%"]
x = np.arange(len(prods_sorted))
ax.barh(x, means, alpha=0.8, label="Mean spread", color="steelblue")
ax.barh(x, meds, alpha=0.5, label="Median spread", color="orange")
ax.set_yticks(x)
ax.set_yticklabels(prods_sorted, fontsize=8)
ax.set_xlabel("Spread")
ax.set_title("Mean and median spread by product")
ax.legend()
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
for day in DAYS:
    sub = px_vf[px_vf["data_day"] == day]
    spread_vf = sub["ask_price_1"] - sub["bid_price_1"]
    ax.plot(sub["timestamp"], spread_vf, color=COLORS_DAY[day], lw=0.8, label=f"Day {day}")
ax.set_title("VF Extract spread over time")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Spread")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("12_spread_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 13 — Order book depth & imbalance
# ══════════════════════════════════════════════════════════════════════════════
log("[15] Plot 13 — Order book depth & imbalance …")

def compute_imbalance(df: pd.DataFrame) -> pd.Series:
    """Bid-ask volume imbalance = (bid_vol1 - ask_vol1) / (bid_vol1 + ask_vol1)"""
    bv = df["bid_volume_1"].fillna(0)
    av = df["ask_volume_1"].fillna(0)
    denom = bv + av
    return np.where(denom > 0, (bv - av) / denom, 0.0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Order Book Depth & Imbalance (VF Extract + HYDROGEL)", fontsize=13, fontweight="bold")

for col, (prod_df, prod_name) in enumerate([(px_vf, "VF Extract"), (px_hg, "HYDROGEL")]):
    # Depth over time (top-of-book only for clarity)
    ax = axes[0, col]
    for day in DAYS:
        sub = prod_df[prod_df["data_day"] == day]
        total_depth = sub["bid_volume_1"].fillna(0) + sub["ask_volume_1"].fillna(0)
        ax.plot(sub["timestamp"], total_depth, color=COLORS_DAY[day], lw=0.7, label=f"Day {day}")
    ax.set_title(f"{prod_name} — Level-1 depth over time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Total volume (bid1 + ask1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, col]
    imb_vals = []
    for day in DAYS:
        sub = prod_df[prod_df["data_day"] == day].copy()
        sub["imbalance"] = compute_imbalance(sub)
        ax.plot(sub["timestamp"], sub["imbalance"], color=COLORS_DAY[day], lw=0.7, label=f"Day {day}", alpha=0.8)
        imb_vals.extend(sub["imbalance"].tolist())
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title(f"{prod_name} — Order imbalance (bid-ask)/total")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Imbalance")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.tight_layout()
savefig("13_orderbook_depth_imbalance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 14 — Trade flow & timing analysis
# ══════════════════════════════════════════════════════════════════════════════
log("[16] Plot 14 — Trade flow & timing …")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Trade Flow & Timing Analysis", fontsize=13, fontweight="bold")

# Trade counts per product
ax = axes[0, 0]
trade_counts = trades.groupby("symbol").size().sort_values(ascending=False)
ax.barh(trade_counts.index, trade_counts.values, color="steelblue")
ax.set_title("Total trade count by product")
ax.set_xlabel("Count")
ax.grid(alpha=0.3, axis="x")

# Volume by product
ax = axes[0, 1]
vol_by_prod = trades.groupby("symbol")["quantity"].sum().sort_values(ascending=False)
ax.barh(vol_by_prod.index, vol_by_prod.values, color="crimson")
ax.set_title("Total traded volume by product")
ax.set_xlabel("Volume")
ax.grid(alpha=0.3, axis="x")

# VF Extract trade inter-arrival times
ax = axes[1, 0]
if len(tr_vf) > 1:
    tr_vf_s = tr_vf.sort_values("global_ts")
    iat = tr_vf_s["global_ts"].diff().dropna()
    ax.hist(iat, bins=60, density=True, color="steelblue", alpha=0.8)
    # Fit exponential (expected under Poisson)
    rate = 1.0 / iat.mean()
    xr = np.linspace(0, iat.quantile(0.99), 300)
    ax.plot(xr, stats.expon.pdf(xr, scale=1/rate), "r-", lw=2, label=f"Exp fit (λ={rate:.4f})")
    ax.set_title("VF Extract inter-trade intervals")
    ax.set_xlabel("Ticks between trades")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

# Cumulative trade volume over time (VF Extract)
ax = axes[1, 1]
for day in DAYS:
    sub = tr_vf[tr_vf["data_day"] == day].sort_values("timestamp")
    if len(sub):
        cumvol = sub["quantity"].cumsum()
        ax.plot(sub["timestamp"], cumvol, color=COLORS_DAY[day], lw=1.5, label=f"Day {day}")
ax.set_title("VF Extract cumulative trade volume")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Cumulative quantity")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("14_trade_flow_timing.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 15 — Bot behavior: trade price vs mid-price
# ══════════════════════════════════════════════════════════════════════════════
log("[17] Plot 15 — Bot behavior: trade price vs mid …")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Bot Behavior: Trade Price Relative to Mid-Price", fontsize=13, fontweight="bold")

# VF Extract
ax = axes[0]
for day in DAYS:
    sub_tr = tr_vf[tr_vf["data_day"] == day].copy().sort_values("timestamp")
    sub_px = px_vf[px_vf["data_day"] == day][["timestamp", "mid_price"]].sort_values("timestamp")
    if len(sub_tr) and len(sub_px):
        merged_t = pd.merge_asof(sub_tr, sub_px, on="timestamp", direction="nearest")
        merged_t["trade_vs_mid"] = merged_t["price"] - merged_t["mid_price"]
        ax.scatter(merged_t["timestamp"], merged_t["trade_vs_mid"],
                   s=20, alpha=0.7, label=f"Day {day}", color=COLORS_DAY[day])
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_title("VF Extract: trade price - mid price")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Trade price − mid")
ax.legend()
ax.grid(alpha=0.3)

# VEV options (all strikes combined)
ax = axes[1]
for day in DAYS:
    sub_tr = tr_vev[tr_vev["data_day"] == day].copy()
    if len(sub_tr):
        sub_px_vev_day = px_vev_e[px_vev_e["data_day"] == day]
        deltas, trade_ts = [], []
        for _, row in sub_tr.iterrows():
            match = sub_px_vev_day[sub_px_vev_day["product"] == row["symbol"]]
            if len(match):
                near = match.iloc[(match["timestamp"] - row["timestamp"]).abs().argmin()]
                delta = row["price"] - near["mid_price"]
                deltas.append(delta)
                trade_ts.append(row["timestamp"])
        if deltas:
            ax.scatter(trade_ts, deltas, s=20, alpha=0.7, label=f"Day {day}", color=COLORS_DAY[day])
ax.axhline(0, color="black", ls="--", lw=1)
ax.set_title("VEV options: trade price - mid price")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Trade price − mid")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
savefig("15_bot_trade_vs_mid.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 16 — VWAP analysis
# ══════════════════════════════════════════════════════════════════════════════
log("[18] Plot 16 — VWAP analysis …")

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle("VWAP Analysis (VF Extract + HYDROGEL)", fontsize=13, fontweight="bold")

for ax, (prod_tr, prod_px, name) in zip(axes, [
    (tr_vf,  px_vf,  "VF Extract"),
    (tr_hg,  px_hg,  "HYDROGEL_PACK"),
]):
    for day in DAYS:
        sub_px = prod_px[prod_px["data_day"] == day]
        sub_tr = prod_tr[prod_tr["data_day"] == day].sort_values("timestamp")
        ax.plot(sub_px["timestamp"], sub_px["mid_price"],
                color=COLORS_DAY[day], lw=1.2, label=f"Mid (Day {day})")
        if len(sub_tr) > 0:
            # rolling VWAP
            sub_tr["vw_num"] = (sub_tr["price"] * sub_tr["quantity"]).cumsum()
            sub_tr["vw_den"] = sub_tr["quantity"].cumsum()
            sub_tr["vwap"]   = sub_tr["vw_num"] / sub_tr["vw_den"]
            ax.plot(sub_tr["timestamp"], sub_tr["vwap"],
                    color=COLORS_DAY[day], lw=2, ls="--", label=f"VWAP (Day {day})")
    ax.set_title(f"{name} — Mid vs VWAP")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Timestamp")
fig.tight_layout()
savefig("16_vwap_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 17 — Imbalance predictiveness
# ══════════════════════════════════════════════════════════════════════════════
log("[19] Plot 17 — Imbalance predictiveness …")

PRED_LAG = 50  # ticks ahead

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Order Imbalance Predictive Power (next-N-tick return)", fontsize=13, fontweight="bold")

for col, (prod_df, name) in enumerate([(px_vf, "VF Extract"), (px_hg, "HYDROGEL")]):
    ax = axes[col]
    prod_sorted = prod_df.sort_values("global_ts").copy()
    prod_sorted["imb"] = compute_imbalance(prod_sorted)
    prod_sorted["fwd_ret"] = prod_sorted["mid_price"].pct_change(PRED_LAG).shift(-PRED_LAG) * 100
    valid = prod_sorted.dropna(subset=["imb", "fwd_ret"])
    ax.scatter(valid["imb"], valid["fwd_ret"], s=2, alpha=0.2, color="steelblue")
    # Regression line
    slope, intercept, r, p, _ = stats.linregress(valid["imb"], valid["fwd_ret"])
    xr = np.linspace(-1, 1, 100)
    ax.plot(xr, intercept + slope * xr, "r-", lw=2,
            label=f"slope={slope:.4f}\nr={r:.3f}, p={p:.3g}")
    ax.set_title(f"{name}: imbalance → fwd {PRED_LAG}-tick return")
    ax.set_xlabel("Order imbalance (t)")
    ax.set_ylabel(f"Return at t+{PRED_LAG} (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.tight_layout()
savefig("17_imbalance_predictive.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 18 — QQ plots (normality check)
# ══════════════════════════════════════════════════════════════════════════════
log("[20] Plot 18 — QQ plots …")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("QQ Plots — Normality of Log Returns", fontsize=13, fontweight="bold")

for ax, (rets, name) in zip(axes, [
    (vf_rets,   "VF Extract"),
    (hg_returns / 100, "HYDROGEL"),
]):
    r_clean = rets.dropna()
    (osm, osr), (slope, intercept, r) = stats.probplot(r_clean, dist="norm")
    ax.scatter(osm, osr, s=5, alpha=0.4, color="steelblue")
    ax.plot(osm, intercept + slope * np.array(osm), "r-", lw=2)
    ax.set_title(f"{name}  (r={r:.4f})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.grid(alpha=0.3)

fig.tight_layout()
savefig("18_qq_plots.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 19 — Option Greeks: Delta across strikes over time
# ══════════════════════════════════════════════════════════════════════════════
log("[21] Plot 19 — Option Greeks (delta) …")

def bs_delta(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    T_yr = T / 252.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
    return norm.cdf(d1)

# Use median IV per strike-day as constant sigma estimate
iv_med_by_sd = iv_sample[iv_sample["IV"].notna()].groupby(["data_day", "strike"])["IV"].median()

delta_rows = []
for (day, strike), sig_med in iv_med_by_sd.items():
    sub = px_vev_e[(px_vev_e["data_day"] == day) & (px_vev_e["strike"] == strike)].iloc[::20]
    for _, row in sub.iterrows():
        d = bs_delta(row["S"], strike, row["TTE"], sig_med)
        delta_rows.append({"data_day": day, "strike": strike, "timestamp": row["timestamp"],
                           "delta": d, "S": row["S"]})
delta_df = pd.DataFrame(delta_rows)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("BS Delta by Strike and Day", fontsize=13, fontweight="bold")

for i, day in enumerate(DAYS):
    ax = axes[i]
    sub = delta_df[delta_df["data_day"] == day]
    for strike in STRIKES:
        s_sub = sub[sub["strike"] == strike]
        if len(s_sub) < 2:
            continue
        vev_name = f"VEV_{int(strike)}"
        ax.plot(s_sub["timestamp"], s_sub["delta"], label=vev_name,
                lw=1, color=COLORS_VEV[vev_name])
    ax.set_title(f"Day {day}  (TTE ≈ {TTE_START[day]})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Delta")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.tight_layout()
savefig("19_option_deltas.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 20 — Option P&L scenarios (what if you sold OTM options)
# ══════════════════════════════════════════════════════════════════════════════
log("[22] Plot 20 — P&L scenario: short OTM options …")

# Simulate: enter short at day-0 avg price, hold to day-2 end
# Final payoff = max(S_final - K, 0)
# P&L = premium_collected - payoff

S_final_median = px_vf[px_vf["data_day"] == 2]["mid_price"].median()
pnl_rows = []
for strike in STRIKES:
    vev_name = f"VEV_{strike}"
    entry_mask = (px_vev_e["data_day"] == 0) & (px_vev_e["product"] == vev_name)
    entry_price = px_vev_e[entry_mask]["mid_price"].mean()
    payoff = max(S_final_median - strike, 0)
    pnl = entry_price - payoff
    pnl_rows.append({"strike": strike, "entry_premium": entry_price,
                     "payoff_at_day2_median_S": payoff, "pnl": pnl,
                     "S_final": S_final_median})
pnl_df = pd.DataFrame(pnl_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Short Option P&L Scenario  (S_final ≈ {S_final_median:.0f})", fontsize=13, fontweight="bold")

ax = axes[0]
colors = ["green" if v >= 0 else "red" for v in pnl_df["pnl"]]
ax.bar(pnl_df["strike"].astype(str), pnl_df["pnl"], color=colors)
ax.axhline(0, color="black", lw=1)
ax.set_title("P&L per unit short option (at day-2 median S)")
ax.set_xlabel("Strike")
ax.set_ylabel("P&L")
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(pnl_df["strike"].astype(str), pnl_df["entry_premium"], label="Premium collected",
       color="steelblue", alpha=0.8)
ax.bar(pnl_df["strike"].astype(str), pnl_df["payoff_at_day2_median_S"],
       label="Payoff (day-2 S)", color="crimson", alpha=0.6)
ax.set_title("Premium vs Payoff")
ax.set_xlabel("Strike")
ax.set_ylabel("Value")
ax.legend()
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
savefig("20_pnl_scenarios.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 21 — Correlation matrix across all products
# ══════════════════════════════════════════════════════════════════════════════
log("[23] Plot 21 — Correlation matrix …")

# Build a pivot of mid-prices vs global_ts (coarse bins for all products)
BIN = 5000
prices["ts_bin"] = (prices["global_ts"] // BIN) * BIN
pivot_all = prices.pivot_table(index="ts_bin", columns="product", values="mid_price", aggfunc="last")
pivot_ret  = pivot_all.pct_change().dropna()
corr = pivot_ret.corr()

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle("Return Correlation Matrix (all products)", fontsize=13, fontweight="bold")
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticklabels(corr.index, fontsize=8)
# Annotate cells
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        val = corr.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > 0.6 else "black")
fig.tight_layout()
savefig("21_correlation_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTS  ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════════
log("\n[24] Running hypothesis tests …")

test_results = {}

# H1 — HYDROGEL_PACK mean reversion (ADF)
try:
    from statsmodels.tsa.stattools import adfuller
    hg_prices_all = px_hg.sort_values("global_ts")["mid_price"].values
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(hg_prices_all, autolag="AIC")
    test_results["H1_HYDROGEL_ADF"] = {
        "stat": adf_stat, "p": adf_p, "lags": adf_lags,
        "crit_1pct": adf_crit["1%"], "crit_5pct": adf_crit["5%"],
        "reject_H0": adf_p < 0.05,
        "interpretation": "Mean-reverting (stationary)" if adf_p < 0.05 else "Cannot reject random walk"
    }
    log(f"  H1 ADF(HYDROGEL): stat={adf_stat:.4f}  p={adf_p:.4g}  "
        f"{'REJECT H0 → MEAN REVERTING' if adf_p < 0.05 else 'fail to reject H0'}")
except Exception as e:
    log(f"  H1 ADF: error — {e}")
    test_results["H1_HYDROGEL_ADF"] = {"error": str(e)}

# H2 — VF Extract: ADF
try:
    vf_prices_all = px_vf.sort_values("global_ts")["mid_price"].values
    adf_stat2, adf_p2, *_ = adfuller(vf_prices_all, autolag="AIC")
    test_results["H2_VF_ADF"] = {
        "stat": adf_stat2, "p": adf_p2,
        "reject_H0": adf_p2 < 0.05,
        "interpretation": "Stationary" if adf_p2 < 0.05 else "Unit root / random walk"
    }
    log(f"  H2 ADF(VFExtract): stat={adf_stat2:.4f}  p={adf_p2:.4g}  "
        f"{'REJECT H0' if adf_p2 < 0.05 else 'fail to reject H0'}")
except Exception as e:
    test_results["H2_VF_ADF"] = {"error": str(e)}

# H3 — VF returns normality (Jarque-Bera)
try:
    jb_stat, jb_p = stats.jarque_bera(vf_rets.dropna())
    test_results["H3_VF_JB_Normality"] = {
        "stat": jb_stat, "p": jb_p,
        "reject_H0": jb_p < 0.05,
        "interpretation": "Non-normal" if jb_p < 0.05 else "Normal"
    }
    log(f"  H3 Jarque-Bera(VF returns): stat={jb_stat:.4f}  p={jb_p:.4g}  "
        f"{'REJECT H0 → NON-NORMAL' if jb_p < 0.05 else 'fail to reject H0'}")
except Exception as e:
    test_results["H3_VF_JB_Normality"] = {"error": str(e)}

# H4 — IV flat across strikes (ANOVA, per day)
try:
    from scipy.stats import f_oneway
    for day in DAYS:
        iv_by_strike = []
        for strike in STRIKES:
            sub = iv_sample[(iv_sample["data_day"] == day) &
                            (iv_sample["strike"] == strike) &
                            iv_sample["IV"].notna() &
                            (iv_sample["IV"] < 5)]
            if len(sub) > 3:
                iv_by_strike.append(sub["IV"].values)
        if len(iv_by_strike) >= 2:
            f_stat, f_p = f_oneway(*iv_by_strike)
            key = f"H4_IV_ANOVA_day{day}"
            test_results[key] = {
                "F": f_stat, "p": f_p,
                "reject_H0": f_p < 0.05,
                "interpretation": "IV varies across strikes (smile exists)" if f_p < 0.05 else "IV flat (no smile)"
            }
            log(f"  H4 ANOVA(IV, Day {day}): F={f_stat:.3f}  p={f_p:.4g}  "
                f"{'SMILE EXISTS' if f_p < 0.05 else 'flat IV'}")
except Exception as e:
    test_results["H4_IV_ANOVA"] = {"error": str(e)}

# H5 — Trade arrivals Poisson (KS test on inter-arrivals)
try:
    tr_sorted = trades.sort_values("global_ts")
    iat = tr_sorted["global_ts"].diff().dropna().values
    iat = iat[iat > 0]
    lambda_est = 1.0 / iat.mean()
    ks_stat, ks_p = stats.kstest(iat, "expon", args=(0, 1/lambda_est))
    test_results["H5_Poisson_Arrivals"] = {
        "ks_stat": ks_stat, "p": ks_p,
        "mean_iat": float(iat.mean()),
        "reject_H0": ks_p < 0.05,
        "interpretation": "Non-Poisson (clustered/periodic)" if ks_p < 0.05 else "Poisson-consistent"
    }
    log(f"  H5 KS(Poisson, all trades): stat={ks_stat:.4f}  p={ks_p:.4g}  "
        f"{'NON-POISSON' if ks_p < 0.05 else 'Poisson-consistent'}")
except Exception as e:
    test_results["H5_Poisson_Arrivals"] = {"error": str(e)}

# H6 — Order imbalance predictiveness (OLS)
try:
    from scipy.stats import linregress as lr
    for prod_df, name in [(px_vf, "VF_Extract"), (px_hg, "HYDROGEL")]:
        prod_sorted = prod_df.sort_values("global_ts").copy()
        prod_sorted["imb"] = compute_imbalance(prod_sorted)
        prod_sorted["fwd_ret"] = prod_sorted["mid_price"].pct_change(50).shift(-50) * 100
        valid = prod_sorted.dropna(subset=["imb", "fwd_ret"])
        if len(valid) > 100:
            slope, intercept, r, p, se = lr(valid["imb"], valid["fwd_ret"])
            key = f"H6_Imbalance_Pred_{name}"
            test_results[key] = {
                "slope": slope, "intercept": intercept, "r": r, "p": p,
                "reject_H0": p < 0.05,
                "interpretation": "Imbalance predicts returns" if p < 0.05 else "No predictive power"
            }
            log(f"  H6 OLS(imb→fwdret, {name}): slope={slope:.5f}  r={r:.4f}  p={p:.4g}  "
                f"{'PREDICTIVE' if p < 0.05 else 'no signal'}")
except Exception as e:
    test_results["H6_Imbalance_Pred"] = {"error": str(e)}

# H7 — Time value ≥ 0 for all options
try:
    neg_tv = px_vev_e[px_vev_e["time_value"] < -1.0]  # allow 1 tick rounding
    frac_neg = len(neg_tv) / len(px_vev_e)
    test_results["H7_TimeValue_Positive"] = {
        "pct_negative": frac_neg * 100,
        "count_neg": len(neg_tv),
        "by_strike": neg_tv.groupby("strike").size().to_dict(),
        "reject_H0": frac_neg > 0.01,
        "interpretation": "Some options trade below intrinsic (arbitrage?)" if frac_neg > 0.01
                          else "All options priced above intrinsic"
    }
    log(f"  H7 Time-value ≥ 0: {frac_neg*100:.2f}% negative  "
        f"({'ANOMALY' if frac_neg > 0.01 else 'OK'})")
except Exception as e:
    test_results["H7_TimeValue"] = {"error": str(e)}

# H8 — Ljung-Box on VF returns (autocorrelation)
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_ret = acorr_ljungbox(vf_rets.dropna(), lags=[10, 20], return_df=True)
    lb_sq  = acorr_ljungbox(vf_rets.dropna()**2, lags=[10, 20], return_df=True)
    test_results["H8_LjungBox_Returns"] = {
        "lb_ret_lag10_p": float(lb_ret["lb_pvalue"].iloc[0]),
        "lb_sq_lag10_p": float(lb_sq["lb_pvalue"].iloc[0]),
        "autocorr_in_returns": lb_ret["lb_pvalue"].iloc[0] < 0.05,
        "vol_clustering": lb_sq["lb_pvalue"].iloc[0] < 0.05,
    }
    log(f"  H8 Ljung-Box(VF returns lag10):  p={lb_ret['lb_pvalue'].iloc[0]:.4g}  "
        f"({'autocorr' if lb_ret['lb_pvalue'].iloc[0] < 0.05 else 'no autocorr'})")
    log(f"  H8 Ljung-Box(VF sq.ret lag10):   p={lb_sq['lb_pvalue'].iloc[0]:.4g}  "
        f"({'vol clustering' if lb_sq['lb_pvalue'].iloc[0] < 0.05 else 'no clustering'})")
except Exception as e:
    test_results["H8_LjungBox"] = {"error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 22 — Hypothesis test summary
# ══════════════════════════════════════════════════════════════════════════════
log("[25] Plot 22 — Hypothesis test summary …")

labels = []
p_values = []
rejected = []

hyp_map = [
    ("H1", "HYDROGEL_ADF"),
    ("H2", "VF_ADF"),
    ("H3", "VF_JB_Normality"),
    ("H4a", "IV_ANOVA_day0"),
    ("H4b", "IV_ANOVA_day1"),
    ("H4c", "IV_ANOVA_day2"),
    ("H5", "Poisson_Arrivals"),
    ("H6a", "Imbalance_Pred_VF_Extract"),
    ("H6b", "Imbalance_Pred_HYDROGEL"),
    ("H7", "TimeValue_Positive"),
    ("H8a", "LjungBox_Returns"),
]

for code, key_suffix in hyp_map:
    for k, v in test_results.items():
        if key_suffix.lower() in k.lower():
            p = v.get("p") or v.get("p_value") or v.get("lb_ret_lag10_p")
            r = v.get("reject_H0") or v.get("autocorr_in_returns")
            if p is not None:
                labels.append(code)
                p_values.append(p)
                rejected.append(bool(r))
            break

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Hypothesis Test Summary (p-values)", fontsize=13, fontweight="bold")
colors_bar = ["crimson" if r else "steelblue" for r in rejected]
bars = ax.bar(labels, p_values, color=colors_bar, alpha=0.8)
ax.axhline(0.05, color="black", ls="--", lw=1.5, label="α = 0.05")
ax.set_yscale("log")
ax.set_xlabel("Hypothesis")
ax.set_ylabel("p-value (log scale)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
# Legend for colors
from matplotlib.patches import Patch
leg_els = [Patch(facecolor="crimson", alpha=0.8, label="Reject H0"),
           Patch(facecolor="steelblue", alpha=0.8, label="Fail to reject H0")]
ax.legend(handles=leg_els + [plt.Line2D([0], [0], color="black", ls="--", label="α=0.05")])
fig.tight_layout()
savefig("22_hypothesis_test_summary.png")

# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("KEY STATISTICS SUMMARY")
log("=" * 70)

vf_mean_by_day = px_vf.groupby("data_day")["mid_price"].agg(["mean","std","min","max"])
log("\nVF Extract price stats by day:")
log(vf_mean_by_day.to_string())

hg_mean_by_day = px_hg.groupby("data_day")["mid_price"].agg(["mean","std","min","max"])
log("\nHYDROGEL price stats by day:")
log(hg_mean_by_day.to_string())

log("\nOption mean mid-prices by strike (all days):")
log(px_vev.groupby(["product","data_day"])["mid_price"].mean().unstack("data_day").to_string())

log("\nRealized volatility (ann.) per day:")
for day in DAYS:
    sub = px_vf[px_vf["data_day"] == day].sort_values("timestamp")
    lrets = np.log(sub["mid_price"]).diff().dropna()
    ann_vol = lrets.std() * np.sqrt(252 * 10_000)
    log(f"  Day {day}: σ_ann = {ann_vol:.4f}")

log("\nMedian IV by strike (all days combined):")
med_iv_all = iv_sample[iv_sample["IV"].notna() & (iv_sample["IV"] < 5)].groupby("strike")["IV"].median()
log(med_iv_all.to_string())

log("\nTrade count and volume by symbol:")
trade_summary = trades.groupby("symbol").agg(count=("quantity","count"), volume=("quantity","sum"),
                                              mean_price=("price","mean")).sort_values("volume",ascending=False)
log(trade_summary.to_string())

log("\nH7 — Negative time-value strikes:")
log(str(test_results.get("H7_TimeValue_Positive", {}).get("by_strike", {})))

# ══════════════════════════════════════════════════════════════════════════════
# Save log and test results
# ══════════════════════════════════════════════════════════════════════════════
log_path = os.path.join(OUT_DIR, "eda_summary.log")
with open(log_path, "w") as f:
    f.write("\n".join(LOG_LINES))
print(f"\n[saved] eda_summary.log")

results_path = os.path.join(OUT_DIR, "test_results.json")
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2, default=str)
print(f"[saved] test_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# Generate FINDINGS.md
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating FINDINGS.md …")

def _fmt(v, fmt=".4f", fallback="N/A"):
    """Safe format: returns fallback string if v is not a real number."""
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(fallback)

# Collect key numbers
vf_day0_mean = vf_mean_by_day.loc[0, "mean"]
vf_day2_mean = vf_mean_by_day.loc[2, "mean"]
vf_day0_std  = vf_mean_by_day.loc[0, "std"]
hg_day0_mean = hg_mean_by_day.loc[0, "mean"]
hg_day0_std  = hg_mean_by_day.loc[0, "std"]

h1_res = test_results.get("H1_HYDROGEL_ADF", {})
h2_res = test_results.get("H2_VF_ADF", {})
h3_res = test_results.get("H3_VF_JB_Normality", {})
h4a    = test_results.get("H4_IV_ANOVA_day0", {})
h5_res = test_results.get("H5_Poisson_Arrivals", {})
h7_res = test_results.get("H7_TimeValue_Positive", {})
h8_res = test_results.get("H8_LjungBox_Returns", {})

# Pre-format all values used in the f-string
h1_stat   = _fmt(h1_res.get("stat"))
h1_p      = _fmt(h1_res.get("p"), ".4g")
h1_interp = h1_res.get("interpretation", "N/A (statsmodels not installed)")
h1_action = "Mean-reversion strategy" if h1_res.get("reject_H0") else "Trend/neutral (or statsmodels missing)"
h2_stat   = _fmt(h2_res.get("stat"))
h2_p      = _fmt(h2_res.get("p"), ".4g")
h2_interp = h2_res.get("interpretation", "N/A")
h3_p      = _fmt(h3_res.get("p"), ".4g")
h3_interp = h3_res.get("interpretation", "N/A")
h4a_f     = _fmt(h4a.get("F"), ".3f")
h4a_p     = _fmt(h4a.get("p"), ".4g")
h4a_interp = h4a.get("interpretation", "N/A")
h5_p      = _fmt(h5_res.get("p"), ".4g")
h5_interp = h5_res.get("interpretation", "N/A")
h5_iat    = _fmt(h5_res.get("mean_iat"), ".0f")
h7_pct    = _fmt(h7_res.get("pct_negative"), ".2f")
h7_interp = h7_res.get("interpretation", "N/A")
h7_arb    = "Check arb opportunity" if h7_res.get("reject_H0") else "No arb"
h8_ret_p  = _fmt(h8_res.get("lb_ret_lag10_p"), ".4g")
h8_sq_p   = _fmt(h8_res.get("lb_sq_lag10_p"), ".4g")
h8_acorr  = "YES" if h8_res.get("autocorr_in_returns") else "NO"
h8_vol    = "YES" if h8_res.get("vol_clustering") else "NO"
h6a_res   = test_results.get("H6_Imbalance_Pred_VF_Extract", {})
h6b_res   = test_results.get("H6_Imbalance_Pred_HYDROGEL", {})
h6a_interp = h6a_res.get("interpretation", "N/A")
h6b_interp = h6b_res.get("interpretation", "N/A")

med_iv_str = "\n".join([f"  - Strike {int(k)}: IV ≈ {v:.3f}" for k, v in med_iv_all.items()])

# Pre-compute per-strike avg prices for table
def _mp(prod): return f"{px_vev_e[px_vev_e['product']==prod]['mid_price'].mean():.1f}"

findings_text = f"""# Round 3 EDA Findings — IMC Prosperity 4
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Products Overview

| Product | Type | Notes |
|---------|------|-------|
| `HYDROGEL_PACK` | Regular tradeable | Mean price ~{hg_day0_mean:.0f}, σ ≈ {hg_day0_std:.1f} |
| `VELVETFRUIT_EXTRACT` | Option underlying | Mean price ~{vf_day0_mean:.0f} (Day 0) → {vf_day2_mean:.0f} (Day 2) |
| `VEV_4000 … VEV_6500` | European call options | TTE = 7 on Day 0, 5 on Day 2 |

**TTE Mapping:** Data day 0 → TTE=7, day 1 → TTE=6, day 2 → TTE=5

---

## 2. HYDROGEL_PACK Analysis

- Mean price: **{hg_day0_mean:.2f}** across all days (highly stable)
- Spread: consistently narrow (1–2 ticks typical)
- **ADF test (H1):** stat={h1_stat}, p={h1_p}
  → **{h1_interp}**
- Implication: {h1_action}

---

## 3. VELVETFRUIT_EXTRACT (Underlying) Analysis

- Day 0 mean: **{vf_day0_mean:.2f}** ± {vf_day0_std:.2f}
- Day 2 mean: **{vf_day2_mean:.2f}**
- **ADF test (H2):** stat={h2_stat}, p={h2_p}
  → **{h2_interp}**
- **Normality (H3, Jarque-Bera):** p={h3_p}
  → **{h3_interp}** (fat tails / non-normal returns expected from bot activity)
- **Ljung-Box (H8):** returns p={h8_ret_p}, squared returns p={h8_sq_p}
  → Autocorrelation in returns: **{h8_acorr}**
  → Vol clustering: **{h8_vol}**

---

## 4. Options Analysis (VEV_*)

### 4.1 Option Price Structure (Day 0)

All options are **European calls** (vouchers to buy VF Extract at strike).

| Strike | Avg Mid | Status @ Day-0 |
|--------|---------|----------------|
| 4000 | ~{_mp('VEV_4000')} | Deep ITM (S-K≈{vf_day0_mean-4000:.0f}) |
| 4500 | ~{_mp('VEV_4500')} | Deep ITM (S-K≈{vf_day0_mean-4500:.0f}) |
| 5000 | ~{_mp('VEV_5000')} | Slightly ITM |
| 5100 | ~{_mp('VEV_5100')} | Near ATM |
| 5200 | ~{_mp('VEV_5200')} | Near ATM |
| 5300 | ~{_mp('VEV_5300')} | Near ATM |
| 5400 | ~{_mp('VEV_5400')} | OTM |
| 5500 | ~{_mp('VEV_5500')} | OTM |
| 6000 | ~0.5 | Deep OTM (floor price) |
| 6500 | ~0.5 | Deep OTM (floor price) |

### 4.2 Implied Volatility

Median implied vol by strike (all days, BS model with r=0, T in Solvenarian days/252):

{med_iv_str}

**IV Smile (H4):** ANOVA test Day 0: F={h4a_f}, p={h4a_p}
→ **{h4a_interp}**

Key observations:
- IV tends to be **higher for OTM options** than ATM (classic smile / right skew)
- VEV_6000 and VEV_6500 have **undefined/unreliable IV** (basically zero-value, no time value)
- IV varies across TTE — examine plot 11 for the term structure

### 4.3 Time Value Analysis (H7)

- **{h7_pct}%** of option snapshots show negative time value
- {h7_interp}
- Negative TV by strike: {h7_res.get('by_strike', {})}

---

## 5. Bot Behavior Analysis

- All `buyer`/`seller` fields in trade data are **NaN** — identity not disclosed
- Currency field = `XIRECS` (settlement currency)
- **Trade timing (H5):** KS test p={h5_p}
  → **{h5_interp}**
  Mean inter-trade interval: {h5_iat} ticks
- Trades are typically **market-aggressive** (at/near ask for buys, bid for sells)
- Trade volume is concentrated in **VELVETFRUIT_EXTRACT** and **VEV_5400/5500** (OTM options)

**Bot Behavioral Pattern:**
- Systematic trades appear every ~{h5_iat} ticks on average
- OTM option trades (VEV_5400, 5500) suggest bots speculate on upside breakouts
- Trade price vs mid-price analysis (plot 15): check if bots consistently buy at ask or sell at bid

---

## 6. Hypothesis Test Summary

| # | Hypothesis | Result | Action |
|---|-----------|--------|--------|
| H1 | HYDROGEL follows random walk | **{h1_interp}** | {h1_action} |
| H2 | VF Extract follows random walk | **{h2_interp}** | {'Mean-reversion' if h2_res.get('reject_H0') else 'Treat as GBM'} |
| H3 | VF returns are normal | **{h3_interp}** | Use heavier tails in model |
| H4 | IV is flat (no smile) | **{h4a_interp}** | Exploit smile — sell rich wings |
| H5 | Trade arrivals are Poisson | **{h5_interp}** | Consider queue/cluster effects |
| H6a | Imbalance predicts VF returns | **{h6a_interp}** | Use imbalance signal in quoting |
| H6b | Imbalance predicts HYDROGEL returns | **{h6b_interp}** | Use imbalance signal in quoting |
| H7 | Options priced above intrinsic | **{h7_interp}** | {h7_arb} |
| H8 | No autocorr in VF returns | Ret-p={h8_ret_p}, Sq-p={h8_sq_p} | {'Use momentum signals' if h8_res.get('autocorr_in_returns') else 'No momentum'} |

---

## 7. Strategy Recommendations

### Strategy A: HYDROGEL_PACK Market Making / Mean Reversion

```python
# Parameters to tune
FAIR_VALUE = {hg_day0_mean:.0f}   # or rolling mean
SPREAD = 2         # half-spread around fair value
MAX_POS = 50       # position limit

# Logic: quote bid at FAIR-SPREAD, ask at FAIR+SPREAD
# Skew quotes as position deviates from 0
```

- **Target:** Fair value ~{hg_day0_mean:.0f}
- **Signal:** Deviation from rolling mean → fade the move
- **Edge:** Bot activity creates predictable flow

### Strategy B: VF Extract Market Making

```python
# Parameters
FAIR_VALUE = rolling_mid   # track dynamically
MAX_POS = 50
SPREAD = 2–4 ticks
```

- If VF Extract is mean-reverting (H2 result), quote around rolling mean
- If it's a random walk, lean on trade imbalance signal (H6)

### Strategy C: Option Delta Hedging + IV Arbitrage

```python
# Core idea: Buy underpriced options, sell overpriced options
# Hedge delta by trading underlying

# Step 1: Compute IV for each strike
# Step 2: Identify strikes where IV deviates from median/theoretical curve
# Step 3: Trade options + hedge with VF Extract
```

- **Sell OTM options** (VEV_5500, VEV_6000, VEV_6500): collect premium,
  these expire worthless if S stays below strikes (historically likely)
- **Buy ATM options** when IV is low relative to realized vol

### Strategy D: Short Deep OTM Options (VEV_6000, VEV_6500)

- These trade at the floor price (0.5)
- Probability of expiring ITM appears very low (S ≈ {vf_day0_mean:.0f}, strikes 6000/6500)
- Risk: Black-swan spike in VF Extract to >6000
- **Recommendation:** Sell these at every opportunity (near-free money)
- **Max loss per unit:** 500 (if S hits 6500) — size carefully

### Strategy E: Options Portfolio Greeks Management

```python
# Maintain delta-neutral book:
# - Track net delta across all VEV positions
# - Use VF Extract to offset: trade -net_delta units of VF Extract

# Theta decay play:
# - Short options decay toward intrinsic as TTE → 0
# - Enter short positions early (high TTE) and ride time decay
```

---

## 8. Implementation Notes for Trader Class

### Position Limits (typical IMC Prosperity Round 3)
- Check official limits — likely 50–200 per product

### Key Computations Needed
```python
def compute_tte(day: int, timestamp: int) -> float:
    \"\"\"TTE in Solvenarian days.\"\"\"
    TTE_START = {{0: 7.0, 1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0}}
    return TTE_START.get(day, 7.0) - timestamp / 1_000_000

def bs_call(S, K, T, sigma, r=0):
    \"\"\"European call price. T in days.\"\"\"
    if T <= 0: return max(S - K, 0.0)
    T_yr = T / 252
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
    d2 = d1 - sigma*np.sqrt(T_yr)
    return S*norm.cdf(d1) - K*np.exp(-r*T_yr)*norm.cdf(d2)

def bs_delta(S, K, T, sigma, r=0):
    if T <= 0: return 1.0 if S > K else 0.0
    T_yr = T / 252
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
    return norm.cdf(d1)
```

### Priority Order of Strategies
1. **Short VEV_6000/VEV_6500** immediately — near-zero risk, free premium
2. **HYDROGEL mean reversion** — high confidence signal if ADF confirms stationarity
3. **VEV ATM option selling** — theta decay, hedge with VF Extract
4. **VF Extract market making** — if stationary, fade moves; if GBM, use imbalance signal

---

## 9. Plots Index

| File | Contents |
|------|----------|
| 01_midprice_overview.png | All products mid-price across 3 days |
| 02_hydrogel_analysis.png | HYDROGEL: timeseries, distribution, spread, returns |
| 03_vf_extract_timeseries.png | VF Extract with trade overlays |
| 04_vf_extract_returns.png | Returns analysis + ACF of returns and squared returns |
| 05_rolling_volatility.png | Rolling annualised vol for VF Extract |
| 06_option_price_timeseries.png | All VEV option prices (log scale + linear zoom) |
| 07_intrinsic_time_value.png | Intrinsic vs time value by strike, per day |
| 08_moneyness_analysis.png | Moneyness (S/K) over time |
| 09_iv_smile.png | IV smile for each day (median ± IQR) |
| 10_iv_surface.png | IV surface heatmap (strike × time) |
| 11_iv_vs_tte.png | ATM IV vs TTE |
| 12_spread_analysis.png | Bid-ask spreads all products |
| 13_orderbook_depth_imbalance.png | Depth and imbalance for VF Extract + HYDROGEL |
| 14_trade_flow_timing.png | Trade counts, volume, inter-arrival times |
| 15_bot_trade_vs_mid.png | Trade prices relative to mid-price |
| 16_vwap_analysis.png | VWAP vs mid for VF Extract and HYDROGEL |
| 17_imbalance_predictive.png | Order imbalance → future return scatter |
| 18_qq_plots.png | QQ normality plots |
| 19_option_deltas.png | BS delta by strike over time |
| 20_pnl_scenarios.png | Short option P&L at day-2 median S |
| 21_correlation_matrix.png | Full cross-product return correlations |
| 22_hypothesis_test_summary.png | All hypothesis test p-values |

---

*Generated by round3_eda.py — IMC Prosperity 4 Round 3*
"""

findings_path = os.path.join(OUT_DIR, "FINDINGS.md")
with open(findings_path, "w") as f:
    f.write(findings_text)
print(f"[saved] FINDINGS.md")

print("\n" + "=" * 70)
print("EDA COMPLETE — all outputs in eda_output/")
print("=" * 70)
