"""
Prosperity 4 - Round 1 EDA
Products: ASH_COATED_OSMIUM (ASH), INTARIAN_PEPPER_ROOT (PEPPER)
Days: -2, -1, 0
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import periodogram

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "round1-data", "ROUND1")
OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = [-2, -1, 0]
PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
SHORT = {"ASH_COATED_OSMIUM": "ASH", "INTARIAN_PEPPER_ROOT": "PEPPER"}
COLORS = {"ASH_COATED_OSMIUM": "#2196F3", "INTARIAN_PEPPER_ROOT": "#FF5722"}
DAY_COLORS = {-2: "#1b7837", -1: "#762a83", 0: "#d6604d"}

# ─── Load Data ────────────────────────────────────────────────────────────────

def load_prices():
    frames = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_1_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    return df

def load_trades():
    frames = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"trades_round_1_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

prices_all = load_prices()
trades_all = load_trades()

# ─── Summary Stats ─────────────────────────────────────────────────────────────

print("=" * 70)
print("PROSPERITY 4 ROUND 1 — EDA STATS")
print("=" * 70)

print(f"\nPrices shape: {prices_all.shape}")
print(f"Trades shape: {trades_all.shape}")
print(f"\nPrices columns: {list(prices_all.columns)}")
print(f"Trades columns: {list(trades_all.columns)}")

for prod in PRODUCTS:
    p = prices_all[prices_all["product"] == prod]
    t = trades_all[trades_all["symbol"] == prod] if "symbol" in trades_all.columns else pd.DataFrame()
    print(f"\n{'─'*60}")
    print(f"PRODUCT: {prod}  ({SHORT[prod]})")
    print(f"{'─'*60}")
    print(f"  Price rows: {len(p)}")
    print(f"  Trade rows: {len(t)}")
    print(f"\n  --- Mid Price Stats (all days) ---")
    print(p["mid_price"].describe().to_string())
    print(f"\n  --- Per-Day Mid Price Stats ---")
    for day in DAYS:
        pd_ = p[p["day"] == day]["mid_price"]
        print(f"  Day {day:2d}: mean={pd_.mean():.2f}  std={pd_.std():.4f}  "
              f"min={pd_.min():.2f}  max={pd_.max():.2f}  range={pd_.max()-pd_.min():.2f}")

    if len(t) > 0:
        print(f"\n  --- Trade Price Stats (all days) ---")
        print(t["price"].describe().to_string())
        print(f"\n  --- Trade Quantity Stats ---")
        print(t["quantity"].describe().to_string())
        print(f"\n  --- Traders/buyers/sellers ---")
        if "buyer" in t.columns:
            print("  Buyers:", t["buyer"].value_counts().to_dict())
        if "seller" in t.columns:
            print("  Sellers:", t["seller"].value_counts().to_dict())

    # Spread analysis
    p2 = p.copy()
    p2["spread"] = p2["ask_price_1"] - p2["bid_price_1"]
    valid_spread = p2[p2["spread"].notna() & (p2["spread"] > 0)]
    if len(valid_spread) > 0:
        print(f"\n  --- Bid-Ask Spread (bid_price_1 vs ask_price_1) ---")
        print(valid_spread["spread"].describe().to_string())
        for day in DAYS:
            ds = valid_spread[valid_spread["day"] == day]["spread"]
            if len(ds) > 0:
                print(f"  Day {day:2d}: mean spread={ds.mean():.2f}  median={ds.median():.2f}")

    # Returns
    for day in DAYS:
        pd_ = p[p["day"] == day]["mid_price"].dropna()
        if len(pd_) > 1:
            rets = pd_.diff().dropna()
            print(f"\n  Day {day:2d} Returns (diff mid_price):")
            print(f"    mean={rets.mean():.4f}  std={rets.std():.4f}  "
                  f"skew={rets.skew():.4f}  kurt={rets.kurtosis():.4f}")
            _, pval = stats.shapiro(rets.sample(min(len(rets), 5000), random_state=42))
            print(f"    Shapiro-Wilk normality p={pval:.4e}  (normal if p>0.05)")

    # Autocorrelation of returns
    for day in DAYS:
        pd_ = p[p["day"] == day]["mid_price"].dropna()
        if len(pd_) > 10:
            rets = pd_.diff().dropna()
            ac1 = rets.autocorr(lag=1)
            ac2 = rets.autocorr(lag=2)
            ac5 = rets.autocorr(lag=5)
            print(f"\n  Day {day:2d} Return Autocorr: lag1={ac1:.4f}  lag2={ac2:.4f}  lag5={ac5:.4f}")

print("\n\n")

# ─── Order Book Depth Analysis ─────────────────────────────────────────────────
for prod in PRODUCTS:
    p = prices_all[prices_all["product"] == prod].copy()
    print(f"\n{'─'*60}")
    print(f"ORDER BOOK DEPTH: {prod}")
    print(f"{'─'*60}")

    for level in [1, 2, 3]:
        bv_col = f"bid_volume_{level}"
        av_col = f"ask_volume_{level}"
        bp_col = f"bid_price_{level}"
        ap_col = f"ask_price_{level}"
        bv = p[bv_col].dropna()
        av = p[av_col].dropna()
        bp = p[bp_col].dropna()
        ap = p[ap_col].dropna()
        print(f"  Level {level}: bid_price presence={len(bp)/len(p)*100:.1f}%  "
              f"bid_vol mean={bv.mean():.1f}  ask_vol mean={av.mean():.1f}")

    # Weighted mid price
    p2 = p.copy()
    mask = p2["bid_price_1"].notna() & p2["ask_price_1"].notna() & \
           p2["bid_volume_1"].notna() & p2["ask_volume_1"].notna()
    sub = p2[mask].copy()
    if len(sub) > 0:
        sub["wmid"] = (sub["bid_price_1"] * sub["ask_volume_1"] +
                       sub["ask_price_1"] * sub["bid_volume_1"]) / \
                      (sub["bid_volume_1"] + sub["ask_volume_1"])
        sub["wmid_vs_mid"] = sub["wmid"] - sub["mid_price"]
        print(f"\n  Weighted mid vs raw mid: mean diff={sub['wmid_vs_mid'].mean():.4f}  "
              f"std={sub['wmid_vs_mid'].std():.4f}")

    # Order imbalance
    mask2 = p2["bid_volume_1"].notna() & p2["ask_volume_1"].notna()
    sub2 = p2[mask2].copy()
    if len(sub2) > 0:
        sub2["imbalance"] = (sub2["bid_volume_1"] - sub2["ask_volume_1"]) / \
                             (sub2["bid_volume_1"] + sub2["ask_volume_1"])
        print(f"\n  Order Imbalance (L1): mean={sub2['imbalance'].mean():.4f}  "
              f"std={sub2['imbalance'].std():.4f}")
        for day in DAYS:
            ds = sub2[sub2["day"] == day]["imbalance"]
            print(f"  Day {day:2d}: imb mean={ds.mean():.4f}  std={ds.std():.4f}")

print("\n\n")

# ─── Trend Analysis (PEPPER) ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("TREND ANALYSIS — INTARIAN_PEPPER_ROOT")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "INTARIAN_PEPPER_ROOT"].copy()
for day in DAYS:
    pd_ = p[p["day"] == day].copy().reset_index(drop=True)
    mid = pd_["mid_price"].dropna()
    if len(mid) > 10:
        x = np.arange(len(mid))
        slope, intercept, r, pval, se = stats.linregress(x, mid)
        print(f"  Day {day:2d}: slope={slope:.6f}/tick  "
              f"total_drift={slope*(len(mid)-1):.2f}  R²={r**2:.4f}  p={pval:.4e}")
        # Normalized slope (per 1000 ticks)
        print(f"          slope per 1000 ticks = {slope*1000:.4f}")

print(f"\n{'─'*60}")
print("TREND ANALYSIS — ASH_COATED_OSMIUM")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "ASH_COATED_OSMIUM"].copy()
for day in DAYS:
    pd_ = p[p["day"] == day].copy().reset_index(drop=True)
    mid = pd_["mid_price"].dropna()
    if len(mid) > 10:
        x = np.arange(len(mid))
        slope, intercept, r, pval, se = stats.linregress(x, mid)
        print(f"  Day {day:2d}: slope={slope:.6f}/tick  "
              f"total_drift={slope*(len(mid)-1):.2f}  R²={r**2:.4f}  p={pval:.4e}")

print("\n\n")

# ─── Mean Reversion Test (ASH) ─────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("MEAN REVERSION — ASH_COATED_OSMIUM")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "ASH_COATED_OSMIUM"].copy()
for day in DAYS:
    pd_ = p[p["day"] == day]["mid_price"].dropna()
    if len(pd_) > 10:
        # ADF test proxy: regress delta on lagged level
        y = pd_.values
        dy = np.diff(y)
        y_lag = y[:-1]
        slope, intercept, r, pval, se = stats.linregress(y_lag, dy)
        print(f"  Day {day:2d}: delta ~ lag coeff={slope:.6f}  p={pval:.4e}  "
              f"(negative & sig => mean-reverting)")
        print(f"          implied half-life = {-np.log(2)/slope:.1f} ticks"
              if slope < 0 else "          no mean reversion detected")

print(f"\n{'─'*60}")
print("MEAN REVERSION — INTARIAN_PEPPER_ROOT (detrended)")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "INTARIAN_PEPPER_ROOT"].copy()
for day in DAYS:
    pd_ = p[p["day"] == day].copy().reset_index(drop=True)
    mid = pd_["mid_price"].dropna()
    if len(mid) > 10:
        x = np.arange(len(mid))
        slope_t, intercept_t, _, _, _ = stats.linregress(x, mid)
        detrended = mid.values - (slope_t * x + intercept_t)
        dy = np.diff(detrended)
        y_lag = detrended[:-1]
        slope, intercept, r, pval, se = stats.linregress(y_lag, dy)
        print(f"  Day {day:2d}: detrended delta ~ lag coeff={slope:.6f}  p={pval:.4e}")
        if slope < 0:
            print(f"          implied half-life = {-np.log(2)/slope:.1f} ticks")

print("\n\n")

# ─── Fair Value Analysis ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("FAIR VALUE ANALYSIS — ASH_COATED_OSMIUM")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "ASH_COATED_OSMIUM"]["mid_price"]
print(f"  Overall mean: {p.mean():.4f}")
print(f"  Overall median: {p.median():.4f}")
print(f"  % time within 5 of 10000: {(abs(p - 10000) <= 5).mean()*100:.1f}%")
print(f"  % time within 10 of 10000: {(abs(p - 10000) <= 10).mean()*100:.1f}%")
print(f"  % time within 20 of 10000: {(abs(p - 10000) <= 20).mean()*100:.1f}%")
bins_near_10k = prices_all[prices_all["product"]=="ASH_COATED_OSMIUM"]["mid_price"]
hist, edges = np.histogram(bins_near_10k.dropna(), bins=50)
mode_idx = np.argmax(hist)
print(f"  Mode bin center: {(edges[mode_idx]+edges[mode_idx+1])/2:.1f}")

print(f"\n{'─'*60}")
print("FAIR VALUE — PEPPER (linear trend per day)")
print(f"{'─'*60}")
p = prices_all[prices_all["product"] == "INTARIAN_PEPPER_ROOT"].copy()
for day in DAYS:
    pd_ = p[p["day"] == day].copy().reset_index(drop=True)
    mid = pd_["mid_price"].dropna()
    x = pd_["timestamp"][mid.index]
    slope, intercept, r, pval, se = stats.linregress(x, mid)
    at_0 = intercept
    at_end = intercept + slope * x.max()
    print(f"  Day {day:2d}: start≈{at_0:.2f}  end≈{at_end:.2f}  slope={slope:.6f}/ts  R²={r**2:.4f}")

# Cross-day continuity for pepper
print(f"\n  --- Cross-day price continuity (PEPPER) ---")
last_mids = {}
first_mids = {}
for day in DAYS:
    pd_ = p[p["day"] == day]["mid_price"].dropna()
    if len(pd_) > 0:
        last_mids[day] = pd_.iloc[-1]
        first_mids[day] = pd_.iloc[0]
        print(f"  Day {day:2d}: first mid={first_mids[day]:.2f}  last mid={last_mids[day]:.2f}")

print("\n\n")

# ─── Trade Analysis ───────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("TRADE ANALYSIS")
print(f"{'─'*60}")
trades_all2 = trades_all.rename(columns={"symbol": "product"}) if "symbol" in trades_all.columns else trades_all

for prod in PRODUCTS:
    t = trades_all2[trades_all2["product"] == prod]
    if len(t) == 0:
        continue
    print(f"\n  {SHORT[prod]}:")
    print(f"    Total trades: {len(t)}")
    print(f"    Avg trade size: {t['quantity'].mean():.2f}")
    print(f"    Trade size distribution: {t['quantity'].describe().to_dict()}")
    for day in DAYS:
        td = t[t["day"] == day]
        print(f"    Day {day:2d}: {len(td)} trades  avg_size={td['quantity'].mean():.2f}  "
              f"total_vol={td['quantity'].sum():.0f}")
    if "buyer" in t.columns:
        buyers = t["buyer"].fillna("MARKET").value_counts()
        print(f"    Top buyers: {buyers.head(5).to_dict()}")
    if "seller" in t.columns:
        sellers = t["seller"].fillna("MARKET").value_counts()
        print(f"    Top sellers: {sellers.head(5).to_dict()}")

    # Trade price vs mid price
    prices_prod = prices_all[prices_all["product"] == prod][["day", "timestamp", "mid_price"]]
    t2 = t.copy()
    day_merged = []
    for d in DAYS:
        t_d = t2[t2["day"] == d].sort_values("timestamp").reset_index(drop=True)
        p_d = prices_prod[prices_prod["day"] == d].sort_values("timestamp").reset_index(drop=True)
        if len(t_d) > 0 and len(p_d) > 0:
            m = pd.merge_asof(t_d, p_d, on="timestamp")
            day_merged.append(m)
    if day_merged:
        merged = pd.concat(day_merged, ignore_index=True)
        if "mid_price" in merged.columns:
            merged["trade_vs_mid"] = merged["price"] - merged["mid_price"]
            print(f"    Trade price vs mid: mean={merged['trade_vs_mid'].mean():.4f}  "
                  f"std={merged['trade_vs_mid'].std():.4f}")

print("\n\n")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
print("Generating plots...")

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

# ── Fig 1: Mid price timeseries for both products, all days ──────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
for ax, prod in zip(axes, PRODUCTS):
    p = prices_all[prices_all["product"] == prod]
    offset = 0
    for day in DAYS:
        pd_ = p[p["day"] == day].copy()
        ax.plot(pd_["global_ts"], pd_["mid_price"],
                color=DAY_COLORS[day], alpha=0.85, lw=0.7, label=f"Day {day}")
    ax.set_title(f"{prod} — Mid Price (all days)", fontsize=11)
    ax.set_xlabel("Global Timestamp")
    ax.set_ylabel("Mid Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Mid Price Timeseries", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "01_midprice_timeseries.png")

# ── Fig 2: Mid price distribution (histogram + KDE) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, prod in zip(axes, PRODUCTS):
    p = prices_all[prices_all["product"] == prod]["mid_price"].dropna()
    ax.hist(p, bins=80, density=True, alpha=0.65, color=COLORS[prod], edgecolor="white", lw=0.3)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(p)
    xs = np.linspace(p.min(), p.max(), 400)
    ax.plot(xs, kde(xs), color="black", lw=1.5)
    ax.axvline(p.mean(), color="red", ls="--", lw=1, label=f"mean={p.mean():.2f}")
    ax.axvline(p.median(), color="orange", ls=":", lw=1, label=f"median={p.median():.2f}")
    ax.set_title(f"{SHORT[prod]} — Mid Price Distribution", fontsize=11)
    ax.set_xlabel("Mid Price")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Mid Price Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "02_midprice_distributions.png")

# ── Fig 3: Returns distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]["mid_price"].dropna()
        rets = p_sub.diff().dropna()
        ax.hist(rets, bins=60, density=True, alpha=0.7, color=COLORS[prod], edgecolor="white", lw=0.3)
        # Normal overlay
        mu, sigma = rets.mean(), rets.std()
        xs = np.linspace(rets.min(), rets.max(), 300)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="black", lw=1.5, label="Normal fit")
        ax.set_title(f"{SHORT[prod]} Day {day} Returns", fontsize=9)
        ax.set_xlabel("Δ Mid Price")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Returns Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "03_returns_distributions.png")

# ── Fig 4: Spread analysis ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy()
        p_sub["spread"] = p_sub["ask_price_1"] - p_sub["bid_price_1"]
        valid = p_sub["spread"].dropna()
        valid = valid[valid > 0]
        if len(valid) > 0:
            ax.hist(valid, bins=40, color=COLORS[prod], alpha=0.75, edgecolor="white", lw=0.3)
            ax.axvline(valid.mean(), color="red", ls="--", lw=1.2, label=f"mean={valid.mean():.2f}")
            ax.axvline(valid.median(), color="orange", ls=":", lw=1.2, label=f"median={valid.median():.2f}")
        ax.set_title(f"{SHORT[prod]} Day {day} — Bid-Ask Spread", fontsize=9)
        ax.set_xlabel("Spread")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Bid-Ask Spread Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "04_spread_distributions.png")

# ── Fig 5: Book imbalance over time ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy()
        mask = p_sub["bid_volume_1"].notna() & p_sub["ask_volume_1"].notna()
        sub = p_sub[mask].copy()
        if len(sub) > 0:
            sub["imb"] = (sub["bid_volume_1"] - sub["ask_volume_1"]) / \
                         (sub["bid_volume_1"] + sub["ask_volume_1"])
            ax.plot(sub["timestamp"], sub["imb"], lw=0.5, alpha=0.7, color=COLORS[prod])
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.axhline(sub["imb"].mean(), color="red", lw=0.8, ls=":", label=f"mean={sub['imb'].mean():.3f}")
        ax.set_title(f"{SHORT[prod]} Day {day} — Order Imbalance", fontsize=9)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Imbalance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — L1 Order Book Imbalance", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "05_order_imbalance.png")

# ── Fig 6: PEPPER detrended + rolling mean ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
prod = "INTARIAN_PEPPER_ROOT"
for ax, day in zip(axes, DAYS):
    p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy().reset_index(drop=True)
    mid = p_sub["mid_price"].dropna()
    x = p_sub["timestamp"][mid.index].values
    slope_t, intercept_t, _, _, _ = stats.linregress(x, mid.values)
    trend_line = slope_t * x + intercept_t
    detrended = mid.values - trend_line

    ax.plot(x, detrended, lw=0.6, alpha=0.8, color=COLORS[prod], label="Detrended mid")
    roll_window = 200
    roll_mean = pd.Series(detrended).rolling(roll_window, center=True).mean()
    ax.plot(x, roll_mean, lw=1.5, color="black", label=f"Rolling mean (w={roll_window})")
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_title(f"PEPPER Day {day} — Detrended Mid Price (slope={slope_t:.6f}/ts)", fontsize=10)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Detrended Price")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — PEPPER Detrended Price", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "06_pepper_detrended.png")

# ── Fig 7: ASH mid price + fair value line ────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
prod = "ASH_COATED_OSMIUM"
for ax, day in zip(axes, DAYS):
    p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy()
    ax.plot(p_sub["timestamp"], p_sub["mid_price"], lw=0.6, alpha=0.9, color=COLORS[prod], label="Mid price")
    ax.axhline(10000, color="red", ls="--", lw=1.2, label="Fair value=10000")
    roll = p_sub["mid_price"].rolling(500, center=True).mean()
    ax.plot(p_sub["timestamp"], roll, lw=1.2, color="black", alpha=0.7, label="Rolling mean (w=500)")
    ax.set_title(f"ASH Day {day}", fontsize=10)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Mid Price")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — ASH Mid Price vs Fair Value", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "07_ash_midprice_fairvalue.png")

# ── Fig 8: Autocorrelation of returns ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
max_lag = 50
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]["mid_price"].dropna()
        rets = p_sub.diff().dropna()
        lags = range(1, max_lag + 1)
        acf_vals = [rets.autocorr(lag=l) for l in lags]
        conf = 1.96 / np.sqrt(len(rets))
        ax.bar(list(lags), acf_vals, color=COLORS[prod], alpha=0.7)
        ax.axhline(conf, color="red", ls="--", lw=0.8, label=f"95% CI (±{conf:.3f})")
        ax.axhline(-conf, color="red", ls="--", lw=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{SHORT[prod]} Day {day} — Return ACF", fontsize=9)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Return Autocorrelation", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "08_return_autocorrelation.png")

# ── Fig 9: Trade activity ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
trades_prod = trades_all.rename(columns={"symbol": "product"}) if "symbol" in trades_all.columns else trades_all
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        t_sub = trades_prod[(trades_prod["product"] == prod) & (trades_prod["day"] == day)]
        if len(t_sub) > 0:
            ax.scatter(t_sub["timestamp"], t_sub["price"], s=t_sub["quantity"]*1.5,
                       alpha=0.5, color=COLORS[prod], edgecolors="none")
            # overlay mid price
            p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]
            ax.plot(p_sub["timestamp"], p_sub["mid_price"], lw=0.5, color="black", alpha=0.6, label="Mid price")
        ax.set_title(f"{SHORT[prod]} Day {day} — Trades", fontsize=9)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Trade Activity (bubble size = quantity)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "09_trade_activity.png")

# ── Fig 10: QQ plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]["mid_price"].dropna()
        rets = p_sub.diff().dropna()
        (osm, osr), (slope, intercept, r) = stats.probplot(rets, dist="norm")
        ax.scatter(osm, osr, s=3, alpha=0.5, color=COLORS[prod])
        ax.plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.5)
        ax.set_title(f"{SHORT[prod]} Day {day} — QQ Plot (returns)", fontsize=9)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — QQ Plots of Returns", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "10_qq_plots.png")

# ── Fig 11: Rolling volatility ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
window = 200
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy()
        p_sub["ret"] = p_sub["mid_price"].diff()
        p_sub["roll_vol"] = p_sub["ret"].rolling(window).std()
        ax.plot(p_sub["timestamp"], p_sub["roll_vol"], lw=0.7, color=COLORS[prod])
        ax.axhline(p_sub["ret"].std(), color="red", ls="--", lw=0.8, label=f"Overall std={p_sub['ret'].std():.3f}")
        ax.set_title(f"{SHORT[prod]} Day {day} — Rolling Vol (w={window})", fontsize=9)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Rolling Std of Returns")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Rolling Volatility", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "11_rolling_volatility.png")

# ── Fig 12: PEPPER trend across all days ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
prod = "INTARIAN_PEPPER_ROOT"
p = prices_all[prices_all["product"] == prod].copy()
ax.plot(p["global_ts"], p["mid_price"], lw=0.5, color=COLORS[prod], alpha=0.8, label="Mid price")
# Fit overall linear trend
x_all = p["global_ts"].values
y_all = p["mid_price"].values
slope_all, intercept_all, _, _, _ = stats.linregress(x_all, y_all)
trend_vals = slope_all * x_all + intercept_all
ax.plot(x_all, trend_vals, color="black", lw=2, ls="--",
        label=f"Overall trend (slope={slope_all:.6f}/ts)")
for day in DAYS:
    p_d = p[p["day"] == day]
    ax.axvline(p_d["global_ts"].min(), color=DAY_COLORS[day], ls=":", lw=1.0,
               label=f"Day {day} start")
ax.set_title("PEPPER — Full Trend Across All Days", fontsize=11)
ax.set_xlabel("Global Timestamp")
ax.set_ylabel("Mid Price")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "12_pepper_full_trend.png")

# ── Fig 13: Volume-weighted price levels ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, prod in zip(axes, PRODUCTS):
    t = trades_prod[trades_prod["product"] == prod]
    if len(t) == 0:
        ax.set_title(f"{SHORT[prod]} — No trades")
        continue
    # VWAP
    vwap = (t["price"] * t["quantity"]).sum() / t["quantity"].sum()
    ax.hist(t["price"], bins=50, weights=t["quantity"], density=True,
            color=COLORS[prod], alpha=0.7, edgecolor="white", lw=0.3, label="Volume-weighted")
    ax.axvline(vwap, color="red", lw=1.5, ls="--", label=f"VWAP={vwap:.2f}")
    ax.set_title(f"{SHORT[prod]} — Volume-Weighted Price Distribution", fontsize=10)
    ax.set_xlabel("Trade Price")
    ax.set_ylabel("Vol-Weighted Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Trade VWAP Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "13_vwap_analysis.png")

# ── Fig 14: Bid/Ask levels (all 3 levels) for each product ───────────────────
for prod in PRODUCTS:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    for ax, day in zip(axes, DAYS):
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]
        ax.plot(p_sub["timestamp"], p_sub["mid_price"], lw=0.7, color="black", label="Mid", zorder=5)
        for level, (bc, ac) in enumerate(zip(["#1565C0","#1E88E5","#64B5F6"],
                                              ["#B71C1C","#E53935","#EF9A9A"]), 1):
            bp_col = f"bid_price_{level}"
            ap_col = f"ask_price_{level}"
            if bp_col in p_sub.columns:
                ax.plot(p_sub["timestamp"], p_sub[bp_col], lw=0.4, alpha=0.6,
                        color=bc, label=f"Bid L{level}")
            if ap_col in p_sub.columns:
                ax.plot(p_sub["timestamp"], p_sub[ap_col], lw=0.4, alpha=0.6,
                        color=ac, label=f"Ask L{level}")
        ax.set_title(f"{SHORT[prod]} Day {day} — Order Book Levels", fontsize=9)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Price")
        ax.legend(fontsize=6, ncol=4)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Prosperity 4 Round 1 — {prod} Order Book Depth", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, f"14_orderbook_depth_{SHORT[prod]}.png")

# ── Fig 15: Imbalance vs next return (predictive signal) ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for row, prod in enumerate(PRODUCTS):
    for col, day in enumerate(DAYS):
        ax = axes[row][col]
        p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)].copy()
        mask = p_sub["bid_volume_1"].notna() & p_sub["ask_volume_1"].notna()
        sub = p_sub[mask].copy().reset_index(drop=True)
        if len(sub) < 50:
            continue
        sub["imb"] = (sub["bid_volume_1"] - sub["ask_volume_1"]) / \
                     (sub["bid_volume_1"] + sub["ask_volume_1"])
        sub["next_ret"] = sub["mid_price"].shift(-1) - sub["mid_price"]
        sub = sub.dropna(subset=["imb", "next_ret"])
        if len(sub) < 20:
            continue
        # Bin imbalance
        sub["imb_bin"] = pd.cut(sub["imb"], bins=10)
        bin_means = sub.groupby("imb_bin")["next_ret"].mean()
        ax.bar(range(len(bin_means)), bin_means.values, color=COLORS[prod], alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        slope_i, intercept_i, r_i, pval_i, _ = stats.linregress(sub["imb"], sub["next_ret"])
        ax.set_title(f"{SHORT[prod]} Day {day} — Imbalance vs Next Return\n"
                     f"r={r_i:.3f} p={pval_i:.3e}", fontsize=8)
        ax.set_xlabel("Imbalance Bin")
        ax.set_ylabel("Mean Next Return")
        ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — Order Imbalance Predictive Power", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "15_imbalance_predictive.png")

# ── Fig 16: ASH deviation from 10000 distribution ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
prod = "ASH_COATED_OSMIUM"
for ax, day in zip(axes, DAYS):
    p_sub = prices_all[(prices_all["product"] == prod) & (prices_all["day"] == day)]["mid_price"].dropna()
    dev = p_sub - 10000
    ax.hist(dev, bins=60, density=True, color=COLORS[prod], alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="0 deviation")
    ax.axvline(dev.mean(), color="orange", lw=1.2, ls=":", label=f"mean={dev.mean():.2f}")
    ax.set_title(f"ASH Day {day} — Deviation from 10000", fontsize=10)
    ax.set_xlabel("Mid Price - 10000")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Prosperity 4 Round 1 — ASH Deviation from Fair Value 10000", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "16_ash_deviation_fairvalue.png")

print(f"\nAll plots saved to: {OUT_DIR}/")
print("Done.")
