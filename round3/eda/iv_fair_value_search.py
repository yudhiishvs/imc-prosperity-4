"""
IV Fair-Value Binary Search
============================
1. Fit a parabola IV(m) = a*m^2 + b*m + c  where m = log(S/K)/sqrt(T)
   to the observed median IVs across the 3 days.
2. Backtest(S_fair): at each tick, for every active strike compute
   model_price = BS(S_fair, K, T, IV_parabola(m)); if model > ask → buy,
   if model < bid → sell; mark residual positions to mid at day-end.
   Returns total PnL.
3. Binary search S_fair in the "pipe chamber" flat region [4500, 5500]
   to maximize PnL.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "ROUND_3")
sys.path.insert(0, os.path.join(HERE, "..", "dashboard"))

# ── Constants ─────────────────────────────────────────────────────────────────
DAYS         = [0, 1, 2]
TTE_START    = {0: 7.0, 1: 6.0, 2: 5.0}
UNDERLYING   = "VELVETFRUIT_EXTRACT"
ALL_STRIKES  = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
# Near-ATM only — deep ITM (4000) and far OTM (6000, 6500) distort the smile fit
SMILE_STRIKES = [4500, 5000, 5100, 5200, 5300, 5400, 5500]
ACTIVE_STRIKES = [4500, 5000, 5100, 5200, 5300, 5400, 5500]  # traded in sim

POS_CAP    = 50     # per-strike position limit
TRADE_SIZE = 10     # units per signal
EOD_BUFFER = 50_000 # ticks before day-end to stop new positions

# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs_call(S, K, T, sigma, r=0.0):
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0:
        return max(S - K, 0.0)
    T_yr = T / 252.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
    d2 = d1 - sigma * np.sqrt(T_yr)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T_yr) * norm.cdf(d2))

def bs_iv(C_mkt, S, K, T, r=0.0):
    intrinsic = max(S - K, 0.0)
    if C_mkt <= intrinsic + 1e-6 or T <= 1e-9 or S <= 0:
        return np.nan
    C_mkt = max(C_mkt, intrinsic + 1e-6)
    def obj(s): return bs_call(S, K, T, s, r) - C_mkt
    try:
        lo, hi = obj(1e-4), obj(10.0)
        if np.sign(lo) == np.sign(hi):
            return np.nan
        return brentq(obj, 1e-4, 10.0, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan

# ── Data loading ──────────────────────────────────────────────────────────────
def load_prices():
    frames = []
    for day in DAYS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"prices_round_3_day_{day}.csv"), sep=";")
        df.columns = df.columns.str.strip()
        df["data_day"] = day
        df["TTE"] = TTE_START[day] - df["timestamp"] / 1_000_000
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df["mid_price"] > 0].copy()

prices = load_prices()
px_vf  = prices[prices["product"] == UNDERLYING].copy()
px_vev = prices[prices["product"].str.startswith("VEV_")].copy()
px_vev["strike"] = px_vev["product"].str.replace("VEV_", "").astype(float)

# Merge nearest underlying price
vev_parts = []
for day in DAYS:
    sub_vev = px_vev[px_vev["data_day"] == day].copy()
    sub_vf  = (px_vf[px_vf["data_day"] == day][["timestamp","mid_price"]]
               .rename(columns={"mid_price":"S"}).sort_values("timestamp"))
    sub_vev = sub_vev.sort_values("timestamp")
    vev_parts.append(pd.merge_asof(sub_vev, sub_vf, on="timestamp", direction="nearest"))
px_vev_e = pd.concat(vev_parts, ignore_index=True)

print(f"Loaded {len(px_vev_e):,} option ticks across {len(DAYS)} days.")

# ── Compute IVs using the observed S ─────────────────────────────────────────
print("Computing IVs …")
sample = px_vev_e[px_vev_e["strike"].isin(SMILE_STRIKES)].iloc[::10].copy()

def compute_iv_row(r):
    return bs_iv(r["mid_price"], r["S"], r["strike"], r["TTE"])

sample["IV"] = sample.apply(compute_iv_row, axis=1)
sample["m"]  = sample.apply(
    lambda r: np.log(r["S"]/r["strike"])/np.sqrt(max(r["TTE"],0.01))
    if r["S"] > 0 and r["strike"] > 0 else np.nan, axis=1)
sample = sample[sample["IV"].notna() & (sample["IV"] < 5)].copy()

# ── Fit parabola  IV = a*m^2 + b*m + c ───────────────────────────────────────
# Fit to all days combined (single global smile)
valid = sample.dropna(subset=["m","IV"])
coeffs = np.polyfit(valid["m"], valid["IV"], 2)
a, b, c = coeffs
print(f"\nGlobal parabola  IV(m) = {a:.4f}·m² + {b:.4f}·m + {c:.4f}")
print(f"Vertex (ATM) at m* = {-b/(2*a):.4f}  (=0 means parabola centred at ATM)")

# The m-vertex corresponds to K* = S * exp(-sqrt(T) * b/(2a))
# Using median TTE and median S as reference:
med_S   = px_vf["mid_price"].median()
med_T   = px_vev_e["TTE"].median()
m_star  = -b / (2*a)
K_star  = med_S * np.exp(-np.sqrt(med_T) * m_star)
print(f"Using median S={med_S:.1f}, median TTE={med_T:.2f}:")
print(f"IV minimum at strike K* ≈ {K_star:.0f}")
print(f"=> Initial S_fair guess: {K_star:.0f}  (the parabola minimum in strike space)")

def parabola_iv(S_fair, K, T):
    """Parabolic IV at (S_fair, K, T) using fitted coefficients."""
    T_safe = max(T, 0.01)
    m = np.log(S_fair / K) / np.sqrt(T_safe)
    iv = a * m**2 + b * m + c
    return max(iv, 0.005)

# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(S_fair, verbose=False):
    """
    Simulate options MM using parabola IV centered at S_fair.
    Returns total PnL (fill improvement + EOD mark-to-mid).
    """
    total_pnl = 0.0

    for day in DAYS:
        pos    = {K: 0 for K in ACTIVE_STRIKES}   # positions
        cash   = {K: 0.0 for K in ACTIVE_STRIKES}
        eod_ts = TTE_START[day] * 1_000_000 - EOD_BUFFER

        day_data = px_vev_e[px_vev_e["data_day"] == day].copy()
        day_data = day_data[day_data["strike"].isin(ACTIVE_STRIKES)].sort_values("timestamp")

        for ts, grp in day_data.groupby("timestamp"):
            is_eod = ts > eod_ts
            for _, row in grp.iterrows():
                K   = row["strike"]
                TTE = row["TTE"]
                if TTE <= 0:
                    continue

                bid1 = row.get("bid_price_1", np.nan)
                ask1 = row.get("ask_price_1", np.nan)
                mid  = row["mid_price"]

                if pd.isna(bid1) or pd.isna(ask1) or bid1 <= 0 or ask1 <= 0:
                    continue

                model = bs_call(S_fair, K, TTE, parabola_iv(S_fair, K, TTE))

                if is_eod:
                    # Flatten at mid
                    p = pos[K]
                    if p != 0:
                        cash[K] += p * mid
                        pos[K]   = 0
                    continue

                # Buy signal: model above ask → cheap option, buy
                if model > ask1 and pos[K] < POS_CAP:
                    qty = min(TRADE_SIZE, POS_CAP - pos[K])
                    pos[K]  += qty
                    cash[K] -= ask1 * qty

                # Sell signal: model below bid → expensive option, sell
                elif model < bid1 and pos[K] > -POS_CAP:
                    qty = min(TRADE_SIZE, pos[K] + POS_CAP)
                    pos[K]  -= qty
                    cash[K] += bid1 * qty

        # Mark residual positions to final mid of the day
        final_mids = (day_data.groupby("strike")["mid_price"].last())
        for K in ACTIVE_STRIKES:
            p = pos[K]
            if p != 0:
                fmid = final_mids.get(K, np.nan)
                if not np.isnan(fmid):
                    cash[K] += p * fmid
                pos[K] = 0
            total_pnl += cash[K]

        if verbose:
            day_pnl = sum(cash.values())
            print(f"  Day {day}: PnL = {day_pnl:+.1f}")

    return total_pnl

# ── Initial estimate ──────────────────────────────────────────────────────────
S0 = K_star
print(f"\n{'='*60}")
print(f"STARTING S_fair = {S0:.1f}")
pnl0 = backtest(S0, verbose=True)
print(f"Total PnL @ S_fair={S0:.1f} → {pnl0:+.1f}")

# ── Binary search over [4200, 5600] ──────────────────────────────────────────
# The "pipe chamber" flat region on the IV curve spans roughly 4500–5500.
S_LO, S_HI = 4200.0, 5600.0
N_COARSE    = 20   # coarse grid first

print(f"\n{'='*60}")
print(f"Coarse grid search over S_fair in [{S_LO:.0f}, {S_HI:.0f}] …")
grid = np.linspace(S_LO, S_HI, N_COARSE)
pnls = [backtest(s) for s in grid]

for s, p in zip(grid, pnls):
    bar = "█" * int(max(0, p) / 500)
    print(f"  S={s:7.1f}  PnL={p:+9.1f}  {bar}")

best_idx   = int(np.argmax(pnls))
best_coarse = grid[best_idx]
print(f"\nCoarse best: S_fair = {best_coarse:.1f}  PnL = {pnls[best_idx]:+.1f}")

# Fine binary search (golden section / scipy minimize_scalar)
print(f"\nFine search in [{grid[max(0,best_idx-1)]:.1f}, {grid[min(N_COARSE-1,best_idx+1)]:.1f}] …")
fine_lo = grid[max(0, best_idx - 1)]
fine_hi = grid[min(N_COARSE - 1, best_idx + 1)]

res = minimize_scalar(lambda s: -backtest(s),
                      bounds=(fine_lo, fine_hi),
                      method="bounded",
                      options={"xatol": 10.0, "maxiter": 20})
S_opt = res.x
pnl_opt = -res.fun

print(f"\n{'='*60}")
print(f"OPTIMAL S_fair = {S_opt:.1f}   max PnL = {pnl_opt:+.1f}")
print(f"(vs initial S_fair={S0:.1f}  PnL={pnl0:+.1f})")
print(f"\nRunning verbose backtest at optimal S_fair …")
_ = backtest(S_opt, verbose=True)

# ── Visualise parabola + search results ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: fitted parabola + data
ax = axes[0]
m_grid = np.linspace(-1.5, 1.5, 300)
ax.plot(m_grid, np.polyval(coeffs, m_grid), "r-", lw=2, label="Fitted parabola")
COLORS_DAY = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
for day in DAYS:
    sub = sample[sample["data_day"] == day]
    ax.scatter(sub["m"], sub["IV"], s=6, alpha=0.4, color=COLORS_DAY[day], label=f"Day {day}")
ax.set_xlabel("Log-moneyness  m = log(S/K)/√T")
ax.set_ylabel("Implied Volatility")
ax.set_title("IV parabola fit  (m-space)")
ax.axvline(0, color="gray", ls="--", lw=1, label="m=0 (ATM)")
ax.axvline(m_star, color="red", ls=":", lw=1.5, label=f"vertex m*={m_star:.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel B: PnL vs S_fair
ax = axes[1]
ax.plot(grid, pnls, "o-", color="steelblue", lw=2, ms=5)
ax.axvline(S0,   color="orange", ls="--", lw=1.5, label=f"Initial S={S0:.0f}")
ax.axvline(S_opt, color="crimson", ls="-",  lw=2,   label=f"Optimal S={S_opt:.0f}")
ax.axvspan(S_LO, S_HI, alpha=0.05, color="green", label="Search region")
ax.set_xlabel("Assumed S_fair (spot price)")
ax.set_ylabel("Simulated PnL")
ax.set_title("PnL vs assumed fair value S")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(f"IV Parabola + Fair Value Search  |  optimal S={S_opt:.0f}  PnL={pnl_opt:+.0f}",
             fontsize=13, fontweight="bold")
fig.tight_layout()

out = os.path.join(HERE, "iv_fair_value_search.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nPlot saved → {out}")
plt.show()

print(f"\n{'='*60}")
print("SUMMARY")
print(f"  Parabola:    IV(m) = {a:.4f}·m² + {b:.4f}·m + {c:.4f}")
print(f"  Vertex m*  = {m_star:.4f}")
print(f"  Initial S  = {S0:.1f}  (parabola minimum in strike space)")
print(f"  Optimal S  = {S_opt:.1f}  (maximises backtest PnL)")
print(f"  PnL lift   = {pnl_opt - pnl0:+.1f}")
