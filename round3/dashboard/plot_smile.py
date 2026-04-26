"""
Quick matplotlib vol smile plot.
Run from the dashboard/ directory:
    python plot_smile.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from data_loader import load_prices, VEV_STRIKES, UNDERLYING, compute_tte
from options_math import implied_vol, log_moneyness


def fit_smile_upward(m, iv, a_min=0.0):
    """
    Fit  a·m² + b·m + c  with constraints:
      a >= a_min  (control minimum curvature; use 1.0 for bid to avoid linear collapse)
      c >= 0      (positive ATM vol)
    Uses scipy curve_fit with bounds instead of unconstrained polyfit.
    Returns (a, b, c) or None if fit fails.
    """
    valid = ~(np.isnan(m) | np.isnan(iv))
    if valid.sum() < 3:
        return None
    mv, ivv = m[valid], iv[valid]

    def parabola(x, a, b, c):
        return a * x**2 + b * x + c

    try:
        popt, _ = curve_fit(
            parabola, mv, ivv,
            p0=[max(a_min, 0.5), 0.0, float(np.median(ivv))],
            bounds=([a_min, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
        return popt   # (a, b, c)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAY = None          # None = all days, or 0 / 1 / 2
SNAPSHOT_TS = None  # global_ts to show single snapshot, or None = all time

# Minimum IV to include in scatter + fit — filters near-zero garbage from
# bids of 1 on deep OTM/ITM options where extrinsic value is essentially 0
MIN_IV = 0.004
# ---------------------------------------------------------------------------

print("Loading prices...", flush=True)
prices = load_prices()

# Underlying spot
spot_df = (
    prices[prices["product"] == UNDERLYING][["day", "timestamp", "mid_price"]]
    .rename(columns={"mid_price": "spot"})
    .drop_duplicates(["day", "timestamp"])
)

if DAY is not None:
    prices  = prices[prices["day"] == DAY]
    spot_df = spot_df[spot_df["day"] == DAY]

# Collect per-strike points
records = []
for product, strike in VEV_STRIKES.items():
    vev = (
        prices[prices["product"] == product]
        [["day", "timestamp", "mid_price", "bid_price_1", "ask_price_1"]]
        .drop_duplicates(["day", "timestamp"])
    ).merge(spot_df, on=["day", "timestamp"], how="inner")

    if vev.empty:
        continue

    S = vev["spot"].values
    T = compute_tte(vev["day"].values, vev["timestamp"].values)
    T = np.clip(T, 0, None)

    global_ts = vev["day"].values * 1_000_000 + vev["timestamp"].values

    if SNAPSHOT_TS is not None:
        mask = np.abs(global_ts - SNAPSHOT_TS) <= 100_000
        S, T = S[mask], T[mask]
        mid   = vev["mid_price"].values[mask]
        bid   = vev["bid_price_1"].values[mask]
        ask   = vev["ask_price_1"].values[mask]
    else:
        mid = vev["mid_price"].values
        bid = vev["bid_price_1"].values
        ask = vev["ask_price_1"].values

    m   = log_moneyness(S, strike, T)
    iv_mid = implied_vol(mid, S, strike, T)
    iv_bid = implied_vol(bid, S, strike, T)
    iv_ask = implied_vol(ask, S, strike, T)

    for i in range(len(m)):
        records.append(dict(m=float(m[i]),
                            iv_mid=float(iv_mid[i]),
                            iv_bid=float(iv_bid[i]),
                            iv_ask=float(iv_ask[i]),
                            strike=strike, product=product))

import pandas as pd
df = pd.DataFrame(records)
print(f"  {len(df):,} data points loaded", flush=True)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#131722")
ax.set_facecolor("#1e222d")

# Scatter: bid (blue) and ask (red) per strike
strike_cmap = plt.cm.tab10
strike_list = list(VEV_STRIKES.keys())
colors = [strike_cmap(i / len(strike_list)) for i in range(len(strike_list))]

for i, product in enumerate(strike_list):
    sub = df[df["product"] == product].dropna(subset=["m"])
    c = colors[i]

    ask_pts = sub[sub["iv_ask"] >= MIN_IV].dropna(subset=["iv_ask"])
    bid_pts = sub[sub["iv_bid"] >= MIN_IV].dropna(subset=["iv_bid"])

    ax.scatter(ask_pts["m"], ask_pts["iv_ask"],
               s=6, alpha=0.3, color="#ef5350", zorder=2)
    ax.scatter(bid_pts["m"], bid_pts["iv_bid"],
               s=6, alpha=0.3, color="#2196f3", zorder=2)

# Fit bid and ask smiles
# ask: unconstrained a>=0 naturally finds a~2.25 (data has clear curvature)
# bid: needs a>=1 to prevent degenerate linear collapse
for iv_col, color, label, a_min, zorder in [
    ("iv_ask", "#ff6b6b", "Ask smile", 0.0, 5),
    ("iv_bid", "#64b5f6", "Bid smile", 1.0, 5),
]:
    valid = df[df[iv_col] >= MIN_IV].dropna(subset=["m", iv_col])
    if len(valid) < 5:
        continue
    coeffs = fit_smile_upward(valid["m"].values, valid[iv_col].values, a_min)
    if coeffs is None:
        continue
    a, b, c_coef = coeffs
    m_rng = np.linspace(valid["m"].min(), valid["m"].max(), 500)
    iv_fit = a * m_rng**2 + b * m_rng + c_coef
    ax.plot(m_rng, iv_fit, color=color, linewidth=2.5,
            linestyle="--", label=label, zorder=zorder)
    ax.annotate(
        f"{label}:  a={a:.4f}  b={b:.4f}  c={c_coef:.4f}",
        xy=(0.02, 0.97 if iv_col == "iv_ask" else 0.91),
        xycoords="axes fraction", fontsize=9,
        color=color, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#131722", ec=color, alpha=0.7),
    )

# ATM line
ax.axvline(0, color="#787b86", linewidth=1.2, linestyle=":", zorder=3)
ax.text(0.002, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
        "ATM", color="#787b86", fontsize=9, va="top")

# Legend patch for bid/ask scatter colours
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef5350',
           markersize=7, label='Ask IV (scatter)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196f3',
           markersize=7, label='Bid IV (scatter)', linestyle='None'),
    Line2D([0], [0], color='#ff6b6b', linewidth=2.5,
           linestyle='--', label='Ask smile fit'),
    Line2D([0], [0], color='#64b5f6', linewidth=2.5,
           linestyle='--', label='Bid smile fit'),
]
ax.legend(handles=legend_elements, facecolor="#1e222d", edgecolor="#2a2e39",
          labelcolor="#d1d4dc", fontsize=9)

ax.set_xlabel("Log-moneyness  log(S/K) / √T   [positive = ITM]",
              color="#787b86", fontsize=10)
ax.set_ylabel("Implied Volatility  (per √Solvenarian day)",
              color="#787b86", fontsize=10)
day_label = f"Day {DAY}" if DAY is not None else "All Days"
snap_label = f"  ·  snapshot @ {SNAPSHOT_TS}" if SNAPSHOT_TS else ""
ax.set_title(f"VEV Options — Volatility Smile  ·  {day_label}{snap_label}",
             color="#d1d4dc", fontsize=12, pad=10)

ax.set_ylim(bottom=0)
ax.tick_params(colors="#787b86")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a2e39")
ax.grid(True, color="#2a2e39", linewidth=0.6, linestyle="-")

plt.tight_layout()
plt.savefig("vol_smile.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: vol_smile.png")
plt.show()
