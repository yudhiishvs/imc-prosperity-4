"""
Plot implied volatility smile overlaid on La trahison des images.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from scipy.optimize import brentq

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "ROUND_3")
IMG_PATH = os.path.join(DATA_DIR, "La_trahison_des_images.png")

DAYS      = [0, 1, 2]
TTE_START = {0: 7.0, 1: 6.0, 2: 5.0}
STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_NAMES = [f"VEV_{k}" for k in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"

DAY_COLORS = {0: "#00e5ff", 1: "#ff9100", 2: "#76ff03"}  # bright on dark bg


def bs_call_price(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    T_yr = T / 252.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_yr) / (sigma * np.sqrt(T_yr))
    d2 = d1 - sigma * np.sqrt(T_yr)
    return S * norm.cdf(d1) - K * np.exp(-r * T_yr) * norm.cdf(d2)


def bs_call_iv(C_market, S, K, T, r=0.0):
    intrinsic = max(S - K, 0.0)
    if C_market < intrinsic - 0.5 or T <= 0 or S <= 0:
        return np.nan
    C_market = max(C_market, intrinsic + 1e-6)
    def obj(sigma):
        return bs_call_price(S, K, T, sigma, r) - C_market
    try:
        lo_val, hi_val = obj(1e-6), obj(10.0)
        if np.sign(lo_val) == np.sign(hi_val):
            return np.nan
        return brentq(obj, 1e-6, 10.0, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan


# ── Load data ──────────────────────────────────────────────────────────────────
frames = []
for day in DAYS:
    df = pd.read_csv(os.path.join(DATA_DIR, f"prices_round_3_day_{day}.csv"), sep=";")
    df.columns = df.columns.str.strip()
    df["data_day"] = day
    df["TTE"] = TTE_START[day] - df["timestamp"] / 1_000_000
    frames.append(df)
prices = pd.concat(frames, ignore_index=True)
prices = prices[prices["mid_price"] > 0]

px_vf  = prices[prices["product"] == UNDERLYING]
px_vev = prices[prices["product"].isin(VEV_NAMES)].copy()
px_vev["strike"] = px_vev["product"].str.replace("VEV_", "").astype(float)

# Merge nearest underlying price per (day, timestamp)
vev_parts = []
for day in DAYS:
    sub_vev = px_vev[px_vev["data_day"] == day].copy()
    sub_vf  = (px_vf[px_vf["data_day"] == day][["timestamp", "mid_price"]]
               .rename(columns={"mid_price": "S"}).sort_values("timestamp"))
    sub_vev = sub_vev.sort_values("timestamp")
    vev_parts.append(pd.merge_asof(sub_vev, sub_vf, on="timestamp", direction="nearest"))
px_vev_e = pd.concat(vev_parts, ignore_index=True)

# Compute IV on 1-in-20 sample (speed)
iv_rows = []
for _, row in px_vev_e.iloc[::20].iterrows():
    iv = bs_call_iv(row["mid_price"], row["S"], row["strike"], row["TTE"])
    iv_rows.append({"data_day": row["data_day"], "strike": row["strike"], "IV": iv})
iv_df = pd.DataFrame(iv_rows)

# Median IV per (day, strike)
iv_med = (iv_df[iv_df["IV"].notna() & (iv_df["IV"] < 5)]
          .groupby(["data_day", "strike"])["IV"]
          .median().reset_index())

print("Median IV per day/strike:")
print(iv_med.pivot(index="strike", columns="data_day", values="IV").round(3).to_string())

# ── Plot ───────────────────────────────────────────────────────────────────────
img = np.array(Image.open(IMG_PATH).convert("RGB"))
img_h, img_w = img.shape[:2]

# Data limits
x_lo, x_hi = 3900, 6600          # strike axis
y_lo, y_hi = 0.0, 0.55           # IV axis (tune to your data)

fig, ax = plt.subplots(figsize=(13, 7))
ax.imshow(img, extent=[x_lo, x_hi, y_lo, y_hi], aspect="auto", zorder=0)

for day in DAYS:
    sub = iv_med[iv_med["data_day"] == day].sort_values("strike")
    if sub.empty:
        continue
    color = DAY_COLORS[day]
    tte   = TTE_START[day]
    ax.plot(sub["strike"], sub["IV"], "o-",
            color=color, lw=2.5, ms=7, zorder=5,
            label=f"Day {day}  (TTE={tte:.0f}d)")
    # IQR band
    iv_q = (iv_df[iv_df["IV"].notna() & (iv_df["IV"] < 5) & (iv_df["data_day"] == day)]
            .groupby("strike")["IV"].quantile([0.25, 0.75]).unstack())
    if not iv_q.empty:
        ax.fill_between(iv_q.index, iv_q[0.25], iv_q[0.75],
                        color=color, alpha=0.18, zorder=4)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel("Strike", fontsize=13, color="white")
ax.set_ylabel("Implied Volatility", fontsize=13, color="white")
ax.set_title("Ceci n'est pas une volatility smile.",
             fontsize=16, color="white", style="italic", pad=12)

ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")

legend = ax.legend(fontsize=11, loc="upper left",
                   framealpha=0.55, facecolor="#111", labelcolor="white")
ax.grid(alpha=0.25, color="white", linestyle="--", zorder=3)

fig.patch.set_facecolor("#1a1a1a")
ax.set_facecolor("#1a1a1a")

out_path = os.path.join(HERE, "iv_on_magritte.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_path}")
plt.show()
