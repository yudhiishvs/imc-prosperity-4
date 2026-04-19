"""
Round 2 Osmium Fair Value & Mean Reversion Analysis
====================================================
1. Mean mid_price per day + cross-day mean & std dev
2. Histogram of mid prices across all 3 days (aggregated)
3. ADF test per day to assess mean reversion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
import shutil

# ── Paths ──
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "ROUND_2"
out_dir = repo_root / "vedant" / "round_2_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

gemini_artifacts_dir = Path(
    "/Users/vedant/.gemini/antigravity/brain/"
    "40238af2-ca9f-40a5-b7dc-1bb1d5b3871c/artifacts"
)
gemini_artifacts_dir.mkdir(parents=True, exist_ok=True)

days = ["-1", "0", "1"]
PRODUCT = "ASH_COATED_OSMIUM"

# ── Load & filter ──
day_frames = {}
for day in days:
    csv_path = data_dir / f"prices_round_2_day_{day}.csv"
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["product"] == PRODUCT].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Replace 0s with NaN (exchange artifact for empty books) and interpolate
    df["mid_price"] = df["mid_price"].replace(0, np.nan).interpolate().bfill().ffill()
    day_frames[day] = df

# ═══════════════════════════════════════════════════════════════
# 1. Per-day means + cross-day statistics
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  FAIR VALUE ANALYSIS — ASH_COATED_OSMIUM (Round 2)")
print("=" * 60)

day_means = {}
for day, df in day_frames.items():
    mean_price = df["mid_price"].mean()
    std_price = df["mid_price"].std()
    min_price = df["mid_price"].min()
    max_price = df["mid_price"].max()
    day_means[day] = mean_price
    print(f"\n  Day {day:>2}:")
    print(f"    Mean mid_price:  {mean_price:>10.2f}")
    print(f"    Std dev:         {std_price:>10.2f}")
    print(f"    Min:             {min_price:>10.2f}")
    print(f"    Max:             {max_price:>10.2f}")
    print(f"    N observations:  {len(df):>10}")

means_array = np.array(list(day_means.values()))
cross_day_mean = means_array.mean()
cross_day_std = means_array.std()

print(f"\n  ── Cross-Day Summary ──")
print(f"    Mean of daily means:  {cross_day_mean:.2f}")
print(f"    Std of daily means:   {cross_day_std:.2f}")
print(f"    Range of means:       {means_array.min():.2f} — {means_array.max():.2f}")

# ═══════════════════════════════════════════════════════════════
# 2. Aggregated Histogram
# ═══════════════════════════════════════════════════════════════
all_mids = pd.concat([df["mid_price"] for df in day_frames.values()])

fig, ax = plt.subplots(figsize=(16, 8))
ax.hist(all_mids, bins=120, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(cross_day_mean, color="red", linestyle="--", linewidth=2,
           label=f"Cross-day mean = {cross_day_mean:.1f}")
ax.axvline(10_000, color="orange", linestyle=":", linewidth=2,
           label="Static fair value = 10,000")
ax.set_title("ASH_COATED_OSMIUM — Mid Price Distribution (Round 2, All Days)",
             fontsize=18)
ax.set_xlabel("Mid Price", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.legend(fontsize=13)
ax.grid(alpha=0.25)
plt.tight_layout()

hist_path = out_dir / "osmium_mid_histogram_r2.png"
plt.savefig(hist_path, dpi=150)
plt.close()
shutil.copy2(hist_path, gemini_artifacts_dir / "osmium_mid_histogram_r2.png")
print(f"\n  Histogram saved → {hist_path.name}")

# ═══════════════════════════════════════════════════════════════
# 3. ADF Test per day
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  AUGMENTED DICKEY-FULLER TEST (Mean Reversion)")
print("=" * 60)

for day, df in day_frames.items():
    series = df["mid_price"].dropna().values
    result = adfuller(series, maxlag=20, autolag="AIC")
    adf_stat, p_value, used_lag, nobs, crit_values, icbest = result

    verdict = "STATIONARY (mean-reverting)" if p_value < 0.05 else "NON-STATIONARY (unit root / drifting)"

    print(f"\n  Day {day}:")
    print(f"    ADF Statistic:   {adf_stat:>10.4f}")
    print(f"    p-value:         {p_value:>10.6f}")
    print(f"    Lags used:       {used_lag:>10}")
    print(f"    Observations:    {nobs:>10}")
    for k, v in crit_values.items():
        print(f"    Critical ({k}): {v:>10.4f}")
    print(f"    Verdict:         {verdict}")

print(f"\n{'=' * 60}")
print("  Analysis complete.")
print("=" * 60)
