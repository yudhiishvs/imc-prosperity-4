"""
HYDROGEL_PACK Market Analysis for IMC Prosperity 4 - Phase 2
=============================================================
Generates a comprehensive set of diagnostic plots to characterize the
asset's behavior and inform a trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import os

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#4a4a6a',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#e0e0e0',
    'text.color':       '#e0e0e0',
    'xtick.color':      '#a0a0c0',
    'ytick.color':      '#a0a0c0',
    'grid.color':       '#2a2a4a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.7,
    'lines.linewidth':  1.0,
    'legend.facecolor': '#0f3460',
    'legend.edgecolor': '#4a4a6a',
})

ACCENT   = '#e94560'   # hot pink/red  — highlights
BLUE     = '#0f6fc6'   # main line colour
GOLD     = '#f5a623'   # secondary
GREEN    = '#39d353'
PURPLE   = '#9b59b6'


# ── Data Loading ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(script_dir, "../../data/ROUND_3")
FILES = [
    ("prices_round_3_day_0.csv", 0),
    ("prices_round_3_day_1.csv", 1),
    ("prices_round_3_day_2.csv", 2),
]

print("Loading data …")
frames = []
for fname, day_idx in FILES:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found — skipping")
        continue
    df = pd.read_csv(path, sep=';')
    df['global_ts'] = df['timestamp'] + day_idx * 1_000_000
    df['day'] = day_idx
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
hg  = raw[raw['product'] == 'HYDROGEL_PACK'].copy().sort_values('global_ts').reset_index(drop=True)

# ── Derived columns ───────────────────────────────────────────────────────────
hg['spread']       = hg['ask_price_1'] - hg['bid_price_1']
hg['mid']          = hg['mid_price']
hg['mid_returns']  = hg['mid'].diff()           # tick-to-tick Δ price
hg['bid_vol']      = hg['bid_volume_1'].fillna(0) + hg['bid_volume_2'].fillna(0) + hg['bid_volume_3'].fillna(0)
hg['ask_vol']      = hg['ask_volume_1'].fillna(0) + hg['ask_volume_2'].fillna(0) + hg['ask_volume_3'].fillna(0)
hg['obi']          = (hg['bid_vol'] - hg['ask_vol']) / (hg['bid_vol'] + hg['ask_vol'])  # order-book imbalance

# Fair value estimate: volume-weighted BBO average
hg['wm_price'] = (hg['bid_price_1'] * hg['ask_volume_1'] + hg['ask_price_1'] * hg['bid_volume_1']) / \
                  (hg['bid_volume_1'] + hg['ask_volume_1'])

# Rolling stats (window = 200 ticks ≈ 20 s)
W = 200
hg['roll_mean'] = hg['mid'].rolling(W, min_periods=1).mean()
hg['roll_std']  = hg['mid'].rolling(W, min_periods=1).std().fillna(0)
hg['z_score']   = (hg['mid'] - hg['roll_mean']) / hg['roll_std'].replace(0, np.nan)

ts  = hg['global_ts']
DAY_BOUNDARIES = [1_000_000, 2_000_000]

print(f"  Loaded {len(hg):,} HYDROGEL_PACK rows across {hg['day'].nunique()} day(s).\n")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Mid-Price & Rolling Mean with Spread Overlay
# ─────────────────────────────────────────────────────────────────────────────
fig1, (ax_price, ax_spread) = plt.subplots(2, 1, figsize=(16, 9),
                                            sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1]})
fig1.suptitle('HYDROGEL_PACK — Price & Bid-Ask Spread (Days 0–2)', fontsize=15, fontweight='bold', y=0.98)

ax_price.plot(ts, hg['mid'],       color=BLUE,   lw=0.8, alpha=0.85, label='Mid Price')
ax_price.plot(ts, hg['roll_mean'], color=GOLD,   lw=1.5, alpha=0.9,  label=f'Rolling Mean (w={W})')
ax_price.fill_between(ts,
                       hg['roll_mean'] - hg['roll_std'],
                       hg['roll_mean'] + hg['roll_std'],
                       color=GOLD, alpha=0.12, label='±1 σ Band')
for db in DAY_BOUNDARIES:
    ax_price.axvline(db, color=ACCENT, lw=1.2, ls=':', alpha=0.8)
ax_price.set_ylabel('Price', fontsize=11)
ax_price.legend(loc='upper right', fontsize=9)
ax_price.yaxis.set_minor_locator(AutoMinorLocator())
ax_price.grid(True)

ax_spread.plot(ts, hg['spread'], color=ACCENT, lw=0.7, alpha=0.9, label='Spread (ask₁ − bid₁)')
ax_spread.fill_between(ts, 0, hg['spread'], color=ACCENT, alpha=0.25)
spread_med = hg['spread'].median()
ax_spread.axhline(spread_med, color=GOLD, lw=1.2, ls='--', label=f'Median = {spread_med:.1f}')
for db in DAY_BOUNDARIES:
    ax_spread.axvline(db, color=ACCENT, lw=1.2, ls=':', alpha=0.8)
ax_spread.set_ylabel('Spread', fontsize=11)
ax_spread.set_xlabel('Timestamp (global)', fontsize=11)
ax_spread.legend(loc='upper right', fontsize=9)
ax_spread.grid(True)

plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_price_spread.png")
fig1.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Return Distribution & Autocorrelation
# ─────────────────────────────────────────────────────────────────────────────
returns = hg['mid_returns'].dropna()

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('HYDROGEL_PACK — Return Distribution & Serial Correlation', fontsize=15, fontweight='bold')

# 2a — return histogram
ax = axes[0]
n, bins, patches = ax.hist(returns, bins=80, color=BLUE, edgecolor='none', alpha=0.85, density=True)
# overlay normal fit
mu, sig = returns.mean(), returns.std()
x = np.linspace(bins[0], bins[-1], 300)
from scipy.stats import norm, kurtosis, skew
ax.plot(x, norm.pdf(x, mu, sig), color=ACCENT, lw=2, label=f'N({mu:.2f}, {sig:.2f}²)')
ax.set_title('Tick Return Distribution')
ax.set_xlabel('Δ Mid Price (per tick)')
ax.set_ylabel('Density')
stats_txt = (f'Mean: {mu:.3f}\nStd:  {sig:.3f}\n'
             f'Skew: {skew(returns):.3f}\nKurt: {kurtosis(returns):.3f}')
ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(facecolor='#0f3460', edgecolor='#4a4a6a', alpha=0.9))
ax.legend(fontsize=9)
ax.grid(True)

# 2b — lag autocorrelation (ACF-style bar chart)
ax = axes[1]
max_lag = 50
acf_vals = [returns.autocorr(lag=l) for l in range(1, max_lag + 1)]
lags = np.arange(1, max_lag + 1)
conf = 1.96 / np.sqrt(len(returns))
colors_acf = [GREEN if v > 0 else ACCENT for v in acf_vals]
ax.bar(lags, acf_vals, color=colors_acf, alpha=0.85, width=0.7)
ax.axhline( conf, color=GOLD, ls='--', lw=1.2, label=f'95% CI (±{conf:.3f})')
ax.axhline(-conf, color=GOLD, ls='--', lw=1.2)
ax.axhline(0, color='white', lw=0.5, alpha=0.5)
ax.set_title('Return Autocorrelation (lags 1–50)')
ax.set_xlabel('Lag (ticks)')
ax.set_ylabel('ACF')
ax.legend(fontsize=9)
ax.grid(True)

# 2c — lag-1 scatter (return[t] vs return[t-1])
ax = axes[2]
r_t  = returns.iloc[1:].values
r_t1 = returns.iloc[:-1].values
ax.scatter(r_t1, r_t, alpha=0.06, s=3, color=BLUE)
corr = np.corrcoef(r_t1, r_t)[0, 1]
m, b = np.polyfit(r_t1, r_t, 1)
x_fit = np.linspace(r_t1.min(), r_t1.max(), 100)
ax.plot(x_fit, m * x_fit + b, color=ACCENT, lw=2, label=f'OLS slope={m:.3f}\nr={corr:.3f}')
ax.set_title('Lag-1 Return Scatter (Mean-Reversion test)')
ax.set_xlabel('Return[t-1]')
ax.set_ylabel('Return[t]')
ax.legend(fontsize=9)
ax.grid(True)

plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_returns_analysis.png")
fig2.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Order Book Depth & Imbalance
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1, 1]})
fig3.suptitle('HYDROGEL_PACK — Order Book Dynamics (Days 0–2)', fontsize=15, fontweight='bold', y=0.99)

# 3a — Bid & Ask levels
ax = axes[0]
ax.plot(ts, hg['bid_price_1'], color=GREEN,  lw=0.8, alpha=0.85, label='Best Bid')
ax.plot(ts, hg['ask_price_1'], color=ACCENT, lw=0.8, alpha=0.85, label='Best Ask')
ax.plot(ts, hg['mid'],         color=BLUE,   lw=1.0, alpha=0.7,  label='Mid')
ax.fill_between(ts, hg['bid_price_1'], hg['ask_price_1'], color=GOLD, alpha=0.15, label='Spread')
for db in DAY_BOUNDARIES:
    ax.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax.set_ylabel('Price', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True)

# 3b — BBO volumes stacked
ax = axes[1]
ax.fill_between(ts,  hg['bid_vol'], color=GREEN,  alpha=0.6, label='Total Bid Volume')
ax.fill_between(ts, -hg['ask_vol'], color=ACCENT, alpha=0.6, label='Total Ask Volume (neg.)')
ax.axhline(0, color='white', lw=0.7, alpha=0.5)
for db in DAY_BOUNDARIES:
    ax.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax.set_ylabel('Volume', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True)

# 3c — OBI (Order Book Imbalance)
ax = axes[2]
ax.plot(ts, hg['obi'], color=PURPLE, lw=0.7, alpha=0.85, label='OBI = (bidVol−askVol)/(bidVol+askVol)')
obi_roll = hg['obi'].rolling(200, min_periods=1).mean()
ax.plot(ts, obi_roll, color=GOLD, lw=1.5, alpha=0.9, label='OBI 200-tick MA')
ax.axhline(0, color='white', lw=0.7, alpha=0.5)
for db in DAY_BOUNDARIES:
    ax.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax.set_ylim(-1, 1)
ax.set_ylabel('OBI', fontsize=11)
ax.set_xlabel('Timestamp (global)', fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True)

plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_orderbook.png")
fig3.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Z-Score Mean Reversion & Regime Map
# ─────────────────────────────────────────────────────────────────────────────
fig4, (ax_z, ax_hist_z) = plt.subplots(1, 2, figsize=(18, 7),
                                         gridspec_kw={'width_ratios': [3, 1]})
fig4.suptitle('HYDROGEL_PACK — Z-Score & Mean-Reversion Signal', fontsize=15, fontweight='bold')

z = hg['z_score']
ax_z.plot(ts, z, color=BLUE, lw=0.6, alpha=0.75, label=f'Z-Score (w={W})')
ax_z.fill_between(ts, z, where=(z >  2), color=ACCENT, alpha=0.5, label='Overbought (z>+2)  → SELL')
ax_z.fill_between(ts, z, where=(z < -2), color=GREEN,  alpha=0.5, label='Oversold  (z<−2)  → BUY')
ax_z.axhline( 2, color=ACCENT, ls='--', lw=1.2)
ax_z.axhline(-2, color=GREEN,  ls='--', lw=1.2)
ax_z.axhline( 1, color=ACCENT, ls=':',  lw=0.8)
ax_z.axhline(-1, color=GREEN,  ls=':',  lw=0.8)
ax_z.axhline( 0, color='white', lw=0.6, alpha=0.5)
for db in DAY_BOUNDARIES:
    ax_z.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax_z.set_xlabel('Timestamp (global)', fontsize=11)
ax_z.set_ylabel('Z-Score', fontsize=11)
ax_z.legend(loc='upper right', fontsize=9)
ax_z.grid(True)

# Histogram of z-scores
from scipy.stats import norm as sp_norm
ax_hist_z.hist(z.dropna(), bins=80, orientation='horizontal',
               color=BLUE, edgecolor='none', alpha=0.85, density=True)
z_clean = z.dropna()
zy = np.linspace(z_clean.min(), z_clean.max(), 300)
ax_hist_z.plot(sp_norm.pdf(zy, 0, 1), zy, color=ACCENT, lw=2, label='N(0,1)')
ax_hist_z.axhline( 2, color=ACCENT, ls='--', lw=1.2)
ax_hist_z.axhline(-2, color=GREEN,  ls='--', lw=1.2)
ax_hist_z.set_xlabel('Density', fontsize=11)
ax_hist_z.set_title('Z-Score Distribution')
ax_hist_z.legend(fontsize=9)
ax_hist_z.grid(True)

plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_zscore.png")
fig4.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Intra-Day Mid-Price Shape (per day overlay)
# ─────────────────────────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(14, 6))
fig5.suptitle('HYDROGEL_PACK — Intra-Day Mid Price per Day', fontsize=15, fontweight='bold')

colors_day = [BLUE, GOLD, GREEN]
for day_idx in sorted(hg['day'].unique()):
    subset = hg[hg['day'] == day_idx]
    label  = f'Day {day_idx}'
    ax5.plot(subset['timestamp'], subset['mid'], color=colors_day[day_idx], lw=0.9, alpha=0.85, label=label)

ax5.set_xlabel('Intra-Day Timestamp', fontsize=11)
ax5.set_ylabel('Mid Price', fontsize=11)
ax5.legend(fontsize=10)
ax5.grid(True)
plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_intraday_overlay.png")
fig5.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Spread Distribution & Spread vs Volume
# ─────────────────────────────────────────────────────────────────────────────
fig6, (ax_sdist, ax_svol) = plt.subplots(1, 2, figsize=(16, 6))
fig6.suptitle('HYDROGEL_PACK — Spread Deep Dive', fontsize=15, fontweight='bold')

spreads = hg['spread'].dropna()
# Spread histogram
ax_sdist.hist(spreads, bins=60, color=ACCENT, edgecolor='none', alpha=0.85, density=True)
ax_sdist.axvline(spreads.mean(),   color=GOLD,  ls='--', lw=1.5, label=f'Mean  = {spreads.mean():.2f}')
ax_sdist.axvline(spreads.median(), color=GREEN,  ls='--', lw=1.5, label=f'Median = {spreads.median():.2f}')
ax_sdist.set_title('Spread Distribution')
ax_sdist.set_xlabel('Bid-Ask Spread')
ax_sdist.set_ylabel('Density')
ax_sdist.legend(fontsize=9)
ax_sdist.grid(True)

# Spread vs total volume scatter
total_vol = hg['bid_vol'] + hg['ask_vol']
ax_svol.scatter(total_vol, hg['spread'], alpha=0.08, s=3, color=PURPLE)
ax_svol.set_title('Spread vs. Total Book Volume')
ax_svol.set_xlabel('Total Visible Volume (bids+asks)')
ax_svol.set_ylabel('Spread')
ax_svol.grid(True)

corr_sv = hg[['spread', 'bid_vol', 'ask_vol']].assign(total_vol=total_vol).corr()['spread']['total_vol']
ax_svol.text(0.05, 0.95, f'Pearson r(spread, vol) = {corr_sv:.3f}',
             transform=ax_svol.transAxes, va='top', fontsize=10,
             bbox=dict(facecolor='#0f3460', edgecolor='#4a4a6a', alpha=0.9))

plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_spread_deep.png")
fig6.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Price Level Drift / Long-Term Trend
# ─────────────────────────────────────────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(16, 6))
fig7.suptitle('HYDROGEL_PACK — Long-Run Price Drift & Deviation from 10 000', fontsize=15, fontweight='bold')

fair_val = 10_000
deviation = hg['mid'] - fair_val
ax7.plot(ts, deviation, color=BLUE,  lw=0.8, alpha=0.85, label='Mid − 10 000')
ax7.fill_between(ts, deviation, where=(deviation > 0), color=GREEN,  alpha=0.3, label='Above par')
ax7.fill_between(ts, deviation, where=(deviation < 0), color=ACCENT, alpha=0.3, label='Below par')
ax7.axhline(0, color='white', lw=1, alpha=0.6, ls='--')
for db in DAY_BOUNDARIES:
    ax7.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax7.set_xlabel('Timestamp (global)', fontsize=11)
ax7.set_ylabel('Deviation from 10 000', fontsize=11)
ax7.legend(fontsize=10)
ax7.grid(True)
plt.tight_layout()
save_path = os.path.join(script_dir, "../../plots/hydrogel/hg_drift.png")
fig7.savefig(save_path, dpi=200)
print(f"Saved {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary Statistics (printed to terminal)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  HYDROGEL_PACK — Summary Statistics")
print("="*60)
print(f"  Mid Price   min={hg['mid'].min():.1f}  max={hg['mid'].max():.1f}  "
      f"mean={hg['mid'].mean():.2f}  std={hg['mid'].std():.2f}")
print(f"  Spread      min={spreads.min():.1f}   max={spreads.max():.1f}   "
      f"mean={spreads.mean():.2f}  median={spreads.median():.2f}")
print(f"  Bid Vol     mean={hg['bid_vol'].mean():.1f}   std={hg['bid_vol'].std():.1f}")
print(f"  Ask Vol     mean={hg['ask_vol'].mean():.1f}   std={hg['ask_vol'].std():.1f}")
print(f"  Return ACF(1) = {returns.autocorr(1):.4f}")
print(f"  Return ACF(2) = {returns.autocorr(2):.4f}")
print(f"  Return ACF(3) = {returns.autocorr(3):.4f}")
print(f"  Return Kurtosis = {kurtosis(returns):.3f}")
print(f"  Return Skewness  = {skew(returns):.3f}")
print(f"  OBI mean = {hg['obi'].mean():.4f}")
print("="*60)
print("\nAll plots saved. Analysis complete.")
