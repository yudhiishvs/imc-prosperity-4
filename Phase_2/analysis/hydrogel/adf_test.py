"""
Augmented Dickey-Fuller (ADF) & KPSS Stationarity Tests — HYDROGEL_PACK
=========================================================================
Uses statsmodels for the ADF and KPSS tests.

ADF:  H₀ = unit root (NOT stationary).  Reject → mean-reverting.
KPSS: H₀ = stationary.  Fail to reject → consistent with stationarity.

Tests run:
  1. ADF on full 3-day series, constant only ('c')
  2. ADF on full 3-day series, constant + trend ('ct')
  3. ADF per day, both specs
  4. ADF on returns (Δmid) — sanity check
  5. KPSS on full series, level and trend
  6. Rolling ADF (window=500, step=50) — plots p-value and γ over time
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(SCRIPT_DIR, "../../data/ROUND_3")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "../../plots/hydrogel")

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
    'axes.edgecolor':   '#4a4a6a', 'axes.labelcolor': '#e0e0e0',
    'axes.titlecolor':  '#e0e0e0', 'text.color':      '#e0e0e0',
    'xtick.color':      '#a0a0c0', 'ytick.color':     '#a0a0c0',
    'grid.color':       '#2a2a4a', 'grid.linestyle':  '--',
    'grid.alpha': 0.7, 'lines.linewidth': 1.0,
    'legend.facecolor': '#0f3460', 'legend.edgecolor': '#4a4a6a',
})
BLUE = '#0f6fc6'; GOLD = '#f5a623'; ACCENT = '#e94560'; GREEN = '#39d353'

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading HYDROGEL_PACK data ...")
frames = []
for day_idx, fname in enumerate(["prices_round_3_day_0.csv",
                                   "prices_round_3_day_1.csv",
                                   "prices_round_3_day_2.csv"]):
    df = pd.read_csv(os.path.join(BASE, fname), sep=';')
    df = df[df['product'] == 'HYDROGEL_PACK'].copy()
    df['global_ts'] = df['timestamp'] + day_idx * 1_000_000
    df['day'] = day_idx
    frames.append(df)

raw = pd.concat(frames, ignore_index=True).sort_values('global_ts').reset_index(drop=True)
mid = raw['mid_price'].values
mid_returns = np.diff(mid)
print(f"  Loaded {len(mid):,} observations across {raw['day'].nunique()} days.\n")

# ── helpers ───────────────────────────────────────────────────────────────────
def run_adf(series, label, regression='c'):
    result   = adfuller(series, regression=regression, autolag='AIC')
    stat, p, n_lags, n_obs, crits = result[0], result[1], result[2], result[3], result[4]
    reject   = p < 0.05
    verdict  = "REJECT H0  =>  stationary / mean-reverting" if reject else \
               "FAIL TO REJECT H0  =>  unit root not excluded"
    print(f"  {'─'*62}")
    print(f"  {label}")
    print(f"  {'─'*62}")
    print(f"  Spec           : regression='{regression}', AIC lags={n_lags}, obs={n_obs}")
    print(f"  ADF Statistic  : {stat:.6f}")
    print(f"  p-value        : {p:.6e}")
    print(f"  Critical values: 1%={crits['1%']:.3f}  5%={crits['5%']:.3f}  10%={crits['10%']:.3f}")
    print(f"  Verdict        : {verdict}\n")
    return {'label': label, 'spec': regression, 'stat': stat,
            'p': p, 'lags': n_lags, 'crits': crits, 'reject': reject}


def run_kpss(series, label, regression='c'):
    # suppress the lag truncation warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, n_lags, crits = kpss(series, regression=regression, nlags='auto')
    reject  = p < 0.05
    verdict = "FAIL TO REJECT H0  =>  consistent with stationarity" if not reject else \
              "REJECT H0  =>  evidence of non-stationarity"
    print(f"  KPSS [{label}]  regression='{regression}'")
    print(f"  KPSS Statistic : {stat:.6f}  (lags={n_lags})")
    print(f"  p-value        : {p:.4f}")
    print(f"  Critical values: 10%={crits['10%']:.3f}  5%={crits['5%']:.3f}  1%={crits['1%']:.3f}")
    print(f"  Verdict        : {verdict}\n")
    return {'label': label, 'spec': regression, 'stat': stat,
            'p': p, 'reject': reject}

# ─────────────────────────────────────────────────────────────────────────────
# 1–5: ADF tests
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  AUGMENTED DICKEY-FULLER (ADF) — HYDROGEL_PACK")
print("=" * 65)
adf_results = []
adf_results.append(run_adf(mid, "Full 3-day series (N=30,000) [constant only]",       'c'))
adf_results.append(run_adf(mid, "Full 3-day series (N=30,000) [constant + trend]",    'ct'))
for d in range(3):
    s = raw[raw['day'] == d]['mid_price'].values
    adf_results.append(run_adf(s, f"Day {d} (N={len(s):,}) [constant only]",       'c'))
    adf_results.append(run_adf(s, f"Day {d} (N={len(s):,}) [constant + trend]",    'ct'))
adf_results.append(run_adf(mid_returns, "Returns dMid (N=29,999) [sanity check]",     'c'))

# ─────────────────────────────────────────────────────────────────────────────
# 6: KPSS tests
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  KPSS TEST (reverse-null cross-validation)")
print("=" * 65)
kpss_results = []
kpss_results.append(run_kpss(mid, "Full 3-day series", 'c'))
kpss_results.append(run_kpss(mid, "Full 3-day series", 'ct'))
for d in range(3):
    s = raw[raw['day'] == d]['mid_price'].values
    kpss_results.append(run_kpss(s, f"Day {d}", 'c'))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — ADF result bars
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax_p, ax_stat) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('ADF Stationarity Test Results — HYDROGEL_PACK', fontsize=14, fontweight='bold')

labels     = [r['label'].replace('(N=30,000)', '').replace('(N=10,000)', '').strip()
              for r in adf_results]
pvals      = [r['p']    for r in adf_results]
stats_vals = [r['stat'] for r in adf_results]
colors     = [GREEN if r['reject'] else ACCENT for r in adf_results]

ax_p.barh(labels, pvals, color=colors, alpha=0.85, edgecolor='none')
ax_p.axvline(0.05, color=GOLD,   lw=2,   ls='--', label='a=0.05')
ax_p.axvline(0.01, color=ACCENT, lw=1.5, ls=':',  label='a=0.01')
ax_p.set_xlabel('p-value', fontsize=11)
ax_p.set_title('ADF p-values  (green = reject H0 = stationary)', fontsize=11)
ax_p.legend(fontsize=9); ax_p.grid(True, axis='x')

crit5 = adf_results[0]['crits']['5%']
crit1 = adf_results[0]['crits']['1%']
ax_stat.barh(labels, stats_vals, color=colors, alpha=0.85, edgecolor='none')
ax_stat.axvline(crit5, color=GOLD,   lw=2,   ls='--', label=f'5% CV={crit5:.2f}')
ax_stat.axvline(crit1, color=ACCENT, lw=1.5, ls=':',  label=f'1% CV={crit1:.2f}')
ax_stat.set_xlabel('ADF Statistic (more negative = stronger rejection)', fontsize=11)
ax_stat.set_title('ADF Statistics  (green = more negative than critical value)', fontsize=11)
ax_stat.legend(fontsize=9); ax_stat.grid(True, axis='x')

plt.tight_layout()
p1 = os.path.join(PLOT_DIR, 'hg_adf_results.png')
fig.savefig(p1, dpi=200)
print(f"Saved {p1}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Rolling ADF (window=500, step=50)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW, STEP = 500, 50
print(f"\nComputing rolling ADF (window={WINDOW}, step={STEP}) ...")
roll_ts, roll_p, roll_stat = [], [], []

import warnings
for start in range(0, len(mid) - WINDOW, STEP):
    seg = mid[start: start + WINDOW]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = adfuller(seg, regression='c', autolag='AIC')
            roll_ts.append(raw['global_ts'].iloc[start + WINDOW])
            roll_p.append(res[1])
            roll_stat.append(res[0])
        except Exception:
            pass

roll_ts   = np.array(roll_ts)
roll_p    = np.array(roll_p)
roll_stat = np.array(roll_stat)

fig2, axes2 = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1.5]})
fig2.suptitle(f'Rolling ADF (window={WINDOW}, step={STEP}) — HYDROGEL_PACK',
              fontsize=13, fontweight='bold')

ax = axes2[0]
ax.plot(raw['global_ts'], mid, color=BLUE, lw=0.7, alpha=0.85, label='Mid Price')
ax.set_ylabel('Mid Price', fontsize=11); ax.legend(fontsize=9); ax.grid(True)
for db in [1_000_000, 2_000_000]:
    ax.axvline(db, color='white', lw=1, ls=':', alpha=0.5)

ax = axes2[1]
ax.plot(roll_ts, roll_p, color=BLUE, lw=0.9, alpha=0.85, label='Rolling ADF p-value')
ax.fill_between(roll_ts, roll_p, where=(roll_p <  0.05), color=GREEN,  alpha=0.4, label='p<0.05  (stationary)')
ax.fill_between(roll_ts, roll_p, where=(roll_p >= 0.05), color=ACCENT, alpha=0.3, label='p>=0.05 (unit root not excluded)')
ax.axhline(0.05, color=GOLD, lw=1.5, ls='--', label='a=0.05')
ax.axhline(0.01, color=ACCENT, lw=1.2, ls=':', label='a=0.01')
for db in [1_000_000, 2_000_000]:
    ax.axvline(db, color='white', lw=1, ls=':', alpha=0.5)
ax.set_xlabel('Global Timestamp', fontsize=11)
ax.set_ylabel('ADF p-value', fontsize=11)
ax.set_ylim(-0.01, min(0.5, roll_p.max() * 1.15))
ax.legend(fontsize=9, loc='upper right'); ax.grid(True)

plt.tight_layout()
p2 = os.path.join(PLOT_DIR, 'hg_rolling_adf.png')
fig2.savefig(p2, dpi=200)
print(f"Saved {p2}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
n_adf_reject = sum(r['reject'] for r in adf_results)
n_kpss_ok    = sum(not r['reject'] for r in kpss_results)
pct_rolling  = (roll_p < 0.05).mean() * 100

print(f"\n{'='*65}")
print("  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  ADF:  {n_adf_reject}/{len(adf_results)} tests rejected H0 at a=0.05")
print(f"  KPSS: {n_kpss_ok}/{len(kpss_results)} tests consistent with stationarity")
print(f"  Rolling ADF: {pct_rolling:.1f}% of 500-tick windows had p < 0.05")
print(f"  Rolling ADF: mean statistic = {roll_stat.mean():.4f}")
print(f"{'='*65}")
