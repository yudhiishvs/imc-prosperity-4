import pandas as pd
import numpy as np
import glob
import os

data_dir = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/data/ROUND_2"
files = glob.glob(os.path.join(data_dir, "prices_round_2_day_*.csv"))

results = []

for f in sorted(files):
    day_name = os.path.basename(f)
    df = pd.read_csv(f, sep=';')
    
    # ── PEPPER Analysis ──
    pep_df = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    if not pep_df.empty:
        pep_mids = pep_df['mid_price'].values
        pep_slope = np.polyfit(np.arange(len(pep_mids)), pep_mids, 1)[0]
        pep_start = pep_mids[0]
        pep_end = pep_mids[-1]
    else:
        pep_slope, pep_start, pep_end = 0, 0, 0
        
    # ── OSMIUM Analysis ──
    ash_df = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    ash_df = ash_df[ash_df['mid_price'] > 0]
    if not ash_df.empty:
        ash_mids = ash_df['mid_price'].values
        ash_mean = np.mean(ash_mids)
        ash_std = np.std(ash_mids)
        # OU check: autocorrelation of residuals (X_t - mu)
        ash_res = ash_mids - ash_mean
        ash_ac = np.corrcoef(ash_res[:-1], ash_res[1:])[0, 1]
        ash_slope = np.polyfit(np.arange(len(ash_mids)), ash_mids, 1)[0]
    else:
        ash_mean, ash_std, ash_ac, ash_slope = 0, 0, 0, 0

    results.append({
        "day": day_name,
        "pep_slope": pep_slope,
        "pep_total_drift": pep_end - pep_start,
        "ash_mean": ash_mean,
        "ash_slope": ash_slope,
        "ash_autocorr": ash_ac
    })

print("--- REGIME GROUND TRUTH ---")
for r in results:
    print(f"\n[{r['day']}]")
    print(f"  PEPPER: Slope: {r['pep_slope']:.6f} | Total Day Drift: {r['pep_total_drift']:+.2f}")
    print(f"  OSMIUM: Mean: {r['ash_mean']:.2f}  | Slope: {r['ash_slope']:.6f} | Lag-1 AC: {r['ash_autocorr']:.4f}")
