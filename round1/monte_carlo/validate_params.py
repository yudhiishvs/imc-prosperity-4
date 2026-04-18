"""
validate_params.py
------------------
Validates that synthetic data matches real data statistically.

Loads 3 real days + N synthetic days, computes the same statistics for both,
prints a side-by-side comparison table, runs KS tests, and saves a multi-panel
figure to validate_output.png.

Usage:
    python validate_params.py [--n-synth 20] [--seed 42]
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROUND1_DIR  = os.path.join(SCRIPT_DIR, "..", "imc-prosperity-4", "ROUND1")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "params.json")
MC_DATA     = os.path.join(SCRIPT_DIR, "mc_data", "round99")
OUT_PNG     = os.path.join(SCRIPT_DIR, "validate_output.png")
PRODUCTS    = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
DAYS        = [-2, -1, 0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_prices():
    """Returns {product: list of (global_ts, mid, spread, bid_vol, ask_vol)}"""
    rows = {p: [] for p in PRODUCTS}
    for day in DAYS:
        path = os.path.join(ROUND1_DIR, f"prices_round_1_day_{day}.csv")
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for cols in reader:
                if len(cols) < 17:
                    continue
                product = cols[2]
                if product not in rows:
                    continue
                try:
                    mid = float(cols[15])
                except ValueError:
                    continue
                if mid == 0:
                    continue
                ts = int(cols[1])
                global_ts = (day + 2) * 1_000_000 + ts
                try:
                    bid1 = int(cols[3]) if cols[3] else None
                    ask1 = int(cols[9]) if cols[9] else None
                    spread = (ask1 - bid1) if bid1 is not None and ask1 is not None else None
                except (ValueError, TypeError):
                    spread = None
                try:
                    bvol = int(cols[4]) if cols[4] else None
                    avol = int(cols[10]) if cols[10] else None
                except (ValueError, TypeError):
                    bvol, avol = None, None
                rows[product].append((global_ts, mid, spread, bvol, avol))
    return rows


def load_synthetic_prices(n_days: int):
    """Returns {product: list of (global_ts, mid, spread, bid_vol, ask_vol)}"""
    rows = {p: [] for p in PRODUCTS}
    for d in range(n_days):
        path = os.path.join(MC_DATA, f"prices_round_99_day_{d}.csv")
        if not os.path.exists(path):
            break
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for cols in reader:
                if len(cols) < 17:
                    continue
                product = cols[2]
                if product not in rows:
                    continue
                try:
                    mid = float(cols[15])
                except ValueError:
                    continue
                ts = int(cols[1])
                global_ts = d * 1_000_000 + ts
                try:
                    bid1 = int(cols[3]) if cols[3] else None
                    ask1 = int(cols[9]) if cols[9] else None
                    spread = (ask1 - bid1) if bid1 is not None and ask1 is not None else None
                except (ValueError, TypeError):
                    spread = None
                try:
                    bvol = int(cols[4]) if cols[4] else None
                    avol = int(cols[10]) if cols[10] else None
                except (ValueError, TypeError):
                    bvol, avol = None, None
                rows[product].append((global_ts, mid, spread, bvol, avol))
    return rows


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(rows: list, product: str, detrend: bool = False):
    """
    Compute key statistics from a list of (global_ts, mid, spread, bvol, avol).
    Returns a dict of stats.
    """
    if not rows:
        return {}

    mids    = [r[1] for r in rows]
    spreads = [r[2] for r in rows if r[2] is not None and 0 < r[2] < 200]
    bvols   = [r[3] for r in rows if r[3] is not None and r[3] > 0]
    avols   = [r[4] for r in rows if r[4] is not None and r[4] > 0]

    # Returns (tick-to-tick differences)
    returns = [mids[i+1] - mids[i] for i in range(len(mids)-1)]

    # Detrend PEPPER by subtracting linear fit
    if detrend:
        ts_vals = np.array([r[0] for r in rows], dtype=float)
        mid_arr = np.array(mids, dtype=float)
        slope, intercept = np.polyfit(ts_vals, mid_arr, 1)
        residuals = mid_arr - (intercept + slope * ts_vals)
        residual_returns = [float(residuals[i+1] - residuals[i]) for i in range(len(residuals)-1)]
    else:
        slope, intercept = 0.0, 0.0
        residuals = np.array(mids)
        residual_returns = returns

    # Autocorrelation at lag 1
    def autocorr_lag1(series):
        if len(series) < 2:
            return float("nan")
        arr = np.array(series)
        mean = arr.mean()
        denom = np.sum((arr - mean)**2)
        if denom == 0:
            return 0.0
        return float(np.sum((arr[:-1] - mean) * (arr[1:] - mean)) / denom)

    # Half-life via OU regression on residuals
    def ou_halflife(series):
        diffs = np.diff(series)
        gaps  = -series[:-1]   # mean = 0 for residuals
        if len(diffs) < 2:
            return float("nan")
        slope_ou = np.dot(gaps, diffs) / np.dot(gaps, gaps) if np.dot(gaps, gaps) > 0 else 0
        return math.log(2) / slope_ou if slope_ou > 0 else float("inf")

    res_arr = np.array(residuals)

    # Kurtosis (excess)
    def excess_kurtosis(series):
        arr = np.array(series)
        n = len(arr)
        if n < 4:
            return float("nan")
        mean = arr.mean()
        std  = arr.std()
        if std == 0:
            return float("nan")
        return float(np.mean(((arr - mean) / std)**4) - 3)

    return {
        "n_ticks":         len(mids),
        "mid_mean":        statistics.mean(mids),
        "mid_std":         statistics.stdev(mids) if len(mids) > 1 else 0,
        "mid_min":         min(mids),
        "mid_max":         max(mids),
        "slope_per_100ts": float(slope) * 100 if detrend else float(np.polyfit(
                               np.array([r[0] for r in rows], dtype=float),
                               np.array(mids, dtype=float), 1)[0]) * 100,
        "return_mean":     statistics.mean(returns) if returns else 0,
        "return_std":      statistics.stdev(returns) if len(returns) > 1 else 0,
        "return_kurtosis": excess_kurtosis(returns),
        "autocorr_lag1":   autocorr_lag1(returns),
        "halflife_ticks":  ou_halflife(res_arr),
        "spread_median":   statistics.median(spreads) if spreads else float("nan"),
        "spread_mean":     statistics.mean(spreads) if spreads else float("nan"),
        "spread_std":      statistics.stdev(spreads) if len(spreads) > 1 else float("nan"),
        "bvol_median":     statistics.median(bvols) if bvols else float("nan"),
        "avol_median":     statistics.median(avols) if avols else float("nan"),
        "mids":            mids,
        "returns":         returns,
        "spreads":         spreads,
        "bvols":           bvols,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(real_stats: dict, synth_stats: dict, product: str):
    FIELDS = [
        ("n_ticks",         "Ticks",               ".0f"),
        ("mid_mean",        "Mid mean",             ".2f"),
        ("mid_std",         "Mid std",              ".3f"),
        ("mid_min",         "Mid min",              ".1f"),
        ("mid_max",         "Mid max",              ".1f"),
        ("slope_per_100ts", "Slope / 100-ts",       ".6f"),
        ("return_mean",     "Return mean",          ".4f"),
        ("return_std",      "Return std",           ".4f"),
        ("return_kurtosis", "Return kurtosis",      ".2f"),
        ("autocorr_lag1",   "Autocorr lag-1",       ".4f"),
        ("halflife_ticks",  "Half-life (ticks)",    ".2f"),
        ("spread_median",   "Spread median",        ".1f"),
        ("spread_mean",     "Spread mean",          ".2f"),
        ("spread_std",      "Spread std",           ".2f"),
        ("bvol_median",     "Bid vol median",       ".1f"),
        ("avol_median",     "Ask vol median",       ".1f"),
    ]

    print(f"\n{'='*60}")
    print(f"  {product}")
    print(f"{'='*60}")
    print(f"  {'Metric':<24} {'Real':>12} {'Synthetic':>12}  {'Δ%':>8}")
    print(f"  {'-'*24} {'-'*12} {'-'*12}  {'-'*8}")

    for key, label, fmt in FIELDS:
        rv = real_stats.get(key)
        sv = synth_stats.get(key)
        if rv is None or sv is None:
            continue
        try:
            rv_f = float(rv)
            sv_f = float(sv)
            delta = ((sv_f - rv_f) / rv_f * 100) if rv_f != 0 else float("nan")
            flag = " !!!" if abs(delta) > 20 and not math.isnan(delta) else ""
            print(f"  {label:<24} {rv_f:{fmt.replace('.',''):>2}{fmt[-2:]}:>12} {sv_f:{fmt.replace('.',''):>2}{fmt[-2:]}:>12}  {delta:>+7.1f}%{flag}")
        except Exception:
            print(f"  {label:<24} {str(rv):>12} {str(sv):>12}")


def ks_test(real_stats: dict, synth_stats: dict, label: str):
    """Run KS test comparing return distributions."""
    r = real_stats.get("returns", [])
    s = synth_stats.get("returns", [])
    if not r or not s:
        return
    stat, pval = sp_stats.ks_2samp(r, s)
    result = "PASS" if pval > 0.05 else "FAIL"
    print(f"  KS test on returns ({label}): statistic={stat:.4f}, p={pval:.4f}  [{result}]")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(real_rows, synth_rows, n_synth_days: int):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Real vs Synthetic Data Validation", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"real": "#1f77b4", "synth": "#ff7f0e"}

    for col, product in enumerate(PRODUCTS):
        real  = real_rows[product]
        synth = synth_rows[product]
        detrend = (product == "INTARIAN_PEPPER_ROOT")

        real_mids  = [r[1] for r in real]
        synth_mids = [r[1] for r in synth]

        real_ts  = list(range(len(real_mids)))
        synth_ts = list(range(len(synth_mids)))

        # --- Row 0: Full price path ---
        ax = fig.add_subplot(gs[0, col*2 : col*2+2])
        # Show 3 real days and first 3 synthetic days only for clarity
        ticks_per_day = 10_000
        ax.plot(real_ts[:3*ticks_per_day],  real_mids[:3*ticks_per_day],
                color=colors["real"],  lw=0.6, alpha=0.8, label="Real (3 days)")
        ax.plot(synth_ts[:3*ticks_per_day], synth_mids[:3*ticks_per_day],
                color=colors["synth"], lw=0.6, alpha=0.8, label="Synthetic (3 days)")
        ax.set_title(f"{product}\nPrice Path (first 3 days)")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mid price")
        ax.legend(fontsize=8)

        # --- Row 1: Mid price distribution ---
        ax = fig.add_subplot(gs[1, col*2])
        # For PEPPER, detrend both
        if detrend:
            real_arr  = np.array(real_mids)
            synth_arr = np.array(synth_mids)
            real_slope,  real_int  = np.polyfit(np.arange(len(real_arr)),  real_arr,  1)
            synth_slope, synth_int = np.polyfit(np.arange(len(synth_arr)), synth_arr, 1)
            real_plot  = real_arr  - (real_int  + real_slope  * np.arange(len(real_arr)))
            synth_plot = synth_arr - (synth_int + synth_slope * np.arange(len(synth_arr)))
            title = "Detrended mid distribution"
        else:
            real_plot  = np.array(real_mids)
            synth_plot = np.array(synth_mids)
            title = "Mid price distribution"

        ax.hist(real_plot,  bins=60, density=True, alpha=0.6, color=colors["real"],  label="Real")
        ax.hist(synth_plot, bins=60, density=True, alpha=0.6, color=colors["synth"], label="Synth")
        ax.set_title(title)
        ax.set_xlabel("Price")
        ax.legend(fontsize=8)

        # --- Row 1: Return distribution ---
        ax = fig.add_subplot(gs[1, col*2+1])
        real_rets  = np.diff(real_plot)
        synth_rets = np.diff(synth_plot)
        lo = np.percentile(np.concatenate([real_rets, synth_rets]), 1)
        hi = np.percentile(np.concatenate([real_rets, synth_rets]), 99)
        bins = np.linspace(lo, hi, 80)
        ax.hist(real_rets,  bins=bins, density=True, alpha=0.6, color=colors["real"],  label="Real")
        ax.hist(synth_rets, bins=bins, density=True, alpha=0.6, color=colors["synth"], label="Synth")
        ax.set_title("Return distribution")
        ax.set_xlabel("Tick return")
        ax.legend(fontsize=8)

        # --- Row 2: Spread distribution ---
        ax = fig.add_subplot(gs[2, col*2])
        real_spreads  = [r[2] for r in real  if r[2] is not None and 0 < r[2] < 200]
        synth_spreads = [r[2] for r in synth if r[2] is not None and 0 < r[2] < 200]
        all_spreads = real_spreads + synth_spreads
        if all_spreads:
            bins_s = np.arange(min(all_spreads)-0.5, max(all_spreads)+1.5, 1)
            ax.hist(real_spreads,  bins=bins_s, density=True, alpha=0.6,
                    color=colors["real"],  label="Real")
            ax.hist(synth_spreads, bins=bins_s, density=True, alpha=0.6,
                    color=colors["synth"], label="Synth")
        ax.set_title("Spread distribution")
        ax.set_xlabel("Bid-ask spread (ticks)")
        ax.legend(fontsize=8)

        # --- Row 2: Volume distribution ---
        ax = fig.add_subplot(gs[2, col*2+1])
        real_vols  = [r[3] for r in real  if r[3] is not None and r[3] > 0]
        synth_vols = [r[3] for r in synth if r[3] is not None and r[3] > 0]
        all_vols = real_vols + synth_vols
        if all_vols:
            bins_v = np.arange(0.5, min(max(all_vols), 80)+1.5, 1)
            ax.hist(real_vols,  bins=bins_v, density=True, alpha=0.6,
                    color=colors["real"],  label="Real")
            ax.hist(synth_vols, bins=bins_v, density=True, alpha=0.6,
                    color=colors["synth"], label="Synth")
        ax.set_title("Bid vol L1 distribution")
        ax.set_xlabel("Volume")
        ax.legend(fontsize=8)

        # --- Row 3: Autocorrelation of returns ---
        ax = fig.add_subplot(gs[3, col*2])
        max_lag = 20
        real_r_arr  = np.array(real_rets  if len(real_rets)  > max_lag else real_rets)
        synth_r_arr = np.array(synth_rets if len(synth_rets) > max_lag else synth_rets)

        def acf(arr, max_lag):
            arr = arr - arr.mean()
            denom = np.dot(arr, arr)
            return [float(np.dot(arr[lag:], arr[:-lag]) / denom) if lag > 0 else 1.0
                    for lag in range(max_lag+1)]

        lags = list(range(1, max_lag+1))
        real_acf  = acf(real_r_arr,  max_lag)[1:]
        synth_acf = acf(synth_r_arr, max_lag)[1:]
        ax.bar([l - 0.2 for l in lags], real_acf,  width=0.4,
               color=colors["real"],  alpha=0.8, label="Real")
        ax.bar([l + 0.2 for l in lags], synth_acf, width=0.4,
               color=colors["synth"], alpha=0.8, label="Synth")
        conf = 1.96 / math.sqrt(len(real_r_arr))
        ax.axhline( conf, color="gray", lw=0.8, ls="--")
        ax.axhline(-conf, color="gray", lw=0.8, ls="--")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("Return autocorrelation (lags 1-20)")
        ax.set_xlabel("Lag")
        ax.legend(fontsize=8)

        # --- Row 3: QQ plot of returns ---
        ax = fig.add_subplot(gs[3, col*2+1])
        # Sample same size for fair comparison
        n = min(len(real_rets), len(synth_rets), 5000)
        rng = random.Random(99)
        r_sample = sorted(rng.sample(list(real_rets),  min(n, len(real_rets))))
        s_sample = sorted(rng.sample(list(synth_rets), min(n, len(synth_rets))))
        # Quantile-quantile: plot real quantiles vs synth quantiles
        q_points = np.linspace(0, 1, min(200, n))
        r_q = np.quantile(r_sample, q_points)
        s_q = np.quantile(s_sample, q_points)
        ax.scatter(r_q, s_q, s=4, alpha=0.6, color="#2ca02c")
        lo_q = min(r_q.min(), s_q.min())
        hi_q = max(r_q.max(), s_q.max())
        ax.plot([lo_q, hi_q], [lo_q, hi_q], "k--", lw=1, label="y=x (perfect match)")
        ax.set_title("QQ plot: Real vs Synthetic returns")
        ax.set_xlabel("Real return quantile")
        ax.set_ylabel("Synthetic return quantile")
        ax.legend(fontsize=8)

        # --- Row 4: Rolling mean and std of mid price ---
        ax = fig.add_subplot(gs[4, col*2])
        window = 500
        real_roll_mean  = [statistics.mean(real_mids[max(0,i-window):i+1])
                           for i in range(0, len(real_mids), 50)]
        synth_roll_mean = [statistics.mean(synth_mids[max(0,i-window):i+1])
                           for i in range(0, len(synth_mids), 50)]
        ax.plot(real_roll_mean,  color=colors["real"],  lw=0.8, alpha=0.8, label="Real")
        ax.plot(synth_roll_mean, color=colors["synth"], lw=0.8, alpha=0.8, label="Synth")
        ax.set_title(f"Rolling mean mid (window={window})")
        ax.set_xlabel("Tick (×50)")
        ax.set_ylabel("Mean price")
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[4, col*2+1])
        real_roll_std  = []
        synth_roll_std = []
        for i in range(0, min(len(real_mids), len(synth_mids)), 50):
            chunk_r = real_mids[max(0,i-window):i+1]
            chunk_s = synth_mids[max(0,i-window):i+1]
            real_roll_std.append(statistics.stdev(chunk_r) if len(chunk_r) > 1 else 0)
            synth_roll_std.append(statistics.stdev(chunk_s) if len(chunk_s) > 1 else 0)
        ax.plot(real_roll_std,  color=colors["real"],  lw=0.8, alpha=0.8, label="Real")
        ax.plot(synth_roll_std, color=colors["synth"], lw=0.8, alpha=0.8, label="Synth")
        ax.set_title(f"Rolling std mid (window={window})")
        ax.set_xlabel("Tick (×50)")
        ax.set_ylabel("Std price")
        ax.legend(fontsize=8)

    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\nPlot saved → {OUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synth", type=int, default=20,
                        help="Number of synthetic days to compare against")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--regen",   action="store_true",
                        help="Regenerate synthetic data before validating")
    args = parser.parse_args()

    # Regenerate synthetic data if requested or missing
    if args.regen or not os.path.exists(
            os.path.join(MC_DATA, f"prices_round_99_day_{args.n_synth - 1}.csv")):
        import subprocess
        print(f"Generating {args.n_synth} synthetic days...")
        if not os.path.exists(PARAMS_FILE):
            subprocess.run([sys.executable,
                            os.path.join(SCRIPT_DIR, "estimate_params.py")], check=True)
        subprocess.run([sys.executable,
                        os.path.join(SCRIPT_DIR, "generate_data.py"),
                        "--n-days", str(args.n_synth),
                        "--seed",   str(args.seed)], check=True)

    print("Loading data...")
    real_rows  = load_real_prices()
    synth_rows = load_synthetic_prices(args.n_synth)

    for product in PRODUCTS:
        detrend = (product == "INTARIAN_PEPPER_ROOT")
        rs = compute_stats(real_rows[product],  product, detrend)
        ss = compute_stats(synth_rows[product], product, detrend)
        print_comparison(rs, ss, product)
        ks_test(rs, ss, product)

    print("\n  NOTE: KS test FAIL means return distributions differ significantly (p<0.05).")
    print("        Large Δ% (!!!) flags metrics where synthetic diverges >20% from real.")

    print("\nGenerating plots...")
    make_plots(real_rows, synth_rows, args.n_synth)


if __name__ == "__main__":
    main()
