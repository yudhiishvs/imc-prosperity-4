"""
estimate_params.py
------------------
Fits statistical process parameters for ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT
from the 3 real price/trade CSV files in ROUND1/.

Models:
  ASH    → OU latent price + bid-ask bounce:
             mid[t] = latent[t] + c * q[t]   (q[t] ∈ {-1, +1} i.i.d.)
             latent[t+1] = latent[t] + θ*(μ-latent[t]) + σ_eff * t_noise
             Bounce c and σ_eff calibrated from Roll (1984) to match autocorr & return_std.
  PEPPER → Linear trend + OU residual:  X(t) = base + slope*t + Z(t)

Noise: Student-t(df) normalized to unit variance. df fitted from excess kurtosis.

Outputs params.json in the same directory.

Usage:
    python estimate_params.py
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict

ROUND1_DIR = os.path.join(os.path.dirname(__file__), "..", "imc-prosperity-4", "ROUND1")
OUT_FILE   = os.path.join(os.path.dirname(__file__), "params.json")

DAYS       = [-2, -1, 0]
PRODUCTS   = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
ASH_MU     = 10000.0   # known constant fair value


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_prices():
    """Returns {product: [(timestamp_global, mid_price, spread, bid_vol1, ask_vol1), ...]}
    Timestamps are made globally monotone across days by adding day_offset.
    """
    rows = defaultdict(list)
    for day in DAYS:
        path = os.path.join(ROUND1_DIR, f"prices_round_1_day_{day}.csv")
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # header
            for cols in reader:
                if len(cols) < 17:
                    continue
                product   = cols[2]
                ts        = int(cols[1])
                # global timestamp: shift each day so they don't overlap
                # day -2 → 0-999900, day -1 → 1000000-1999900, day 0 → 2000000-2999900
                day_offset = (day + 2) * 1_000_000
                ts_global = day_offset + ts
                try:
                    mid = float(cols[15])
                except ValueError:
                    continue
                if mid == 0:
                    continue

                # spread
                try:
                    bid1 = int(cols[3]) if cols[3] else None
                    ask1 = int(cols[9]) if cols[9] else None
                    spread = (ask1 - bid1) if (bid1 is not None and ask1 is not None) else None
                except (ValueError, TypeError):
                    spread = None

                # volumes
                try:
                    bvol = int(cols[4]) if cols[4] else None
                    avol = int(cols[10]) if cols[10] else None
                except (ValueError, TypeError):
                    bvol, avol = None, None

                rows[product].append((ts_global, ts, day, mid, spread, bvol, avol))

    return rows


def load_trades():
    """Returns {product: [(timestamp, price, qty), ...]}"""
    rows = defaultdict(list)
    for day in DAYS:
        path = os.path.join(ROUND1_DIR, f"trades_round_1_day_{day}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for cols in reader:
                if len(cols) < 7:
                    continue
                try:
                    ts    = int(cols[0])
                    sym   = cols[3]
                    price = float(cols[5])
                    qty   = int(cols[6])
                    rows[sym].append((ts, price, qty))
                except (ValueError, IndexError):
                    continue
    return rows


# ---------------------------------------------------------------------------
# OU regression helper
# ---------------------------------------------------------------------------

def fit_ou(series, mu=0.0):
    """
    Fit OU parameters θ and σ from a 1-D time series using OLS.
    Regresses (X[t+1] - X[t]) on (mu - X[t]).
    Returns (theta, sigma_per_step).
    """
    diffs  = [series[i+1] - series[i] for i in range(len(series)-1)]
    gaps   = [mu - series[i]           for i in range(len(series)-1)]

    n = len(diffs)
    if n < 2:
        return 0.0, 0.0

    mean_g = sum(gaps)  / n
    mean_d = sum(diffs) / n

    cov = sum((gaps[i] - mean_g) * (diffs[i] - mean_d) for i in range(n))
    var = sum((gaps[i] - mean_g)**2 for i in range(n))

    theta = cov / var if var > 0 else 0.0

    residuals = [diffs[i] - theta * gaps[i] for i in range(n)]
    sigma = statistics.stdev(residuals) if len(residuals) > 1 else 0.0

    return theta, sigma


def excess_kurtosis(series):
    n = len(series)
    if n < 4:
        return 0.0
    mean = sum(series) / n
    std  = (sum((x - mean)**2 for x in series) / n) ** 0.5
    if std == 0:
        return 0.0
    return sum(((x - mean) / std)**4 for x in series) / n - 3


def fit_linear(xs, ys):
    """Simple OLS linear regression. Returns (slope, intercept)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var = sum((xs[i] - mean_x)**2 for i in range(n))
    slope = cov / var if var > 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


# ---------------------------------------------------------------------------
# Main estimation
# ---------------------------------------------------------------------------

def estimate():
    price_rows = load_prices()
    trade_rows = load_trades()

    params = {}

    # ------------------------------------------------------------------
    # ASH_COATED_OSMIUM — Ornstein-Uhlenbeck around 10000
    # ------------------------------------------------------------------
    ash = price_rows["ASH_COATED_OSMIUM"]
    ash_mids    = [r[3] for r in ash]
    ash_spreads = [r[4] for r in ash if r[4] is not None and 0 < r[4] < 100]
    ash_bvols   = [r[5] for r in ash if r[5] is not None and r[5] > 0]
    ash_avols   = [r[6] for r in ash if r[6] is not None and r[6] > 0]

    ash_theta, _ = fit_ou(ash_mids, mu=ASH_MU)

    # Bid-ask bounce calibration (Roll 1984) — solve 3 equations simultaneously:
    #   (1) autocorr = -c² / (σ_eff² + 2c²)         [return autocorrelation]
    #   (2) return_std² = σ_eff² + 2c²               [return variance]
    #   (3) mid_std²  ≈ σ_eff²/(2θ_latent) + c²      [steady-state mid price variance]
    #
    # From (1)+(2):  c² = -autocorr * return_std²
    #                σ_eff² = return_std² - 2c²
    # From (3):      θ_latent = σ_eff² / (2*(mid_std² - c²))
    #
    # This correctly separates the fast bounce (tick-to-tick) from the slow latent
    # OU drift (hundreds of ticks), giving the right mid price range.
    ash_rets = [ash_mids[i+1] - ash_mids[i] for i in range(len(ash_mids)-1)]
    ash_r_std  = statistics.stdev(ash_rets)
    ash_ac_n   = sum(ash_rets[i] * ash_rets[i+1] for i in range(len(ash_rets)-1))
    ash_ac_d   = sum(x*x for x in ash_rets)
    ash_autocorr = ash_ac_n / ash_ac_d if ash_ac_d > 0 else 0
    ash_mid_std  = statistics.stdev(ash_mids)

    ash_bounce_c2 = max(0.0, -ash_autocorr * ash_r_std**2)
    ash_bounce_c  = math.sqrt(ash_bounce_c2)
    ash_sigma_eff2 = max(1e-6, ash_r_std**2 - 2 * ash_bounce_c2)
    ash_sigma_eff  = math.sqrt(ash_sigma_eff2)

    # θ from mid_std (equation 3)
    denom_theta = 2.0 * (ash_mid_std**2 - ash_bounce_c2)
    ash_theta_latent = ash_sigma_eff2 / denom_theta if denom_theta > 0 else ash_theta

    # Student-t degrees of freedom from excess kurtosis: df = 4 + 6/kurt
    ash_kurt = excess_kurtosis(ash_rets)
    ash_t_df = max(4.5, 4.0 + 6.0 / ash_kurt) if ash_kurt > 0 else 30.0

    # Spread and volume distributions stored as sorted lists for sampling
    ash_spreads.sort()
    ash_bvols.sort()
    ash_avols.sort()

    ash_latent_halflife = math.log(2) / ash_theta_latent if ash_theta_latent > 0 else float("inf")

    params["ASH_COATED_OSMIUM"] = {
        "model": "ou_bounce",
        "mu": ASH_MU,
        "theta": ash_theta_latent,
        "sigma_eff": ash_sigma_eff,
        "bounce_c": ash_bounce_c,
        "t_df": ash_t_df,
        "halflife_ticks": ash_latent_halflife,
        "spread_dist": ash_spreads,
        "bid_vol_dist": ash_bvols,
        "ask_vol_dist": ash_avols,
    }

    print("=== ASH_COATED_OSMIUM ===")
    print(f"  μ (fair value)       = {ASH_MU}")
    print(f"  θ_latent             = {ash_theta_latent:.6f}  → half-life ≈ {ash_latent_halflife:.1f} ticks")
    print(f"  bounce c             = {ash_bounce_c:.4f}  (Roll bid-ask bounce)")
    print(f"  σ_eff (latent noise) = {ash_sigma_eff:.4f}")
    print(f"  t-df (kurtosis)      = {ash_t_df:.2f}  (kurtosis={ash_kurt:.2f})")
    pred_ac  = -ash_bounce_c**2 / (ash_sigma_eff**2 + 2*ash_bounce_c**2)
    pred_std = math.sqrt(ash_sigma_eff**2 + 2*ash_bounce_c**2)
    pred_mid_std = math.sqrt(ash_sigma_eff**2 / (2*ash_theta_latent) + ash_bounce_c**2)
    print(f"  Predicted autocorr   = {pred_ac:.4f}  (real: {ash_autocorr:.4f})")
    print(f"  Predicted r_std      = {pred_std:.4f}  (real: {ash_r_std:.4f})")
    print(f"  Predicted mid_std    = {pred_mid_std:.4f}  (real: {ash_mid_std:.4f})")
    print(f"  Spread: median={statistics.median(ash_spreads)}, mean={statistics.mean(ash_spreads):.2f}")
    print(f"  Bid vol: median={statistics.median(ash_bvols)}, mean={statistics.mean(ash_bvols):.2f}")

    # ------------------------------------------------------------------
    # INTARIAN_PEPPER_ROOT — Linear trend + OU residual
    # ------------------------------------------------------------------
    pepper = price_rows["INTARIAN_PEPPER_ROOT"]

    # Use within-day timestamp (col[1]) and construct a global tick index
    # so the linear trend is continuous across days
    pepper_ts   = []
    pepper_mids = []
    tick_idx = 0
    prev_day = None
    for r in sorted(pepper, key=lambda x: (x[2], x[1])):  # sort by day, then ts
        ts_global_ticks = tick_idx
        pepper_ts.append(float(r[1] + (r[2] + 2) * 1_000_000))  # use same global ts as loaded
        pepper_mids.append(r[3])
        tick_idx += 1

    # Fit trend on actual timestamps (not tick index) for interpretability
    ts_vals = [r[0] for r in sorted(pepper, key=lambda x: (x[2], x[1]))]
    slope, intercept = fit_linear(ts_vals, pepper_mids)

    # Residuals
    residuals = [pepper_mids[i] - (intercept + slope * ts_vals[i]) for i in range(len(pepper_mids))]

    # Fit OU on residuals (mean-revert to 0)
    pep_theta, pep_sigma = fit_ou(residuals, mu=0.0)
    pep_halflife = math.log(2) / pep_theta if pep_theta > 0 else float("inf")

    # Student-t df from excess kurtosis of returns
    pep_rets = [pepper_mids[i+1] - pepper_mids[i] for i in range(len(pepper_mids)-1)]
    pep_kurt = excess_kurtosis(pep_rets)
    pep_t_df = max(4.5, 4.0 + 6.0 / pep_kurt) if pep_kurt > 0 else 30.0

    pepper_spreads = [r[4] for r in pepper if r[4] is not None and 0 < r[4] < 200]
    pepper_bvols   = [r[5] for r in pepper if r[5] is not None and r[5] > 0]
    pepper_avols   = [r[6] for r in pepper if r[6] is not None and r[6] > 0]
    pepper_spreads.sort()
    pepper_bvols.sort()
    pepper_avols.sort()

    # Starting price for synthetic generation: last real mid price
    pepper_start = pepper_mids[-1] if pepper_mids else 12000.0

    params["INTARIAN_PEPPER_ROOT"] = {
        "model": "trend_ou",
        "slope": slope,
        "intercept": intercept,
        "start_mid": pepper_start,
        "ou_theta": pep_theta,
        "ou_sigma": pep_sigma,
        "t_df": pep_t_df,
        "halflife_ticks": pep_halflife,
        "spread_dist": pepper_spreads,
        "bid_vol_dist": pepper_bvols,
        "ask_vol_dist": pepper_avols,
    }

    print("\n=== INTARIAN_PEPPER_ROOT ===")
    print(f"  slope (per global tick) = {slope:.8f}")
    print(f"  slope (per 100-ts step) = {slope * 100:.6f}")
    print(f"  intercept               = {intercept:.2f}")
    print(f"  start_mid               = {pepper_start:.2f}")
    print(f"  OU θ on residuals       = {pep_theta:.6f}  → half-life ≈ {pep_halflife:.2f} ticks")
    print(f"  OU σ on residuals       = {pep_sigma:.4f}")
    print(f"  t-df (kurtosis)         = {pep_t_df:.2f}  (kurtosis={pep_kurt:.2f})")
    print(f"  Spread: median={statistics.median(pepper_spreads)}, mean={statistics.mean(pepper_spreads):.2f}")

    # ------------------------------------------------------------------
    # Trade distributions (both products)
    # ------------------------------------------------------------------
    for product in PRODUCTS:
        trades = trade_rows.get(product, [])
        if not trades:
            params[product]["trade_rate_per_100ts"] = 0.05
            params[product]["trade_size_dist"] = [5]
            params[product]["price_offset_dist"] = [0]
            continue

        # Count trades per 100-ts window across all days
        ts_counts = defaultdict(int)
        for ts, price, qty in trades:
            window = (ts // 100) * 100
            ts_counts[window] += 1

        # Total 100-ts windows across 3 days
        total_windows = 3 * 10000
        trade_rate = len(trades) / total_windows

        sizes   = sorted([qty for _, _, qty in trades])
        offsets = []  # price offset from mid not easily available here; use 0-centred distribution

        params[product]["trade_rate_per_100ts"] = trade_rate
        params[product]["trade_size_dist"]       = sizes
        params[product]["price_offset_dist"]     = [0]  # trades occur near mid; backtester handles matching

    print(f"\n  ASH   trade rate: {params['ASH_COATED_OSMIUM']['trade_rate_per_100ts']:.4f} trades/100ts")
    print(f"  PEPPER trade rate: {params['INTARIAN_PEPPER_ROOT']['trade_rate_per_100ts']:.4f} trades/100ts")

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {OUT_FILE}")

    return params


if __name__ == "__main__":
    estimate()
