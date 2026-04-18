"""
estimate_params.py  (Round 2)
------------------------------
Fits statistical process parameters for ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT
from the 3 real Round-2 price/trade CSV files.

Models
------
  ASH    → OU latent price + bid-ask bounce (Roll 1984)
             mid[t] = latent[t] + bounce_c * q[t]
             latent[t+1] = latent[t] + θ*(μ - latent[t]) + σ_eff * t_noise

  PEPPER → Linear trend + OU residual

Bot archetypes estimated
------------------------
  Desperate Liquidator  — crosses the spread unconditionally
      liquidator_rate_per_100ts : avg number of liquidator trades per 100-ts window
      Detected as trades whose price matches bid_price_1 or ask_price_1 exactly.

  Mean-Reverter (Oscillator) — trades against deviation from fair value
      reverter_rate_per_100ts : avg number of mean-reverter trades per 100-ts window
      Detected as all other trades (inside or at mid).

Order-book stats
  imbalance_mean / imbalance_std : from (bid_vol1 - ask_vol1)/(bid_vol1 + ask_vol1)
  l1_trade_frac : fraction of trades at L1 (vs L2/L3)

Usage:
    python estimate_params.py
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
OUT_FILE   = os.path.join(SCRIPT_DIR, "params.json")

DAYS     = [-1, 0, 1]
PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
ASH_MU   = 10_000.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_prices():
    rows = defaultdict(list)
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_2_day_{day}.csv")
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for cols in reader:
                if len(cols) < 17:
                    continue
                product = cols[2]
                ts      = int(cols[1])
                try:
                    mid = float(cols[15])
                except ValueError:
                    continue
                if mid == 0:
                    continue

                try:
                    bid1 = float(cols[3]) if cols[3] else None
                    ask1 = float(cols[9]) if cols[9] else None
                    spread = (ask1 - bid1) if (bid1 is not None and ask1 is not None) else None
                except (ValueError, TypeError):
                    bid1 = ask1 = spread = None

                try:
                    bid2 = float(cols[5]) if cols[5] else None
                    ask2 = float(cols[11]) if cols[11] else None
                except (ValueError, TypeError):
                    bid2 = ask2 = None

                try:
                    bvol1 = int(cols[4]) if cols[4] else None
                    avol1 = int(cols[10]) if cols[10] else None
                except (ValueError, TypeError):
                    bvol1 = avol1 = None

                imbalance = None
                if bvol1 is not None and avol1 is not None and (bvol1 + avol1) > 0:
                    imbalance = (bvol1 - avol1) / (bvol1 + avol1)

                rows[product].append({
                    "day": day, "ts": ts,
                    "ts_global": (day + 1) * 1_000_000 + ts,
                    "mid": mid,
                    "bid1": bid1, "ask1": ask1,
                    "bid2": bid2, "ask2": ask2,
                    "spread": spread,
                    "bvol1": bvol1, "avol1": avol1,
                    "imbalance": imbalance,
                })
    return rows


def load_trades():
    rows = defaultdict(list)
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"trades_round_2_day_{day}.csv")
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
                    rows[sym].append({"day": day, "ts": ts, "price": price, "qty": qty})
                except (ValueError, IndexError):
                    continue
    return rows


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def fit_ou(series, mu=0.0):
    diffs = [series[i+1] - series[i] for i in range(len(series)-1)]
    gaps  = [mu - series[i]          for i in range(len(series)-1)]
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
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var = sum((xs[i] - mean_x)**2 for i in range(n))
    slope = cov / var if var > 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


# ---------------------------------------------------------------------------
# ε (price aggression beyond half-spread) estimation
# ---------------------------------------------------------------------------

def estimate_epsilon(trade_rows, price_rows, product):
    """
    For each observed trade compute:
        ε = |trade_price - mid| - spread/2

    ε > 0 : bot traded past the best bid/ask (crosses our passive quote)
    ε < 0 : bot traded inside the half-spread (misses our quote)
    ε = 0 : bot traded exactly at the best bid/ask

    Returns (epsilon_mean, epsilon_std).
    """
    # Build sorted price list for fast backward-lookup
    price_list = sorted(
        price_rows[product],
        key=lambda r: (r["day"], r["ts"]),
    )
    # Index by (day, ts) for O(1) lookup of closest prior tick
    from bisect import bisect_right
    keys = [(r["day"], r["ts"]) for r in price_list]

    epsilons = []
    for t in trade_rows.get(product, []):
        key = (t["day"], t["ts"])
        idx = bisect_right(keys, key) - 1
        if idx < 0:
            continue
        pr = price_list[idx]
        if pr["mid"] is None or pr["spread"] is None or pr["spread"] <= 0:
            continue
        half_spread = pr["spread"] / 2.0
        dev = abs(t["price"] - pr["mid"])
        epsilons.append(dev - half_spread)

    if len(epsilons) < 2:
        return 0.0, 1.0

    mean = sum(epsilons) / len(epsilons)
    std  = (sum((e - mean)**2 for e in epsilons) / (len(epsilons) - 1)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# Bot archetype classification
# ---------------------------------------------------------------------------

def classify_trades(trade_rows, price_rows):
    """
    Classify each trade as Desperate Liquidator or Mean-Reverter.

    Liquidator: trade price matches bid_price_1 or ask_price_1 at that timestamp
                (crossed the spread at the best visible level).
    Reverter:   everything else (inside spread, at mid, or deep level).

    Returns per-product dict with:
      liquidator_count, reverter_count, l1_count, l2plus_count, total_windows
    """
    # Build price lookup: product -> ts -> row
    price_lookup = defaultdict(dict)
    for product, rows in price_rows.items():
        for r in rows:
            price_lookup[product][r["ts"] + r["day"] * 1_000_000] = r  # keyed by global ts

    result = {}
    for product in PRODUCTS:
        trades = trade_rows.get(product, [])
        price_map = price_lookup[product]

        liq_count = 0
        rev_count = 0
        l1_count  = 0
        l2p_count = 0

        for t in trades:
            g_ts = t["ts"] + (t["day"] + 1) * 1_000_000
            # Find the closest price tick at or before this trade
            candidates = [(k, v) for k, v in price_map.items() if k <= g_ts]
            if not candidates:
                rev_count += 1
                continue
            _, pr = max(candidates, key=lambda x: x[0])

            bid1 = pr["bid1"]
            ask1 = pr["ask1"]
            bid2 = pr["bid2"]
            ask2 = pr["ask2"]

            price = t["price"]

            is_l1 = (bid1 is not None and abs(price - bid1) < 0.6) or \
                    (ask1 is not None and abs(price - ask1) < 0.6)
            is_l2 = (bid2 is not None and abs(price - bid2) < 0.6) or \
                    (ask2 is not None and abs(price - ask2) < 0.6)

            is_liq = is_l1 or is_l2  # crossing the visible spread

            if is_liq:
                liq_count += 1
            else:
                rev_count += 1

            if is_l1:
                l1_count += 1
            elif is_l2:
                l2p_count += 1

        total_windows = 3 * 10_000  # 3 days * 10000 ticks/day
        result[product] = {
            "liquidator_count":      liq_count,
            "reverter_count":        rev_count,
            "l1_count":              l1_count,
            "l2plus_count":          l2p_count,
            "total_windows":         total_windows,
            "liquidator_rate":       liq_count / total_windows,
            "reverter_rate":         rev_count / total_windows,
            "l1_trade_frac":         l1_count  / max(1, l1_count + l2p_count),
        }
    return result


# ---------------------------------------------------------------------------
# Main estimation
# ---------------------------------------------------------------------------

def estimate():
    print("Loading Round 2 data...")
    price_rows = load_prices()
    trade_rows = load_trades()

    print("Classifying bot archetypes...")
    bot_stats = classify_trades(trade_rows, price_rows)

    print("Estimating ε (price aggression beyond half-spread)...")
    ash_eps_mean, ash_eps_std   = estimate_epsilon(trade_rows, price_rows, "ASH_COATED_OSMIUM")
    pep_eps_mean, pep_eps_std   = estimate_epsilon(trade_rows, price_rows, "INTARIAN_PEPPER_ROOT")

    params = {}

    # ------------------------------------------------------------------
    # ASH_COATED_OSMIUM — OU around 10000
    # ------------------------------------------------------------------
    ash = price_rows["ASH_COATED_OSMIUM"]
    ash_mids    = [r["mid"]    for r in ash]
    ash_spreads = [r["spread"] for r in ash if r["spread"] is not None and 0 < r["spread"] < 100]
    ash_bvols   = [r["bvol1"]  for r in ash if r["bvol1"] is not None and r["bvol1"] > 0]
    ash_avols   = [r["avol1"]  for r in ash if r["avol1"] is not None and r["avol1"] > 0]
    ash_imbs    = [r["imbalance"] for r in ash if r["imbalance"] is not None]

    ash_rets    = [ash_mids[i+1] - ash_mids[i] for i in range(len(ash_mids)-1)]
    ash_r_std   = statistics.stdev(ash_rets)
    ash_ac_n    = sum(ash_rets[i] * ash_rets[i+1] for i in range(len(ash_rets)-1))
    ash_ac_d    = sum(x*x for x in ash_rets)
    ash_autocorr = ash_ac_n / ash_ac_d if ash_ac_d > 0 else 0
    ash_mid_std  = statistics.stdev(ash_mids)

    ash_bounce_c2   = max(0.0, -ash_autocorr * ash_r_std**2)
    ash_bounce_c    = math.sqrt(ash_bounce_c2)
    ash_sigma_eff2  = max(1e-6, ash_r_std**2 - 2 * ash_bounce_c2)
    ash_sigma_eff   = math.sqrt(ash_sigma_eff2)
    denom_theta     = 2.0 * (ash_mid_std**2 - ash_bounce_c2)
    ash_theta_lat   = ash_sigma_eff2 / denom_theta if denom_theta > 0 else 0.001
    ash_kurt        = excess_kurtosis(ash_rets)
    ash_t_df        = max(4.5, 4.0 + 6.0 / ash_kurt) if ash_kurt > 0 else 30.0
    ash_halflife    = math.log(2) / ash_theta_lat if ash_theta_lat > 0 else float("inf")

    ash_bots        = bot_stats["ASH_COATED_OSMIUM"]
    ash_trades      = trade_rows.get("ASH_COATED_OSMIUM", [])
    ash_sizes       = sorted([t["qty"] for t in ash_trades])
    ash_imb_mean    = sum(ash_imbs) / len(ash_imbs) if ash_imbs else 0.0
    ash_imb_std     = statistics.stdev(ash_imbs) if len(ash_imbs) > 1 else 0.1

    ash_spreads.sort()
    ash_bvols.sort()
    ash_avols.sort()

    params["ASH_COATED_OSMIUM"] = {
        "model":                   "ou_bounce",
        "mu":                      ASH_MU,
        "theta":                   ash_theta_lat,
        "sigma_eff":               ash_sigma_eff,
        "bounce_c":                ash_bounce_c,
        "t_df":                    ash_t_df,
        "halflife_ticks":          ash_halflife,
        "spread_dist":             ash_spreads,
        "bid_vol_dist":            ash_bvols,
        "ask_vol_dist":            ash_avols,
        "trade_size_dist":         ash_sizes if ash_sizes else [5],
        "imbalance_mean":          ash_imb_mean,
        "imbalance_std":           ash_imb_std,
        "liquidator_rate_per_100ts": ash_bots["liquidator_rate"],
        "reverter_rate_per_100ts":   ash_bots["reverter_rate"],
        "l1_trade_frac":             ash_bots["l1_trade_frac"],
        "epsilon_mean":              ash_eps_mean,
        "epsilon_std":               ash_eps_std,
    }

    print("\n=== ASH_COATED_OSMIUM ===")
    print(f"  μ (fair value)        = {ASH_MU}")
    print(f"  θ_latent              = {ash_theta_lat:.6f}  → half-life ≈ {ash_halflife:.1f} ticks")
    print(f"  bounce c              = {ash_bounce_c:.4f}")
    print(f"  σ_eff                 = {ash_sigma_eff:.4f}")
    print(f"  t-df                  = {ash_t_df:.2f}  (kurtosis={ash_kurt:.2f})")
    pred_ac  = -ash_bounce_c**2 / (ash_sigma_eff**2 + 2*ash_bounce_c**2)
    pred_std = math.sqrt(ash_sigma_eff**2 + 2*ash_bounce_c**2)
    print(f"  Predicted autocorr    = {pred_ac:.4f}  (real: {ash_autocorr:.4f})")
    print(f"  Predicted r_std       = {pred_std:.4f}  (real: {ash_r_std:.4f})")
    print(f"  Spread median         = {statistics.median(ash_spreads) if ash_spreads else 'N/A'}")
    print(f"  Imbalance mean/std    = {ash_imb_mean:.4f} / {ash_imb_std:.4f}")
    print(f"  Liquidator rate       = {ash_bots['liquidator_rate']:.4f} trades/100ts")
    print(f"  Reverter rate         = {ash_bots['reverter_rate']:.4f} trades/100ts")
    print(f"  L1 trade fraction     = {ash_bots['l1_trade_frac']:.2%}")
    print(f"  ε mean / std          = {ash_eps_mean:.4f} / {ash_eps_std:.4f}  "
          f"(+ε → past quote, -ε → inside spread)")

    # ------------------------------------------------------------------
    # INTARIAN_PEPPER_ROOT — Trend + OU residual
    # ------------------------------------------------------------------
    pepper = price_rows["INTARIAN_PEPPER_ROOT"]
    pepper_sorted = sorted(pepper, key=lambda r: (r["day"], r["ts"]))
    pep_ts_global = [r["ts_global"] for r in pepper_sorted]
    pep_mids      = [r["mid"]       for r in pepper_sorted]

    slope, intercept = fit_linear(pep_ts_global, pep_mids)
    residuals = [pep_mids[i] - (intercept + slope * pep_ts_global[i]) for i in range(len(pep_mids))]
    pep_theta, pep_sigma = fit_ou(residuals, mu=0.0)
    pep_halflife = math.log(2) / pep_theta if pep_theta > 0 else float("inf")

    pep_rets = [pep_mids[i+1] - pep_mids[i] for i in range(len(pep_mids)-1)]
    pep_kurt = excess_kurtosis(pep_rets)
    pep_t_df = max(4.5, 4.0 + 6.0 / pep_kurt) if pep_kurt > 0 else 30.0

    pep_spreads = [r["spread"] for r in pepper if r["spread"] is not None and 0 < r["spread"] < 500]
    pep_bvols   = [r["bvol1"]  for r in pepper if r["bvol1"] is not None and r["bvol1"] > 0]
    pep_avols   = [r["avol1"]  for r in pepper if r["avol1"] is not None and r["avol1"] > 0]
    pep_imbs    = [r["imbalance"] for r in pepper if r["imbalance"] is not None]
    pep_spreads.sort()
    pep_bvols.sort()
    pep_avols.sort()

    pep_bots   = bot_stats["INTARIAN_PEPPER_ROOT"]
    pep_trades = trade_rows.get("INTARIAN_PEPPER_ROOT", [])
    pep_sizes  = sorted([t["qty"] for t in pep_trades])
    pep_imb_mean = sum(pep_imbs) / len(pep_imbs) if pep_imbs else 0.0
    pep_imb_std  = statistics.stdev(pep_imbs) if len(pep_imbs) > 1 else 0.1

    # Residual std for normalizing reverter signal
    res_std = statistics.stdev(residuals) if len(residuals) > 1 else 1.0

    params["INTARIAN_PEPPER_ROOT"] = {
        "model":                   "trend_ou",
        "slope":                   slope,
        "intercept":               intercept,
        "start_mid":               pep_mids[-1] if pep_mids else 12000.0,
        "ou_theta":                pep_theta,
        "ou_sigma":                pep_sigma,
        "ou_residual_std":         res_std,
        "t_df":                    pep_t_df,
        "halflife_ticks":          pep_halflife,
        "spread_dist":             pep_spreads,
        "bid_vol_dist":            pep_bvols,
        "ask_vol_dist":            pep_avols,
        "trade_size_dist":         pep_sizes if pep_sizes else [5],
        "imbalance_mean":          pep_imb_mean,
        "imbalance_std":           pep_imb_std,
        "liquidator_rate_per_100ts": pep_bots["liquidator_rate"],
        "reverter_rate_per_100ts":   pep_bots["reverter_rate"],
        "l1_trade_frac":             pep_bots["l1_trade_frac"],
        "epsilon_mean":              pep_eps_mean,
        "epsilon_std":               pep_eps_std,
    }

    print("\n=== INTARIAN_PEPPER_ROOT ===")
    print(f"  slope (per global ts) = {slope:.8f}")
    print(f"  intercept             = {intercept:.2f}")
    print(f"  start_mid             = {pep_mids[-1]:.2f}")
    print(f"  OU θ on residuals     = {pep_theta:.6f}  → half-life ≈ {pep_halflife:.2f} ticks")
    print(f"  OU σ on residuals     = {pep_sigma:.4f}")
    print(f"  Residual std          = {res_std:.4f}")
    print(f"  t-df                  = {pep_t_df:.2f}  (kurtosis={pep_kurt:.2f})")
    print(f"  Imbalance mean/std    = {pep_imb_mean:.4f} / {pep_imb_std:.4f}")
    print(f"  Liquidator rate       = {pep_bots['liquidator_rate']:.4f} trades/100ts")
    print(f"  Reverter rate         = {pep_bots['reverter_rate']:.4f} trades/100ts")
    print(f"  L1 trade fraction     = {pep_bots['l1_trade_frac']:.2%}")
    print(f"  ε mean / std          = {pep_eps_mean:.4f} / {pep_eps_std:.4f}")

    with open(OUT_FILE, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {OUT_FILE}")
    return params


if __name__ == "__main__":
    estimate()
