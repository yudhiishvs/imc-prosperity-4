"""
generate_data.py
----------------
Generates N synthetic trading days for both products using fitted parameters.
Writes CSVs to mc_data/round99/ in the exact format the backtester expects.

Usage:
    python generate_data.py --n-days 50 [--seed 42]
    python generate_data.py --n-days 50 --seed 42 --params params.json
"""

import argparse
import json
import math
import os
import random

SCRIPT_DIR = os.path.dirname(__file__)
MC_DATA    = os.path.join(SCRIPT_DIR, "mc_data", "round99")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "params.json")

TICKS_PER_DAY = 10_000          # timestamps 0 .. 999900 step 100
TS_STEP       = 100
PRODUCTS      = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_empirical(dist: list, rng: random.Random):
    """Uniform sample from empirical distribution list."""
    return rng.choice(dist)


def sample_t(df: float, rng: random.Random) -> float:
    """
    Sample from Student-t(df) normalized to unit variance.
    Uses the representation t = Z / sqrt(Chi2(df)/df), Z ~ N(0,1).
    Normalized so Var = 1: divide by sqrt(df/(df-2)).
    For df≈6 this produces fat tails with excess kurtosis ≈ 6/(df-4) ≈ 3.
    """
    idf = int(df)
    z   = rng.gauss(0, 1)
    chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(idf))
    t = z / math.sqrt(chi2 / df)
    # Normalize to unit variance
    return t / math.sqrt(df / (df - 2))


def poisson_sample(lam: float, rng: random.Random) -> int:
    """Simple Poisson sample via inversion for small lambda."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Price path generators
# ---------------------------------------------------------------------------

def generate_ash_path(params: dict, n_ticks: int, rng: random.Random) -> list:
    """
    OU latent price + bid-ask bounce model for ASH (Roll 1984).

    mid[t] = latent[t] + bounce_c * q[t]   where q[t] ∈ {-1, +1} i.i.d.
    latent[t+1] = latent[t] + theta*(mu - latent[t]) + sigma_eff * t_noise

    Calibrated so:
      autocorr(returns) ≈ -bounce_c² / (sigma_eff² + 2*bounce_c²) ≈ -0.495
      std(returns) ≈ sqrt(sigma_eff² + 2*bounce_c²) ≈ 3.73
    """
    mu        = params["mu"]
    theta     = params["theta"]
    sigma_eff = params["sigma_eff"]
    bounce_c  = params["bounce_c"]
    t_df      = params["t_df"]

    latent = mu + rng.gauss(0, sigma_eff * 3)
    path = []
    for _ in range(n_ticks):
        latent = latent + theta * (mu - latent) + sigma_eff * sample_t(t_df, rng)
        latent = max(mu - 50, min(mu + 50, latent))
        q = 1 if rng.random() < 0.5 else -1
        mid = latent + bounce_c * q
        path.append(round(mid * 2) / 2)   # half-tick resolution
    return path


def generate_pepper_path(params: dict, n_ticks: int, day_num: int,
                          prev_z: float, rng: random.Random):
    """
    Trend + OU residual path for PEPPER.
    day_num: which synthetic day (0-indexed) — used to continue the trend.
    prev_z:  OU residual carried over from the previous day.
    Returns (path, final_z) where path is length n_ticks.
    """
    slope     = params["slope"]
    intercept = params["intercept"]
    ou_theta  = params["ou_theta"]
    ou_sigma  = params["ou_sigma"]

    # Global tick offset: each day is 1_000_000 global ticks apart (matching real data spacing)
    # Synthetic days start after the 3 real days (day offset = 3 * 1_000_000)
    day_global_offset = (3 + day_num) * 1_000_000

    t_df = params["t_df"]

    Z = prev_z
    path = []
    for i in range(n_ticks):
        ts_global = day_global_offset + i * TS_STEP
        trend = intercept + slope * ts_global
        Z = Z + ou_theta * (0 - Z) + ou_sigma * sample_t(t_df, rng)
        mid = trend + Z
        path.append(round(mid * 2) / 2)

    return path, Z


# ---------------------------------------------------------------------------
# Order book construction
# ---------------------------------------------------------------------------

def build_order_book(mid: float, params: dict, rng: random.Random):
    """
    Returns (bid_prices, bid_vols, ask_prices, ask_vols) — each up to 3 levels.
    """
    spread_dist = params["spread_dist"]
    bvol_dist   = params["bid_vol_dist"]
    avol_dist   = params["ask_vol_dist"]

    spread = sample_empirical(spread_dist, rng)
    spread = max(1, spread)   # minimum 1 tick spread

    # Level 1: symmetric around mid
    bid1 = int(mid - spread / 2)
    ask1 = bid1 + spread

    bv1 = sample_empirical(bvol_dist, rng)
    av1 = sample_empirical(avol_dist, rng)

    bid_prices = [bid1]
    bid_vols   = [bv1]
    ask_prices = [ask1]
    ask_vols   = [av1]

    # Level 2: ~70% chance
    if rng.random() < 0.70:
        bid_prices.append(bid1 - rng.randint(1, 3))
        bid_vols.append(sample_empirical(bvol_dist, rng))
        ask_prices.append(ask1 + rng.randint(1, 3))
        ask_vols.append(sample_empirical(avol_dist, rng))

        # Level 3: ~40% chance (only if level 2 exists)
        if rng.random() < 0.40:
            bid_prices.append(bid_prices[-1] - rng.randint(1, 3))
            bid_vols.append(sample_empirical(bvol_dist, rng))
            ask_prices.append(ask_prices[-1] + rng.randint(1, 3))
            ask_vols.append(sample_empirical(avol_dist, rng))

    return bid_prices, bid_vols, ask_prices, ask_vols


# ---------------------------------------------------------------------------
# CSV row formatting
# ---------------------------------------------------------------------------

def format_price_row(day: int, ts: int, product: str,
                     bid_prices, bid_vols, ask_prices, ask_vols,
                     mid: float) -> str:
    """Format one row of the prices CSV (semicolon-delimited, 17 cols)."""
    cols = [str(day), str(ts), product]

    # Bid levels 1-3 (price, vol interleaved)
    for lvl in range(3):
        if lvl < len(bid_prices):
            cols.append(str(bid_prices[lvl]))
            cols.append(str(bid_vols[lvl]))
        else:
            cols.append("")
            cols.append("")

    # Ask levels 1-3
    for lvl in range(3):
        if lvl < len(ask_prices):
            cols.append(str(ask_prices[lvl]))
            cols.append(str(ask_vols[lvl]))
        else:
            cols.append("")
            cols.append("")

    cols.append(f"{mid:.1f}")   # mid_price
    cols.append("0.0")          # profit_and_loss

    return ";".join(cols)


def format_trade_row(ts: int, product: str, price: float, qty: int) -> str:
    """Format one row of the trades CSV."""
    return f"{ts};;;{product};XIRECS;{price:.1f};{qty}"


# ---------------------------------------------------------------------------
# Day generation
# ---------------------------------------------------------------------------

def generate_day(day_num: int, params: dict, rng: random.Random,
                 pepper_prev_z: float = 0.0):
    """
    Generates all tick data for one synthetic day.
    Returns (price_rows, trade_rows, pepper_final_z).
    """
    ash_params    = params["ASH_COATED_OSMIUM"]
    pepper_params = params["INTARIAN_PEPPER_ROOT"]

    # Generate mid price paths
    ash_path = generate_ash_path(ash_params, TICKS_PER_DAY, rng)
    pepper_path, pepper_final_z = generate_pepper_path(
        pepper_params, TICKS_PER_DAY, day_num, pepper_prev_z, rng
    )

    price_rows = []
    trade_rows = []

    ash_trade_rate    = ash_params["trade_rate_per_100ts"]
    pepper_trade_rate = pepper_params["trade_rate_per_100ts"]

    for i in range(TICKS_PER_DAY):
        ts = i * TS_STEP

        for product, path, p_params, trade_rate in [
            ("ASH_COATED_OSMIUM",     ash_path,    ash_params,    ash_trade_rate),
            ("INTARIAN_PEPPER_ROOT",  pepper_path, pepper_params, pepper_trade_rate),
        ]:
            mid = path[i]
            bp, bv, ap, av = build_order_book(mid, p_params, rng)
            price_rows.append(format_price_row(day_num, ts, product, bp, bv, ap, av, mid))

            # Generate trades for this 100-ts window
            n_trades = poisson_sample(trade_rate, rng)
            for _ in range(n_trades):
                trade_ts  = ts + rng.randint(0, TS_STEP - 1)
                trade_qty = sample_empirical(p_params["trade_size_dist"], rng)
                # Trade price: bid or ask randomly
                trade_price = rng.choice([bp[0], ap[0]]) if bp and ap else mid
                trade_rows.append(format_trade_row(trade_ts, product, trade_price, trade_qty))

    # Sort trades by timestamp
    trade_rows.sort(key=lambda r: int(r.split(";")[0]))

    return price_rows, trade_rows, pepper_final_z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PRICE_HEADER = (
    "day;timestamp;product;"
    "bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
    "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;"
    "mid_price;profit_and_loss"
)
TRADE_HEADER = "timestamp;buyer;seller;symbol;currency;price;quantity"


def generate(n_days: int, seed: int = 42, params_path: str = PARAMS_FILE):
    rng = random.Random(seed)

    with open(params_path) as f:
        params = json.load(f)

    os.makedirs(MC_DATA, exist_ok=True)

    pepper_z = 0.0   # carry OU residual across days

    for d in range(n_days):
        price_rows, trade_rows, pepper_z = generate_day(d, params, rng, pepper_z)

        prices_path = os.path.join(MC_DATA, f"prices_round_99_day_{d}.csv")
        trades_path = os.path.join(MC_DATA, f"trades_round_99_day_{d}.csv")

        with open(prices_path, "w") as f:
            f.write(PRICE_HEADER + "\n")
            f.write("\n".join(price_rows))

        with open(trades_path, "w") as f:
            f.write(TRADE_HEADER + "\n")
            f.write("\n".join(trade_rows))

    print(f"Generated {n_days} synthetic days → {MC_DATA}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days",  type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--params",  type=str, default=PARAMS_FILE)
    args = parser.parse_args()

    generate(args.n_days, args.seed, args.params)
