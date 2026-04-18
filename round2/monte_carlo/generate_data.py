"""
generate_data.py  (Round 2)
---------------------------
Generates N synthetic trading days for both products using fitted parameters.
Writes CSVs to mc_data/round99/ in the exact backtester format.

Bot archetypes injected into every tick
---------------------------------------
1. Desperate Liquidator
   - Crosses the spread unconditionally (trades at visible bid/ask levels)
   - buy_prob = clip(0.5 + imbalance * 0.3, 0.15, 0.85)
   - Level distribution: 70% L1, 30% L2/L3  (from Plot 04 / market_mc spec)

2. Mean-Reverter (Oscillator)
   - Sells when price is above fair value, buys when below
   - Strength proportional to deviation / (4 * sigma_eff)
   - Always trades at L1 (passive pressure, not aggressive crossing)

For PEPPER: "fair value" at a given tick = trend(t_global) + 0 residual,
i.e., the reverter trades against the OU residual.

Each day also writes a <day>_meta.json sidecar with regime stats used by
run_mc.py for adversarial path detection.

Usage:
    python generate_data.py --n-days 1000 [--seed 42]
"""

import argparse
import json
import math
import os
import random

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MC_DATA     = os.path.join(SCRIPT_DIR, "mc_data", "round99")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "params.json")

TICKS_PER_DAY = 10_000
TS_STEP       = 100
PRODUCTS      = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]

PRICE_HEADER = (
    "day;timestamp;product;"
    "bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;"
    "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;"
    "mid_price;profit_and_loss"
)
TRADE_HEADER = "timestamp;buyer;seller;symbol;currency;price;quantity"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_empirical(dist, rng):
    return rng.choice(dist)


def sample_t(df, rng):
    idf = int(df)
    z   = rng.gauss(0, 1)
    chi2 = sum(rng.gauss(0, 1)**2 for _ in range(idf))
    t = z / math.sqrt(chi2 / df)
    return t / math.sqrt(df / (df - 2))


def poisson_sample(lam, rng):
    if lam <= 0:
        return 0
    L = math.exp(-min(lam, 20))
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Price path generators
# ---------------------------------------------------------------------------

def generate_ash_path(params, n_ticks, rng):
    mu        = params["mu"]
    theta     = params["theta"]
    sigma_eff = params["sigma_eff"]
    bounce_c  = params["bounce_c"]
    t_df      = params["t_df"]

    latent = mu + rng.gauss(0, sigma_eff * 3)
    path   = []
    for _ in range(n_ticks):
        latent = latent + theta * (mu - latent) + sigma_eff * sample_t(t_df, rng)
        latent = max(mu - 60, min(mu + 60, latent))
        q   = 1 if rng.random() < 0.5 else -1
        mid = latent + bounce_c * q
        path.append(round(mid * 2) / 2)
    return path


def generate_pepper_path(params, n_ticks, day_num, prev_z, rng):
    slope    = params["slope"]
    intercept = params["intercept"]
    ou_theta = params["ou_theta"]
    ou_sigma = params["ou_sigma"]
    t_df     = params["t_df"]

    # Synthetic days placed after the 3 real days (global offsets 0, 1M, 2M are real)
    day_global_offset = (3 + day_num) * 1_000_000

    Z    = prev_z
    path = []
    trend_vals = []
    for i in range(n_ticks):
        ts_global = day_global_offset + i * TS_STEP
        trend     = intercept + slope * ts_global
        Z         = Z + ou_theta * (0 - Z) + ou_sigma * sample_t(t_df, rng)
        mid       = trend + Z
        path.append(round(mid * 2) / 2)
        trend_vals.append(trend)
    return path, Z, trend_vals


# ---------------------------------------------------------------------------
# Order book construction
# ---------------------------------------------------------------------------

def build_order_book(mid, params, rng):
    spread_dist = params["spread_dist"]
    bvol_dist   = params["bid_vol_dist"]
    avol_dist   = params["ask_vol_dist"]

    spread = max(1, sample_empirical(spread_dist, rng))
    half   = spread / 2
    bid1   = round(mid - half)
    ask1   = bid1 + spread

    bv1 = sample_empirical(bvol_dist, rng)
    av1 = sample_empirical(avol_dist, rng)

    bid_prices = [bid1];  bid_vols = [bv1]
    ask_prices = [ask1];  ask_vols = [av1]

    if rng.random() < 0.70:
        bid_prices.append(bid1 - rng.randint(1, 3))
        bid_vols.append(sample_empirical(bvol_dist, rng))
        ask_prices.append(ask1 + rng.randint(1, 3))
        ask_vols.append(sample_empirical(avol_dist, rng))

        if rng.random() < 0.40:
            bid_prices.append(bid_prices[-1] - rng.randint(1, 3))
            bid_vols.append(sample_empirical(bvol_dist, rng))
            ask_prices.append(ask_prices[-1] + rng.randint(1, 3))
            ask_vols.append(sample_empirical(avol_dist, rng))

    return bid_prices, bid_vols, ask_prices, ask_vols


# ---------------------------------------------------------------------------
# Bot trade injection
# ---------------------------------------------------------------------------

def inject_bot_trades(ts, mid, fair_value, deviation,
                      bid_prices, ask_prices,
                      imbalance, params, rng):
    """
    Inject trades from two bot archetypes.

    Parameters
    ----------
    ts          : int    current timestamp
    mid         : float  current mid price
    fair_value  : float  fair value at this tick (ASH: 10000, PEPPER: trend(t))
    deviation   : float  mid - fair_value  (positive → above fair)
    bid_prices  : list   visible bid levels
    ask_prices  : list   visible ask levels
    imbalance   : float  (bid_vol - ask_vol) / total ∈ [-1, 1]
    params      : dict   product params from params.json
    rng         : Random

    Returns
    -------
    list of (timestamp, price, qty)
    meta dict with liq_n and rev_n counts for this tick
    """
    trades = []
    liq_n  = 0
    rev_n  = 0

    sigma_ref = params.get("sigma_eff", params.get("ou_sigma", 5.0))

    # ── Bot 1: Desperate Liquidator ───────────────────────────────────────────
    # Price model: Mid ± (Spread/2 + ε)
    #   ε > 0 → trades past the best quote (crosses our passive limit order)
    #   ε < 0 → trades inside the half-spread (misses our quote)
    # ε ~ Normal(epsilon_mean, epsilon_std) from observed trade-vs-mid distribution.
    liq_rate    = params.get("liquidator_rate_per_100ts", 0.05)
    eps_mean    = params.get("epsilon_mean", 0.0)
    eps_std     = max(0.01, params.get("epsilon_std", 1.0))
    spread      = ask_prices[0] - bid_prices[0]
    half_spread = spread / 2.0
    buy_prob    = min(0.85, max(0.15, 0.5 + imbalance * 0.3))

    n_liq = poisson_sample(liq_rate, rng)
    for _ in range(n_liq):
        is_buy = rng.random() < buy_prob
        eps    = rng.gauss(eps_mean, eps_std)
        # Price = Mid ± (Spread/2 + ε), rounded to nearest 0.5
        raw    = mid + (half_spread + eps) if is_buy else mid - (half_spread + eps)
        price  = round(raw * 2) / 2
        qty    = sample_empirical(params["trade_size_dist"], rng)
        trades.append((ts + rng.randint(0, TS_STEP - 1), price, qty))
        liq_n += 1

    # ── Bot 2: Mean-Reverter (Oscillator) ─────────────────────────────────────
    # Sells when price is above fair value, buys when below.
    # Probability of selling proportional to normalized deviation.
    rev_rate = params.get("reverter_rate_per_100ts", 0.03)
    sell_bias = deviation / max(1.0, 4.0 * sigma_ref)
    sell_prob = min(0.92, max(0.08, 0.5 + sell_bias))

    n_rev = poisson_sample(rev_rate, rng)
    for _ in range(n_rev):
        is_sell = rng.random() < sell_prob
        # Reverter hits L1 only (not aggressive enough for deeper levels)
        price = bid_prices[0] if is_sell else ask_prices[0]
        qty   = sample_empirical(params["trade_size_dist"], rng)
        trades.append((ts + rng.randint(0, TS_STEP - 1), price, qty))
        rev_n += 1

    return trades, {"liq_n": liq_n, "rev_n": rev_n}


# ---------------------------------------------------------------------------
# CSV row formatting
# ---------------------------------------------------------------------------

def format_price_row(day, ts, product, bid_prices, bid_vols, ask_prices, ask_vols, mid):
    cols = [str(day), str(ts), product]
    for lvl in range(3):
        if lvl < len(bid_prices):
            cols += [str(bid_prices[lvl]), str(bid_vols[lvl])]
        else:
            cols += ["", ""]
    for lvl in range(3):
        if lvl < len(ask_prices):
            cols += [str(ask_prices[lvl]), str(ask_vols[lvl])]
        else:
            cols += ["", ""]
    cols += [f"{mid:.1f}", "0.0"]
    return ";".join(cols)


def format_trade_row(ts, product, price, qty):
    return f"{ts};;;{product};XIRECS;{price:.1f};{qty}"


# ---------------------------------------------------------------------------
# Day generation
# ---------------------------------------------------------------------------

def generate_day(day_num, params, rng, pepper_prev_z=0.0):
    ash_p   = params["ASH_COATED_OSMIUM"]
    pep_p   = params["INTARIAN_PEPPER_ROOT"]

    ash_path                    = generate_ash_path(ash_p, TICKS_PER_DAY, rng)
    pepper_path, pep_final_z, pepper_trends = generate_pepper_path(
        pep_p, TICKS_PER_DAY, day_num, pepper_prev_z, rng
    )

    price_rows  = []
    trade_rows  = []
    day_meta    = {
        "ASH_COATED_OSMIUM":    {"liq_total": 0, "rev_total": 0, "imbalances": [], "deviations": []},
        "INTARIAN_PEPPER_ROOT": {"liq_total": 0, "rev_total": 0, "imbalances": [], "deviations": []},
    }

    for i in range(TICKS_PER_DAY):
        ts = i * TS_STEP

        for product, path, p_params, fair_fn in [
            ("ASH_COATED_OSMIUM",    ash_path,    ash_p,
             lambda _i: ash_p["mu"]),
            ("INTARIAN_PEPPER_ROOT", pepper_path, pep_p,
             lambda _i: pepper_trends[_i]),
        ]:
            mid        = path[i]
            fair       = fair_fn(i)
            deviation  = mid - fair

            bp, bv, ap, av = build_order_book(mid, p_params, rng)

            # Compute L1 imbalance for this tick
            imbalance = 0.0
            if bv and av:
                total = bv[0] + av[0]
                if total > 0:
                    imbalance = (bv[0] - av[0]) / total

            price_rows.append(format_price_row(day_num, ts, product, bp, bv, ap, av, mid))

            bot_trades, tick_meta = inject_bot_trades(
                ts, mid, fair, deviation, bp, ap, imbalance, p_params, rng
            )
            for ts_t, price_t, qty_t in bot_trades:
                trade_rows.append(format_trade_row(ts_t, product, price_t, qty_t))

            m = day_meta[product]
            m["liq_total"] += tick_meta["liq_n"]
            m["rev_total"] += tick_meta["rev_n"]
            m["imbalances"].append(imbalance)
            m["deviations"].append(deviation)

    trade_rows.sort(key=lambda r: int(r.split(";")[0]))

    # Summarise metadata (don't store full per-tick arrays)
    meta = {}
    for product in PRODUCTS:
        m   = day_meta[product]
        imb = m["imbalances"]
        dev = m["deviations"]
        meta[product] = {
            "liq_total":       m["liq_total"],
            "rev_total":       m["rev_total"],
            "imb_mean":        sum(imb) / len(imb) if imb else 0.0,
            "imb_std":         (sum((x - sum(imb)/len(imb))**2 for x in imb) / len(imb))**0.5
                               if len(imb) > 1 else 0.0,
            "dev_mean":        sum(dev) / len(dev) if dev else 0.0,
            "dev_std":         (sum((x - sum(dev)/len(dev))**2 for x in dev) / len(dev))**0.5
                               if len(dev) > 1 else 0.0,
            "dev_abs_max":     max(abs(x) for x in dev) if dev else 0.0,
        }

    return price_rows, trade_rows, pep_final_z, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(n_days=1000, seed=42, params_path=PARAMS_FILE):
    rng = random.Random(seed)

    with open(params_path) as f:
        params = json.load(f)

    os.makedirs(MC_DATA, exist_ok=True)

    pepper_z = 0.0

    for d in range(n_days):
        price_rows, trade_rows, pepper_z, meta = generate_day(d, params, rng, pepper_z)

        prices_path = os.path.join(MC_DATA, f"prices_round_99_day_{d}.csv")
        trades_path = os.path.join(MC_DATA, f"trades_round_99_day_{d}.csv")
        meta_path   = os.path.join(MC_DATA, f"{d}_meta.json")

        with open(prices_path, "w") as f:
            f.write(PRICE_HEADER + "\n")
            f.write("\n".join(price_rows))

        with open(trades_path, "w") as f:
            f.write(TRADE_HEADER + "\n")
            f.write("\n".join(trade_rows))

        with open(meta_path, "w") as f:
            json.dump({"day": d, "products": meta}, f)

        if (d + 1) % 100 == 0 or d == 0:
            print(f"  Generated day {d+1}/{n_days}")

    print(f"Done — {n_days} synthetic days → {MC_DATA}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days",  type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--params",  type=str, default=PARAMS_FILE)
    args = parser.parse_args()
    generate(args.n_days, args.seed, args.params)
