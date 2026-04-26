"""
grid_search.py — Round 3 parameter optimisation for v6 strategy

Replays price CSV data tick-by-tick.  Fill model:
  • Taker orders  : fill immediately at market price (deterministic)
  • Passive quotes: fill if next tick's best bid/ask crosses our price
    (conservative — misses same-tick aggressor fills, overestimates adverse
     selection, but good enough for relative comparison)

Parameters swept:
  OPT_TAKER_EDGE   — min BS-fair vs market gap to take (seashells)
  OPT_POS_CAP      — max long/short per ATM option strike
  HEDGE_DEADBAND   — min |Δδ| before we submit a VF hedge trade
  HG_EMA_ALPHA     — speed of HG fair-value EMA
  HG_SOFT_LIMIT    — position at which HG forces inventory reduction

Run:
  python3 grid_search.py
"""

import csv
import itertools
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "../data/ROUND_3")

PRICE_FILES = {
    0: os.path.join(DATA_DIR, "prices_round_3_day_0.csv"),
    1: os.path.join(DATA_DIR, "prices_round_3_day_1.csv"),
    2: os.path.join(DATA_DIR, "prices_round_3_day_2.csv"),
}

# ─── Products ─────────────────────────────────────────────────────────────────
HG  = "HYDROGEL_PACK"
VF  = "VELVETFRUIT_EXTRACT"
VEV = {
    5000: "VOLCANIC_ROCK_VOUCHER_5000",   # may differ; handle by substring match
    5100: "VOLCANIC_ROCK_VOUCHER_5100",
    5200: "VOLCANIC_ROCK_VOUCHER_5200",
    5300: "VOLCANIC_ROCK_VOUCHER_5300",
    5400: "VOLCANIC_ROCK_VOUCHER_5400",
    5500: "VOLCANIC_ROCK_VOUCHER_5500",
}
ATM_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]
FLOOR_STRIKES = [6000, 6500]

HG_LIMIT  = 200
VF_LIMIT  = 200
OPT_LIMIT = 300

TTE_BY_DAY = {0: 5.0, 1: 4.0, 2: 3.0}

# ─── Black-Scholes (same as v6) ───────────────────────────────────────────────

def _ncdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * (
        t * (0.319381530 + t * (-0.356563782
        + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    return p if x >= 0.0 else 1.0 - p

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _bs_call(S, K, T_days, sigma):
    if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0:
        return max(S - K, 0.0)
    T = T_days / 252.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    return S * _ncdf(d1) - K * _ncdf(d2)

def _bs_delta(S, K, T_days, sigma):
    if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0:
        return 1.0 if S > K else 0.0
    T = T_days / 252.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return _ncdf(d1)

def _bs_vega(S, K, T_days, sigma):
    if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0:
        return 0.0
    T = T_days / 252.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    return S * math.sqrt(T) * _npdf(d1)

def _bs_iv(C, S, K, T_days):
    intrinsic = max(S - K, 0.0)
    if C < intrinsic - 0.5 or T_days <= 1e-6 or S <= 0:
        return float("nan")
    C = max(C, intrinsic + 1e-6)
    T = T_days / 252.0
    sigma = math.sqrt(2.0 * math.pi / T) * C / S
    sigma = max(0.05, min(sigma, 5.0))
    for _ in range(50):
        price = _bs_call(S, K, T_days, sigma)
        vega  = _bs_vega(S, K, T_days, sigma)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - C) / vega
        sigma  = max(1e-6, min(sigma, 10.0))
    return sigma

def _fit_smile(pairs):
    if len(pairs) < 3:
        return None
    sx4=sx3=sx2=sx1=sn=syx2=syx1=sy=0.0
    for x, y in pairs:
        x2=x*x
        sx4+=x2*x2; sx3+=x2*x; sx2+=x2; sx1+=x; sn+=1
        syx2+=y*x2;  syx1+=y*x; sy+=y
    M=[[sx4,sx3,sx2],[sx3,sx2,sx1],[sx2,sx1,sn]]
    r=[syx2,syx1,sy]
    def det3(m):
        return(m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
              -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
              +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    def sub(m,v,c):
        o=[row[:] for row in m]
        for i in range(3): o[i][c]=v[i]
        return o
    D=det3(M)
    if abs(D)<1e-12: return None
    return det3(sub(M,r,0))/D, det3(sub(M,r,1))/D, det3(sub(M,r,2))/D

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_price_data():
    """Returns list of tick dicts sorted by (day, timestamp)."""
    ticks = []
    for day, path in PRICE_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                row["_day"] = day
                ticks.append(row)
    return ticks

def build_book_index(ticks):
    """Group ticks by (day, timestamp, product) for O(1) lookup."""
    idx = defaultdict(dict)   # (day, ts) -> {product: row}
    for row in ticks:
        key = (row["_day"], int(row["timestamp"]))
        idx[key][row["product"]] = row
    return idx

def get_book(row) -> Optional[Tuple[float, float, float, float]]:
    """Returns (best_bid, bid_vol, best_ask, ask_vol) or None."""
    try:
        bb  = float(row["bid_price_1"])
        bbv = float(row["bid_volume_1"])
        ba  = float(row["ask_price_1"])
        bav = float(row["ask_volume_1"])
        if bb <= 0 or ba <= 0 or bbv <= 0 or bav <= 0 or bb >= ba:
            return None
        return bb, bbv, ba, bav
    except (ValueError, KeyError):
        return None

def get_mid(row) -> Optional[float]:
    try:
        return float(row["mid_price"])
    except (ValueError, KeyError):
        return None

# ─── Determine product name mapping from actual CSV ───────────────────────────

def detect_product_names(ticks):
    """Auto-detect correct product name for each VEV strike."""
    products = set(r["product"] for r in ticks)
    vev_map = {}
    for K in [5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
        for p in products:
            if str(K) in p and ("VEV" in p or "VOUCHER" in p or "VOLCANIC" in p):
                vev_map[K] = p
                break
    return vev_map

# ─── Single backtest run ───────────────────────────────────────────────────────

def run_backtest(ticks, book_idx, vev_map, params):
    """
    Simulates the v6 strategy with given params.
    Returns dict of per-product P&L and total.
    """
    OPT_TAKER_EDGE  = params["OPT_TAKER_EDGE"]
    OPT_POS_CAP     = params["OPT_POS_CAP"]
    HEDGE_DEADBAND  = params["HEDGE_DEADBAND"]
    HG_EMA_ALPHA    = params["HG_EMA_ALPHA"]
    HG_SOFT_LIMIT   = params["HG_SOFT_LIMIT"]
    HG_QUOTE_TICK   = params.get("HG_QUOTE_TICK", 2)
    HG_QUOTE_SIZE   = params.get("HG_QUOTE_SIZE", 20)
    OPT_TAKER_SIZE  = params.get("OPT_TAKER_SIZE", 10)
    OPT_PASSIVE_TICK = params.get("OPT_PASSIVE_TICK", 1)
    OPT_PASSIVE_SIZE = params.get("OPT_PASSIVE_SIZE", 10)
    OPT_MIN_TV      = params.get("OPT_MIN_TV", 0.3)
    HG_HARD_LIMIT   = 190
    HG_SKEW_TICKS   = 3.0

    # State
    pos: Dict[str, int] = defaultdict(int)  # symbol -> position
    cash = 0.0
    hg_ema = None

    # Passive orders outstanding from previous tick (simplified: 1-tick pending)
    # Format: (symbol, price, qty_signed)
    pending_passives: List[Tuple[str, float, int]] = []

    def buy_room(sym, limit=None):
        lim = limit or OPT_POS_CAP
        return max(0, lim - pos[sym])

    def sell_room(sym, limit=None):
        lim = limit or OPT_POS_CAP
        return max(0, lim + pos[sym])

    def fill(sym, price, qty):
        """qty > 0 = buy, qty < 0 = sell."""
        nonlocal cash
        pos[sym] += qty
        cash -= price * qty

    # Collect unique (day, ts) keys in order
    keys = sorted(book_idx.keys())

    for i, key in enumerate(keys):
        day, ts = key
        books = book_idx[key]
        next_books = book_idx.get(keys[i + 1]) if i + 1 < len(keys) else None

        tte_base = TTE_BY_DAY.get(day, 3.0)
        tte = max(0.01, tte_base - ts / 1_000_000)

        # ── Fill pending passive orders from last tick ────────────────────────
        if next_books:
            for sym, p_price, p_qty in pending_passives:
                nb = next_books.get(sym)
                if nb is None:
                    continue
                nbk = get_book(nb)
                if nbk is None:
                    continue
                nbb, _, nba, _ = nbk
                if p_qty > 0 and nbb >= p_price:    # passive bid crossed by market
                    fill(sym, p_price, p_qty)
                elif p_qty < 0 and nba <= p_price:  # passive ask crossed by market
                    fill(sym, p_price, p_qty)
        pending_passives = []

        # ── HG: EMA fair value + taker + passive quotes ───────────────────────
        hg_row = books.get(HG)
        if hg_row:
            hg_bk = get_book(hg_row)
            if hg_bk:
                hg_bb, hg_bbv, hg_ba, hg_bav = hg_bk
                wmid = (hg_bb * hg_bav + hg_ba * hg_bbv) / (hg_bbv + hg_bav)
                hg_ema = wmid if hg_ema is None else (
                    (1 - HG_EMA_ALPHA) * hg_ema + HG_EMA_ALPHA * wmid)
                fair = hg_ema

                hg_pos = pos[HG]
                skew   = (hg_pos / HG_LIMIT) * HG_SKEW_TICKS
                fair_q = fair - skew
                bid_p  = math.floor(fair_q - HG_QUOTE_TICK)
                ask_p  = math.ceil(fair_q + HG_QUOTE_TICK)

                # Taker: cross spread when clear edge
                if hg_ba <= fair - 5 and buy_room(HG, HG_LIMIT) > 0:
                    qty = min(10, int(hg_bav), buy_room(HG, HG_LIMIT))
                    if qty > 0:
                        fill(HG, hg_ba, qty)
                elif hg_bb >= fair + 5 and sell_room(HG, HG_LIMIT) > 0:
                    qty = min(10, int(hg_bbv), sell_room(HG, HG_LIMIT))
                    if qty > 0:
                        fill(HG, hg_bb, -qty)

                # Hard limit emergency
                hg_pos = pos[HG]
                if hg_pos >= HG_HARD_LIMIT:
                    qty = min(30, int(hg_bbv), sell_room(HG, HG_LIMIT))
                    if qty > 0:
                        fill(HG, hg_bb, -qty)
                elif hg_pos <= -HG_HARD_LIMIT:
                    qty = min(30, int(hg_bav), buy_room(HG, HG_LIMIT))
                    if qty > 0:
                        fill(HG, hg_ba, qty)
                else:
                    hg_pos = pos[HG]
                    bq = 0 if hg_pos >= HG_SOFT_LIMIT else min(HG_QUOTE_SIZE,
                                                                 buy_room(HG, HG_LIMIT))
                    aq = 0 if hg_pos <= -HG_SOFT_LIMIT else min(HG_QUOTE_SIZE,
                                                                  sell_room(HG, HG_LIMIT))
                    # Forced unwind
                    if hg_pos >= HG_SOFT_LIMIT:
                        aq = min(HG_QUOTE_SIZE, sell_room(HG, HG_LIMIT))
                        ask_p = min(ask_p, hg_bb + 1)
                    if hg_pos <= -HG_SOFT_LIMIT:
                        bq = min(HG_QUOTE_SIZE, buy_room(HG, HG_LIMIT))
                        bid_p = max(bid_p, hg_ba - 1)
                    if bq > 0:
                        pending_passives.append((HG, bid_p, bq))
                    if aq > 0:
                        pending_passives.append((HG, ask_p, -aq))

        # ── Get VF book (needed for smile fit and hedging) ────────────────────
        vf_row = books.get(VF)
        vf_bk  = get_book(vf_row) if vf_row else None
        if vf_bk is None:
            continue
        vf_bb, vf_bbv, vf_ba, vf_bav = vf_bk
        S = (vf_bb + vf_ba) / 2.0

        if tte < 0.3:
            continue

        # ── Smile fit ─────────────────────────────────────────────────────────
        smile_pairs = []
        opt_bks  = {}
        opt_ivs  = {}

        for K in ATM_STRIKES:
            sym = vev_map.get(K)
            if sym is None:
                continue
            row = books.get(sym)
            if row is None:
                continue
            bk = get_book(row)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            mid = (bb + ba) / 2.0
            tv  = mid - max(S - K, 0.0)
            if tv < OPT_MIN_TV:
                continue
            iv = _bs_iv(mid, S, float(K), tte)
            if math.isnan(iv) or iv < 0.05 or iv > 3.0:
                continue
            m = math.log(K / S)
            smile_pairs.append((m, iv))
            opt_bks[K]  = (bb, bbv, ba, bav)
            opt_ivs[K]  = iv

        coeffs = _fit_smile(smile_pairs) if len(smile_pairs) >= 3 else None

        # ── ATM option aggressive MM ──────────────────────────────────────────
        fitted_ivs = {}

        for K in ATM_STRIKES:
            if K not in opt_bks:
                continue
            sym = vev_map.get(K)
            if sym is None:
                continue
            bb, bbv, ba, bav = opt_bks[K]
            opt_pos = pos[sym]
            m = math.log(K / S)

            if coeffs is not None:
                a, b, c = coeffs
                fiv = max(0.05, a * m * m + b * m + c)
            else:
                fiv = opt_ivs.get(K, float("nan"))
                if math.isnan(fiv):
                    continue

            fitted_ivs[K] = fiv
            fair = _bs_call(S, float(K), tte, fiv)

            # Taker buy
            if ba < fair - OPT_TAKER_EDGE:
                room = buy_room(sym)
                qty  = min(OPT_TAKER_SIZE, int(bav), room)
                if qty > 0:
                    fill(sym, ba, qty)

            # Taker sell
            elif bb > fair + OPT_TAKER_EDGE:
                room = sell_room(sym)
                qty  = min(OPT_TAKER_SIZE, int(bbv), room)
                if qty > 0:
                    fill(sym, bb, -qty)

            # Passive quotes
            bid_p = round(fair) - OPT_PASSIVE_TICK
            ask_p = round(fair) + OPT_PASSIVE_TICK
            if bid_p > 0:
                bq = min(OPT_PASSIVE_SIZE, buy_room(sym))
                if bq > 0:
                    pending_passives.append((sym, float(bid_p), bq))
            if ask_p > 0:
                aq = min(OPT_PASSIVE_SIZE, sell_room(sym))
                if aq > 0:
                    pending_passives.append((sym, float(ask_p), -aq))

        # ── Delta hedge VF ────────────────────────────────────────────────────
        total_delta = 0.0
        for K in ATM_STRIKES:
            opt_sym = vev_map.get(K)
            if opt_sym is None:
                continue
            opt_pos = pos[opt_sym]
            if opt_pos == 0:
                continue
            fiv = fitted_ivs.get(K)
            if fiv is None:
                continue
            total_delta += opt_pos * _bs_delta(S, float(K), tte, fiv)

        target_vf = -round(total_delta)
        target_vf = max(-VF_LIMIT, min(VF_LIMIT, target_vf))
        current_vf = pos[VF]
        hedge = target_vf - current_vf

        if abs(hedge) >= HEDGE_DEADBAND and hedge != 0:
            if hedge > 0:
                qty = min(int(hedge), int(vf_bav), buy_room(VF, VF_LIMIT))
                if qty > 0:
                    fill(VF, vf_ba, qty)
            else:
                qty = min(int(-hedge), int(vf_bbv), sell_room(VF, VF_LIMIT))
                if qty > 0:
                    fill(VF, vf_bb, -qty)

    # ── Final mark-to-market ──────────────────────────────────────────────────
    final_books = book_idx.get(keys[-1], {}) if keys else {}
    mtm = 0.0
    for sym, qty in pos.items():
        if qty == 0:
            continue
        row = final_books.get(sym)
        if row:
            mid = get_mid(row)
            if mid:
                mtm += qty * mid

    pnl_by_product = {}
    # Return summary metrics
    return {
        "total_pnl": cash + mtm,
        "cash": cash,
        "mtm": mtm,
        "final_positions": dict(pos),
    }

# ─── Grid search ──────────────────────────────────────────────────────────────

PARAM_GRID = {
    "OPT_TAKER_EDGE":  [1.0, 1.5, 2.0, 3.0],
    "OPT_POS_CAP":     [20, 35, 50],
    "HEDGE_DEADBAND":  [0, 5, 10],
    "HG_EMA_ALPHA":    [0.005, 0.010, 0.020],
    "HG_SOFT_LIMIT":   [100, 140],
    # Fixed
    "HG_QUOTE_TICK":   [2],
    "HG_QUOTE_SIZE":   [20],
    "OPT_TAKER_SIZE":  [10],
    "OPT_PASSIVE_TICK":[1],
    "OPT_PASSIVE_SIZE":[10],
    "OPT_MIN_TV":      [0.3],
}

def main():
    print("Loading price data...")
    ticks = load_price_data()
    if not ticks:
        print("ERROR: no data loaded. Check DATA_DIR path.")
        return

    print(f"Loaded {len(ticks)} rows")
    vev_map = detect_product_names(ticks)
    print(f"VEV map: {vev_map}")

    book_idx = build_book_index(ticks)
    print(f"Unique (day, ts) keys: {len(book_idx)}")

    # Build combinations (exclude fixed params)
    sweep_keys   = ["OPT_TAKER_EDGE", "OPT_POS_CAP", "HEDGE_DEADBAND",
                    "HG_EMA_ALPHA", "HG_SOFT_LIMIT"]
    fixed_params = {k: v[0] for k, v in PARAM_GRID.items() if k not in sweep_keys}

    combos = list(itertools.product(*[PARAM_GRID[k] for k in sweep_keys]))
    print(f"\nRunning {len(combos)} parameter combinations...")
    print("-" * 90)

    results = []

    for combo in combos:
        params = dict(zip(sweep_keys, combo))
        params.update(fixed_params)
        out = run_backtest(ticks, book_idx, vev_map, params)
        results.append((out["total_pnl"], params, out))

    results.sort(key=lambda x: -x[0])

    print(f"\n{'Rank':>4}  {'Total PnL':>12}  {'Cash':>10}  {'MtM':>10}  "
          f"{'Edge':>5}  {'Cap':>4}  {'Hdge':>5}  {'EMA':>6}  {'Soft':>5}")
    print("-" * 90)
    for rank, (pnl, p, out) in enumerate(results[:30], 1):
        print(f"{rank:4d}  {pnl:+12.2f}  {out['cash']:+10.2f}  {out['mtm']:+10.2f}  "
              f"{p['OPT_TAKER_EDGE']:5.1f}  {p['OPT_POS_CAP']:4d}  "
              f"{p['HEDGE_DEADBAND']:5d}  {p['HG_EMA_ALPHA']:6.3f}  "
              f"{p['HG_SOFT_LIMIT']:5d}")

    print()
    best_pnl, best_p, best_out = results[0]
    print("=== BEST PARAMS ===")
    for k, v in best_p.items():
        if k in sweep_keys:
            print(f"  {k}: {v}")
    print(f"  → Total PnL: {best_pnl:+.2f}  (cash={best_out['cash']:+.2f}  mtm={best_out['mtm']:+.2f})")
    print(f"  → Final positions: {best_out['final_positions']}")

    # Also show sensitivity: fix all to best, vary one at a time
    print("\n=== PARAMETER SENSITIVITY (others at best) ===")
    for key in sweep_keys:
        vals = PARAM_GRID[key]
        if len(vals) == 1:
            continue
        print(f"\n  {key}:")
        for v in vals:
            p = dict(best_p)
            p[key] = v
            out = run_backtest(ticks, book_idx, vev_map, p)
            marker = " ← best" if v == best_p[key] else ""
            print(f"    {v:8}  →  {out['total_pnl']:+10.2f}{marker}")

if __name__ == "__main__":
    main()
