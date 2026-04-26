"""
v6_round3_prosperity.py
IMC Prosperity 4 — Round 3  |  Fresh strategy

Design (based on Frankfurt Hedgehog IMC Prosperity 3 approach):
─────────────────────────────────────────────────────────────────
1. FIT IV SMILE every tick
   • For each ATM strike (5000–5500): compute market IV via Newton-Raphson
   • Moneyness: m = log(K / S)  (negative = ITM, positive = OTM)
   • Fit quadratic: IV(m) = a·m² + b·m + c  via least-squares / Cramer's rule
   • This gives the "fair" implied vol for any strike

2. AGGRESSIVE OPTION MM (VEV_5000–5500)
   • Fair price = BS_call(S, K, TTE, fitted_IV)
   • If market_ask < fair - TAKER_EDGE  →  take: buy at market_ask
   • If market_bid > fair + TAKER_EDGE  →  take: sell at market_bid
   • Passive quotes: bid at round(fair) - 1, ask at round(fair) + 1
     (no maker-only constraint — willing to cross like the Hedgehog)
   • Position cap: 50 per strike → max delta ≤ 50 × 6 × 0.8 ≈ 240 < VF limit

3. DELTA HEDGE with VF Extract after every tick
   • total_delta = Σ pos[K] × BS_delta(S, K, TTE, fitted_IV[K])
   • target_vf = −round(total_delta)
   • Submit aggressive VF orders to reach target_vf each tick
   • This makes the book delta-neutral — only exposed to IV, not direction

4. VEV_6000 / VEV_6500  →  passive short at ask=1 (floor price)
   • P(expiry ITM) ≈ 0.3% and 0.0006%. Selling at 1 = free premium.
   • No delta hedge needed (delta ≈ 0)

5. HYDROGEL_PACK  →  simple EMA market maker
   • EMA fair value (α=0.005), inventory skew ±3 ticks, quote ±2 ticks
   • Forced unwind when pos ≥ soft limit

Position limits (official):
  HYDROGEL_PACK : 200   VELVETFRUIT_EXTRACT : 200   VEV_* : 300 each
TTE mapping (live):
  day 0 → TTE=5, day 1 → TTE=4, ..., day 4 → TTE=1
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]],
              conversions: int, trader_data: str) -> None:
        base = len(self.to_json([self.compress_state(state, ""),
                                  self.compress_orders(orders), conversions, "", ""]))
        max_item = max(0, (self.max_log_length - base) // 3)
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item),
            self.truncate(self.logs, max_item),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[str, List[Any]]:
        return {sym: [od.buy_orders, od.sell_orders] for sym, od in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for tl in trades.values() for t in tl]

    def compress_observations(self, observations: Observation) -> List[Any]:
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                   o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in observations.conversionObservations.items()}
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if max_length <= 0:
            return ""
        lo, hi, out = 0, min(len(value), max_length), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:

    # ── Products ──────────────────────────────────────────────────────────────
    HG  = "HYDROGEL_PACK"
    VF  = "VELVETFRUIT_EXTRACT"
    VEV = {
        4000: "VEV_4000", 4500: "VEV_4500",
        5000: "VEV_5000", 5100: "VEV_5100",
        5200: "VEV_5200", 5300: "VEV_5300",
        5400: "VEV_5400", 5500: "VEV_5500",
        6000: "VEV_6000", 6500: "VEV_6500",
    }
    ATM_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]   # smile fit + aggressive MM
    FLOOR_STRIKES = [6000, 6500]                            # passive short at ask=1

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT  = 200
    VF_LIMIT  = 200
    VEV_LIMIT = 300

    # Option MM: cap at 50 per strike so total delta fits inside VF_LIMIT=200
    # Worst case: 50 units × 6 strikes × delta≈0.65 ≈ 195  ✓
    OPT_POS_CAP = 50

    # Floor-price strikes: short up to 200 (delta≈0, no hedge needed)
    FLOOR_POS_CAP = 200

    # ── TTE mapping (live Round 3) ─────────────────────────────────────────────
    TTE_BY_DAY = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}

    # ── Option MM parameters ───────────────────────────────────────────────────
    OPT_TAKER_EDGE   = 1.5    # take when fair vs market exceeds this (seashells)
    OPT_PASSIVE_TICK = 1      # passive quote offset from fair price (ticks)
    OPT_TAKER_SIZE   = 10     # max units per taker fill
    OPT_PASSIVE_SIZE = 10     # max units per passive quote
    OPT_MIN_TTE      = 0.3    # stop trading very near expiry
    OPT_MIN_TV       = 0.3    # skip if time value below this (not enough edge)
    FLOOR_SHORT_SIZE = 20     # passive sell size per tick for floor-price strikes

    # ── Smile fit ─────────────────────────────────────────────────────────────
    SMILE_MIN_POINTS = 3      # need at least 3 IV points to fit parabola

    # ── HG MM parameters ──────────────────────────────────────────────────────
    HG_EMA_ALPHA    = 0.005
    HG_QUOTE_TICK   = 2       # quote ±2 from fair
    HG_SKEW_TICKS   = 3.0    # max inventory skew at position limit
    HG_QUOTE_SIZE   = 20
    HG_SOFT_LIMIT   = 140
    HG_HARD_LIMIT   = 190
    HG_UNWIND_SIZE  = 30

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes math  (no scipy / numpy)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ncdf(x: float) -> float:
        """Normal CDF — Abramowitz & Stegun 26.2.17, error < 7.5e-8."""
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        p = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * (
            t * (0.319381530 + t * (-0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        )
        return p if x >= 0.0 else 1.0 - p

    @staticmethod
    def _npdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def _d1(cls, S: float, K: float, T: float, sigma: float) -> float:
        return (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))

    @classmethod
    def _bs_call(cls, S: float, K: float, T_days: float, sigma: float) -> float:
        """European call price. T_days in Solvenarian days (1 day = 1/252 yr)."""
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return max(S - K, 0.0)
        T = T_days / 252.0
        d1 = cls._d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return S * cls._ncdf(d1) - K * cls._ncdf(d2)

    @classmethod
    def _bs_delta(cls, S: float, K: float, T_days: float, sigma: float) -> float:
        """BS call delta = N(d1). Range: (0, 1)."""
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return 1.0 if S > K else 0.0
        T = T_days / 252.0
        return cls._ncdf(cls._d1(S, K, T, sigma))

    @classmethod
    def _bs_vega(cls, S: float, K: float, T_days: float, sigma: float) -> float:
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return 0.0
        T = T_days / 252.0
        d1 = cls._d1(S, K, T, sigma)
        return S * math.sqrt(T) * cls._npdf(d1)

    @classmethod
    def _bs_iv(cls, C: float, S: float, K: float, T_days: float) -> float:
        """Implied vol via Newton-Raphson. Returns NaN if unsolvable."""
        intrinsic = max(S - K, 0.0)
        if C < intrinsic - 0.5 or T_days <= 1e-6 or S <= 0.0:
            return float("nan")
        C = max(C, intrinsic + 1e-6)
        T = T_days / 252.0
        sigma = math.sqrt(2.0 * math.pi / T) * C / S   # Brenner-Subrahmanyam init
        sigma = max(0.05, min(sigma, 5.0))
        for _ in range(50):
            price = cls._bs_call(S, K, T_days, sigma)
            vega  = cls._bs_vega(S, K, T_days, sigma)
            if abs(vega) < 1e-10:
                break
            sigma -= (price - C) / vega
            sigma = max(1e-6, min(sigma, 10.0))
        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # Parabola smile fit: IV = a·m² + b·m + c  where m = log(K/S)
    # Solved via normal equations + Cramer's rule (no numpy).
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_smile(pairs: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """Least-squares quadratic fit to (moneyness, iv) pairs. Returns (a,b,c) or None."""
        if len(pairs) < 3:
            return None
        sx4 = sx3 = sx2 = sx1 = sn = syx2 = syx1 = sy = 0.0
        for x, y in pairs:
            x2 = x * x
            sx4 += x2 * x2; sx3 += x2 * x; sx2 += x2; sx1 += x; sn += 1
            syx2 += y * x2;  syx1 += y * x; sy += y
        M = [[sx4, sx3, sx2], [sx3, sx2, sx1], [sx2, sx1, sn]]
        r = [syx2, syx1, sy]

        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                  - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                  + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

        def sub(m, v, c):
            o = [row[:] for row in m]
            for i in range(3): o[i][c] = v[i]
            return o

        D = det3(M)
        if abs(D) < 1e-12:
            return None
        return det3(sub(M, r, 0))/D, det3(sub(M, r, 1))/D, det3(sub(M, r, 2))/D

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ewma(prev: Optional[float], val: float, alpha: float) -> float:
        return float(val) if prev is None else (1.0 - alpha) * float(prev) + alpha * float(val)

    @staticmethod
    def _book(depth: Optional[OrderDepth]) -> Optional[Tuple[int, int, int, int]]:
        """Returns (best_bid, bid_vol, best_ask, ask_vol) or None."""
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None
        bb = max(depth.buy_orders); ba = min(depth.sell_orders)
        bbv = depth.buy_orders[bb]; bav = -depth.sell_orders[ba]
        if bb <= 0 or ba <= 0 or bbv <= 0 or bav <= 0 or bb >= ba:
            return None
        return bb, bbv, ba, bav

    @staticmethod
    def _buy_room(pos: int, pending: int, limit: int) -> int:
        return max(0, limit - pos - pending)

    @staticmethod
    def _sell_room(pos: int, pending: int, limit: int) -> int:
        return max(0, limit + pos - pending)

    def _tte(self, day: int, timestamp: int) -> float:
        base = self.TTE_BY_DAY.get(day, max(0.1, 5.0 - day))
        return max(0.0, base - timestamp / 1_000_000)

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, raw: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "hg_ema": None, "hg_prev_mid": None,
            "day": 0, "prev_ts": -1,
        }
        if not raw:
            return default
        try:
            d = json.loads(raw)
        except Exception:
            return default
        if not isinstance(d, dict):
            return default
        for k, v in default.items():
            if k not in d:
                d[k] = v
        return d

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HYDROGEL MM — simple EMA + inventory skew + forced unwind
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_hg(self, state: TradingState, data: Dict) -> Tuple[List[Order], Dict]:
        depth = state.order_depths.get(self.HG)
        bk = self._book(depth)
        if bk is None:
            return [], data

        bb, bbv, ba, bav = bk
        pos = state.position.get(self.HG, 0)
        orders: List[Order] = []
        pb = ps = 0

        wmid = (bb * bav + ba * bbv) / (bbv + bav)
        ema = self._ewma(data["hg_ema"], wmid, self.HG_EMA_ALPHA)
        data["hg_ema"] = ema
        data["hg_prev_mid"] = (bb + ba) / 2.0

        # Inventory skew: shift fair value against our position
        skew = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew

        bid_p = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p = math.ceil(fair_q + self.HG_QUOTE_TICK)

        # Hard limit emergency
        if pos >= self.HG_HARD_LIMIT:
            qty = min(pos - self.HG_HARD_LIMIT + self.HG_UNWIND_SIZE,
                      self.HG_UNWIND_SIZE, bbv,
                      self._sell_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, bb, -qty))
            return orders, data
        if pos <= -self.HG_HARD_LIMIT:
            qty = min(abs(pos) - self.HG_HARD_LIMIT + self.HG_UNWIND_SIZE,
                      self.HG_UNWIND_SIZE, bav,
                      self._buy_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, ba, qty))
            return orders, data

        # Soft limit: stop adding in the direction we're already leaning
        bq = 0 if pos >= self.HG_SOFT_LIMIT else min(self.HG_QUOTE_SIZE,
                                                      self._buy_room(pos, 0, self.HG_LIMIT))
        aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(self.HG_QUOTE_SIZE,
                                                       self._sell_room(pos, 0, self.HG_LIMIT))

        # Forced unwind: when stuck at soft limit, ensure we're quoting the reducing side
        if pos >= self.HG_SOFT_LIMIT:
            aq = min(self.HG_QUOTE_SIZE, self._sell_room(pos, 0, self.HG_LIMIT))
            ask_p = min(ask_p, bb + 1)   # inside spread to guarantee fills
        if pos <= -self.HG_SOFT_LIMIT:
            bq = min(self.HG_QUOTE_SIZE, self._buy_room(pos, 0, self.HG_LIMIT))
            bid_p = max(bid_p, ba - 1)

        if bq > 0:
            orders.append(Order(self.HG, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.HG, ask_p, -aq))

        logger.print(f"[HG] pos={pos} fair={ema:.1f} bid={bid_p}×{bq} ask={ask_p}×{aq}")
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FLOOR-PRICE PASSIVE SHORT — VEV_6000 / VEV_6500
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_floor_short(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        for K in self.FLOOR_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            pos  = state.position.get(sym, 0)
            room = self._sell_room(pos, 0, self.FLOOR_POS_CAP)
            qty  = min(self.FLOOR_SHORT_SIZE, room)
            if qty > 0:
                result[sym] = [Order(sym, 1, -qty)]
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ATM OPTIONS — smile fit + aggressive MM + delta hedge
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        Returns (option_orders, vf_hedge_orders).
        Fits IV smile, prices each ATM strike, places aggressive + passive orders,
        then computes total option delta and hedges with VF Extract.
        """
        opt_orders: Dict[str, List[Order]] = {}
        vf_orders:  List[Order]            = []

        if tte < self.OPT_MIN_TTE:
            return opt_orders, vf_orders

        # ── Get underlying price ──────────────────────────────────────────────
        vf_bk = self._book(state.order_depths.get(self.VF))
        if vf_bk is None:
            return opt_orders, vf_orders
        S = (vf_bk[0] + vf_bk[2]) / 2.0

        # ── Step 1: collect (moneyness, IV) pairs for smile fit ───────────────
        smile_pairs: List[Tuple[float, float]]           = []
        opt_books:   Dict[int, Tuple[int, int, int, int]] = {}
        opt_iv:      Dict[int, float]                    = {}

        for K in self.ATM_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            bk    = self._book(depth)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            mid = (bb + ba) / 2.0

            # Skip if time value is negligible
            tv = mid - max(S - K, 0.0)
            if tv < self.OPT_MIN_TV:
                continue

            iv = self._bs_iv(mid, S, float(K), tte)
            if math.isnan(iv) or iv < 0.05 or iv > 3.0:
                continue

            m = math.log(K / S)   # log-moneyness: negative=ITM, positive=OTM
            smile_pairs.append((m, iv))
            opt_books[K] = (bb, bbv, ba, bav)
            opt_iv[K]    = iv

        # ── Step 2: fit smile parabola ────────────────────────────────────────
        coeffs = self._fit_smile(smile_pairs) if len(smile_pairs) >= self.SMILE_MIN_POINTS else None

        # ── Step 3: for each strike, compute fair price and place orders ──────
        fitted_iv: Dict[int, float] = {}

        for K in self.ATM_STRIKES:
            if K not in opt_books:
                continue

            sym = self.VEV[K]
            bb, bbv, ba, bav = opt_books[K]
            pos = state.position.get(sym, 0)
            m   = math.log(K / S)

            # Determine fitted IV (use parabola if available, else raw market IV)
            if coeffs is not None:
                a, b, c = coeffs
                fiv = a * m * m + b * m + c
                fiv = max(0.05, fiv)
            else:
                fiv = opt_iv.get(K, float("nan"))
                if math.isnan(fiv):
                    continue

            fitted_iv[K] = fiv

            # Fair price from Black-Scholes with fitted IV
            fair = self._bs_call(S, float(K), tte, fiv)

            orders: List[Order] = []
            pb = ps = 0

            # ── Aggressive taker ──────────────────────────────────────────────
            # Buy if market is cheap relative to fair
            if ba < fair - self.OPT_TAKER_EDGE:
                room = self._buy_room(pos, pb, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pb += qty
                    logger.print(f"[OPT TAKE BUY]  {sym} fair={fair:.2f} ask={ba} pos={pos}")

            # Sell if market is rich relative to fair
            elif bb > fair + self.OPT_TAKER_EDGE:
                room = self._sell_room(pos, ps, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bbv)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    ps += qty
                    logger.print(f"[OPT TAKE SELL] {sym} fair={fair:.2f} bid={bb} pos={pos}")

            # ── Passive quotes (willing to cross — no maker-only constraint) ──
            bid_p = round(fair) - self.OPT_PASSIVE_TICK
            ask_p = round(fair) + self.OPT_PASSIVE_TICK

            bq = min(self.OPT_PASSIVE_SIZE, self._buy_room(pos, pb, self.OPT_POS_CAP))
            aq = min(self.OPT_PASSIVE_SIZE, self._sell_room(pos, ps, self.OPT_POS_CAP))

            if bid_p > 0 and bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if ask_p > 0 and aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders

        # ── Step 4: delta hedge with VF Extract ──────────────────────────────
        # Compute total option delta across all ATM positions
        total_delta = 0.0
        for K in self.ATM_STRIKES:
            pos = state.position.get(self.VEV[K], 0)
            if pos == 0:
                continue
            fiv = fitted_iv.get(K)
            if fiv is None or math.isnan(fiv):
                continue
            delta = self._bs_delta(S, float(K), tte, fiv)
            total_delta += pos * delta

        # Target VF position to neutralize delta
        target_vf  = -round(total_delta)
        target_vf  = max(-self.VF_LIMIT, min(self.VF_LIMIT, target_vf))
        current_vf = state.position.get(self.VF, 0)
        vf_bb, vf_bbv, vf_ba, vf_bav = vf_bk

        hedge_needed = target_vf - current_vf

        if hedge_needed > 0:
            # Need to buy VF
            qty = min(hedge_needed, vf_bav, self._buy_room(current_vf, 0, self.VF_LIMIT))
            if qty > 0:
                vf_orders.append(Order(self.VF, vf_ba, qty))  # taker: lift the ask
                logger.print(f"[HEDGE BUY VF] delta={total_delta:.1f} target={target_vf} qty={qty}")

        elif hedge_needed < 0:
            # Need to sell VF
            qty = min(-hedge_needed, vf_bbv, self._sell_room(current_vf, 0, self.VF_LIMIT))
            if qty > 0:
                vf_orders.append(Order(self.VF, vf_bb, -qty))  # taker: hit the bid
                logger.print(f"[HEDGE SELL VF] delta={total_delta:.1f} target={target_vf} qty={qty}")

        return opt_orders, vf_orders

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load(state.traderData)

        # Day tracking: timestamp reset signals new game-day
        prev_ts = data["prev_ts"]
        day     = data["day"]
        if prev_ts >= 0 and state.timestamp < prev_ts - 500_000:
            day += 1
            logger.print(f"[DAY] day={day}")
        data["day"]     = day
        data["prev_ts"] = state.timestamp

        tte = self._tte(day, state.timestamp)

        result: Dict[Symbol, List[Order]] = {}

        # 1. HYDROGEL MM
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. Floor-price passive short (VEV_6000/6500)
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 3. ATM option aggressive MM + delta hedge
        opt_ords, vf_hedge = self._trade_options(state, tte)
        for sym, ords in opt_ords.items():
            result.setdefault(sym, []).extend(ords)
        if vf_hedge:
            result[self.VF] = vf_hedge

        # Ensure empty list for every VEV product not otherwise touched
        for sym in self.VEV.values():
            if sym not in result:
                result[sym] = []

        trader_data = json.dumps(data, separators=(",", ":"))
        conversions = 0
        logger.flush(state=state, orders=result, conversions=conversions, trader_data=trader_data)
        return result, conversions, trader_data