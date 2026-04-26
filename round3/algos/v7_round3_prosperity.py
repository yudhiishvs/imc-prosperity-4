"""
v7_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes from v6:
────────────────────────────────────────────────────────────────────────
1. HG taker logic (was passive-only → big loser during trending moves)
   • HG_EMA_ALPHA = 0.020 (was 0.005 — much faster, tracks price better)
   • HG_TAKE_EDGE = 8: sell at bid when bb ≥ EMA+8, buy at ask when ba ≤ EMA-8
   • Passive MM only fires when no taker signal active
   • (Determined by grid search: alpha=0.020 and edge=8 far outperform slower/tighter)

2. Option position exit at cap (was: permanently stuck at ±OPT_POS_CAP)
   • If pos ≥ OPT_POS_CAP: sell at bid to reduce (no edge required)
   • If pos ≤ -OPT_POS_CAP: buy at ask to reduce (no edge required)
   • Normal taker/passive logic only fires when below cap

3. Delta hedge deadband = 20 (was: hedge every tick, paying spread constantly)
   • Only hedge if |target_vf - current_vf| ≥ 20
   • (Grid search: deadband=20 far outperforms 0/5/10)

4. Updated params from grid search (720-combo sweep):
   • OPT_TAKER_EDGE = 1.0  (was 1.5)
   • OPT_POS_CAP    = 35   (was 50)

Design (unchanged from v6):
────────────────────────────────────────────────────────────────────────
1. FIT IV SMILE every tick (quadratic in log-moneyness, Cramer's rule)
2. AGGRESSIVE OPTION MM: take when market ≠ BS fair by > TAKER_EDGE
3. DELTA HEDGE with VF Extract (deadband to avoid over-trading)
4. VEV_6000/6500: passive short at ask=1 (floor price)
5. HYDROGEL_PACK: EMA MM with taker signal + inventory skew
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
    ATM_STRIKES   = [5000, 5100, 5200, 5300, 5400, 5500]
    FLOOR_STRIKES = [6000, 6500]

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT  = 200
    VF_LIMIT  = 200
    VEV_LIMIT = 300

    # ── HG MM parameters (v7: taker-first) ────────────────────────────────────
    HG_EMA_ALPHA   = 0.020   # v6=0.005 — faster EMA tracks price better (grid search)
    HG_TAKE_EDGE   = 8       # NEW: taker threshold in ticks from EMA
    HG_TAKER_SIZE  = 10      # NEW: max units per taker fill
    HG_QUOTE_TICK  = 2       # passive quote offset from fair
    HG_SKEW_TICKS  = 3.0    # max inventory skew at position limit
    HG_QUOTE_SIZE  = 20
    HG_SOFT_LIMIT  = 140
    HG_HARD_LIMIT  = 190
    HG_UNWIND_SIZE = 30

    # ── Option MM parameters (v7: updated from grid search) ───────────────────
    OPT_TAKER_EDGE   = 1.0   # v6=1.5 — tighter edge captures more opportunities
    OPT_POS_CAP      = 35    # v6=50  — less max accumulation risk
    OPT_PASSIVE_TICK = 1     # passive quote offset from fair
    OPT_TAKER_SIZE   = 10    # max units per taker fill
    OPT_PASSIVE_SIZE = 10    # max units per passive quote
    OPT_MIN_TTE      = 0.3   # stop trading very near expiry
    OPT_MIN_TV       = 0.3   # skip if time value below this (not enough edge)
    FLOOR_SHORT_SIZE = 20    # passive sell size per tick for floor-price strikes

    # ── Delta hedge parameters (v7: deadband to avoid over-hedging) ────────────
    HEDGE_DEADBAND = 20      # NEW: only hedge if |needed| ≥ this (grid search: 20 optimal)

    # ── Smile fit ─────────────────────────────────────────────────────────────
    SMILE_MIN_POINTS = 3     # need at least 3 IV points to fit parabola

    # ── TTE mapping (live Round 3) ─────────────────────────────────────────────
    TTE_BY_DAY = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes math  (no scipy / numpy)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ncdf(x: float) -> float:
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
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return max(S - K, 0.0)
        T = T_days / 252.0
        d1 = cls._d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return S * cls._ncdf(d1) - K * cls._ncdf(d2)

    @classmethod
    def _bs_delta(cls, S: float, K: float, T_days: float, sigma: float) -> float:
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
        intrinsic = max(S - K, 0.0)
        if C < intrinsic - 0.5 or T_days <= 1e-6 or S <= 0.0:
            return float("nan")
        C = max(C, intrinsic + 1e-6)
        T = T_days / 252.0
        sigma = math.sqrt(2.0 * math.pi / T) * C / S
        sigma = max(0.05, min(sigma, 5.0))
        for _ in range(20):
            price = cls._bs_call(S, K, T_days, sigma)
            vega  = cls._bs_vega(S, K, T_days, sigma)
            if abs(vega) < 1e-10:
                break
            sigma -= (price - C) / vega
            sigma = max(1e-6, min(sigma, 10.0))
        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # Parabola smile fit: IV = a·m² + b·m + c  where m = log(K/S)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_smile(pairs: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
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
    # 1. HYDROGEL MM — taker-first, then passive with inventory skew
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_hg(self, state: TradingState, data: Dict) -> Tuple[List[Order], Dict]:
        depth = state.order_depths.get(self.HG)
        bk = self._book(depth)
        if bk is None:
            return [], data

        bb, bbv, ba, bav = bk
        pos = state.position.get(self.HG, 0)
        orders: List[Order] = []

        wmid = (bb * bav + ba * bbv) / (bbv + bav)
        ema = self._ewma(data["hg_ema"], wmid, self.HG_EMA_ALPHA)
        data["hg_ema"] = ema
        data["hg_prev_mid"] = (bb + ba) / 2.0

        # ── Hard limit emergency: forced unwind at market ──────────────────────
        if pos >= self.HG_HARD_LIMIT:
            qty = min(self.HG_UNWIND_SIZE, bbv,
                      self._sell_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, bb, -qty))
            return orders, data
        if pos <= -self.HG_HARD_LIMIT:
            qty = min(self.HG_UNWIND_SIZE, bav,
                      self._buy_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, ba, qty))
            return orders, data

        # ── Taker signal: cross when price significantly off EMA ───────────────
        # Sell into strength when bid is high vs EMA (price above fair)
        if bb >= ema + self.HG_TAKE_EDGE:
            qty = min(self.HG_TAKER_SIZE, bbv,
                      self._sell_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, bb, -qty))
                logger.print(f"[HG TAKE SELL] bb={bb} ema={ema:.1f} pos={pos}")
            return orders, data   # don't post passive same tick as taker

        # Buy into weakness when ask is low vs EMA (price below fair)
        if ba <= ema - self.HG_TAKE_EDGE:
            qty = min(self.HG_TAKER_SIZE, bav,
                      self._buy_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, ba, qty))
                logger.print(f"[HG TAKE BUY] ba={ba} ema={ema:.1f} pos={pos}")
            return orders, data   # don't post passive same tick as taker

        # ── Passive MM: no taker signal — quote around fair ───────────────────
        skew  = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew
        bid_p = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p = math.ceil(fair_q + self.HG_QUOTE_TICK)

        bq = 0 if pos >= self.HG_SOFT_LIMIT  else min(self.HG_QUOTE_SIZE,
                                                        self._buy_room(pos, 0, self.HG_LIMIT))
        aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(self.HG_QUOTE_SIZE,
                                                        self._sell_room(pos, 0, self.HG_LIMIT))

        # Forced unwind at soft limit: quote aggressively on the reducing side
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
            room = self._sell_room(pos, 0, self.VEV_LIMIT)
            qty  = min(self.FLOOR_SHORT_SIZE, room)
            if qty > 0:
                result[sym] = [Order(sym, 1, -qty)]
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ATM OPTIONS — smile fit + taker MM + cap-exit + delta hedge
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
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
        smile_pairs: List[Tuple[float, float]]            = []
        opt_books:   Dict[int, Tuple[int, int, int, int]] = {}
        opt_iv:      Dict[int, float]                     = {}

        for K in self.ATM_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            bk    = self._book(depth)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            mid = (bb + ba) / 2.0
            tv  = mid - max(S - K, 0.0)
            if tv < self.OPT_MIN_TV:
                continue
            iv = self._bs_iv(mid, S, float(K), tte)
            if math.isnan(iv) or iv < 0.05 or iv > 3.0:
                continue
            m = math.log(K / S)
            smile_pairs.append((m, iv))
            opt_books[K] = (bb, bbv, ba, bav)
            opt_iv[K]    = iv

        # ── Step 2: fit smile parabola ────────────────────────────────────────
        coeffs = self._fit_smile(smile_pairs) if len(smile_pairs) >= self.SMILE_MIN_POINTS else None

        # ── Step 3: per-strike order logic ────────────────────────────────────
        fitted_iv: Dict[int, float] = {}

        for K in self.ATM_STRIKES:
            if K not in opt_books:
                continue

            sym = self.VEV[K]
            bb, bbv, ba, bav = opt_books[K]
            pos = state.position.get(sym, 0)
            m   = math.log(K / S)

            if coeffs is not None:
                a, b, c = coeffs
                fiv = max(0.05, a * m * m + b * m + c)
            else:
                fiv = opt_iv.get(K, float("nan"))
                if math.isnan(fiv):
                    continue

            fitted_iv[K] = fiv
            fair = self._bs_call(S, float(K), tte, fiv)

            orders: List[Order] = []
            pb = ps = 0

            # ── Priority 1: Exit when at position cap (v7 fix) ────────────────
            # If stuck long at cap, sell at bid without requiring edge.
            # This ensures we always have an exit and never accumulate indefinitely.
            if pos >= self.OPT_POS_CAP:
                qty = min(self.OPT_TAKER_SIZE, bbv)
                if qty > 0 and bb > 0:
                    orders.append(Order(sym, bb, -qty))
                    ps += qty
                    logger.print(f"[OPT EXIT SELL] {sym} pos={pos} bb={bb} fair={fair:.2f}")
                # Still allow passive ask below to catch more fills
                ask_p = round(fair) + self.OPT_PASSIVE_TICK
                aq = min(self.OPT_PASSIVE_SIZE, self._sell_room(pos, ps, self.OPT_POS_CAP))
                if ask_p > 0 and aq > 0:
                    orders.append(Order(sym, ask_p, -aq))
                if orders:
                    opt_orders[sym] = orders
                continue

            if pos <= -self.OPT_POS_CAP:
                qty = min(self.OPT_TAKER_SIZE, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pb += qty
                    logger.print(f"[OPT EXIT BUY]  {sym} pos={pos} ba={ba} fair={fair:.2f}")
                # Still allow passive bid below
                bid_p = round(fair) - self.OPT_PASSIVE_TICK
                bq = min(self.OPT_PASSIVE_SIZE, self._buy_room(pos, pb, self.OPT_POS_CAP))
                if bid_p > 0 and bq > 0:
                    orders.append(Order(sym, bid_p, bq))
                if orders:
                    opt_orders[sym] = orders
                continue

            # ── Priority 2: Aggressive taker ──────────────────────────────────
            did_taker_buy = did_taker_sell = False

            if ba < fair - self.OPT_TAKER_EDGE:
                room = self._buy_room(pos, pb, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pb += qty
                    did_taker_buy = True
                    logger.print(f"[OPT TAKE BUY]  {sym} fair={fair:.2f} ask={ba} pos={pos}")

            elif bb > fair + self.OPT_TAKER_EDGE:
                room = self._sell_room(pos, ps, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bbv)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    ps += qty
                    did_taker_sell = True
                    logger.print(f"[OPT TAKE SELL] {sym} fair={fair:.2f} bid={bb} pos={pos}")

            # ── Priority 3: Passive quotes ────────────────────────────────────
            # Suppress passive bid if we just bought aggressively (don't double-buy)
            # Suppress passive ask if we just sold aggressively (don't double-sell)
            bid_p = round(fair) - self.OPT_PASSIVE_TICK
            ask_p = round(fair) + self.OPT_PASSIVE_TICK

            bq = 0 if did_taker_buy  else min(self.OPT_PASSIVE_SIZE,
                                               self._buy_room(pos, pb, self.OPT_POS_CAP))
            aq = 0 if did_taker_sell else min(self.OPT_PASSIVE_SIZE,
                                               self._sell_room(pos, ps, self.OPT_POS_CAP))

            if bid_p > 0 and bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if ask_p > 0 and aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders

        # ── Step 4: delta hedge with VF Extract (deadband = 20) ───────────────
        total_delta = 0.0
        for K in self.ATM_STRIKES:
            p = state.position.get(self.VEV[K], 0)
            if p == 0:
                continue
            fiv = fitted_iv.get(K)
            if fiv is None or math.isnan(fiv):
                continue
            total_delta += p * self._bs_delta(S, float(K), tte, fiv)

        target_vf  = max(-self.VF_LIMIT, min(self.VF_LIMIT, -round(total_delta)))
        current_vf = state.position.get(self.VF, 0)
        hedge_needed = target_vf - current_vf

        # Only hedge when imbalance exceeds deadband (avoid paying spread every tick)
        if abs(hedge_needed) >= self.HEDGE_DEADBAND:
            vf_bb, vf_bbv, vf_ba, vf_bav = vf_bk
            if hedge_needed > 0:
                qty = min(hedge_needed, vf_bav,
                          self._buy_room(current_vf, 0, self.VF_LIMIT))
                if qty > 0:
                    vf_orders.append(Order(self.VF, vf_ba, qty))
                    logger.print(f"[HEDGE BUY VF]  delta={total_delta:.1f} tgt={target_vf} qty={qty}")
            else:
                qty = min(-hedge_needed, vf_bbv,
                          self._sell_room(current_vf, 0, self.VF_LIMIT))
                if qty > 0:
                    vf_orders.append(Order(self.VF, vf_bb, -qty))
                    logger.print(f"[HEDGE SELL VF] delta={total_delta:.1f} tgt={target_vf} qty={qty}")

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

        # 1. HYDROGEL MM (taker-first)
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. Floor-price passive short (VEV_6000/6500)
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 3. ATM option MM + delta hedge (with cap-exit and deadband)
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
