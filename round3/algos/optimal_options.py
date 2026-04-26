"""
v19_round3_prosperity.py
IMC Prosperity 4 — Round 3

CORE ALPHA (from EDA chart — VEV mid vs Strike by VFE quantile bucket):
  Option mids are nearly CONSTANT across all VFE quantile buckets (5198–5300).
  => Bots price VEVs at a fixed/slow anchor (~median VFE ≈ 5249), not real-time spot.
  => When VFE > anchor: options CHEAP vs fair BS model → BUY aggressively.
  => When VFE < anchor: options EXPENSIVE → SELL vol premium hard.

Changes vs v18:
──────────────────────────────────────────────────────────────────────────────
1. Track VFE_EMA (alpha=0.002) as proxy for bot anchor price in traderData.
2. delta_S = S_current - vf_ema; apply size skew per option:
     delta_S > +VF_SKEW_THRESH (+15): bid_mult=2.5  ask_mult=0.4  (buy cheap opts)
     delta_S < -VF_SKEW_THRESH (-15): bid_mult=0.4  ask_mult=2.5  (sell vol premium)
     |delta_S| <= 15: symmetric (1.0/1.0)
3. Anchor-taker: when delta_S confirms direction AND market is mispriced vs current BS:
     High VFE: if ask <= BS(S, K, T, sigma_mid) - ANCHOR_TAKER_EDGE → TAKE BUY
     Low  VFE: if bid >= BS(S, K, T, sigma_mid) + ANCHOR_TAKER_EDGE → TAKE SELL
4. OPT_TAKER_EDGE     0.5  → 0.3 (more capture)
5. OPT_PASSIVE_SIZE   25   → 30
6. OPT_TAKER_SIZE     10   → 15
7. VF_QUOTE_SIZE      25   → 30
8. VF_MM_POS_CAP      150  → 175
9. FLOOR_SHORT_SIZE   20   → 35
10. ANCHOR_TAKER_EDGE = 1.0  (separate from OPT_TAKER_EDGE for anchor taker)
11. VF_SKEW_THRESH    = 15.0 (price units; ~1/3 of observed VFE half-range)
    VF_SKEW_STRONG    = 2.5
    VF_SKEW_WEAK      = 0.4
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

    ACTIVE_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400]
    FLOOR_STRIKES  = [6000, 6500]
    DEEP_ITM_STRIKES: list = []

    SMILE_STRIKES = [5000, 5100, 5200, 5300, 5400]

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT  = 200
    VF_LIMIT  = 200
    VEV_LIMIT = 300

    OPT_POS_CAP   = 150
    OPT_SOFT_CAP  = 60
    ITM_POS_CAP   = 50
    ITM_SOFT_CAP  = 25
    FLOOR_POS_CAP = 200

    # ── Pre-calibrated vol smile ────────────────────────────────────────────
    # IV = a*m^2 + b*m + c, m = log(S/K)/sqrt(T)
    # Flat ~10% IV fitted from 3-day market data (pipe-shape).
    # Updated Constants for v20


    # ── TTE ───────────────────────────────────────────────────────────────────
    TTE_START      = 7.0
    MAX_TS_PER_DAY = 999_900

    # ── Option MM parameters ──────────────────────────────────────────────────
    OPT_CAP_SIZE     = 15
    OPT_PASSIVE_SIZE = 30   # v19: up from 25
    OPT_MIN_TTE      = 0.3
    OPT_MIN_TV       = 0.5

    FLOOR_SHORT_SIZE = 35   # v19: up from 20

    VOL_SKEW_THRESH  = 0.0008
    VOL_SKEW_BOOST   = 1.50
    VOL_SKEW_CUT     = 0.75

    # ── Model-fair taker ──────────────────────────────────────────────────────
    OPT_TAKER_EDGE = 0.3    # v19: down from 0.5 (more capture)
    OPT_TAKER_SIZE = 15     # v19: up from 10

    # ── VFE anchor skew (CORE v19 ALPHA) ─────────────────────────────────────
    # Bots price options at a fixed anchor ~= slow VFE EMA.
    # When current VFE deviates from anchor, we skew our quoting:
    #   VFE > anchor (delta_S > +THRESH): options cheap → bid×STRONG, ask×WEAK
    #   VFE < anchor (delta_S < -THRESH): options rich  → bid×WEAK, ask×STRONG
    VF_EMA_ALPHA      = 0.002   # very slow: proxy for bot anchor
    VF_SKEW_THRESH    = 15.0    # price units (≈1/3 of observed VFE half-range ~50)
    VF_SKEW_STRONG    = 2.5     # multiplier for favored side
    VF_SKEW_WEAK      = 0.4     # multiplier for adverse side

    # Anchor-based taker: fires when VFE is far from anchor AND market is mispriced
    ANCHOR_TAKER_EDGE = 1.0
    ANCHOR_TAKER_SIZE = 15

    # ── Delta hedge — disabled ────────────────────────────────────────────────
    HEDGE_DEADBAND = 999

    # ── VF passive MM parameters ──────────────────────────────────────────────
    VF_QUOTE_SIZE   = 30    # v19: up from 25
    VF_MM_POS_CAP   = 175   # v19: up from 150
    VF_MM_SOFT_CAP  = 87    # v19: up from 75
    VF_EOD_TS       = 950_000


    ASK_SMILE = (0.18, 0.12, 0.095)
    BID_SMILE = (0.15, 0.08, 0.085)
    VF_EMA_ALPHA = 0.0015
    VF_SKEW_THRESH = 12.0
    VF_SKEW_STRONG = 3.0
    VF_SKEW_WEAK = 0.3
    OPT_TAKER_EDGE = 0.25
    ANCHOR_TAKER_EDGE = 0.8
    OPT_PASSIVE_SIZE = 40

    # ── HG MM parameters ──────────────────────────────────────────────────────
    HG_EMA_ALPHA   = 0.050
    HG_TREND_ALPHA = 0.005
    HG_TREND_GAP   = 8
    HG_TAKE_EDGE   = 20
    HG_TAKER_SIZE  = 8
    HG_QUOTE_TICK  = 2
    HG_SKEW_TICKS  = 5.0
    HG_QUOTE_SIZE  = 15
    HG_SOFT_LIMIT  = 30
    HG_HARD_LIMIT  = 190
    HG_UNWIND_SIZE = 30
    HG_VOL_ALPHA   = 0.10
    HG_VOL_THRESH  = 2.5
    HG_VOL_SIZE    = 8
    HG_EOD_TS      = 950_000




    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes
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
    def _bs_call(cls, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 1e-8 or sigma <= 1e-8 or S <= 0.0:
            return max(S - K, 0.0)
        sq = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
        d2 = d1 - sigma * sq
        return S * cls._ncdf(d1) - K * cls._ncdf(d2)

    @classmethod
    def _bs_delta(cls, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 1e-8 or sigma <= 1e-8 or S <= 0.0:
            return 1.0 if S > K else 0.0
        sq = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
        return cls._ncdf(d1)

    @classmethod
    def _bs_iv_bisect(cls, C: float, S: float, K: float, T: float) -> float:
        intrinsic = max(S - K, 0.0)
        if C <= intrinsic + 1e-6 or T <= 1e-8 or S <= 0.0:
            return float("nan")
        lo, hi = 1e-6, 5.0
        if cls._bs_call(S, K, T, hi) < C:
            return float("nan")
        for _ in range(60):
            mid_s = (lo + hi) / 2.0
            if cls._bs_call(S, K, T, mid_s) >= C:
                hi = mid_s
            else:
                lo = mid_s
            if hi - lo < 1e-7:
                break
        return (lo + hi) / 2.0

    # ─────────────────────────────────────────────────────────────────────────
    # Smile helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smile_iv(m: float, params: Tuple[float, float, float]) -> float:
        a, b, c = params
        return max(0.001, a * m * m + b * m + c)

    # ─────────────────────────────────────────────────────────────────────────
    # Live IV smile fit (for vol-skew sizing)
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_smile(
        self, state: TradingState, S: float, tte: float
    ) -> Dict[int, float]:
        if tte < self.OPT_MIN_TTE:
            return {}

        points: List[Tuple[float, float, int]] = []
        for K in self.SMILE_STRIKES:
            sym = self.VEV[K]
            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            bb = max(depth.buy_orders)
            ba = min(depth.sell_orders)
            if bb <= 0 or ba <= 0 or bb >= ba:
                continue
            mid = (bb + ba) / 2.0
            if mid <= max(S - float(K), 0.0) + 0.5:
                continue
            iv = self._bs_iv_bisect(mid, S, float(K), tte)
            if math.isnan(iv):
                continue
            m = math.log(S / float(K)) / math.sqrt(tte)
            points.append((m, iv, K))

        if len(points) < 3:
            return {}

        n   = len(points)
        sx  = sum(p[0] for p in points)
        sy  = sum(p[1] for p in points)
        sxy = sum(p[0] * p[1] for p in points)
        sx2 = sum(p[0] * p[0] for p in points)
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-15:
            return {}
        slope     = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n

        residuals: Dict[int, float] = {}
        for m, iv, K in points:
            fitted = slope * m + intercept
            residuals[K] = iv - fitted
        return residuals

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
        bb = max(depth.buy_orders);  ba = min(depth.sell_orders)
        bbv = depth.buy_orders[bb];  bav = -depth.sell_orders[ba]
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
        return max(0.0, self.TTE_START - day - timestamp / self.MAX_TS_PER_DAY)

    def _infer_tte_from_market(self, state: TradingState) -> Optional[float]:
        vf_bk = self._book(state.order_depths.get(self.VF))
        if vf_bk is None:
            return None
        S = (vf_bk[0] + vf_bk[2]) / 2.0

        ma = (self.ASK_SMILE[0] + self.BID_SMILE[0]) / 2.0
        mb = (self.ASK_SMILE[1] + self.BID_SMILE[1]) / 2.0
        mc = (self.ASK_SMILE[2] + self.BID_SMILE[2]) / 2.0

        estimates: List[float] = []
        for K in self.ACTIVE_STRIKES:
            sym = self.VEV[K]
            bk = self._book(state.order_depths.get(sym))
            if bk is None:
                continue
            C_obs = (bk[0] + bk[2]) / 2.0
            intrinsic = max(S - K, 0.0)
            if C_obs <= intrinsic + 0.5:
                continue

            lo, hi = 0.05, self.TTE_START
            for _ in range(40):
                T = (lo + hi) / 2.0
                m = math.log(S / float(K)) / math.sqrt(T)
                sigma = max(0.001, ma * m * m + mb * m + mc)
                C_theo = self._bs_call(S, float(K), T, sigma)
                if C_theo > C_obs:
                    hi = T
                else:
                    lo = T
                if hi - lo < 0.01:
                    break
            estimates.append((lo + hi) / 2.0)

        if len(estimates) < 2:
            return None
        estimates.sort()
        return estimates[len(estimates) // 2]

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, raw: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "hg_ema": None, "hg_trend_ema": None, "hg_prev_mid": None,
            "hg_vol": None,
            "day": 0, "prev_ts": -1,
            "vf_ema": None,  # v19: VFE anchor EMA
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
    # 1. HYDROGEL MM
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

        prev_mid = data["hg_prev_mid"]
        if prev_mid is not None:
            price_chg = abs(wmid - prev_mid)
            hg_vol = self._ewma(data["hg_vol"], price_chg, self.HG_VOL_ALPHA)
        else:
            hg_vol = data["hg_vol"] if data["hg_vol"] is not None else 0.0
        data["hg_vol"] = hg_vol

        high_vol = hg_vol > self.HG_VOL_THRESH
        q_size = self.HG_VOL_SIZE if high_vol else self.HG_QUOTE_SIZE

        ema       = self._ewma(data["hg_ema"],       wmid, self.HG_EMA_ALPHA)
        trend_ema = self._ewma(data["hg_trend_ema"], wmid, self.HG_TREND_ALPHA)
        data["hg_ema"]       = ema
        data["hg_trend_ema"] = trend_ema
        data["hg_prev_mid"]  = wmid

        fast_vs_trend = ema - trend_ema
        in_downtrend  = fast_vs_trend < -self.HG_TREND_GAP
        in_uptrend    = fast_vs_trend >  self.HG_TREND_GAP

        if state.timestamp > self.HG_EOD_TS and pos != 0:
            if pos > 0:
                qty = min(pos, bbv, self._sell_room(pos, 0, self.HG_LIMIT))
                if qty > 0:
                    orders.append(Order(self.HG, bb, -qty))
                    logger.print(f"[HG EOD SELL] pos={pos} qty={qty}@{bb}")
            else:
                qty = min(-pos, bav, self._buy_room(pos, 0, self.HG_LIMIT))
                if qty > 0:
                    orders.append(Order(self.HG, ba, qty))
                    logger.print(f"[HG EOD BUY ] pos={pos} qty={qty}@{ba}")
            return orders, data

        if pos >= self.HG_HARD_LIMIT:
            qty = min(self.HG_UNWIND_SIZE, bbv, self._sell_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, bb, -qty))
            return orders, data
        if pos <= -self.HG_HARD_LIMIT:
            qty = min(self.HG_UNWIND_SIZE, bav, self._buy_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, ba, qty))
            return orders, data

        if bb >= ema + self.HG_TAKE_EDGE:
            qty = min(self.HG_TAKER_SIZE, bbv, self._sell_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, bb, -qty))
                logger.print(f"[HG TAKE SELL] bb={bb} ema={ema:.1f}")
            return orders, data
        if ba <= ema - self.HG_TAKE_EDGE:
            qty = min(self.HG_TAKER_SIZE, bav, self._buy_room(pos, 0, self.HG_LIMIT))
            if qty > 0:
                orders.append(Order(self.HG, ba, qty))
                logger.print(f"[HG TAKE BUY] ba={ba} ema={ema:.1f}")
            return orders, data

        if ba - bb < 3:
            return orders, data
        bid_p = bb + 1
        ask_p = ba - 1

        base_bq = 0 if pos >= self.HG_SOFT_LIMIT else min(
            q_size, self._buy_room(pos, 0, self.HG_LIMIT))
        base_aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(
            q_size, self._sell_room(pos, 0, self.HG_LIMIT))

        if in_downtrend:
            bq = 0
            aq = base_aq
        elif in_uptrend:
            bq = base_bq
            aq = 0
        else:
            bq = base_bq
            aq = base_aq

        if pos >= self.HG_SOFT_LIMIT:
            aq = min(q_size, self._sell_room(pos, 0, self.HG_LIMIT))
        if pos <= -self.HG_SOFT_LIMIT:
            bq = min(q_size, self._buy_room(pos, 0, self.HG_LIMIT))

        if bq > 0:
            orders.append(Order(self.HG, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.HG, ask_p, -aq))

        trend_str = "DOWN" if in_downtrend else ("UP" if in_uptrend else "flat")
        logger.print(
            f"[HG] pos={pos} ema={ema:.1f} trend={trend_ema:.1f}({trend_str})"
            f" vol={hg_vol:.2f} bid={bid_p}×{bq} ask={ask_p}×{aq}"
        )
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VF PASSIVE MM
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vf(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.VF)
        bk = self._book(depth)
        if bk is None:
            return []

        bb, bbv, ba, bav = bk
        pos = state.position.get(self.VF, 0)
        spread = ba - bb
        orders: List[Order] = []

        if state.timestamp > self.VF_EOD_TS and pos != 0:
            if pos > 0:
                qty = min(pos, bbv, self._sell_room(pos, 0, self.VF_MM_POS_CAP))
                if qty > 0:
                    orders.append(Order(self.VF, bb, -qty))
                    logger.print(f"[VF EOD SELL] pos={pos} qty={qty}@{bb}")
            else:
                qty = min(-pos, bav, self._buy_room(pos, 0, self.VF_MM_POS_CAP))
                if qty > 0:
                    orders.append(Order(self.VF, ba, qty))
                    logger.print(f"[VF EOD BUY ] pos={pos} qty={qty}@{ba}")
            return orders

        if spread < 3:
            return []

        bid_p = bb + 1
        ask_p = ba - 1

        if pos >= self.VF_MM_SOFT_CAP:
            aq = min(self.VF_QUOTE_SIZE, self._sell_room(pos, 0, self.VF_MM_POS_CAP))
            if aq > 0:
                orders.append(Order(self.VF, ask_p, -aq))
        elif pos <= -self.VF_MM_SOFT_CAP:
            bq = min(self.VF_QUOTE_SIZE, self._buy_room(pos, 0, self.VF_MM_POS_CAP))
            if bq > 0:
                orders.append(Order(self.VF, bid_p, bq))
        else:
            bq = min(self.VF_QUOTE_SIZE, self._buy_room(pos, 0, self.VF_MM_POS_CAP))
            aq = min(self.VF_QUOTE_SIZE, self._sell_room(pos, 0, self.VF_MM_POS_CAP))
            if bq > 0:
                orders.append(Order(self.VF, bid_p, bq))
            if aq > 0:
                orders.append(Order(self.VF, ask_p, -aq))

        if orders:
            logger.print(f"[VF] pos={pos} spread={spread} bid={bid_p} ask={ask_p}")
        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FLOOR-PRICE PASSIVE SHORT — VEV_6000 / VEV_6500
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
    # 4. OPTIONS MM + VFE anchor skew (CORE v19 ALPHA)
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float, vf_ema: Optional[float]
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        Passive MM for ACTIVE_STRIKES.

        v19 alpha: bots price VEV options at a fixed anchor (slow VFE EMA).
        When current VFE deviates from anchor:
          delta_S > +VF_SKEW_THRESH: options cheap → bid×STRONG, ask×WEAK
                                     anchor-taker fires on cheap asks
          delta_S < -VF_SKEW_THRESH: options rich  → bid×WEAK, ask×STRONG
                                     anchor-taker fires on rich bids (vol premium)
        """
        opt_orders: Dict[str, List[Order]] = {}
        vf_orders:  List[Order]            = []

        if tte < self.OPT_MIN_TTE:
            return opt_orders, vf_orders

        vf_bk = self._book(state.order_depths.get(self.VF))
        if vf_bk is None:
            return opt_orders, vf_orders
        vf_bb, vf_bbv, vf_ba, vf_bav = vf_bk
        S = (vf_bb + vf_ba) / 2.0

        # ── VFE anchor skew: compute bid/ask multipliers ──────────────────────
        anchor = vf_ema if vf_ema is not None else S
        delta_S = S - anchor

        if delta_S > self.VF_SKEW_THRESH:
            bid_mult = self.VF_SKEW_STRONG   # options cheap: buy hard
            ask_mult = self.VF_SKEW_WEAK
            anchor_mode = "HIGH"
        elif delta_S < -self.VF_SKEW_THRESH:
            bid_mult = self.VF_SKEW_WEAK
            ask_mult = self.VF_SKEW_STRONG   # options rich: sell vol premium
            anchor_mode = "LOW"
        else:
            bid_mult = 1.0
            ask_mult = 1.0
            anchor_mode = "FLAT"

        logger.print(
            f"[VFE ANCHOR] S={S:.1f} anchor={anchor:.1f} dS={delta_S:+.1f}"
            f" mode={anchor_mode} bid_mult={bid_mult} ask_mult={ask_mult}"
        )

        # Live IV smile for cross-strike vol-skew sizing (unchanged from v18)
        vol_residuals = self._fit_smile(state, S, tte)

        mid_iv_by_K: Dict[int, float] = {}

        for K in self.ACTIVE_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            bk    = self._book(depth)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            pos     = state.position.get(sym, 0)
            spread  = ba - bb

            if spread < 2:
                continue

            is_itm   = K <= 4500
            pos_cap  = self.ITM_POS_CAP  if is_itm else self.OPT_POS_CAP
            soft_cap = self.ITM_SOFT_CAP if is_itm else self.OPT_SOFT_CAP
            p_size   = min(self.OPT_PASSIVE_SIZE, pos_cap // 2)

            market_mid = (bb + ba) / 2.0
            if market_mid - max(S - float(K), 0.0) < self.OPT_MIN_TV:
                continue

            live_iv = self._bs_iv_bisect(market_mid, S, float(K), tte)
            if not math.isnan(live_iv):
                mid_iv_by_K[K] = live_iv

            orders: List[Order] = []

            # Emergency cap exit
            if pos >= pos_cap:
                qty = min(self.OPT_CAP_SIZE, bbv, self._sell_room(pos, 0, pos_cap))
                if qty > 0 and bb > 0:
                    orders.append(Order(sym, bb, -qty))
                    logger.print(f"[CAP SELL] {sym} pos={pos} bb={bb}")
                if orders:
                    opt_orders[sym] = orders
                continue

            if pos <= -pos_cap:
                qty = min(self.OPT_CAP_SIZE, bav, self._buy_room(pos, 0, pos_cap))
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    logger.print(f"[CAP BUY ] {sym} pos={pos} ba={ba}")
                if orders:
                    opt_orders[sym] = orders
                continue

            # ── Model fair value (pre-calibrated parabolic smile) ────────────
            m         = math.log(S / float(K)) / math.sqrt(tte)
            sigma_ask = self._smile_iv(m, self.ASK_SMILE)
            sigma_bid = self._smile_iv(m, self.BID_SMILE)
            sigma_mid = (sigma_ask + sigma_bid) / 2.0
            fair_ask  = self._bs_call(S, float(K), tte, sigma_ask)
            fair_bid  = self._bs_call(S, float(K), tte, sigma_bid)
            fair_mid  = self._bs_call(S, float(K), tte, sigma_mid)

            # ── Standard model-fair taker (unchanged from v18) ────────────────
            if ba <= fair_bid - self.OPT_TAKER_EDGE:
                qty = min(self.OPT_TAKER_SIZE, bav, self._buy_room(pos, 0, pos_cap))
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    logger.print(f"[OPT TAKE BUY ] {sym} ba={ba} fair_bid={fair_bid:.2f}")
                if orders:
                    opt_orders[sym] = orders
                continue

            if bb >= fair_ask + self.OPT_TAKER_EDGE:
                qty = min(self.OPT_TAKER_SIZE, bbv, self._sell_room(pos, 0, pos_cap))
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    logger.print(f"[OPT TAKE SELL] {sym} bb={bb} fair_ask={fair_ask:.2f}")
                if orders:
                    opt_orders[sym] = orders
                continue

            # ── v19 ANCHOR TAKER: fire on bot mispricing vs current spot ──────
            # Bots priced at anchor; if anchor != S, market has stale price.
            # HIGH VFE: anchor < S → bots priced too low → lifting their ask is
            #           cheap (we pay anchor price, worth current-S fair).
            # LOW  VFE: anchor > S → bots priced too high → hitting their bid
            #           collects vol premium (sell at their anchor price).
            if anchor_mode == "HIGH":
                # Options cheap: take the ask if ba < fair_mid - ANCHOR_EDGE
                if ba <= fair_mid - self.ANCHOR_TAKER_EDGE:
                    qty = min(self.ANCHOR_TAKER_SIZE, bav, self._buy_room(pos, 0, pos_cap))
                    if qty > 0:
                        orders.append(Order(sym, ba, qty))
                        logger.print(
                            f"[ANCHOR BUY ] {sym} ba={ba} fair_mid={fair_mid:.2f}"
                            f" dS={delta_S:+.1f}"
                        )
                    if orders:
                        opt_orders[sym] = orders
                    continue

            elif anchor_mode == "LOW":
                # Options rich: hit the bid if bb > fair_mid + ANCHOR_EDGE
                if bb >= fair_mid + self.ANCHOR_TAKER_EDGE:
                    qty = min(self.ANCHOR_TAKER_SIZE, bbv, self._sell_room(pos, 0, pos_cap))
                    if qty > 0:
                        orders.append(Order(sym, bb, -qty))
                        logger.print(
                            f"[ANCHOR SELL] {sym} bb={bb} fair_mid={fair_mid:.2f}"
                            f" dS={delta_S:+.1f}"
                        )
                    if orders:
                        opt_orders[sym] = orders
                    continue

            # ── Passive quoting ───────────────────────────────────────────────
            can_bid = fair_bid >= float(bb)
            can_ask = fair_ask <= float(ba)

            bid_p = bb + 1
            ask_p = ba - 1

            bq = 0
            aq = 0

            if pos <= -soft_cap:
                if can_bid and bid_p < ba:
                    bq = min(p_size, self._buy_room(pos, 0, pos_cap))
                if can_ask and ask_p > bb:
                    aq = min(p_size // 2, self._sell_room(pos, 0, pos_cap))

            elif pos >= soft_cap:
                if can_ask and ask_p > bb:
                    aq = min(p_size, self._sell_room(pos, 0, pos_cap))
                if can_bid and bid_p < ba:
                    bq = min(p_size // 2, self._buy_room(pos, 0, pos_cap))

            else:
                if can_bid:
                    bq = min(p_size, self._buy_room(pos, 0, pos_cap))
                if can_ask:
                    aq = min(p_size, self._sell_room(pos, 0, pos_cap))

            # ── IV smile skew (v18 logic) ─────────────────────────────────────
            resid = vol_residuals.get(K, 0.0)
            if abs(resid) >= self.VOL_SKEW_THRESH:
                if resid > 0:
                    aq = min(pos_cap, int(aq * self.VOL_SKEW_BOOST))
                    bq = max(0, int(bq * self.VOL_SKEW_CUT))
                else:
                    bq = min(pos_cap, int(bq * self.VOL_SKEW_BOOST))
                    aq = max(0, int(aq * self.VOL_SKEW_CUT))

            # ── v19: Apply VFE anchor skew to final sizes ─────────────────────
            bq = min(pos_cap, int(bq * bid_mult))
            aq = min(pos_cap, int(aq * ask_mult))

            # Final room cap
            bq = min(bq, self._buy_room(pos, 0, pos_cap))
            aq = min(aq, self._sell_room(pos, 0, pos_cap))

            # Spread=2 safety: can't cross our own orders
            if spread == 2 and bq > 0 and aq > 0:
                if pos >= 0:
                    bq = 0
                else:
                    aq = 0

            if bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders
                resid_str = f"{resid:+.5f}" if resid != 0.0 else "n/a  "
                logger.print(
                    f"[OPT] {sym} pos={pos} spread={spread}"
                    f" bid={bid_p}×{bq}(gate={can_bid}) ask={ask_p}×{aq}(gate={can_ask})"
                    f" fair={fair_bid:.1f}/{fair_ask:.1f} resid={resid_str}"
                    f" bm={bid_mult} am={ask_mult}"
                )

        # Delta hedge — disabled (DEADBAND=999)
        total_delta = 0.0
        for K, iv in mid_iv_by_K.items():
            pos_k = state.position.get(self.VEV[K], 0)
            if pos_k == 0:
                continue
            total_delta += pos_k * self._bs_delta(S, float(K), tte, iv)

        target_vf  = max(-self.VF_LIMIT, min(self.VF_LIMIT, -round(total_delta)))
        current_vf = state.position.get(self.VF, 0)
        hedge      = target_vf - current_vf

        if abs(hedge) >= self.HEDGE_DEADBAND:
            if hedge > 0:
                qty = min(hedge, vf_bav, self._buy_room(current_vf, 0, self.VF_LIMIT))
                if qty > 0:
                    vf_orders.append(Order(self.VF, vf_ba, qty))
            else:
                qty = min(-hedge, vf_bbv, self._sell_room(current_vf, 0, self.VF_LIMIT))
                if qty > 0:
                    vf_orders.append(Order(self.VF, vf_bb, -qty))

        return opt_orders, vf_orders

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load(state.traderData)

        prev_ts = data["prev_ts"]
        day     = data["day"]

        if prev_ts < 0:
            inferred_tte = self._infer_tte_from_market(state)
            if inferred_tte is not None:
                inferred_day = round(self.TTE_START - inferred_tte)
                inferred_day = max(0, min(2, inferred_day))
                day = inferred_day
                logger.print(f"[INIT] inferred TTE={inferred_tte:.3f} → day={day}")
            else:
                logger.print(f"[INIT] could not infer TTE, defaulting day=0")
        elif state.timestamp < prev_ts - 500_000:
            day += 1
            logger.print(f"[DAY] new day={day}")

        data["day"]     = day
        data["prev_ts"] = state.timestamp

        tte = self._tte(day, state.timestamp)
        logger.print(f"[TICK] ts={state.timestamp} day={day} TTE={tte:.4f}")

        # ── v19: Update VFE EMA (bot anchor proxy) ───────────────────────────
        vf_depth = state.order_depths.get(self.VF)
        vf_bk_raw = self._book(vf_depth)
        if vf_bk_raw is not None:
            vf_mid_now = (vf_bk_raw[0] + vf_bk_raw[2]) / 2.0
            data["vf_ema"] = self._ewma(data["vf_ema"], vf_mid_now, self.VF_EMA_ALPHA)

        result: Dict[Symbol, List[Order]] = {}

        # 1. HG: dual-EMA + vol signal + EOD close
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. VF: passive MM + EOD close
        vf_mm_ords = self._trade_vf(state)

        # 3. Floor-price passive shorts (VEV_6000 / VEV_6500)
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 4. Options MM with VFE anchor skew + disabled delta hedge
        opt_ords, vf_hedge = self._trade_options(state, tte, data.get("vf_ema"))
        for sym, ords in opt_ords.items():
            result.setdefault(sym, []).extend(ords)

        vf_all = vf_mm_ords + vf_hedge
        if vf_all:
            result[self.VF] = vf_all

        for sym in self.VEV.values():
            if sym not in result:
                result[sym] = []

        trader_data = json.dumps(data, separators=(",", ":"))
        conversions = 0
        logger.flush(state=state, orders=result, conversions=conversions, trader_data=trader_data)
        return result, conversions, trader_data
