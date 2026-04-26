"""
v16_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes vs v14 (HG only — all other logic identical)
──────────────────────────────────────────────────────────────────────────────
EDA showed 4 structural problems in v14 HG:

1. TREND FILTER THRASHING  — gap (fast_EMA - trend_EMA) oscillates ±40 ticks.
   With HG_TREND_GAP=8, the filter fires ~62% of ticks (31% UP + 31% DOWN),
   meaning we quote only one side 62% of the time — not due to real trends,
   just noise. The filter switches every few ticks. Fix: raise HG_TREND_GAP
   8→25. At gap=25 the filter fires ~10% each side (80% flat = both sides
   quoted). More fills, same directional protection for large moves.

2. TAKE_EDGE DEAD CODE  — EMA tracking error std=5.6 ticks; 0.0% of ticks
   ever reach TAKE_EDGE=20. The taker never fires. Fix: lower to 12. At std
   5.6 this triggers ~3-4% of ticks (≈1.8σ deviations from EMA = genuine
   mean-reversion opportunities at ±12 tick extremes).

3. POSITION CAPS TOO LOOSE  — HARD_LIMIT=190 allows buildup of 190-unit
   adverse position. v14 PnL peaked at +1530 then dropped ~1000 from holding
   short=-23 through a 40+ tick recovery. Fix: lower HARD_LIMIT 190→120,
   SOFT_LIMIT 60→50. Reduces maximum MTM exposure from adverse positions.

4. IMBALANCE SIGNAL UNUSED  — EDA confirms order imbalance predicts 50-tick
   HG returns: slope 3–9 bps/unit (p=2.1e-12). Currently v14 ignores it.
   Fix: shift fair_q by imb * HG_IMB_SCALE (=2 ticks) each tick. When more
   bid volume → lean bid up slightly (expect up move), reducing adverse fills.

HG_SKEW_TICKS 5.0→7.0: slightly more aggressive position-reducing skew.
All VF/options logic unchanged from v14.
Delta hedge: still disabled (DEADBAND=999).
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

    # Strikes included in live IV smile fit (near-ATM only; deep ITM excluded
    # because bid < intrinsic distorts the BS fit).
    SMILE_STRIKES = [5000, 5100, 5200, 5300, 5400]

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT  = 200
    VF_LIMIT  = 200
    VEV_LIMIT = 300

    OPT_POS_CAP   = 150
    OPT_SOFT_CAP  = 60
    # K=4500 raised: 16-tick spread, highest edge/fill in options
    ITM_POS_CAP   = 50
    ITM_SOFT_CAP  = 25
    FLOOR_POS_CAP = 200

    # ── Pre-calibrated vol smile — TTE inference only ────────────────────────
    ASK_SMILE: Tuple[float, float, float] = (2.2480,  0.1860, 0.0136)
    BID_SMILE: Tuple[float, float, float] = (1.0000, -0.0469, 0.0130)

    # ── TTE ───────────────────────────────────────────────────────────────────
    TTE_START      = 7.0
    MAX_TS_PER_DAY = 999_900

    # ── Option MM parameters ──────────────────────────────────────────────────
    OPT_CAP_SIZE     = 15
    OPT_PASSIVE_SIZE = 25   # up from 15 — more capture per bot event
    OPT_MIN_TTE      = 0.3
    OPT_MIN_TV       = 0.5

    FLOOR_SHORT_SIZE = 20

    # IV vol-skew sizing (Orin: "find the odd ones out")
    # When a strike's IV is above the smile → sell expensive vol (boost ask)
    # When below → buy cheap vol (boost bid)
    VOL_SKEW_THRESH  = 0.0008   # min |residual| to apply skew
    VOL_SKEW_BOOST   = 1.50     # multiply favored side
    VOL_SKEW_CUT     = 0.75     # multiply adverse side

    # ── Delta hedge — disabled ────────────────────────────────────────────────
    HEDGE_DEADBAND = 999

    # ── VF passive MM parameters ──────────────────────────────────────────────
    VF_QUOTE_SIZE   = 25    # up from 15
    VF_MM_POS_CAP   = 150   # up from 100
    VF_MM_SOFT_CAP  = 75    # up from 50
    VF_EOD_TS       = 950_000

    # ── HG MM parameters ──────────────────────────────────────────────────────
    HG_EMA_ALPHA   = 0.050
    HG_TREND_ALPHA = 0.003
    HG_TREND_GAP   = 25    # v16: 8→25 — stops filter thrashing (62%→~20% trend ticks)
    HG_TAKE_EDGE   = 12    # v16: 20→12 — activates taker (was 0% trigger; now ~3-4%)
    HG_TAKER_SIZE  = 8
    HG_QUOTE_TICK  = 2
    HG_SKEW_TICKS  = 7.0   # v16: 5→7 — more aggressive position-lean
    HG_QUOTE_SIZE  = 15
    HG_SOFT_LIMIT  = 50    # v16: 60→50 — tighter one-sided switch
    HG_HARD_LIMIT  = 120   # v16: 190→120 — cap adverse position earlier
    HG_UNWIND_SIZE = 30
    HG_VOL_ALPHA   = 0.10
    HG_VOL_THRESH  = 2.5
    HG_VOL_SIZE    = 8
    HG_EOD_TS      = 950_000
    HG_IMB_SCALE   = 2.0   # v16: NEW — shift fair_q by imbalance * 2 ticks

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes (bisection IV — stable for all moneyness levels)
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
        """
        Implied vol via bisection — O(60) evaluations but globally stable.
        Avoids NR divergence for far OTM/ITM options. Returns nan if infeasible.
        """
        intrinsic = max(S - K, 0.0)
        if C <= intrinsic + 1e-6 or T <= 1e-8 or S <= 0.0:
            return float("nan")
        # Bracket: lo=1e-6 (C→intrinsic), hi=5 (C very large)
        lo, hi = 1e-6, 5.0
        # Check if hi is an actual upper bound
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
    # Smile helpers (TTE inference)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smile_iv(m: float, params: Tuple[float, float, float]) -> float:
        a, b, c = params
        return max(0.001, a * m * m + b * m + c)

    # ─────────────────────────────────────────────────────────────────────────
    # Live IV smile fit
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_smile(
        self, state: TradingState, S: float, tte: float
    ) -> Dict[int, float]:
        """
        Compute live IV for SMILE_STRIKES, fit a linear smile IV = a*m + b,
        and return {K: residual} where residual > 0 means expensive vol.

        Uses mid-price IV for fitting stability. Returns empty dict if fewer
        than 3 strikes have valid IV.
        """
        if tte < self.OPT_MIN_TTE:
            return {}

        points: List[Tuple[float, float, int]] = []  # (m, iv_mid, K)
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
            # Skip if mid ≤ intrinsic (no time value; IV undefined)
            if mid <= max(S - float(K), 0.0) + 0.5:
                continue
            iv = self._bs_iv_bisect(mid, S, float(K), tte)
            if math.isnan(iv):
                continue
            m = math.log(S / float(K)) / math.sqrt(tte)
            points.append((m, iv, K))

        if len(points) < 3:
            return {}

        # Linear regression: IV = slope * m + intercept
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
            residuals[K] = iv - fitted  # >0: expensive, <0: cheap
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

        # EOD: taker-close open position
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

        # Imbalance skew: positive imb (more bid vol) → lean up (expect up move)
        imb    = (bbv - bav) / (bbv + bav) if (bbv + bav) > 0 else 0.0
        skew   = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew + imb * self.HG_IMB_SCALE
        bid_p  = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p  = math.ceil(fair_q  + self.HG_QUOTE_TICK)

        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)

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
            ask_p = min(ask_p, bb + 1)
        if pos <= -self.HG_SOFT_LIMIT:
            bq = min(q_size, self._buy_room(pos, 0, self.HG_LIMIT))
            bid_p = max(bid_p, ba - 1)

        if bq > 0:
            orders.append(Order(self.HG, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.HG, ask_p, -aq))

        trend_str = "DOWN" if in_downtrend else ("UP" if in_uptrend else "flat")
        logger.print(
            f"[HG] pos={pos} ema={ema:.1f} trend={trend_ema:.1f}({trend_str})"
            f" imb={imb:+.2f} vol={hg_vol:.2f} bid={bid_p}×{bq} ask={ask_p}×{aq}"
        )
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VF PASSIVE MM
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vf(self, state: TradingState) -> List[Order]:
        """
        Passive MM on VELVETFRUIT_EXTRACT.
        - Two-sided inside-market when spread ≥ 3 (bid@bb+1, ask@ba-1)
        - EOD close at ts > VF_EOD_TS to avoid overnight directional exposure
        """
        depth = state.order_depths.get(self.VF)
        bk = self._book(depth)
        if bk is None:
            return []

        bb, bbv, ba, bav = bk
        pos = state.position.get(self.VF, 0)
        spread = ba - bb
        orders: List[Order] = []

        # EOD: close VF position with taker
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
    # 4. OPTIONS MM + IV smile skew
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        Passive MM for ACTIVE_STRIKES with live IV smile-based size skew.

        Spread handling:
          spread ≥ 3: bid@bb+1, ask@ba-1     (fully inside, queue priority)
          spread ≤ 2: skip — no reliable edge.
            spread=2: bb+1==ba-1, can't post both inside; one-sided fills
            create directional accumulation with no unwind path (confirmed
            in v13: VEV_5300 stuck long 6 units at avg=51, MTM loss -5.2).
            VEV_5400 alternates 58% spread=1 / 42% spread=2 — inaccessible.

        IV smile (SMILE_STRIKES only): fit live IV vs moneyness each tick.
          Expensive vol (IV > fit): cut bid size, boost ask size
          Cheap vol    (IV < fit): boost bid size, cut ask size
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

        # Live IV smile for vol-skewed sizing
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

            # Skip tight spreads — need ≥ 3 ticks for inside-market quoting
            if spread < 3:
                continue

            is_itm   = K <= 4500
            pos_cap  = self.ITM_POS_CAP  if is_itm else self.OPT_POS_CAP
            soft_cap = self.ITM_SOFT_CAP if is_itm else self.OPT_SOFT_CAP
            p_size   = min(self.OPT_PASSIVE_SIZE, pos_cap // 2)

            # Skip near-intrinsic (no time value)
            market_mid = (bb + ba) / 2.0
            if market_mid - max(S - float(K), 0.0) < self.OPT_MIN_TV:
                continue

            # Store live IV for delta hedge (disabled but kept for future use)
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

            # ── Quote prices: inside spread for queue priority ────────────────
            # spread >= 3 guaranteed here (checked above)
            bid_p = bb + 1
            ask_p = ba - 1

            # ── Base sizes from position ──────────────────────────────────────
            bq = 0
            aq = 0

            if pos <= -soft_cap:
                if bid_p < ba:
                    bq = min(p_size, self._buy_room(pos, 0, pos_cap))
                aq = min(p_size // 2, self._sell_room(pos, 0, pos_cap))

            elif pos >= soft_cap:
                if ask_p > bb:
                    aq = min(p_size, self._sell_room(pos, 0, pos_cap))
                bq = min(p_size // 2, self._buy_room(pos, 0, pos_cap))

            else:
                bq = min(p_size, self._buy_room(pos, 0, pos_cap))
                aq = min(p_size, self._sell_room(pos, 0, pos_cap))

            # ── IV smile skew: Orin's "find the odd ones out" ─────────────────
            # vol_residuals only covers SMILE_STRIKES (5000-5400).
            # Deep ITM (4000/4500) not in smile fit → no skew applied.
            resid = vol_residuals.get(K, 0.0)
            if abs(resid) >= self.VOL_SKEW_THRESH:
                if resid > 0:
                    # Expensive vol: prefer selling → boost ask, cut bid
                    aq = min(pos_cap, int(aq * self.VOL_SKEW_BOOST))
                    bq = max(0, int(bq * self.VOL_SKEW_CUT))
                else:
                    # Cheap vol: prefer buying → boost bid, cut ask
                    bq = min(pos_cap, int(bq * self.VOL_SKEW_BOOST))
                    aq = max(0, int(aq * self.VOL_SKEW_CUT))

            # Final room cap
            bq = min(bq, self._buy_room(pos, 0, pos_cap))
            aq = min(aq, self._sell_room(pos, 0, pos_cap))

            if bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders
                resid_str = f"{resid:+.5f}" if resid != 0.0 else "n/a  "
                logger.print(
                    f"[OPT] {sym} pos={pos} spread={spread}"
                    f" bid={bid_p}×{bq} ask={ask_p}×{aq} resid={resid_str}"
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

        result: Dict[Symbol, List[Order]] = {}

        # 1. HG: dual-EMA + vol signal + EOD close
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. VF: passive MM + EOD close
        vf_mm_ords = self._trade_vf(state)

        # 3. Floor-price passive shorts
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 4. Options MM (IV smile skew) + disabled delta hedge
        opt_ords, vf_hedge = self._trade_options(state, tte)
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
