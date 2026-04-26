"""
v12_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes vs v11_itm
──────────────────────────────────────────────────────────────────────────────
1. REMOVE VEV_5500  — Discord: "getting fried 5500". Our data: 0 own fills.

2. VF PASSIVE MM  — VELVETFRUIT_EXTRACT had 52 market_trades/day in v11 logs
   that we never intercepted. Add inside-market MM at bb+1/ba-1 (like options).
   Position cap ±100 (separate from the now-disabled delta hedge).

3. EOD HG CLOSE  — v11 ended with pos=-23 short; price recovered 33 ticks
   → lost ~747 in MTM. At ts > 950000 (last 5%), use taker to close any
   open HG position. Saves that unrealised gain at day close.

4. DISABLE DELTA HEDGE  — HEDGE_DEADBAND raised to 999.
   Discord insight: "remove BS gets better results". The hedge fired 0 times
   in v11 (DEADBAND=20 never triggered). The only BS usage left was the hedge.
   Removing it lets VF be used purely for passive MM.
   VF position cap set to ±100 to manage directional risk independently.

5. VOLATILITY SIGNAL FOR HG  — Track EWMA of |mid price change| per tick.
   When recent vol > HG_VOL_THRESH, reduce quote size to avoid getting run
   over in fast-moving markets (aligned with "volatility is a signal" Discord).

HG parameters unchanged: dual-EMA trend filter, maker-only guard, SOFT_LIMIT=60.
Options parameters unchanged: inside-market bb+1/ba-1 for queue priority.
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

    # VEV_5500 removed: Discord "getting fried 5500" + 0 own fills in our logs
    ACTIVE_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400]
    FLOOR_STRIKES  = [6000, 6500]
    DEEP_ITM_STRIKES: list = []

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT  = 200
    VF_LIMIT  = 200
    VEV_LIMIT = 300

    # Near-ATM caps
    OPT_POS_CAP   = 150
    OPT_SOFT_CAP  = 60
    # Deep ITM caps (delta ~0.95-1.0)
    ITM_POS_CAP   = 25
    ITM_SOFT_CAP  = 12
    FLOOR_POS_CAP = 200

    # ── Pre-calibrated vol smile — TTE inference only ────────────────────────
    ASK_SMILE: Tuple[float, float, float] = (2.2480,  0.1860, 0.0136)
    BID_SMILE: Tuple[float, float, float] = (1.0000, -0.0469, 0.0130)

    # ── TTE ───────────────────────────────────────────────────────────────────
    TTE_START      = 7.0
    MAX_TS_PER_DAY = 999_900

    # ── Option MM parameters ──────────────────────────────────────────────────
    OPT_CAP_SIZE     = 15
    OPT_PASSIVE_SIZE = 15
    OPT_MIN_TTE      = 0.3
    OPT_MIN_TV       = 0.5

    FLOOR_SHORT_SIZE = 20

    # ── Delta hedge — disabled (DEADBAND=999 never triggers) ─────────────────
    # "remove BS gets better results" — Discord. VF position now managed by
    # its own passive MM (see _trade_vf). Delta hedge never fired in v11 anyway.
    HEDGE_DEADBAND = 999

    # ── VF passive MM parameters ──────────────────────────────────────────────
    # VF spread is ~5 ticks; 52 market_trades/day we missed in v11 logs.
    VF_QUOTE_SIZE   = 15
    VF_MM_POS_CAP   = 100   # hard cap: taker exit
    VF_MM_SOFT_CAP  = 50    # soft cap: quote only the closing side

    # ── HG MM parameters ──────────────────────────────────────────────────────
    HG_EMA_ALPHA   = 0.050
    HG_TREND_ALPHA = 0.003
    HG_TREND_GAP   = 8
    HG_TAKE_EDGE   = 20
    HG_TAKER_SIZE  = 8
    HG_QUOTE_TICK  = 2
    HG_SKEW_TICKS  = 5.0
    HG_QUOTE_SIZE  = 15
    HG_SOFT_LIMIT  = 60
    HG_HARD_LIMIT  = 190
    HG_UNWIND_SIZE = 30

    # Volatility signal: reduce quote size when HG is moving fast
    HG_VOL_ALPHA  = 0.10    # EWMA alpha for |mid change| per tick
    HG_VOL_THRESH = 2.5     # ticks — above this, use reduced quote size
    HG_VOL_SIZE   = 8       # reduced quote size in high-vol regime

    # EOD close threshold — last 5% of day (ts ∈ [0, 999900])
    HG_EOD_TS = 950_000

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
    def _bs_vega(cls, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 1e-8 or sigma <= 1e-8 or S <= 0.0:
            return 0.0
        sq = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
        return S * sq * cls._npdf(d1)

    @classmethod
    def _bs_iv(cls, C: float, S: float, K: float, T: float) -> float:
        intrinsic = max(S - K, 0.0)
        if C <= intrinsic + 1e-6 or T <= 1e-8 or S <= 0.0:
            return float("nan")
        sigma = math.sqrt(2.0 * math.pi / T) * C / S
        sigma = max(0.001, min(sigma, 5.0))
        for _ in range(60):
            price = cls._bs_call(S, K, T, sigma)
            vega  = cls._bs_vega(S, K, T, sigma)
            if abs(vega) < 1e-10:
                break
            step = (price - C) / vega
            step = max(-0.5 * sigma, min(0.5 * sigma, step))
            sigma -= step
            sigma = max(1e-6, min(sigma, 10.0))
        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # Smile helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smile_iv(m: float, params: Tuple[float, float, float]) -> float:
        a, b, c = params
        return max(0.001, a * m * m + b * m + c)

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
        mid = len(estimates) // 2
        return estimates[mid]

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self, raw: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "hg_ema": None, "hg_trend_ema": None, "hg_prev_mid": None,
            "hg_vol": None,     # NEW: EWMA of |mid change| for vol signal
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

        # ── Volatility signal: EWMA of |mid change| per tick ─────────────────
        prev_mid = data["hg_prev_mid"]
        if prev_mid is not None:
            price_chg = abs(wmid - prev_mid)
            hg_vol = self._ewma(data["hg_vol"], price_chg, self.HG_VOL_ALPHA)
        else:
            hg_vol = data["hg_vol"] if data["hg_vol"] is not None else 0.0
        data["hg_vol"] = hg_vol

        # Quote size: reduce in high-vol regime to avoid getting run over
        high_vol = hg_vol > self.HG_VOL_THRESH
        q_size = self.HG_VOL_SIZE if high_vol else self.HG_QUOTE_SIZE

        # ── Dual EMA: fast (fair value) + slow (trend direction) ─────────────
        ema       = self._ewma(data["hg_ema"],       wmid, self.HG_EMA_ALPHA)
        trend_ema = self._ewma(data["hg_trend_ema"], wmid, self.HG_TREND_ALPHA)
        data["hg_ema"]       = ema
        data["hg_trend_ema"] = trend_ema
        data["hg_prev_mid"]  = wmid

        fast_vs_trend = ema - trend_ema
        in_downtrend  = fast_vs_trend < -self.HG_TREND_GAP
        in_uptrend    = fast_vs_trend >  self.HG_TREND_GAP

        # ── EOD close: last 5% of day → taker-close open position ────────────
        # Avoids carrying an open short/long overnight when price mean-reverts.
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
            return orders, data  # skip normal MM during EOD close

        # ── Emergency hard-limit unwind ───────────────────────────────────────
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

        # ── Taker (safety valve — large mean-reversion spikes only) ──────────
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

        # ── Passive MM ────────────────────────────────────────────────────────
        skew   = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew
        bid_p  = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p  = math.ceil(fair_q  + self.HG_QUOTE_TICK)

        # Maker-only guard: never cross the spread
        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)

        base_bq = 0 if pos >= self.HG_SOFT_LIMIT else min(
            q_size, self._buy_room(pos, 0, self.HG_LIMIT))
        base_aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(
            q_size, self._sell_room(pos, 0, self.HG_LIMIT))

        # Two-EMA trend filter: stop quoting the adverse side in a trend
        if in_downtrend:
            bq = 0
            aq = base_aq
        elif in_uptrend:
            bq = base_bq
            aq = 0
        else:
            bq = base_bq
            aq = base_aq

        # Inventory unwind overrides trend filter when beyond soft limit
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
        vol_str   = "HI" if high_vol else "lo"
        logger.print(
            f"[HG] pos={pos} ema={ema:.1f} trend={trend_ema:.1f}({trend_str})"
            f" vol={hg_vol:.2f}({vol_str}) sz={q_size}"
            f" bid={bid_p}×{bq} ask={ask_p}×{aq}"
        )
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # 2. VF PASSIVE MM  (NEW in v12)
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vf(self, state: TradingState) -> List[Order]:
        """
        Passive market-making on VELVETFRUIT_EXTRACT.

        In v11 logs: 52 VF market_trades/day at bid/ask; we had 0 own fills
        because we only used VF for (disabled) delta hedging.

        Strategy: quote inside spread at bb+1 / ba-1 for queue priority.
        Same pattern as options MM. Position cap ±100 (leaves ±100 for hedge if
        re-enabled in a future version).
        """
        depth = state.order_depths.get(self.VF)
        bk = self._book(depth)
        if bk is None:
            return []

        bb, bbv, ba, bav = bk
        pos = state.position.get(self.VF, 0)
        spread = ba - bb

        # Only MM when spread ≥ 3 to earn at least 1 tick on each side
        if spread < 3:
            return []

        bid_p = bb + 1
        ask_p = ba - 1

        orders: List[Order] = []

        if pos >= self.VF_MM_SOFT_CAP:
            # Too long: only quote ask side to close
            aq = min(self.VF_QUOTE_SIZE, self._sell_room(pos, 0, self.VF_MM_POS_CAP))
            if aq > 0:
                orders.append(Order(self.VF, ask_p, -aq))

        elif pos <= -self.VF_MM_SOFT_CAP:
            # Too short: only quote bid side to close
            bq = min(self.VF_QUOTE_SIZE, self._buy_room(pos, 0, self.VF_MM_POS_CAP))
            if bq > 0:
                orders.append(Order(self.VF, bid_p, bq))

        else:
            # Neutral: two-sided inside-market quotes
            bq = min(self.VF_QUOTE_SIZE, self._buy_room(pos, 0, self.VF_MM_POS_CAP))
            aq = min(self.VF_QUOTE_SIZE, self._sell_room(pos, 0, self.VF_MM_POS_CAP))
            if bq > 0:
                orders.append(Order(self.VF, bid_p, bq))
            if aq > 0:
                orders.append(Order(self.VF, ask_p, -aq))

        if orders:
            logger.print(
                f"[VF] pos={pos} spread={spread}"
                f" bid={bid_p} ask={ask_p}"
            )
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
    # 4. OPTIONS MM + (disabled) delta hedge
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        Passive MM for ACTIVE_STRIKES. Delta hedge disabled (HEDGE_DEADBAND=999).
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

        mid_iv_by_K: Dict[int, float] = {}

        for K in self.ACTIVE_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            bk    = self._book(depth)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            pos = state.position.get(sym, 0)

            is_itm   = K <= 4500
            pos_cap  = self.ITM_POS_CAP  if is_itm else self.OPT_POS_CAP
            soft_cap = self.ITM_SOFT_CAP if is_itm else self.OPT_SOFT_CAP
            p_size   = min(self.OPT_PASSIVE_SIZE, pos_cap // 2)

            market_mid = (bb + ba) / 2.0
            if market_mid - max(S - K, 0.0) < self.OPT_MIN_TV:
                continue

            live_iv = self._bs_iv(market_mid, S, float(K), tte)
            if not math.isnan(live_iv):
                mid_iv_by_K[K] = live_iv

            orders: List[Order] = []

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

            bq = 0
            aq = 0
            bid_p = bb
            ask_p = ba

            if pos <= -soft_cap:
                if bb + 1 < ba:
                    bq    = min(p_size, self._buy_room(pos, 0, pos_cap))
                    bid_p = bb + 1
                aq = min(p_size // 2, self._sell_room(pos, 0, pos_cap))

            elif pos >= soft_cap:
                if ba - 1 > bb:
                    aq    = min(p_size, self._sell_room(pos, 0, pos_cap))
                    ask_p = ba - 1
                bq = min(p_size // 2, self._buy_room(pos, 0, pos_cap))

            else:
                if ba - bb >= 3:
                    bid_p = bb + 1
                    ask_p = ba - 1
                bq = min(p_size, self._buy_room(pos, 0, pos_cap))
                aq = min(p_size, self._sell_room(pos, 0, pos_cap))

            if bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders
                iv_str = f"{live_iv:.5f}" if not math.isnan(live_iv) else "nan"
                logger.print(
                    f"[OPT] {sym} pos={pos} spread={ba-bb}"
                    f" bid={bid_p}×{bq} ask={ask_p}×{aq} iv={iv_str}"
                )

        # Delta hedge — effectively disabled (DEADBAND=999)
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
                    logger.print(f"[HEDGE BUY ] delta={total_delta:.2f} tgt={target_vf} qty={qty}")
            else:
                qty = min(-hedge, vf_bbv, self._sell_room(current_vf, 0, self.VF_LIMIT))
                if qty > 0:
                    vf_orders.append(Order(self.VF, vf_bb, -qty))
                    logger.print(f"[HEDGE SELL] delta={total_delta:.2f} tgt={target_vf} qty={qty}")

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

        # 1. HG: dual-EMA MM + EOD close + vol signal
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. VF: passive inside-market MM (NEW in v12)
        vf_mm_ords = self._trade_vf(state)

        # 3. Floor-price passive shorts
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 4. Options MM + (disabled) delta hedge
        opt_ords, vf_hedge = self._trade_options(state, tte)
        for sym, ords in opt_ords.items():
            result.setdefault(sym, []).extend(ords)

        # Merge VF orders: passive MM + delta hedge (hedge disabled, so just MM)
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
