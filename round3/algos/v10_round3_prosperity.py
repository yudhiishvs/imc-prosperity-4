"""
v10_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes vs v9
─────────────
HG REWRITE — slow-EMA mean-reversion MM + maker-only guard + stress sizing

Root cause of v9 HG loss (-7011):
  No maker-only guard.  When EMA > market (downtrend), bid_p = EMA-2 > ba
  → the exchange treats it as a market buy (taker) → bought aggressively into
  falling market all day.  avg_buy=9977, avg_sell=9965 → -12.80/unit adverse.

Fixes:
1. HG_EMA_ALPHA = 0.005  (slow mean-reversion EMA, same as +486 reference algo)
   With slow EMA, fair value ≈ long-run mean.  Market deviations → mean-revert quotes.

2. Maker-only guard:
     bid_p = min(bid_p, ba - 1)   # never cross market ask
     ask_p = max(ask_p, bb + 1)   # never cross market bid
   Prevents inadvertent market-order behavior.

3. Stress-based size reduction:
   When |mid - EMA| > HG_STRESS_GAP (20 ticks), reduce quote sizes to 4.
   Limits exposure when market has trended far from fair value.

4. HG_SKEW_TICKS = 6.0, HG_SOFT_LIMIT = 80  (earlier/stronger inventory skew)

5. Taker edge raised to 15 (only fires on genuine large spikes vs slow EMA)

Options (unchanged from v9)
────────────────────────────
• Neutral MM at bb+1/ba-1 for queue priority (spread ≥ 3 required).
• VEV_5000/5100/5200 getting fills (+41 PnL in v9 test).
• TTE_START = 7.0, OPT_POS_CAP = 150, OPT_SOFT_CAP = 60.

Position limits (official):
  HYDROGEL_PACK: 200   VELVETFRUIT_EXTRACT: 200   VEV_*: 300 each
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

    # Near-ATM strikes — active bid/ask smile MM
    ACTIVE_STRIKES   = [5000, 5100, 5200, 5300, 5400, 5500]
    # Deep OTM — passive short at floor price 1
    FLOOR_STRIKES    = [6000, 6500]
    # Deep ITM — not actively MM'd but included in delta hedge if holding
    DEEP_ITM_STRIKES = [4000, 4500]

    # ── Position limits ────────────────────────────────────────────────────────
    HG_LIMIT    = 200
    VF_LIMIT    = 200
    VEV_LIMIT   = 300

    # Per-strike cap.  6 strikes × 150 × avg_delta≈0.45 ≈ 405 — clipped to VF_LIMIT=200 in hedge.
    OPT_POS_CAP   = 150   # hard cap: emergency market exit
    OPT_SOFT_CAP  = 60    # soft cap: start skewing quotes to reduce inventory
    FLOOR_POS_CAP = 200

    # ── Pre-calibrated vol smile — kept for TTE inference only ────────────────
    # (NOT used for quoting prices any more — we use live market prices instead)
    ASK_SMILE: Tuple[float, float, float] = (2.2480,  0.1860, 0.0136)
    BID_SMILE: Tuple[float, float, float] = (1.0000, -0.0469, 0.0130)

    # ── TTE ───────────────────────────────────────────────────────────────────
    TTE_START      = 7.0       # Solvenarian days at round-3 day-0 ts-0
    MAX_TS_PER_DAY = 999_900

    # ── Option MM parameters ──────────────────────────────────────────────────
    OPT_CAP_SIZE     = 15     # units per emergency cap-exit (market taker)
    OPT_PASSIVE_SIZE = 15     # units per passive resting quote
    OPT_MIN_TTE      = 0.3    # stop trading very near expiry
    OPT_MIN_TV       = 0.5    # skip if time-value < this (near intrinsic, no MM edge)

    FLOOR_SHORT_SIZE = 20     # passive sell size per tick for floor-price strikes

    # ── Delta hedge ───────────────────────────────────────────────────────────
    HEDGE_DEADBAND = 20       # only hedge when |imbalance| ≥ this (avoid paying spread every tick)

    # ── HG MM parameters ──────────────────────────────────────────────────────
    HG_EMA_ALPHA   = 0.005   # slow mean-reversion EMA (same as +486 reference algo)
    HG_TAKE_EDGE   = 15      # taker fires only on large spikes vs slow EMA
    HG_TAKER_SIZE  = 8
    HG_QUOTE_TICK  = 2       # ticks from fair (clamped by maker guard to inside mkt)
    HG_SKEW_TICKS  = 6.0     # inventory skew in ticks per unit (stronger than v9)
    HG_QUOTE_SIZE  = 20
    HG_SOFT_LIMIT  = 80      # start skewing earlier (was 140)
    HG_HARD_LIMIT  = 190
    HG_UNWIND_SIZE = 30
    HG_STRESS_GAP  = 20      # |mid - EMA| threshold for stress-reduced sizing

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes in Solvenarian days
    # sigma in per-√Solvenarian-day units (e.g. 0.013 ATM)
    # T in Solvenarian days
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
    def _bs_call(cls, S: float, K: float, T: float, sigma: float) -> float:
        """European call. T in Solvenarian days, sigma per √Solvenarian-day."""
        if T <= 1e-8 or sigma <= 1e-8 or S <= 0.0:
            return max(S - K, 0.0)
        sq = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
        d2 = d1 - sigma * sq
        return S * cls._ncdf(d1) - K * cls._ncdf(d2)

    @classmethod
    def _bs_delta(cls, S: float, K: float, T: float, sigma: float) -> float:
        """BS call delta = N(d1). T in Solvenarian days."""
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
        """
        Implied vol via Newton-Raphson with Brenner-Subrahmanyam initialisation.
        T in Solvenarian days; returns sigma per √Solvenarian-day, or nan.
        """
        intrinsic = max(S - K, 0.0)
        if C <= intrinsic + 1e-6 or T <= 1e-8 or S <= 0.0:
            return float("nan")
        # Brenner-Subrahmanyam: sigma_0 ≈ sqrt(2π/T) * C/S
        sigma = math.sqrt(2.0 * math.pi / T) * C / S
        sigma = max(0.001, min(sigma, 5.0))
        for _ in range(60):
            price = cls._bs_call(S, K, T, sigma)
            vega  = cls._bs_vega(S, K, T, sigma)
            if abs(vega) < 1e-10:
                break
            step = (price - C) / vega
            step = max(-0.5 * sigma, min(0.5 * sigma, step))  # damped step
            sigma -= step
            sigma = max(1e-6, min(sigma, 10.0))
        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # Smile helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smile_iv(m: float, params: Tuple[float, float, float]) -> float:
        """Evaluate pre-calibrated parabola: IV = a·m² + b·m + c, floored at 0.001."""
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
        """Returns (best_bid, bid_vol, best_ask, ask_vol) or None."""
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
        """
        Back-solve TTE from observed option mid-prices using mid-smile BS.

        Uses bisection on TTE ∈ [0.1, TTE_START].  Returns median estimate
        across active strikes, or None if fewer than 2 strikes are usable.
        """
        vf_bk = self._book(state.order_depths.get(self.VF))
        if vf_bk is None:
            return None
        S = (vf_bk[0] + vf_bk[2]) / 2.0

        # Mid-smile coefficients (average of ask and bid)
        ma = (self.ASK_SMILE[0] + self.BID_SMILE[0]) / 2.0
        mb = (self.ASK_SMILE[1] + self.BID_SMILE[1]) / 2.0
        mc = (self.ASK_SMILE[2] + self.BID_SMILE[2]) / 2.0

        estimates: List[float] = []
        for K in self.ACTIVE_STRIKES:
            sym = self.VEV[K]
            bk = self._book(state.order_depths.get(sym))
            if bk is None:
                continue
            C_obs = (bk[0] + bk[2]) / 2.0  # market mid price
            intrinsic = max(S - K, 0.0)
            if C_obs <= intrinsic + 0.5:
                continue  # no time value, skip

            # Bisect TTE ∈ [0.05, TTE_START]
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
    # 1. HYDROGEL MM — taker-first (v7 logic unchanged)
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

        # Taker: hit when price significantly off EMA
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

        # Passive MM
        skew   = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew
        bid_p  = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p  = math.ceil(fair_q + self.HG_QUOTE_TICK)

        # Maker-only guard — CRITICAL: prevent inadvertent market-order behavior.
        # Without this, when EMA > market (downtrend), bid_p > ba → exchange
        # treats it as a taker buy (fills at ask) → buys aggressively into decline.
        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)

        # Stress-based size reduction: when market is far from EMA (trending),
        # the EMA-based fair value is unreliable — shrink exposure.
        gap = abs(wmid - ema)
        if gap > self.HG_STRESS_GAP:
            cap_size = 4
        else:
            cap_size = self.HG_QUOTE_SIZE

        bq = 0 if pos >= self.HG_SOFT_LIMIT  else min(cap_size,
                                                        self._buy_room(pos, 0, self.HG_LIMIT))
        aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(cap_size,
                                                        self._sell_room(pos, 0, self.HG_LIMIT))
        # Inventory unwind when beyond soft limit
        if pos >= self.HG_SOFT_LIMIT:
            aq = min(self.HG_QUOTE_SIZE, self._sell_room(pos, 0, self.HG_LIMIT))
            ask_p = min(ask_p, bb + 1)
        if pos <= -self.HG_SOFT_LIMIT:
            bq = min(self.HG_QUOTE_SIZE, self._buy_room(pos, 0, self.HG_LIMIT))
            bid_p = max(bid_p, ba - 1)

        if bq > 0:
            orders.append(Order(self.HG, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.HG, ask_p, -aq))

        logger.print(f"[HG] pos={pos} ema={ema:.1f} gap={gap:.1f} bid={bid_p}×{bq} ask={ask_p}×{aq}")
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
    # 3. OPTIONS MM — pre-calibrated bid/ask smile + delta hedge
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        Passive market-making for each ACTIVE_STRIKE.

        Strategy
        --------
        Post resting limit orders at the market bid/ask (not at stale fair values).
        Let market bots cross our quotes — we earn the spread.
        Pre-calibrated smile is NOT used for pricing; it was miscalibrated for
        current vol level (market ATM IV ~0.015 vs calibrated 0.013).

        Quote logic (position-skewed):
          |pos| ≤ SOFT_CAP  → quote bid@(bb+1) and ask@(ba-1) when spread≥3,
                               else bb/ba (inside market for queue priority)
          pos > SOFT_CAP    → quote ask@(ba-1) only (close long, no new buys)
          pos < -SOFT_CAP   → quote bid@(bb+1) only (close short, no new sells)
          |pos| ≥ POS_CAP   → emergency market exit (taker), then stop

        Live IV from market mid is used for delta hedge.
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

        # live IV per strike → delta hedge
        mid_iv_by_K: Dict[int, float] = {}

        # ── Per-strike passive MM ─────────────────────────────────────────────
        for K in self.ACTIVE_STRIKES:
            sym   = self.VEV[K]
            depth = state.order_depths.get(sym)
            bk    = self._book(depth)
            if bk is None:
                continue
            bb, bbv, ba, bav = bk
            pos = state.position.get(sym, 0)

            # Skip near-intrinsic options (no time value to capture)
            market_mid = (bb + ba) / 2.0
            if market_mid - max(S - K, 0.0) < self.OPT_MIN_TV:
                continue

            # Live IV from market mid — for delta hedge
            live_iv = self._bs_iv(market_mid, S, float(K), tte)
            if not math.isnan(live_iv):
                mid_iv_by_K[K] = live_iv

            orders: List[Order] = []

            # ── Emergency cap exit (market taker — use sparingly) ─────────────
            if pos >= self.OPT_POS_CAP:
                qty = min(self.OPT_CAP_SIZE, bbv,
                          self._sell_room(pos, 0, self.OPT_POS_CAP))
                if qty > 0 and bb > 0:
                    orders.append(Order(sym, bb, -qty))
                    logger.print(f"[CAP SELL] {sym} pos={pos} bb={bb}")
                if orders:
                    opt_orders[sym] = orders
                continue

            if pos <= -self.OPT_POS_CAP:
                qty = min(self.OPT_CAP_SIZE, bav,
                          self._buy_room(pos, 0, self.OPT_POS_CAP))
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    logger.print(f"[CAP BUY ] {sym} pos={pos} ba={ba}")
                if orders:
                    opt_orders[sym] = orders
                continue

            # ── Passive quotes at market bid / ask with inventory skew ─────────
            bq = 0
            aq = 0
            bid_p = bb
            ask_p = ba

            if pos <= -self.OPT_SOFT_CAP:
                # Short-heavy: prefer to buy back, ease off selling
                if bb + 1 < ba:
                    bq = min(self.OPT_PASSIVE_SIZE,
                             self._buy_room(pos, 0, self.OPT_POS_CAP))
                    bid_p = bb + 1   # improve bid to close short faster
                # Still quote half-size ask so we don't go even shorter
                aq = min(self.OPT_PASSIVE_SIZE // 2,
                         self._sell_room(pos, 0, self.OPT_POS_CAP))

            elif pos >= self.OPT_SOFT_CAP:
                # Long-heavy: prefer to sell, ease off buying
                if ba - 1 > bb:
                    aq = min(self.OPT_PASSIVE_SIZE,
                             self._sell_room(pos, 0, self.OPT_POS_CAP))
                    ask_p = ba - 1   # improve ask to close long faster
                # Still quote half-size bid
                bq = min(self.OPT_PASSIVE_SIZE // 2,
                         self._buy_room(pos, 0, self.OPT_POS_CAP))

            else:
                # Neutral: quote inside the spread for queue priority.
                # Existing market bots sit at bb/ba with 20-unit queues — if we
                # join at the same price we're behind them and never fill.
                # Posting at bb+1 / ba-1 makes us best bid/ask so aggressive
                # market-order flow hits us first.  Only improve when spread≥3
                # to avoid crossing our own orders.
                if ba - bb >= 3:
                    bid_p = bb + 1
                    ask_p = ba - 1
                bq = min(self.OPT_PASSIVE_SIZE,
                         self._buy_room(pos, 0, self.OPT_POS_CAP))
                aq = min(self.OPT_PASSIVE_SIZE,
                         self._sell_room(pos, 0, self.OPT_POS_CAP))

            if bq > 0:
                orders.append(Order(sym, bid_p, bq))
            if aq > 0:
                orders.append(Order(sym, ask_p, -aq))

            if orders:
                opt_orders[sym] = orders
                iv_str = f"{live_iv:.5f}" if not math.isnan(live_iv) else "nan"
                logger.print(
                    f"[OPT] {sym} pos={pos} bid={bid_p}×{bq} ask={ask_p}×{aq} iv={iv_str}"
                )

        # ── Live IV for any deep-ITM / floor-strike positions we hold ─────────
        for K in self.DEEP_ITM_STRIKES + self.FLOOR_STRIKES:
            if state.position.get(self.VEV[K], 0) == 0:
                continue
            bk2 = self._book(state.order_depths.get(self.VEV[K]))
            if bk2 is None:
                continue
            mid2 = (bk2[0] + bk2[2]) / 2.0
            iv2  = self._bs_iv(mid2, S, float(K), tte)
            if not math.isnan(iv2):
                mid_iv_by_K[K] = iv2

        # ── Delta hedge with VF Extract ───────────────────────────────────────
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
            # First tick ever — traderData was empty (fresh submission).
            # Infer TTE from observed option prices to avoid defaulting to day=0
            # when the algo is actually starting mid-round (e.g. day 2).
            inferred_tte = self._infer_tte_from_market(state)
            if inferred_tte is not None:
                # TTE = TTE_START - day - ts/MAX_TS, at ts≈0 → day ≈ TTE_START - tte
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

        # 1. HG taker-first MM
        if self.HG in state.order_depths:
            hg_ords, data = self._trade_hg(state, data)
            result[self.HG] = hg_ords

        # 2. Floor-price passive shorts
        for sym, ords in self._trade_floor_short(state).items():
            result[sym] = ords

        # 3. Options MM (pre-calibrated smile) + delta hedge
        opt_ords, vf_hedge = self._trade_options(state, tte)
        for sym, ords in opt_ords.items():
            result.setdefault(sym, []).extend(ords)
        if vf_hedge:
            result[self.VF] = vf_hedge

        for sym in self.VEV.values():
            if sym not in result:
                result[sym] = []

        trader_data = json.dumps(data, separators=(",", ":"))
        conversions = 0
        logger.flush(state=state, orders=result, conversions=conversions, trader_data=trader_data)
        return result, conversions, trader_data
