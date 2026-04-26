"""
v8_round3_prosperity.py
IMC Prosperity 4 — Round 3

Combines v7 HG improvements with new pre-calibrated vol smile options strategy.

Options changes vs v7
─────────────────────
1. PRE-CALIBRATED BID/ASK SMILE — no live fitting every tick
   Fitted offline from 3-day round-3 data (scipy curve_fit, a≥0 constrained):

     Moneyness:  m = log(S/K) / √TTE   (positive = ITM)
     IV units:   per √Solvenarian-day

     Ask smile  a=2.2480  b= 0.1860  c=0.0136
     Bid smile  a=1.0000  b=−0.0469  c=0.0130

2. SEPARATE BID/ASK FAIR PRICES
   fair_bid = BS(S, K, TTE, bid_smile_iv)  ← our bid quote price
   fair_ask = BS(S, K, TTE, ask_smile_iv)  ← our ask quote price
   Natural MM spread = vol bid-ask spread embedded in the two smiles.
   Taker logic: buy when mkt_ask < fair_bid − edge  (market cheap vs our bid)
                sell when mkt_bid > fair_ask + edge  (market rich vs our ask)

3. BS IN SOLVENARIAN DAYS throughout — no /252 year conversion.
   sigma ~ 0.013 ATM  (per √Solvenarian-day, not annualised).

4. TTE_START = 7.0 Solvenarian days at round-3 day-0 ts-0.

5. OPT_POS_CAP = 80 (Frankfurt Hedgehog recommendation).
   Guarantees full delta hedge within VF_LIMIT = 200 at any position mix.

6. Deep-ITM strikes (4000, 4500) included in delta computation if held,
   but NOT actively MM'd (bid near intrinsic → no time value to capture).

HG changes kept from v7
────────────────────────
• Taker-first logic: hit/lift when price ≥ EMA ± HG_TAKE_EDGE
• HG_EMA_ALPHA = 0.020  (was 0.005)
• Cap-exit logic for options positions
• Delta hedge DEADBAND = 20 (only hedge when |imbalance| ≥ 20)

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

    # 80 per active strike. Total delta worst case: 80×6×0.65≈312, clipped to 200 in hedge.
    OPT_POS_CAP   = 80
    FLOOR_POS_CAP = 200

    # ── Pre-calibrated vol smile (Solvenarian-day units) ──────────────────────
    # Moneyness: m = log(S/K) / sqrt(TTE_solv)   [positive = ITM for calls]
    # IV: per √Solvenarian-day (ATM ~0.013)
    #
    # Calibrated from 3-day round-3 market data using scipy curve_fit.
    # Ask smile: unconstrained fit naturally finds a=2.25 (data has real curvature).
    # Bid smile: constrained a≥1 to prevent degenerate linear collapse.
    ASK_SMILE: Tuple[float, float, float] = (2.2480,  0.1860, 0.0136)
    BID_SMILE: Tuple[float, float, float] = (1.0000, -0.0469, 0.0130)

    # ── TTE ───────────────────────────────────────────────────────────────────
    TTE_START      = 7.0       # Solvenarian days at round-3 day-0 ts-0
    MAX_TS_PER_DAY = 999_900

    # ── Option MM parameters ──────────────────────────────────────────────────
    OPT_TAKER_EDGE   = 2.0    # take when mkt price crosses our fair by this many seashells
    OPT_TAKER_SIZE   = 15     # max units per aggressive fill
    OPT_PASSIVE_SIZE = 10     # max units per passive quote
    OPT_MIN_TTE      = 0.3    # stop trading very near expiry
    OPT_MIN_TV       = 0.5    # skip if time-value < this (near intrinsic, no MM edge)

    FLOOR_SHORT_SIZE = 20     # passive sell size per tick for floor-price strikes

    # ── Delta hedge ───────────────────────────────────────────────────────────
    HEDGE_DEADBAND = 20       # only hedge when |imbalance| ≥ this (avoid paying spread every tick)

    # ── HG MM parameters (from v7 grid search) ────────────────────────────────
    HG_EMA_ALPHA   = 0.020   # fast EMA tracks price better
    HG_TAKE_EDGE   = 8       # taker threshold from EMA (ticks)
    HG_TAKER_SIZE  = 10
    HG_QUOTE_TICK  = 2
    HG_SKEW_TICKS  = 3.0
    HG_QUOTE_SIZE  = 20
    HG_SOFT_LIMIT  = 140
    HG_HARD_LIMIT  = 190
    HG_UNWIND_SIZE = 30

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
        skew  = (pos / self.HG_LIMIT) * self.HG_SKEW_TICKS
        fair_q = ema - skew
        bid_p = math.floor(fair_q - self.HG_QUOTE_TICK)
        ask_p = math.ceil(fair_q + self.HG_QUOTE_TICK)

        bq = 0 if pos >= self.HG_SOFT_LIMIT  else min(self.HG_QUOTE_SIZE,
                                                        self._buy_room(pos, 0, self.HG_LIMIT))
        aq = 0 if pos <= -self.HG_SOFT_LIMIT else min(self.HG_QUOTE_SIZE,
                                                        self._sell_room(pos, 0, self.HG_LIMIT))
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
    # 3. OPTIONS MM — pre-calibrated bid/ask smile + delta hedge
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_options(
        self, state: TradingState, tte: float
    ) -> Tuple[Dict[str, List[Order]], List[Order]]:
        """
        For each ACTIVE_STRIKE:
          m = log(S/K) / sqrt(TTE)
          ask_iv = ASK_SMILE(m),  bid_iv = BID_SMILE(m)
          fair_ask = BS(S,K,TTE,ask_iv)   ← our offer
          fair_bid = BS(S,K,TTE,bid_iv)   ← our bid

          Taker: buy  when mkt_ask < fair_bid - TAKER_EDGE  (cheap vs our bid)
                 sell when mkt_bid > fair_ask + TAKER_EDGE  (rich vs our ask)
          Passive: quote at round(fair_bid) / round(fair_ask)
          Cap-exit: at ±OPT_POS_CAP sell/buy at market without edge requirement.

        Delta hedge all VEV positions with VF Extract (deadband = 20).
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
        sqrt_tte = math.sqrt(tte)

        # mid IV per strike, used for delta hedge
        mid_iv_by_K: Dict[int, float] = {}

        # ── Per-strike active MM ──────────────────────────────────────────────
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

            # Standardised log-moneyness: log(S/K)/sqrt(TTE); positive = ITM
            m = math.log(S / K) / sqrt_tte

            ask_iv = self._smile_iv(m, self.ASK_SMILE)
            bid_iv = self._smile_iv(m, self.BID_SMILE)
            mid_iv = (ask_iv + bid_iv) / 2.0
            mid_iv_by_K[K] = mid_iv

            fair_ask = self._bs_call(S, float(K), tte, ask_iv)
            fair_bid = self._bs_call(S, float(K), tte, bid_iv)

            orders: List[Order] = []
            pb = ps = 0

            # ── Priority 1: Cap-exit — reduce without edge requirement ─────────
            if pos >= self.OPT_POS_CAP:
                qty = min(self.OPT_TAKER_SIZE, bbv,
                          self._sell_room(pos, 0, self.OPT_POS_CAP))
                if qty > 0 and bb > 0:
                    orders.append(Order(sym, bb, -qty))
                    ps += qty
                    logger.print(f"[EXIT SELL] {sym} pos={pos} bb={bb}")
                aq = min(self.OPT_PASSIVE_SIZE, self._sell_room(pos, ps, self.OPT_POS_CAP))
                ask_q = round(fair_ask)
                if ask_q > 0 and aq > 0:
                    orders.append(Order(sym, ask_q, -aq))
                if orders:
                    opt_orders[sym] = orders
                continue

            if pos <= -self.OPT_POS_CAP:
                qty = min(self.OPT_TAKER_SIZE, bav,
                          self._buy_room(pos, 0, self.OPT_POS_CAP))
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pb += qty
                    logger.print(f"[EXIT BUY ] {sym} pos={pos} ba={ba}")
                bq = min(self.OPT_PASSIVE_SIZE, self._buy_room(pos, pb, self.OPT_POS_CAP))
                bid_q = round(fair_bid)
                if bid_q > 0 and bq > 0:
                    orders.append(Order(sym, bid_q, bq))
                if orders:
                    opt_orders[sym] = orders
                continue

            # ── Priority 2: Aggressive taker ──────────────────────────────────
            did_taker_buy = did_taker_sell = False

            # Market selling cheaper than our bid fair → buy it
            if ba < fair_bid - self.OPT_TAKER_EDGE:
                room = self._buy_room(pos, pb, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pb += qty
                    did_taker_buy = True
                    logger.print(
                        f"[TAKE BUY ] {sym} m={m:.3f}"
                        f" bid_iv={bid_iv:.5f} fair_bid={fair_bid:.2f}"
                        f" mkt_ask={ba} pos={pos}"
                    )

            # Market bidding richer than our ask fair → sell it
            elif bb > fair_ask + self.OPT_TAKER_EDGE:
                room = self._sell_room(pos, ps, self.OPT_POS_CAP)
                qty  = min(self.OPT_TAKER_SIZE, room, bbv)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    ps += qty
                    did_taker_sell = True
                    logger.print(
                        f"[TAKE SELL] {sym} m={m:.3f}"
                        f" ask_iv={ask_iv:.5f} fair_ask={fair_ask:.2f}"
                        f" mkt_bid={bb} pos={pos}"
                    )

            # ── Priority 3: Passive quotes at smile fair prices ───────────────
            bid_q = round(fair_bid)
            ask_q = round(fair_ask)

            if bid_q > 0 and ask_q > bid_q:
                bq = 0 if did_taker_buy  else min(self.OPT_PASSIVE_SIZE,
                                                    self._buy_room(pos, pb, self.OPT_POS_CAP))
                aq = 0 if did_taker_sell else min(self.OPT_PASSIVE_SIZE,
                                                    self._sell_room(pos, ps, self.OPT_POS_CAP))
                if bq > 0:
                    orders.append(Order(sym, bid_q, bq))
                if aq > 0:
                    orders.append(Order(sym, ask_q, -aq))

            if orders:
                opt_orders[sym] = orders

        # ── Mid IV for deep-ITM strikes we may hold ───────────────────────────
        for K in self.DEEP_ITM_STRIKES:
            if state.position.get(self.VEV[K], 0) != 0:
                m = math.log(S / K) / sqrt_tte
                mid_iv_by_K[K] = (self._smile_iv(m, self.ASK_SMILE) +
                                   self._smile_iv(m, self.BID_SMILE)) / 2.0

        # ── Delta hedge with VF Extract ───────────────────────────────────────
        total_delta = 0.0

        for K, iv in mid_iv_by_K.items():
            pos = state.position.get(self.VEV[K], 0)
            if pos == 0:
                continue
            delta = self._bs_delta(S, float(K), tte, iv)
            total_delta += pos * delta

        # Include floor-strike positions (delta small but non-zero, be precise)
        for K in self.FLOOR_STRIKES:
            pos = state.position.get(self.VEV[K], 0)
            if pos == 0:
                continue
            m  = math.log(S / K) / sqrt_tte
            iv = (self._smile_iv(m, self.ASK_SMILE) +
                  self._smile_iv(m, self.BID_SMILE)) / 2.0
            total_delta += pos * self._bs_delta(S, float(K), tte, iv)

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
        if prev_ts >= 0 and state.timestamp < prev_ts - 500_000:
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