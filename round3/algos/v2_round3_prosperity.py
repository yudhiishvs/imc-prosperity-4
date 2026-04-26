"""
v2_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes from v1 (413807):
  KEPT (already working):
    - HYDROGEL EMA fair value, EWMA vol, stationarity stress, OIM signal,
      inventory skew, dynamic sizing, soft/hard limits, emergency unwind,
      kill switches, vol-weighted mid, wall detection (kept but capped).

  ADDED:
    - Taker logic for HYDROGEL (cross spread on clear edge).
    - VF Extract market making: EMA fair + imbalance skew (p=2.2e-21 from EDA).
    - Short VEV_6000 + VEV_6500: post passive asks at 1 every tick (floor-price
      options, near-zero expiry probability for S≈5250, free premium).
    - IV scalping for VEV_5000–VEV_5500: BS fair price (fixed σ from EDA) vs
      market mid; trade deviations > threshold.
    - Day tracking + TTE computation for option pricing.
    - Inline BS math (norm_cdf, call price, vega, IV via Newton) — no scipy.

  NOTE:
    Wall detection is kept but EDA proved wall_mid == mid_price always in Round 3
    data (100% exact match). The WALL_BLEND=0.25 cap means max ±1 tick effect on
    fair value. It is harmless but effectively a no-op. Remove in v3 if desired.

Position limits (confirmed from official Round 3 docs):
    HYDROGEL_PACK       : 200
    VELVETFRUIT_EXTRACT : 200
    VEV_*               : 300 each
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import jsonpickle

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
# Logger (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = max(0, (self.max_log_length - base_length) // 3)
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed: List[List[Any]] = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[str, List[Any]]:
        compressed: Dict[str, List[Any]] = {}
        for symbol, od in order_depths.items():
            compressed[symbol] = [od.buy_orders, od.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed: List[List[Any]] = []
        for trade_list in trades.values():
            for t in trade_list:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed: List[List[Any]] = []
        for order_list in orders.values():
            for o in order_list:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed

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
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:

    # ── Products ──────────────────────────────────────────────────────────────

    HG_PRODUCT  = "HYDROGEL_PACK"
    VF_PRODUCT  = "VELVETFRUIT_EXTRACT"

    # Options — strikes mapped to symbols
    VEV_STRIKES: Dict[int, str] = {
        4000: "VEV_4000",
        4500: "VEV_4500",
        5000: "VEV_5000",
        5100: "VEV_5100",
        5200: "VEV_5200",
        5300: "VEV_5300",
        5400: "VEV_5400",
        5500: "VEV_5500",
        6000: "VEV_6000",
        6500: "VEV_6500",
    }

    # Deep OTM — short these passively (floor-price, near-zero expiry risk)
    VEV_SHORT_STRIKES  = [6000, 6500]

    # ATM band — scalp these using BS fair value
    VEV_SCALP_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]

    # Deep ITM — skip (wide spread, low time value signal)
    VEV_SKIP_STRIKES   = [4000, 4500]

    # ── Position limits (confirmed from official Round 3 docs) ───────────────

    HG_POSITION_LIMIT  = 200
    VF_POSITION_LIMIT  = 200   # confirmed: same as HYDROGEL
    VEV_POSITION_LIMIT = 300   # per strike (confirmed)

    # ── TTE: for LIVE Round 3 submission, day_counter=0 → TTE=5 ─────────────
    # Historical data context: day 0=tutorial(TTE=8), day 1=R1(TTE=7), day 2=R2(TTE=6)
    # Live execution: Round 3 starts at TTE=5 (7 days - 2 rounds elapsed)
    # day_counter increments on each timestamp reset (new game day within submission)

    TTE_BY_DAY: Dict[int, float] = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}

    # ── Fixed implied vols from EDA (median across 3 days, BS r=0, T/252) ────
    # Used as fair-price reference for v2 scalping; replace with online smile
    # fit in v3.

    VEV_FIXED_IV: Dict[int, float] = {
        5000: 0.218,
        5100: 0.217,
        5200: 0.220,
        5300: 0.224,
        5400: 0.208,
        5500: 0.226,
    }

    # ── HYDROGEL parameters (unchanged from v1) ───────────────────────────────

    HG_EMA_ALPHA          = 0.005
    HG_VOL_EMA_ALPHA      = 0.10
    HG_STRESS_EMA_ALPHA   = 0.05
    HG_INITIAL_FV         = 9900.0

    WALL_ENABLED          = True
    WALL_VOLUME_MULTIPLIER = 1.35
    WALL_MIN_VOLUME       = 18
    WALL_MAX_ADJUST_TICKS = 4.0
    WALL_BLEND            = 0.25

    HG_BID_OFFSET         = 1
    HG_ASK_OFFSET         = 1
    HG_BASE_QUOTE_SIZE    = 20
    HG_MIN_QUOTE_SIZE     = 4
    HG_INVENTORY_SKEW_TICKS = 6.0
    HG_OIM_THRESHOLD      = 0.05

    HG_TARGET_VOL         = 1.50
    HG_EXTREME_VOL        = 6.00
    HG_KILL_VOL           = 6.00
    HG_TARGET_STRESS_Z    = 4.0
    HG_HIGH_STRESS_Z      = 8.0
    HG_KILL_STRESS_Z      = 12.0

    HG_SOFT_POSITION_LIMIT   = 140
    HG_HARD_POSITION_LIMIT   = 190
    HG_EMERGENCY_TARGET      = 150
    HG_MAX_EMERGENCY_TRADE   = 30

    # NEW: taker logic — cross spread when |best_price - fair| > this
    HG_TAKE_EDGE          = 5
    HG_MAX_TAKE_SIZE      = 15

    # ── VF Extract parameters (NEW) ───────────────────────────────────────────

    VF_EMA_ALPHA          = 0.010   # faster than HG to track ~8 tick/day drift
    VF_VOL_EMA_ALPHA      = 0.10
    VF_INITIAL_FV         = 5250.0

    VF_BID_OFFSET         = 2
    VF_ASK_OFFSET         = 2
    VF_BASE_QUOTE_SIZE    = 20     # scaled up (limit=200, same as HG)
    VF_MIN_QUOTE_SIZE     = 4
    VF_INVENTORY_SKEW_TICKS = 3.0
    VF_OIM_THRESHOLD      = 0.05
    VF_OIM_SKEW_TICKS     = 1.5   # shift quote prices by this × imbalance

    VF_TAKE_EDGE          = 3
    VF_MAX_TAKE_SIZE      = 15    # scaled up

    VF_SOFT_LIMIT         = 140   # ~70% of 200 (same ratio as HG)
    VF_HARD_LIMIT         = 190   # ~95% of 200

    # ── VEV passive short parameters (NEW) ───────────────────────────────────

    VEV_SHORT_PRICE       = 1    # ask at 1 (floor ask in data)
    VEV_SHORT_SIZE        = 20   # units per tick; capped by position room

    # ── VEV scalping parameters (NEW) ─────────────────────────────────────────

    VEV_SCALP_THRESHOLD   = 2.0  # δ_price ticks to enter
    VEV_SCALP_EXIT_TICKS  = 0.5  # exit when δ_price < this (converged)
    VEV_SCALP_SIZE        = 5    # units per trade
    VEV_SCALP_MAX_POS     = 50   # per strike (~17% of 300 limit — conservative)
    VEV_MIN_TTE           = 0.5  # stop scalping if TTE < this (near expiry)
    VEV_MIN_TIME_VALUE    = 0.5  # skip if option has negligible time value

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes math (inline — no scipy)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Abramowitz & Stegun 26.2.17 approximation. Max error: 7.5e-8."""
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (
            0.319381530
            + t * (-0.356563782
            + t * (1.781477937
            + t * (-1.821255978
            + t * 1.330274429)))
        )
        p = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * poly
        return p if x >= 0.0 else 1.0 - p

    @staticmethod
    def _bs_call(S: float, K: float, T_days: float, sigma: float, r: float = 0.0) -> float:
        """European call price. T_days in Solvenarian days (1 day = 1/252 year)."""
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return max(S - K, 0.0)
        T = T_days / 252.0
        sq = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
        d2 = d1 - sq
        N = Trader._norm_cdf
        return S * N(d1) - K * math.exp(-r * T) * N(d2)

    @staticmethod
    def _bs_vega(S: float, K: float, T_days: float, sigma: float, r: float = 0.0) -> float:
        """BS vega (∂C/∂σ)."""
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return 0.0
        T = T_days / 252.0
        sq = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
        return S * math.sqrt(T) * (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)

    @staticmethod
    def _bs_iv(C: float, S: float, K: float, T_days: float, r: float = 0.0) -> float:
        """
        Implied vol via Newton–Raphson.
        Returns NaN if the option has no meaningful time value or is unsolvable.
        """
        intrinsic = max(S - K, 0.0)
        if C < intrinsic - 0.5 or T_days <= 1e-6 or S <= 0.0:
            return float("nan")
        C = max(C, intrinsic + 1e-6)

        # Brenner-Subrahmanyam initial guess
        T = T_days / 252.0
        sigma = math.sqrt(2.0 * math.pi / T) * C / S
        sigma = max(0.05, min(sigma, 5.0))

        for _ in range(50):
            price = Trader._bs_call(S, K, T_days, sigma, r)
            vega  = Trader._bs_vega(S, K, T_days, sigma, r)
            if abs(vega) < 1e-10:
                break
            delta = (price - C) / vega
            sigma -= delta
            sigma = max(1e-6, min(sigma, 10.0))
            if abs(delta) < 1e-7:
                break
        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self, raw: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            # Hydrogel
            "hg_fair_ema": None,
            "hg_prev_mid": None,
            "hg_vol_ema": 1.5,
            "hg_stress_ema": 0.0,
            # VF Extract
            "vf_fair_ema": None,
            "vf_prev_mid": None,
            "vf_vol_ema": 5.0,
            # Day tracking (for TTE)
            "day_counter": 0,
            "prev_timestamp": -1,
        }
        if not raw:
            return default
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default
        if not isinstance(data, dict):
            return default
        for key, value in default.items():
            if key not in data or data[key] is None:
                data[key] = value
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities (unchanged from v1 except generalised for any product)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ewma(previous: Optional[float], value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return (1.0 - alpha) * float(previous) + alpha * float(value)

    @staticmethod
    def _safe_book(depth: Optional[OrderDepth]) -> Optional[Tuple[int, int, int, int]]:
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bbv = depth.buy_orders[best_bid]
        bav = -depth.sell_orders[best_ask]
        if best_bid <= 0 or best_ask <= 0 or bbv <= 0 or bav <= 0 or best_bid >= best_ask:
            return None
        return best_bid, bbv, best_ask, bav

    @staticmethod
    def _vol_weighted_mid(bb: int, bbv: int, ba: int, bav: int) -> float:
        total = bbv + bav
        if total <= 0:
            return (bb + ba) / 2.0
        return (bb * bav + ba * bbv) / total

    @staticmethod
    def _buy_room(position: int, pending_buys: int, limit: int) -> int:
        return max(0, limit - (position + pending_buys))

    @staticmethod
    def _sell_room(position: int, pending_sells: int, limit: int) -> int:
        return max(0, limit + (position - pending_sells))

    def _compute_tte(self, day_counter: int, timestamp: int) -> float:
        base = self.TTE_BY_DAY.get(day_counter, max(0.1, 5.0 - day_counter))
        return max(0.0, base - timestamp / 1_000_000)

    # ─────────────────────────────────────────────────────────────────────────
    # Wall helpers (unchanged from v1 — kept but low impact per EDA)
    # ─────────────────────────────────────────────────────────────────────────

    def _largest_wall(self, orders: Dict[int, int], is_sell: bool) -> Tuple[Optional[int], int, float]:
        levels: List[Tuple[int, int]] = []
        for price, vol in orders.items():
            v = abs(vol) if is_sell else vol
            if price > 0 and v > 0:
                levels.append((price, v))
        if len(levels) < 2:
            return None, 0, 0.0
        avg = sum(v for _, v in levels) / len(levels)
        if avg <= 0:
            return None, 0, 0.0
        wp, wv = max(levels, key=lambda x: x[1])
        strength = wv / avg
        if wv < self.WALL_MIN_VOLUME or strength < self.WALL_VOLUME_MULTIPLIER:
            return None, 0, 0.0
        return wp, wv, strength

    def _wall_adjusted_fair(self, depth: OrderDepth, base_fair: float) -> float:
        if not self.WALL_ENABLED:
            return base_fair
        bwp, _, _ = self._largest_wall(depth.buy_orders, False)
        awp, _, _ = self._largest_wall(depth.sell_orders, True)
        if bwp is not None and awp is not None and bwp < awp:
            anchor = (bwp + awp) / 2.0
        elif bwp is not None:
            anchor = max(base_fair, float(bwp))
        elif awp is not None:
            anchor = min(base_fair, float(awp))
        else:
            return base_fair
        raw_adj = anchor - base_fair
        capped  = max(-self.WALL_MAX_ADJUST_TICKS, min(self.WALL_MAX_ADJUST_TICKS, raw_adj))
        return base_fair + self.WALL_BLEND * capped

    # ─────────────────────────────────────────────────────────────────────────
    # HYDROGEL strategy (v1 logic + taker)
    # ─────────────────────────────────────────────────────────────────────────

    def _dynamic_hg_quote_size(self, vol: float, stress: float, position: int) -> int:
        safe_vol = max(0.25, vol)
        if safe_vol >= self.HG_EXTREME_VOL:
            vm = 0.25
        else:
            vm = max(0.25, min(1.0, self.HG_TARGET_VOL / safe_vol))

        if stress >= self.HG_HIGH_STRESS_Z:
            sm = 0.40
        elif stress >= self.HG_TARGET_STRESS_Z:
            sm = 0.70
        else:
            sm = 1.00

        ir = abs(position) / self.HG_POSITION_LIMIT
        im = 1.0 if ir < 0.25 else (0.80 if ir < 0.50 else (0.60 if ir < 0.75 else 0.40))

        return max(self.HG_MIN_QUOTE_SIZE, min(self.HG_BASE_QUOTE_SIZE, int(round(
            self.HG_BASE_QUOTE_SIZE * vm * sm * im
        ))))

    def _trade_hydrogel(
        self, state: TradingState, data: Dict[str, Any]
    ) -> Tuple[List[Order], Dict[str, Any]]:
        depth = state.order_depths.get(self.HG_PRODUCT)
        if depth is None:
            return [], data
        book = self._safe_book(depth)
        if book is None:
            return [], data

        bb, bbv, ba, bav = book
        orders: List[Order] = []
        pos = state.position.get(self.HG_PRODUCT, 0)
        pb = ps = 0

        raw_mid    = (bb + ba) / 2.0
        wmid       = self._vol_weighted_mid(bb, bbv, ba, bav)

        # ── 1. EMA fair value ─────────────────────────────────────────────────
        ema = data["hg_fair_ema"]
        if ema is None:
            ema = wmid
        ema = self._ewma(ema, wmid, self.HG_EMA_ALPHA)
        data["hg_fair_ema"] = ema

        # ── 2. Wall adjustment (low impact but harmless) ──────────────────────
        fair = self._wall_adjusted_fair(depth, ema)

        # ── 3. Volatility EMA ─────────────────────────────────────────────────
        prev_mid = data.get("hg_prev_mid")
        vol_ema  = data["hg_vol_ema"]
        if prev_mid is not None:
            vol_ema = self._ewma(vol_ema, abs(raw_mid - prev_mid), self.HG_VOL_EMA_ALPHA)
        data["hg_vol_ema"]  = vol_ema
        data["hg_prev_mid"] = raw_mid

        # ── 4. Stationarity stress ────────────────────────────────────────────
        stress_z = abs(raw_mid - fair) / max(1.0, vol_ema)
        stress   = self._ewma(data["hg_stress_ema"], stress_z, self.HG_STRESS_EMA_ALPHA)
        data["hg_stress_ema"] = stress

        # ── 5. Order imbalance ────────────────────────────────────────────────
        oim = (bbv - bav) / (bbv + bav) if (bbv + bav) > 0 else 0.0

        # ── 6. NEW: Taker logic — cross spread on clear edge ─────────────────
        #   Take BEFORE placing passive quotes; only up to HG_MAX_TAKE_SIZE.
        take_buy_cap  = min(self.HG_MAX_TAKE_SIZE, self._buy_room(pos, pb, self.HG_POSITION_LIMIT))
        take_sell_cap = min(self.HG_MAX_TAKE_SIZE, self._sell_room(pos, ps, self.HG_POSITION_LIMIT))

        if ba <= round(fair) - self.HG_TAKE_EDGE and take_buy_cap > 0:
            # Ask is cheap relative to fair → take it
            take_qty = min(take_buy_cap, bav)
            if take_qty > 0:
                orders.append(Order(self.HG_PRODUCT, ba, take_qty))
                pb += take_qty
                logger.print(f"[HG TAKE BUY] fair={fair:.1f} ask={ba} qty={take_qty}")

        if bb >= round(fair) + self.HG_TAKE_EDGE and take_sell_cap > 0:
            # Bid is rich relative to fair → hit it
            take_qty = min(take_sell_cap, bbv)
            if take_qty > 0:
                orders.append(Order(self.HG_PRODUCT, bb, -take_qty))
                ps += take_qty
                logger.print(f"[HG TAKE SELL] fair={fair:.1f} bid={bb} qty={take_qty}")

        # ── 7. Emergency unwind ───────────────────────────────────────────────
        if pos >= self.HG_HARD_POSITION_LIMIT:
            red = min(pos - self.HG_EMERGENCY_TARGET, self.HG_MAX_EMERGENCY_TRADE,
                      bbv, self._sell_room(pos, ps, self.HG_POSITION_LIMIT))
            if red > 0:
                orders.append(Order(self.HG_PRODUCT, bb, -red))
                ps += red
                logger.print(f"[HG EMERGENCY LONG] pos={pos} sell={bb}x{red}")
            return orders, data

        if pos <= -self.HG_HARD_POSITION_LIMIT:
            red = min(abs(pos) - self.HG_EMERGENCY_TARGET, self.HG_MAX_EMERGENCY_TRADE,
                      bav, self._buy_room(pos, pb, self.HG_POSITION_LIMIT))
            if red > 0:
                orders.append(Order(self.HG_PRODUCT, ba, red))
                pb += red
                logger.print(f"[HG EMERGENCY SHORT] pos={pos} buy={ba}x{red}")
            return orders, data

        # ── 8. Kill switches ──────────────────────────────────────────────────
        if vol_ema >= self.HG_KILL_VOL or stress >= self.HG_KILL_STRESS_Z:
            return orders, data

        # ── 9. Quote sizing and prices ────────────────────────────────────────
        qsize = self._dynamic_hg_quote_size(vol_ema, stress, pos)
        skew  = (pos / self.HG_POSITION_LIMIT) * self.HG_INVENTORY_SKEW_TICKS
        fq    = fair - skew

        bid_p = math.floor(fq - self.HG_BID_OFFSET)
        ask_p = math.ceil(fq + self.HG_ASK_OFFSET)
        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)
        if bid_p >= ask_p:
            bid_p = ask_p - 1

        bq = min(qsize, self._buy_room(pos, pb, self.HG_POSITION_LIMIT))
        aq = min(qsize, self._sell_room(pos, ps, self.HG_POSITION_LIMIT))

        # ── 10. OIM filter ────────────────────────────────────────────────────
        if oim > self.HG_OIM_THRESHOLD:
            aq = 0
        elif oim < -self.HG_OIM_THRESHOLD:
            bq = 0

        # ── 11. Stationarity side filter ──────────────────────────────────────
        if stress >= self.HG_TARGET_STRESS_Z:
            if raw_mid > fair:
                bq = 0
            else:
                aq = 0

        # ── 12. Soft limit ────────────────────────────────────────────────────
        if pos >= self.HG_SOFT_POSITION_LIMIT:
            bq = 0
        if pos <= -self.HG_SOFT_POSITION_LIMIT:
            aq = 0

        # ── 13. Place passive quotes ──────────────────────────────────────────
        if bq > 0:
            orders.append(Order(self.HG_PRODUCT, bid_p, bq))
            pb += bq
        if aq > 0:
            orders.append(Order(self.HG_PRODUCT, ask_p, -aq))
            ps += aq

        logger.print(
            f"[HG] ts={state.timestamp} pos={pos} fair={fair:.1f} "
            f"vol={vol_ema:.2f} stress={stress:.2f} oim={oim:.2f} "
            f"bid={bid_p}x{bq} ask={ask_p}x{aq}"
        )
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # VF Extract strategy (NEW)
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vf_extract(
        self, state: TradingState, data: Dict[str, Any]
    ) -> List[Order]:
        depth = state.order_depths.get(self.VF_PRODUCT)
        if depth is None:
            return []
        book = self._safe_book(depth)
        if book is None:
            return []

        bb, bbv, ba, bav = book
        orders: List[Order] = []
        pos = state.position.get(self.VF_PRODUCT, 0)
        pb = ps = 0

        raw_mid = (bb + ba) / 2.0
        wmid    = self._vol_weighted_mid(bb, bbv, ba, bav)

        # Fair value EMA (faster α to track ~8 tick/day drift identified in EDA)
        ema = data.get("vf_fair_ema")
        if ema is None:
            ema = wmid
        ema = self._ewma(ema, wmid, self.VF_EMA_ALPHA)
        data["vf_fair_ema"] = ema

        # Volatility EMA
        prev_mid = data.get("vf_prev_mid")
        vol_ema  = data.get("vf_vol_ema", 5.0)
        if prev_mid is not None:
            vol_ema = self._ewma(vol_ema, abs(raw_mid - prev_mid), self.VF_VOL_EMA_ALPHA)
        data["vf_vol_ema"]  = vol_ema
        data["vf_prev_mid"] = raw_mid

        fair = ema

        # Order imbalance (EDA: p=2.2e-21, slope=0.0178 — significant predictor)
        # Positive imbalance → price likely rising → lean ask toward fair (don't sell cheap)
        oim = (bbv - bav) / (bbv + bav) if (bbv + bav) > 0 else 0.0

        # ── Taker logic ───────────────────────────────────────────────────────
        take_buy  = min(self.VF_MAX_TAKE_SIZE, self._buy_room(pos, pb, self.VF_POSITION_LIMIT))
        take_sell = min(self.VF_MAX_TAKE_SIZE, self._sell_room(pos, ps, self.VF_POSITION_LIMIT))

        if ba <= round(fair) - self.VF_TAKE_EDGE and take_buy > 0:
            qty = min(take_buy, bav)
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, ba, qty))
                pb += qty

        if bb >= round(fair) + self.VF_TAKE_EDGE and take_sell > 0:
            qty = min(take_sell, bbv)
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, bb, -qty))
                ps += qty

        # ── Hard limit guard ──────────────────────────────────────────────────
        if pos >= self.VF_HARD_LIMIT:
            # Emergency: sell at bid
            qty = min(self.VF_MAX_TAKE_SIZE, bbv, self._sell_room(pos, ps, self.VF_POSITION_LIMIT))
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, bb, -qty))
            return orders
        if pos <= -self.VF_HARD_LIMIT:
            qty = min(self.VF_MAX_TAKE_SIZE, bav, self._buy_room(pos, pb, self.VF_POSITION_LIMIT))
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, ba, qty))
            return orders

        # ── Quote prices with inventory + imbalance skew ──────────────────────
        inv_skew = (pos / self.VF_POSITION_LIMIT) * self.VF_INVENTORY_SKEW_TICKS
        oim_skew = oim * self.VF_OIM_SKEW_TICKS   # lean into the imbalance direction
        fq = fair - inv_skew + oim_skew

        bid_p = math.floor(fq - self.VF_BID_OFFSET)
        ask_p = math.ceil(fq + self.VF_ASK_OFFSET)
        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)
        if bid_p >= ask_p:
            bid_p = ask_p - 1

        bq = min(self.VF_BASE_QUOTE_SIZE, self._buy_room(pos, pb, self.VF_POSITION_LIMIT))
        aq = min(self.VF_BASE_QUOTE_SIZE, self._sell_room(pos, ps, self.VF_POSITION_LIMIT))

        # OIM strong signal: suppress one side
        if oim > self.VF_OIM_THRESHOLD:
            aq = 0
        elif oim < -self.VF_OIM_THRESHOLD:
            bq = 0

        # Soft limit
        if pos >= self.VF_SOFT_LIMIT:
            bq = 0
        if pos <= -self.VF_SOFT_LIMIT:
            aq = 0

        if bq > 0:
            orders.append(Order(self.VF_PRODUCT, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.VF_PRODUCT, ask_p, -aq))

        logger.print(
            f"[VF] ts={state.timestamp} pos={pos} fair={fair:.1f} "
            f"oim={oim:.2f} oim_skew={oim_skew:.2f} "
            f"bid={bid_p}x{bq} ask={ask_p}x{aq}"
        )
        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # VEV_6000 / VEV_6500 passive short (NEW)
    # EDA: these trade at bid=0, ask=1, mid=0.5 with std=0 across ALL 30k ticks.
    # Underlying S≈5250 → strike 6000/6500 are 14%/24% OTM.
    # Collect 1 XIRECS per unit. Max loss = (S_expiry - K) if S spikes (very low prob).
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vev_passive_short(
        self, state: TradingState
    ) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        for strike in self.VEV_SHORT_STRIKES:
            sym = self.VEV_STRIKES[strike]
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            pos = state.position.get(sym, 0)
            # Negative position = short; we short up to VEV_SCALP_MAX_POS
            room = self._sell_room(pos, 0, self.VEV_POSITION_LIMIT)
            qty  = min(self.VEV_SHORT_SIZE, room)
            if qty <= 0:
                continue
            # Post passive ask at price 1 (floor ask from EDA)
            orders = [Order(sym, self.VEV_SHORT_PRICE, -qty)]
            result[sym] = orders
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # VEV IV scalping (NEW)
    # For each ATM strike, compute BS theoretical price using fixed σ from EDA.
    # Trade when |market_mid - BS_price| > VEV_SCALP_THRESHOLD.
    # v3 will replace fixed σ with online parabola-fit (Hedgehog approach).
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vev_scalp(
        self, state: TradingState, data: Dict[str, Any], tte: float
    ) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        if tte < self.VEV_MIN_TTE:
            # Too close to expiry — BS pricing unreliable
            return result

        # Get underlying price from VF Extract book
        vf_depth = state.order_depths.get(self.VF_PRODUCT)
        if vf_depth is None:
            return result
        vf_book = self._safe_book(vf_depth)
        if vf_book is None:
            return result
        S = (vf_book[0] + vf_book[2]) / 2.0   # VF mid price

        for strike in self.VEV_SCALP_STRIKES:
            sym = self.VEV_STRIKES[strike]
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            book = self._safe_book(depth)
            if book is None:
                continue

            bb, bbv, ba, bav = book
            mid_price = (bb + ba) / 2.0
            pos = state.position.get(sym, 0)

            # Skip options with negligible time value
            intrinsic  = max(S - strike, 0.0)
            time_value = mid_price - intrinsic
            if time_value < self.VEV_MIN_TIME_VALUE:
                continue

            # BS theoretical price using fixed σ from EDA
            sigma = self.VEV_FIXED_IV.get(strike)
            if sigma is None:
                continue
            bs_price = self._bs_call(S, float(strike), tte, sigma)
            delta    = mid_price - bs_price

            # Trade signal
            orders: List[Order] = []

            if delta > self.VEV_SCALP_THRESHOLD:
                # Option is RICH → sell
                room = self._sell_room(pos, 0, self.VEV_SCALP_MAX_POS)
                qty  = min(self.VEV_SCALP_SIZE, room, bav)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))   # hit bid (aggressive)
                    logger.print(
                        f"[VEV SELL] {sym} S={S:.1f} bs={bs_price:.2f} "
                        f"mid={mid_price:.1f} δ={delta:.2f} qty={qty}"
                    )

            elif delta < -self.VEV_SCALP_THRESHOLD:
                # Option is CHEAP → buy
                room = self._buy_room(pos, 0, self.VEV_SCALP_MAX_POS)
                qty  = min(self.VEV_SCALP_SIZE, room, bbv)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))    # lift ask (aggressive)
                    logger.print(
                        f"[VEV BUY]  {sym} S={S:.1f} bs={bs_price:.2f} "
                        f"mid={mid_price:.1f} δ={delta:.2f} qty={qty}"
                    )

            if orders:
                result[sym] = orders

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)

        # ── Day tracking: detect timestamp reset to increment day counter ─────
        prev_ts = data.get("prev_timestamp", -1)
        day     = data.get("day_counter", 0)
        if prev_ts >= 0 and state.timestamp < prev_ts - 500_000:
            day += 1
            logger.print(f"[DAY] New day detected: day_counter={day}")
        data["day_counter"]    = day
        data["prev_timestamp"] = state.timestamp

        tte = self._compute_tte(day, state.timestamp)

        # ── Build orders ──────────────────────────────────────────────────────
        result: Dict[Symbol, List[Order]] = {}

        # HYDROGEL: existing MM + new taker logic
        if self.HG_PRODUCT in state.order_depths:
            hg_orders, data = self._trade_hydrogel(state, data)
            result[self.HG_PRODUCT] = hg_orders

        # VF Extract: new EMA MM + imbalance skew
        if self.VF_PRODUCT in state.order_depths:
            result[self.VF_PRODUCT] = self._trade_vf_extract(state, data)

        # VEV_6000 / VEV_6500: passive short (floor-price free premium)
        vev_short = self._trade_vev_passive_short(state)
        for sym, orders in vev_short.items():
            result[sym] = orders

        # VEV_5000–5500: IV scalping vs BS fair price
        vev_scalp = self._trade_vev_scalp(state, data, tte)
        for sym, orders in vev_scalp.items():
            result.setdefault(sym, []).extend(orders)

        # Ensure all products have an explicit empty list (good practice)
        for strike_sym in self.VEV_STRIKES.values():
            if strike_sym not in result:
                result[strike_sym] = []

        # ── Flush ─────────────────────────────────────────────────────────────
        trader_data = jsonpickle.encode(data)
        conversions = 0
        logger.flush(state=state, orders=result, conversions=conversions, trader_data=trader_data)
        return result, conversions, trader_data
