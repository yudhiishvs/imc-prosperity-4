"""
v4_round3_prosperity.py
IMC Prosperity 4 — Round 3

Changes from v3:
  1. Size-15 insider signal (EDA: size-15 at L1 predicts +3–8 tick move in next 100 ticks)
       • HG:  detect size-15 at best bid/ask → ±1 tick fair-value nudge
       • VF:  detect size-15 at best bid/ask → ±2 tick fair-value nudge + quote-side flip
     Signal is stateless (no memory needed; appears on 10–30% of ticks per EDA).

  2. VF drift compensation (EDA: +8 tick/day systematic upward drift)
       • EMA already tracks the drift, but slow (α=0.010).
       • We add an explicit drift offset: drift_ticks = VF_DRIFT_PER_DAY * intraday_fraction
         applied on top of the EMA fair value so quotes lean long from tick 0 of each day.
       • Drift resets each new game-day (day_counter increments on timestamp reset).

  3. VEV scalp: sign-flip exit (new exit condition)
       • If pos > 0 and residual > VEV_SMILE_THRESHOLD  → option has become rich; exit long.
       • If pos < 0 and residual < -VEV_SMILE_THRESHOLD → option has become cheap; exit short.
       This catches convergence overshoot and locks in profit faster than waiting for |δ|<0.003.

  Unchanged from v3:
    • HYDROGEL EMA MM (wall detection, vol/stress kill switches)
    • VF Extract EMA MM with OIM skew
    • Passive short at VEV_6000/6500 (floor price, ask=1)
    • Online parabola smile fit + EDA prior confirmation for VEV_5000–5500

  Position limits (official Round 3 docs):
    HYDROGEL_PACK : 200   VF_EXTRACT : 200   VEV_* : 300 each

  TTE mapping (live Round 3 execution):
    day_counter=0 → TTE=5   (Round 3 starts at TTE=5)
    day_counter increments on timestamp reset (new game-day).

EDA residual priors (Day 2 = TTE≈4.5):
  5000 → residual −0.016  BUY (underpriced)
  5100 → residual −0.010  BUY
  5200 → residual +0.003  SELL (overpriced)
  5300 → residual +0.007  SELL
  5400 → residual −0.015  BUY
  5500 → residual −0.007  BUY
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
# Logger (unchanged from v1–v3)
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

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[str, List[Any]]:
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

    HG_PRODUCT = "HYDROGEL_PACK"
    VF_PRODUCT = "VELVETFRUIT_EXTRACT"

    VEV_STRIKES: Dict[int, str] = {
        4000: "VEV_4000", 4500: "VEV_4500",
        5000: "VEV_5000", 5100: "VEV_5100",
        5200: "VEV_5200", 5300: "VEV_5300",
        5400: "VEV_5400", 5500: "VEV_5500",
        6000: "VEV_6000", 6500: "VEV_6500",
    }

    # Strikes used for online smile fit (reliable IV, meaningful time value)
    VEV_SMILE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]

    # Floor-price deep-OTM: sell passively at ask=1 (near-zero expiry probability)
    VEV_SHORT_STRIKES = [6000, 6500]

    # Deep ITM (4000, 4500): time value ≈ 0, spread >> edge → skip
    VEV_SKIP_STRIKES  = [4000, 4500]

    # ── Position limits (confirmed from official Round 3 docs) ────────────────
    HG_POSITION_LIMIT  = 200
    VF_POSITION_LIMIT  = 200
    VEV_POSITION_LIMIT = 300   # per strike

    # ── TTE: live Round 3, day_counter=0 → TTE=5 ─────────────────────────────
    TTE_BY_DAY: Dict[int, float] = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}

    # ── EDA residual priors: sign tells us if market over/underprices vol ────
    # Positive residual = market IV > smile → SELL  (−1)
    # Negative residual = market IV < smile → BUY   (+1)
    VEV_PRIOR_DIRECTION: Dict[int, int] = {
        5000: +1,   # BUY  (Day2 residual −0.016)
        5100: +1,   # BUY  (Day2 residual −0.010)
        5200: -1,   # SELL (Day2 residual +0.003)
        5300: -1,   # SELL (Day2 residual +0.007)
        5400: +1,   # BUY  (Day2 residual −0.015)
        5500: +1,   # BUY  (Day2 residual −0.007)
    }

    # Max positions per scalp strike (scaled from conviction map)
    VEV_MAX_POS: Dict[int, int] = {
        5000: 40,
        5100: 25,
        5200: 15,
        5300: 20,
        5400: 40,
        5500: 20,
    }

    # ── VEV smile scalp parameters ────────────────────────────────────────────
    VEV_SMILE_THRESHOLD  = 0.008   # |δIV| to enter (≈1σ of residual noise from EDA)
    VEV_SMILE_EXIT       = 0.003   # |δIV| to exit on convergence
    VEV_SCALP_SIZE       = 5       # units per aggressive fill
    VEV_MIN_TTE          = 0.5     # stop scalping near expiry
    VEV_MIN_TIME_VALUE   = 0.5     # skip if option has negligible time value

    # ── VEV passive short (floor price) ──────────────────────────────────────
    VEV_SHORT_PRICE      = 1
    VEV_SHORT_SIZE       = 20      # per tick, capped by position room
    VEV_SHORT_MAX_POS    = 200     # maximum short per floor-price strike

    # ── HYDROGEL parameters ───────────────────────────────────────────────────
    HG_EMA_ALPHA          = 0.005
    HG_VOL_EMA_ALPHA      = 0.10
    HG_STRESS_EMA_ALPHA   = 0.05
    HG_INITIAL_FV         = 9900.0

    WALL_ENABLED           = True
    WALL_VOLUME_MULTIPLIER = 1.35
    WALL_MIN_VOLUME        = 18
    WALL_MAX_ADJUST_TICKS  = 4.0
    WALL_BLEND             = 0.25

    HG_BID_OFFSET            = 1
    HG_ASK_OFFSET            = 1
    HG_BASE_QUOTE_SIZE       = 20
    HG_MIN_QUOTE_SIZE        = 4
    HG_INVENTORY_SKEW_TICKS  = 6.0
    HG_OIM_THRESHOLD         = 0.05

    # NEW v4: insider signal nudge for HG (size-15 at L1 → ±1 tick on fair)
    HG_INSIDER_NUDGE_TICKS   = 1.0

    HG_TARGET_VOL    = 1.50
    HG_EXTREME_VOL   = 6.00
    HG_KILL_VOL      = 6.00
    HG_TARGET_STRESS_Z = 4.0
    HG_HIGH_STRESS_Z   = 8.0
    HG_KILL_STRESS_Z   = 12.0

    HG_SOFT_POSITION_LIMIT = 140
    HG_HARD_POSITION_LIMIT = 190
    HG_EMERGENCY_TARGET    = 150
    HG_MAX_EMERGENCY_TRADE = 30

    HG_TAKE_EDGE     = 5
    HG_MAX_TAKE_SIZE = 15

    # ── VF Extract parameters ─────────────────────────────────────────────────
    VF_EMA_ALPHA            = 0.010
    VF_VOL_EMA_ALPHA        = 0.10
    VF_INITIAL_FV           = 5250.0

    VF_BID_OFFSET           = 2
    VF_ASK_OFFSET           = 2
    VF_BASE_QUOTE_SIZE      = 20
    VF_MIN_QUOTE_SIZE       = 4
    VF_INVENTORY_SKEW_TICKS = 3.0
    VF_OIM_THRESHOLD        = 0.05
    VF_OIM_SKEW_TICKS       = 1.5

    # NEW v4: insider signal nudge for VF (size-15 at L1 → ±2 ticks on fair)
    VF_INSIDER_NUDGE_TICKS  = 2.0

    # NEW v4: explicit drift compensation (EDA: +8 ticks/day upward drift on VF)
    # Applied as: drift_offset = VF_DRIFT_PER_DAY * (timestamp / 1_000_000)
    # The EMA with α=0.010 already tracks the drift but lags; this front-loads it.
    VF_DRIFT_PER_DAY        = 8.0

    VF_TAKE_EDGE     = 3
    VF_MAX_TAKE_SIZE = 15

    VF_SOFT_LIMIT = 140
    VF_HARD_LIMIT = 190

    # ── Insider bot detection ─────────────────────────────────────────────────
    # EDA: size-15 orders at L1 appear on 10–30% of ticks and predict
    # +3–8 tick move in the following 100 ticks (bullish when at bid).
    INSIDER_SIZE = 15

    # ─────────────────────────────────────────────────────────────────────────
    # Black-Scholes math (inline — no scipy/numpy)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Abramowitz & Stegun 26.2.17. Max error: 7.5e-8."""
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        poly = t * (
            0.319381530 + t * (-0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
        p = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x) * poly
        return p if x >= 0.0 else 1.0 - p

    @staticmethod
    def _bs_call(S: float, K: float, T_days: float, sigma: float, r: float = 0.0) -> float:
        """European call price. T_days in Solvenarian days (1 day = 1/252 yr)."""
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
        if T_days <= 1e-6 or sigma <= 1e-8 or S <= 0.0:
            return 0.0
        T = T_days / 252.0
        sq = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
        return S * math.sqrt(T) * (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)

    @staticmethod
    def _bs_iv(C: float, S: float, K: float, T_days: float, r: float = 0.0) -> float:
        """Implied vol via Newton–Raphson. Returns NaN if unsolvable."""
        intrinsic = max(S - K, 0.0)
        if C < intrinsic - 0.5 or T_days <= 1e-6 or S <= 0.0:
            return float("nan")
        C = max(C, intrinsic + 1e-6)
        T = T_days / 252.0
        sigma = math.sqrt(2.0 * math.pi / T) * C / S   # Brenner-Subrahmanyam init
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
    # Online parabola smile fit — pure Python, no numpy
    # Fits: IV = a·x² + b·x + c  where x = log(S/K)
    # via normal equations solved with Cramer's rule (3×3 system)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_parabola(pairs: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """Least-squares quadratic fit to (x, y) pairs. Returns (a, b, c) or None."""
        if len(pairs) < 3:
            return None

        sx4 = sx3 = sx2 = sx1 = s1 = 0.0
        syx2 = syx1 = sy = 0.0
        for x, y in pairs:
            x2 = x * x
            sx4 += x2 * x2; sx3 += x2 * x; sx2 += x2; sx1 += x; s1 += 1.0
            syx2 += y * x2;  syx1 += y * x; sy += y

        M = [
            [sx4, sx3, sx2],
            [sx3, sx2, sx1],
            [sx2, sx1,  s1],
        ]
        rhs = [syx2, syx1, sy]

        def det3(m: List[List[float]]) -> float:
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        D = det3(M)
        if abs(D) < 1e-12:
            return None

        def replace_col(m: List[List[float]], v: List[float], col: int) -> List[List[float]]:
            out = [row[:] for row in m]
            for i in range(3):
                out[i][col] = v[i]
            return out

        a = det3(replace_col(M, rhs, 0)) / D
        b = det3(replace_col(M, rhs, 1)) / D
        c = det3(replace_col(M, rhs, 2)) / D
        return (a, b, c)

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_data(self, raw: str) -> Dict[str, Any]:
        default: Dict[str, Any] = {
            "hg_fair_ema":    None,
            "hg_prev_mid":    None,
            "hg_vol_ema":     1.5,
            "hg_stress_ema":  0.0,
            "vf_fair_ema":    None,
            "vf_prev_mid":    None,
            "vf_vol_ema":     5.0,
            "day_counter":    0,
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
        for key, val in default.items():
            if key not in data or data[key] is None:
                data[key] = val
        return data

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ewma(prev: Optional[float], value: float, alpha: float) -> float:
        if prev is None:
            return float(value)
        return (1.0 - alpha) * float(prev) + alpha * float(value)

    @staticmethod
    def _safe_book(depth: Optional[OrderDepth]) -> Optional[Tuple[int, int, int, int]]:
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None
        bb = max(depth.buy_orders.keys())
        ba = min(depth.sell_orders.keys())
        bbv = depth.buy_orders[bb]
        bav = -depth.sell_orders[ba]
        if bb <= 0 or ba <= 0 or bbv <= 0 or bav <= 0 or bb >= ba:
            return None
        return bb, bbv, ba, bav

    @staticmethod
    def _vol_weighted_mid(bb: int, bbv: int, ba: int, bav: int) -> float:
        total = bbv + bav
        if total <= 0:
            return (bb + ba) / 2.0
        return (bb * bav + ba * bbv) / total

    @staticmethod
    def _buy_room(pos: int, pending: int, limit: int) -> int:
        return max(0, limit - (pos + pending))

    @staticmethod
    def _sell_room(pos: int, pending: int, limit: int) -> int:
        return max(0, limit + (pos - pending))

    def _compute_tte(self, day_counter: int, timestamp: int) -> float:
        base = self.TTE_BY_DAY.get(day_counter, max(0.1, 5.0 - day_counter))
        return max(0.0, base - timestamp / 1_000_000)

    # ─────────────────────────────────────────────────────────────────────────
    # NEW v4: Insider bot detection
    # EDA finding: size-15 orders at best bid/ask on 10–30% of ticks.
    # When at best bid → price rises +3–8 ticks over next 100 ticks.
    # Returns: +1.0 (bullish), -1.0 (bearish), 0.0 (no signal)
    # ─────────────────────────────────────────────────────────────────────────

    def _insider_signal(self, depth: OrderDepth) -> float:
        if not depth.buy_orders or not depth.sell_orders:
            return 0.0
        bb  = max(depth.buy_orders.keys())
        ba  = min(depth.sell_orders.keys())
        bbv = depth.buy_orders.get(bb, 0)
        bav = abs(depth.sell_orders.get(ba, 0))
        if bbv == self.INSIDER_SIZE:
            return 1.0   # insider at best bid → bullish
        if bav == self.INSIDER_SIZE:
            return -1.0  # insider at best ask → bearish
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Wall helpers (unchanged from v1–v3)
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
        capped = max(-self.WALL_MAX_ADJUST_TICKS, min(self.WALL_MAX_ADJUST_TICKS, raw_adj))
        return base_fair + self.WALL_BLEND * capped

    # ─────────────────────────────────────────────────────────────────────────
    # HYDROGEL strategy — v4: adds insider signal nudge
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

        raw_mid = (bb + ba) / 2.0
        wmid    = self._vol_weighted_mid(bb, bbv, ba, bav)

        ema = data["hg_fair_ema"]
        if ema is None:
            ema = wmid
        ema = self._ewma(ema, wmid, self.HG_EMA_ALPHA)
        data["hg_fair_ema"] = ema

        # Wall adjustment then insider nudge (v4 addition)
        fair = self._wall_adjusted_fair(depth, ema)
        insider = self._insider_signal(depth)
        fair += insider * self.HG_INSIDER_NUDGE_TICKS

        prev_mid = data.get("hg_prev_mid")
        vol_ema  = data["hg_vol_ema"]
        if prev_mid is not None:
            vol_ema = self._ewma(vol_ema, abs(raw_mid - prev_mid), self.HG_VOL_EMA_ALPHA)
        data["hg_vol_ema"]  = vol_ema
        data["hg_prev_mid"] = raw_mid

        stress_z = abs(raw_mid - fair) / max(1.0, vol_ema)
        stress   = self._ewma(data["hg_stress_ema"], stress_z, self.HG_STRESS_EMA_ALPHA)
        data["hg_stress_ema"] = stress

        oim = (bbv - bav) / (bbv + bav) if (bbv + bav) > 0 else 0.0

        # Taker — cross spread only when clear edge
        take_buy  = min(self.HG_MAX_TAKE_SIZE, self._buy_room(pos, pb, self.HG_POSITION_LIMIT))
        take_sell = min(self.HG_MAX_TAKE_SIZE, self._sell_room(pos, ps, self.HG_POSITION_LIMIT))
        if ba <= round(fair) - self.HG_TAKE_EDGE and take_buy > 0:
            qty = min(take_buy, bav)
            if qty > 0:
                orders.append(Order(self.HG_PRODUCT, ba, qty))
                pb += qty
        if bb >= round(fair) + self.HG_TAKE_EDGE and take_sell > 0:
            qty = min(take_sell, bbv)
            if qty > 0:
                orders.append(Order(self.HG_PRODUCT, bb, -qty))
                ps += qty

        # Emergency unwind
        if pos >= self.HG_HARD_POSITION_LIMIT:
            red = min(pos - self.HG_EMERGENCY_TARGET, self.HG_MAX_EMERGENCY_TRADE,
                      bbv, self._sell_room(pos, ps, self.HG_POSITION_LIMIT))
            if red > 0:
                orders.append(Order(self.HG_PRODUCT, bb, -red))
                ps += red
            return orders, data
        if pos <= -self.HG_HARD_POSITION_LIMIT:
            red = min(abs(pos) - self.HG_EMERGENCY_TARGET, self.HG_MAX_EMERGENCY_TRADE,
                      bav, self._buy_room(pos, pb, self.HG_POSITION_LIMIT))
            if red > 0:
                orders.append(Order(self.HG_PRODUCT, ba, red))
                pb += red
            return orders, data

        # Kill switches
        if vol_ema >= self.HG_KILL_VOL or stress >= self.HG_KILL_STRESS_Z:
            return orders, data

        # Quote sizing and prices
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

        if oim > self.HG_OIM_THRESHOLD:
            aq = 0
        elif oim < -self.HG_OIM_THRESHOLD:
            bq = 0

        if stress >= self.HG_TARGET_STRESS_Z:
            if raw_mid > fair:
                bq = 0
            else:
                aq = 0

        if pos >= self.HG_SOFT_POSITION_LIMIT:
            bq = 0
        if pos <= -self.HG_SOFT_POSITION_LIMIT:
            aq = 0

        if bq > 0:
            orders.append(Order(self.HG_PRODUCT, bid_p, bq))
        if aq > 0:
            orders.append(Order(self.HG_PRODUCT, ask_p, -aq))

        logger.print(
            f"[HG] ts={state.timestamp} pos={pos} fair={fair:.1f} ins={insider:+.0f} "
            f"vol={vol_ema:.2f} stress={stress:.2f} oim={oim:.2f} "
            f"bid={bid_p}x{bq} ask={ask_p}x{aq}"
        )
        return orders, data

    # ─────────────────────────────────────────────────────────────────────────
    # VF Extract strategy — v4: adds insider signal + drift compensation
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

        ema = data.get("vf_fair_ema")
        if ema is None:
            ema = wmid
        ema = self._ewma(ema, wmid, self.VF_EMA_ALPHA)
        data["vf_fair_ema"] = ema

        prev_mid = data.get("vf_prev_mid")
        vol_ema  = data.get("vf_vol_ema", 5.0)
        if prev_mid is not None:
            vol_ema = self._ewma(vol_ema, abs(raw_mid - prev_mid), self.VF_VOL_EMA_ALPHA)
        data["vf_vol_ema"]  = vol_ema
        data["vf_prev_mid"] = raw_mid

        # NEW v4: drift offset — EDA found +8 ticks/day systematic upward drift.
        # intraday_fraction ∈ [0, 1] based on timestamp within day (0..1_000_000).
        intraday_frac = min(1.0, state.timestamp / 1_000_000)
        drift_offset  = self.VF_DRIFT_PER_DAY * intraday_frac

        # NEW v4: insider signal nudge
        insider = self._insider_signal(depth)
        insider_offset = insider * self.VF_INSIDER_NUDGE_TICKS

        fair = ema + drift_offset + insider_offset
        oim  = (bbv - bav) / (bbv + bav) if (bbv + bav) > 0 else 0.0

        # Taker
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

        # Hard limit guard
        if pos >= self.VF_HARD_LIMIT:
            qty = min(self.VF_MAX_TAKE_SIZE, bbv, self._sell_room(pos, ps, self.VF_POSITION_LIMIT))
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, bb, -qty))
            return orders
        if pos <= -self.VF_HARD_LIMIT:
            qty = min(self.VF_MAX_TAKE_SIZE, bav, self._buy_room(pos, pb, self.VF_POSITION_LIMIT))
            if qty > 0:
                orders.append(Order(self.VF_PRODUCT, ba, qty))
            return orders

        # Passive quotes with inventory + imbalance skew
        inv_skew = (pos / self.VF_POSITION_LIMIT) * self.VF_INVENTORY_SKEW_TICKS
        oim_skew = oim * self.VF_OIM_SKEW_TICKS
        fq = fair - inv_skew + oim_skew

        bid_p = math.floor(fq - self.VF_BID_OFFSET)
        ask_p = math.ceil(fq + self.VF_ASK_OFFSET)
        bid_p = min(bid_p, ba - 1)
        ask_p = max(ask_p, bb + 1)
        if bid_p >= ask_p:
            bid_p = ask_p - 1

        bq = min(self.VF_BASE_QUOTE_SIZE, self._buy_room(pos, pb, self.VF_POSITION_LIMIT))
        aq = min(self.VF_BASE_QUOTE_SIZE, self._sell_room(pos, ps, self.VF_POSITION_LIMIT))

        if oim > self.VF_OIM_THRESHOLD:
            aq = 0
        elif oim < -self.VF_OIM_THRESHOLD:
            bq = 0

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
            f"drift={drift_offset:+.1f} ins={insider:+.0f} "
            f"oim={oim:.2f} bid={bid_p}x{bq} ask={ask_p}x{aq}"
        )
        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # VEV_6000 / VEV_6500 — passive short at floor price (unchanged from v2/v3)
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vev_passive_short(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        for strike in self.VEV_SHORT_STRIKES:
            sym   = self.VEV_STRIKES[strike]
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            pos  = state.position.get(sym, 0)
            room = self._sell_room(pos, 0, self.VEV_SHORT_MAX_POS)
            qty  = min(self.VEV_SHORT_SIZE, room)
            if qty > 0:
                result[sym] = [Order(sym, self.VEV_SHORT_PRICE, -qty)]
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # VEV ATM scalp — online parabola smile fit + EDA prior confirmation
    # v4 adds: sign-flip exit (lock in profit when residual overshoots)
    #
    # Exit conditions (now three):
    #   A. |residual| < VEV_SMILE_EXIT                → convergence (original)
    #   B. pos > 0 and residual > VEV_SMILE_THRESHOLD → option flipped rich; exit long
    #   C. pos < 0 and residual < -VEV_SMILE_THRESHOLD → option flipped cheap; exit short
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_vev_scalp(
        self, state: TradingState, tte: float
    ) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        if tte < self.VEV_MIN_TTE:
            return result

        # Underlying price
        vf_depth = state.order_depths.get(self.VF_PRODUCT)
        if vf_depth is None:
            return result
        vf_book = self._safe_book(vf_depth)
        if vf_book is None:
            return result
        S = (vf_book[0] + vf_book[2]) / 2.0

        # Collect IV data points
        iv_pairs:  List[Tuple[float, float]]           = []
        book_data: Dict[int, Tuple[int, int, int, int]] = {}
        iv_data:   Dict[int, float]                     = {}

        for strike in self.VEV_SMILE_STRIKES:
            sym   = self.VEV_STRIKES[strike]
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            book = self._safe_book(depth)
            if book is None:
                continue
            bb, bbv, ba, bav = book
            mid_price = (bb + ba) / 2.0

            intrinsic  = max(S - strike, 0.0)
            time_value = mid_price - intrinsic
            if time_value < self.VEV_MIN_TIME_VALUE:
                continue

            iv = self._bs_iv(mid_price, S, float(strike), tte)
            if math.isnan(iv) or iv < 0.05 or iv > 3.0:
                continue

            lm = math.log(S / strike)
            iv_pairs.append((lm, iv))
            iv_data[strike]   = iv
            book_data[strike] = (bb, bbv, ba, bav)

        # Fit smile
        coeffs = self._fit_parabola(iv_pairs) if len(iv_pairs) >= 3 else None
        if coeffs is None:
            return result

        a, b, c = coeffs

        # Compute residuals and place orders
        for strike in self.VEV_SMILE_STRIKES:
            if strike not in iv_data or strike not in book_data:
                continue

            sym       = self.VEV_STRIKES[strike]
            market_iv = iv_data[strike]
            bb, bbv, ba, bav = book_data[strike]
            lm        = math.log(S / strike)
            fitted_iv = a * lm * lm + b * lm + c
            residual  = market_iv - fitted_iv   # + = rich, − = cheap

            prior_dir = self.VEV_PRIOR_DIRECTION.get(strike, 0)
            pos       = state.position.get(sym, 0)
            max_p     = self.VEV_MAX_POS.get(strike, 20)

            orders: List[Order] = []

            # SELL: market IV > smile AND prior confirms SELL
            if (residual > self.VEV_SMILE_THRESHOLD
                    and prior_dir == -1
                    and pos > -max_p):
                room = self._sell_room(pos, 0, max_p)
                qty  = min(self.VEV_SCALP_SIZE, room, bbv)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    logger.print(
                        f"[VEV SELL] {sym} S={S:.0f} iv={market_iv:.4f} "
                        f"fit={fitted_iv:.4f} δ={residual:.4f} pos={pos} qty={qty}"
                    )

            # BUY: market IV < smile AND prior confirms BUY
            elif (residual < -self.VEV_SMILE_THRESHOLD
                    and prior_dir == +1
                    and pos < max_p):
                room = self._buy_room(pos, 0, max_p)
                qty  = min(self.VEV_SCALP_SIZE, room, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    logger.print(
                        f"[VEV BUY]  {sym} S={S:.0f} iv={market_iv:.4f} "
                        f"fit={fitted_iv:.4f} δ={residual:.4f} pos={pos} qty={qty}"
                    )

            # EXIT A: residual has converged to near-zero
            elif abs(residual) < self.VEV_SMILE_EXIT:
                if pos > 0:
                    qty = min(pos, self.VEV_SCALP_SIZE, bbv)
                    if qty > 0:
                        orders.append(Order(sym, bb, -qty))
                        logger.print(f"[VEV EXIT-A] {sym} residual converged δ={residual:.4f}")
                elif pos < 0:
                    qty = min(-pos, self.VEV_SCALP_SIZE, bav)
                    if qty > 0:
                        orders.append(Order(sym, ba, qty))
                        logger.print(f"[VEV EXIT-A] {sym} residual converged δ={residual:.4f}")

            # EXIT B (NEW v4): sign-flip — trade has worked and overshot; lock in profit
            elif pos > 0 and residual > self.VEV_SMILE_THRESHOLD:
                # We're long, but option is now rich → sell
                qty = min(pos, self.VEV_SCALP_SIZE, bbv)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    logger.print(
                        f"[VEV EXIT-B] {sym} long→now rich δ={residual:.4f} pos={pos}"
                    )

            elif pos < 0 and residual < -self.VEV_SMILE_THRESHOLD:
                # We're short, but option is now cheap → buy back
                qty = min(-pos, self.VEV_SCALP_SIZE, bav)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    logger.print(
                        f"[VEV EXIT-B] {sym} short→now cheap δ={residual:.4f} pos={pos}"
                    )

            if orders:
                result[sym] = orders

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)

        # Day tracking: timestamp reset signals new game-day
        prev_ts = data.get("prev_timestamp", -1)
        day     = data.get("day_counter", 0)
        if prev_ts >= 0 and state.timestamp < prev_ts - 500_000:
            day += 1
            logger.print(f"[DAY] New day: day_counter={day}")
        data["day_counter"]    = day
        data["prev_timestamp"] = state.timestamp

        tte = self._compute_tte(day, state.timestamp)

        result: Dict[Symbol, List[Order]] = {}

        # 1. HYDROGEL: EMA MM + taker + insider nudge
        if self.HG_PRODUCT in state.order_depths:
            hg_orders, data = self._trade_hydrogel(state, data)
            result[self.HG_PRODUCT] = hg_orders

        # 2. VF Extract: EMA MM + OIM skew + drift offset + insider nudge
        if self.VF_PRODUCT in state.order_depths:
            result[self.VF_PRODUCT] = self._trade_vf_extract(state, data)

        # 3. VEV_6000 / VEV_6500: passive short at floor ask=1
        for sym, orders in self._trade_vev_passive_short(state).items():
            result[sym] = orders

        # 4. VEV_5000–5500: online smile-fit residual scalp + sign-flip exit
        for sym, orders in self._trade_vev_scalp(state, tte).items():
            result.setdefault(sym, []).extend(orders)

        # Ensure explicit empty list for every VEV product
        for sym in self.VEV_STRIKES.values():
            if sym not in result:
                result[sym] = []

        trader_data = jsonpickle.encode(data)
        conversions = 0
        logger.flush(state=state, orders=result, conversions=conversions, trader_data=trader_data)
        return result, conversions, trader_data