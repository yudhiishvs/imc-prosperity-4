"""
trader_template.py  (Round 2)
-----------------------------
Parameterized version of v1_round2_prosperity.py.
All strategy constants are replaced by self.p[key] lookups so the optimizer
can inject any parameter vector without modifying files.

Usage:
    from trader_template import ParameterizedTrader, DEFAULT_PARAMS, PARAM_BOUNDS, ASH_PARAMS, PEPPER_PARAMS
    trader = ParameterizedTrader(DEFAULT_PARAMS)

MAF note:
    The backtester does not yet support a 4th MAF return value.
    We model MAF as "pos_limit" in the params (80 = no MAF, 100 = with MAF).
    The optimizer computes the PnL delta between limit=80 and limit=100 per day,
    which is the break-even MAF you should bid to justify paying for the contract.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle


# ── Default parameter vector (reproduces v1 exactly) ─────────────────────────
DEFAULT_PARAMS: dict = {
    # ── ASH_COATED_OSMIUM ─────────────────────────────────────────────────────
    "ash_fair":        10_000,   # known constant fair value
    "ash_quote_dist":  5,        # L1 passive quote distance from fair
    "ash_quote_size":  14,       # L1 quote size per side
    "ash_l2_dist":     8,        # L2 backstop distance from fair
    "ash_l2_size":     20,       # L2 backstop size per side
    "ash_soft_limit":  40,       # inventory skew ramp starts here
    "ash_hard_limit":  70,       # emergency flatten triggers here

    # ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────────
    "pepper_slope":    0.1001,   # ticks per timestamp unit
    "pepper_take_buf": 3,        # take asks up to fair + this
    "pepper_bid_off":  3,        # passive bids at fair - this
    "pepper_bid_size": 20,       # size per passive bid
    "pepper_hard_buy": 8,        # aggressive fill if position < this

    # ── MAF / position limits ─────────────────────────────────────────────────
    "pos_limit":       80,       # 80 = no MAF, 100 = MAF won (25% extra)
}

# Param names that belong to each asset — used to isolate sweeps
ASH_PARAMS    = ["ash_quote_dist", "ash_quote_size", "ash_l2_dist",
                 "ash_l2_size", "ash_soft_limit", "ash_hard_limit"]
PEPPER_PARAMS = ["pepper_slope", "pepper_take_buf", "pepper_bid_off",
                 "pepper_bid_size", "pepper_hard_buy"]

# Optimizer search bounds: (lo, hi, type)
PARAM_BOUNDS: dict = {
    "ash_quote_dist":  (2,      8,      int),
    "ash_quote_size":  (6,      22,     int),
    "ash_l2_dist":     (4,      14,     int),
    "ash_l2_size":     (10,     35,     int),
    "ash_soft_limit":  (20,     60,     int),
    "ash_hard_limit":  (55,     78,     int),

    "pepper_slope":    (0.0985, 0.1015, float),
    "pepper_take_buf": (1,      7,      int),
    "pepper_bid_off":  (1,      7,      int),
    "pepper_bid_size": (8,      35,     int),
    "pepper_hard_buy": (2,      25,     int),
}


class ParameterizedTrader:

    ASH    = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    def __init__(self, params: dict = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    # ── helpers ────────────────────────────────────────────────────────────────

    def _load(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            d = jsonpickle.decode(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _pos_limit(self, product: str) -> int:
        return self.p["pos_limit"]

    @staticmethod
    def _best(depth: OrderDepth) -> Tuple[int, int]:
        bid = max(depth.buy_orders)  if depth.buy_orders  else 0
        ask = min(depth.sell_orders) if depth.sell_orders else 0
        return bid, ask

    def _buy(self, orders, product, price, qty, pos, pb, limit=None) -> int:
        room = self._pos_limit(product) - (pos + pb)
        qty  = min(qty, room)
        if limit is not None:
            qty = min(qty, limit)
        if qty > 0:
            orders.append(Order(product, price, qty))
            pb += qty
        return pb

    def _sell(self, orders, product, price, qty, pos, ps, limit=None) -> int:
        room = self._pos_limit(product) + (pos - ps)
        qty  = min(qty, room)
        if limit is not None:
            qty = min(qty, limit)
        if qty > 0:
            orders.append(Order(product, price, -qty))
            ps += qty
        return ps

    # ── ASH strategy ──────────────────────────────────────────────────────────

    def _trade_ash(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.ASH)
        if depth is None:
            return []

        orders: List[Order] = []
        pos  = state.position.get(self.ASH, 0)
        pb   = ps = 0
        fair = self.p["ash_fair"]

        # Layer 1: take mispriced liquidity
        for ask in sorted(depth.sell_orders):
            if ask >= fair:
                break
            vol = -depth.sell_orders[ask]
            pb = self._buy(orders, self.ASH, ask, vol, pos, pb)

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair:
                break
            vol = depth.buy_orders[bid]
            ps = self._sell(orders, self.ASH, bid, vol, pos, ps)

        # Layer 2: inventory-skewed passive quotes
        proj = pos + pb - ps
        soft = self.p["ash_soft_limit"]
        hard = self.p["ash_hard_limit"]

        long_ratio  = max(0.0, proj / soft)
        short_ratio = max(0.0, -proj / soft)
        bid_scale   = max(0.0, 1.0 - long_ratio)
        ask_scale   = max(0.0, 1.0 - short_ratio)

        if proj >= hard:
            bid_scale = 0.0
        if proj <= -hard:
            ask_scale = 0.0

        best_bid, best_ask = self._best(depth)

        # Sit inside best visible quote, clamped to fair±1
        passive_bid = (min(best_bid + 1, fair - 1) if best_bid > 0
                       else fair - self.p["ash_quote_dist"])
        passive_ask = (max(best_ask - 1, fair + 1) if best_ask > 0
                       else fair + self.p["ash_quote_dist"])
        passive_bid = min(passive_bid, fair - 1)
        passive_ask = max(passive_ask, fair + 1)

        bid_qty = int(self.p["ash_quote_size"] * bid_scale)
        ask_qty = int(self.p["ash_quote_size"] * ask_scale)

        if bid_qty > 0:
            pb = self._buy(orders, self.ASH, passive_bid, bid_qty, pos, pb)
        if ask_qty > 0:
            ps = self._sell(orders, self.ASH, passive_ask, ask_qty, pos, ps)

        # L2 backstop
        l2_dist = self.p["ash_l2_dist"]
        l2_size = self.p["ash_l2_size"]

        if bid_scale > 0:
            l2_bid = fair - l2_dist
            if l2_bid < passive_bid:
                l2_qty = int(l2_size * bid_scale)
                if l2_qty > 0:
                    pb = self._buy(orders, self.ASH, l2_bid, l2_qty, pos, pb)

        if ask_scale > 0:
            l2_ask = fair + l2_dist
            if l2_ask > passive_ask:
                l2_qty = int(l2_size * ask_scale)
                if l2_qty > 0:
                    ps = self._sell(orders, self.ASH, l2_ask, l2_qty, pos, ps)

        # Layer 3: emergency flatten
        proj = pos + pb - ps
        if proj > hard and best_bid > 0:
            flatten = proj - soft
            ps = self._sell(orders, self.ASH, best_bid, flatten, pos, ps)
        elif proj < -hard and best_ask > 0:
            flatten = abs(proj) - soft
            pb = self._buy(orders, self.ASH, best_ask, flatten, pos, pb)

        return orders

    # ── PEPPER strategy ────────────────────────────────────────────────────────

    def _pepper_fair(self, ts: int, data: dict) -> float:
        first_mid = data.get("pepper_first_mid")
        first_ts  = data.get("pepper_first_ts")
        if first_mid is None or first_ts is None:
            return None
        return first_mid + self.p["pepper_slope"] * (ts - first_ts)

    def _trade_pepper(self, state: TradingState, data: dict) -> List[Order]:
        depth = state.order_depths.get(self.PEPPER)
        if depth is None:
            return []

        orders: List[Order] = []
        pos   = state.position.get(self.PEPPER, 0)
        pb    = ps = 0
        ts    = state.timestamp
        limit = self._pos_limit(self.PEPPER)

        best_bid, best_ask = self._best(depth)
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None

        if data.get("pepper_first_mid") is None and mid is not None:
            data["pepper_first_mid"] = mid
            data["pepper_first_ts"]  = ts

        fair = self._pepper_fair(ts, data)
        if fair is None:
            fair = mid if mid is not None else 12000.0

        pb_room = limit - (pos + pb)

        # Layer 1: take asks ≤ fair + buffer
        if pb_room > 0:
            for ask in sorted(depth.sell_orders):
                if ask > fair + self.p["pepper_take_buf"]:
                    break
                vol   = -depth.sell_orders[ask]
                pb    = self._buy(orders, self.PEPPER, ask, vol, pos, pb)
                pb_room = limit - (pos + pb)
                if pb_room <= 0:
                    break

        # Layer 2: aggressive fill if position critically low
        proj = pos + pb - ps
        if proj < self.p["pepper_hard_buy"] and best_ask > 0:
            gap = self.p["pepper_hard_buy"] - proj
            pb  = self._buy(orders, self.PEPPER, best_ask, gap, pos, pb)

        # Layer 3: passive bids at fair - offset
        proj = pos + pb - ps
        if proj < limit:
            bid_price = int(fair - self.p["pepper_bid_off"])
            bid_qty   = min(self.p["pepper_bid_size"], limit - proj)
            if bid_qty > 0 and bid_price > 0:
                pb = self._buy(orders, self.PEPPER, bid_price, bid_qty, pos, pb)

        # Safety: never short
        proj = pos + pb - ps
        if proj < 0 and best_ask > 0:
            pb = self._buy(orders, self.PEPPER, best_ask + 2, abs(proj), pos, pb)

        return orders

    # ── main entry point ───────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        if self.ASH in state.order_depths:
            result[self.ASH] = self._trade_ash(state)
        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self._trade_pepper(state, data)

        return result, 0, jsonpickle.encode(data)
