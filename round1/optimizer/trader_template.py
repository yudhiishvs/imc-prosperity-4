"""
trader_template.py
------------------
Parameterized version of the v35 Trader.
All hardcoded scalars are replaced with self.p[key] lookups so the optimizer
can inject any parameter vector without touching file I/O.

Usage:
    from trader_template import ParameterizedTrader, DEFAULT_PARAMS
    trader = ParameterizedTrader(DEFAULT_PARAMS)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle


# ── Default parameter vector (reproduces v35 exactly) ──────────────────────
DEFAULT_PARAMS: dict = {
    # ── ASH_COATED_OSMIUM ──────────────────────────────────────────────────
    # Market-making basis vector
    "ash_base_quote_size":       40,     # total passive quote qty per side
    "ash_inner_qty_frac":        0.7,    # fraction placed at inner level
    "ash_outer_offset":          2,      # ticks between inner and outer quote

    # Inventory-skew basis vector
    "ash_volume_skew":           1.5,    # aggressiveness of size scaling
    "ash_kill_switch":           70,     # |pos| where one side is zeroed
    "ash_emergency_threshold":   60,     # |pos| that triggers emergency flatten
    "ash_emergency_target":      30,     # |pos| to flatten down to
    "ash_directional_threshold": 999,    # |pos| hard-stop adding side (999=off)

    # Momentum/drift basis vector
    "ash_ema_alpha":             0.2,    # EMA smoothing for fair value
    "ash_momentum_bid_scale":    1.35,   # bid size boost when price fell last tick
    "ash_momentum_ask_scale":    1.35,   # ask size boost when price rose last tick
    "ash_momentum_neutral_scale":1.0,    # size scale when no last-tick change

    # ── INTARIAN_PEPPER_ROOT ───────────────────────────────────────────────
    # Trend-carry basis vector
    "pepper_slope":              0.001,  # price per timestamp tick
    "pepper_buy_ceiling":        10,     # aggressive take up to fair + this
    "pepper_passive_bid_1":      2,      # first passive bid: fair + this
    "pepper_passive_bid_2":      1,      # second passive bid: fair + this
}

# Parameter space bounds for the optimizer (lo, hi, type)
PARAM_BOUNDS: dict = {
    "ash_base_quote_size":        (10,  60,   int),
    "ash_inner_qty_frac":         (0.4, 0.9,  float),
    "ash_outer_offset":           (1,   4,    int),
    "ash_volume_skew":            (0.5, 3.0,  float),
    "ash_kill_switch":            (55,  80,   int),
    "ash_emergency_threshold":    (40,  72,   int),
    "ash_emergency_target":       (10,  45,   int),
    "ash_directional_threshold":  (30,  999,  int),
    "ash_ema_alpha":              (0.05,0.5,  float),
    "ash_momentum_bid_scale":     (1.0, 2.0,  float),
    "ash_momentum_ask_scale":     (1.0, 2.0,  float),
    "ash_momentum_neutral_scale": (0.8, 1.2,  float),
    "pepper_slope":               (0.00090, 0.00110, float),
    "pepper_buy_ceiling":         (4,   14,   int),
    "pepper_passive_bid_1":       (0,   6,    int),
    "pepper_passive_bid_2":       (0,   5,    int),
}


class ParameterizedTrader:
    """Drop-in replacement for Trader that accepts an external params dict."""

    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    FAIR_VALUE     = {"ASH_COATED_OSMIUM": 10_000}

    def __init__(self, params: dict = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_data(self, raw: str) -> dict:
        default = {"pepper_base": None, "osmium_ema": None, "osmium_last_mid": None}
        if not raw:
            return default
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default
        if not isinstance(data, dict):
            return default

        pb = data.get("pepper_base")
        if not isinstance(pb, (int, float)):
            pb = data.get("pepper_base_estimate")
            pb = float(pb) if isinstance(pb, (int, float)) else None
        else:
            pb = float(pb)

        ema = data.get("osmium_ema")
        ema = float(ema) if isinstance(ema, (int, float)) else None

        lm = data.get("osmium_last_mid")
        lm = float(lm) if isinstance(lm, (int, float)) else None

        return {"pepper_base": pb, "osmium_ema": ema, "osmium_last_mid": lm}

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
        bb = max(depth.buy_orders)  if depth.buy_orders  else 0
        ba = min(depth.sell_orders) if depth.sell_orders else 0
        return bb, ba

    @staticmethod
    def _mid_price(depth: OrderDepth):
        bb = max(depth.buy_orders)  if depth.buy_orders  else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is not None and ba is not None: return (bb + ba) / 2
        if bb is not None: return float(bb)
        if ba is not None: return float(ba)
        return None

    def _buy_room(self, product, position, pending_buys):
        return self.POSITION_LIMIT[product] - (position + pending_buys)

    def _sell_room(self, product, position, pending_sells):
        return self.POSITION_LIMIT[product] + (position - pending_sells)

    def _place_buy(self, orders, product, price, qty, position, pending_buys):
        room = self._buy_room(product, position, pending_buys)
        q = min(qty, room)
        if q > 0:
            orders.append(Order(product, price, q))
            pending_buys += q
        return pending_buys

    def _place_sell(self, orders, product, price, qty, position, pending_sells):
        room = self._sell_room(product, position, pending_sells)
        q = min(qty, room)
        if q > 0:
            orders.append(Order(product, price, -q))
            pending_sells += q
        return pending_sells

    def _take_asks(self, orders, product, depth, max_price, position,
                   pending_buys, max_total=None):
        bought = 0
        for ask in sorted(depth.sell_orders):
            if ask > max_price or (max_total is not None and bought >= max_total):
                break
            room = self._buy_room(product, position, pending_buys)
            if room <= 0:
                break
            size = min(-depth.sell_orders[ask], room)
            if max_total is not None:
                size = min(size, max_total - bought)
            if size <= 0:
                continue
            orders.append(Order(product, ask, size))
            pending_buys += size
            bought += size
        return pending_buys

    def _inside_bid(self, bb, ba, ticks, fallback):
        if bb is not None and ba is not None:
            sp = max(0, ba - bb - 1)
            return bb + min(ticks, sp) if sp > 0 else bb
        if bb is not None: return bb
        if ba is not None: return ba - 1
        return fallback

    def _inside_ask(self, bb, ba, ticks, fallback):
        if bb is not None and ba is not None:
            sp = max(0, ba - bb - 1)
            return ba - min(ticks, sp) if sp > 0 else ba
        if ba is not None: return ba
        if bb is not None: return bb + 1
        return fallback

    def _take_mispriced(self, orders, product, depth, position,
                        pending_buys, pending_sells, fair_value,
                        buy_inclusive=False, sell_inclusive=False):
        for ask in sorted(depth.sell_orders):
            if ask > fair_value or (ask == fair_value and not buy_inclusive):
                break
            vol = -depth.sell_orders[ask]
            pending_buys = self._place_buy(orders, product, ask, vol, position, pending_buys)
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < fair_value or (bid == fair_value and not sell_inclusive):
                break
            vol = depth.buy_orders[bid]
            pending_sells = self._place_sell(orders, product, bid, vol, position, pending_sells)
        return pending_buys, pending_sells

    def _flatten_at_fair(self, orders, product, depth, position,
                         pending_buys, pending_sells, fair_value):
        fp = int(round(fair_value))
        proj = position + pending_buys - pending_sells
        if proj > 0 and fp in depth.buy_orders:
            qty = min(depth.buy_orders[fp], proj)
            pending_sells = self._place_sell(orders, product, fp, qty, position, pending_sells)
        elif proj < 0 and fp in depth.sell_orders:
            qty = min(-depth.sell_orders[fp], abs(proj))
            pending_buys = self._place_buy(orders, product, fp, qty, position, pending_buys)
        return pending_buys, pending_sells

    def _emergency_flatten(self, orders, product, depth, position,
                           pending_buys, pending_sells):
        proj = position + pending_buys - pending_sells
        et   = self.p["ash_emergency_threshold"]
        tgt  = self.p["ash_emergency_target"]
        if abs(proj) <= et:
            return False, pending_buys, pending_sells
        bb, ba = self._best_bid_ask(depth)
        if proj > et:
            qty = proj - tgt
            if bb > 0 and qty > 0:
                pending_sells = self._place_sell(orders, product, bb, qty, position, pending_sells)
            return True, pending_buys, pending_sells
        if proj < -et:
            qty = abs(proj) - tgt
            if ba > 0 and qty > 0:
                pending_buys = self._place_buy(orders, product, ba, qty, position, pending_buys)
            return True, pending_buys, pending_sells
        return False, pending_buys, pending_sells

    def _penny_jump_quotes(self, orders, product, depth, position,
                           pending_buys, pending_sells, fair_value,
                           quote_shift, bid_signal_scale, ask_signal_scale,
                           directional_long=False, directional_short=False):
        bb_raw, ba_raw = self._best_bid_ask(depth)
        bb = bb_raw if bb_raw != 0 else None
        ba = ba_raw if ba_raw != 0 else None
        fair_floor = int(fair_value)
        fair_ceil  = fair_floor if fair_value == fair_floor else fair_floor + 1

        proj  = position + pending_buys - pending_sells
        ratio = proj / self.POSITION_LIMIT[product]
        skew  = self.p["ash_volume_skew"]
        bid_scale = max(0.0, 1.0 - max(0.0,  ratio) * skew)
        ask_scale = max(0.0, 1.0 + min(0.0,  ratio) * skew)

        bqs = self.p["ash_base_quote_size"]
        total_bid_qty = int(round(bqs * bid_scale * bid_signal_scale))
        total_ask_qty = int(round(bqs * ask_scale * ask_signal_scale))

        if directional_long:
            total_bid_qty = 0
        elif directional_short:
            total_ask_qty = 0

        ks = self.p["ash_kill_switch"]
        if proj >= ks:
            total_bid_qty = 0
        elif proj <= -ks:
            total_ask_qty = 0

        frac = self.p["ash_inner_qty_frac"]
        oo   = self.p["ash_outer_offset"]
        inner_bid_qty = int(round(total_bid_qty * frac))
        outer_bid_qty = max(0, total_bid_qty - inner_bid_qty)
        inner_ask_qty = int(round(total_ask_qty * frac))
        outer_ask_qty = max(0, total_ask_qty - inner_ask_qty)

        inner_bid = self._inside_bid(bb, ba, 1, fair_floor - 1 + quote_shift)
        inner_bid = min(inner_bid, fair_floor)
        outer_bid = min(inner_bid - oo, fair_floor - oo + quote_shift)
        if ba is not None:
            outer_bid = min(outer_bid, ba - 1)

        inner_ask = self._inside_ask(bb, ba, 1, fair_ceil + 1 + quote_shift)
        inner_ask = max(inner_ask, fair_ceil)
        if directional_long:
            inner_ask = max(fair_ceil, inner_ask - 1)
        outer_ask = max(inner_ask + oo, fair_ceil + oo + quote_shift)
        if bb is not None:
            outer_ask = max(outer_ask, bb + 1)
        if directional_short:
            inner_bid = min(fair_floor, inner_bid + 1)

        if inner_bid_qty > 0 and (ba is None or inner_bid < ba):
            pending_buys = self._place_buy(orders, product, inner_bid, inner_bid_qty, position, pending_buys)
        if outer_bid_qty > 0 and outer_bid > 0 and (ba is None or outer_bid < ba):
            pending_buys = self._place_buy(orders, product, outer_bid, outer_bid_qty, position, pending_buys)
        if inner_ask_qty > 0 and (bb is None or inner_ask > bb):
            pending_sells = self._place_sell(orders, product, inner_ask, inner_ask_qty, position, pending_sells)
        if outer_ask_qty > 0 and (bb is None or outer_ask > bb):
            pending_sells = self._place_sell(orders, product, outer_ask, outer_ask_qty, position, pending_sells)

        return pending_buys, pending_sells

    # ── ASH trading logic ──────────────────────────────────────────────────

    def _trade_osmium(self, state: TradingState, fair_value, current_mid, last_mid) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        depth   = state.order_depths.get(product)
        if depth is None:
            return []

        orders: List[Order] = []
        position      = state.position.get(product, 0)
        pending_buys  = 0
        pending_sells = 0
        fair          = self.FAIR_VALUE[product] if fair_value is None else fair_value
        last_change   = 0.0 if (current_mid is None or last_mid is None) else current_mid - last_mid

        dt = self.p["ash_directional_threshold"]
        directional_long  = position >  dt
        directional_short = position < -dt

        pending_buys, pending_sells = self._take_mispriced(
            orders, product, depth, position, pending_buys, pending_sells,
            fair_value=fair, buy_inclusive=True, sell_inclusive=True,
        )
        pending_buys, pending_sells = self._flatten_at_fair(
            orders, product, depth, position, pending_buys, pending_sells, fair_value=fair,
        )
        triggered, pending_buys, pending_sells = self._emergency_flatten(
            orders, product, depth, position, pending_buys, pending_sells,
        )
        if triggered:
            return orders

        ns  = self.p["ash_momentum_neutral_scale"]
        bs  = self.p["ash_momentum_bid_scale"]
        as_ = self.p["ash_momentum_ask_scale"]
        if last_change > 0:
            quote_shift, bid_signal_scale, ask_signal_scale = -1, ns, as_
        elif last_change < 0:
            quote_shift, bid_signal_scale, ask_signal_scale =  1, bs, ns
        else:
            quote_shift, bid_signal_scale, ask_signal_scale =  0, ns, ns

        self._penny_jump_quotes(
            orders, product, depth, position, pending_buys, pending_sells,
            fair, quote_shift, bid_signal_scale, ask_signal_scale,
            directional_long=directional_long, directional_short=directional_short,
        )
        return orders

    # ── PEPPER trading logic ───────────────────────────────────────────────

    def _trade_pepper_root(self, state: TradingState, fair_value) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        depth   = state.order_depths.get(product)
        if depth is None or fair_value is None:
            return []

        orders: List[Order] = []
        position      = state.position.get(product, 0)
        limit         = self.POSITION_LIMIT[product]
        pending_buys  = 0
        pending_sells = 0
        best_bid, _   = self._best_bid_ask(depth)

        if position > limit:
            sp = best_bid if best_bid != 0 else int(fair_value)
            self._place_sell(orders, product, sp, position - limit, position, pending_sells)
            return orders

        deficit = limit - position
        if deficit <= 0:
            return orders

        ceiling = int(fair_value) + self.p["pepper_buy_ceiling"]
        pending_buys = self._take_asks(
            orders, product, depth, ceiling, position, pending_buys, max_total=deficit,
        )

        remaining = limit - (position + pending_buys)
        if remaining > 0:
            p1 = int(fair_value) + self.p["pepper_passive_bid_1"]
            p2 = int(fair_value) + self.p["pepper_passive_bid_2"]
            q1 = (remaining + 1) // 2
            q2 = remaining - q1
            pending_buys = self._place_buy(orders, product, p1, q1, position, pending_buys)
            if q2 > 0:
                pending_buys = self._place_buy(orders, product, p2, q2, position, pending_buys)

        return orders

    # ── main entry point ───────────────────────────────────────────────────

    def run(self, state: TradingState):
        data            = self._load_data(state.traderData)
        pepper_base     = data.get("pepper_base")
        osmium_ema      = data.get("osmium_ema")
        osmium_last_mid = data.get("osmium_last_mid")

        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        pepper_mid   = self._mid_price(pepper_depth) if pepper_depth else None
        timestamp    = state.timestamp

        if pepper_base is None and pepper_mid is not None:
            pepper_base = float(pepper_mid) - self.p["pepper_slope"] * float(timestamp)
        pepper_fair = (
            None if pepper_base is None
            else pepper_base + self.p["pepper_slope"] * float(timestamp)
        )

        osmium_depth = state.order_depths.get("ASH_COATED_OSMIUM")
        osmium_mid   = self._mid_price(osmium_depth) if osmium_depth else None
        alpha        = self.p["ash_ema_alpha"]
        if osmium_mid is not None:
            osmium_ema = (
                float(osmium_mid) if osmium_ema is None
                else alpha * float(osmium_mid) + (1 - alpha) * float(osmium_ema)
            )
        osmium_fair = (
            float(osmium_ema) if osmium_ema is not None
            else float(self.FAIR_VALUE["ASH_COATED_OSMIUM"])
        )

        result: Dict[str, List[Order]] = {}
        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(
                state, osmium_fair, osmium_mid, osmium_last_mid
            )
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self._trade_pepper_root(state, pepper_fair)

        data["pepper_base"]     = pepper_base
        data["osmium_ema"]      = osmium_ema
        data["osmium_last_mid"] = osmium_mid if osmium_mid is not None else osmium_last_mid
        return result, 0, jsonpickle.encode(data)
