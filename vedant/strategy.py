from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle


class Trader:
    """
    ============================================
    ASH_COATED_OSMIUM: baseline market making (static fair, penny jump, no OIM / no inventory skew).
    INTARIAN_PEPPER_ROOT: detrended long bias + capped scalp sells into bid spikes above FV.
    """

    # ── Osmium: baseline MM (see osmium_basic_mm.py) ───────────
    OSMIUM_POSITION_LIMIT = 80
    OSMIUM_FAIR_VALUE = 10_000
    OSMIUM_BASE_QUOTE_SIZE = 10

    # ── Pepper (aligned with best_strat_logs; tune PEPPER_MAX_SCALP_VOLUME) ──
    PEPPER_SLOPE = 0.001
    PEPPER_POSITION_LIMIT = 80
    PEPPER_INITIAL_ACC_THRESH = 9
    PEPPER_SCALP_MIN_MARGIN = 4
    PEPPER_MAX_SCALP_VOLUME = 5  # max units sold per tick into bid(s) at/above scalp threshold; sweep 1..80
    PEPPER_RECOUP_MAX_MARGIN = 3

    def bid(self) -> int:
        return 15

    # ── State Persistence ────────────────────────────────────
    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_base_estimate": None,
            "pepper_reached_80": False,
        }
        if not raw:
            return default_data
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default_data
        if not isinstance(data, dict):
            return default_data

        base_estimate = data.get("pepper_base_estimate")
        if not isinstance(base_estimate, (int, float)):
            base_estimate = None
        else:
            base_estimate = float(base_estimate)

        pepper_reached_80 = data.get("pepper_reached_80", False)
        if not isinstance(pepper_reached_80, bool):
            pepper_reached_80 = False

        return {
            "pepper_base_estimate": base_estimate,
            "pepper_reached_80": pepper_reached_80,
        }

    # ── Shared Helpers ───────────────────────────────────────
    def _get_position_limit(self, product: str) -> int:
        if product == "INTARIAN_PEPPER_ROOT":
            return self.PEPPER_POSITION_LIMIT
        if product == "ASH_COATED_OSMIUM":
            return self.OSMIUM_POSITION_LIMIT
        return 20

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        return best_bid, best_ask

    @staticmethod
    def _mid_price(depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _buy_room(self, product: str, position: int, pending_buys: int) -> int:
        return self._get_position_limit(product) - (position + pending_buys)

    def _sell_room(self, product: str, position: int, pending_sells: int) -> int:
        return self._get_position_limit(product) + (position - pending_sells)

    def _place_buy(self, orders: List[Order], product: str, price: int, desired_qty: int, position: int, pending_buys: int) -> int:
        room = self._buy_room(product, position, pending_buys)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, qty))
            pending_buys += qty
        return pending_buys

    def _place_sell(self, orders: List[Order], product: str, price: int, desired_qty: int, position: int, pending_sells: int) -> int:
        room = self._sell_room(product, position, pending_sells)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, -qty))
            pending_sells += qty
        return pending_sells

    def _take_asks(self, orders: List[Order], product: str, depth: OrderDepth, max_price: int, position: int, pending_buys: int, max_total: int = None) -> int:
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

    def _take_bids(self, orders: List[Order], product: str, depth: OrderDepth, min_price: int, position: int, pending_sells: int, max_total: int = None) -> int:
        sold = 0
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid < min_price or (max_total is not None and sold >= max_total):
                break
            room = self._sell_room(product, position, pending_sells)
            if room <= 0:
                break
            size = min(depth.buy_orders[bid], room)
            if max_total is not None:
                size = min(size, max_total - sold)
            if size <= 0:
                continue
            orders.append(Order(product, bid, -size))
            pending_sells += size
            sold += size
        return pending_sells

    # ── Arbitrage & Quoting Helpers ──────────────────────────
    def _take_mispriced(self, orders: List[Order], product: str, depth: OrderDepth, position: int, pending_buys: int, pending_sells: int, fair: int) -> Tuple[int, int]:
        if not depth.sell_orders and not depth.buy_orders:
            return pending_buys, pending_sells

        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price >= fair:
                break
            ask_vol = -depth.sell_orders[ask_price]
            pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)

        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price <= fair:
                break
            bid_vol = depth.buy_orders[bid_price]
            pending_sells = self._place_sell(orders, product, bid_price, bid_vol, position, pending_sells)

        return pending_buys, pending_sells

    def _flatten_at_fair(self, orders: List[Order], product: str, depth: OrderDepth, position: int, pending_buys: int, pending_sells: int, fair: int) -> Tuple[int, int]:
        projected = position + pending_buys - pending_sells
        if projected > 0 and fair in depth.buy_orders:
            bid_vol = depth.buy_orders[fair]
            flatten_qty = min(bid_vol, projected)
            pending_sells = self._place_sell(orders, product, fair, flatten_qty, position, pending_sells)
        elif projected < 0 and fair in depth.sell_orders:
            ask_vol = -depth.sell_orders[fair]
            flatten_qty = min(ask_vol, abs(projected))
            pending_buys = self._place_buy(orders, product, fair, flatten_qty, position, pending_buys)
        return pending_buys, pending_sells

    # ── Trending Logic (PEPPER_ROOT) ─────────────────────────
    def _pepper_base_estimate(self, current_mid, timestamp: int, stored_base=None):
        if isinstance(stored_base, (int, float)):
            return float(stored_base)
        if current_mid is None:
            return None
        return float(current_mid) - self.PEPPER_SLOPE * float(timestamp)

    # ── Asset Execution Pipelines ────────────────────────────
    def _trade_osmium(self, state: TradingState) -> List[Order]:
        """Static fair + misprice takes + flatten-at-fair + symmetric penny jump (no OIM, no inventory skew)."""
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None:
            return []
        if not depth.buy_orders and not depth.sell_orders:
            return []

        fair = self.OSMIUM_FAIR_VALUE
        orders: List[Order] = []
        position = state.position.get(product, 0)
        pb = ps = 0

        pb, ps = self._take_mispriced(orders, product, depth, position, pb, ps, fair)
        pb, ps = self._flatten_at_fair(orders, product, depth, position, pb, ps, fair)

        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid == 0 or best_ask == 0:
            return orders

        penny_bid = min(best_bid + 1, fair - 1)
        penny_ask = max(best_ask - 1, fair + 1)
        pb = self._place_buy(orders, product, penny_bid, self.OSMIUM_BASE_QUOTE_SIZE, position, pb)
        ps = self._place_sell(orders, product, penny_ask, self.OSMIUM_BASE_QUOTE_SIZE, position, ps)

        return orders

    def _trade_pepper_root(self, state: TradingState, base_estimate, reached_80: bool) -> Tuple[List[Order], bool]:
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        if depth is None:
            return [], reached_80

        current_mid = self._mid_price(depth)
        if current_mid is None:
            return [], reached_80

        orders: List[Order] = []
        position = state.position.get(product, 0)

        if position >= self.PEPPER_POSITION_LIMIT:
            reached_80 = True

        if base_estimate is None:
            base_estimate = current_mid

        timestamp = state.timestamp
        fair = float(base_estimate) + self.PEPPER_SLOPE * float(timestamp)
        fair_center = int(round(fair))

        pending_buys = 0
        pending_sells = 0

        if reached_80 and position > 0:
            target_sell_price = fair_center + self.PEPPER_SCALP_MIN_MARGIN
            pending_sells = self._take_bids(
                orders,
                product,
                depth,
                target_sell_price,
                position,
                pending_sells,
                max_total=self.PEPPER_MAX_SCALP_VOLUME,
            )

        deficit = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
        if deficit > 0:
            if not reached_80:
                max_buy_price = fair_center + self.PEPPER_INITIAL_ACC_THRESH
                pending_buys = self._take_asks(orders, product, depth, max_buy_price, position, pending_buys, max_total=deficit)

                remaining = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
                if remaining > 0:
                    first_bid_qty = (remaining + 1) // 2
                    second_bid_qty = remaining - first_bid_qty
                    pending_buys = self._place_buy(orders, product, fair_center + 2, first_bid_qty, position, pending_buys)
                    if second_bid_qty > 0:
                        pending_buys = self._place_buy(orders, product, fair_center + 1, second_bid_qty, position, pending_buys)
            else:
                max_buy_price = fair_center + self.PEPPER_RECOUP_MAX_MARGIN
                pending_buys = self._take_asks(orders, product, depth, max_buy_price, position, pending_buys, max_total=deficit)

                remaining = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
                if remaining > 0:
                    bid_post_price = min(fair_center, fair_center + self.PEPPER_RECOUP_MAX_MARGIN)
                    pending_buys = self._place_buy(orders, product, bid_post_price, remaining, position, pending_buys)

        return orders, reached_80

    # ── Main Entry ───────────────────────────────────────────
    def run(self, state: TradingState):
        data = self._load_data(state.traderData)

        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        pepper_mid = self._mid_price(pepper_depth) if pepper_depth else None

        pepper_base_estimate = self._pepper_base_estimate(
            pepper_mid,
            state.timestamp,
            data.get("pepper_base_estimate"),
        )

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            pepper_orders, pepper_reached_80 = self._trade_pepper_root(
                state,
                pepper_base_estimate,
                data.get("pepper_reached_80", False),
            )
            result["INTARIAN_PEPPER_ROOT"] = pepper_orders
            data["pepper_reached_80"] = pepper_reached_80

        data["pepper_base_estimate"] = pepper_base_estimate

        trader_data = jsonpickle.encode(data)
        return result, 0, trader_data
