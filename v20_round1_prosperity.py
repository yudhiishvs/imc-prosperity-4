from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import statistics
import jsonpickle


class Trader:
    """
    Round 2 Strategy
    ============================================
    Assets: ASH_COATED_OSMIUM (Pegged/Stationary), INTARIAN_PEPPER_ROOT (Trend Carry)
    """

    FAIR_VALUE = {"ASH_COATED_OSMIUM": 10_000}
    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    BASE_QUOTE_SIZE = {"ASH_COATED_OSMIUM": 30}
    VOLUME_SKEW_AGGRESSION = {"ASH_COATED_OSMIUM": 1}
    EMERGENCY_THRESHOLD = {"ASH_COATED_OSMIUM": 70}
    EMERGENCY_TARGET = {"ASH_COATED_OSMIUM": 40}
    KILL_SWITCH_THRESHOLD = {"ASH_COATED_OSMIUM": 80}

    PEPPER_SLOPE = 0.001
    PEPPER_EMERGENCY_THRESHOLD = 65
    PEPPER_EMERGENCY_TARGET = 45

    def bid(self) -> int:
        return 15

    def _load_data(self, raw: str) -> dict:
        default_data = {"pepper_base": None}
        if not raw:
            return default_data
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default_data
        if not isinstance(data, dict):
            return default_data

        pepper_base = data.get("pepper_base")
        if not isinstance(pepper_base, (int, float)):
            legacy_base = data.get("pepper_base_estimate")
            pepper_base = float(legacy_base) if isinstance(legacy_base, (int, float)) else None
        else:
            pepper_base = float(pepper_base)

        return {"pepper_base": pepper_base}

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
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _buy_room(self, product: str, position: int, pending_buys: int) -> int:
        return self.POSITION_LIMIT.get(product, 20) - (position + pending_buys)

    def _sell_room(self, product: str, position: int, pending_sells: int) -> int:
        return self.POSITION_LIMIT.get(product, 20) + (position - pending_sells)

    def _place_buy(
        self,
        orders: List[Order],
        product: str,
        price: int,
        desired_qty: int,
        position: int,
        pending_buys: int,
    ) -> int:
        room = self._buy_room(product, position, pending_buys)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, qty))
            pending_buys += qty
        return pending_buys

    def _place_sell(
        self,
        orders: List[Order],
        product: str,
        price: int,
        desired_qty: int,
        position: int,
        pending_sells: int,
    ) -> int:
        room = self._sell_room(product, position, pending_sells)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, -qty))
            pending_sells += qty
        return pending_sells

    def _take_asks(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        max_price: int,
        position: int,
        pending_buys: int,
        max_total: int = None,
    ) -> int:
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

    def _take_bids(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        min_price: int,
        position: int,
        pending_sells: int,
        max_total: int = None,
    ) -> int:
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

    def _inside_bid(self, best_bid, best_ask, inside_ticks: int, fallback: int) -> int:
        if best_bid is not None and best_ask is not None:
            spread_space = max(0, best_ask - best_bid - 1)
            if spread_space > 0:
                return best_bid + min(inside_ticks, spread_space)
            return best_bid
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask - 1
        return fallback

    def _inside_ask(self, best_bid, best_ask, inside_ticks: int, fallback: int) -> int:
        if best_bid is not None and best_ask is not None:
            spread_space = max(0, best_ask - best_bid - 1)
            if spread_space > 0:
                return best_ask - min(inside_ticks, spread_space)
            return best_ask
        if best_ask is not None:
            return best_ask
        if best_bid is not None:
            return best_bid + 1
        return fallback

    def _take_mispriced(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        fair = self.FAIR_VALUE[product]
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

    def _flatten_at_fair(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        fair = self.FAIR_VALUE[product]
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

    def _penny_jump_quotes(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        fair = self.FAIR_VALUE[product]
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid == 0 or best_ask == 0:
            return pending_buys, pending_sells

        penny_bid = best_bid + 1
        penny_ask = best_ask - 1
        penny_bid = min(penny_bid, fair - 1)
        penny_ask = max(penny_ask, fair + 1)

        projected = position + pending_buys - pending_sells
        position_ratio = projected / self.POSITION_LIMIT[product]
        bid_scale = max(0.0, 1.0 - max(0.0, position_ratio) * self.VOLUME_SKEW_AGGRESSION[product])
        ask_scale = max(0.0, 1.0 + min(0.0, position_ratio) * self.VOLUME_SKEW_AGGRESSION[product])

        bid_qty = int(round(self.BASE_QUOTE_SIZE[product] * bid_scale))
        ask_qty = int(round(self.BASE_QUOTE_SIZE[product] * ask_scale))

        if projected >= self.KILL_SWITCH_THRESHOLD[product]:
            bid_qty = 0
        elif projected <= -self.KILL_SWITCH_THRESHOLD[product]:
            ask_qty = 0

        if bid_qty > 0:
            pending_buys = self._place_buy(orders, product, penny_bid, bid_qty, position, pending_buys)
        if ask_qty > 0:
            pending_sells = self._place_sell(orders, product, penny_ask, ask_qty, position, pending_sells)
        return pending_buys, pending_sells

    def _emergency_flatten(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[bool, int, int]:
        projected = position + pending_buys - pending_sells
        if abs(projected) <= self.EMERGENCY_THRESHOLD[product]:
            return False, pending_buys, pending_sells

        best_bid, best_ask = self._best_bid_ask(depth)
        if projected > self.EMERGENCY_THRESHOLD[product]:
            flatten_qty = projected - self.EMERGENCY_TARGET[product]
            if best_bid > 0 and flatten_qty > 0:
                pending_sells = self._place_sell(orders, product, best_bid, flatten_qty, position, pending_sells)
            return True, pending_buys, pending_sells
        if projected < -self.EMERGENCY_THRESHOLD[product]:
            flatten_qty = abs(projected) - self.EMERGENCY_TARGET[product]
            if best_ask > 0 and flatten_qty > 0:
                pending_buys = self._place_buy(orders, product, best_ask, flatten_qty, position, pending_buys)
            return True, pending_buys, pending_sells
        return False, pending_buys, pending_sells

    def _trade_pegged_asset(self, state: TradingState, product: str) -> List[Order]:
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pb = 0
        ps = 0

        pb, ps = self._take_mispriced(orders, product, depth, position, pb, ps)
        pb, ps = self._flatten_at_fair(orders, product, depth, position, pb, ps)
        triggered, pb, ps = self._emergency_flatten(orders, product, depth, position, pb, ps)

        if not triggered:
            pb, ps = self._penny_jump_quotes(orders, product, depth, position, pb, ps)

        return orders

    def _trade_osmium(self, state: TradingState) -> List[Order]:
        return self._trade_pegged_asset(state, "ASH_COATED_OSMIUM")

    def _trade_pepper_root(self, state: TradingState, fair_value) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        if depth is None or fair_value is None:
            return []

        orders: List[Order] = []
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMIT[product]
        pending_buys = 0
        pending_sells = 0
        best_bid, _ = self._best_bid_ask(depth)

        if position > limit:
            excess = position - limit
            sell_price = best_bid if best_bid != 0 else int(fair_value)
            self._place_sell(orders, product, sell_price, excess, position, pending_sells)
            return orders

        deficit = limit - position
        if deficit <= 0:
            return orders

        max_buy_price = int(fair_value) + 3
        pending_buys = self._take_asks(
            orders,
            product,
            depth,
            max_buy_price,
            position,
            pending_buys,
            max_total=deficit,
        )

        remaining = limit - (position + pending_buys)
        if remaining > 0:
            first_bid_qty = (remaining + 1) // 2
            second_bid_qty = remaining - first_bid_qty
            pending_buys = self._place_buy(
                orders,
                product,
                int(fair_value),
                first_bid_qty,
                position,
                pending_buys,
            )
            if second_bid_qty > 0:
                pending_buys = self._place_buy(
                    orders,
                    product,
                    int(fair_value) - 1,
                    second_bid_qty,
                    position,
                    pending_buys,
                )

        return orders

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        pepper_base = data.get("pepper_base")

        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        pepper_mid = self._mid_price(pepper_depth) if pepper_depth else None
        timestamp = state.timestamp

        if pepper_base is None and pepper_mid is not None:
            pepper_base = float(pepper_mid) - self.PEPPER_SLOPE * float(timestamp)

        pepper_fair = None if pepper_base is None else pepper_base + self.PEPPER_SLOPE * float(timestamp)

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self._trade_pepper_root(state, pepper_fair)

        data["pepper_base"] = pepper_base
        return result, 0, jsonpickle.encode(data)
