from datamodel import Order, OrderDepth, TradingState
from typing import Callable, Dict, List, Optional, Tuple
import json
import math
import statistics


class Trader:
    POSITION_LIMIT = {
        "ASH_COATED_OSMIUM": 20,
        "INTARIAN_PEPPER_ROOT": 20,
    }

    ASH = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"
    ASH_FAIR_VALUE = 10000
    PEPPER_HISTORY_KEY = "pepper_adjusted_mids"
    HISTORY_LIMIT = 50

    def load_data(self, raw: str) -> Dict[str, List[float]]:
        if not raw:
            return {self.PEPPER_HISTORY_KEY: []}
        try:
            data = json.loads(raw)
        except Exception:
            return {self.PEPPER_HISTORY_KEY: []}
        if not isinstance(data, dict):
            return {self.PEPPER_HISTORY_KEY: []}

        history = data.get(self.PEPPER_HISTORY_KEY, [])
        if not isinstance(history, list):
            history = []

        cleaned_history: List[float] = []
        for value in history:
            if isinstance(value, (int, float)):
                cleaned_history.append(float(value))

        return {self.PEPPER_HISTORY_KEY: cleaned_history[-self.HISTORY_LIMIT :]}

    def dump_data(self, data: Dict[str, List[float]]) -> str:
        return json.dumps({self.PEPPER_HISTORY_KEY: data.get(self.PEPPER_HISTORY_KEY, [])[-self.HISTORY_LIMIT :]})

    def get_sorted_books(self, order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        sell_levels: List[Tuple[int, int]] = []
        buy_levels: List[Tuple[int, int]] = []

        if order_depth is None:
            return sell_levels, buy_levels

        if getattr(order_depth, "sell_orders", None):
            sell_levels = sorted(order_depth.sell_orders.items(), key=lambda level: level[0])

        if getattr(order_depth, "buy_orders", None):
            buy_levels = sorted(order_depth.buy_orders.items(), key=lambda level: level[0], reverse=True)

        return sell_levels, buy_levels

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        sell_levels, buy_levels = self.get_sorted_books(order_depth)
        best_ask = sell_levels[0][0] if sell_levels else None
        best_bid = buy_levels[0][0] if buy_levels else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth, fallback: Optional[float] = None) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return fallback

    def compute_imbalance(self, order_depth: OrderDepth) -> float:
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return 0.0

        buy_orders = getattr(order_depth, "buy_orders", {}) or {}
        sell_orders = getattr(order_depth, "sell_orders", {}) or {}
        best_bid_volume = abs(buy_orders.get(best_bid, 0))
        best_ask_volume = abs(sell_orders.get(best_ask, 0))
        denominator = best_bid_volume + best_ask_volume

        if denominator <= 0:
            return 0.0
        return (best_bid_volume - best_ask_volume) / denominator

    def clamp_order_size_to_limit(
        self,
        product: str,
        position: int,
        pending_buys: int,
        pending_sells: int,
        desired_quantity: int,
    ) -> int:
        limit = self.POSITION_LIMIT[product]
        projected_position = position + pending_buys - pending_sells

        if desired_quantity > 0:
            if projected_position >= limit:
                return 0
            return max(0, min(desired_quantity, limit - projected_position))

        if desired_quantity < 0:
            if projected_position <= -limit:
                return 0
            return -max(0, min(-desired_quantity, projected_position + limit))

        return 0

    def place_aggressive_orders(
        self,
        product: str,
        orders: List[Order],
        levels: List[Tuple[int, int]],
        side: str,
        position: int,
        pending_buys: int,
        pending_sells: int,
        should_trade: Callable[[int, int], bool],
        target_position: Optional[int] = None,
    ) -> Tuple[int, int]:
        for price, book_volume in levels:
            current_position = position + pending_buys - pending_sells
            if not should_trade(price, current_position):
                continue

            visible_size = abs(book_volume)
            if visible_size <= 0:
                continue

            if side == "buy":
                desired_size = visible_size
                if target_position is not None:
                    desired_size = min(desired_size, max(0, target_position - current_position))
                signed_quantity = self.clamp_order_size_to_limit(
                    product, position, pending_buys, pending_sells, desired_size
                )
                if signed_quantity > 0:
                    orders.append(Order(product, int(price), int(signed_quantity)))
                    pending_buys += signed_quantity
            else:
                desired_size = visible_size
                if target_position is not None:
                    desired_size = min(desired_size, max(0, current_position - target_position))
                signed_quantity = self.clamp_order_size_to_limit(
                    product, position, pending_buys, pending_sells, -desired_size
                )
                if signed_quantity < 0:
                    orders.append(Order(product, int(price), int(signed_quantity)))
                    pending_sells += -signed_quantity

        return pending_buys, pending_sells

    def _scaled_capacity(self, capacity: int, scale: float) -> int:
        if capacity <= 0 or scale <= 0:
            return 0
        scaled = int(math.floor(capacity * scale))
        return min(capacity, max(1, scaled))

    def _split_size(self, total: int, inner_ratio: float) -> Tuple[int, int]:
        if total <= 0:
            return 0, 0
        inner = int(math.floor(total * inner_ratio))
        inner = min(total, max(1, inner))
        outer = total - inner
        return inner, outer

    def _make_passive_bid_price(
        self,
        base_price: int,
        reservation_price: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[int]:
        reservation_cap = int(math.floor(reservation_price - 1))
        price = min(int(base_price), reservation_cap)

        if best_bid is not None:
            price = max(price, min(best_bid + 1, reservation_cap))
        if best_ask is not None:
            price = min(price, best_ask - 1)

        if best_ask is not None and price >= best_ask:
            return None
        if price > reservation_cap:
            return None
        return int(price)

    def _make_passive_ask_price(
        self,
        base_price: int,
        reservation_price: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[int]:
        reservation_floor = int(math.ceil(reservation_price + 1))
        price = max(int(base_price), reservation_floor)

        if best_ask is not None:
            price = min(price, max(best_ask - 1, reservation_floor))
        if best_bid is not None:
            price = max(price, best_bid + 1)

        if best_bid is not None and price <= best_bid:
            return None
        if price < reservation_floor:
            return None
        return int(price)

    def place_layered_quotes(
        self,
        product: str,
        orders: List[Order],
        position: int,
        pending_buys: int,
        pending_sells: int,
        reservation_price: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
        inner_bid_width: int,
        outer_bid_width: int,
        inner_ask_width: int,
        outer_ask_width: int,
        buy_inner_ratio: float,
        sell_inner_ratio: float,
        buy_scale: float,
        sell_scale: float,
        reduce_only_threshold: int,
    ) -> Tuple[int, int]:
        projected_position = position + pending_buys - pending_sells
        limit = self.POSITION_LIMIT[product]

        allow_buys = True
        allow_sells = True
        if abs(projected_position) >= reduce_only_threshold:
            allow_buys = projected_position < 0
            allow_sells = projected_position > 0

        buy_capacity = max(0, limit - projected_position) if allow_buys else 0
        sell_capacity = max(0, limit + projected_position) if allow_sells else 0

        buy_total = self._scaled_capacity(buy_capacity, buy_scale)
        sell_total = self._scaled_capacity(sell_capacity, sell_scale)

        inner_buy_size, outer_buy_size = self._split_size(buy_total, buy_inner_ratio)
        inner_sell_size, outer_sell_size = self._split_size(sell_total, sell_inner_ratio)

        quote_buys: Dict[int, int] = {}
        quote_sells: Dict[int, int] = {}

        inner_bid_base = int(math.floor(reservation_price - inner_bid_width))
        outer_bid_base = int(math.floor(reservation_price - outer_bid_width))
        inner_ask_base = int(math.ceil(reservation_price + inner_ask_width))
        outer_ask_base = int(math.ceil(reservation_price + outer_ask_width))

        inner_bid_price = self._make_passive_bid_price(inner_bid_base, reservation_price, best_bid, best_ask)
        outer_bid_price = self._make_passive_bid_price(outer_bid_base, reservation_price, best_bid, best_ask)
        inner_ask_price = self._make_passive_ask_price(inner_ask_base, reservation_price, best_bid, best_ask)
        outer_ask_price = self._make_passive_ask_price(outer_ask_base, reservation_price, best_bid, best_ask)

        if inner_buy_size > 0 and inner_bid_price is not None:
            quote_buys[inner_bid_price] = quote_buys.get(inner_bid_price, 0) + inner_buy_size
        if outer_buy_size > 0 and outer_bid_price is not None:
            quote_buys[outer_bid_price] = quote_buys.get(outer_bid_price, 0) + outer_buy_size
        if inner_sell_size > 0 and inner_ask_price is not None:
            quote_sells[inner_ask_price] = quote_sells.get(inner_ask_price, 0) + inner_sell_size
        if outer_sell_size > 0 and outer_ask_price is not None:
            quote_sells[outer_ask_price] = quote_sells.get(outer_ask_price, 0) + outer_sell_size

        for price in sorted(quote_buys.keys(), reverse=True):
            signed_quantity = self.clamp_order_size_to_limit(
                product, position, pending_buys, pending_sells, quote_buys[price]
            )
            if signed_quantity > 0:
                orders.append(Order(product, int(price), int(signed_quantity)))
                pending_buys += signed_quantity

        for price in sorted(quote_sells.keys()):
            signed_quantity = self.clamp_order_size_to_limit(
                product, position, pending_buys, pending_sells, -quote_sells[price]
            )
            if signed_quantity < 0:
                orders.append(Order(product, int(price), int(signed_quantity)))
                pending_sells += -signed_quantity

        return pending_buys, pending_sells

    def _trade_ash(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.ASH)
        if depth is None:
            return []

        orders: List[Order] = []
        position = state.position.get(self.ASH, 0)
        pending_buys = 0
        pending_sells = 0

        sell_levels, buy_levels = self.get_sorted_books(depth)
        best_bid, best_ask = self.get_best_bid_ask(depth)
        imbalance = self.compute_imbalance(depth)

        pending_buys, pending_sells = self.place_aggressive_orders(
            self.ASH,
            orders,
            sell_levels,
            "buy",
            position,
            pending_buys,
            pending_sells,
            lambda price, current_position: price < self.ASH_FAIR_VALUE
            or (price == self.ASH_FAIR_VALUE and current_position < 0),
        )
        pending_buys, pending_sells = self.place_aggressive_orders(
            self.ASH,
            orders,
            buy_levels,
            "sell",
            position,
            pending_buys,
            pending_sells,
            lambda price, current_position: price > self.ASH_FAIR_VALUE
            or (price == self.ASH_FAIR_VALUE and current_position > 0),
        )

        projected_position = position + pending_buys - pending_sells
        reservation_price = self.ASH_FAIR_VALUE + round(6 * imbalance) - 0.25 * projected_position

        buy_scale = 1.0
        sell_scale = 1.0
        limit = self.POSITION_LIMIT[self.ASH]
        if projected_position > 0:
            buy_scale = max(0.25, 1.0 - (projected_position / float(limit)))
        elif projected_position < 0:
            sell_scale = max(0.25, 1.0 - (abs(projected_position) / float(limit)))

        self.place_layered_quotes(
            self.ASH,
            orders,
            position,
            pending_buys,
            pending_sells,
            reservation_price,
            best_bid,
            best_ask,
            inner_bid_width=2,
            outer_bid_width=4,
            inner_ask_width=2,
            outer_ask_width=4,
            buy_inner_ratio=0.60,
            sell_inner_ratio=0.60,
            buy_scale=buy_scale,
            sell_scale=sell_scale,
            reduce_only_threshold=limit - 2,
        )

        return orders

    def _trade_pepper(self, state: TradingState, pepper_fair: Optional[float]) -> List[Order]:
        depth = state.order_depths.get(self.PEPPER)
        if depth is None or pepper_fair is None:
            return []

        orders: List[Order] = []
        position = state.position.get(self.PEPPER, 0)
        pending_buys = 0
        pending_sells = 0

        sell_levels, buy_levels = self.get_sorted_books(depth)
        best_bid, best_ask = self.get_best_bid_ask(depth)
        imbalance = self.compute_imbalance(depth)

        limit = self.POSITION_LIMIT[self.PEPPER]
        timestamp = getattr(state, "timestamp", 0)
        if timestamp < 70000:
            target_position = limit - 2
        elif timestamp < 85000:
            target_position = limit - 6
        else:
            target_position = limit - 10
        lower_buy_threshold = int(math.floor(pepper_fair - 1))
        higher_sell_threshold = int(math.ceil(pepper_fair + 2))
        target_buy_threshold = int(math.ceil(pepper_fair + 2))
        target_sell_threshold = int(math.floor(pepper_fair))

        pending_buys, pending_sells = self.place_aggressive_orders(
            self.PEPPER,
            orders,
            sell_levels,
            "buy",
            position,
            pending_buys,
            pending_sells,
            lambda price, current_position: price <= lower_buy_threshold,
        )
        pending_buys, pending_sells = self.place_aggressive_orders(
            self.PEPPER,
            orders,
            buy_levels,
            "sell",
            position,
            pending_buys,
            pending_sells,
            lambda price, current_position: price >= higher_sell_threshold,
        )

        if position + pending_buys - pending_sells < target_position:
            pending_buys, pending_sells = self.place_aggressive_orders(
                self.PEPPER,
                orders,
                sell_levels,
                "buy",
                position,
                pending_buys,
                pending_sells,
                lambda price, current_position: price > lower_buy_threshold and price <= target_buy_threshold and current_position < target_position,
                target_position=target_position,
            )

        if position + pending_buys - pending_sells > target_position + 2:
            pending_buys, pending_sells = self.place_aggressive_orders(
                self.PEPPER,
                orders,
                buy_levels,
                "sell",
                position,
                pending_buys,
                pending_sells,
                lambda price, current_position: price >= target_sell_threshold and price < higher_sell_threshold and current_position > target_position,
                target_position=target_position,
            )

        projected_position = position + pending_buys - pending_sells
        reservation_price = pepper_fair + 3 * imbalance + 0.30 * (target_position - projected_position)
        inner_ask_width = 3 if position > 10 else 4
        outer_ask_width = 5 if position > 10 else 6

        if projected_position >= target_position:
            buy_scale = 0.5
            sell_scale = 1.0
        elif projected_position <= target_position - 4:
            buy_scale = 1.0
            sell_scale = 0.5
        else:
            buy_scale = 0.8
            sell_scale = 1.0

        self.place_layered_quotes(
            self.PEPPER,
            orders,
            position,
            pending_buys,
            pending_sells,
            reservation_price,
            best_bid,
            best_ask,
            inner_bid_width=2,
            outer_bid_width=4,
            inner_ask_width=inner_ask_width,
            outer_ask_width=outer_ask_width,
            buy_inner_ratio=0.70,
            sell_inner_ratio=0.50,
            buy_scale=buy_scale,
            sell_scale=sell_scale,
            reduce_only_threshold=limit - 1,
        )

        return orders

    def run(self, state: TradingState):
        data = self.load_data(getattr(state, "traderData", ""))
        history = data.get(self.PEPPER_HISTORY_KEY, [])

        pepper_depth = state.order_depths.get(self.PEPPER)
        pepper_mid = self.get_mid_price(pepper_depth, fallback=None) if pepper_depth is not None else None

        timestamp = getattr(state, "timestamp", 0)
        updated_history = list(history)
        if pepper_mid is not None:
            updated_history.append(float(pepper_mid) - 0.001 * float(timestamp))
            updated_history = updated_history[-self.HISTORY_LIMIT :]

        if history:
            pepper_base = float(statistics.median(updated_history))
        elif pepper_mid is not None:
            pepper_base = float(round(float(pepper_mid) / 1000.0) * 1000)
        elif updated_history:
            pepper_base = float(statistics.median(updated_history))
        else:
            pepper_base = None

        pepper_fair = None if pepper_base is None else pepper_base + 0.001 * float(timestamp)

        result: Dict[str, List[Order]] = {
            self.ASH: self._trade_ash(state),
            self.PEPPER: self._trade_pepper(state, pepper_fair),
        }

        data[self.PEPPER_HISTORY_KEY] = updated_history[-self.HISTORY_LIMIT :]
        return result, 0, self.dump_data(data)