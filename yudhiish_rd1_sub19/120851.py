from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import statistics
import jsonpickle


class Trader:
    """
    Round 2 Strategy
    ============================================
    Assets: ASH_COATED_OSMIUM (Pegged/Stationary), INTARIAN_PEPPER_ROOT (Trend + Residual)
    """

    FAIR_VALUE = {"ASH_COATED_OSMIUM": 10_000}
    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    BASE_QUOTE_SIZE = {"ASH_COATED_OSMIUM": 22}
    VOLUME_SKEW_AGGRESSION = {"ASH_COATED_OSMIUM": 1}
    EMERGENCY_THRESHOLD = {"ASH_COATED_OSMIUM": 70}
    EMERGENCY_TARGET = {"ASH_COATED_OSMIUM": 40}
    KILL_SWITCH_THRESHOLD = {"ASH_COATED_OSMIUM": 80}

    PEPPER_SLOPE = 0.001
    PEPPER_ADJUSTED_HISTORY_LENGTH = 25
    PEPPER_SIGNAL_THRESHOLD = 1.9
    PEPPER_INVENTORY_SKEW = 0.08
    PEPPER_EMERGENCY_THRESHOLD = 65
    PEPPER_EMERGENCY_TARGET = 45

    PEPPER_WARMUP_TICKS = 2
    PEPPER_WARMUP_DISTANCE = 5
    PEPPER_WARMUP_SIZE = 4
    PEPPER_PASSIVE_DISTANCE = 2
    PEPPER_PASSIVE_SIZE = 6

    def bid(self) -> int:
        return 15

    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_adjusted_history": [],
            "pepper_base_estimate": None,
            "pepper_slope_estimate": self.PEPPER_SLOPE,
        }
        if not raw:
            return default_data
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default_data
        if not isinstance(data, dict):
            return default_data

        adjusted_history = data.get("pepper_adjusted_history", [])
        if not isinstance(adjusted_history, list):
            adjusted_history = []
        cleaned_adjusted = []
        for value in adjusted_history:
            if isinstance(value, (int, float)):
                cleaned_adjusted.append(float(value))
        cleaned_adjusted = cleaned_adjusted[-self.PEPPER_ADJUSTED_HISTORY_LENGTH :]

        base_estimate = data.get("pepper_base_estimate")
        if not isinstance(base_estimate, (int, float)):
            base_estimate = None
        else:
            base_estimate = float(base_estimate)

        slope_estimate = data.get("pepper_slope_estimate")
        if not isinstance(slope_estimate, (int, float)):
            slope_estimate = self.PEPPER_SLOPE
        else:
            slope_estimate = float(slope_estimate)

        return {
            "pepper_adjusted_history": cleaned_adjusted,
            "pepper_base_estimate": base_estimate,
            "pepper_slope_estimate": slope_estimate,
        }

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

    def _pepper_size_plan(self, signal_abs: float) -> tuple:
        if signal_abs >= 12:
            return 50, 22, 2
        if signal_abs >= 8:
            return 30, 15, 2
        if signal_abs >= 5:
            return 20, 12, 1
        return 10, 10, 1

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

    def _pepper_target_position(self, timestamp: int) -> int:
        limit = self.POSITION_LIMIT["INTARIAN_PEPPER_ROOT"]
        if timestamp < 200000:
            return limit - 2
        if timestamp < 700000:
            return limit - 8
        return limit - 16

    def _pepper_floor_position(self, timestamp: int) -> int:
        limit = self.POSITION_LIMIT["INTARIAN_PEPPER_ROOT"]
        if timestamp < 200000:
            return limit - 4
        if timestamp < 700000:
            return limit - 12
        return limit - 28

    def _pepper_base_estimate(self, adjusted_history: list, current_mid, timestamp: int, stored_base=None):
        if adjusted_history:
            return float(statistics.median(adjusted_history))
        if isinstance(stored_base, (int, float)):
            return float(stored_base)
        if current_mid is None:
            return None
        return float(current_mid) - self.PEPPER_SLOPE * float(timestamp)

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

    def _trade_pepper_root(self, state: TradingState, adjusted_history: list, base_estimate) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        current_mid = self._mid_price(depth)
        if current_mid is None:
            return []

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0
        best_bid, best_ask = self._best_bid_ask(depth)

        bb = best_bid if best_bid != 0 else None
        ba = best_ask if best_ask != 0 else None
        rounded_mid = int(round(current_mid))

        if position > self.PEPPER_EMERGENCY_THRESHOLD:
            flatten_qty = position - self.PEPPER_EMERGENCY_TARGET
            emergency_price = best_bid if best_bid != 0 else rounded_mid
            self._place_sell(orders, product, emergency_price, flatten_qty, position, pending_sells)
            return orders

        if position < -self.PEPPER_EMERGENCY_THRESHOLD:
            flatten_qty = abs(position) - self.PEPPER_EMERGENCY_TARGET
            emergency_price = best_ask if best_ask != 0 else rounded_mid
            self._place_buy(orders, product, emergency_price, flatten_qty, position, pending_buys)
            return orders

        if len(adjusted_history) < self.PEPPER_WARMUP_TICKS or base_estimate is None:
            pending_buys = self._place_buy(
                orders,
                product,
                rounded_mid - self.PEPPER_WARMUP_DISTANCE,
                self.PEPPER_WARMUP_SIZE,
                position,
                pending_buys,
            )
            pending_sells = self._place_sell(
                orders,
                product,
                rounded_mid + self.PEPPER_WARMUP_DISTANCE,
                self.PEPPER_WARMUP_SIZE,
                position,
                pending_sells,
            )
            return orders

        timestamp = state.timestamp
        fair = float(base_estimate) + self.PEPPER_SLOPE * float(timestamp)
        signal = float(current_mid) - fair
        target_position = self._pepper_target_position(timestamp)
        target_gap = target_position - position
        floor_position = self._pepper_floor_position(timestamp)
        floor_gap = floor_position - position

        inventory_shift = max(-1.0, min(1.0, self.PEPPER_INVENTORY_SKEW * target_gap))
        buy_threshold = max(0.5, self.PEPPER_SIGNAL_THRESHOLD - inventory_shift)
        sell_threshold = max(0.5, self.PEPPER_SIGNAL_THRESHOLD + inventory_shift)

        if floor_gap > 0:
            if floor_gap >= 20:
                core_buy_max = int(fair) + 2
            elif floor_gap >= 8:
                core_buy_max = int(fair) + 1
            else:
                core_buy_max = int(fair)

            core_take_qty = min(floor_gap, 35)
            pending_buys = self._take_asks(
                orders,
                product,
                depth,
                core_buy_max,
                position,
                pending_buys,
                max_total=core_take_qty,
            )

            if position + pending_buys - pending_sells < floor_position:
                build_quote_qty = min(18, max(8, floor_gap))
                build_bid = self._inside_bid(bb, ba, 1, core_buy_max)
                build_bid = min(build_bid, core_buy_max)
                pending_buys = self._place_buy(
                    orders,
                    product,
                    build_bid,
                    build_quote_qty,
                    position,
                    pending_buys,
                )

        if signal <= -buy_threshold:
            take_qty, quote_qty, inside_ticks = self._pepper_size_plan(
                max(abs(signal), float(self.PEPPER_SIGNAL_THRESHOLD))
            )
            max_buy_price = int(fair) if target_gap > 0 else int(fair) - 1
            pending_buys = self._take_asks(
                orders,
                product,
                depth,
                max_buy_price,
                position,
                pending_buys,
                max_total=take_qty,
            )

            aggressive_bid = self._inside_bid(bb, ba, inside_ticks, max_buy_price)
            aggressive_bid = min(aggressive_bid, max_buy_price)
            pending_buys = self._place_buy(
                orders,
                product,
                aggressive_bid,
                quote_qty,
                position,
                pending_buys,
            )

        elif signal >= sell_threshold and (position > floor_position or signal >= sell_threshold + 3):
            take_qty, quote_qty, inside_ticks = self._pepper_size_plan(
                max(abs(signal), float(self.PEPPER_SIGNAL_THRESHOLD))
            )
            min_sell_price = int(fair) + 1 if target_gap > 0 else int(fair)
            pending_sells = self._take_bids(
                orders,
                product,
                depth,
                min_sell_price,
                position,
                pending_sells,
                max_total=take_qty,
            )

            aggressive_ask = self._inside_ask(bb, ba, inside_ticks, min_sell_price)
            aggressive_ask = max(aggressive_ask, min_sell_price)
            pending_sells = self._place_sell(
                orders,
                product,
                aggressive_ask,
                quote_qty,
                position,
                pending_sells,
            )

        else:
            fair_center = int(round(fair))
            bid_distance = self.PEPPER_PASSIVE_DISTANCE
            ask_distance = self.PEPPER_PASSIVE_DISTANCE
            bid_size = self.PEPPER_PASSIVE_SIZE
            ask_size = self.PEPPER_PASSIVE_SIZE
            bid_inside = 0
            ask_inside = 0

            if position < floor_position:
                if target_gap > 3:
                    bid_distance = max(1, bid_distance - 1)
                    bid_size += 2
                    bid_inside = 1
                if floor_gap > 8:
                    bid_distance = max(1, bid_distance - 1)
                    bid_size += 2
                    bid_inside = 1

                bid_price = self._inside_bid(bb, ba, bid_inside, fair_center - bid_distance)
                bid_price = min(bid_price, int(fair))
                pending_buys = self._place_buy(
                    orders,
                    product,
                    bid_price,
                    bid_size,
                    position,
                    pending_buys,
                )
            else:
                if target_gap > 3:
                    bid_distance = max(1, bid_distance - 1)
                    ask_distance += 1
                    bid_size += 2
                    ask_size = max(2, ask_size - 2)
                    bid_inside = 1
                elif target_gap < -3:
                    bid_distance += 1
                    ask_distance = max(1, ask_distance - 1)
                    ask_size += 2
                    bid_size = max(2, bid_size - 2)
                    ask_inside = 1

                bid_price = self._inside_bid(bb, ba, bid_inside, fair_center - bid_distance)
                ask_price = self._inside_ask(bb, ba, ask_inside, fair_center + ask_distance)
                bid_price = min(bid_price, int(fair))
                ask_price = max(ask_price, int(fair) + 1)

                pending_buys = self._place_buy(
                    orders,
                    product,
                    bid_price,
                    bid_size,
                    position,
                    pending_buys,
                )
                pending_sells = self._place_sell(
                    orders,
                    product,
                    ask_price,
                    ask_size,
                    position,
                    pending_sells,
                )

        return orders

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        pepper_adjusted_history = data.get("pepper_adjusted_history", [])

        pepper_depth = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        pepper_mid = self._mid_price(pepper_depth) if pepper_depth else None

        if pepper_mid is not None:
            adjusted_value = float(pepper_mid) - self.PEPPER_SLOPE * float(state.timestamp)
            pepper_adjusted_history = (pepper_adjusted_history + [adjusted_value])[-self.PEPPER_ADJUSTED_HISTORY_LENGTH :]

        pepper_base_estimate = self._pepper_base_estimate(
            pepper_adjusted_history,
            pepper_mid,
            state.timestamp,
            data.get("pepper_base_estimate"),
        )

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self._trade_pepper_root(
                state,
                pepper_adjusted_history,
                pepper_base_estimate,
            )

        data["pepper_adjusted_history"] = pepper_adjusted_history
        data["pepper_base_estimate"] = pepper_base_estimate
        data["pepper_slope_estimate"] = self.PEPPER_SLOPE

        return result, 0, jsonpickle.encode(data)