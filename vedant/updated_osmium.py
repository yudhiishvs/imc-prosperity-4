from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle


class Trader:
    """
    ============================================
    ASH_COATED_OSMIUM: EMA fair + A-S reservation-price inventory layer
                       + momentum-tilted layered quoting.
    INTARIAN_PEPPER_ROOT: detrended long bias + capped scalp sells into bid spikes above FV.
    """

    # ── Osmium: Parameterized configuration ───────────
    FAIR_VALUE = {"ASH_COATED_OSMIUM": 10_000}
    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    BASE_QUOTE_SIZE = {"ASH_COATED_OSMIUM": 40}

    # Fair Value
    OSMIUM_EMA_ALPHA = 0.3

    # Reservation Price (A-S Inventory Cost)
    OSMIUM_INVENTORY_SKEW = 0.06  # ticks shift per unit of position; reservation = ema - pos * this

    # Volume Scaling (Smooth Power Curve — replaces linear skew + kill switch + emergency flatten)
    OSMIUM_SKEW_POWER = 2.0       # shape: 1=linear, 2=quadratic, 3=cubic
    OSMIUM_ACCUM_FLOOR = 0.0      # min accumulation-side scale at position limit (0=kill switch)
    OSMIUM_UNWIND_CEILING = 2.0   # max unwind-side scale at position limit

    # Taking Policy
    OSMIUM_TAKE_UNWIND_WIDTH = 1  # extra ticks past reservation to sweep on unwind side
    OSMIUM_TAKE_ACCUM_WIDTH = 0   # ticks tighter than EMA for accumulation-side taking
    OSMIUM_SYMMETRIC_ZONE = 15    # |position| below this → symmetric EMA-based taking

    # Quote Structure
    OSMIUM_INNER_QUOTE_OFFSET = 0
    OSMIUM_OUTER_QUOTE_OFFSET = 1
    OSMIUM_INNER_QTY_RATIO = 0.9
    OSMIUM_DYNAMIC_OUTER_ANCHOR = True

    # Momentum Fade Skewing
    OSMIUM_MOMENTUM_QUOTE_SHIFT = 4
    OSMIUM_MOMENTUM_AGRESS_SCALE = 1.7
    OSMIUM_MOMENTUM_DEFENSE_SCALE = 1.2

    # ── Pepper (Optimized via Grid Search) ──
    PEPPER_SLOPE = 0.001
    PEPPER_POSITION_LIMIT = 80
    PEPPER_INITIAL_ACC_THRESH = 8
    PEPPER_SCALP_MIN_MARGIN = 4
    PEPPER_MAX_SCALP_VOLUME = 3  # max units sold per tick into bid(s) at/above scalp threshold; sweep 1..80
    PEPPER_RECOUP_MAX_MARGIN = -2
    # Post-reach market making (long-biased): penny-jump both sides, favor bids.
    PEPPER_MM_BASE_QUOTE_SIZE = 15
    PEPPER_MM_BID_WEIGHT = 0.5
    PEPPER_MM_MIN_LONG_POSITION = 77
    # L2 quality gate: only penny-jump when L1 is sufficiently close to L2.
    PEPPER_MM_L2_MAX_BID_GAP = 6  # max |L1 bid - L2 bid|
    PEPPER_MM_L2_MAX_ASK_GAP = 5  # max |L2 ask - L1 ask|
    # OIM Dynamic Shifts
    PEPPER_OIM_BASE_THRESHOLD = 0.0
    PEPPER_OIM_MAX_SHIFT = 2

    def bid(self) -> int:
        return 15

    # ── State Persistence ────────────────────────────────────
    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_base_estimate": None,
            "pepper_reached_80": False,
            "osmium_ema": None,
            "osmium_last_mid": None,
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

        osmium_ema = data.get("osmium_ema")
        if not isinstance(osmium_ema, (int, float)):
            osmium_ema = None
        else:
            osmium_ema = float(osmium_ema)

        osmium_last_mid = data.get("osmium_last_mid")
        if not isinstance(osmium_last_mid, (int, float)):
            osmium_last_mid = None
        else:
            osmium_last_mid = float(osmium_last_mid)

        return {
            "pepper_base_estimate": base_estimate,
            "pepper_reached_80": pepper_reached_80,
            "osmium_ema": osmium_ema,
            "osmium_last_mid": osmium_last_mid,
        }

    # ── Shared Helpers ───────────────────────────────────────
    def _get_position_limit(self, product: str) -> int:
        if product == "INTARIAN_PEPPER_ROOT":
            return self.PEPPER_POSITION_LIMIT
        return self.POSITION_LIMIT.get(product, 20)

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        return best_bid, best_ask

    @staticmethod
    def _second_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None]:
        bid2 = None
        ask2 = None
        if len(depth.buy_orders) >= 2:
            bid2 = sorted(depth.buy_orders.keys(), reverse=True)[1]
        if len(depth.sell_orders) >= 2:
            ask2 = sorted(depth.sell_orders.keys())[1]
        return bid2, ask2

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

    # ── Arbitrage & Quoting Helpers ──────────────────────────
    def _take_mispriced(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
        fair_value=None,
        buy_inclusive: bool = False,
        sell_inclusive: bool = False,
    ) -> Tuple[int, int]:
        if not depth.sell_orders and not depth.buy_orders:
            return pending_buys, pending_sells
        fair = self.FAIR_VALUE[product] if fair_value is None else fair_value

        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price > fair or (ask_price == fair and not buy_inclusive):
                break
            ask_vol = -depth.sell_orders[ask_price]
            pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)

        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price < fair or (bid_price == fair and not sell_inclusive):
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
        fair_value=None,
    ) -> Tuple[int, int]:
        fair = self.FAIR_VALUE[product] if fair_value is None else fair_value
        fair_price = int(round(fair))
        projected = position + pending_buys - pending_sells
        if projected > 0 and fair_price in depth.buy_orders:
            bid_vol = depth.buy_orders[fair_price]
            flatten_qty = min(bid_vol, projected)
            pending_sells = self._place_sell(orders, product, fair_price, flatten_qty, position, pending_sells)
        elif projected < 0 and fair_price in depth.sell_orders:
            ask_vol = -depth.sell_orders[fair_price]
            flatten_qty = min(ask_vol, abs(projected))
            pending_buys = self._place_buy(orders, product, fair_price, flatten_qty, position, pending_buys)
        return pending_buys, pending_sells

    def _penny_jump_quotes(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
        fair_value,
        quote_shift: int,
        bid_signal_scale: float,
        ask_signal_scale: float,
    ) -> Tuple[int, int]:
        best_bid, best_ask = self._best_bid_ask(depth)
        bb = best_bid if best_bid != 0 else None
        ba = best_ask if best_ask != 0 else None
        fair_floor = int(fair_value)
        fair_ceil = fair_floor if fair_value == fair_floor else fair_floor + 1

        projected = position + pending_buys - pending_sells

        # ── A-S power-curve volume scaling ──
        pos_limit = self.POSITION_LIMIT[product]
        position_frac = min(abs(projected) / pos_limit, 1.0) if pos_limit > 0 else 0.0
        skew = position_frac ** self.OSMIUM_SKEW_POWER

        accum_scale = self.OSMIUM_ACCUM_FLOOR + (1.0 - self.OSMIUM_ACCUM_FLOOR) * (1.0 - skew)
        unwind_scale = 1.0 + (self.OSMIUM_UNWIND_CEILING - 1.0) * skew

        if projected > 0:
            bid_scale, ask_scale = accum_scale, unwind_scale
        elif projected < 0:
            bid_scale, ask_scale = unwind_scale, accum_scale
        else:
            bid_scale, ask_scale = 1.0, 1.0

        total_bid_qty = int(round(self.BASE_QUOTE_SIZE[product] * bid_scale * bid_signal_scale))
        total_ask_qty = int(round(self.BASE_QUOTE_SIZE[product] * ask_scale * ask_signal_scale))

        inner_bid_qty = int(round(total_bid_qty * self.OSMIUM_INNER_QTY_RATIO))
        outer_bid_qty = max(0, total_bid_qty - inner_bid_qty)
        inner_ask_qty = int(round(total_ask_qty * self.OSMIUM_INNER_QTY_RATIO))
        outer_ask_qty = max(0, total_ask_qty - inner_ask_qty)

        inner_bid = self._inside_bid(bb, ba, self.OSMIUM_INNER_QUOTE_OFFSET, fair_floor - self.OSMIUM_INNER_QUOTE_OFFSET + quote_shift)
        inner_bid = min(inner_bid, fair_floor)
        
        if self.OSMIUM_DYNAMIC_OUTER_ANCHOR and bb is not None and ba is not None:
            spread = max(1, ba - bb)
            outer_bid = inner_bid - spread
        else:
            outer_bid = min(inner_bid - self.OSMIUM_OUTER_QUOTE_OFFSET, fair_floor - self.OSMIUM_OUTER_QUOTE_OFFSET + quote_shift)
            
        if ba is not None:
            outer_bid = min(outer_bid, ba - 1)

        inner_ask = self._inside_ask(bb, ba, self.OSMIUM_INNER_QUOTE_OFFSET, fair_ceil + self.OSMIUM_INNER_QUOTE_OFFSET + quote_shift)
        inner_ask = max(inner_ask, fair_ceil)
        
        if self.OSMIUM_DYNAMIC_OUTER_ANCHOR and bb is not None and ba is not None:
            spread = max(1, ba - bb)
            outer_ask = inner_ask + spread
        else:
            outer_ask = max(inner_ask + self.OSMIUM_OUTER_QUOTE_OFFSET, fair_ceil + self.OSMIUM_OUTER_QUOTE_OFFSET + quote_shift)
            
        if bb is not None:
            outer_ask = max(outer_ask, bb + 1)

        if inner_bid_qty > 0 and (ba is None or inner_bid < ba):
            pending_buys = self._place_buy(orders, product, inner_bid, inner_bid_qty, position, pending_buys)
        if outer_bid_qty > 0 and outer_bid > 0 and (ba is None or outer_bid < ba):
            pending_buys = self._place_buy(orders, product, outer_bid, outer_bid_qty, position, pending_buys)
        if inner_ask_qty > 0 and (bb is None or inner_ask > bb):
            pending_sells = self._place_sell(orders, product, inner_ask, inner_ask_qty, position, pending_sells)
        if outer_ask_qty > 0 and (bb is None or outer_ask > bb):
            pending_sells = self._place_sell(orders, product, outer_ask, outer_ask_qty, position, pending_sells)

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
        if abs(projected) <= self.OSMIUM_EMERGENCY_THRESHOLD:
            return False, pending_buys, pending_sells

        best_bid, best_ask = self._best_bid_ask(depth)
        if projected > self.OSMIUM_EMERGENCY_THRESHOLD:
            flatten_qty = projected - self.OSMIUM_EMERGENCY_TARGET
            if best_bid > 0 and flatten_qty > 0:
                pending_sells = self._place_sell(orders, product, best_bid, flatten_qty, position, pending_sells)
            return True, pending_buys, pending_sells
        if projected < -self.OSMIUM_EMERGENCY_THRESHOLD:
            flatten_qty = abs(projected) - self.OSMIUM_EMERGENCY_TARGET
            if best_ask > 0 and flatten_qty > 0:
                pending_buys = self._place_buy(orders, product, best_ask, flatten_qty, position, pending_buys)
            return True, pending_buys, pending_sells
        return False, pending_buys, pending_sells

    # ── Trending Logic (PEPPER_ROOT) ─────────────────────────
    def _pepper_base_estimate(self, current_mid, timestamp: int, stored_base=None):
        if isinstance(stored_base, (int, float)):
            return float(stored_base)
        if current_mid is None:
            return None
        return float(current_mid) - self.PEPPER_SLOPE * float(timestamp)

    # ── Reservation-Aware Taking ──────────────────────────────
    def _take_reservation_aware(
        self,
        orders: List[Order],
        product: str,
        depth: OrderDepth,
        position: int,
        pending_buys: int,
        pending_sells: int,
        ema_fair: float,
        reservation: float,
    ) -> Tuple[int, int]:
        """Position-aware taking: symmetric near flat, asymmetric at high inventory.

        Near-flat (|position| < SYMMETRIC_ZONE): take everything past EMA (current behaviour).
        At high inventory:
          - Unwind side: take past *reservation* (more aggressive, price-aware)
          - Accumulation side: take only clear mispricings (tighter than EMA)
        """
        if not depth.sell_orders and not depth.buy_orders:
            return pending_buys, pending_sells

        if abs(position) < self.OSMIUM_SYMMETRIC_ZONE:
            # Symmetric EMA-based taking (same as old _take_mispriced inclusive)
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price > ema_fair:
                    break
                ask_vol = -depth.sell_orders[ask_price]
                pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)

            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price < ema_fair:
                    break
                bid_vol = depth.buy_orders[bid_price]
                pending_sells = self._place_sell(orders, product, bid_price, bid_vol, position, pending_sells)

            return pending_buys, pending_sells

        # ── Asymmetric taking for high inventory ──
        if position > 0:
            # Long: unwind by selling, accumulation by buying
            # UNWIND (sell): take bids at or above (reservation - width)
            unwind_threshold = reservation - self.OSMIUM_TAKE_UNWIND_WIDTH
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price < unwind_threshold:
                    break
                bid_vol = depth.buy_orders[bid_price]
                pending_sells = self._place_sell(orders, product, bid_price, bid_vol, position, pending_sells)

            # ACCUMULATION (buy): only take asks clearly below EMA
            accum_threshold = ema_fair - self.OSMIUM_TAKE_ACCUM_WIDTH
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price > accum_threshold:
                    break
                ask_vol = -depth.sell_orders[ask_price]
                pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)

        else:
            # Short: unwind by buying, accumulation by selling
            # UNWIND (buy): take asks at or below (reservation + width)
            unwind_threshold = reservation + self.OSMIUM_TAKE_UNWIND_WIDTH
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price > unwind_threshold:
                    break
                ask_vol = -depth.sell_orders[ask_price]
                pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)

            # ACCUMULATION (sell): only take bids clearly above EMA
            accum_threshold = ema_fair + self.OSMIUM_TAKE_ACCUM_WIDTH
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price < accum_threshold:
                    break
                bid_vol = depth.buy_orders[bid_price]
                pending_sells = self._place_sell(orders, product, bid_price, bid_vol, position, pending_sells)

        return pending_buys, pending_sells

    # ── Asset Execution Pipelines ────────────────────────────
    def _trade_osmium(self, state: TradingState, fair_value, current_mid, last_mid) -> List[Order]:
        """Osmium: EMA fair + A-S reservation-price inventory layer
        + momentum-tilted layered quoting."""
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None:
            return []
        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0
        fair = self.FAIR_VALUE[product] if fair_value is None else fair_value
        last_change = 0.0 if current_mid is None or last_mid is None else current_mid - last_mid

        # ── Step 1: Reservation-aware taking ──
        reservation = fair - position * self.OSMIUM_INVENTORY_SKEW
        pending_buys, pending_sells = self._take_reservation_aware(
            orders, product, depth, position,
            pending_buys, pending_sells,
            ema_fair=fair, reservation=reservation,
        )

        # ── Step 2: Momentum overlay (unchanged) ──
        if last_change > 0:
            quote_shift = -self.OSMIUM_MOMENTUM_QUOTE_SHIFT
            bid_signal_scale = self.OSMIUM_MOMENTUM_DEFENSE_SCALE
            ask_signal_scale = self.OSMIUM_MOMENTUM_AGRESS_SCALE
        elif last_change < 0:
            quote_shift = self.OSMIUM_MOMENTUM_QUOTE_SHIFT
            bid_signal_scale = self.OSMIUM_MOMENTUM_AGRESS_SCALE
            ask_signal_scale = self.OSMIUM_MOMENTUM_DEFENSE_SCALE
        else:
            quote_shift = 0
            bid_signal_scale = 1.0
            ask_signal_scale = 1.0

        # ── Step 3: Penny-jump quotes with power-curve volume skew ──
        self._penny_jump_quotes(
            orders,
            product,
            depth,
            position,
            pending_buys,
            pending_sells,
            fair,
            quote_shift,
            bid_signal_scale,
            ask_signal_scale,
        )

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

        # OIM Calculation & Shift
        bid_shift = 0
        ask_shift = 0
        best_bid, best_ask = self._best_bid_ask(depth)
        
        if best_bid > 0 and best_ask > 0:
            bid_vol = depth.buy_orders.get(best_bid, 0)
            ask_vol = -depth.sell_orders.get(best_ask, 0)
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                oim = (bid_vol - ask_vol) / total_vol
                if abs(oim) > self.PEPPER_OIM_BASE_THRESHOLD:
                    shift_magnitude = int((abs(oim) - self.PEPPER_OIM_BASE_THRESHOLD) / (1 - self.PEPPER_OIM_BASE_THRESHOLD) * self.PEPPER_OIM_MAX_SHIFT) + 1
                    shift_magnitude = min(shift_magnitude, self.PEPPER_OIM_MAX_SHIFT)
                    if oim < 0: # Ask heavy -> fade bid down
                        bid_shift = -shift_magnitude
                    else:       # Bid heavy -> lift ask up
                        ask_shift = shift_magnitude

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

            # Long-biased penny-jump market making after reaching full inventory.
            best_bid, best_ask = self._best_bid_ask(depth)
            bid2, ask2 = self._second_bid_ask(depth)
            l2_gate_ok = (
                bid2 is not None
                and ask2 is not None
                and abs(best_bid - bid2) <= self.PEPPER_MM_L2_MAX_BID_GAP
                and abs(ask2 - best_ask) <= self.PEPPER_MM_L2_MAX_ASK_GAP
            )

            if best_bid > 0 and best_ask > 0 and best_bid + 1 < best_ask and l2_gate_ok:
                mm_bid = best_bid + 1 + bid_shift
                mm_ask = best_ask - 1 + ask_shift

                base = self.PEPPER_MM_BASE_QUOTE_SIZE
                
                # Skew the base quote up/down based on deviation from 80 limit
                projected = position + pending_buys - pending_sells
                deficit = self.PEPPER_POSITION_LIMIT - projected
                
                # For every unit we are short of 80, we add to bid and subtract from ask
                bid_qty = int(round(base * self.PEPPER_MM_BID_WEIGHT)) + deficit
                ask_qty = max(0, base - bid_qty)

                # Preserve long carry: avoid adding more asks if inventory slips.
                if projected < self.PEPPER_MM_MIN_LONG_POSITION:
                    ask_qty = 0
                    bid_qty = base + deficit
                
                # Tight bounds checking
                room_to_buy = self._buy_room(product, position, pending_buys)
                room_to_sell = self._sell_room(product, position, pending_sells)
                
                bid_qty = min(bid_qty, room_to_buy)
                ask_qty = min(ask_qty, room_to_sell)

                if bid_qty > 0:
                    pending_buys = self._place_buy(orders, product, mm_bid, bid_qty, position, pending_buys)
                if ask_qty > 0:
                    pending_sells = self._place_sell(orders, product, mm_ask, ask_qty, position, pending_sells)

        deficit = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
        if deficit > 0:
            if not reached_80:
                max_buy_price = fair_center + self.PEPPER_INITIAL_ACC_THRESH
                pending_buys = self._take_asks(orders, product, depth, max_buy_price, position, pending_buys, max_total=deficit)

                remaining = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
                if remaining > 0:
                    pending_buys = self._place_buy(orders, product, fair_center, remaining, position, pending_buys)

            else:
                max_buy_price = fair_center + self.PEPPER_RECOUP_MAX_MARGIN
                pending_buys = self._take_asks(
                    orders,
                    product,
                    depth,
                    max_buy_price,
                    position,
                    pending_buys,
                    max_total=deficit,
                )

                remaining = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
                if remaining > 0:
                    bid_post_price = min(fair_center, fair_center + self.PEPPER_RECOUP_MAX_MARGIN)
                    pending_buys = self._place_buy(
                        orders, product, bid_post_price, remaining, position, pending_buys
                    )

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

        osmium_depth = state.order_depths.get("ASH_COATED_OSMIUM")
        osmium_mid = self._mid_price(osmium_depth) if osmium_depth else None
        osmium_ema = data.get("osmium_ema")
        osmium_last_mid = data.get("osmium_last_mid")

        if osmium_mid is not None:
            if osmium_ema is None:
                osmium_ema = float(osmium_mid)
            else:
                osmium_ema = self.OSMIUM_EMA_ALPHA * float(osmium_mid) + (1 - self.OSMIUM_EMA_ALPHA) * float(osmium_ema)
        osmium_fair = float(osmium_ema) if osmium_ema is not None else float(self.FAIR_VALUE["ASH_COATED_OSMIUM"])

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(
                state,
                osmium_fair,
                osmium_mid,
                osmium_last_mid,
            )

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            pepper_orders, pepper_reached_80 = self._trade_pepper_root(
                state,
                pepper_base_estimate,
                data.get("pepper_reached_80", False),
            )
            result["INTARIAN_PEPPER_ROOT"] = pepper_orders
            data["pepper_reached_80"] = pepper_reached_80

        data["pepper_base_estimate"] = pepper_base_estimate
        data["osmium_ema"] = osmium_ema
        data["osmium_last_mid"] = osmium_mid if osmium_mid is not None else osmium_last_mid

        trader_data = jsonpickle.encode(data)
        return result, 0, trader_data
