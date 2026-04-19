from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle


class Trader:
    """
    ============================================
    ASH_COATED_OSMIUM: Static-fair penny-jump MM with OIM signal.
                       No EMA. No panic flattening. Follow order imbalance.
    INTARIAN_PEPPER_ROOT: detrended long bias + capped scalp sells into bid spikes above FV.
    """

    # ── Osmium: OIM-Led Regime Discovery MM ───────────────
    OSMIUM_POSITION_LIMIT = 80
    
    # OSMIUM_BASE_QUOTE_SIZE = 48
    # OSMIUM_VOLUME_SKEW_AGGRESSION = 0.590
    # OSMIUM_KILL_SWITCH_THRESHOLD = 78
    
    # OSMIUM_OIM_THRESHOLD = 0.900
    # OSMIUM_OIM_SHIFT = 1
    # OSMIUM_OIM_EDGE_SCALE = 1.000
    # OSMIUM_OIM_FADE_SCALE = 1.000

    # OSMIUM_INNER_OFFSET = 3
    # OSMIUM_OUTER_OFFSET = 26
    
    # # Continuous Risk Multipliers
    # OSMIUM_FV_TETHER_SCALE = 0.050
    # OSMIUM_OIM_TAKE_SCALE = 1.000
    
    OSMIUM_FAIR_VALUE                   =    10009
    OSMIUM_INNER_OFFSET                 =        8
    OSMIUM_OUTER_OFFSET                 =        8
    OSMIUM_VOLUME_SKEW_AGGRESSION       =      1.354
    OSMIUM_OIM_SHIFT                    =        0
    OSMIUM_BASE_QUOTE_SIZE              =       78
    OSMIUM_KILL_SWITCH_THRESHOLD        =       80
    OSMIUM_OIM_THRESHOLD                =      0.853
    OSMIUM_OIM_FADE_SCALE               =      0.006
    OSMIUM_OIM_EDGE_SCALE               =      1.774
    OSMIUM_OIM_TAKE_SCALE               =      0.906
    OSMIUM_FV_TETHER_SCALE              =       0.07
    OSMIUM_L2_QUOTE_SIZE                =       80


    # ── Pepper (Optimized via Grid Search — UNTOUCHED) ──
    PEPPER_SLOPE = 0.001
    PEPPER_POSITION_LIMIT = 80
    PEPPER_INITIAL_ACC_THRESH = 8
    PEPPER_SCALP_MIN_MARGIN = 4
    PEPPER_MAX_SCALP_VOLUME = 3
    PEPPER_RECOUP_MAX_MARGIN = -2
    PEPPER_MM_BASE_QUOTE_SIZE = 17
    PEPPER_MM_BID_WEIGHT = 0.45
    PEPPER_MM_MIN_LONG_POSITION = 60
    PEPPER_MM_L2_MAX_BID_GAP = 6
    PEPPER_MM_L2_MAX_ASK_GAP = 5
    PEPPER_OIM_BASE_THRESHOLD = 0.1
    PEPPER_OIM_MAX_SHIFT = 2

    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

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

    @staticmethod
    def _two_sided_mid(depth: OrderDepth):
        """Mid price only when BOTH bid and ask exist. Returns None on one-sided books."""
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
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

    # ── Osmium Execution Pipeline ────────────────────────────
    def _trade_osmium(self, state: TradingState) -> List[Order]:
        """OIM-Led Regime Discovery MM. Centers around localized mid_price with FV tethering."""
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0

        # Stage 1: Market State Analysis 
        best_bid, best_ask = self._best_bid_ask(depth)
        current_mid = self._two_sided_mid(depth)
        
        if current_mid is None:
            if best_bid > 0:
                current_mid = best_bid + self.OSMIUM_OUTER_OFFSET
            elif best_ask > 0:
                current_mid = best_ask - self.OSMIUM_OUTER_OFFSET
            else:
                current_mid = self.OSMIUM_FAIR_VALUE
                
        # Stage 2: Calculate Leading Indicator (OIM)
        # Empirical Data: L1 is highly predictive (99% hit rate h=5). L2/L3 is spoofed noise.
        oim = 0.0
        bid_vol = depth.buy_orders.get(best_bid, 0)
        ask_vol = abs(depth.sell_orders.get(best_ask, 0))
            
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            oim = (bid_vol - ask_vol) / total_vol

        # Global toggle mapping: disable OIM completely if variables are set to 0.
        if self.OSMIUM_OIM_THRESHOLD <= 0.0 or (self.OSMIUM_OIM_EDGE_SCALE == 0.0 and self.OSMIUM_OIM_FADE_SCALE == 0.0):
            oim = 0.0

        # Stage 3: Trend Taking (Toxic Liquidator)
        if abs(oim) >= self.OSMIUM_OIM_THRESHOLD and self.OSMIUM_OIM_TAKE_SCALE > 0:
            take_fraction = min(1.0, abs(oim) * self.OSMIUM_OIM_TAKE_SCALE)
            # NEVER cross the spread unconditionally. Only take if quote provides strict mathematical FV edge.
            if oim > 0 and best_ask > 0 and best_ask < self.OSMIUM_FAIR_VALUE:
                room = self._buy_room(product, position, pending_buys)
                desired_take = int(round(abs(depth.sell_orders[best_ask]) * take_fraction))
                take_vol = min(desired_take, room)
                if take_vol > 0:
                    pending_buys = self._place_buy(orders, product, best_ask, take_vol, position, pending_buys)
            elif oim < 0 and best_bid > 0 and best_bid > self.OSMIUM_FAIR_VALUE:
                room = self._sell_room(product, position, pending_sells)
                desired_take = int(round(depth.buy_orders[best_bid] * take_fraction))
                take_vol = min(desired_take, room)
                if take_vol > 0:
                    pending_sells = self._place_sell(orders, product, best_bid, take_vol, position, pending_sells)
                
        # Stage 4: Quote Parameterization & Skew
        projected = position + pending_buys - pending_sells
        pos_ratio = projected / self.OSMIUM_POSITION_LIMIT
        
        # Weak Gravity Tether to Global FV shifts the effective inventory perception
        tether_skew = (current_mid - self.OSMIUM_FAIR_VALUE) * self.OSMIUM_FV_TETHER_SCALE
        effective_pos_ratio = pos_ratio + tether_skew
            
        bid_scale = max(0.0, 1.0 - max(0.0, effective_pos_ratio) * self.OSMIUM_VOLUME_SKEW_AGGRESSION)
        ask_scale = max(0.0, 1.0 + min(0.0, effective_pos_ratio) * self.OSMIUM_VOLUME_SKEW_AGGRESSION)
        
        bid_shift = 0
        ask_shift = 0
        bid_signal_scale = 1.0
        ask_signal_scale = 1.0

        if oim > self.OSMIUM_OIM_THRESHOLD:
            bid_shift = self.OSMIUM_OIM_SHIFT
            ask_shift = self.OSMIUM_OIM_SHIFT
            bid_signal_scale = self.OSMIUM_OIM_EDGE_SCALE
            ask_signal_scale = self.OSMIUM_OIM_FADE_SCALE
        elif oim < -self.OSMIUM_OIM_THRESHOLD:
            bid_shift = -self.OSMIUM_OIM_SHIFT
            ask_shift = -self.OSMIUM_OIM_SHIFT
            bid_signal_scale = self.OSMIUM_OIM_FADE_SCALE
            ask_signal_scale = self.OSMIUM_OIM_EDGE_SCALE

        total_bid_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * bid_scale * bid_signal_scale))
        total_ask_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * ask_scale * ask_signal_scale))

        if projected >= self.OSMIUM_KILL_SWITCH_THRESHOLD:
            total_bid_qty = 0
        elif projected <= -self.OSMIUM_KILL_SWITCH_THRESHOLD:
            total_ask_qty = 0

        import math
        mm_bid = math.floor(current_mid) - self.OSMIUM_INNER_OFFSET + bid_shift
        mm_ask = math.ceil(current_mid) + self.OSMIUM_INNER_OFFSET + ask_shift

        # L1/L2 Penny Jumping Component
        bid1, ask1 = best_bid, best_ask
        bid2, ask2 = self._second_bid_ask(depth)
        
        max_acceptable_bid = math.floor(self.OSMIUM_FAIR_VALUE) - 1 + bid_shift
        min_acceptable_ask = math.ceil(self.OSMIUM_FAIR_VALUE) + 1 + ask_shift

        # Try to aggressively penny-jump L1. If L1 violates our bounds, fall back to capturing L2 priority.
        jump_l2_bid = False
        jump_l2_ask = False

        if bid1 > 0 and bid1 + 1 <= max_acceptable_bid:
            mm_bid = max(mm_bid, bid1 + 1)
        elif bid2 is not None and bid2 + 1 <= max_acceptable_bid:
            mm_bid = max(mm_bid, bid2 + 1)
            jump_l2_bid = True

        if ask1 > 0 and ask1 - 1 >= min_acceptable_ask:
            mm_ask = min(mm_ask, ask1 - 1)
        elif ask2 is not None and ask2 - 1 >= min_acceptable_ask:
            mm_ask = min(mm_ask, ask2 - 1)
            jump_l2_ask = True

        # Stage 5: Safety bounding (Never cross our own spread, respect global FV extreme bounds)
        if total_bid_qty > 0:
            qty = total_bid_qty
            if jump_l2_bid:
                room = self._buy_room(product, position, pending_buys)
                qty = max(self.OSMIUM_L2_QUOTE_SIZE, room)

            if best_ask > 0:
                mm_bid = min(mm_bid, best_ask - 1)
            # Cap at extremely absurd prices
            mm_bid = min(mm_bid, self.OSMIUM_FAIR_VALUE + self.OSMIUM_OUTER_OFFSET)
            pending_buys = self._place_buy(orders, product, mm_bid, qty, position, pending_buys)
        
        if total_ask_qty > 0:
            qty = total_ask_qty
            if jump_l2_ask:
                room = self._sell_room(product, position, pending_sells)
                qty = max(self.OSMIUM_L2_QUOTE_SIZE, room)

            if best_bid > 0:
                mm_ask = max(mm_ask, best_bid + 1)
            # Floor at extremely absurd prices
            mm_ask = max(mm_ask, self.OSMIUM_FAIR_VALUE - self.OSMIUM_OUTER_OFFSET)
            pending_sells = self._place_sell(orders, product, mm_ask, qty, position, pending_sells)
             
        return orders

    # ── Trending Logic (PEPPER_ROOT) — UNTOUCHED ─────────────
    def _pepper_base_estimate(self, current_mid, timestamp: int, stored_base=None):
        if isinstance(stored_base, (int, float)):
            return float(stored_base)
        if current_mid is None:
            return None
        return float(current_mid) - self.PEPPER_SLOPE * float(timestamp)

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

        result: Dict[str, List[Order]] = {}

        # ── Osmium ──
        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        # ── Pepper Root ──
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