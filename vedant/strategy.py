from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import statistics
import jsonpickle

class Trader:
    """
    Advanced Data-Driven Strategy
    ============================================
    Assets: ASH_COATED_OSMIUM (OIM-Skewed Peg), INTARIAN_PEPPER_ROOT (Detrended Long Bias)
    """

    # ── Penny-Jump Configuration (OSMIUM) ───────────────────
    FAIR_VALUE = {"ASH_COATED_OSMIUM": 10_000}
    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    BASE_QUOTE_SIZE = {"ASH_COATED_OSMIUM": 20}
    VOLUME_SKEW_AGGRESSION = {"ASH_COATED_OSMIUM": 1}
    EMERGENCY_THRESHOLD = {"ASH_COATED_OSMIUM": 70}
    EMERGENCY_TARGET = {"ASH_COATED_OSMIUM": 40}
    KILL_SWITCH_THRESHOLD = {"ASH_COATED_OSMIUM": 80}
    OSMIUM_OIM_MULTIPLIER = 4  # Ticks to skew fair value based on full book imbalance

    # ── Trending Model Configuration (PEPPER_ROOT) ───────────
    PEPPER_SLOPE = 0.001       # Deterministic upward trend per timestamp
    PEPPER_DUMP_MIN = 5        # Scalp out at FV + 5
    PEPPER_RECOUP_MAX = 0      # Re-buy scalped units safely at FV or below

    def bid(self) -> int:
        return 15

    # ── State Persistence ────────────────────────────────────
    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_base_estimate": None,
            "pepper_reached_80": False
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
            "pepper_reached_80": pepper_reached_80
        }

    # ── Shared Helpers ───────────────────────────────────────
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
        return self.POSITION_LIMIT.get(product, 20) - (position + pending_buys)

    def _sell_room(self, product: str, position: int, pending_sells: int) -> int:
        return self.POSITION_LIMIT.get(product, 20) + (position - pending_sells)

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
    def _take_mispriced(self, orders: List[Order], product: str, depth: OrderDepth, position: int, pending_buys: int, pending_sells: int, fair: int) -> Tuple[int, int]:
        # Empty book protection
        if not depth.sell_orders and not depth.buy_orders:
            return pending_buys, pending_sells
            
        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price >= fair: break
            ask_vol = -depth.sell_orders[ask_price]
            pending_buys = self._place_buy(orders, product, ask_price, ask_vol, position, pending_buys)
            
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price <= fair: break
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

    def _emergency_flatten(self, orders: List[Order], product: str, depth: OrderDepth, position: int, pending_buys: int, pending_sells: int) -> Tuple[bool, int, int]:
        projected = position + pending_buys - pending_sells
        threshold = self.EMERGENCY_THRESHOLD.get(product, 70)
        target = self.EMERGENCY_TARGET.get(product, 40)
        
        if abs(projected) <= threshold:
            return False, pending_buys, pending_sells

        best_bid, best_ask = self._best_bid_ask(depth)
        
        if projected > threshold:
            flatten_qty = projected - target
            if best_bid > 0 and flatten_qty > 0:
                pending_sells = self._place_sell(orders, product, best_bid, flatten_qty, position, pending_sells)
            return True, pending_buys, pending_sells
            
        elif projected < -threshold:
            flatten_qty = abs(projected) - target
            if best_ask > 0 and flatten_qty > 0:
                pending_buys = self._place_buy(orders, product, best_ask, flatten_qty, position, pending_buys)
            return True, pending_buys, pending_sells
            
        return False, pending_buys, pending_sells

    def _penny_jump_quotes(self, orders: List[Order], product: str, depth: OrderDepth, position: int, pending_buys: int, pending_sells: int, fair: int) -> Tuple[int, int]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid == 0 or best_ask == 0:
            return pending_buys, pending_sells

        penny_bid = best_bid + 1
        penny_ask = best_ask - 1
        
        # Don't quote past calculated fair
        penny_bid = min(penny_bid, fair - 1)
        penny_ask = max(penny_ask, fair + 1)

        projected = position + pending_buys - pending_sells
        position_ratio = projected / self.POSITION_LIMIT.get(product, 80)
        aggression = self.VOLUME_SKEW_AGGRESSION.get(product, 1)
        
        bid_scale = max(0.0, 1.0 - max(0.0, position_ratio) * aggression)
        ask_scale = max(0.0, 1.0 + min(0.0, position_ratio) * aggression)

        base_qty = self.BASE_QUOTE_SIZE.get(product, 20)
        bid_qty = int(round(base_qty * bid_scale))
        ask_qty = int(round(base_qty * ask_scale))

        kill_switch = self.KILL_SWITCH_THRESHOLD.get(product, 80)
        if projected >= kill_switch: bid_qty = 0
        elif projected <= -kill_switch: ask_qty = 0

        if bid_qty > 0:
            pending_buys = self._place_buy(orders, product, penny_bid, bid_qty, position, pending_buys)
        if ask_qty > 0:
            pending_sells = self._place_sell(orders, product, penny_ask, ask_qty, position, pending_sells)
            
        return pending_buys, pending_sells

    # ── Trending Logic Integrations (PEPPER_ROOT) ────────────
    def _pepper_base_estimate(self, current_mid, timestamp: int, stored_base=None):
        if isinstance(stored_base, (int, float)):
            return float(stored_base)
        if current_mid is None:
            return None
        return float(current_mid) - self.PEPPER_SLOPE * float(timestamp)


    # ── Asset Execution Pipelines ────────────────────────────
    def _trade_osmium(self, state: TradingState) -> List[Order]:
        """Osmium execution with OIM-driven fair value skews."""
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None: return []

        # Empty book protection
        if not depth.buy_orders and not depth.sell_orders:
            return []

        # Calculate Total Book Imbalance
        total_bid_vol = sum(depth.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in depth.sell_orders.values())
        oim = 0.0
        if total_bid_vol + total_ask_vol > 0:
            oim = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

        dynamic_fair = self.FAIR_VALUE[product] + int(round(oim * self.OSMIUM_OIM_MULTIPLIER))

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pb = ps = 0

        # Pass the customized Dynamic Fair to strictly arb mispricings
        pb, ps = self._take_mispriced(orders, product, depth, position, pb, ps, dynamic_fair)
        pb, ps = self._flatten_at_fair(orders, product, depth, position, pb, ps, dynamic_fair)
        
        triggered, pb, ps = self._emergency_flatten(orders, product, depth, position, pb, ps)
        if not triggered:
            pb, ps = self._penny_jump_quotes(orders, product, depth, position, pb, ps, dynamic_fair)

        return orders


    def _trade_pepper_root(self, state: TradingState, base_estimate, reached_80: bool) -> Tuple[List[Order], bool]:
        """Two-Stage Macro-Accumulation and Opportunistic Scalping."""
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        if depth is None: return [], reached_80

        current_mid = self._mid_price(depth)
        if current_mid is None: return [], reached_80

        orders: List[Order] = []
        position = state.position.get(product, 0)
        
        if position == 80:
            reached_80 = True
        
        if base_estimate is None:
            base_estimate = current_mid  # Fallback for tick 0 instantly

        timestamp = state.timestamp
        fair = float(base_estimate) + self.PEPPER_SLOPE * float(timestamp)
        fair_center = int(round(fair))

        limit = self.POSITION_LIMIT.get(product, 80)
        pending_buys = 0
        pending_sells = 0

        # ── Stage 2: Opportunistic Scalping (Only if already reached 80) ──
        if reached_80 and position > 0:
            target_sell_price = fair_center + self.PEPPER_DUMP_MIN
            # Hit resting bids at target_sell_price or better (e.g. FV+5)
            pending_sells = self._take_bids(orders, product, depth, target_sell_price, position, pending_sells, max_total=position)

        # ── Accumulation / Recoup Drive ──
        deficit = limit - (position + pending_buys)
        if deficit > 0:
            if not reached_80:
                # Stage 1: Sweep up to FV + 12 absolutely indiscriminately
                max_buy_price = fair_center + 10
                pending_buys = self._take_asks(orders, product, depth, max_buy_price, position, pending_buys, max_total=deficit)

                remaining = limit - (position + pending_buys)
                if remaining > 0:
                    first_bid_qty = (remaining + 1) // 2
                    second_bid_qty = remaining - first_bid_qty
                    pending_buys = self._place_buy(orders, product, fair_center + 2, first_bid_qty, position, pending_buys)
                    if second_bid_qty > 0:
                        pending_buys = self._place_buy(orders, product, fair_center + 1, second_bid_qty, position, pending_buys)
            else:
                # Stage 2: Patient Recouping up to FV + RECOUP_MAX
                max_buy_price = fair_center + self.PEPPER_RECOUP_MAX
                # Sweep resting asks at or below target
                pending_buys = self._take_asks(orders, product, depth, max_buy_price, position, pending_buys, max_total=deficit)
                
                remaining = limit - (position + pending_buys)
                if remaining > 0:
                    # Post passive Bid wall safely near the drift line
                    bid_post_price = min(fair_center + 1, fair_center + self.PEPPER_RECOUP_MAX)
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
            data.get("pepper_base_estimate")
        )

        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            pepper_orders, pepper_reached_80 = self._trade_pepper_root(
                state,
                pepper_base_estimate,
                data.get("pepper_reached_80", False)
            )
            result["INTARIAN_PEPPER_ROOT"] = pepper_orders
            data["pepper_reached_80"] = pepper_reached_80

        data["pepper_base_estimate"] = pepper_base_estimate

        return result, 0, jsonpickle.encode(data)