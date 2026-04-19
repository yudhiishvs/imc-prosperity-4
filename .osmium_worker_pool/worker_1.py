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
    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}
    
    OSMIUM_FAIR_VALUE = 10_000      # hardcoded constant, not tuned
    
    # Quote structure
    OSMIUM_BASE_QUOTE_SIZE = 18     # base size per side of the book

    # Price-deviation volume skew
    OSMIUM_DEV_SCALE = 6.7132          # normalization: deviation of this many ticks = "fully skewed"
    OSMIUM_DEV_POWER = 1.3018          # shape of skew curve: 1=linear, 2=quadratic, 3=cubic, etc.
    OSMIUM_UNWIND_CEILING = 1.5965     # max volume multiplier on the side moving TOWARD 10k
    OSMIUM_ACCUM_FLOOR = 0.8271       # min volume multiplier on the side moving AWAY from 10k

    # Inventory gate (hysteresis — pulls all accumulation-side quotes when deeply offside)
    OSMIUM_KILL_QUOTE_THRESHOLD = 80  # |pos| at which we PULL all accumulation-side quotes
    OSMIUM_RESUME_QUOTE_THRESHOLD = 79 # |pos| at which we RESUME quoting both sides again

    # ── Pepper (Optimized via Grid Search) ──
    PEPPER_SLOPE = 0.001
    PEPPER_POSITION_LIMIT = 80
    PEPPER_INITIAL_ACC_THRESH = 8
    PEPPER_SCALP_MIN_MARGIN = 4
    PEPPER_MAX_SCALP_VOLUME = 3  # max units sold per tick into bid(s) at/above scalp threshold; sweep 1..80
    PEPPER_RECOUP_MAX_MARGIN = -2
    # Post-reach market making (long-biased): penny-jump both sides, favor bids.
    PEPPER_MM_BASE_QUOTE_SIZE = 17
    PEPPER_MM_BID_WEIGHT = 0.45
    PEPPER_MM_MIN_LONG_POSITION = 60
    # L2 quality gate: only penny-jump when L1 is sufficiently close to L2.
    PEPPER_MM_L2_MAX_BID_GAP = 7  # max |L1 bid - L2 bid|
    PEPPER_MM_L2_MAX_ASK_GAP = 5  # max |L2 ask - L1 ask|
    # OIM Dynamic Shifts
    PEPPER_OIM_BASE_THRESHOLD = 0.1
    PEPPER_OIM_MAX_SHIFT = 2

    def bid(self) -> int:
        return 15

    # ── State Persistence ────────────────────────────────────
    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_base_estimate": None,
            "pepper_reached_80": False,
            "osmium_gated": False,
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

        osmium_gated = data.get("osmium_gated", False)
        if not isinstance(osmium_gated, bool):
            osmium_gated = False

        return {
            "pepper_base_estimate": base_estimate,
            "pepper_reached_80": pepper_reached_80,
            "osmium_gated": osmium_gated,
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

    def _trade_osmium(self, state: TradingState, is_gated: bool) -> Tuple[List[Order], bool]:
        """Osmium: Mean-reversion market making anchored to 10,000.

        Pipeline:
          1. Hysteresis inventory gate (pull accumulation-side quotes when deeply offside)
          2. Taking — sweep free money strictly beyond FV, then position-reduce AT FV
          3. Deviation-based volume skew (more size toward 10k, less size away)
          4. Penny-jump quoting with hard FV cap (never bid above 10k, never offer below 10k)
        """
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None:
            return [], is_gated

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0
        current_mid = self._mid_price(depth)
        if current_mid is None:
            return [], is_gated

        fair = self.OSMIUM_FAIR_VALUE   # always 10,000
        fair_int = int(fair)

        # ── 1. Hysteresis Inventory Gate ──────────────────────────────────
        abs_pos = abs(position)
        if abs_pos >= self.OSMIUM_KILL_QUOTE_THRESHOLD:
            is_gated = True
        elif abs_pos < self.OSMIUM_RESUME_QUOTE_THRESHOLD:
            is_gated = False

        # ── 2. Taking ────────────────────────────────────────────────────
        # 2a. Free money: sweep asks BELOW fair and bids ABOVE fair (always positive edge).
        pending_buys = self._take_asks(orders, product, depth, fair_int - 1, position, pending_buys)
        pending_sells = self._take_bids(orders, product, depth, fair_int + 1, position, pending_sells)

        # 2b. Position reduction AT fair value.
        #     At exactly 10k there's zero directional edge, so only trade here
        #     if it REDUCES our current inventory toward flat.
        projected = position + pending_buys - pending_sells
        if projected > 0 and fair_int in depth.buy_orders:
            # We're long — sell to bids at 10k to flatten
            bid_vol = depth.buy_orders[fair_int]
            reduce_qty = min(bid_vol, projected)
            pending_sells = self._place_sell(orders, product, fair_int, reduce_qty, position, pending_sells)
        elif projected < 0 and fair_int in depth.sell_orders:
            # We're short — buy asks at 10k to flatten
            ask_vol = -depth.sell_orders[fair_int]
            reduce_qty = min(ask_vol, abs(projected))
            pending_buys = self._place_buy(orders, product, fair_int, reduce_qty, position, pending_buys)

        # ── 3. Deviation-based volume skew ───────────────────────────────
        deviation = current_mid - fair
        if self.OSMIUM_DEV_SCALE > 0:
            dev_frac = max(-1.0, min(1.0, deviation / self.OSMIUM_DEV_SCALE))
        else:
            dev_frac = 0.0

        skew = abs(dev_frac) ** self.OSMIUM_DEV_POWER

        if deviation > 0:
            # Price above 10k -> sell aggressively (toward 10k), buy cautiously
            ask_scale = 1.0 + (self.OSMIUM_UNWIND_CEILING - 1.0) * skew
            bid_scale = self.OSMIUM_ACCUM_FLOOR + (1.0 - self.OSMIUM_ACCUM_FLOOR) * (1.0 - skew)
        elif deviation < 0:
            # Price below 10k -> buy aggressively (toward 10k), sell cautiously
            bid_scale = 1.0 + (self.OSMIUM_UNWIND_CEILING - 1.0) * skew
            ask_scale = self.OSMIUM_ACCUM_FLOOR + (1.0 - self.OSMIUM_ACCUM_FLOOR) * (1.0 - skew)
        else:
            bid_scale = 1.0
            ask_scale = 1.0

        # Kill gate: zero out the accumulation side when deeply offside
        if is_gated:
            if position > 0:
                bid_scale = 0.0
            elif position < 0:
                ask_scale = 0.0

        # ── 4. Penny-jump quoting with FV cap ────────────────────────────
        # Penny-jump the best bid/ask, but NEVER bid above 10k and
        # NEVER offer below 10k.  This ensures we only accumulate at a
        # discount and only sell at a premium relative to the anchor.
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid != 0 and best_ask != 0:
            inner_bid = best_bid + 1
            inner_ask = best_ask - 1
            if inner_bid >= inner_ask:   # would cross or lock
                inner_bid = best_bid
                inner_ask = best_ask
        else:
            inner_bid = best_bid if best_bid != 0 else fair_int - 1
            inner_ask = best_ask if best_ask != 0 else fair_int + 1

        # Hard cap at fair value — the core quote-shift mechanism.
        # When price drifts above 10k, our bid gets pinned to 10k (wide from mid),
        # while our ask penny-jumps normally (tight to mid). Vice-versa below 10k.
        inner_bid = min(inner_bid, fair_int)
        inner_ask = max(inner_ask, fair_int)

        bid_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * bid_scale))
        ask_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * ask_scale))

        if bid_qty > 0:
            pending_buys = self._place_buy(orders, product, inner_bid, bid_qty, position, pending_buys)
        if ask_qty > 0:
            pending_sells = self._place_sell(orders, product, inner_ask, ask_qty, position, pending_sells)

        return orders, is_gated

    # ── Trending Logic (PEPPER_ROOT) ─────────────────────────
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

        if "ASH_COATED_OSMIUM" in state.order_depths:
            osmium_orders, osmium_gated = self._trade_osmium(
                state,
                data.get("osmium_gated", False),
            )
            result["ASH_COATED_OSMIUM"] = osmium_orders
            data["osmium_gated"] = osmium_gated

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
