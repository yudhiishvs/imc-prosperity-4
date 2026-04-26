from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple, Optional
import jsonpickle
import math


class Trader:
    """
    Robustified 301k baseline:
    - Base: Yudhiish's OIM-Led Osmium and Defensive Trend-Tracking Pepper.
    - Alpha Inject: Vedant's Penny-Jumping L2 Gate and Deficit-Skewed MM.
    - Fixed: State persistence fallbacks and None-safety.
    """

    # -- Osmium: Hybrid Regime Discovery MM --
    OSMIUM_POSITION_LIMIT = 80
    OSMIUM_INITIAL_FV = 10004

    OSMIUM_EMA_ALPHA = 0.081 
    OSMIUM_INNER_OFFSET = 9   
    OSMIUM_OUTER_OFFSET = 6   
    OSMIUM_VOLUME_SKEW_AGGRESSION = 0.823 
    OSMIUM_OIM_SHIFT = 2
    OSMIUM_BASE_QUOTE_SIZE = 55 
    OSMIUM_L2_QUOTE_SIZE = 35   
    OSMIUM_KILL_SWITCH_THRESHOLD = 80
    OSMIUM_OIM_THRESHOLD = 0.031 
    OSMIUM_OIM_FADE_SCALE = 0.341 
    OSMIUM_OIM_EDGE_SCALE = 3.389 
    OSMIUM_OIM_TAKE_SCALE = 4.526 
    OSMIUM_FV_TETHER_SCALE = 0.053 
    OSMIUM_VOL_EMA_ALPHA = 0.100
    OSMIUM_SPREAD_EMA_ALPHA = 0.050
    OSMIUM_OIM_EDGE_ALPHA = 0.080
    OSMIUM_VOL_STRESS_LEVEL = 3.200 
    OSMIUM_WIDE_SPREAD_MULT = 1.750
    OSMIUM_LATE_REDUCE_TS = 90000
    OSMIUM_LATE_TARGET_ABS = 55
    OSMIUM_CLOSE_REDUCE_TS = 96000
    OSMIUM_CLOSE_TARGET_ABS = 35
    OSMIUM_LATE_FLATTEN_CHUNK = 10

    # -- Pepper --
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
    PEPPER_BASE_UPDATE_ALPHA = 0.025
    PEPPER_RESID_EMA_ALPHA = 0.035
    PEPPER_TREND_EMA_ALPHA = 0.070
    PEPPER_SPREAD_EMA_ALPHA = 0.060
    PEPPER_SOFT_BREAK_RESID_EMA = -2.6
    PEPPER_SLOW_DRIFT_LEVEL = -0.50
    PEPPER_NEG_DRIFT_LEVEL = -0.03
    PEPPER_STRESS_SPREAD_MULT = 1.70
    PEPPER_REGIME_REDUCE_TS = 99200
    PEPPER_LATE_REDUCE_TS = 99200
    PEPPER_CLOSE_REDUCE_TS = 99700
    PEPPER_HARD_CLOSE_TS = 99900
    PEPPER_LATE_TARGET = 80
    PEPPER_CLOSE_TARGET = 78
    PEPPER_HARD_CLOSE_TARGET = 62
    PEPPER_SOFT_BREAK_TARGET = 76
    PEPPER_NEG_BREAK_TARGET = 60
    PEPPER_STALE_POS_TS = 25000
    PEPPER_STALE_POS_TS_HARD = 35000
    PEPPER_STALE_TARGET = 76
    PEPPER_STALE_HARD_TARGET = 70
    PEPPER_LATE_SELL_MARGIN = 2
    PEPPER_CLOSE_SELL_MARGIN = 0
    PEPPER_STRONG_BREAK_TS = 65000
    PEPPER_STRONG_BREAK_RESID_EMA = -3.5
    PEPPER_STRONG_BREAK_TARGET = 50
    PEPPER_DEFENSIVE_TRIM_CHUNK = 10
    PEPPER_EXTRA_REDUCE_CHUNK = 12
    PEPPER_EARLY_POST_OFFSET = 1
    PEPPER_EARLY_POST_TS = 20000

    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    def _load_data(self, raw: str) -> dict:
        default_data = {
            "pepper_base_estimate": None,
            "pepper_reached_80": False,
            "pepper_resid_ema": None,
            "pepper_prev_mid": None,
            "pepper_trend_ema": 0.0,
            "pepper_spread_ema": 14.0,
            "pepper_prev_position": None,
            "pepper_last_pos_change_ts": 0,
            "osmium_ema": None,
            "osmium_prev_mid": None,
            "osmium_prev_oim": None,
            "osmium_vol_ema": 1.5,
            "osmium_spread_ema": 12.0,
            "osmium_oim_edge": 0.0,
        }
        if not raw:
            return default_data
        try:
            data = jsonpickle.decode(raw)
        except Exception:
            return default_data
        if not isinstance(data, dict):
            return default_data

        # Merge with default_data to ensure all keys are present
        for k, v in default_data.items():
            if k not in data or data[k] is None:
                data[k] = v
        return data

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _ewma(previous: Optional[float], value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return (1.0 - alpha) * float(previous) + alpha * float(value)

    def _get_position_limit(self, product: str) -> int:
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
        return best_bid or best_ask

    @staticmethod
    def _two_sided_mid(depth: OrderDepth):
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
            size = min(-depth.sell_orders[ask], room)
            if max_total is not None:
                size = min(size, max_total - bought)
            if size > 0:
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
            size = min(depth.buy_orders[bid], room)
            if max_total is not None:
                size = min(size, max_total - sold)
            if size > 0:
                orders.append(Order(product, bid, -size))
                pending_sells += size
                sold += size
        return pending_sells

    def _trade_osmium(self, state: TradingState, data: dict) -> Tuple[List[Order], dict]:
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None: return [], {}
        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0

        best_bid, best_ask = self._best_bid_ask(depth)
        current_mid = self._two_sided_mid(depth)
        # Fixed: Fallback to initial FV if ema is missing or None
        osmium_ema = data.get("osmium_ema")
        if osmium_ema is None:
            osmium_ema = float(self.OSMIUM_INITIAL_FV)
            
        if current_mid is None:
            current_mid = osmium_ema

        osmium_ema = (1.0 - self.OSMIUM_EMA_ALPHA) * osmium_ema + self.OSMIUM_EMA_ALPHA * current_mid
        osmium_fv = osmium_ema

        spread_now = float(best_ask - best_bid) if best_bid > 0 and best_ask > 0 else data.get("osmium_spread_ema", 12.0)
        prev_mid = data.get("osmium_prev_mid")
        ret = 0.0 if prev_mid is None else float(current_mid) - float(prev_mid)
        osmium_vol_ema = self._ewma(data.get("osmium_vol_ema"), abs(ret), self.OSMIUM_VOL_EMA_ALPHA)
        osmium_spread_ema = self._ewma(data.get("osmium_spread_ema"), spread_now, self.OSMIUM_SPREAD_EMA_ALPHA)

        oim = 0.0
        bid_vol = depth.buy_orders.get(best_bid, 0)
        ask_vol = abs(depth.sell_orders.get(best_ask, 0))
        if (bid_vol + ask_vol) > 0:
            oim = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        prev_oim = data.get("osmium_prev_oim")
        oim_edge_sample = 0.0
        if prev_mid is not None and prev_oim is not None:
            scale = max(1.0, osmium_vol_ema, osmium_spread_ema)
            oim_edge_sample = self._clamp(float(prev_oim) * ret / scale, -1.5, 1.5)
        osmium_oim_edge = self._ewma(data.get("osmium_oim_edge"), oim_edge_sample, self.OSMIUM_OIM_EDGE_ALPHA)
        effective_oim = oim * self._clamp(0.55 + 0.45 * osmium_oim_edge, 0.20, 1.20)

        if abs(effective_oim) >= self.OSMIUM_OIM_THRESHOLD:
            take_fraction = min(1.0, abs(effective_oim) * self.OSMIUM_OIM_TAKE_SCALE)
            edge = 0.12 * max(1.0, osmium_vol_ema)
            if effective_oim > 0 and best_ask > 0 and best_ask < osmium_fv - edge:
                pending_buys = self._place_buy(orders, product, best_ask, int(round(abs(depth.sell_orders[best_ask]) * take_fraction)), position, pending_buys)
            elif effective_oim < 0 and best_bid > 0 and best_bid > osmium_fv + edge:
                pending_sells = self._place_sell(orders, product, best_bid, int(round(depth.buy_orders[best_bid] * take_fraction)), position, pending_sells)

        projected = position + pending_buys - pending_sells
        effective_pos_ratio = (projected / self.OSMIUM_POSITION_LIMIT) + (current_mid - osmium_fv) * self.OSMIUM_FV_TETHER_SCALE
        bid_scale = max(0.0, 1.0 - max(0.0, effective_pos_ratio) * self.OSMIUM_VOLUME_SKEW_AGGRESSION)
        ask_scale = max(0.0, 1.0 + min(0.0, effective_pos_ratio) * self.OSMIUM_VOLUME_SKEW_AGGRESSION)

        bid_shift, ask_shift, b_sig, a_sig = 0, 0, 1.0, 1.0
        if effective_oim > self.OSMIUM_OIM_THRESHOLD:
            bid_shift, ask_shift, b_sig, a_sig = self.OSMIUM_OIM_SHIFT, self.OSMIUM_OIM_SHIFT, self.OSMIUM_OIM_EDGE_SCALE, self.OSMIUM_OIM_FADE_SCALE
        elif effective_oim < -self.OSMIUM_OIM_THRESHOLD:
            bid_shift, ask_shift, b_sig, a_sig = -self.OSMIUM_OIM_SHIFT, -self.OSMIUM_OIM_SHIFT, self.OSMIUM_OIM_FADE_SCALE, self.OSMIUM_OIM_EDGE_SCALE

        total_bid_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * bid_scale * b_sig))
        total_ask_qty = int(round(self.OSMIUM_BASE_QUOTE_SIZE * ask_scale * a_sig))

        # Dynamic Stress Scaling
        offset_bump = 0
        if osmium_vol_ema > self.OSMIUM_VOL_STRESS_LEVEL:
            offset_bump = 2
            total_bid_qty = int(round(total_bid_qty * 0.65))
            total_ask_qty = int(round(total_ask_qty * 0.65))

        # Settlement Flattening
        ts = state.timestamp
        if ts >= self.OSMIUM_CLOSE_REDUCE_TS:
            if projected > self.OSMIUM_CLOSE_TARGET_ABS: total_bid_qty = 0
            if projected < -self.OSMIUM_CLOSE_TARGET_ABS: total_ask_qty = 0
        elif ts >= self.OSMIUM_LATE_REDUCE_TS:
            if projected > self.OSMIUM_LATE_TARGET_ABS: total_bid_qty = 0
            if projected < -self.OSMIUM_LATE_TARGET_ABS: total_ask_qty = 0
        
        mm_bid = math.floor(current_mid) - (self.OSMIUM_INNER_OFFSET + offset_bump) + bid_shift
        mm_ask = math.ceil(current_mid) + (self.OSMIUM_INNER_OFFSET + offset_bump) + ask_shift
        
        bid1, ask1 = best_bid, best_ask
        bid2, ask2 = self._second_bid_ask(depth)
        max_b, min_a = math.floor(osmium_fv) - 1 + bid_shift, math.ceil(osmium_fv) + 1 + ask_shift

        jump_b, jump_a = False, False
        if bid1 > 0 and bid1 + 1 <= max_b: mm_bid = max(mm_bid, bid1 + 1)
        elif bid2 is not None and bid2 + 1 <= max_b: mm_bid = max(mm_bid, bid2 + 1); jump_b = True
        if ask1 > 0 and ask1 - 1 >= min_a: mm_ask = min(mm_ask, ask1 - 1)
        elif ask2 is not None and ask2 - 1 >= min_a: mm_ask = min(mm_ask, ask2 - 1); jump_a = True

        if total_bid_qty > 0 and projected < self.OSMIUM_KILL_SWITCH_THRESHOLD:
            qty = max(self.OSMIUM_L2_QUOTE_SIZE, total_bid_qty) if jump_b else total_bid_qty
            pending_buys = self._place_buy(orders, product, mm_bid, qty, position, pending_buys)
        if total_ask_qty > 0 and projected > -self.OSMIUM_KILL_SWITCH_THRESHOLD:
            qty = max(self.OSMIUM_L2_QUOTE_SIZE, total_ask_qty) if jump_a else total_ask_qty
            pending_sells = self._place_sell(orders, product, mm_ask, qty, position, pending_sells)

        data.update({
            "osmium_ema": osmium_ema, 
            "osmium_prev_mid": float(current_mid), 
            "osmium_prev_oim": float(oim),
            "osmium_vol_ema": float(osmium_vol_ema), 
            "osmium_spread_ema": float(osmium_spread_ema), 
            "osmium_oim_edge": float(osmium_oim_edge),
        })
        return orders, data

    def _pepper_base_estimate(self, current_mid, timestamp: int, stored_base=None):
        if current_mid is None: return stored_base
        observed = float(current_mid) - self.PEPPER_SLOPE * float(timestamp)
        return self._ewma(stored_base, observed, self.PEPPER_BASE_UPDATE_ALPHA)

    def _pepper_target_position(self, state: TradingState, depth: OrderDepth, position: int, resid_ema: float, data: dict) -> int:
        best_bid, best_ask = self._best_bid_ask(depth)
        stale_for = state.timestamp - data.get("pepper_last_pos_change_ts", 0)
        target = self.PEPPER_POSITION_LIMIT
        ts = state.timestamp

        if ts >= self.PEPPER_HARD_CLOSE_TS: target = 62
        elif ts >= self.PEPPER_CLOSE_REDUCE_TS: target = 78
        elif ts >= self.PEPPER_LATE_REDUCE_TS: target = 80

        if stale_for > self.PEPPER_STALE_POS_TS and ts > 30000: target = min(target, self.PEPPER_STALE_TARGET)
        if resid_ema is not None:
            if resid_ema <= self.PEPPER_STRONG_BREAK_RESID_EMA: target = min(target, self.PEPPER_STRONG_BREAK_TARGET)
            elif resid_ema <= self.PEPPER_SOFT_BREAK_RESID_EMA: target = min(target, self.PEPPER_SOFT_BREAK_TARGET)
        
        return int(target)

    def _trade_pepper_root(self, state: TradingState, base_estimate, resid_ema, data: dict) -> Tuple[List[Order], bool]:
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        if depth is None: return [], data.get("pepper_reached_80", False)
        current_mid = self._mid_price(depth)
        if current_mid is None: return [], data.get("pepper_reached_80", False)

        orders, position = [], state.position.get(product, 0)
        reached_80 = data.get("pepper_reached_80", False)
        if position >= self.PEPPER_POSITION_LIMIT: reached_80 = True

        fair = float(base_estimate or current_mid) + self.PEPPER_SLOPE * float(state.timestamp)
        fair_center = int(round(fair))
        target_pos = self._pepper_target_position(state, depth, position, resid_ema, data)
        pending_buys, pending_sells = 0, 0

        # OIM Shift logic
        bid_shift, ask_shift = 0, 0
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid > 0 and best_ask > 0:
            vol = float(depth.buy_orders[best_bid] + abs(depth.sell_orders[best_ask]))
            if vol > 0:
                oim = (depth.buy_orders[best_bid] - abs(depth.sell_orders[best_ask])) / vol
                if abs(oim) > self.PEPPER_OIM_BASE_THRESHOLD:
                    mag = min(self.PEPPER_OIM_MAX_SHIFT, int((abs(oim)-0.1)/0.9 * self.PEPPER_OIM_MAX_SHIFT) + 1)
                    if oim < 0: bid_shift = -mag
                    else: ask_shift = mag

        # Defensive Trim & Scalp
        if position > target_pos:
            margin = self.PEPPER_CLOSE_SELL_MARGIN if state.timestamp > 99000 or (resid_ema and resid_ema < -2.5) else self.PEPPER_SCALP_MIN_MARGIN
            excess = position - target_pos
            pending_sells = self._take_bids(orders, product, depth, fair_center + margin, position, pending_sells, max_total=excess)

        # AGGRESSIVE MM INJECT
        if reached_80 and position > 40:
            bid2, ask2 = self._second_bid_ask(depth)
            if bid2 and ask2 and abs(best_bid - bid2) <= self.PEPPER_MM_L2_MAX_BID_GAP and abs(ask2 - best_ask) <= self.PEPPER_MM_L2_MAX_ASK_GAP:
                mm_bid, mm_ask = best_bid + 1 + bid_shift, best_ask - 1 + ask_shift
                deficit = self.PEPPER_POSITION_LIMIT - (position + pending_buys - pending_sells)
                bid_qty = int(round(self.PEPPER_MM_BASE_QUOTE_SIZE * self.PEPPER_MM_BID_WEIGHT)) + deficit
                ask_qty = max(0, self.PEPPER_MM_BASE_QUOTE_SIZE - bid_qty) if position >= self.PEPPER_MM_MIN_LONG_POSITION else 0
                pending_buys = self._place_buy(orders, product, mm_bid, bid_qty, position, pending_buys)
                pending_sells = self._place_sell(orders, product, mm_ask, ask_qty, position, pending_sells)

        # Basic Accumulation
        remaining = target_pos - (position + pending_buys - pending_sells)
        if remaining > 0:
            if not reached_80:
                pending_buys = self._take_asks(orders, product, depth, fair_center + self.PEPPER_INITIAL_ACC_THRESH, position, pending_buys, max_total=remaining)
                remaining = target_pos - (position + pending_buys - pending_sells)
                if remaining > 0: pending_buys = self._place_buy(orders, product, fair_center, remaining, position, pending_buys)
            else:
                pending_buys = self._place_buy(orders, product, fair_center + self.PEPPER_RECOUP_MAX_MARGIN, remaining, position, pending_buys)

        data["pepper_reached_80"] = reached_80
        return orders, data

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        
        # Safe Pepper Signal Update
        pepper_mid = self._mid_price(state.order_depths.get("INTARIAN_PEPPER_ROOT"))
        base = self._pepper_base_estimate(pepper_mid, state.timestamp, data.get("pepper_base_estimate"))
        
        resid = data.get("pepper_resid_ema")
        if pepper_mid and base:
            resid_sample = float(pepper_mid) - (float(base) + self.PEPPER_SLOPE * float(state.timestamp))
            resid = self._ewma(resid, resid_sample, self.PEPPER_RESID_EMA_ALPHA)
            
        result: Dict[str, List[Order]] = {}
        
        if "ASH_COATED_OSMIUM" in state.order_depths:
            o_orders, data = self._trade_osmium(state, data)
            result["ASH_COATED_OSMIUM"] = o_orders
        
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            p_orders, data = self._trade_pepper_root(state, base, resid, data)
            result["INTARIAN_PEPPER_ROOT"] = p_orders
            
            p_pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            if p_pos != data.get("pepper_prev_position"):
                data["pepper_last_pos_change_ts"] = state.timestamp
            data["pepper_prev_position"] = p_pos

        data.update({"pepper_base_estimate": base, "pepper_resid_ema": resid, "pepper_prev_mid": pepper_mid})
        traderData = jsonpickle.encode(data)
        return result, 0, traderData