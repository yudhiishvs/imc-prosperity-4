from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle

# ─────────────────────────────────────────────────────────────────────────────
# v1_round2_prosperity.py — Baseline strategy from EDA findings
#
# ASH_COATED_OSMIUM
#   Roll (1984) bid-ask bounce model. Fair value = 10,000 (constant).
#   Mean reversion half-life 2-3 ticks. Spread median 16 ticks.
#   All trades land exactly at bid or ask — never inside spread.
#   Two bot types: whale (single crossing trade, reverses in 1-2 ticks, 30%
#   of breaks) and reverter (oscillates around fair, 9:1 ratio vs whale).
#   67% of price breaks have no trade — phantom reshuffling, ignore them.
#   Strategy: passive quotes anchored to fair value (not mid), immediate
#   re-post after fill, inventory skew to stay flat.
#
# INTARIAN_PEPPER_ROOT
#   Linear trend +0.1001 ticks/timestamp = +1000 ticks/day (R²=0.9999).
#   Bots trade near mid (inside half-spread), no aggressive spread crossing.
#   Never short. Get to +80 as fast as possible, hold all day.
#   Fair value at time t estimated from first observed price + slope * elapsed.
#   Strategy: take any ask at or below fair + buffer, post aggressive bids
#   at fair - small_offset to fill passively as trend moves up.
# ─────────────────────────────────────────────────────────────────────────────


class Trader:

    # ── ASH constants ─────────────────────────────────────────────────────────
    ASH = "ASH_COATED_OSMIUM"
    ASH_FAIR        = 10_000        # known constant fair value
    ASH_POS_LIMIT   = 80
    ASH_QUOTE_DIST  = 5             # distance from fair value for passive quotes
    ASH_QUOTE_SIZE  = 14            # matches observed L1 volume
    ASH_L2_DIST     = 8             # secondary backstop quote distance
    ASH_L2_SIZE     = 20            # larger than L1 (matches observed L2 ~24)
    ASH_SOFT_LIMIT  = 40            # reduce aggression above this
    ASH_HARD_LIMIT  = 70            # emergency flatten above this

    # ── PEPPER constants ──────────────────────────────────────────────────────
    PEPPER = "INTARIAN_PEPPER_ROOT"
    PEPPER_SLOPE     = 0.1001       # ticks per timestamp unit
    PEPPER_POS_LIMIT = 80
    PEPPER_TAKE_BUF  = 3            # take asks up to fair + this many ticks
    PEPPER_BID_OFF   = 3            # post passive bids at fair - this
    PEPPER_BID_SIZE  = 20           # size per passive bid order
    PEPPER_HARD_BUY  = 8            # if below this position, buy aggressively

    def _load(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            d = jsonpickle.decode(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    # ── Order book helpers ────────────────────────────────────────────────────

    @staticmethod
    def _best(depth: OrderDepth) -> Tuple[int, int]:
        bid = max(depth.buy_orders)  if depth.buy_orders  else 0
        ask = min(depth.sell_orders) if depth.sell_orders else 0
        return bid, ask

    def _buy_room(self, pos: int, pb: int) -> int:
        return self.ASH_POS_LIMIT - (pos + pb)

    def _sell_room(self, pos: int, ps: int) -> int:
        return self.ASH_POS_LIMIT + (pos - ps)

    def _buy(self, orders, product, price, qty, pos, pb, limit=None) -> int:
        room = self.ASH_POS_LIMIT - (pos + pb) if product == self.ASH \
               else self.PEPPER_POS_LIMIT - (pos + pb)
        qty = min(qty, room)
        if limit is not None:
            qty = min(qty, limit)
        if qty > 0:
            orders.append(Order(product, price, qty))
            pb += qty
        return pb

    def _sell(self, orders, product, price, qty, pos, ps, limit=None) -> int:
        room = self.ASH_POS_LIMIT + (pos - ps) if product == self.ASH \
               else self.PEPPER_POS_LIMIT + (pos - ps)
        qty = min(qty, room)
        if limit is not None:
            qty = min(qty, limit)
        if qty > 0:
            orders.append(Order(product, price, -qty))
            ps += qty
        return ps

    # ─────────────────────────────────────────────────────────────────────────
    # ASH strategy
    # ─────────────────────────────────────────────────────────────────────────

    def _trade_ash(self, state: TradingState) -> List[Order]:
        depth = state.order_depths.get(self.ASH)
        if depth is None:
            return []

        orders: List[Order] = []
        pos = state.position.get(self.ASH, 0)
        pb = ps = 0
        fair = self.ASH_FAIR

        # ── Layer 1: Take any mispriced liquidity ─────────────────────────
        # Ask below fair → free money, buy it
        for ask in sorted(depth.sell_orders):
            if ask >= fair:
                break
            vol = -depth.sell_orders[ask]
            pb = self._buy(orders, self.ASH, ask, vol, pos, pb)

        # Bid above fair → free money, sell it
        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair:
                break
            vol = depth.buy_orders[bid]
            ps = self._sell(orders, self.ASH, bid, vol, pos, ps)

        # ── Layer 2: Inventory-skewed passive quotes ───────────────────────
        # Projected position after layer 1
        proj = pos + pb - ps

        # Scale quote sizes based on inventory — reduce side that adds exposure
        # Full size when flat, zero when at soft limit
        long_ratio  = max(0.0, proj / self.ASH_SOFT_LIMIT)
        short_ratio = max(0.0, -proj / self.ASH_SOFT_LIMIT)

        bid_scale = max(0.0, 1.0 - long_ratio)
        ask_scale = max(0.0, 1.0 - short_ratio)

        # Hard stops: don't add to a position already at/near limit
        if proj >= self.ASH_HARD_LIMIT:
            bid_scale = 0.0
        if proj <= -self.ASH_HARD_LIMIT:
            ask_scale = 0.0

        # Primary L1 quotes: fair ± QUOTE_DIST
        # Findings: quote at real executable levels, not mid.
        # L1-L2 gap always 2-3 ticks → we can sit 1 tick inside existing L1.
        best_bid, best_ask = self._best(depth)

        # Bid: sit just inside the best visible bid but no higher than fair-1
        if best_bid > 0:
            passive_bid = min(best_bid + 1, fair - 1)
        else:
            passive_bid = fair - self.ASH_QUOTE_DIST

        # Ask: sit just inside the best visible ask but no lower than fair+1
        if best_ask > 0:
            passive_ask = max(best_ask - 1, fair + 1)
        else:
            passive_ask = fair + self.ASH_QUOTE_DIST

        # Clamp to fair value bounds so we never cross our own fair value
        passive_bid = min(passive_bid, fair - 1)
        passive_ask = max(passive_ask, fair + 1)

        bid_qty = int(self.ASH_QUOTE_SIZE * bid_scale)
        ask_qty = int(self.ASH_QUOTE_SIZE * ask_scale)

        if bid_qty > 0:
            pb = self._buy(orders, self.ASH, passive_bid, bid_qty, pos, pb)
        if ask_qty > 0:
            ps = self._sell(orders, self.ASH, passive_ask, ask_qty, pos, ps)

        # Secondary L2 backstop quotes: fair ± L2_DIST, larger size
        # Absorbs the whale if it blows through L1
        if bid_scale > 0:
            l2_bid = fair - self.ASH_L2_DIST
            if l2_bid < passive_bid:
                l2_bid_qty = int(self.ASH_L2_SIZE * bid_scale)
                if l2_bid_qty > 0:
                    pb = self._buy(orders, self.ASH, l2_bid, l2_bid_qty, pos, pb)

        if ask_scale > 0:
            l2_ask = fair + self.ASH_L2_DIST
            if l2_ask > passive_ask:
                l2_ask_qty = int(self.ASH_L2_SIZE * ask_scale)
                if l2_ask_qty > 0:
                    ps = self._sell(orders, self.ASH, l2_ask, l2_ask_qty, pos, ps)

        # ── Layer 3: Emergency flatten ─────────────────────────────────────
        # Findings: fat tails (kurtosis ~3) → hard limit at 70 to survive
        # consecutive same-direction whale hits
        proj = pos + pb - ps
        if proj > self.ASH_HARD_LIMIT and best_bid > 0:
            flatten = proj - self.ASH_SOFT_LIMIT
            ps = self._sell(orders, self.ASH, best_bid, flatten, pos, ps)
        elif proj < -self.ASH_HARD_LIMIT and best_ask > 0:
            flatten = abs(proj) - self.ASH_SOFT_LIMIT
            pb = self._buy(orders, self.ASH, best_ask, flatten, pos, pb)

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # PEPPER strategy
    # ─────────────────────────────────────────────────────────────────────────

    def _pepper_fair(self, ts: int, state: dict) -> float:
        """
        Estimate PEPPER fair value at this timestamp.
        First tick: anchor to mid price. Subsequent ticks: anchor + slope * elapsed.
        Findings: slope = +0.1001/ts, R² = 0.9999 — essentially deterministic.
        """
        first_mid = state.get("pepper_first_mid")
        first_ts  = state.get("pepper_first_ts")
        if first_mid is None or first_ts is None:
            return None
        return first_mid + self.PEPPER_SLOPE * (ts - first_ts)

    def _trade_pepper(self, state: TradingState, data: dict) -> List[Order]:
        depth = state.order_depths.get(self.PEPPER)
        if depth is None:
            return []

        orders: List[Order] = []
        pos = state.position.get(self.PEPPER, 0)
        pb = ps = 0
        ts = state.timestamp

        best_bid, best_ask = self._best(depth)
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None

        # Initialise anchor on first tick
        if data.get("pepper_first_mid") is None and mid is not None:
            data["pepper_first_mid"] = mid
            data["pepper_first_ts"]  = ts

        fair = self._pepper_fair(ts, data)
        if fair is None:
            fair = mid if mid is not None else 12000.0

        pb_room = self.PEPPER_POS_LIMIT - (pos + pb)

        # ── Layer 1: Take asks at or below fair + buffer ───────────────────
        # Findings: trend guarantees fair rises 0.1/ts — any fill at current
        # fair is profitable. Buffer of 3 ticks gives wiggle room for spread.
        if pb_room > 0:
            for ask in sorted(depth.sell_orders):
                if ask > fair + self.PEPPER_TAKE_BUF:
                    break
                vol = -depth.sell_orders[ask]
                pb = self._buy(orders, self.PEPPER, ask, vol, pos, pb)
                pb_room = self.PEPPER_POS_LIMIT - (pos + pb)
                if pb_room <= 0:
                    break

        # ── Layer 2: Aggressive fill if position dangerously low ──────────
        # If we're far below 80 and there are asks available, buy more
        # aggressively to not miss the trend
        proj = pos + pb - ps
        if proj < self.PEPPER_HARD_BUY and best_ask > 0:
            gap = self.PEPPER_HARD_BUY - proj
            pb = self._buy(orders, self.PEPPER, best_ask, gap, pos, pb)

        # ── Layer 3: Passive bids at fair - offset ─────────────────────────
        # Post bids slightly below fair so the trend walks price up through us.
        # Findings: bots trade inside spread (~mid), so we will fill passively.
        proj = pos + pb - ps
        if proj < self.PEPPER_POS_LIMIT:
            bid_price = int(fair - self.PEPPER_BID_OFF)
            bid_qty   = min(self.PEPPER_BID_SIZE, self.PEPPER_POS_LIMIT - proj)
            if bid_qty > 0 and bid_price > 0:
                pb = self._buy(orders, self.PEPPER, bid_price, bid_qty, pos, pb)

        # ── Safety: never go short on PEPPER ──────────────────────────────
        # Findings: +1000 ticks/day means a short bleeds 0.1 ticks/ts. Buy back
        # immediately at any cost.
        proj = pos + pb - ps
        if proj < 0 and best_ask > 0:
            cover = abs(proj)
            pb = self._buy(orders, self.PEPPER, best_ask + 2, cover, pos, pb)

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load(state.traderData)

        result: Dict[str, List[Order]] = {}

        if self.ASH in state.order_depths:
            result[self.ASH] = self._trade_ash(state)

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self._trade_pepper(state, data)

        return result, 0, jsonpickle.encode(data)