from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import random

class Trader:
    """
    ============================================
    BOT_CHECK v3 — ASH_COATED_OSMIUM:
      - Every tick: sweep the ENTIRE orderbook (take ALL bids & ALL asks) with no price caps.
      - After sweeping, post a bid at (fair - PROBE_OFFSET) and an ask at (fair + PROBE_OFFSET).
      - PROBE_OFFSET is randomized every tick within [MIN_OFS, MAX_OFS] to map the fill curve.
      - Inventory reduction: always post/take at fair to stay flat if skewed.

    INTARIAN_PEPPER_ROOT: DISABLED.
    ============================================
    """

    OSMIUM_FAIR_VALUE     = 10_004
    OSMIUM_POSITION_LIMIT = 80

    # The range to randomly test probing out empty-book liquidity
    # Progressively narrow these bounds across multiple submissions
    MIN_PROBE_OFFSET = 11
    MAX_PROBE_OFFSET = 13

    # Size to post after clearing
    PROBE_QUOTE_SIZE = 80

    POSITION_LIMIT = {"ASH_COATED_OSMIUM": 80}

    # ── Helpers ──────────────────────────────────────────────
    def _get_position_limit(self, product: str) -> int:
        return self.OSMIUM_POSITION_LIMIT if product == "ASH_COATED_OSMIUM" else 20

    def _buy_room(self, product: str, position: int, pending_buys: int) -> int:
        return self._get_position_limit(product) - (position + pending_buys)

    def _sell_room(self, product: str, position: int, pending_sells: int) -> int:
        return self._get_position_limit(product) + (position - pending_sells)

    def _place_buy(self, orders, product, price, desired_qty, position, pending_buys):
        room = self._buy_room(product, position, pending_buys)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, qty))
            pending_buys += qty
        return pending_buys

    def _place_sell(self, orders, product, price, desired_qty, position, pending_sells):
        room = self._sell_room(product, position, pending_sells)
        qty = min(desired_qty, room)
        if qty > 0:
            orders.append(Order(product, price, -qty))
            pending_sells += qty
        return pending_sells

    # ── Main Osmium Logic ─────────────────────────────────────
    def _trade_osmium(self, state: TradingState) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(product)
        if depth is None:
            return []

        orders: List[Order] = []
        position = state.position.get(product, 0)
        pending_buys = 0
        pending_sells = 0
        fair = self.OSMIUM_FAIR_VALUE

        # ── PHASE 1: Sweep ENTIRE orderbook (No price limits) ──
        # Take EVERY ask on the book.
        for ask in sorted(depth.sell_orders.keys()):
            room = self._buy_room(product, position, pending_buys)
            if room <= 0:
                break
            qty = min(-depth.sell_orders[ask], room)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                pending_buys += qty

        # Take EVERY bid on the book.
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            room = self._sell_room(product, position, pending_sells)
            if room <= 0:
                break
            qty = min(depth.buy_orders[bid], room)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                pending_sells += qty

        # ── PHASE 2: Inventory flatten at fair ──
        # (Handles any position built from sweeping ticks)
        projected = position + pending_buys - pending_sells
        if projected > 0:
            # Post passive asks at fair to unwind longs
            pending_sells = self._place_sell(
                orders, product, fair, projected, position, pending_sells
            )
        elif projected < 0:
            # Post passive bids at fair to unwind shorts
            pending_buys = self._place_buy(
                orders, product, fair, abs(projected), position, pending_buys
            )

        # ── PHASE 3: Post random extreme probe quotes ──
        probe_offset = random.randint(self.MIN_PROBE_OFFSET, self.MAX_PROBE_OFFSET)
        probe_bid = int(fair - probe_offset)
        probe_ask = int(fair + probe_offset)

        pending_buys = self._place_buy(
            orders, product, probe_bid, self.PROBE_QUOTE_SIZE, position, pending_buys
        )
        pending_sells = self._place_sell(
            orders, product, probe_ask, self.PROBE_QUOTE_SIZE, position, pending_sells
        )

        return orders

    # ── Main Entry ────────────────────────────────────────────
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osmium(state)

        return result, 0, ""
