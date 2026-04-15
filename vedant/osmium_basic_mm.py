from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


class Trader:
    """
    Baseline Osmium market maker:
    - Static fair value (10_000)
    - Pure penny jumping inside the spread
    - Mispricing takes around fair
    - Flatten inventory only when fair is directly tradable
    No OIM skew, no inventory-based quote skew.
    """

    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR = 10_000
    LIMIT = 80
    BASE_QTY = 10

    @staticmethod
    def _best_bid_ask(depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 0
        return best_bid, best_ask

    def _buy_room(self, position: int, pending_buys: int) -> int:
        return self.LIMIT - (position + pending_buys)

    def _sell_room(self, position: int, pending_sells: int) -> int:
        return self.LIMIT + (position - pending_sells)

    def _place_buy(
        self,
        orders: List[Order],
        price: int,
        desired_qty: int,
        position: int,
        pending_buys: int,
    ) -> int:
        qty = min(desired_qty, self._buy_room(position, pending_buys))
        if qty > 0:
            orders.append(Order(self.PRODUCT, price, qty))
            pending_buys += qty
        return pending_buys

    def _place_sell(
        self,
        orders: List[Order],
        price: int,
        desired_qty: int,
        position: int,
        pending_sells: int,
    ) -> int:
        qty = min(desired_qty, self._sell_room(position, pending_sells))
        if qty > 0:
            orders.append(Order(self.PRODUCT, price, -qty))
            pending_sells += qty
        return pending_sells

    def _take_mispriced(
        self,
        depth: OrderDepth,
        orders: List[Order],
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        # Buy asks cheaper than fair.
        for ask in sorted(depth.sell_orders.keys()):
            if ask >= self.FAIR:
                break
            avail = -depth.sell_orders[ask]
            pending_buys = self._place_buy(orders, ask, avail, position, pending_buys)

        # Sell bids richer than fair.
        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid <= self.FAIR:
                break
            avail = depth.buy_orders[bid]
            pending_sells = self._place_sell(orders, bid, avail, position, pending_sells)

        return pending_buys, pending_sells

    def _flatten_at_fair(
        self,
        depth: OrderDepth,
        orders: List[Order],
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        projected = position + pending_buys - pending_sells
        if projected > 0 and self.FAIR in depth.buy_orders:
            pending_sells = self._place_sell(
                orders, self.FAIR, min(projected, depth.buy_orders[self.FAIR]), position, pending_sells
            )
        elif projected < 0 and self.FAIR in depth.sell_orders:
            pending_buys = self._place_buy(
                orders, self.FAIR, min(abs(projected), -depth.sell_orders[self.FAIR]), position, pending_buys
            )
        return pending_buys, pending_sells

    def _penny_jump(
        self,
        depth: OrderDepth,
        orders: List[Order],
        position: int,
        pending_buys: int,
        pending_sells: int,
    ) -> Tuple[int, int]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid <= 0 or best_ask <= 0:
            return pending_buys, pending_sells

        bid_px = min(best_bid + 1, self.FAIR - 1)
        ask_px = max(best_ask - 1, self.FAIR + 1)

        pending_buys = self._place_buy(orders, bid_px, self.BASE_QTY, position, pending_buys)
        pending_sells = self._place_sell(orders, ask_px, self.BASE_QTY, position, pending_sells)
        return pending_buys, pending_sells

    def run(self, state: TradingState):
        depth = state.order_depths.get(self.PRODUCT)
        if depth is None:
            return {}, 0, ""

        orders: List[Order] = []
        position = state.position.get(self.PRODUCT, 0)
        pb = ps = 0

        pb, ps = self._take_mispriced(depth, orders, position, pb, ps)
        pb, ps = self._flatten_at_fair(depth, orders, position, pb, ps)
        pb, ps = self._penny_jump(depth, orders, position, pb, ps)

        return {self.PRODUCT: orders}, 0, ""

