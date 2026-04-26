"""
Submission-only Osmium flip probe.

Behavior:
1. Buy exactly 1 ASH_COATED_OSMIUM at first ask.
2. After FLIP_START_TIMESTAMP, sell that 1 unit at best bid.
3. Stay flat afterward.

Purpose:
- Validate mark reconstruction during hold segment.
- Validate that PnL becomes flat after position is closed.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    from prosperity4bt.datamodel import OrderDepth, TradingState, Order


PRODUCT = "ASH_COATED_OSMIUM"
FLIP_START_TIMESTAMP = 20000


class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders: Dict[str, List[Order]] = {}
        depth = state.order_depths.get(PRODUCT)
        if depth is None:
            return orders, 0, ""

        pos = state.position.get(PRODUCT, 0)
        ts = int(state.timestamp)

        # Entry leg: buy one unit at first visible ask.
        if pos <= 0 and ts == 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            orders[PRODUCT] = [Order(PRODUCT, best_ask, 1)]
            return orders, 0, ""

        # Exit leg: after flip time, liquidate any long at best bid.
        if pos > 0 and ts >= FLIP_START_TIMESTAMP and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            orders[PRODUCT] = [Order(PRODUCT, best_bid, -pos)]

        return orders, 0, ""

