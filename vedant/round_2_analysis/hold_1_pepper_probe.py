"""
Submission-only hold-1 probe for INTARIAN_PEPPER_ROOT.

Purpose:
- Recover server mark path for Pepper via:
  server_mark(t) = pnl(t) + buy_price
after buying exactly 1 unit and holding.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    from prosperity4bt.datamodel import OrderDepth, TradingState, Order


PRODUCT = "INTARIAN_PEPPER_ROOT"
PROBE_QTY = 1


class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders: Dict[str, List[Order]] = {}

        depth = state.order_depths.get(PRODUCT)
        if depth is None:
            return orders, 0, ""

        pos = state.position.get(PRODUCT, 0)
        qty_needed = PROBE_QTY - pos
        if qty_needed > 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            orders[PRODUCT] = [Order(PRODUCT, best_ask, qty_needed)]

        return orders, 0, ""

