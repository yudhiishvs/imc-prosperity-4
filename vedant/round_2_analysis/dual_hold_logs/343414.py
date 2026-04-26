"""
Submission-only dual hold probe.

Buys and holds exactly 1 unit of each Round 2 product:
- ASH_COATED_OSMIUM
- INTARIAN_PEPPER_ROOT

Purpose:
- Recover both product mark paths from a single run.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    from prosperity4bt.datamodel import OrderDepth, TradingState, Order


TARGET_QTY = 1
PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")


class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders: Dict[str, List[Order]] = {}

        for product in PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            pos = state.position.get(product, 0)
            qty_needed = TARGET_QTY - pos
            if qty_needed > 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                orders[product] = [Order(product, best_ask, qty_needed)]

        return orders, 0, ""