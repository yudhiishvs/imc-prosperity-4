"""
Submission-only Phase 1 probe for ASH_COATED_OSMIUM.

Upload this file to Prosperity when you want to run the hold-1-unit calibration.
Do not mix offline analysis code into this file.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    from prosperity4bt.datamodel import OrderDepth, TradingState, Order


PRODUCT = "ASH_COATED_OSMIUM"
PROBE_QTY = 1


class Trader:
    """
    Buys exactly one unit of ASH_COATED_OSMIUM at the first visible ask and then
    never intentionally trades again.

    Once the fill occurs, the submission holds exactly one unit, so the server
    PnL path can be inverted offline as:

      server_mark(t) = pnl(t) + buy_price
    """

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        orders: Dict[str, List[Order]] = {}

        depth = state.order_depths.get(PRODUCT)
        if depth is None:
            return orders, 0, ""

        position = state.position.get(PRODUCT, 0)
        qty_needed = PROBE_QTY - position

        if qty_needed > 0 and depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            orders[PRODUCT] = [Order(PRODUCT, best_ask, qty_needed)]

        return orders, 0, ""
