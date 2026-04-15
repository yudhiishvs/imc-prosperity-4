from datamodel import OrderDepth, TradingState, Order, ProsperityEncoder, Symbol
from typing import Any, Dict, List, Tuple
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Minimal flush to satisfy the visualizer/backtester format.
        print(
            json.dumps(
                [
                    [state.timestamp, "", [], {k: [v.buy_orders, v.sell_orders] for k, v in state.order_depths.items()}, [], [], state.position, [{}, {}]],
                    [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
                    conversions,
                    trader_data,
                    self.logs,
                ],
                cls=ProsperityEncoder,
                separators=(",", ":"),
            )
        )
        self.logs = ""


logger = Logger()


class Trader:
    """
    Baseline: Buy-and-hold INTARIAN_PEPPER_ROOT.
    - Buys up to the position limit as quickly as possible.
    - Never sells (ignores any opportunities to exit).
    """

    PEPPER_POSITION_LIMIT = 80

    @staticmethod
    def _best_ask(depth: OrderDepth) -> Tuple[int, int]:
        if not depth.sell_orders:
            return 0, 0
        best_ask = min(depth.sell_orders.keys())
        best_ask_vol = -depth.sell_orders[best_ask]
        return best_ask, best_ask_vol

    def run(self, state: TradingState):
        product = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(product)
        pos = state.position.get(product, 0)

        orders: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = ""

        if depth is not None and pos < self.PEPPER_POSITION_LIMIT and depth.sell_orders:
            room = self.PEPPER_POSITION_LIMIT - pos
            # Sweep asks starting from best ask until filled or book exhausted.
            olist: List[Order] = []
            for ask in sorted(depth.sell_orders.keys()):
                if room <= 0:
                    break
                avail = -depth.sell_orders[ask]
                qty = min(room, avail)
                if qty > 0:
                    olist.append(Order(product, int(ask), int(qty)))
                    room -= qty
            if olist:
                orders[product] = olist

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

