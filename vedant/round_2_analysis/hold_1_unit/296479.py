from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    ============================================
    Phase 1 Calibration: HOLD-1-UNIT TRADER
    ============================================
    This trader simply buys exactly 1 unit of ASH_COATED_OSMIUM 
    on the first possible tick and holds it forever.
    
    By tracking the PnL of this strategy over time, we can back out
    the internal continuous Fair Value (FV) used by the matching engine.
    PnL(t) = server_FV(t) - buy_price
    ============================================
    """
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        
        product = "ASH_COATED_OSMIUM"
        
        if product in state.order_depths:
            position = state.position.get(product, 0)
            
            # If we don't own it yet, buy exactly 1 unit.
            if position == 0:
                order_depth: OrderDepth = state.order_depths[product]
                # Check for available sell orders to take
                if len(order_depth.sell_orders) > 0:
                    # Get the best ask (the lowest price someone is willing to sell for)
                    best_ask = min(order_depth.sell_orders.keys())
                    
                    # Send a buy order for 1 unit at the best ask price
                    result[product] = [Order(product, best_ask, 1)]
                    
        return result, 0, ""