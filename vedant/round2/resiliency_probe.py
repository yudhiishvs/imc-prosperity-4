from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:

    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == 'ASH_COATED_OSMIUM':
                fv = 10004
                
                # We want to perform two tests:
                # 1. Snipe 50% of the best ask/bid.
                # 2. Penny jump the best bid/ask.
                
                # Check for best bid
                if len(order_depth.buy_orders) > 0:
                    best_bid_price = max(order_depth.buy_orders.keys())
                    best_bid_vol = order_depth.buy_orders[best_bid_price]
                    
                    # Test 1: Snipe exactly 50% of the best bid (aggressively sell)
                    snipe_vol = max(1, best_bid_vol // 2)
                    orders.append(Order(product, best_bid_price, -snipe_vol))
                    
                    # Test 2: Penny-jump the best bid (passively bid 1 tick higher)
                    jump_price = best_bid_price + 1
                    orders.append(Order(product, jump_price, 1))

                # Check for best ask
                if len(order_depth.sell_orders) > 0:
                    best_ask_price = min(order_depth.sell_orders.keys())
                    best_ask_vol = order_depth.sell_orders[best_ask_price]
                    
                    # Test 1: Snipe exactly 50% of the best ask (aggressively buy)
                    # Sell volume is negative, so volume // 2 is still negative, we want positive order to buy
                    snipe_vol = max(1, abs(best_ask_vol) // 2)
                    orders.append(Order(product, best_ask_price, snipe_vol))
                    
                    # Test 2: Penny-jump the best ask (passively sell 1 tick lower)
                    jump_price = best_ask_price - 1
                    orders.append(Order(product, jump_price, -1))
                
                result[product] = orders

        return result, 1, "Resiliency Probe"
