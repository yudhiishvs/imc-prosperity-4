import json
from collections import defaultdict

log_file = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/best_strat/307924.log"

with open(log_file) as f:
    data = json.load(f)

trades = data.get("tradeHistory", [])

print("--- OSMIUM TRADE ANALYSIS ---")
# trades is a list of dictionary or lists. Wait, tradeHistory is usually a JSON string? or list of dicts.
# Let's see how tradeHistory is structured. If it's a list of dicts:
if isinstance(trades, str):
    trades = json.loads(trades)

submission_trades = [t for t in trades if t.get('buyer') == 'SUBMISSION' or t.get('seller') == 'SUBMISSION']
osmium_sub = [t for t in submission_trades if t.get('symbol') == 'ASH_COATED_OSMIUM']

print(f"Total OSMIUM trades for SUBMISSION: {len(osmium_sub)}")

if osmium_sub:
    prices = defaultdict(int)
    for t in osmium_sub:
        prices[t.get('price')] += t.get('quantity')

    print("Summary of OSMIUM execute prices (Price: Vol):")
    for p in sorted(prices.keys()):
        print(f"{p}: {prices[p]}")

    # Calculate average sell price vs average buy price
    sells = [t for t in osmium_sub if t.get('seller') == 'SUBMISSION']
    buys = [t for t in osmium_sub if t.get('buyer') == 'SUBMISSION']

    if sells:
        avg_sell = sum(t['price'] * t['quantity'] for t in sells) / sum(t['quantity'] for t in sells)
        print(f"\nAvg Sell: {avg_sell:.2f}")
    if buys:    
        avg_buy = sum(t['price'] * t['quantity'] for t in buys) / sum(t['quantity'] for t in buys)
        print(f"Avg Buy : {avg_buy:.2f}")
        
    if sells and buys:
        print(f"Spread capture: {avg_sell - avg_buy:.2f} ticks")
else:
    print("No SUBMISSION trades found for osmium, let's explore if 'buyer'/'seller' keys are different.")
    if len(trades) > 0:
        print("Example trade obj: ", trades[0])
