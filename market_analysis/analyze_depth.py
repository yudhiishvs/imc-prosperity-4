import json

log_file = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/308866/308866.log"

with open(log_file) as f:
    data = json.load(f)

activities = data.get("activitiesLog", [])

if isinstance(activities, str):
    activities = activities.strip().split('\n')
    header = activities[0].split(';')
    parsed_log = []
    for line in activities[1:]:
        if not line: continue
        row = line.split(';')
        parsed_log.append(dict(zip(header, row)))
else:
    parsed_log = activities

osmium_log = [r for r in parsed_log if r.get('product') == 'ASH_COATED_OSMIUM']
os_dict = {int(r['timestamp']): r for r in osmium_log}

trades = data.get("tradeHistory", [])
if isinstance(trades, str):
    trades = json.loads(trades)

submission_trades = [t for t in trades if t.get('buyer') == 'SUBMISSION' or t.get('seller') == 'SUBMISSION']
osmium_sub = [t for t in submission_trades if t.get('symbol') == 'ASH_COATED_OSMIUM']

print("--- QUOTE DEPTH ANALYSIS ---")

distances = []

for trade in osmium_sub:
    ts = trade['timestamp']
    price = trade['price']
    side = "BUY" if trade.get('buyer') == 'SUBMISSION' else "SELL"
    
    # Check the state at ts-100 (the state we quoted against)
    prev_ts = ts - 100
    if prev_ts in os_dict:
        state = os_dict[prev_ts]
        try:
            b1 = int(state['bid_price_1']) if state['bid_price_1'] else None
            a1 = int(state['ask_price_1']) if state['ask_price_1'] else None
            
            if side == "BUY":
                if b1 is not None:
                    dist = price - b1
                    distances.append(dist)
                    # print(f"TS {ts}: We bought at {price}, Best Bid was {b1}. Distance: {dist}")
            else:
                if a1 is not None:
                    dist = a1 - price
                    distances.append(dist)
                    # print(f"TS {ts}: We sold at {price}, Best Ask was {a1}. Distance: {dist}")
        except ValueError:
            pass

if distances:
    avg_dist = sum(distances) / len(distances)
    print(f"Average distance of our fill from market L1: {avg_dist:.2f} ticks")
    print(f"Max distance: {max(distances)} ticks")
    print(f"Min distance: {min(distances)} ticks")
    
    penny_jumps = [d for d in distances if d == 1]
    deep_quotes = [d for d in distances if d > 1]
    
    print(f"Percentage of fills that were 'True Penny Jumps' (Distance == 1): {(len(penny_jumps)/len(distances))*100:.2f}%")
    print(f"Percentage of fills that were 'Deep Inside' (Distance > 1): {(len(deep_quotes)/len(distances))*100:.2f}%")
else:
    print("No distances recorded.")
