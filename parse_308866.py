import json
import collections

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
osmium_sub.sort(key=lambda t: t['timestamp'])

print("\n--- 1. VOLUME STATISTICS & FILL RATE ---")
total_osmium_vol = sum(t['quantity'] for t in osmium_sub)
buy_vol = sum(t['quantity'] for t in osmium_sub if t.get('buyer') == 'SUBMISSION')
sell_vol = sum(t['quantity'] for t in osmium_sub if t.get('seller') == 'SUBMISSION')
timestamps_with_trade = len(set(t['timestamp'] for t in osmium_sub))

print(f"Total OSMIUM trades for SUBMISSION: {len(osmium_sub)}")
print(f"Total OSMIUM volume traded: {total_osmium_vol} (Buy: {buy_vol}, Sell: {sell_vol})")
print(f"Fill Rate (Timestamps with trades): {timestamps_with_trade} out of {len(os_dict)} total timestamps ({(timestamps_with_trade/max(1, len(os_dict)))*100:.2f}%)")

print("\n--- 2. FV ALIGNMENT ---")
# Calculate average sell price vs average buy price
sells = [t for t in osmium_sub if t.get('seller') == 'SUBMISSION']
buys = [t for t in osmium_sub if t.get('buyer') == 'SUBMISSION']

avg_sell = sum(t['price'] * t['quantity'] for t in sells) / max(sum(t['quantity'] for t in sells), 1) if sells else 0
avg_buy = sum(t['price'] * t['quantity'] for t in buys) / max(sum(t['quantity'] for t in buys), 1) if buys else 0

print(f"Avg Buy : {avg_buy:.2f}")
print(f"Avg Sell: {avg_sell:.2f}")
if sells and buys:
    print(f"Spread capture: {avg_sell - avg_buy:.2f} ticks")
    print(f"Implied Mid (FV Tracker): {(avg_sell + avg_buy) / 2:.2f}")

print("\n--- 3. AGGREGATE FILL DYNAMICS (WHALE VS PASSIVE VS JUMP) ---")

sudden_jump_fills = 0
staircase_fills = 0
passive_fills = 0

for trade in osmium_sub:
    ts = trade['timestamp']
    side = "BUY" if trade.get('buyer') == 'SUBMISSION' else "SELL"
    
    if ts - 300 in os_dict and ts - 200 in os_dict and ts - 100 in os_dict and ts in os_dict:
        t_0 = os_dict[ts]
        t_1 = os_dict[ts - 100]
        t_2 = os_dict[ts - 200]
        
        try:
            b_0 = int(t_0['bid_price_1']) if t_0['bid_price_1'] else 0
            b_1 = int(t_1['bid_price_1']) if t_1['bid_price_1'] else 0
            b_2 = int(t_2['bid_price_1']) if t_2['bid_price_1'] else 0
            
            a_0 = int(t_0['ask_price_1']) if t_0['ask_price_1'] else 99999
            a_1 = int(t_1['ask_price_1']) if t_1['ask_price_1'] else 99999
            a_2 = int(t_2['ask_price_1']) if t_2['ask_price_1'] else 99999
            
            if side == "SELL":
                # If sudden bid appeared exactly at t_0
                if b_0 > b_1 + 1:
                    sudden_jump_fills += 1
                elif b_2 < b_1 and b_1 < b_0:
                    staircase_fills += 1
                else:
                    passive_fills += 1
                    
            if side == "BUY":
                if a_0 < a_1 - 1:
                    sudden_jump_fills += 1
                elif a_2 > a_1 and a_1 > a_0:
                    staircase_fills += 1
                else:
                    passive_fills += 1
        except ValueError:
            pass

classified = sudden_jump_fills + staircase_fills + passive_fills
print(f"Total classified fills: {classified}")
print(f"Fills caused by SUDDEN JUMPS (Whale appearing): {sudden_jump_fills} ({sudden_jump_fills/max(1, classified)*100:.1f}%)")
print(f"Fills caused by STAIRCASE (Penny Jumping): {staircase_fills} ({staircase_fills/max(1, classified)*100:.1f}%)")
print(f"Fills caused by PASSIVE HITS (Stable resting orders): {passive_fills} ({passive_fills/max(1, classified)*100:.1f}%)")

print("\n--- 4. GENERAL PENNY-JUMPING DYNAMICS (ALL TIMESTAMPS) ---")
count_spread_shrink = 0
count_total_moves = 0
staircase_lengths = []
current_staircase = 0

prev_ts = min(os_dict.keys())
prev_b1 = int(os_dict[prev_ts]['bid_price_1']) if os_dict[prev_ts]['bid_price_1'] else 0
prev_a1 = int(os_dict[prev_ts]['ask_price_1']) if os_dict[prev_ts]['ask_price_1'] else 99999

for t in sorted(os_dict.keys())[1:]:
    if t - 100 == prev_ts:
        curr = os_dict[t]
        try:
            c_b1 = int(curr['bid_price_1']) if curr['bid_price_1'] else 0
            c_a1 = int(curr['ask_price_1']) if curr['ask_price_1'] else 99999
            
            c_spread = c_a1 - c_b1
            p_spread = prev_a1 - prev_b1
            
            if c_spread < p_spread and (c_b1 == prev_b1 + 1 or c_a1 == prev_a1 - 1):
                count_spread_shrink += 1
                current_staircase += 1
            else:
                if current_staircase > 0:
                    staircase_lengths.append(current_staircase)
                current_staircase = 0
                
            if c_spread != p_spread or c_b1 != prev_b1:
                count_total_moves += 1
            
            prev_b1 = c_b1
            prev_a1 = c_a1
            
        except ValueError:
            pass
            
    prev_ts = t

print(f"Total orderbook state updates (moves): {count_total_moves}")
print(f"Number of times spread shrunk by exactly 1 tick (penny jumping): {count_spread_shrink}")
print(f"Average length of a consecutive penny-jump staircase: {sum(staircase_lengths)/len(staircase_lengths) if staircase_lengths else 0:.2f} ticks")
print(f"Max staircase length observed: {max(staircase_lengths) if staircase_lengths else 0}")

print("\n--- 5. PROFITABILITY ---")
try:
    final_osmium = float(osmium_log[-1]['profit_and_loss'])
    print(f"Final OSMIUM PnL: {final_osmium}")
    
    pepper_log = [r for r in parsed_log if r.get('product') == 'INTARIAN_PEPPER_ROOT']
    final_pepper = float(pepper_log[-1]['profit_and_loss']) if pepper_log else 0
    print(f"Final PEPPER PnL: {final_pepper}")
except Exception as e:
    print("Could not retrieve final PnL:", e)

