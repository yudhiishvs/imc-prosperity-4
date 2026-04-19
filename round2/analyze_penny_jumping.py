import json
import collections

log_file = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/best_strat/307924.log"

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

print("\n--- AGGREGATE FILL DYNAMICS ---")

sudden_jump_fills = 0
staircase_fills = 0
passive_fills = 0

for trade in osmium_sub:
    ts = trade['timestamp']
    price = trade['price']
    side = "BUY" if trade.get('buyer') == 'SUBMISSION' else "SELL"
    
    # Analyze the 3 timestamps before
    if ts - 300 in os_dict and ts - 200 in os_dict and ts - 100 in os_dict and ts in os_dict:
        t_0 = os_dict[ts]
        t_1 = os_dict[ts - 100]
        t_2 = os_dict[ts - 200]
        t_3 = os_dict[ts - 300]
        
        try:
            b_0 = int(t_0['bid_price_1']) if t_0['bid_price_1'] else 0
            b_1 = int(t_1['bid_price_1']) if t_1['bid_price_1'] else 0
            b_2 = int(t_2['bid_price_1']) if t_2['bid_price_1'] else 0
            b_3 = int(t_3['bid_price_1']) if t_3['bid_price_1'] else 0
            
            a_0 = int(t_0['ask_price_1']) if t_0['ask_price_1'] else 99999
            a_1 = int(t_1['ask_price_1']) if t_1['ask_price_1'] else 99999
            a_2 = int(t_2['ask_price_1']) if t_2['ask_price_1'] else 99999
            a_3 = int(t_3['ask_price_1']) if t_3['ask_price_1'] else 99999
            
            if side == "SELL":
                # We sold. This means either someone hit our ask, or we swept their bid.
                # If there's a sudden bid that popped up exactly at t_0 far above b_1
                if b_0 > b_1 + 1:
                    sudden_jump_fills += 1
                elif b_2 < b_1 and b_1 < b_0:
                    staircase_fills += 1
                else:
                    passive_fills += 1
                    
            if side == "BUY":
                # We bought. This means someone hit our bid, or we swept their ask.
                if a_0 < a_1 - 1:
                    sudden_jump_fills += 1
                elif a_2 > a_1 and a_1 > a_0:
                    staircase_fills += 1
                else:
                    passive_fills += 1
        except ValueError:
            pass

print(f"Total classified fills: {sudden_jump_fills + staircase_fills + passive_fills}")
print(f"Fills caused by SUDDEN JUMPS (Whale appearing out of nowhere inside spread): {sudden_jump_fills}")
print(f"Fills caused by STAIRCASE (bots slowly penny jumping each other to the middle): {staircase_fills}")
print(f"Fills caused by PASSIVE HITS (Stable resting orders hit by normal flow): {passive_fills}")


print("\n--- GENERAL SPREAD DYNAMICS OVER ALL 10,000 TIMESTAMPS ---")
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
