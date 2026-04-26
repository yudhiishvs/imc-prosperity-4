import json
import numpy as np

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
mids = [float(r['mid_price']) for r in osmium_log if r['mid_price']]

print("--- MID-PRICE MOMENTUM ANALYSIS (308866) ---")
if len(mids) > 100:
    start_avg = np.mean(mids[:100])
    end_avg = np.mean(mids[-100:])
    total_avg = np.mean(mids)
    std_dev = np.std(mids)
    
    print(f"Start (first 100 ts) Avg: {start_avg:.2f}")
    print(f"End (last 100 ts) Avg  : {end_avg:.2f}")
    print(f"Total Session Avg      : {total_avg:.2f}")
    print(f"Session StdDev         : {std_dev:.2f}")
    print(f"Net Drift              : {end_avg - start_avg:+.2f} ticks")
    
    # Check for trend
    z = np.polyfit(range(len(mids)), mids, 1)
    print(f"Linear Slope (ticks/tick): {z[0]:.6f}")
    print(f"Predicted Drift over 100k ts: {z[0] * 100000:+.2f} ticks")
else:
    print("Not enough data points.")
