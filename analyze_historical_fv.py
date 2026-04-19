import pandas as pd
import glob
import os

data_dir = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4/data/ROUND_2"
files = glob.glob(os.path.join(data_dir, "prices_round_2_day_*.csv"))

results = []

for f in sorted(files):
    day_str = os.path.basename(f).split('_')[-1].replace('.csv', '')
    df = pd.read_csv(f, sep=';')
    
    ash_df = df[(df['product'] == 'ASH_COATED_OSMIUM') & (df['mid_price'] > 0)]
    
    if not ash_df.empty:
        avg_mid = ash_df['mid_price'].mean()
        median_mid = ash_df['mid_price'].median()
        mode_mid = ash_df['mid_price'].mode()[0]
        
        results.append({
            "day": day_str,
            "avg_mid": avg_mid,
            "median_mid": median_mid,
            "mode_mid": mode_mid
        })

print("--- REFINED HISTORICAL MID-PRICE ANALYSIS ---")
for res in results:
    print(f"Day {res['day']}: Avg: {res['avg_mid']:.2f} | Median: {res['median_mid']:.2f} | Mode: {res['mode_mid']:.2f}")
