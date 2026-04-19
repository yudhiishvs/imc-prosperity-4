import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/normal_quotes/297739.log'
    print(f"Loading {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    # --- Part 1: Mid Price & Mean Revalidation ---
    csv_data = data['activitiesLog']
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    osmium.sort_values(by='timestamp', inplace=True)
    
    valid_mids = osmium[osmium['mid_price'] > 5000]['mid_price']
    mean_mid = valid_mids.mean()
    std_mid = valid_mids.std()
    min_mid = valid_mids.min()
    max_mid = valid_mids.max()
    print("=== ACTIVITIES LOG STATS (Revalidation) ===")
    print(f"Mean Mid Price (valid ticks): {mean_mid:.4f}")
    print(f"Std Dev Mid Price (valid ticks): {std_mid:.4f}")
    print(f"Min Mid Price: {min_mid:.4f}")
    print(f"Max Mid Price: {max_mid:.4f}")
    
    # --- Part 2: Trade History & Extreme Fills ---
    trade_data = data.get('tradeHistory', '')
    if len(trade_data) == 0:
        print("No tradeHistory data found!")
        return
        
    if isinstance(trade_data, str):
        try:
            trades = pd.read_csv(io.StringIO(trade_data), sep=';')
        except:
            print("Failed to parse string trade history.")
            return
    else:
        trades = pd.DataFrame(trade_data)
        
    sym_col = 'symbol' if 'symbol' in trades.columns else 'product' if 'product' in trades.columns else None
    if sym_col is None:
        print("Could not find product/symbol column in tradeHistory.")
        return

    os_trades = trades[trades[sym_col] == 'ASH_COATED_OSMIUM'].copy()
    if len(os_trades) == 0:
        print("No osmium trades found in history.")
        return

    print("\n=== FILL DYNAMICS ===")
    print(f"Total Osmium Trades: {len(os_trades)}")
    print(f"Absolute Minimum executed trade price: {os_trades['price'].min()}")
    print(f"Absolute Maximum executed trade price: {os_trades['price'].max()}")
    
    # To determine if we actually posted passive quotes or just hit limits, 
    # we just need to see if we got any fills at our internal random generation range [1, 20] from 10004.
    # We posted bids at 9984 to 10003. And asks at 10005 to 10024.
    
    # Filter out the core "sweep" zones where the natural book sits (usually ~ 9998-10001 bid, 10007-10010 ask)
    # Wait, any execution below 9997 is a great sign the bots stepped down to hit our probe bid!
    # Any execution above 10011 is a great sign the bots stepped up to hit our probe ask!
    
    down_probes = os_trades[os_trades['price'] <= 9995]
    up_probes = os_trades[os_trades['price'] >= 10013]
    
    print(f"\n--- Fills extending downwards (Price <= 9995) ---")
    if len(down_probes) > 0:
        print(down_probes['price'].value_counts().sort_index(ascending=False))
    else:
        print("None.")
        
    print(f"\n--- Fills extending upwards (Price >= 10013) ---")
    if len(up_probes) > 0:
        print(up_probes['price'].value_counts().sort_index())
    else:
        print("None.")
        
    print("\n--- Core Fills Profile (9996 to 10012) ---")
    core_probes = os_trades[(os_trades['price'] > 9995) & (os_trades['price'] < 10013)]
    print(core_probes['price'].value_counts().sort_index())

if __name__ == "__main__":
    main()
