import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/sweep_and_quote/297254.log'
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
    print(f"Mean Mid Price (valid): {mean_mid:.4f}")
    print(f"Std Dev Mid Price (valid): {std_mid:.4f}")
    print(f"Min Mid Price: {min_mid:.4f}")
    print(f"Max Mid Price: {max_mid:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(osmium['timestamp'], osmium['mid_price'], label='Mid Price', color='blue', alpha=0.7)
    plt.axhline(mean_mid, color='red', linestyle='--', label=f'Mean ({mean_mid:.2f})')
    # tightly bound y-axis so it isn't massive scaling
    plt.ylim(mean_mid - 30, mean_mid + 30)
    plt.title('ASH_COATED_OSMIUM: Sweep Mid Price Dynamics')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('sweep_mean_analysis.png')
    plt.close()
    print("Plot saved to sweep_mean_analysis.png")
    
    # --- Part 2: Trade History & Extreme Fills ---
    trade_data = data.get('tradeHistory', '')
    if len(trade_data) == 0:
        print("No tradeHistory data found!")
        return
        
    # tradeHistory is usually structured as a list of dicts in pure json, or string csv. Let's handle both.
    if isinstance(trade_data, str):
        # We assume it's a CSV like activitiesLog
        try:
            trades = pd.read_csv(io.StringIO(trade_data), sep=';')
        except:
            print("Failed to parse string trade history.")
            return
    else:
        # If it's a JSON array
        trades = pd.DataFrame(trade_data)
        
    # Standardize column name (symbol or product)
    sym_col = 'symbol' if 'symbol' in trades.columns else 'product' if 'product' in trades.columns else None
    if sym_col is None:
        print("Could not find product/symbol column in tradeHistory.")
        print(f"Columns: {trades.columns.tolist()}")
        # Let's just print a bit of it for debug
        print("Sample:", trades.head(1).to_dict('records'))
        return

    os_trades = trades[trades[sym_col] == 'ASH_COATED_OSMIUM'].copy()
    if len(os_trades) == 0:
        print("No osmium trades found in history.")
        return

    # To find which extreme bounds filled us, we just look for all trades executed by anyone
    # at extremely wide prices since the bots only fill at those levels if we quoted there.
    # Our bot sweeps the natural book, meaning prices around 10000 will be our active sweeps.
    # It also quotes blindly out to bounded limits (e.g. 21 to 9999 offset).
    # IF we see trades at 10500, we know the bots hit us!
    
    extreme_buys = os_trades[os_trades['price'] < 9992]
    extreme_sells = os_trades[os_trades['price'] > 10016]
    
    print("\n=== FILL DYNAMICS ===")
    print(f"Total Osmium Trades: {len(os_trades)}")
    print(f"Absolute Minimum executed trade price: {os_trades['price'].min()}")
    print(f"Absolute Maximum executed trade price: {os_trades['price'].max()}")
    
    # Group by price to see density of extreme fills
    if len(extreme_buys) > 0:
        print("\nExtreme Bid Fills (Bots sold to us aggressively low):")
        print(extreme_buys['price'].value_counts().sort_index().head(20))
        max_fill_bid = extreme_buys['price'].max()  # The highest the bot was willing to reach down to hit
        print(f"Maximum bot 'throw' down to hit our bid: {max_fill_bid} (Offset: {fair - max_fill_bid if 'fair' in locals() else 'N/A'})")
    else:
        print("\nNo Extreme Bid Fills extracted. The offset range [21, 9999] was likely too wide, or limits not hit.")

    if len(extreme_sells) > 0:
        print("\nExtreme Ask Fills (Bots bought from us aggressively high):")
        print(extreme_sells['price'].value_counts().sort_index().head(20))
        min_fill_ask = extreme_sells['price'].min()  # The lowest the bot was willing to reach up to hit
        print(f"Minimum bot 'throw' up to hit our ask: {min_fill_ask}")
    else:
        print("\nNo Extreme Ask Fills extracted.")

if __name__ == "__main__":
    main()
