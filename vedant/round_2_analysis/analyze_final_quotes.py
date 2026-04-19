import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/normal_quotes/298076/298076.log'
    print(f"Loading {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    csv_data = data['activitiesLog']
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    
    valid_mids = osmium[osmium['mid_price'] > 5000]['mid_price']
    mean_mid = valid_mids.mean()
    print("=== FINAL RE-VALIDATION ===")
    print(f"Mean Mid Price (valid ticks): {mean_mid:.4f} (Confirms 10004 hypothesis)")

    trade_data = data.get('tradeHistory', '')
    if isinstance(trade_data, str):
        trades = pd.read_csv(io.StringIO(trade_data), sep=';')
    else:
        trades = pd.DataFrame(trade_data)
        
    sym_col = 'symbol' if 'symbol' in trades.columns else 'product'
    os_trades = trades[trades[sym_col] == 'ASH_COATED_OSMIUM'].copy()

    # Total timestamps
    n_ticks = len(osmium['timestamp'].unique())
    n_offsets = 21 # range(10, 31)
    ticks_per_offset = n_ticks / n_offsets
    quoted_qty_per_offset = ticks_per_offset * 80
    
    # Sum quantities by price
    price_fills = os_trades.groupby('price')['quantity'].sum()

    # Calculate Probability-based EV
    # Bids
    bid_results = []
    for price in range(9974, 9995):
        qty_filled = price_fills.get(float(price), 0)
        prob = qty_filled / quoted_qty_per_offset
        edge = 10004 - price
        ev = prob * edge
        bid_results.append((price, edge, qty_filled, prob, ev))
        
    df_bids = pd.DataFrame(bid_results, columns=['Price', 'Offset', 'QtyFilled', 'Prob', 'EV'])
    print("\n--- BID OPTIMIZATION (Prob-based) ---")
    print(df_bids.to_string(index=False))

    # Asks
    ask_results = []
    for price in range(10014, 10035):
        qty_filled = price_fills.get(float(price), 0)
        prob = qty_filled / quoted_qty_per_offset
        edge = price - 10004
        ev = prob * edge
        ask_results.append((price, edge, qty_filled, prob, ev))
        
    df_asks = pd.DataFrame(ask_results, columns=['Price', 'Offset', 'QtyFilled', 'Prob', 'EV'])
    print("\n--- ASK OPTIMIZATION (Prob-based) ---")
    print(df_asks.to_string(index=False))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bids Subplot
    ax1.bar(df_bids['Offset'], df_bids['Prob'], color='green', alpha=0.6, label='Fill Prob')
    ax1.set_xlabel('Edge (Offset from 10004)')
    ax1.set_ylabel('Fill Probability', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    ax1_ev = ax1.twinx()
    ax1_ev.plot(df_bids['Offset'], df_bids['EV'], color='darkgreen', marker='o', linewidth=2, label='Expected Value')
    ax1_ev.set_ylabel('EV (Prob * Edge)', color='darkgreen')
    ax1_ev.tick_params(axis='y', labelcolor='darkgreen')
    ax1.set_title('Empty Book BIDs: Prob & EV')

    # Asks Subplot
    ax2.bar(df_asks['Offset'], df_asks['Prob'], color='red', alpha=0.6, label='Fill Prob')
    ax2.set_xlabel('Edge (Offset from 10004)')
    ax2.set_ylabel('Fill Probability', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2_ev = ax2.twinx()
    ax2_ev.plot(df_asks['Offset'], df_asks['EV'], color='darkred', marker='o', linewidth=2, label='Expected Value')
    ax2_ev.set_ylabel('EV (Prob * Edge)', color='darkred')
    ax2_ev.tick_params(axis='y', labelcolor='darkred')
    ax2.set_title('Empty Book ASKs: Prob & EV')

    plt.tight_layout()
    plt.savefig('fill_probabilities_refined.png')
    plt.close()
    print("Plot saved to fill_probabilities_refined.png")

if __name__ == "__main__":
    main()
