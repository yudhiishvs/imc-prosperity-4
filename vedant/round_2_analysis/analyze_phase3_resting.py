import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/sweep_and_quote/298540/298540.log'
    print(f"Loading {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    csv_data = data['activitiesLog']
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    
    # Calculate Final PnL from activities
    final_pnl = osmium['profit_and_loss'].iloc[-1] if len(osmium) > 0 else 0
    print(f"=== PnL Analysis ===")
    print(f"Final PnL achieved on run 298540: {final_pnl}")

    trade_data = data.get('tradeHistory', '')
    if isinstance(trade_data, str):
        trades = pd.read_csv(io.StringIO(trade_data), sep=';')
    else:
        trades = pd.DataFrame(trade_data)
        
    sym_col = 'symbol' if 'symbol' in trades.columns else 'product'
    os_trades = trades[trades[sym_col] == 'ASH_COATED_OSMIUM'].copy()

    print("\n=== Phase 3: Resting Book Analysis (Mining our Sweeps) ===")
    # Our bot actively swept all resting orders (bids + asks).
    # Since FV is 10004, the exact prices we swept are the bots' passive resting bounds!
    
    # Let's filter out trades that are our passive "Max EV" quotes getting filled (11, 12, 13 offset)
    # 10004 - 11 to 13 = 9991, 9992, 9993
    # 10004 + 11 to 13 = 10015, 10016, 10017
    
    probe_bid_prices = [9991, 9992, 9993]
    probe_ask_prices = [10015, 10016, 10017]
    
    # Assume any trade outside these 6 specific prices (and maybe edge noise) is a sweep into the Natural Book.
    # The natural book is roughly 9998-10002 for bids, 10006-10010 for asks.
    
    natural_sweeps_bids = os_trades[(os_trades['price'] >= 9996) & (os_trades['price'] <= 10003)]
    natural_sweeps_asks = os_trades[(os_trades['price'] >= 10005) & (os_trades['price'] <= 10012)]
    
    print("\n[Natural Bot Resting Bids] (Where they quote to buy passively):")
    bid_volumes = natural_sweeps_bids.groupby('price')['quantity'].sum().sort_index(ascending=False)
    for price, vol in bid_volumes.items():
        print(f"Price: {price} (Offset from 10004: {10004 - price}) -> Total Volume Swept: {vol}")

    print("\n[Natural Bot Resting Asks] (Where they quote to sell passively):")
    ask_volumes = natural_sweeps_asks.groupby('price')['quantity'].sum().sort_index()
    for price, vol in ask_volumes.items():
        print(f"Price: {price} (Offset from 10004: {price - 10004}) -> Total Volume Swept: {vol}")

    # Analyze our Probe Performance
    print("\n=== Evaluating EV Drop at 'Optimal' 11-13 ===")
    probe_bids_data = os_trades[os_trades['price'].isin(probe_bid_prices)]
    probe_asks_data = os_trades[os_trades['price'].isin(probe_ask_prices)]
    
    bid_fills = probe_bids_data.groupby('price')['quantity'].sum()
    ask_fills = probe_asks_data.groupby('price')['quantity'].sum()
    
    print("\nProbe Bid Fills (Target: 9991, 9992, 9993):")
    print(bid_fills)
    print("\nProbe Ask Fills (Target: 10015, 10016, 10017):")
    print(ask_fills)
    
if __name__ == "__main__":
    main()
