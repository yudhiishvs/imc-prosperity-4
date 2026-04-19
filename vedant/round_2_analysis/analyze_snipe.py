import json
import io
import pandas as pd

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/snipe/298967.log'
    print(f"Loading {log_file}...")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
        
    csv_data = data['activitiesLog']
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    osmium.sort_values(by='timestamp', inplace=True)
    osmium.reset_index(drop=True, inplace=True)

    trade_data = data.get('tradeHistory', '')
    if isinstance(trade_data, str):
        trades = pd.read_csv(io.StringIO(trade_data), sep=';')
    else:
        trades = pd.DataFrame(trade_data)
        
    sym_col = 'symbol' if 'symbol' in trades.columns else 'product'
    os_trades = trades[trades[sym_col] == 'ASH_COATED_OSMIUM'].copy()
    
    print("=== Bot Inventory Refill Analysis ===")
    # We want to see if we snipe the volume, does the next tick replenish the full expected volume?
    # Expected Volume is around 13-15 for the inside levels.
    
    # Let's see the sequence of volumes when prices stay the same.
    prices_bids = osmium['bid_price_1']
    vols_bids = osmium['bid_volume_1']
    
    refills_count = 0
    total_snipes = 0
    
    snipes = os_trades[abs(os_trades['quantity']) > 1]
    
    for _, snipe in snipes.iterrows():
        ts = snipe['timestamp']
        price = snipe['price']
        qty = abs(snipe['quantity'])
        
        # Look at the tick after the snipe
        next_tick = osmium[osmium['timestamp'] > ts].head(1)
        if len(next_tick) == 0:
            continue
            
        next_ts = next_tick['timestamp'].values[0]
        bid1 = next_tick['bid_price_1'].values[0]
        bidv1 = next_tick['bid_volume_1'].values[0]
        ask1 = next_tick['ask_price_1'].values[0]
        askv1 = next_tick['ask_volume_1'].values[0]
        
        total_snipes += 1
        
        # If we sniped bid, check if bid re-appeared at same price with full volume
        if snipe['quantity'] < 0: # we sold to their bid
            if bid1 == price:
               if bidv1 > qty: # They refilled more than what we took
                   refills_count += 1
        else: # we bought from their ask
            if ask1 == price:
               if askv1 > qty:
                   refills_count += 1
                   
    print(f"Total Snipes analyzed: {total_snipes}")
    print(f"Number of times the bot refilled its volume at the same price immediately: {refills_count}")
    print(f"Refill probability: {refills_count / total_snipes:.2%}")

    print("\n=== Penny Jump Analysis ===")
    jumping_trades = os_trades[abs(os_trades['quantity']) == 1]
    print(f"We posted {len(osmium)} penny jump orders.")
    print(f"We actually got filled on our penny jump orders {len(jumping_trades)} times.")
    print("If we get filled on our passive orders, it means the bot DID NOT aggressively penny-jump us to take priority.")
    print("If we rarely get filled, it means the bot stepped in front of us.")
    
if __name__ == "__main__":
    main()
