import json
import io
import pandas as pd

def main():
    log_file = '/Users/vedant/Quant/Prosperity4/imc-prosperity-4/current_strat_performance/301407.log'
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

    print(f"Total Osmium Trades: {len(os_trades)}")
    
    if len(os_trades) > 0:
        os_trades['is_buy'] = os_trades['buyer'] == 'V' # Replace 'V' placeholder, actually let's just look at quantity
        buy_trades = os_trades[os_trades['buyer'] == 'SUBMISSION'] 
        sell_trades = os_trades[os_trades['seller'] == 'SUBMISSION']
        
        print("\n--- Buy Trades ---")
        if len(buy_trades) > 0:
            print(buy_trades.groupby('price')['quantity'].sum())
        
        print("\n--- Sell Trades ---")
        if len(sell_trades) > 0:
            print(sell_trades.groupby('price')['quantity'].sum())

    print("\n--- First 20 Ticks ---")
    print(osmium[['timestamp', 'mid_price', 'profit_and_loss']].head(20))
    
if __name__ == "__main__":
    main()
