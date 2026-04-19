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

    jumping_count = 0
    total_samples = 0
    
    for i in range(1, len(osmium)):
        prev_row = osmium.iloc[i-1]
        curr_row = osmium.iloc[i]
        
        # We know we posted a 1-lot order 1 tick better than the best bid/ask
        # Assume prev_row best bid was B. We posted at B+1.
        # If curr_row best bid is B+2, then the bot stepped in front of our B+1 order!
        
        prev_bid = prev_row['bid_price_1']
        curr_bid = curr_row['bid_price_1']
        
        if pd.notna(prev_bid) and pd.notna(curr_bid):
            total_samples += 1
            if curr_bid > prev_bid + 1: # They jumped over our jump
                jumping_count += 1
                
        prev_ask = prev_row['ask_price_1']
        curr_ask = curr_row['ask_price_1']
        if pd.notna(prev_ask) and pd.notna(curr_ask):
            total_samples += 1
            if curr_ask < prev_ask - 1: # They undercut our undercut
                jumping_count += 1
                
    print(f"Total timestamps analyzed: {total_samples}")
    print(f"Count of explicit bot penny jumps over our quotes: {jumping_count}")

if __name__ == "__main__":
    main()
