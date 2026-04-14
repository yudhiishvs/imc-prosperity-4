import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "round_1"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "advanced_eda"
FIGS = OUT_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def load_data():
    price_files = sorted(DATA.glob("prices_round_1_day_*.csv"))
    frames = []
    for fp in price_files:
        df = pd.read_csv(fp, sep=";")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if c not in ['product', 'timestamp', 'day']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.sort_values(["day", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def analyze_pepper():
    df = load_data()
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    
    # Needs a continuous time for deterministic fair value calculation.
    # The prices reset per day, so let's calculate the initial mid_price per day and offset.
    # Although Yudhiish used a fixed slope across the whole day.
    
    results = []
    for day, group in pepper.groupby('day'):
        group = group.copy()
        # Filter NA
        valid = group.dropna(subset=['mid_price'])
        if len(valid) == 0: continue
        initial_mid = valid.iloc[0]['mid_price']
        
        group['deterministic_fv'] = initial_mid + 0.001 * group['timestamp']
        
        for idx, row in group.iterrows():
            if pd.isna(row['deterministic_fv']): continue
            fv = row['deterministic_fv']
            
            # Distance of max bid from FV (Opportunity to Dump)
            if not pd.isna(row['bid_price_1']):
                dump_distance = row['bid_price_1'] - fv
                results.append({'type': 'bid', 'distance': dump_distance, 'vol': row['bid_volume_1']})
                
            # Distance of min ask from FV (Opportunity to Accumulate)
            if not pd.isna(row['ask_price_1']):
                acc_distance = row['ask_price_1'] - fv
                results.append({'type': 'ask', 'distance': acc_distance, 'vol': row['ask_volume_1']})
                
    res_df = pd.DataFrame(results)
    
    # Calculate statistics
    bids = res_df[res_df['type'] == 'bid']['distance']
    asks = res_df[res_df['type'] == 'ask']['distance']
    
    print("\nBid Distance from Deterministic FV (We dump into these Bids)")
    print("Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.9, 99.99, 100]:
        print(f"p{p}: {np.percentile(bids, p):.2f} ticks")

    print("\nAsk Distance from Deterministic FV (We accumulate from these Asks)")
    print("Percentiles:")
    for p in [50, 25, 10, 5, 1, 0.1, 0.01, 0]:
        print(f"p{p}: {np.percentile(asks, p):.2f} ticks")
        
    # Plot Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(bids, bins=50, range=(-10, 10), color='green', alpha=0.7)
    axes[0].set_title('Bids relative to FV (Positive means Bid > FV)')
    axes[0].axvline(0, color='r', linestyle='--')
    
    axes[1].hist(asks, bins=50, range=(-10, 10), color='red', alpha=0.7)
    axes[1].set_title('Asks relative to FV (Negative means Ask < FV)')
    axes[1].axvline(0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGS / 'pepper_spread_dist.png', dpi=150)
    print(f"\nSaved histogram to {FIGS / 'pepper_spread_dist.png'}")

if __name__ == "__main__":
    analyze_pepper()
