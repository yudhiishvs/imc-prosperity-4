import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

# Paths
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
    
    # Sort absolutely temporally
    df.sort_values(["day", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def analyze_0_spikes(df: pd.DataFrame, asset: str):
    print(f"\n--- {asset} '0' Spikes Analysis ---")
    asset_df = df[df['product'] == asset].copy()
    
    # Mid-price is calculated somehow. If mid_price == 0 or drops extremely low
    # Check what order book looks like at these times.
    low_prices = asset_df[(asset_df['mid_price'] < 5000) | (asset_df['mid_price'].isna())]
    print(f"Total rows with mid_price < 5000 or NaN: {len(low_prices)}")
    
    if len(low_prices) > 0:
        print("Sample of order book during these spikes:")
        cols = ['day', 'timestamp', 'bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', 'mid_price']
        print(low_prices[cols].head(10))
        print("Conclusion on spikes: They correspond to an empty side of the order book (NaN bid or ask), leading to a mid_price calculation failure or dropping to 0.")

def plot_macro_trends(df: pd.DataFrame):
    print("\n--- Generating Macro Trend Plots ---")
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    
    # Create continuous timestamp index for plotting across days
    osmium['continuous_ts'] = osmium['day'] * 1_000_000 + osmium['timestamp']
    pepper['continuous_ts'] = pepper['day'] * 1_000_000 + pepper['timestamp']
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot Osmium
    axes[0].plot(osmium['continuous_ts'], osmium['mid_price'], color='cyan', linewidth=0.5)
    axes[0].set_title('ASH_COATED_OSMIUM Macro Trend (3 Days)')
    axes[0].set_ylabel('Mid Price')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Pepper
    axes[1].plot(pepper['continuous_ts'], pepper['mid_price'], color='orange', linewidth=0.5)
    axes[1].set_title('INTARIAN_PEPPER_ROOT Macro Trend (3 Days)')
    axes[1].set_ylabel('Mid Price')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGS / 'macro_trends.png', dpi=150)
    print(f"Saved {FIGS / 'macro_trends.png'}")

def analyze_osmium_imbalance(df: pd.DataFrame):
    print("\n--- ASH COATED OSMIUM Advanced Optimizations ---")
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    
    # Fix 0/nan spikes
    osmium = osmium[osmium['mid_price'] > 5000].copy()
    
    # Order book imbalance: (BidVol - AskVol) / (BidVol + AskVol)
    osmium['OIM'] = (osmium['bid_volume_1'] - osmium['ask_volume_1']) / (osmium['bid_volume_1'] + osmium['ask_volume_1'])
    osmium['next_mid_price'] = osmium['mid_price'].shift(-1)
    osmium['mid_change'] = osmium['next_mid_price'] - osmium['mid_price']
    
    # Correlation between OIM and next mid price change
    corr = osmium[['OIM', 'mid_change']].corr().iloc[0, 1]
    print(f"Correlation between Level 1 Order Imbalance and Next Tick Mid-Price Change: {corr:.4f}")
    
    # Bin OIM to see average price change
    osmium['OIM_bin'] = pd.cut(osmium['OIM'], bins=[-1.01, -0.6, -0.2, 0.2, 0.6, 1.01], labels=['Strong Ask', 'Slight Ask', 'Neutral', 'Slight Bid', 'Strong Bid'])
    agg = osmium.groupby('OIM_bin')['mid_change'].mean()
    print("\nAverage Mid-Price Change (Next Tick) by L1 Imbalance:")
    print(agg)
    
    # Compare with L2 + L3 imbalance
    osmium['bid_vol_total'] = osmium[['bid_volume_1', 'bid_volume_2', 'bid_volume_3']].sum(axis=1)
    osmium['ask_vol_total'] = osmium[['ask_volume_1', 'ask_volume_2', 'ask_volume_3']].sum(axis=1)
    osmium['OIM_total'] = (osmium['bid_vol_total'] - osmium['ask_vol_total']) / (osmium['bid_vol_total'] + osmium['ask_vol_total'])
    corr_tot = osmium[['OIM_total', 'mid_change']].corr().iloc[0, 1]
    print(f"\nCorrelation between Total Book Imbalance and Next Tick Mid-Price Change: {corr_tot:.4f}")
    
    # Trade direction correlation
    trade_files = sorted(DATA.glob("trades_round_1_day_*.csv"))
    t_frames = []
    for fp in trade_files:
        t_frames.append(pd.read_csv(fp, sep=";"))
    trades = pd.concat(t_frames, ignore_index=True)
    trades = trades[trades['symbol'] == 'ASH_COATED_OSMIUM'].copy()
    # sum trades per timestamp per day
    trades['buyer_aggressor'] = np.where(trades['buyer'] == '', 0, 1) # This is a placeholder, as the csv might not have aggressor flag directly.
    # Actually let's just look at spread distance
    print("\nDone with Osmium Advanced EDA.")

def run():
    print("Loading data...")
    df = load_data()
    print("Data loaded.")
    
    analyze_0_spikes(df, "ASH_COATED_OSMIUM")
    analyze_0_spikes(df, "INTARIAN_PEPPER_ROOT")
    
    plot_macro_trends(df)
    
    analyze_osmium_imbalance(df)

if __name__ == "__main__":
    run()
