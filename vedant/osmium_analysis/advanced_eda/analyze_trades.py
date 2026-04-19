import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "round_1"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "advanced_eda" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    price_frames, trade_frames = [], []
    for day in [-2, -1, 0]:
        pf = DATA / f"prices_round_1_day_{day}.csv"
        tf = DATA / f"trades_round_1_day_{day}.csv"
        if pf.exists():
            pdf = pd.read_csv(pf, sep=";")
            pdf['day'] = day
            price_frames.append(pdf)
        if tf.exists():
            tdf = pd.read_csv(tf, sep=";")
            tdf['day'] = day
            trade_frames.append(tdf)
            
    pdf = pd.concat(price_frames, ignore_index=True)
    for c in pdf.columns:
        if c not in ['product', 'timestamp', 'day']:
            pdf[c] = pd.to_numeric(pdf[c], errors='coerce')
            
    tdf = pd.concat(trade_frames, ignore_index=True)
    for c in tdf.columns:
        if c not in ['symbol', 'timestamp', 'day', 'buyer', 'seller', 'currency']:
            tdf[c] = pd.to_numeric(tdf[c], errors='coerce')
    tdf.rename(columns={'symbol': 'product'}, inplace=True)
            
    return pdf, tdf

def analyze_trades():
    print("Loading Data...")
    prices, trades = load_data()
    
    for asset in ['ASH_COATED_OSMIUM', 'INTARIAN_PEPPER_ROOT']:
        print(f"\n==============================")
        print(f"--- Trade Analysis: {asset} ---")
        p_asset = prices[prices['product'] == asset].copy()
        t_asset = trades[trades['product'] == asset].copy()
        
        if len(t_asset) == 0:
            print("No trades found.")
            continue
            
        # Clean Mid-Price Interpolation
        p_asset['raw_mid'] = (p_asset['bid_price_1'] + p_asset['ask_price_1']) / 2.0
        p_asset['raw_mid'] = p_asset['raw_mid'].replace(0, np.nan)
        p_asset['clean_mid'] = p_asset['raw_mid'].interpolate(method='linear').bfill().ffill()
        
        # Merge trades with price book state AT THE SAME TIMESTAMP
        merged = pd.merge(t_asset, p_asset[['day', 'timestamp', 'clean_mid', 'bid_price_1', 'ask_price_1', 'bid_price_2', 'ask_price_2']], on=['day', 'timestamp'], how='left')
        
        print(f"Total recorded trades: {len(merged)}")
        
        # Analyze trade direction and aggression
        merged['trade_vs_mid'] = merged['price'] - merged['clean_mid']
        
        # Trades executed AT or ABOVE Ask (Buy Market Orders)
        buy_aggressors = merged[merged['price'] >= merged['ask_price_1']]
        # Trades executed AT or BELOW Bid (Sell Market Orders)
        sell_aggressors = merged[merged['price'] <= merged['bid_price_1']]
        
        print(f"Trades hitting the ASK (Buy Pressure): {len(buy_aggressors)} ({(len(buy_aggressors)/len(merged))*100:.1f}%)")
        print(f"Trades hitting the BID (Sell Pressure): {len(sell_aggressors)} ({(len(sell_aggressors)/len(merged))*100:.1f}%)")
        inside_spread = merged[(merged['price'] < merged['ask_price_1']) & (merged['price'] > merged['bid_price_1'])]
        print(f"Trades matching inside L1 spread: {len(inside_spread)} ({(len(inside_spread)/len(merged))*100:.1f}%)")
        
        # Distance from mid
        print("\nStatistical absolute distance from mid-price (spread capture margin):")
        abs_dist = merged['trade_vs_mid'].abs()
        print(abs_dist.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]))
        
        # Volume profile
        print(f"Average trade volume: {merged['quantity'].mean():.2f}")
        print(f"90th percentile trade volume: {merged['quantity'].quantile(0.9):.2f}")
        
        # L2 sweeps
        # Need to handle NaNs for L2 prices if book was thin
        hits_l2_ask = merged[(merged['ask_price_2'].notna()) & (merged['price'] >= merged['ask_price_2'])]
        hits_l2_bid = merged[(merged['bid_price_2'].notna()) & (merged['price'] <= merged['bid_price_2'])]
        print(f"\nTrades deeply sweeping to L2 ASK: {len(hits_l2_ask)}")
        print(f"Trades deeply sweeping to L2 BID: {len(hits_l2_bid)}")
        
        # Plot Histograms
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(merged['trade_vs_mid'], bins=50, color='purple', alpha=0.7)
        ax.axvline(0, color='black', linestyle='--')
        ax.set_title(f"Distance of Trade Executions vs Mid Price\n{asset}")
        ax.set_xlabel("Ticks away from Mid (+ means hit Ask, - means hit Bid)")
        ax.set_ylabel("Number of Trades")
        
        plot_name = OUT_DIR / f"{asset.lower()}_trade_dist.png"
        plt.tight_layout()
        plt.savefig(plot_name, dpi=150)
        print(f"Saved Trade Distribution Plot to {plot_name}")

if __name__ == "__main__":
    analyze_trades()
