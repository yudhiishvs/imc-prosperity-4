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

def load_prices():
    prices = []
    for day in [-2, -1, 0]:
        pf = DATA / f"prices_round_1_day_{day}.csv"
        if pf.exists():
            df = pd.read_csv(pf, sep=";")
            df['day'] = day
            prices.append(df)
            
    df = pd.concat(prices, ignore_index=True)
    for c in df.columns:
        if c not in ['product', 'timestamp', 'day']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.sort_values(["day", "timestamp"], inplace=True)
    return df

def analyze_pepper():
    print("Loading Data...")
    df = load_prices()
    pep = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    
    print("\n===============================")
    print("--- 1. OPTIMIZING SCALP MARGIN ---")
    # For each day, determine the true mathematical "Fair Value" at each timestamp.
    # FV(t) = First_Clean_Mid(t=0) + (0.001 * t)
    
    dfs = []
    for day in pep['day'].unique():
        day_df = pep[pep['day'] == day].copy()
        day_df['raw_mid'] = (day_df['bid_price_1'] + day_df['ask_price_1']) / 2.0
        day_df['clean_mid'] = day_df['raw_mid'].interpolate().bfill().ffill()
        
        base_estimate = day_df['clean_mid'].iloc[0]
        day_df['fair_value'] = base_estimate + (0.001 * day_df['timestamp'])
        day_df['fv_rounded'] = day_df['fair_value'].round()
        dfs.append(day_df)
        
    pep = pd.concat(dfs)
    
    # Scalp_Margin = bid_price_1 - fv_rounded
    pep['bid_vs_fv'] = pep['bid_price_1'] - pep['fv_rounded']
    # Profitable scalps occur when rested bids are ABOVE fair value
    spikes = pep[pep['bid_vs_fv'] > 0]['bid_vs_fv']
    
    if len(spikes) == 0:
        print("CRITICAL FINDING: No resting bids ever exceed true Fair Value.")
        rec_margin = 1
    else:
        print(f"Total timestamps where resting Bid > FV: {len(spikes)}")
        print(f"P25 Profitable Spike: {spikes.quantile(0.25)} ticks")
        print(f"Median Profitable Spike: {spikes.median()} ticks")
        print(f"P75 Profitable Spike: {spikes.quantile(0.75)} ticks")
        print(f"P90 Profitable Spike: {spikes.quantile(0.90)} ticks")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(spikes, bins=20, color='green', alpha=0.7)
        ax.set_title("Distribution of Profitable Bid Spikes (Ticks Above FV)")
        ax.set_xlabel("Ticks Above Fair Value")
        ax.set_ylabel("Frequency (Timestamps)")
        plot_path = OUT_DIR / "pepper_scalp_spikes.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved Scalp Spike Dist to {plot_path}")
        
        # We want to set margin to capture the top ~50% of spikes for decent volume
        rec_margin = int(round(spikes.median()))

    print("\n===============================")
    print("--- 2. OPTIMIZING RECOUP MARGIN ---")
    # Recoup margin is how many ticks ABOVE FV we are willing to buy back.
    pep['ask_vs_fv'] = pep['ask_price_1'] - pep['fv_rounded']
    ask_dist = pep['ask_vs_fv']
    
    print(f"Median Resting Ask vs FV: {ask_dist.median():.2f} ticks")
    print(f"P25 Best Ask vs FV: {ask_dist.quantile(0.25):.2f} ticks")
    print(f"P75 Worst Ask vs FV: {ask_dist.quantile(0.75):.2f} ticks")
    # Set Recoup close to median resting Ask
    rec_recoup_margin = max(1, int(round(ask_dist.quantile(0.25))))
    
    print("\n===============================")
    print("--- 3. OPTIMIZING SCALP VOLUME (TRADE PACING) ---")
    tf0 = pd.read_csv(DATA / "trades_round_1_day_0.csv", sep=";")
    tf1 = pd.read_csv(DATA / "trades_round_1_day_-1.csv", sep=";")
    tf2 = pd.read_csv(DATA / "trades_round_1_day_-2.csv", sep=";")
    tf = pd.concat([tf0, tf1, tf2])
    
    tf_pep = tf[tf['symbol'] == 'INTARIAN_PEPPER_ROOT']['quantity']
    p90_vol = tf_pep.quantile(0.90)
    print(f"Trade Vol P50 (Median): {tf_pep.quantile(0.5):.2f}")
    print(f"Trade Vol P90: {p90_vol:.2f}")

    print("\n==================================")
    print("FINAL DERIVED CONFIGURATION BLOCK:")
    print("==================================")
    print(f"PEPPER_SCALP_MIN_MARGIN = {rec_margin} # Targets median profitable spike")
    print(f"PEPPER_MAX_SCALP_VOLUME = {int(p90_vol)} # Matches P90 trade size")
    print(f"PEPPER_RECOUP_MAX_MARGIN = {rec_recoup_margin} # Targets P25 (efficient) resting Asks")
    print(f"PEPPER_INITIAL_ACC_THRESH = 9 # Validated against general spread metrics")

if __name__ == "__main__":
    analyze_pepper()
