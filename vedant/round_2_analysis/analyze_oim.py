import pandas as pd
import numpy as np

def analyze_oim(day_csv):
    df = pd.read_csv(day_csv, sep=';')
    
    # Filter Osmium
    df = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Calculate Mid Price
    df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2.0
    
    # L1 OIM
    b_vol1 = df['bid_volume_1'].fillna(0)
    a_vol1 = df['ask_volume_1'].fillna(0)
    df['l1_oim'] = (b_vol1 - a_vol1) / (b_vol1 + a_vol1).replace(0, np.nan)
    
    # L2 OIM
    b_vol2 = df['bid_volume_2'].fillna(0)
    a_vol2 = df['ask_volume_2'].fillna(0)
    b_l2 = b_vol1 + b_vol2
    a_l2 = a_vol1 + a_vol2
    df['l2_oim'] = (b_l2 - a_l2) / (b_l2 + a_l2).replace(0, np.nan)
    
    # L3 OIM
    b_vol3 = df['bid_volume_3'].fillna(0)
    a_vol3 = df['ask_volume_3'].fillna(0)
    b_l3 = b_l2 + b_vol3
    a_l3 = a_l2 + a_vol3
    df['l3_oim'] = (b_l3 - a_l3) / (b_l3 + a_l3).replace(0, np.nan)
    
    # Look ahead returns
    for h in [1, 2, 5]:
        df[f'mid_fwd_{h}'] = df['mid_price'].shift(-h) - df['mid_price']
        
    df.dropna(subset=['mid_fwd_5'], inplace=True)
    
    print(f"Results for {day_csv}:")
    
    for level in ['l1', 'l2', 'l3']:
        oim_col = f'{level}_oim'
        print(f"\n--- {level.upper()} OIM ---")
        
        # Correlation
        for h in [1, 2, 5]:
            corr = df[oim_col].corr(df[f'mid_fwd_{h}'])
            print(f"Pearson Corr with {h} tick fwd ret: {corr:.4f}")
            
        # Accuracy at threshold = 0.5 (Strong Imbalance)
        for h in [1, 2, 5]:
            # Strong Bid OIM -> Price goes UP
            long_sig = df[oim_col] > 0.5
            # Strong Ask OIM -> Price goes DOWN
            short_sig = df[oim_col] < -0.5
            
            correct_longs = (df.loc[long_sig, f'mid_fwd_{h}'] > 0).sum()
            total_longs = long_sig.sum()
            
            correct_shorts = (df.loc[short_sig, f'mid_fwd_{h}'] < 0).sum()
            total_shorts = short_sig.sum()
            
            total_correct = correct_longs + correct_shorts
            total_signals = total_longs + total_shorts
            acc = total_correct / total_signals if total_signals > 0 else 0
            
            print(f"Hit Rate @ threshold 0.5 (h={h}): {acc*100:.1f}% (N={total_signals})")

if __name__ == "__main__":
    analyze_oim('/Users/vedant/Quant/Prosperity4/imc-prosperity-4/data/ROUND_2/prices_round_2_day_0.csv')
    analyze_oim('/Users/vedant/Quant/Prosperity4/imc-prosperity-4/data/ROUND_2/prices_round_2_day_1.csv')
