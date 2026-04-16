import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

# Paths
def resolve_repo_root(start: pathlib.Path) -> pathlib.Path:
    cur = start
    for _ in range(6):
        if (cur / "pyproject.toml").exists() or (cur / "data").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start

def resolve_data_dir(root: pathlib.Path) -> pathlib.Path:
    candidates = [
        root / "data" / "round_1",
        root / "data" / "ROUND_1",
        root / "data" / "round1",
        root / "data" / "ROUND1",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

ROOT = resolve_repo_root(pathlib.Path(__file__).resolve().parent)
DATA = resolve_data_dir(ROOT)
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
    
    # Use a clean mid derived from the book to avoid mid_price==0 artifacts.
    valid = osmium["bid_price_1"].notna() & osmium["ask_price_1"].notna()
    osmium["raw_mid"] = np.where(valid, (osmium["bid_price_1"] + osmium["ask_price_1"]) / 2.0, np.nan)
    osmium["clean_mid"] = osmium.groupby("day")["raw_mid"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    osmium = osmium.dropna(subset=["clean_mid"]).copy()
    
    # Order book imbalance: (BidVol - AskVol) / (BidVol + AskVol)
    den1 = (osmium["bid_volume_1"].fillna(0) + osmium["ask_volume_1"].fillna(0)).replace(0, np.nan)
    osmium["OIM"] = ((osmium["bid_volume_1"].fillna(0) - osmium["ask_volume_1"].fillna(0)) / den1).fillna(0.0)
    osmium["next_mid"] = osmium.groupby("day")["clean_mid"].shift(-1)
    osmium["mid_change"] = osmium["next_mid"] - osmium["clean_mid"]
    
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
    denT = (osmium["bid_vol_total"].fillna(0) + osmium["ask_vol_total"].fillna(0)).replace(0, np.nan)
    osmium['OIM_total'] = ((osmium['bid_vol_total'].fillna(0) - osmium['ask_vol_total'].fillna(0)) / denT).fillna(0.0)
    corr_tot = osmium[['OIM_total', 'mid_change']].corr().iloc[0, 1]
    print(f"\nCorrelation between Total Book Imbalance and Next Tick Mid-Price Change: {corr_tot:.4f}")
    
    # Binned view (more interpretable than a raw correlation)
    osmium["OIM_total_bin"] = pd.cut(
        osmium["OIM_total"],
        bins=[-1.01, -0.6, -0.2, 0.2, 0.6, 1.01],
        labels=["Strong Ask", "Slight Ask", "Neutral", "Slight Bid", "Strong Bid"],
    )
    agg2 = osmium.groupby("OIM_total_bin")["mid_change"].mean()
    print("\nAverage Next-Tick Clean-Mid Change by Total-Book Imbalance:")
    print(agg2)
    print("\nDone with Osmium Advanced EDA.")

def analyze_pepper_detailed(df: pd.DataFrame):
    print("\n--- INTARIAN_PEPPER_ROOT Detailed Analysis ---")
    pepper = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    
    # Calculate basic clean mid
    valid = pepper["bid_price_1"].notna() & pepper["ask_price_1"].notna()
    pepper["raw_mid"] = np.where(valid, (pepper["bid_price_1"] + pepper["ask_price_1"]) / 2.0, np.nan)
    pepper["clean_mid"] = pepper.groupby("day")["raw_mid"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    pepper = pepper.dropna(subset=["clean_mid"]).copy()
    
    # 1. Macro Trend (Slope)
    # Fit a linear regression on continuous_ts vs clean_mid
    pepper['continuous_ts'] = pepper['day'] * 1_000_000 + pepper['timestamp']
    z = np.polyfit(pepper['continuous_ts'], pepper['clean_mid'], 1)
    print(f"Overall Linear Slope (price change per unit time): {z[0]:.6e}")
    # Let's get the slope per timestamp unit (usually timestamp increases by 100 per tick)
    slope_per_tick = z[0] * 100
    print(f"Expected price change per tick (dt=100) based on slope: {slope_per_tick:.6f}")
    
    # 2. Spread and Volatility
    pepper['spread'] = pepper['ask_price_1'] - pepper['bid_price_1']
    print(f"Average Spread: {pepper['spread'].mean():.4f}")
    pepper["mid_change"] = pepper.groupby("day")["clean_mid"].shift(-1) - pepper["clean_mid"]
    # STD of tick-to-tick changes
    print(f"Volatility (std of tick-to-tick mid change): {pepper['mid_change'].std():.4f}")
    
    # 3. Autocorrelation (Mean Reversion vs Momentum)
    # Calculate autocorrelations for lags 1, 2, 3
    for lag in [1, 2, 3, 5, 10]:
        ac = pepper['mid_change'].autocorr(lag=lag)
        print(f"Autocorrelation of mid_change (lag={lag}): {ac:.4f}")
        
    # 4. Order Imbalance
    den1 = (pepper["bid_volume_1"].fillna(0) + pepper["ask_volume_1"].fillna(0)).replace(0, np.nan)
    pepper["OIM"] = ((pepper["bid_volume_1"].fillna(0) - pepper["ask_volume_1"].fillna(0)) / den1).fillna(0.0)
    
    corr = pepper[['OIM', 'mid_change']].corr().iloc[0, 1]
    print(f"Correlation between L1 Order Imbalance and Next Tick Mid-Price Change: {corr:.4f}")
    
    pepper['OIM_bin'] = pd.cut(pepper['OIM'], bins=[-1.01, -0.6, -0.2, 0.2, 0.6, 1.01], labels=['Strong Ask', 'Slight Ask', 'Neutral', 'Slight Bid', 'Strong Bid'])
    agg = pepper.groupby('OIM_bin')['mid_change'].mean()
    print("\nAverage Mid-Price Change (Next Tick) by L1 Imbalance:")
    print(agg)

    # 5. Opportunity for Scalping / Market Making
    # How often does price revert when moving N ticks from trend?
    pepper['trend_pred'] = z[0] * pepper['continuous_ts'] + z[1]
    pepper['detrended_mid'] = pepper['clean_mid'] - pepper['trend_pred']
    print(f"Std dev of detrended mid: {pepper['detrended_mid'].std():.4f}")


def run():
    print("Loading data...")
    df = load_data()
    print("Data loaded.")
    
    analyze_0_spikes(df, "ASH_COATED_OSMIUM")
    analyze_0_spikes(df, "INTARIAN_PEPPER_ROOT")
    
    plot_macro_trends(df)
    
    analyze_osmium_imbalance(df)
    
    analyze_pepper_detailed(df)

if __name__ == "__main__":
    run()
