import pandas as pd
import numpy as np
import pathlib
from scipy.stats import pearsonr

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

def load_data(day: int):
    # Prices
    price_file = DATA / f"prices_round_1_day_{day}.csv"
    if not price_file.exists(): return None, None
    prices = pd.read_csv(price_file, sep=";")
    prices = prices[prices['product'] == 'ASH_COATED_OSMIUM'].copy()
    for c in prices.columns:
        if c not in ['product', 'timestamp', 'day']:
            prices[c] = pd.to_numeric(prices[c], errors='coerce')
    prices.sort_values("timestamp", inplace=True)
    prices.reset_index(drop=True, inplace=True)
    prices['l1_mid'] = (prices['bid_price_1'] + prices['ask_price_1']) / 2.0
    prices['l1_mid'] = prices['l1_mid'].interpolate(method='linear')
    
    # Trades
    trade_file = DATA / f"trades_round_1_day_{day}.csv"
    if not trade_file.exists(): return prices, pd.DataFrame()
    trades = pd.read_csv(trade_file, sep=";")
    trades = trades[trades['symbol'] == 'ASH_COATED_OSMIUM'].copy()
    trades.sort_values("timestamp", inplace=True)
    
    return prices, trades

def analyze_informed_flow():
    days = [-2, -1, 0]
    all_prices = []
    
    for day in days:
        prices, trades = load_data(day)
        if prices is None or trades.empty: continue
        
        # Merge trades with prices to find prevailing mid and determine aggressor side
        trades = pd.merge_asof(trades, prices[['timestamp', 'l1_mid']], on='timestamp', direction='backward')
        
        # Determine aggressor volume (+ for buys, - for sells)
        trades['aggressor_vol'] = np.where(trades['price'] > trades['l1_mid'], trades['quantity'], 
                                   np.where(trades['price'] < trades['l1_mid'], -trades['quantity'], 0))
        
        # Group by timestamp (to sum up aggressive volume occurring at the exact same tick)
        tick_trades = trades.groupby('timestamp')['aggressor_vol'].sum().reset_index()
        
        # Merge back onto prices
        prices = pd.merge(prices, tick_trades, on='timestamp', how='left')
        prices['aggressor_vol'] = prices['aggressor_vol'].fillna(0)
        
        # Calculate Rolling Net Aggressor Volume over last N ticks (e.g., 20 ticks = 2 seconds)
        window = 20
        prices['rolling_net_vol'] = prices['aggressor_vol'].rolling(window=window, min_periods=1).sum()
        
        # Calculate Forward Returns over next M ticks
        horizons = [10, 30, 60]  # 1s, 3s, 6s forward
        for h in horizons:
            prices[f'fwd_ret_{h}'] = prices['l1_mid'].shift(-h) - prices['l1_mid']
            
        all_prices.append(prices)
        
    df = pd.concat(all_prices, ignore_index=True)
    
    print("=== Informed Trade Flow Analysis (ASH_COATED_OSMIUM) ===")
    print("Does Net Aggressive Trade Volume predict future price movements?\n")
    
    horizons = [10, 30, 60]
    for h in horizons:
        valid = df.dropna(subset=['rolling_net_vol', f'fwd_ret_{h}'])
        corr, _ = pearsonr(valid['rolling_net_vol'], valid[f'fwd_ret_{h}'])
        
        # Calculate expected tick move per 10 units of aggressive imbalance
        slope = np.polyfit(valid['rolling_net_vol'], valid[f'fwd_ret_{h}'], 1)[0]
        impact_per_10u = slope * 10
        
        print(f"Horizon: T+{h} ticks")
        print(f"  Correlation (Net Vol vs Ret): {corr:.4f}")
        print(f"  Impact per 10 net units hit:  {impact_per_10u:.3f} ticks")
        print("-" * 40)
        
    print("\nLarge Trade Analysis:")
    # Look at moments where rolling volume is extreme (> 95th percentile absolute)
    threshold = df['rolling_net_vol'].abs().quantile(0.95)
    large_buys = df[df['rolling_net_vol'] > threshold]
    large_sells = df[df['rolling_net_vol'] < -threshold]
    
    print(f"Threshold for 'Large' volume imbalance: {threshold:.1f} units over 20 ticks")
    print(f"When Large Aggressor BUY occurs, AVG forward return (T+30): {large_buys['fwd_ret_30'].mean():.3f} ticks")
    print(f"When Large Aggressor SELL occurs, AVG forward return (T+30): {large_sells['fwd_ret_30'].mean():.3f} ticks")

if __name__ == "__main__":
    analyze_informed_flow()
