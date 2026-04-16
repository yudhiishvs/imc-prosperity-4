import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

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
OUT_DIR = pathlib.Path(__file__).resolve().parent / "advanced_eda" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(day: int):
    # Load Prices
    price_file = DATA / f"prices_round_1_day_{day}.csv"
    if not price_file.exists():
        return None, None
        
    prices = pd.read_csv(price_file, sep=";")
    prices = prices[prices['product'] == 'ASH_COATED_OSMIUM'].copy()
    for c in prices.columns:
        if c not in ['product', 'timestamp', 'day']:
            prices[c] = pd.to_numeric(prices[c], errors='coerce')
            
    prices.sort_values("timestamp", inplace=True)
    prices.reset_index(drop=True, inplace=True)
    
    # Calculate Mids
    prices['l1_mid'] = (prices['bid_price_1'] + prices['ask_price_1']) / 2.0
    prices['l2_mid'] = (prices['bid_price_2'] + prices['ask_price_2']) / 2.0
    
    # Load Trades
    trade_file = DATA / f"trades_round_1_day_{day}.csv"
    if not trade_file.exists():
        trades = pd.DataFrame()
    else:
        trades = pd.read_csv(trade_file, sep=";")
        trades = trades[trades['symbol'] == 'ASH_COATED_OSMIUM'].copy()
        trades.sort_values("timestamp", inplace=True)
        
    return prices, trades

def plot_osmium_day(day: int):
    print(f"Plotting Day {day}...")
    prices, trades = load_data(day)
    if prices is None or prices.empty:
        print(f"No price data found for Day {day}")
        return
        
    # We will interpolate missing L1 values purely for plotting aesthetics
    prices['bid_price_1'].interpolate(method='linear', inplace=True)
    prices['ask_price_1'].interpolate(method='linear', inplace=True)
    prices['l1_mid'].interpolate(method='linear', inplace=True)
    
    # Merge trades with prices to classify aggressor side if possible
    # We merge_asof to get the prevailing mid at the exact or closest previous timestamp
    if not trades.empty:
        trades = pd.merge_asof(trades, prices[['timestamp', 'l1_mid']], on='timestamp', direction='backward')
        
        # Determine aggressor: if trade price >= mid, likely an aggressor buy. Else sell.
        aggressor_buys = trades[trades['price'] > trades['l1_mid']]
        aggressor_sells = trades[trades['price'] < trades['l1_mid']]
        mid_trades = trades[trades['price'] == trades['l1_mid']]
    else:
        aggressor_buys = pd.DataFrame()
        aggressor_sells = pd.DataFrame()
        mid_trades = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Prices
    ax.plot(prices['timestamp'], prices['ask_price_2'], color='pink', alpha=0.9, linewidth=1, label='L2 Ask')
    ax.plot(prices['timestamp'], prices['ask_price_1'], color='red', alpha=0.9, linewidth=1.5, label='L1 Ask')
    
    ax.plot(prices['timestamp'], prices['bid_price_1'], color='green', alpha=0.9, linewidth=1.5, label='L1 Bid')
    ax.plot(prices['timestamp'], prices['bid_price_2'], color='lightgreen', alpha=0.9, linewidth=1, label='L2 Bid')
    
    ax.plot(prices['timestamp'], prices['l2_mid'], color='purple', alpha=0.4, linewidth=1, linestyle=':', label='L2 Mid')
    ax.plot(prices['timestamp'], prices['l1_mid'], color='blue', alpha=0.7, linewidth=1.5, linestyle='--', label='L1 Mid')
    
    # Trades
    if not aggressor_buys.empty:
        # Green Up Triangles for Aggressive Buys (Someone hit the ask)
        ax.scatter(aggressor_buys['timestamp'], aggressor_buys['price'], 
                   color='lime', marker='^', s=40, zorder=5, label='Aggressor Buy Trade')
                   
    if not aggressor_sells.empty:
        # Red Down Triangles for Aggressive Sells (Someone hit the bid)
        ax.scatter(aggressor_sells['timestamp'], aggressor_sells['price'], 
                   color='red', marker='v', s=40, zorder=5, label='Aggressor Sell Trade')
                   
    if not mid_trades.empty:
        # Yellow Circles for trades matching exact mid (Unusual, maybe block trades)
        ax.scatter(mid_trades['timestamp'], mid_trades['price'], 
                   color='yellow', marker='o', edgecolors='black', s=40, zorder=5, label='Mid-Price Trade')

    ax.set_title(f"Osmium orderbook and trades (Day {day})", fontsize=18)
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("Price", fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = OUT_DIR / f"osmium_trades_day_{day}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved: {out_file}")

def main():
    days = [-2, -1, 0]
    for d in days:
        plot_osmium_day(d)

if __name__ == "__main__":
    main()
