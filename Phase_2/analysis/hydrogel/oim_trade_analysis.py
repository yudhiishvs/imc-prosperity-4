import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "../../data/ROUND_3")
PLOT_DIR = os.path.join(SCRIPT_DIR, "../../plots/hydrogel")

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e',
    'axes.edgecolor': '#4a4a6a', 'axes.labelcolor': '#e0e0e0',
    'axes.titlecolor': '#e0e0e0', 'text.color': '#e0e0e0',
    'xtick.color': '#a0a0c0', 'ytick.color': '#a0a0c0',
    'grid.color': '#2a2a4a', 'grid.linestyle': '--',
    'grid.alpha': 0.7, 'lines.linewidth': 1.5
})
BLUE = '#0f6fc6'; GOLD = '#f5a623'; ACCENT = '#e94560'; GREEN = '#39d353'

def load_data():
    price_dfs = []
    trade_dfs = []
    
    for day in range(3):
        # Load prices
        p_df = pd.read_csv(os.path.join(BASE_PATH, f"prices_round_3_day_{day}.csv"), sep=";")
        p_df = p_df[p_df['product'] == 'HYDROGEL_PACK'].copy()
        p_df['day'] = day
        # Ensure timestamp is sorted
        p_df = p_df.sort_values('timestamp').reset_index(drop=True)
        price_dfs.append(p_df)
        
        # Load trades
        t_df = pd.read_csv(os.path.join(BASE_PATH, f"trades_round_3_day_{day}.csv"), sep=";")
        t_df = t_df[t_df['symbol'] == 'HYDROGEL_PACK'].copy()
        t_df['day'] = day
        trade_dfs.append(t_df)
        
    all_prices = pd.concat(price_dfs, ignore_index=True)
    all_trades = pd.concat(trade_dfs, ignore_index=True)
    return all_prices, all_trades

def process_data(prices, trades):
    print("Processing data...")
    # Calculate OIM and Volume
    prices['bid_volume_1'].fillna(0, inplace=True)
    prices['ask_volume_1'].fillna(0, inplace=True)
    
    # In the dataset, ask volumes are usually positive or negative depending on round, let's take absolute
    prices['ask_volume_1'] = prices['ask_volume_1'].abs()
    
    total_vol = prices['bid_volume_1'] + prices['ask_volume_1']
    prices['OIM'] = np.where(total_vol > 0, (prices['bid_volume_1'] - prices['ask_volume_1']) / total_vol, 0)
    prices['Total_Volume'] = total_vol
    
    # Forward Returns
    # We want to see the return from the current mid price to the future mid price
    for h in [1, 3, 5, 10]:
        # group by day to prevent leak across days
        prices[f'mid_fwd_ret_{h}'] = prices.groupby('day')['mid_price'].shift(-h) - prices['mid_price']
        
    # Aggregate trades by timestamp and day
    # We want to classify trades. We merge prices into trades to get the bid/ask at that timestamp.
    # We assume trades happen at the state they are matched. 
    trades_with_prices = pd.merge(trades, prices[['day', 'timestamp', 'bid_price_1', 'ask_price_1', 'mid_price']], 
                                  on=['day', 'timestamp'], how='left')
    
    # Classify trades
    # Buyer-initiated: price >= ask_price_1 (or price > mid_price)
    trades_with_prices['buyer_initiated'] = trades_with_prices['price'] >= trades_with_prices['ask_price_1']
    trades_with_prices['seller_initiated'] = trades_with_prices['price'] <= trades_with_prices['bid_price_1']
    
    # For some trades that fall in between, we can use mid_price as a threshold
    trades_with_prices.loc[(~trades_with_prices['buyer_initiated']) & (~trades_with_prices['seller_initiated']), 'buyer_initiated'] = trades_with_prices['price'] > trades_with_prices['mid_price']
    trades_with_prices.loc[(~trades_with_prices['buyer_initiated']) & (~trades_with_prices['seller_initiated']), 'seller_initiated'] = trades_with_prices['price'] < trades_with_prices['mid_price']

    # Aggregate to timestamp level
    trade_agg = trades_with_prices.groupby(['day', 'timestamp']).agg(
        buy_trade_qty=('quantity', lambda x: x[trades_with_prices.loc[x.index, 'buyer_initiated']].sum()),
        sell_trade_qty=('quantity', lambda x: x[trades_with_prices.loc[x.index, 'seller_initiated']].sum())
    ).reset_index()
    
    # Merge back to prices, but shift by -1 because we want OIM at t to predict trade at t+1
    # Actually, in the simulator, if we see a state at t, the trades that occur between t and t+1 are 
    # reported at t+100 (or t+1 tick). We should shift trade_agg timestamp by -100 to align trade at t+100 with state at t.
    trade_agg['target_state_ts'] = trade_agg['timestamp'] - 100
    
    prices = pd.merge(prices, trade_agg[['day', 'target_state_ts', 'buy_trade_qty', 'sell_trade_qty']], 
                      left_on=['day', 'timestamp'], right_on=['day', 'target_state_ts'], how='left')
    
    prices['buy_trade_qty'] = prices['buy_trade_qty'].fillna(0)
    prices['sell_trade_qty'] = prices['sell_trade_qty'].fillna(0)
    prices['any_trade_next_tick'] = (prices['buy_trade_qty'] > 0) | (prices['sell_trade_qty'] > 0)
    
    return prices

def analyze_and_plot(df):
    print("Generating plots...")
    # Bin OIM into 10 buckets
    df['OIM_Bin'] = pd.cut(df['OIM'], bins=10)
    
    # 1. Probability of Trade vs OIM
    prob_buy = df.groupby('OIM_Bin')['buy_trade_qty'].apply(lambda x: (x > 0).mean())
    prob_sell = df.groupby('OIM_Bin')['sell_trade_qty'].apply(lambda x: (x > 0).mean())
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x_labels = [str(i.mid) for i in prob_buy.index]
    x_pos = np.arange(len(x_labels))
    
    width = 0.35
    ax1.bar(x_pos - width/2, prob_buy, width, label='Prob of Buy Trade (Adverse Ask)', color=GREEN)
    ax1.bar(x_pos + width/2, prob_sell, width, label='Prob of Sell Trade (Adverse Bid)', color=ACCENT)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{i.left:.1f} to {i.right:.1f}" for i in prob_buy.index], rotation=45)
    ax1.set_xlabel("Order Imbalance (OIM) Bin")
    ax1.set_ylabel("Probability of Trade in Next Tick")
    ax1.set_title("Does OIM predict market trades in the next tick?")
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(PLOT_DIR, "hg_oim_vs_trade_prob.png"), dpi=200)
    
    # 2. Forward Returns vs OIM
    fwd_returns = df.groupby('OIM_Bin')[['mid_fwd_ret_1', 'mid_fwd_ret_3', 'mid_fwd_ret_5', 'mid_fwd_ret_10']].mean()
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_pos, fwd_returns['mid_fwd_ret_1'], marker='o', label='T+1 Tick', color=GOLD)
    ax2.plot(x_pos, fwd_returns['mid_fwd_ret_3'], marker='s', label='T+3 Ticks', color=BLUE)
    ax2.plot(x_pos, fwd_returns['mid_fwd_ret_5'], marker='^', label='T+5 Ticks', color=GREEN)
    ax2.plot(x_pos, fwd_returns['mid_fwd_ret_10'], marker='d', label='T+10 Ticks', color=ACCENT)
    
    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{i.left:.1f} to {i.right:.1f}" for i in prob_buy.index], rotation=45)
    ax2.set_xlabel("Order Imbalance (OIM) Bin")
    ax2.set_ylabel("Expected Mid-Price Change (Ticks)")
    ax2.set_title("OIM Predictive Power on Short-Term Price Movement")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, "hg_oim_vs_fwd_ret.png"), dpi=200)

    # 3. Print threshold analysis
    print("\nOIM Threshold Analysis:")
    print("-" * 50)
    print(f"{'OIM Range':<25} | {'P(Adverse Ask)':<15} | {'P(Adverse Bid)':<15} | {'E[Ret T+3]':<15}")
    for bin_val, p_buy, p_sell, ret in zip(prob_buy.index, prob_buy, prob_sell, fwd_returns['mid_fwd_ret_3']):
        print(f"{str(bin_val):<25} | {p_buy:.2%}         | {p_sell:.2%}         | {ret:.4f}")
        
if __name__ == "__main__":
    prices, trades = load_data()
    df = process_data(prices, trades)
    analyze_and_plot(df)
    print("\nAnalysis complete. Plots saved to Phase_2/plots/hydrogel/")
