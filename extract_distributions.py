"""
Distribution Extraction for prosperity4mcbt Calibration
========================================================
Extracts proper statistical distributions from ALL available data sources:
  - Historical CSVs (data/ROUND_2/prices_round_2_day_{-1,0,1}.csv)
  - Historical trades (data/ROUND_2/trades_round_2_day_{-1,0,1}.csv)
  - Live test logs: 307924 (best_strat), 308866 (gradient), 298967 (snipe),
    296479 (hold_1_unit), 297254 (sweep_and_quote), 297739 (normal_quotes)
"""

import json
import glob
import os
import numpy as np
import pandas as pd
from collections import defaultdict

BASE = "/Users/vedant/Quant/Prosperity4/imc-prosperity-4"

# ──────────────────────────────────────────────────────────────────────────────
# 1. HISTORICAL PRICE DATA (The Ground Truth)
# ──────────────────────────────────────────────────────────────────────────────

def load_historical_prices():
    """Load all 3 days of price CSVs."""
    files = sorted(glob.glob(os.path.join(BASE, "data/ROUND_2/prices_round_2_day_*.csv")))
    dfs = []
    for f in files:
        day = os.path.basename(f).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(f, sep=';')
        df['day_label'] = day
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_historical_trades():
    """Load all 3 days of trade CSVs."""
    files = sorted(glob.glob(os.path.join(BASE, "data/ROUND_2/trades_round_2_day_*.csv")))
    dfs = []
    for f in files:
        day = os.path.basename(f).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(f, sep=';')
        df['day_label'] = day
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. LIVE LOG DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_live_log(path):
    """Load a prosperity .log file and parse activitiesLog + tradeHistory."""
    with open(path) as f:
        data = json.load(f)
    
    activities = data.get("activitiesLog", [])
    if isinstance(activities, str):
        lines = activities.strip().split('\n')
        header = lines[0].split(';')
        parsed = []
        for line in lines[1:]:
            if not line: continue
            row = line.split(';')
            parsed.append(dict(zip(header, row)))
        activities = parsed
    
    trades = data.get("tradeHistory", [])
    if isinstance(trades, str):
        trades = json.loads(trades)
    
    return activities, trades

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_ash_spread_distribution(prices_df):
    """Extract the full spread distribution for ASH."""
    ash = prices_df[prices_df['product'] == 'ASH_COATED_OSMIUM'].copy()
    ash = ash.dropna(subset=['bid_price_1', 'ask_price_1'])
    ash = ash[(ash['bid_price_1'] > 0) & (ash['ask_price_1'] > 0)]
    ash['spread'] = ash['ask_price_1'] - ash['bid_price_1']
    
    print("=" * 70)
    print("ASH SPREAD DISTRIBUTION")
    print("=" * 70)
    print(f"  N = {len(ash)}")
    print(f"  Mean: {ash['spread'].mean():.2f}")
    print(f"  Median: {ash['spread'].median():.2f}")
    print(f"  Std: {ash['spread'].std():.2f}")
    print(f"  Min: {ash['spread'].min()}")
    print(f"  Max: {ash['spread'].max()}")
    print(f"\n  Value counts (top 10):")
    vc = ash['spread'].value_counts().sort_index()
    for val, count in vc.items():
        pct = count / len(ash) * 100
        print(f"    Spread={val:5.0f}: {count:5d} ({pct:5.1f}%)")
    return ash

def analyze_ash_l1_volume_distribution(prices_df):
    """Extract L1 bid/ask volume distributions for ASH."""
    ash = prices_df[prices_df['product'] == 'ASH_COATED_OSMIUM'].copy()
    ash = ash.dropna(subset=['bid_volume_1', 'ask_volume_1'])
    
    print("\n" + "=" * 70)
    print("ASH L1 VOLUME DISTRIBUTION")
    print("=" * 70)
    for side, col in [("BID", "bid_volume_1"), ("ASK", "ask_volume_1")]:
        vols = ash[col].dropna()
        vols = vols[vols > 0]
        print(f"\n  {side} Volume (N={len(vols)}):")
        print(f"    Mean: {vols.mean():.2f}")
        print(f"    Median: {vols.median():.2f}")
        print(f"    Std: {vols.std():.2f}")
        print(f"    Min: {vols.min()}")
        print(f"    Max: {vols.max()}")
        vc = vols.value_counts().sort_index()
        print(f"    Value counts:")
        for val, count in vc.items():
            pct = count / len(vols) * 100
            if pct > 0.5:
                print(f"      Vol={val:5.0f}: {count:5d} ({pct:5.1f}%)")

def analyze_ash_l2_volume_distribution(prices_df):
    """Extract L2 bid/ask volume distributions for ASH."""
    ash = prices_df[prices_df['product'] == 'ASH_COATED_OSMIUM'].copy()
    
    print("\n" + "=" * 70)
    print("ASH L2 VOLUME & PRESENCE DISTRIBUTION")
    print("=" * 70)
    
    for side, vol_col, price_col in [("BID", "bid_volume_2", "bid_price_2"), ("ASK", "ask_volume_2", "ask_price_2")]:
        has_l2 = ash[ash[price_col].notna() & (ash[price_col] > 0)] if side == "BID" else ash[ash[price_col].notna() & (ash[price_col] > 0)]
        pct_present = len(has_l2) / len(ash) * 100
        
        vols = has_l2[vol_col].dropna()
        vols = vols[vols > 0]
        
        print(f"\n  {side} L2 (Present {pct_present:.1f}% of ticks, N={len(vols)}):")
        print(f"    Mean: {vols.mean():.2f}")
        print(f"    Median: {vols.median():.2f}")
        print(f"    Std: {vols.std():.2f}")
        print(f"    Min: {vols.min()}")
        print(f"    Max: {vols.max()}")
        vc = vols.value_counts().sort_index()
        print(f"    Value counts:")
        for val, count in vc.items():
            pct = count / len(vols) * 100
            if pct > 1.0:
                print(f"      Vol={val:5.0f}: {count:5d} ({pct:5.1f}%)")

    # L1-L2 gap distribution
    print("\n  L1-L2 GAP DISTRIBUTION (bid side):")
    has_both = ash[ash['bid_price_2'].notna() & (ash['bid_price_2'] > 0) & (ash['bid_price_1'] > 0)]
    gaps = has_both['bid_price_1'] - has_both['bid_price_2']
    if len(gaps) > 0:
        print(f"    Mean gap: {gaps.mean():.2f}")
        print(f"    Median gap: {gaps.median():.2f}")
        vc = gaps.value_counts().sort_index()
        for val, count in vc.items():
            pct = count / len(gaps) * 100
            if pct > 1.0:
                print(f"      Gap={val:5.0f}: {count:5d} ({pct:5.1f}%)")

def analyze_ash_trade_distributions(trades_df):
    """Extract trade quantity, inter-arrival time, and price deviation distributions."""
    ash = trades_df[trades_df['symbol'] == 'ASH_COATED_OSMIUM'].copy()
    
    print("\n" + "=" * 70)
    print("ASH TRADE DISTRIBUTIONS")
    print("=" * 70)
    
    # Quantity distribution
    print(f"\n  QUANTITY DISTRIBUTION (N={len(ash)}):")
    print(f"    Mean: {ash['quantity'].mean():.2f}")
    print(f"    Median: {ash['quantity'].median():.2f}")
    print(f"    Std: {ash['quantity'].std():.2f}")
    print(f"    Min: {ash['quantity'].min()}")
    print(f"    Max: {ash['quantity'].max()}")
    vc = ash['quantity'].value_counts().sort_index()
    print(f"    Value counts:")
    for val, count in vc.items():
        pct = count / len(ash) * 100
        print(f"      Qty={val:5d}: {count:5d} ({pct:5.1f}%)")
    
    # Inter-arrival time distribution
    print(f"\n  INTER-ARRIVAL TIME DISTRIBUTION:")
    for day in sorted(ash['day_label'].unique()):
        day_trades = ash[ash['day_label'] == day].sort_values('timestamp')
        if len(day_trades) > 1:
            iat = day_trades['timestamp'].diff().dropna()
            print(f"    Day {day}: Mean={iat.mean():.0f}ts, Median={iat.median():.0f}ts, Std={iat.std():.0f}ts, Min={iat.min():.0f}ts, Max={iat.max():.0f}ts")
            # Percentiles
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"      P{p}: {iat.quantile(p/100):.0f}ts")

def analyze_ash_whale_distribution(prices_df, trades_df):
    """Classify whale vs reverter trades and extract their separate distributions."""
    ash_prices = prices_df[prices_df['product'] == 'ASH_COATED_OSMIUM'].copy()
    ash_prices = ash_prices[ash_prices['mid_price'] > 0]
    
    ash_trades = trades_df[trades_df['symbol'] == 'ASH_COATED_OSMIUM'].copy()
    
    print("\n" + "=" * 70)
    print("ASH WHALE vs REVERTER CLASSIFICATION")
    print("=" * 70)
    
    # Build mid_price lookup per day+timestamp
    mid_lookup = {}
    for _, row in ash_prices.iterrows():
        key = (row['day_label'], row['timestamp'])
        mid_lookup[key] = row['mid_price']
    
    # For each trade, compute the mid move around it
    whale_qtys = []
    reverter_qtys = []
    whale_moves = []
    reverter_moves = []
    
    for _, trade in ash_trades.iterrows():
        day = trade['day_label']
        ts = trade['timestamp']
        qty = trade['quantity']
        
        prev_mid = mid_lookup.get((day, ts - 100))
        curr_mid = mid_lookup.get((day, ts))
        next_mid = mid_lookup.get((day, ts + 100))
        
        if prev_mid is not None and curr_mid is not None:
            move = abs(curr_mid - prev_mid)
            if move >= 6:  # "Break" threshold from FINDINGS.md
                whale_qtys.append(qty)
                whale_moves.append(move)
            else:
                reverter_qtys.append(qty)
                reverter_moves.append(move)
    
    print(f"\n  WHALE trades (|move| >= 6 at time of trade): {len(whale_qtys)}")
    if whale_qtys:
        wq = np.array(whale_qtys)
        print(f"    Quantity: Mean={wq.mean():.2f}, Median={np.median(wq):.1f}, Std={wq.std():.2f}, Min={wq.min()}, Max={wq.max()}")
        vc = pd.Series(whale_qtys).value_counts().sort_index()
        for val, count in vc.items():
            pct = count / len(whale_qtys) * 100
            print(f"      Qty={val:5d}: {count:5d} ({pct:5.1f}%)")
        
        wm = np.array(whale_moves)
        print(f"    Move Size: Mean={wm.mean():.2f}, Median={np.median(wm):.1f}, Std={wm.std():.2f}, Min={wm.min()}, Max={wm.max()}")
    
    print(f"\n  REVERTER trades (|move| < 6): {len(reverter_qtys)}")
    if reverter_qtys:
        rq = np.array(reverter_qtys)
        print(f"    Quantity: Mean={rq.mean():.2f}, Median={np.median(rq):.1f}, Std={rq.std():.2f}, Min={rq.min()}, Max={rq.max()}")
        vc = pd.Series(reverter_qtys).value_counts().sort_index()
        for val, count in vc.items():
            pct = count / len(reverter_qtys) * 100
            if pct > 1.0:
                print(f"      Qty={val:5d}: {count:5d} ({pct:5.1f}%)")

def analyze_ash_ou_params(prices_df):
    """Estimate OU parameters from historical data per day."""
    print("\n" + "=" * 70)
    print("ASH ORNSTEIN-UHLENBECK PARAMETER ESTIMATION")
    print("=" * 70)
    
    for day in sorted(prices_df['day_label'].unique()):
        ash = prices_df[(prices_df['product'] == 'ASH_COATED_OSMIUM') & (prices_df['day_label'] == day)].copy()
        ash = ash[ash['mid_price'] > 0].sort_values('timestamp')
        mids = ash['mid_price'].values
        
        if len(mids) < 100:
            continue
        
        # Estimate OU: dX = theta*(mu - X)dt + sigma*dW
        # Using regression: X_{t+1} - X_t = a + b*X_t + eps
        dx = np.diff(mids)
        x = mids[:-1]
        
        # Linear regression
        A = np.column_stack([np.ones_like(x), x])
        result = np.linalg.lstsq(A, dx, rcond=None)
        a, b = result[0]
        
        theta = -b  # mean-reversion speed
        mu = a / theta if theta != 0 else 0  # long-run mean
        residuals = dx - (a + b * x)
        sigma = residuals.std()
        half_life = np.log(2) / theta if theta > 0 else float('inf')
        
        # Lag-1 autocorrelation of returns
        ac1 = np.corrcoef(dx[:-1], dx[1:])[0, 1]
        
        print(f"\n  Day {day}:")
        print(f"    theta (reversion speed): {theta:.4f}")
        print(f"    mu (long-run mean): {mu:.2f}")
        print(f"    sigma (noise): {sigma:.4f}")
        print(f"    half-life: {half_life:.2f} ticks")
        print(f"    lag-1 AC of returns: {ac1:.4f}")
        print(f"    actual mean of mids: {np.mean(mids):.2f}")
        print(f"    std of mids: {np.std(mids):.2f}")

def analyze_pepper_distributions(prices_df, trades_df):
    """Extract PEPPER spread and trade distributions."""
    print("\n" + "=" * 70)
    print("PEPPER SPREAD DISTRIBUTION")
    print("=" * 70)
    
    pep = prices_df[prices_df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
    pep = pep.dropna(subset=['bid_price_1', 'ask_price_1'])
    pep = pep[(pep['bid_price_1'] > 0) & (pep['ask_price_1'] > 0)]
    pep['spread'] = pep['ask_price_1'] - pep['bid_price_1']
    
    for day in sorted(pep['day_label'].unique()):
        day_df = pep[pep['day_label'] == day]
        sp = day_df['spread']
        print(f"\n  Day {day} (N={len(day_df)}):")
        print(f"    Mean: {sp.mean():.2f}, Median: {sp.median():.2f}, Std: {sp.std():.2f}")
        vc = sp.value_counts().sort_index()
        for val, count in vc.items():
            pct = count / len(day_df) * 100
            if pct > 1.0:
                print(f"      Spread={val:5.0f}: {count:5d} ({pct:5.1f}%)")
    
    # PEPPER trade quantity distribution
    pep_trades = trades_df[trades_df['symbol'] == 'INTARIAN_PEPPER_ROOT'].copy()
    print(f"\n  PEPPER TRADE QUANTITY (N={len(pep_trades)}):")
    print(f"    Mean: {pep_trades['quantity'].mean():.2f}")
    print(f"    Std: {pep_trades['quantity'].std():.2f}")
    vc = pep_trades['quantity'].value_counts().sort_index()
    for val, count in vc.items():
        pct = count / len(pep_trades) * 100
        print(f"      Qty={val:5d}: {count:5d} ({pct:5.1f}%)")

def analyze_live_logs():
    """Parse all live test logs for additional validation data."""
    print("\n" + "=" * 70)
    print("LIVE LOG CROSS-VALIDATION")
    print("=" * 70)
    
    log_files = [
        ("best_strat/307924.log", "best_strat"),
        ("308866/308866.log", "gradient_ascent"),
        ("snipe/298967.log", "snipe_probe"),
        ("hold_1_unit/296479.log", "hold_1_unit"),
        ("sweep_and_quote/297254.log", "sweep_and_quote"),
        ("normal_quotes/297739.log", "normal_quotes"),
    ]
    
    for log_path, label in log_files:
        full_path = os.path.join(BASE, log_path)
        if not os.path.exists(full_path):
            print(f"\n  [{label}] - FILE NOT FOUND: {full_path}")
            continue
        
        try:
            activities, trades = load_live_log(full_path)
        except Exception as e:
            print(f"\n  [{label}] - PARSE ERROR: {e}")
            continue
        
        # Extract OSMIUM mid prices from activities log
        osmium_acts = [r for r in activities if r.get('product') == 'ASH_COATED_OSMIUM']
        
        if osmium_acts:
            mids = [float(r['mid_price']) for r in osmium_acts if r.get('mid_price') and float(r.get('mid_price', 0)) > 0]
            
            # Spread from activities
            spreads = []
            for r in osmium_acts:
                try:
                    b1 = float(r.get('bid_price_1', 0) or 0)
                    a1 = float(r.get('ask_price_1', 0) or 0)
                    if b1 > 0 and a1 > 0:
                        spreads.append(a1 - b1)
                except (ValueError, TypeError):
                    pass
            
            print(f"\n  [{label}] OSMIUM (ticks={len(osmium_acts)}):")
            if mids:
                print(f"    Mid: Mean={np.mean(mids):.2f}, Median={np.median(mids):.2f}, Std={np.std(mids):.2f}")
            if spreads:
                print(f"    Spread: Mean={np.mean(spreads):.2f}, Median={np.median(spreads):.2f}")
        
        # Extract OSMIUM trades
        osmium_trades = [t for t in trades if t.get('symbol') == 'ASH_COATED_OSMIUM']
        non_sub_trades = [t for t in osmium_trades if t.get('buyer') != 'SUBMISSION' and t.get('seller') != 'SUBMISSION']
        
        if non_sub_trades:
            qtys = [t['quantity'] for t in non_sub_trades]
            print(f"    Non-SUBMISSION trades: {len(non_sub_trades)}, Avg Qty={np.mean(qtys):.2f}, Max Qty={max(qtys)}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading historical data...")
    prices_df = load_historical_prices()
    trades_df = load_historical_trades()
    
    print(f"Loaded {len(prices_df)} price rows, {len(trades_df)} trade rows\n")
    
    analyze_ash_spread_distribution(prices_df)
    analyze_ash_l1_volume_distribution(prices_df)
    analyze_ash_l2_volume_distribution(prices_df)
    analyze_ash_trade_distributions(trades_df)
    analyze_ash_whale_distribution(prices_df, trades_df)
    analyze_ash_ou_params(prices_df)
    analyze_pepper_distributions(prices_df, trades_df)
    analyze_live_logs()
    
    print("\n\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
