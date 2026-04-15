import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
from scipy.stats import linregress

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

def clean_mid_per_day(osm: pd.DataFrame) -> pd.DataFrame:
    osm = osm.copy()
    osm.sort_values(["day", "timestamp"], inplace=True)
    osm.reset_index(drop=True, inplace=True)

    valid = osm["bid_price_1"].notna() & osm["ask_price_1"].notna()
    osm["raw_mid"] = np.where(valid, (osm["bid_price_1"] + osm["ask_price_1"]) / 2.0, np.nan)
    osm["clean_mid"] = osm.groupby("day")["raw_mid"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    osm["spread_l1"] = (osm["ask_price_1"] - osm["bid_price_1"]).astype(float)
    osm.loc[~valid, "spread_l1"] = np.nan
    return osm


def compute_oim_features(osm: pd.DataFrame) -> pd.DataFrame:
    osm = osm.copy()
    b1 = osm["bid_volume_1"].fillna(0.0)
    a1 = osm["ask_volume_1"].fillna(0.0)
    den1 = (b1 + a1).replace(0.0, np.nan)
    osm["oim_l1"] = ((b1 - a1) / den1).fillna(0.0)

    bid_tot = osm[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0.0).sum(axis=1)
    ask_tot = osm[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0.0).sum(axis=1)
    dent = (bid_tot + ask_tot).replace(0.0, np.nan)
    osm["oim_total"] = ((bid_tot - ask_tot) / dent).fillna(0.0)
    return osm


def ar1_half_life(x: pd.Series) -> float | None:
    s = x.dropna().astype(float)
    if len(s) < 50:
        return None
    x0 = s.iloc[:-1].values
    x1 = s.iloc[1:].values
    if np.std(x0) < 1e-9:
        return None
    beta = np.polyfit(x0, x1, 1)[0]
    if beta <= 0 or beta >= 1:
        return None
    # half-life in ticks for AR(1): ln(0.5)/ln(beta)
    return float(np.log(0.5) / np.log(beta))


def analyze_osmium():
    print("Loading data...")
    df = load_data()
    if df.empty: return
    
    osmium = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()
    osmium = clean_mid_per_day(osmium)
    osmium = compute_oim_features(osmium)
    
    print("\n--- 1. Data Cleaning (Mid Price Spikes) ---")
    print(f"Total rows: {len(osmium)}")
    print(f"Missing/Zero mid_prices before interpolation: {osmium['raw_mid'].isna().sum()}")
    print(f"Missing/Zero mid_prices after interpolation: {osmium['clean_mid'].isna().sum()}")
    print(f"Rows with missing L1 side (bid or ask): {(osmium['bid_price_1'].isna() | osmium['ask_price_1'].isna()).sum()}")

    print("\n--- 1b. Spread + Volatility Baselines ---")
    spread = osmium["spread_l1"].dropna()
    if len(spread) > 0:
        print("L1 spread stats (ticks):")
        print(spread.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    osmium["ret_1"] = osmium.groupby("day")["clean_mid"].diff()
    r1 = osmium["ret_1"].dropna()
    if len(r1) > 0:
        print("\n1-tick clean-mid changes (ticks):")
        print(r1.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    print("\n--- 2. Drift vs. Mean Reversion Analysis ---")
    slopes = []
    means = []
    for day in osmium['day'].unique():
        day_data = osmium[osmium['day'] == day]
        slope, intercept, r_value, p_value, std_err = linregress(day_data['timestamp'], day_data['clean_mid'])
        slopes.append(slope)
        mean_val = np.mean(day_data['clean_mid'])
        means.append(mean_val)
        print(f"Day {day} Slope: {slope:.10f} | Mean: {mean_val:.4f} | R2: {r_value**2:.6f}")
        
    print(f"\nAverage Slope across all days: {np.mean(slopes):.10f}")
    data_fv = np.mean(osmium['clean_mid'])
    print(f"Global Mean Price (Data-Backed FV): {data_fv:.4f}")

    print("\n--- 2b. Mean Reversion Half-Life (AR(1) on deviations) ---")
    hl_all = ar1_half_life(osmium["clean_mid"] - data_fv)
    if hl_all is not None:
        print(f"Approx half-life vs global mean (ticks): {hl_all:.1f}")
    for day in sorted(osmium["day"].unique()):
        day_s = osmium.loc[osmium["day"] == day, "clean_mid"]
        day_mean = float(day_s.mean())
        hl = ar1_half_life(day_s - day_mean)
        if hl is not None:
            print(f"Day {day} half-life vs day mean (ticks): {hl:.1f}")

    print("\n--- 3. Orderbook Imbalance (OIM) Predictivity ---")
    for feat in ["oim_l1", "oim_total"]:
        print(f"\nFeature: {feat}")
        for ticks in [1, 2, 5, 10]:
            osmium[f"mid_change_{ticks}"] = osmium.groupby("day")["clean_mid"].shift(-ticks) - osmium["clean_mid"]
            corr = osmium[feat].corr(osmium[f"mid_change_{ticks}"])
            print(f"Correlation to +{ticks} tick change: {corr:.4f}")

        # Simple out-of-sample check: train on days -2/-1, test on day 0 using a linear slope.
        train = osmium[osmium["day"].isin([-2, -1])].dropna(subset=["clean_mid"])
        test = osmium[osmium["day"] == 0].dropna(subset=["clean_mid"])
        if len(train) > 1000 and len(test) > 1000:
            horizon = 1
            train_y = train.groupby("day")["clean_mid"].shift(-horizon) - train["clean_mid"]
            test_y = test.groupby("day")["clean_mid"].shift(-horizon) - test["clean_mid"]
            tr = train.assign(y=train_y).dropna(subset=["y"])
            te = test.assign(y=test_y).dropna(subset=["y"])
            slope = np.cov(tr[feat], tr["y"])[0, 1] / (np.var(tr[feat]) + 1e-9)
            pred = slope * te[feat]
            dir_acc = (np.sign(pred) == np.sign(te["y"])).mean()
            print(f"Train(-2,-1) -> Test(0): slope={slope:.4f} | direction-accuracy={dir_acc:.3f}")

    print("\n--- 4. Visualizing Spread Dynamics ---")
    day_plot = osmium[osmium['day'] == osmium['day'].iloc[0]].copy()
    
    subset = day_plot[day_plot['timestamp'] <= 100000]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(subset['timestamp'], subset['ask_price_1'], label='Ask L1', color='red', alpha=0.8, linewidth=1)
    ax.plot(subset['timestamp'], subset['bid_price_1'], label='Bid L1', color='green', alpha=0.8, linewidth=1)
    ax.plot(subset['timestamp'], subset['ask_price_2'], label='Ask L2', color='salmon', alpha=0.5, linewidth=1)
    ax.plot(subset['timestamp'], subset['bid_price_2'], label='Bid L2', color='lightgreen', alpha=0.5, linewidth=1)
    ax.plot(subset['timestamp'], subset['clean_mid'], label='Clean Mid (Interpolated)', color='blue', alpha=0.9, linewidth=1.5, linestyle='--')
    ax.axhline(data_fv, color='black', linestyle='-', label=f'Data-Backed FV ({data_fv:.1f})')

    ax.set_title(f"ASH_COATED_OSMIUM Spread Dynamics (L1, L2, Clean Mid)\nData-Backed FV = {data_fv:.2f}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plot1 = OUT_DIR / "osmium_spread_dynamics.png"
    plt.savefig(plot1, dpi=200)
    print(f"Saved spread dynamics plot to: {plot1}")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    micro = day_plot.head(200)
    
    ax2.plot(micro['timestamp'], micro['ask_price_1'], label='Ask L1', color='red', marker='o', markersize=3)
    ax2.plot(micro['timestamp'], micro['bid_price_1'], label='Bid L1', color='green', marker='o', markersize=3)
    ax2.plot(micro['timestamp'], micro['ask_price_2'], label='Ask L2', color='salmon', marker='x', markersize=3)
    ax2.plot(micro['timestamp'], micro['bid_price_2'], label='Bid L2', color='lightgreen', marker='x', markersize=3)
    ax2.plot(micro['timestamp'], micro['clean_mid'], label='Clean Mid', color='blue', linestyle='--')
    ax2.axhline(data_fv, color='black', linestyle='-', label=f'Global FV ({data_fv:.1f})')
    
    ax2.set_title("Osmium Micro-Structure (First 200 Trades)")
    ax2.legend()
    plt.tight_layout()
    plot2 = OUT_DIR / "osmium_spread_micro.png"
    plt.savefig(plot2, dpi=150)
    print(f"Saved micro structure plot to: {plot2}")

    print("\n--- 5. OIM vs Next-Tick Move (Binned) ---")
    tmp = osmium.dropna(subset=["clean_mid"]).copy()
    tmp["y1"] = tmp.groupby("day")["clean_mid"].shift(-1) - tmp["clean_mid"]
    tmp = tmp.dropna(subset=["y1"])

    for feat in ["oim_l1", "oim_total"]:
        # Use quantile bins for interpretability; duplicates can happen if distribution is spiky.
        bins = pd.qcut(tmp[feat], q=10, duplicates="drop")
        agg = tmp.groupby(bins)["y1"].mean()
        print(f"\nMean next-tick move by {feat} decile:")
        print(agg)

if __name__ == "__main__":
    analyze_osmium()
