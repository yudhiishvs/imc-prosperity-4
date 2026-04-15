import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib

def resolve_repo_root(start: pathlib.Path) -> pathlib.Path:
    cur = start
    for _ in range(6):
        if (cur / "pyproject.toml").exists() and (cur / "data").exists():
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

def clean_osmium(prices: pd.DataFrame) -> pd.DataFrame:
    osm = prices[prices["product"] == "ASH_COATED_OSMIUM"].copy()
    osm.sort_values(["day", "timestamp"], inplace=True)
    osm.reset_index(drop=True, inplace=True)

    # Use only real book mids (both sides present); avoid treating broken book as a price.
    valid = osm["bid_price_1"].notna() & osm["ask_price_1"].notna()
    osm["raw_mid"] = np.where(valid, (osm["bid_price_1"] + osm["ask_price_1"]) / 2.0, np.nan)

    # Interpolate within each day only (prevents cross-day leakage).
    osm["clean_mid"] = (
        osm.groupby("day")["raw_mid"]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        .astype(float)
    )

    osm["spread_l1"] = (osm["ask_price_1"] - osm["bid_price_1"]).astype(float)
    osm.loc[~valid, "spread_l1"] = np.nan

    # OIM features (L1 and total-book) with safe denominator.
    b1 = osm["bid_volume_1"].fillna(0.0)
    a1 = osm["ask_volume_1"].fillna(0.0)
    den1 = (b1 + a1).replace(0.0, np.nan)
    osm["oim_l1"] = ((b1 - a1) / den1).fillna(0.0)

    bid_tot = osm[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0.0).sum(axis=1)
    ask_tot = osm[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0.0).sum(axis=1)
    dent = (bid_tot + ask_tot).replace(0.0, np.nan)
    osm["oim_total"] = ((bid_tot - ask_tot) / dent).fillna(0.0)

    return osm


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

def analyze_parameters():
    print("Loading Data...")
    df = load_prices()

    osm = clean_osmium(df)
    n_total = len(osm)
    n_valid = int(osm["clean_mid"].notna().sum())
    print(f"OSMIUM rows: {n_total:,} | usable mid rows (after per-day interpolation): {n_valid:,}")

    print("\n===============================")
    print("--- 0. SPREAD + MID MOVE BASELINES ---")
    spread = osm["spread_l1"].dropna()
    if len(spread) > 0:
        print("L1 spread stats (ticks):")
        print(spread.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

    # 1-tick mid changes (per day) to avoid cross-day discontinuities
    osm["mid_change_1"] = osm.groupby("day")["clean_mid"].shift(-1) - osm["clean_mid"]
    mc = osm["mid_change_1"].dropna()
    if len(mc) > 0:
        print("\n1-tick clean-mid change stats (ticks):")
        print(mc.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
    
    print("\n===============================")
    print("--- 1. OPTIMIZING EMA ALPHA ---")
    # Goal: find an EMA that tracks the time-varying center of price *within a day*.
    # We minimize MSE between EMA(t) and mid(t + horizon).
    horizon = 5
    osm["target_mid"] = osm.groupby("day")["clean_mid"].shift(-horizon)
    osm = osm.dropna(subset=["target_mid", "clean_mid"]).copy()
    
    alphas = np.linspace(0.01, 1.0, 100)
    mses = []
    
    for alpha in alphas:
        ema_col = osm.groupby("day")["clean_mid"].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
        mse = np.mean((osm["target_mid"] - ema_col) ** 2)
        mses.append(mse)
        
    optimal_alpha = alphas[np.argmin(mses)]
    print(f"Optimal EMA Alpha (Minimized MSE vs +{horizon} ticks): {optimal_alpha:.4f}")
    
    plt.figure(figsize=(8,5))
    plt.plot(alphas, mses, color='blue', linewidth=2)
    plt.title(f"EMA Alpha Optimization (Targeting +{horizon} Future Ticks)")
    plt.xlabel("EMA Weight (Alpha)")
    plt.ylabel("Mean Squared Error vs Future Mid-Price")
    plt.axvline(optimal_alpha, color='red', linestyle='--', label=f'Optimal: {optimal_alpha:.2f}')
    plt.legend()
    plt.tight_layout()
    plot_alpha = OUT_DIR / "osmium_ema_optimization.png"
    plt.savefig(plot_alpha, dpi=150)
    print(f"Saved Alpha Optimization plot to {plot_alpha}")
    
    print("\n===============================")
    print("--- 2. OPTIMIZING OIM MULTIPLIER ---")
    # We estimate how much OIM predicts *future mid change*.
    # Then we translate that into a "fair skew" multiplier in ticks.
    
    osm["opt_ema"] = osm.groupby("day")["clean_mid"].transform(lambda s: s.ewm(alpha=optimal_alpha, adjust=False).mean())
    osm["residual"] = osm["target_mid"] - osm["opt_ema"]
    
    from scipy.stats import linregress
    for feat in ["oim_l1", "oim_total"]:
        slope, intercept, r_value, p_value, std_err = linregress(osm[feat], osm["residual"])
        print(f"\nFeature: {feat}")
        print(f"Residual ≈ intercept + slope * OIM")
        print(f"slope (ticks per OIM unit): {slope:.4f}")
        print(f"R^2: {r_value**2:.4f}")
        print(f"StdErr(slope): {std_err:.4f}")

    print("\n===============================")
    print("--- 3. OPTIMIZING QUOTING PARAMETERS ---")
    # Base Quote Size derived from empirical volume percentiles of market trades.
    tf0 = pd.read_csv(DATA / "trades_round_1_day_0.csv", sep=";")
    tf1 = pd.read_csv(DATA / "trades_round_1_day_-1.csv", sep=";")
    tf2 = pd.read_csv(DATA / "trades_round_1_day_-2.csv", sep=";")
    tf = pd.concat([tf0, tf1, tf2])
    
    tf_osm = tf[tf['symbol'] == 'ASH_COATED_OSMIUM']['quantity']
    
    p90_vol = tf_osm.quantile(0.90)
    p50_vol = tf_osm.quantile(0.50)
    
    print(f"Trade Vol Median: {p50_vol}")
    print(f"Trade Vol P90: {p90_vol}")
    
    recommended_base = int(round(p90_vol))
    recommended_thresh = int(80 - (2 * recommended_base))
    
    print("\n==================================")
    print("SUGGESTED STARTING CONFIG (DATA-DRIVEN):")
    print("==================================")
    print(f"OSMIUM_EMA_ALPHA ~= {optimal_alpha:.2f}")
    print(f"OSMIUM_BASE_QUOTE_SIZE ~= {recommended_base}  (from trade size P90)")
    print(f"OSMIUM_EMERGENCY_THRESHOLD ~= {recommended_thresh}  (rule-of-thumb vs base size)")
    print(f"OSMIUM_EMERGENCY_TARGET ~= {int(recommended_thresh/2)}")
    print("\nNote: Do NOT round OIM slope to int; treat as a small continuous skew and tune via backtest.")

if __name__ == "__main__":
    analyze_parameters()
