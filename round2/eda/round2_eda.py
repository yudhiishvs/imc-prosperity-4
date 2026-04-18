"""
Prosperity 4 — Round 2 EDA
Products: ASH_COATED_OSMIUM (ASH), INTARIAN_PEPPER_ROOT (PEPPER)
Days: -1, 0, 1

Notebook usage:
    %run round2_eda.py          # run everything
    from round2_eda import load_data, plot_inefficiency_map
    prices, trades, merged = load_data()
    plot_inefficiency_map(merged)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── Paths & constants ─────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, "..", "data")
OUT_DIR   = os.path.join(_HERE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS     = [-1, 0, 1]
PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
SHORT    = {"ASH_COATED_OSMIUM": "ASH", "INTARIAN_PEPPER_ROOT": "PEPPER"}
COLORS   = {"ASH_COATED_OSMIUM": "#2196F3", "INTARIAN_PEPPER_ROOT": "#FF5722"}
DAY_COLORS = {-1: "#762a83", 0: "#d6604d", 1: "#1b7837"}

ASH_FAIR_VALUE = 10_000.0

# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices(days=DAYS) -> pd.DataFrame:
    frames = []
    for day in days:
        path = os.path.join(DATA_DIR, f"prices_round_2_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
    df = df[df["mid_price"] > 0]
    return df


def load_trades(days=DAYS) -> pd.DataFrame:
    frames = []
    for day in days:
        path = os.path.join(DATA_DIR, f"trades_round_2_day_{day}.csv")
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"symbol": "product"})
    return df


def load_data(days=DAYS):
    """
    Returns (prices, trades, merged).
    `merged` is trades joined to the most-recent prior mid-price tick,
    with a `deviation` column = trade_price - mid.
    """
    prices = load_prices(days)
    trades = load_trades(days)

    parts = []
    for product in PRODUCTS:
        for day in days:
            p = prices[(prices["product"] == product) & (prices["day"] == day)][
                ["timestamp", "mid_price"]
            ].sort_values("timestamp")
            t = trades[(trades["product"] == product) & (trades["day"] == day)].copy()
            if t.empty or p.empty:
                continue
            t = pd.merge_asof(
                t.sort_values("timestamp"),
                p.rename(columns={"mid_price": "mid"}),
                on="timestamp",
                direction="backward",
            )
            t["deviation"] = t["price"] - t["mid"]
            parts.append(t)

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return prices, trades, merged

# ── Utilities ─────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {name}")

# ── Stats printout ─────────────────────────────────────────────────────────────

def print_stats(prices=None, trades=None):
    if prices is None or trades is None:
        prices, trades, _ = load_data()

    print("=" * 70)
    print("PROSPERITY 4  ROUND 2 — EDA STATS")
    print("=" * 70)
    print(f"\nPrices shape : {prices.shape}")
    print(f"Trades shape : {trades.shape}")
    print(f"Prices cols  : {list(prices.columns)}")
    print(f"Trades cols  : {list(trades.columns)}")

    for prod in PRODUCTS:
        p = prices[prices["product"] == prod]
        t = trades[trades["product"] == prod]
        print(f"\n{'─'*60}")
        print(f"PRODUCT: {prod}  ({SHORT[prod]})")
        print(f"{'─'*60}")
        print(f"  Price rows : {len(p)}")
        print(f"  Trade rows : {len(t)}")
        print(f"\n  Mid Price (all days):\n{p['mid_price'].describe().to_string()}")

        for day in DAYS:
            m = p[p["day"] == day]["mid_price"]
            print(f"\n  Day {day:+d}: mean={m.mean():.2f}  std={m.std():.4f}  "
                  f"min={m.min():.2f}  max={m.max():.2f}  range={m.max()-m.min():.2f}")

        if len(t):
            print(f"\n  Trade price:\n{t['price'].describe().to_string()}")
            print(f"\n  Trade quantity:\n{t['quantity'].describe().to_string()}")
            for col in ("buyer", "seller"):
                if col in t.columns:
                    print(f"  {col.capitalize()}s: {t[col].fillna('MARKET').value_counts().head(5).to_dict()}")

        # Spread
        p2 = p.copy()
        p2["spread"] = p2["ask_price_1"] - p2["bid_price_1"]
        vs = p2[(p2["spread"].notna()) & (p2["spread"] > 0)]
        if len(vs):
            print(f"\n  Bid-Ask Spread:\n{vs['spread'].describe().to_string()}")
            for day in DAYS:
                s = vs[vs["day"] == day]["spread"]
                print(f"  Day {day:+d}: mean={s.mean():.2f}  median={s.median():.2f}")

        # Returns + autocorr
        for day in DAYS:
            m = p[p["day"] == day]["mid_price"].dropna()
            if len(m) < 2:
                continue
            r = m.diff().dropna()
            ac1, ac2, ac5 = r.autocorr(1), r.autocorr(2), r.autocorr(5)
            _, pval = stats.shapiro(r.sample(min(len(r), 5000), random_state=42))
            print(f"\n  Day {day:+d} Returns: mean={r.mean():.4f}  std={r.std():.4f}  "
                  f"skew={r.skew():.3f}  kurt={r.kurtosis():.3f}  normality_p={pval:.2e}")
            print(f"           ACF: lag1={ac1:.4f}  lag2={ac2:.4f}  lag5={ac5:.4f}")

    # Order book depth
    print(f"\n{'─'*60}")
    print("ORDER BOOK DEPTH")
    print(f"{'─'*60}")
    for prod in PRODUCTS:
        p = prices[prices["product"] == prod].copy()
        print(f"\n  {SHORT[prod]}:")
        for lvl in [1, 2, 3]:
            bp = p[f"bid_price_{lvl}"].dropna()
            bv = p[f"bid_volume_{lvl}"].dropna()
            av = p[f"ask_volume_{lvl}"].dropna()
            print(f"    L{lvl}: presence={len(bp)/len(p)*100:.1f}%  "
                  f"bid_vol_mean={bv.mean():.1f}  ask_vol_mean={av.mean():.1f}")
        mask = p["bid_volume_1"].notna() & p["ask_volume_1"].notna()
        sub = p[mask].copy()
        if len(sub):
            sub["imb"] = (sub["bid_volume_1"] - sub["ask_volume_1"]) / \
                         (sub["bid_volume_1"] + sub["ask_volume_1"])
            sub["wmid"] = (sub["bid_price_1"] * sub["ask_volume_1"] +
                           sub["ask_price_1"] * sub["bid_volume_1"]) / \
                          (sub["bid_volume_1"] + sub["ask_volume_1"])
            sub["wmid_diff"] = sub["wmid"] - sub["mid_price"]
            print(f"    Imbalance: mean={sub['imb'].mean():.4f}  std={sub['imb'].std():.4f}")
            print(f"    WMid−Mid:  mean={sub['wmid_diff'].mean():.4f}  std={sub['wmid_diff'].std():.4f}")

    # Trend analysis
    print(f"\n{'─'*60}")
    print("TREND ANALYSIS")
    print(f"{'─'*60}")
    for prod in PRODUCTS:
        print(f"\n  {SHORT[prod]}:")
        p = prices[prices["product"] == prod]
        for day in DAYS:
            m = p[p["day"] == day]["mid_price"].dropna().reset_index(drop=True)
            if len(m) < 10:
                continue
            x = np.arange(len(m))
            sl, ic, r, pv, _ = stats.linregress(x, m)
            print(f"    Day {day:+d}: slope={sl:.6f}/tick  total_drift={sl*(len(m)-1):.2f}  "
                  f"R²={r**2:.4f}  p={pv:.2e}")

    # Mean reversion
    print(f"\n{'─'*60}")
    print("MEAN REVERSION (delta ~ lagged level)")
    print(f"{'─'*60}")
    for prod in PRODUCTS:
        print(f"\n  {SHORT[prod]}:")
        p = prices[prices["product"] == prod]
        for day in DAYS:
            m = p[p["day"] == day]["mid_price"].dropna().values
            if len(m) < 10:
                continue
            dy, yl = np.diff(m), m[:-1]
            sl, _, _, pv, _ = stats.linregress(yl, dy)
            hl = f"{-np.log(2)/sl:.1f} ticks" if sl < 0 else "n/a"
            print(f"    Day {day:+d}: coeff={sl:.6f}  p={pv:.2e}  half-life={hl}")

    # Trade vs mid
    print(f"\n{'─'*60}")
    print("TRADE PRICE vs MID PRICE")
    print(f"{'─'*60}")
    for prod in PRODUCTS:
        t = trades[trades["product"] == prod]
        p_mid = prices[prices["product"] == prod][["day", "timestamp", "mid_price"]]
        parts = []
        for day in DAYS:
            td = t[t["day"] == day].sort_values("timestamp")
            pd_ = p_mid[p_mid["day"] == day].sort_values("timestamp")
            if td.empty or pd_.empty:
                continue
            m = pd.merge_asof(td, pd_.drop(columns=["day"]), on="timestamp")
            m["day"] = day
            parts.append(m)
        if not parts:
            continue
        mdf = pd.concat(parts)
        mdf["dev"] = mdf["price"] - mdf["mid_price"]
        print(f"\n  {SHORT[prod]}: mean_dev={mdf['dev'].mean():.3f}  "
              f"std={mdf['dev'].std():.3f}  "
              f"|dev|_mean={mdf['dev'].abs().mean():.3f}")
        for day in DAYS:
            dv = mdf[mdf["day"] == day]["dev"]
            if not dv.empty:
                print(f"    Day {day:+d}: mean={dv.mean():.3f}  std={dv.std():.3f}  "
                      f"pct_outside_half_spread={_pct_outside_spread(prices, prod, day, dv):.1f}%")


def _pct_outside_spread(prices, product, day, dev_series):
    p = prices[(prices["product"] == product) & (prices["day"] == day)]
    sp = (p["ask_price_1"] - p["bid_price_1"]).median()
    if np.isnan(sp):
        return float("nan")
    return (dev_series.abs() > sp / 2).mean() * 100


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_midprice_timeseries(prices=None, save=True):
    """Fig 01 — Mid price timeseries for both products across all days."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, prod in zip(axes, PRODUCTS):
        p = prices[prices["product"] == prod]
        for day in DAYS:
            pd_ = p[p["day"] == day]
            ax.plot(pd_["global_ts"], pd_["mid_price"],
                    color=DAY_COLORS[day], lw=0.7, alpha=0.85, label=f"Day {day:+d}")
        ax.set_title(f"{prod} — Mid Price", fontsize=11)
        ax.set_xlabel("Global Timestamp")
        ax.set_ylabel("Mid Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Mid Price Timeseries", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "01_midprice_timeseries.png")
    else:
        plt.show()
    return fig


def plot_midprice_distributions(prices=None, save=True):
    """Fig 02 — Mid price histogram + KDE for each product (all days combined)."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, prod in zip(axes, PRODUCTS):
        m = prices[prices["product"] == prod]["mid_price"].dropna()
        ax.hist(m, bins=80, density=True, alpha=0.65, color=COLORS[prod],
                edgecolor="white", lw=0.3)
        kde = gaussian_kde(m)
        xs = np.linspace(m.min(), m.max(), 400)
        ax.plot(xs, kde(xs), color="black", lw=1.5)
        ax.axvline(m.mean(), color="red", ls="--", lw=1, label=f"mean={m.mean():.2f}")
        ax.axvline(m.median(), color="orange", ls=":", lw=1, label=f"median={m.median():.2f}")
        ax.set_title(f"{SHORT[prod]} — Mid Price Distribution", fontsize=11)
        ax.set_xlabel("Mid Price")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Mid Price Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "02_midprice_distributions.png")
    else:
        plt.show()
    return fig


def plot_returns_distributions(prices=None, save=True):
    """Fig 03 — Returns (Δ mid) distribution per product per day."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            m = prices[(prices["product"] == prod) & (prices["day"] == day)]["mid_price"].dropna()
            r = m.diff().dropna()
            ax.hist(r, bins=60, density=True, alpha=0.7, color=COLORS[prod],
                    edgecolor="white", lw=0.3)
            mu, sig = r.mean(), r.std()
            xs = np.linspace(r.min(), r.max(), 300)
            ax.plot(xs, stats.norm.pdf(xs, mu, sig), color="black", lw=1.5, label="Normal fit")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d}  Returns", fontsize=9)
            ax.set_xlabel("Δ Mid Price")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Returns Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "03_returns_distributions.png")
    else:
        plt.show()
    return fig


def plot_spread_distributions(prices=None, save=True):
    """Fig 04 — Bid-ask spread distribution per product per day."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy()
            p["spread"] = p["ask_price_1"] - p["bid_price_1"]
            vs = p["spread"].dropna()
            vs = vs[vs > 0]
            if len(vs):
                ax.hist(vs, bins=40, color=COLORS[prod], alpha=0.75, edgecolor="white", lw=0.3)
                ax.axvline(vs.mean(), color="red", ls="--", lw=1.2,
                           label=f"mean={vs.mean():.2f}")
                ax.axvline(vs.median(), color="orange", ls=":", lw=1.2,
                           label=f"median={vs.median():.2f}")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Spread", fontsize=9)
            ax.set_xlabel("Spread (ticks)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Bid-Ask Spread Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "04_spread_distributions.png")
    else:
        plt.show()
    return fig


def plot_order_imbalance(prices=None, save=True):
    """Fig 05 — L1 order book imbalance over time."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy()
            mask = p["bid_volume_1"].notna() & p["ask_volume_1"].notna()
            sub = p[mask].copy()
            if len(sub):
                sub["imb"] = (sub["bid_volume_1"] - sub["ask_volume_1"]) / \
                             (sub["bid_volume_1"] + sub["ask_volume_1"])
                ax.plot(sub["timestamp"], sub["imb"], lw=0.5, alpha=0.7, color=COLORS[prod])
                ax.axhline(0, color="black", lw=0.8, ls="--")
                ax.axhline(sub["imb"].mean(), color="red", lw=0.8, ls=":",
                           label=f"mean={sub['imb'].mean():.3f}")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Order Imbalance", fontsize=9)
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("(BidVol − AskVol) / Total")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — L1 Order Book Imbalance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "05_order_imbalance.png")
    else:
        plt.show()
    return fig


def plot_pepper_detrended(prices=None, save=True):
    """Fig 06 — PEPPER price detrended per day with rolling mean of residuals."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    prod = "INTARIAN_PEPPER_ROOT"
    for ax, day in zip(axes, DAYS):
        p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy().reset_index(drop=True)
        m = p["mid_price"].dropna()
        x = p["timestamp"][m.index].values
        sl, ic, _, _, _ = stats.linregress(x, m.values)
        trend = sl * x + ic
        detrended = m.values - trend
        ax.plot(x, detrended, lw=0.6, alpha=0.8, color=COLORS[prod], label="Detrended mid")
        roll = pd.Series(detrended).rolling(200, center=True).mean()
        ax.plot(x, roll.values, lw=1.5, color="black", label="Rolling mean (w=200)")
        ax.axhline(0, color="red", ls="--", lw=0.8)
        ax.set_title(f"PEPPER  Day {day:+d} — Detrended  (slope={sl:.6f}/ts)", fontsize=10)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Residual Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — PEPPER Detrended Price", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "06_pepper_detrended.png")
    else:
        plt.show()
    return fig


def plot_ash_vs_fairvalue(prices=None, save=True):
    """Fig 07 — ASH mid price vs fair value = 10000 with rolling mean."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    prod = "ASH_COATED_OSMIUM"
    for ax, day in zip(axes, DAYS):
        p = prices[(prices["product"] == prod) & (prices["day"] == day)]
        ax.plot(p["timestamp"], p["mid_price"], lw=0.6, alpha=0.9,
                color=COLORS[prod], label="Mid price")
        ax.axhline(ASH_FAIR_VALUE, color="red", ls="--", lw=1.2,
                   label=f"Fair value={ASH_FAIR_VALUE:.0f}")
        roll = p["mid_price"].rolling(500, center=True).mean()
        ax.plot(p["timestamp"], roll, lw=1.2, color="black", alpha=0.7,
                label="Rolling mean (w=500)")
        ax.set_title(f"ASH  Day {day:+d}", fontsize=10)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Mid Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — ASH Mid Price vs Fair Value", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "07_ash_midprice_fairvalue.png")
    else:
        plt.show()
    return fig


def plot_return_autocorrelation(prices=None, max_lag=50, save=True):
    """Fig 08 — ACF of tick returns up to max_lag lags."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            m = prices[(prices["product"] == prod) & (prices["day"] == day)]["mid_price"].dropna()
            r = m.diff().dropna()
            lags = range(1, max_lag + 1)
            acf = [r.autocorr(lag=l) for l in lags]
            conf = 1.96 / np.sqrt(len(r))
            ax.bar(list(lags), acf, color=COLORS[prod], alpha=0.7)
            ax.axhline(conf,  color="red", ls="--", lw=0.8, label=f"±95% CI ({conf:.3f})")
            ax.axhline(-conf, color="red", ls="--", lw=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Return ACF", fontsize=9)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Return Autocorrelation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "08_return_autocorrelation.png")
    else:
        plt.show()
    return fig


def plot_trade_activity(prices=None, trades=None, save=True):
    """Fig 09 — Trades as bubble scatter on top of mid price."""
    if prices is None or trades is None:
        prices, trades, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            t = trades[(trades["product"] == prod) & (trades["day"] == day)]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)]
            if len(t):
                ax.scatter(t["timestamp"], t["price"],
                           s=np.clip(t["quantity"] * 2, 10, 120),
                           alpha=0.55, color=COLORS[prod], edgecolors="none")
            ax.plot(p["timestamp"], p["mid_price"], lw=0.5, color="black",
                    alpha=0.6, label="Mid price")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Trades", fontsize=9)
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Price")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Trade Activity (bubble size = quantity)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "09_trade_activity.png")
    else:
        plt.show()
    return fig


def plot_qq(prices=None, save=True):
    """Fig 10 — Q-Q plots of returns vs normal distribution."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            m = prices[(prices["product"] == prod) & (prices["day"] == day)]["mid_price"].dropna()
            r = m.diff().dropna()
            (osm, osr), (sl, ic, _) = stats.probplot(r, dist="norm")
            ax.scatter(osm, osr, s=3, alpha=0.5, color=COLORS[prod])
            ax.plot(osm, sl * np.array(osm) + ic, color="red", lw=1.5)
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — QQ", fontsize=9)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — QQ Plots of Returns (vs Normal)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "10_qq_plots.png")
    else:
        plt.show()
    return fig


def plot_rolling_volatility(prices=None, window=200, save=True):
    """Fig 11 — Rolling std of mid-price returns."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy()
            p["ret"] = p["mid_price"].diff()
            p["rvol"] = p["ret"].rolling(window).std()
            ax.plot(p["timestamp"], p["rvol"], lw=0.7, color=COLORS[prod])
            ax.axhline(p["ret"].std(), color="red", ls="--", lw=0.8,
                       label=f"Overall σ={p['ret'].std():.3f}")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Rolling Vol (w={window})", fontsize=9)
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Rolling Std")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Rolling Volatility", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "11_rolling_volatility.png")
    else:
        plt.show()
    return fig


def plot_pepper_full_trend(prices=None, save=True):
    """Fig 12 — PEPPER price across all days with overall linear trend."""
    if prices is None:
        prices, _, _ = load_data()
    fig, ax = plt.subplots(figsize=(16, 6))
    prod = "INTARIAN_PEPPER_ROOT"
    p = prices[prices["product"] == prod].copy()
    ax.plot(p["global_ts"], p["mid_price"], lw=0.5, color=COLORS[prod], alpha=0.8,
            label="Mid price")
    x_all = p["global_ts"].values
    sl, ic, _, _, _ = stats.linregress(x_all, p["mid_price"].values)
    ax.plot(x_all, sl * x_all + ic, color="black", lw=2, ls="--",
            label=f"Trend (slope={sl:.6f}/ts)")
    for day in DAYS:
        gts = p[p["day"] == day]["global_ts"]
        ax.axvline(gts.min(), color=DAY_COLORS[day], ls=":", lw=1,
                   label=f"Day {day:+d} start")
    ax.set_title("PEPPER — Full Trend Across All Days", fontsize=11)
    ax.set_xlabel("Global Timestamp")
    ax.set_ylabel("Mid Price")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save(fig, "12_pepper_full_trend.png")
    else:
        plt.show()
    return fig


def plot_vwap(trades=None, save=True):
    """Fig 13 — Volume-weighted price distribution per product."""
    if trades is None:
        _, trades, _ = load_data()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, prod in zip(axes, PRODUCTS):
        t = trades[trades["product"] == prod]
        if t.empty:
            ax.set_title(f"{SHORT[prod]} — No trades")
            continue
        vwap = (t["price"] * t["quantity"]).sum() / t["quantity"].sum()
        ax.hist(t["price"], bins=50, weights=t["quantity"], density=True,
                color=COLORS[prod], alpha=0.7, edgecolor="white", lw=0.3,
                label="Vol-weighted")
        ax.axvline(vwap, color="red", lw=1.5, ls="--", label=f"VWAP={vwap:.2f}")
        ax.set_title(f"{SHORT[prod]} — VWAP Analysis", fontsize=10)
        ax.set_xlabel("Trade Price")
        ax.set_ylabel("Vol-Weighted Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Trade VWAP Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "13_vwap_analysis.png")
    else:
        plt.show()
    return fig


def plot_orderbook_depth(prices=None, save=True):
    """Fig 14 — Bid/ask levels 1-3 over time for each product."""
    if prices is None:
        prices, _, _ = load_data()
    for prod in PRODUCTS:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        for ax, day in zip(axes, DAYS):
            p = prices[(prices["product"] == prod) & (prices["day"] == day)]
            ax.plot(p["timestamp"], p["mid_price"], lw=0.7, color="black",
                    label="Mid", zorder=5)
            bid_colors = ["#1565C0", "#1E88E5", "#64B5F6"]
            ask_colors = ["#B71C1C", "#E53935", "#EF9A9A"]
            for lvl, (bc, ac) in enumerate(zip(bid_colors, ask_colors), 1):
                bp_col, ap_col = f"bid_price_{lvl}", f"ask_price_{lvl}"
                if bp_col in p.columns:
                    ax.plot(p["timestamp"], p[bp_col], lw=0.4, alpha=0.6,
                            color=bc, label=f"Bid L{lvl}")
                if ap_col in p.columns:
                    ax.plot(p["timestamp"], p[ap_col], lw=0.4, alpha=0.6,
                            color=ac, label=f"Ask L{lvl}")
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Order Book Levels", fontsize=9)
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Price")
            ax.legend(fontsize=6, ncol=4)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Round 2 — {prod} Order Book Depth", fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save:
            _save(fig, f"14_orderbook_depth_{SHORT[prod]}.png")
        else:
            plt.show()


def plot_imbalance_predictive(prices=None, save=True):
    """Fig 15 — Order imbalance vs next-tick return (predictive power check)."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy()
            mask = p["bid_volume_1"].notna() & p["ask_volume_1"].notna()
            sub = p[mask].copy().reset_index(drop=True)
            if len(sub) < 50:
                continue
            sub["imb"] = (sub["bid_volume_1"] - sub["ask_volume_1"]) / \
                         (sub["bid_volume_1"] + sub["ask_volume_1"])
            sub["next_ret"] = sub["mid_price"].shift(-1) - sub["mid_price"]
            sub = sub.dropna(subset=["imb", "next_ret"])
            if len(sub) < 20:
                continue
            sub["imb_bin"] = pd.cut(sub["imb"], bins=10)
            bin_means = sub.groupby("imb_bin")["next_ret"].mean()
            ax.bar(range(len(bin_means)), bin_means.values, color=COLORS[prod], alpha=0.7)
            ax.axhline(0, color="black", lw=0.8)
            sl, _, r, pv, _ = stats.linregress(sub["imb"], sub["next_ret"])
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — Imbalance vs Next Return\n"
                         f"r={r:.3f}  p={pv:.2e}", fontsize=8)
            ax.set_xlabel("Imbalance Bin")
            ax.set_ylabel("Mean Next Return")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — Order Imbalance Predictive Power", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "15_imbalance_predictive.png")
    else:
        plt.show()
    return fig


def plot_ash_deviation(prices=None, save=True):
    """Fig 16 — ASH deviation from fair value = 10000 per day."""
    if prices is None:
        prices, _, _ = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    prod = "ASH_COATED_OSMIUM"
    for ax, day in zip(axes, DAYS):
        m = prices[(prices["product"] == prod) & (prices["day"] == day)]["mid_price"].dropna()
        dev = m - ASH_FAIR_VALUE
        ax.hist(dev, bins=60, density=True, color=COLORS[prod], alpha=0.7,
                edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", lw=1.5, ls="--", label="Zero deviation")
        ax.axvline(dev.mean(), color="orange", lw=1.2, ls=":",
                   label=f"mean={dev.mean():.2f}")
        ax.set_title(f"ASH  Day {day:+d} — Deviation from {ASH_FAIR_VALUE:.0f}", fontsize=10)
        ax.set_xlabel(f"Mid Price − {ASH_FAIR_VALUE:.0f}")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Round 2 — ASH Deviation from Fair Value", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        _save(fig, "16_ash_deviation_fairvalue.png")
    else:
        plt.show()
    return fig


# ── Plot A — Inefficiency Map ─────────────────────────────────────────────────

def plot_inefficiency_map(merged=None, prices=None, save=True):
    """
    Fig A — Trade price deviation from mid-price (the Inefficiency Map).

    X-axis: timestamp
    Y-axis: trade_price − mid_price at the time of trade
    Bubble size: trade quantity
    Color gradient: red = above mid, green = below mid
    Dashed lines: ±half bid-ask spread reference

    What to look for
    ─────────────────
    - Large clusters far from zero: bots crossing the spread aggressively.
      If frequent, the MAF extra-25% volume is very valuable — you can capture
      those aggressive orders at better prices.
    - Asymmetry above/below zero: persistent buy or sell pressure in the market.
    - Day-over-day widening: increasing aggression or regime shift worth modeling.
    - % annotation: fraction of trades that cross more than half the spread —
      directly sizes the opportunity for a passive market-maker.
    """
    if merged is None or prices is None:
        prices, _, merged = load_data()

    if merged.empty:
        print("No merged trade data available.")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Round 2 — Inefficiency Map: Trade Price Deviation from Mid\n"
        "(clusters far from zero = aggressive spread-crossing; "
        "bubble size = trade quantity)",
        fontsize=12, y=0.99,
    )

    gs = gridspec.GridSpec(len(PRODUCTS), len(DAYS), figure=fig,
                           hspace=0.48, wspace=0.28)

    global_abs_max = merged["deviation"].abs().quantile(0.995)

    for row, prod in enumerate(PRODUCTS):
        color = COLORS[prod]
        short = SHORT[prod]

        for col, day in enumerate(DAYS):
            ax = fig.add_subplot(gs[row, col])
            sub = merged[(merged["product"] == prod) & (merged["day"] == day)]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Half-spread from prices
            p_day = prices[(prices["product"] == prod) & (prices["day"] == day)]
            spread_med = (p_day["ask_price_1"] - p_day["bid_price_1"]).median()
            half_spread = spread_med / 2 if not np.isnan(spread_med) else None

            # Scatter
            sizes = np.clip(sub["quantity"] * 3, 10, 120)
            sc = ax.scatter(
                sub["timestamp"], sub["deviation"],
                c=sub["deviation"],
                cmap="RdYlGn_r",
                vmin=-global_abs_max, vmax=global_abs_max,
                s=sizes, alpha=0.65, edgecolors="none",
            )

            # Reference lines
            ax.axhline(0, color="black", lw=0.9, zorder=0)
            if half_spread is not None:
                ax.axhline(+half_spread, color=color, lw=0.9, ls="--", alpha=0.75,
                           label=f"+½spread ({half_spread:.0f})")
                ax.axhline(-half_spread, color=color, lw=0.9, ls="--", alpha=0.75,
                           label=f"−½spread ({-half_spread:.0f})")

            ax.set_xlim(sub["timestamp"].min(), sub["timestamp"].max())
            ax.set_ylim(-global_abs_max * 1.1, global_abs_max * 1.1)
            ax.set_title(f"{short}  ·  Day {day:+d}", fontsize=10)
            ax.set_xlabel("Timestamp", fontsize=8)
            if col == 0:
                ax.set_ylabel("Trade price − mid", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

            # % trades crossing ±half-spread
            if half_spread is not None:
                pct = (sub["deviation"].abs() > half_spread).mean() * 100
                ax.annotate(
                    f"{pct:.0f}% cross ½-spread",
                    xy=(0.03, 0.93), xycoords="axes fraction",
                    fontsize=7.5, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0),
                )

            # Colorbar on rightmost column
            if col == len(DAYS) - 1:
                plt.colorbar(sc, ax=ax, label="deviation (ticks)", pad=0.02)
            if row == 0 or col == 0:
                ax.legend(fontsize=6.5, loc="lower right")

    if save:
        _save(fig, "A_inefficiency_map.png")
    else:
        plt.show()
    return fig


def plot_spread_vs_trades(prices=None, trades=None, save=True):
    """
    Fig B — Bid-ask spread over time (shaded area) with trade dots overlaid.

    What to look for
    ─────────────────
    - Trades clustering during wide-spread periods: bots crossing aggressively —
      passive MM opportunity; post wide-spread trades with limit orders.
    - Trades clustering during narrow-spread periods: tight-market activity,
      less edge available per trade.
    - Spread spikes: low-liquidity moments; potential for wider quotes.
    """
    if prices is None or trades is None:
        prices, trades, _ = load_data()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Round 2 — Spread Over Time vs. Trade Frequency\n"
        "(shaded = bid-ask spread; dots = trade price, size = quantity)",
        fontsize=12, y=1.01,
    )

    for row, prod in enumerate(PRODUCTS):
        for col, day in enumerate(DAYS):
            ax = axes[row][col]
            p = prices[(prices["product"] == prod) & (prices["day"] == day)].copy()
            t = trades[(trades["product"] == prod) & (trades["day"] == day)].copy()

            p = p.dropna(subset=["bid_price_1", "ask_price_1"])
            p["spread"] = p["ask_price_1"] - p["bid_price_1"]
            p = p[p["spread"] > 0]

            if p.empty:
                ax.set_visible(False)
                continue

            ax.fill_between(
                p["timestamp"],
                p["bid_price_1"],
                p["ask_price_1"],
                alpha=0.25,
                color=COLORS[prod],
                label="Bid-ask spread",
            )
            ax.plot(p["timestamp"], p["mid_price"], lw=0.6, color="black", alpha=0.5, label="Mid")

            if not t.empty:
                sizes = np.clip(t["quantity"] * 3, 8, 100)
                p_ts = p.set_index("timestamp")[["bid_price_1", "ask_price_1"]]
                t_aligned = pd.merge_asof(
                    t.sort_values("timestamp"),
                    p_ts.reset_index(),
                    on="timestamp",
                    direction="backward",
                )
                inside = (t_aligned["price"] >= t_aligned["bid_price_1"]) & \
                         (t_aligned["price"] <= t_aligned["ask_price_1"])
                ax.scatter(
                    t_aligned.loc[inside, "timestamp"],
                    t_aligned.loc[inside, "price"],
                    s=sizes[inside.values],
                    color="#2e7d32", alpha=0.7, edgecolors="none",
                    zorder=3, label="Trade (inside spread)",
                )
                ax.scatter(
                    t_aligned.loc[~inside, "timestamp"],
                    t_aligned.loc[~inside, "price"],
                    s=sizes[~inside.values],
                    color="#c62828", alpha=0.8, edgecolors="none",
                    zorder=4, label="Trade (outside spread)",
                )
                pct_outside = (~inside).mean() * 100
                ax.annotate(
                    f"{pct_outside:.0f}% outside spread",
                    xy=(0.03, 0.95), xycoords="axes fraction",
                    fontsize=7.5, color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0),
                )

            med_spread = p["spread"].median()
            ax.set_title(f"{SHORT[prod]}  Day {day:+d} — median spread={med_spread:.1f}", fontsize=9)
            ax.set_xlabel("Timestamp", fontsize=8)
            if col == 0:
                ax.set_ylabel("Price", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save:
        _save(fig, "B_spread_vs_trades.png")
    else:
        plt.show()
    return fig


# ── run_all ───────────────────────────────────────────────────────────────────

def run_all():
    print("Loading Round 2 data...")
    prices, trades, merged = load_data()
    print(f"  prices={len(prices):,}  trades={len(trades):,}  merged={len(merged):,}")
    print()

    print_stats(prices, trades)

    print("\nGenerating plots...")
    plot_midprice_timeseries(prices)
    plot_midprice_distributions(prices)
    plot_returns_distributions(prices)
    plot_spread_distributions(prices)
    plot_order_imbalance(prices)
    plot_pepper_detrended(prices)
    plot_ash_vs_fairvalue(prices)
    plot_return_autocorrelation(prices)
    plot_trade_activity(prices, trades)
    plot_qq(prices)
    plot_rolling_volatility(prices)
    plot_pepper_full_trend(prices)
    plot_vwap(trades)
    plot_orderbook_depth(prices)
    plot_imbalance_predictive(prices)
    plot_ash_deviation(prices)
    plot_inefficiency_map(merged, prices)
    plot_spread_vs_trades(prices, trades)

    print(f"\nAll output → {OUT_DIR}")


if __name__ == "__main__":
    import datetime, io

    log_path = os.path.join(OUT_DIR, "eda_summary.log")

    class _Tee(io.TextIOBase):
        def __init__(self, *streams):
            self._streams = streams
        def write(self, s):
            for st in self._streams:
                st.write(s)
            return len(s)
        def flush(self):
            for st in self._streams:
                st.flush()

    with open(log_path, "w") as lf:
        lf.write(f"# EDA run: {datetime.datetime.now().isoformat()}\n\n")
        sys.stdout = _Tee(sys.__stdout__, lf)
        try:
            run_all()
        finally:
            sys.stdout = sys.__stdout__
    print(f"Log saved → {log_path}")
