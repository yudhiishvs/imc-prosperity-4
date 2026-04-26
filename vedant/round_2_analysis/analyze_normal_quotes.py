from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from submission_log_utils import load_submission_log


def _product_trades(trades: pd.DataFrame, product: str) -> pd.DataFrame:
    if trades.empty:
        return trades
    if "symbol" in trades.columns:
        return trades[trades["symbol"] == product].copy()
    if "product" in trades.columns:
        return trades[trades["product"] == product].copy()
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a normal-quote submission log for Round 2 products."
    )
    parser.add_argument("log_file", help="Path to submission .log or .json")
    parser.add_argument("--product", default="ASH_COATED_OSMIUM")
    args = parser.parse_args()

    log = load_submission_log(Path(args.log_file).expanduser().resolve())
    activities = log.activities

    if activities.empty:
        raise ValueError("activitiesLog is empty or could not be parsed.")

    product = args.product
    sub = activities[activities["product"] == product].copy()
    if sub.empty:
        raise ValueError(f"No rows found for {product}.")
    sub = sub.sort_values("timestamp").reset_index(drop=True)

    mid_all = sub["mid_price"].dropna()
    valid_mid = mid_all[mid_all > 0]
    print("=== ACTIVITY SUMMARY ===")
    print(f"log_file: {log.path}")
    print(f"product: {product}")
    print(f"n_ticks: {len(sub)}")
    print(f"n_mid_all: {len(mid_all)}")
    print(f"n_mid_valid_gt_0: {len(valid_mid)}")
    print(f"mid_zero_or_missing_rate: {(1 - len(valid_mid) / len(sub)):.4f}")
    if len(valid_mid) > 0:
        print(f"mid_mean_valid: {valid_mid.mean():.6f}")
        print(f"mid_std_valid: {valid_mid.std(ddof=0):.6f}")
        print(f"mid_min_valid: {valid_mid.min():.6f}")
        print(f"mid_max_valid: {valid_mid.max():.6f}")

    bids = sub[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1, skipna=True)
    asks = sub[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1, skipna=True)
    spread = asks - bids
    print("\n=== BOOK SUMMARY ===")
    print(f"best_spread_mean: {spread.mean():.6f}")
    print(f"best_spread_median: {spread.median():.6f}")
    print(f"best_spread_min: {spread.min():.6f}")
    print(f"best_spread_max: {spread.max():.6f}")

    trades = _product_trades(log.trades, product)
    print("\n=== TRADE SUMMARY ===")
    print(f"n_trades: {len(trades)}")
    if not trades.empty:
        prices = trades["price"].astype(float)
        print(f"trade_price_min: {prices.min():.6f}")
        print(f"trade_price_max: {prices.max():.6f}")
        print(f"trade_price_mean: {prices.mean():.6f}")
        print("top_trade_prices:")
        vc = prices.value_counts().head(15)
        for px, n in vc.items():
            print(f"  price={px:.3f} count={int(n)}")
    else:
        print("No tradeHistory rows parsed for this artifact.")


if __name__ == "__main__":
    main()
