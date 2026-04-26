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
        description="Analyze sweep-and-quote behavior from a submission artifact."
    )
    parser.add_argument("log_file", help="Path to submission .log or .json")
    parser.add_argument("--product", default="ASH_COATED_OSMIUM")
    parser.add_argument(
        "--plot-out",
        default="imc-prosperity-4/vedant/round_2_analysis/sweep_mean_analysis.png",
        help="Output path for mid-price plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plot generation (useful in headless/CI runs).",
    )
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
    mean_mid = float(valid_mid.mean())
    std_mid = float(valid_mid.std(ddof=0))
    min_mid = float(valid_mid.min())
    max_mid = float(valid_mid.max())

    print("=== ACTIVITY SUMMARY ===")
    print(f"log_file: {log.path}")
    print(f"product: {product}")
    print(f"n_ticks: {len(sub)}")
    print(f"n_mid_all: {len(mid_all)}")
    print(f"n_mid_valid_gt_0: {len(valid_mid)}")
    print(f"mid_zero_or_missing_rate: {(1 - len(valid_mid) / len(sub)):.4f}")
    print(f"mid_mean_valid: {mean_mid:.6f}")
    print(f"mid_std_valid: {std_mid:.6f}")
    print(f"mid_min_valid: {min_mid:.6f}")
    print(f"mid_max_valid: {max_mid:.6f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        out_path = Path(args.plot_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(sub["timestamp"], sub["mid_price"], label="Mid Price", color="blue", alpha=0.7)
        plt.axhline(mean_mid, color="red", linestyle="--", label=f"Mean ({mean_mid:.2f})")
        plt.ylim(mean_mid - 30, mean_mid + 30)
        plt.title(f"{product}: Sweep Mid Price Dynamics")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"plot_saved_to: {out_path}")
    else:
        print("plot_skipped: --no-plot enabled")

    trades = _product_trades(log.trades, product)
    print("\n=== TRADE SUMMARY ===")
    print(f"n_trades: {len(trades)}")
    if trades.empty:
        print("No tradeHistory rows parsed for this artifact.")
        return

    prices = trades["price"].astype(float)
    print(f"trade_price_min: {prices.min():.6f}")
    print(f"trade_price_max: {prices.max():.6f}")
    print(f"trade_price_mean: {prices.mean():.6f}")
    print("top_trade_prices:")
    vc = prices.value_counts().head(20)
    for px, n in vc.items():
        print(f"  price={px:.3f} count={int(n)}")


if __name__ == "__main__":
    main()
