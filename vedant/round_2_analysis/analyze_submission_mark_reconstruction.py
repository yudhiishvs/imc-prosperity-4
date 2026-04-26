"""
Reconstruct hidden mark from submission trade history + per-product PnL.

This works for logs where the submission has nonzero inventory in the target
product and tradeHistory includes SUBMISSION-side fills.

Usage:
  python3 analyze_submission_mark_reconstruction.py /path/to/submission.log --product INTARIAN_PEPPER_ROOT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from submission_log_utils import load_submission_log


def _submission_trades_for_product(trades: pd.DataFrame, product: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    symbol_col = "symbol" if "symbol" in trades.columns else "product" if "product" in trades.columns else None
    if symbol_col is None:
        return pd.DataFrame()
    sub = trades[trades[symbol_col] == product].copy()
    if sub.empty:
        return sub
    sub = sub[(sub["buyer"] == "SUBMISSION") | (sub["seller"] == "SUBMISSION")].copy()
    return sub.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct hidden mark from submission inventory accounting."
    )
    parser.add_argument("log_file", help="Path to submission .log or .json")
    parser.add_argument("--product", default="INTARIAN_PEPPER_ROOT")
    args = parser.parse_args()

    log = load_submission_log(Path(args.log_file).expanduser().resolve())
    if log.activities.empty:
        raise ValueError("activitiesLog is empty.")

    product = args.product
    a = log.activities[log.activities["product"] == product].copy().sort_values("timestamp").reset_index(drop=True)
    if a.empty:
        raise ValueError(f"No rows found for {product} in activitiesLog.")

    t = _submission_trades_for_product(log.trades, product)
    if t.empty:
        raise ValueError(f"No SUBMISSION tradeHistory rows found for {product}.")

    pos = 0
    cash = 0.0
    by_ts: dict[int, tuple[int, float]] = {}
    for ts, grp in t.groupby("timestamp"):
        for _, row in grp.iterrows():
            q = int(row["quantity"])
            px = float(row["price"])
            if row["buyer"] == "SUBMISSION":
                pos += q
                cash -= q * px
            elif row["seller"] == "SUBMISSION":
                pos -= q
                cash += q * px
        by_ts[int(ts)] = (pos, cash)

    pos = 0
    cash = 0.0
    trade_ts = set(by_ts.keys())
    pos_series = []
    cash_series = []
    traded_series = []
    for ts in a["timestamp"].astype(int):
        if ts in by_ts:
            pos, cash = by_ts[ts]
        pos_series.append(pos)
        cash_series.append(cash)
        traded_series.append(ts in trade_ts)

    a["recon_pos"] = pos_series
    a["recon_cash"] = cash_series
    a["had_submission_trade"] = traded_series

    nz = a[a["recon_pos"] != 0].copy()
    if nz.empty:
        raise ValueError("No nonzero position ticks; cannot reconstruct mark.")

    nz["mark"] = (nz["profit_and_loss"] - nz["recon_cash"]) / nz["recon_pos"]
    nz["best_bid"] = nz[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1, skipna=True)
    nz["best_ask"] = nz[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1, skipna=True)
    nz["mid"] = (nz["best_bid"] + nz["best_ask"]) / 2.0

    mark = nz["mark"].astype(float)
    dmark = mark.diff().dropna()

    nz["prev_pos"] = nz["recon_pos"].shift(1)
    stable = nz[(~nz["had_submission_trade"]) & (nz["prev_pos"] == nz["recon_pos"])].copy()
    stable_mark = stable["mark"].astype(float) if not stable.empty else pd.Series(dtype=float)
    stable_dmark = stable_mark.diff().dropna() if len(stable_mark) >= 2 else pd.Series(dtype=float)

    valid_mid = nz.dropna(subset=["mid"]).copy()
    mae = (valid_mid["mark"] - valid_mid["mid"]).abs()
    inside = (valid_mid["mark"] >= valid_mid["best_bid"]) & (valid_mid["mark"] <= valid_mid["best_ask"])

    print("=== SUBMISSION MARK RECONSTRUCTION ===")
    print(f"log_file: {log.path}")
    print(f"product: {product}")
    print(f"submission_trade_rows: {len(t)}")
    print(f"ticks_total: {len(a)}")
    print(f"ticks_nonzero_position: {len(nz)}")
    print(f"max_abs_position: {int(nz['recon_pos'].abs().max())}")
    print(f"mark_mean: {float(mark.mean()):.6f}")
    print(f"mark_std: {float(mark.std(ddof=0)):.6f}")
    print(f"mark_min: {float(mark.min()):.6f}")
    print(f"mark_max: {float(mark.max()):.6f}")
    print(f"lag1_mark_autocorr: {float(mark.autocorr()):.6f}")
    print(f"dmark_mean: {float(dmark.mean()):.6f}")
    print(f"dmark_std: {float(dmark.std(ddof=0)):.6f}")
    print(f"mark_on_1_over_1024_grid: {bool(np.allclose(mark, np.round(mark * 1024) / 1024))}")
    print(f"corr_mark_mid: {float(valid_mid['mark'].corr(valid_mid['mid'])):.6f}")
    print(f"mean_abs_mark_minus_mid: {float(mae.mean()):.6f}")
    print(f"inside_best_spread_rate: {float(inside.mean()):.6f}")
    print(f"stable_ticks: {len(stable)}")
    if len(stable_mark) >= 2:
        stable_slope = float(np.polyfit(stable["timestamp"].astype(float), stable_mark.to_numpy(dtype=float), deg=1)[0] * 100.0)
        print(f"stable_slope_per_tick: {stable_slope:.6f}")
    if len(stable_dmark) > 0:
        print(f"stable_dmark_mean: {float(stable_dmark.mean()):.6f}")
        print(f"stable_dmark_std: {float(stable_dmark.std(ddof=0)):.6f}")


if __name__ == "__main__":
    main()
