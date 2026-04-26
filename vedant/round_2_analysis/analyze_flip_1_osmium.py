"""
Offline analysis for flip-1 Osmium probe logs.

Usage:
  python3 analyze_flip_1_osmium.py /path/to/submission.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from submission_log_utils import load_submission_log


PRODUCT = "ASH_COATED_OSMIUM"


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
    parser = argparse.ArgumentParser(description="Analyze a flip-1 Osmium probe log.")
    parser.add_argument("log_file")
    args = parser.parse_args()

    log = load_submission_log(Path(args.log_file).expanduser().resolve())
    if log.activities.empty:
        raise ValueError("activitiesLog is empty.")

    a = log.activities[log.activities["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    if a.empty:
        raise ValueError(f"No {PRODUCT} rows in activitiesLog.")

    trades = _submission_trades_for_product(log.trades, PRODUCT)
    if trades.empty:
        raise ValueError("No submission tradeHistory rows found for Osmium.")

    buys = trades[trades["buyer"] == "SUBMISSION"].copy()
    sells = trades[trades["seller"] == "SUBMISSION"].copy()
    if buys.empty or sells.empty:
        raise ValueError("Could not find both buy and sell legs in tradeHistory.")

    entry_trade = buys.iloc[0]
    exit_trade = sells.iloc[-1]
    entry_ts = int(entry_trade["timestamp"])
    exit_ts = int(exit_trade["timestamp"])
    buy_price = float(entry_trade["price"])

    a["server_mark"] = np.where(
        (a["timestamp"] > entry_ts) & (a["timestamp"] <= exit_ts),
        a["profit_and_loss"] + buy_price,
        np.nan,
    )
    hold = a.dropna(subset=["server_mark"]).copy()
    if hold.empty:
        raise ValueError("No hold-period rows available for mark reconstruction.")

    hold_mark = hold["server_mark"].to_numpy(dtype=float)
    hold_dmark = np.diff(hold_mark)

    flat = a[a["timestamp"] > exit_ts].copy()
    flat_pnl_std = float(flat["profit_and_loss"].std(ddof=0)) if not flat.empty else float("nan")
    flat_pnl_range = (
        float(flat["profit_and_loss"].max() - flat["profit_and_loss"].min())
        if not flat.empty
        else float("nan")
    )

    print("=== FLIP-1 OSMIUM ANALYSIS ===")
    print(f"log_file: {log.path}")
    print(f"entry_ts: {entry_ts} buy_price: {buy_price:.6f}")
    print(f"exit_ts: {exit_ts} sell_price: {float(exit_trade['price']):.6f}")
    print(f"n_hold_ticks: {len(hold)}")
    print(f"hold_mark_mean: {hold_mark.mean():.6f}")
    print(f"hold_mark_std: {hold_mark.std():.6f}")
    print(f"hold_dmark_mean: {hold_dmark.mean():.6f}")
    print(f"hold_dmark_std: {hold_dmark.std():.6f}")
    print(f"hold_mark_on_1_over_1024_grid: {bool(np.allclose(hold_mark, np.round(hold_mark * 1024) / 1024))}")
    print(f"n_flat_ticks: {len(flat)}")
    print(f"flat_pnl_std: {flat_pnl_std:.6f}")
    print(f"flat_pnl_range: {flat_pnl_range:.6f}")


if __name__ == "__main__":
    main()

