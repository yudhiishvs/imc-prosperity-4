"""
Offline analysis for hold-1 Pepper probe logs.

Usage:
  python3 analyze_hold_1_pepper.py /path/to/submission.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from submission_log_utils import load_submission_log


PRODUCT = "INTARIAN_PEPPER_ROOT"


def _infer_buy_price(df):
    entry_ts = int(df["timestamp"].min())
    row = df[df["timestamp"] == entry_ts].iloc[0]
    asks = [float(row[c]) for c in ("ask_price_1", "ask_price_2", "ask_price_3") if row[c] == row[c]]
    if not asks:
        raise ValueError("No ask observed at entry tick.")
    return entry_ts, min(asks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a hold-1 Pepper submission log.")
    parser.add_argument("log_file")
    args = parser.parse_args()

    log = load_submission_log(Path(args.log_file).expanduser().resolve())
    if log.activities.empty:
        raise ValueError("activitiesLog is empty.")

    p = log.activities[log.activities["product"] == PRODUCT].copy().sort_values("timestamp").reset_index(drop=True)
    if p.empty:
        raise ValueError(f"No {PRODUCT} rows in activitiesLog.")

    entry_ts, buy_px = _infer_buy_price(p)
    p["server_mark"] = np.where(p["timestamp"] > entry_ts, p["profit_and_loss"] + buy_px, np.nan)
    cal = p.dropna(subset=["server_mark"]).copy()
    if cal.empty:
        raise ValueError("No post-entry rows available.")

    mark = cal["server_mark"].to_numpy(dtype=float)
    dmark = np.diff(mark)
    slope = np.polyfit(cal["timestamp"].to_numpy(dtype=float), mark, deg=1)[0] * 100.0

    print("=== HOLD-1 PEPPER ANALYSIS ===")
    print(f"log_file: {log.path}")
    print(f"entry_timestamp: {entry_ts}")
    print(f"inferred_buy_price: {buy_px:.6f}")
    print(f"n_post_entry_ticks: {len(cal)}")
    print(f"mark_mean: {mark.mean():.6f}")
    print(f"mark_std: {mark.std():.6f}")
    print(f"mark_min: {mark.min():.6f}")
    print(f"mark_max: {mark.max():.6f}")
    print(f"lag1_mark_autocorr: {np.corrcoef(mark[:-1], mark[1:])[0, 1]:.6f}")
    print(f"dmark_mean: {dmark.mean():.6f}")
    print(f"dmark_std: {dmark.std():.6f}")
    print(f"estimated_trend_per_tick: {slope:.6f}")
    print(f"mark_on_1_over_1024_grid: {bool(np.allclose(mark, np.round(mark * 1024) / 1024))}")


if __name__ == "__main__":
    main()

