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


def _is_submission_fill(row: pd.Series) -> bool:
    return row.get("buyer") == "SUBMISSION" or row.get("seller") == "SUBMISSION"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze snipe and refill behavior from a submission artifact."
    )
    parser.add_argument("log_file", help="Path to submission .log or .json")
    parser.add_argument("--product", default="ASH_COATED_OSMIUM")
    args = parser.parse_args()

    log = load_submission_log(Path(args.log_file).expanduser().resolve())
    activities = log.activities
    if activities.empty:
        raise ValueError("activitiesLog is empty or could not be parsed.")

    product = args.product
    book = activities[activities["product"] == product].copy()
    if book.empty:
        raise ValueError(f"No rows found for {product}.")
    book = book.sort_values("timestamp").reset_index(drop=True)
    by_ts = book.set_index("timestamp")

    trades = _product_trades(log.trades, product)
    if trades.empty:
        print("No product trades found in tradeHistory.")
        return

    submission = trades[trades.apply(_is_submission_fill, axis=1)].copy()
    if submission.empty:
        print("No submission fills found for this product.")
        return

    submission["timestamp"] = submission["timestamp"].astype(int)
    submission["price"] = submission["price"].astype(float)
    submission["quantity"] = submission["quantity"].astype(int)

    total_fills = len(submission)
    snipe_like = submission[submission["quantity"].abs() > 1].copy()
    penny_like = submission[submission["quantity"].abs() == 1].copy()

    refill_hits = 0
    refill_trials = 0

    for _, fill in snipe_like.iterrows():
        ts = int(fill["timestamp"])
        next_rows = book[book["timestamp"] > ts].head(1)
        if next_rows.empty:
            continue
        nxt = next_rows.iloc[0]
        price = float(fill["price"])
        qty = abs(int(fill["quantity"]))
        refill_trials += 1

        if fill.get("seller") == "SUBMISSION":
            # We sold into bid, check if best bid at same price comes back with depth.
            if pd.notna(nxt["bid_price_1"]) and float(nxt["bid_price_1"]) == price and float(nxt["bid_volume_1"]) > qty:
                refill_hits += 1
        elif fill.get("buyer") == "SUBMISSION":
            # We bought from ask, check if best ask at same price comes back with depth.
            ask_vol = abs(float(nxt["ask_volume_1"])) if pd.notna(nxt["ask_volume_1"]) else 0.0
            if pd.notna(nxt["ask_price_1"]) and float(nxt["ask_price_1"]) == price and ask_vol > qty:
                refill_hits += 1

    print("=== SNIPE SUMMARY ===")
    print(f"log_file: {log.path}")
    print(f"product: {product}")
    print(f"total_submission_fills: {total_fills}")
    print(f"snipe_like_fills_abs_qty_gt_1: {len(snipe_like)}")
    print(f"penny_like_fills_abs_qty_eq_1: {len(penny_like)}")

    print("\n=== REFILL CHECK ===")
    print(f"refill_trials: {refill_trials}")
    print(f"refill_hits_same_price_next_tick: {refill_hits}")
    if refill_trials > 0:
        print(f"refill_rate: {refill_hits / refill_trials:.2%}")
    else:
        print("refill_rate: N/A")

    # Classify fills as aggressive vs passive against visible L1 at fill timestamp.
    aggressive = 0
    passive = 0
    unknown = 0
    for _, fill in submission.iterrows():
        ts = int(fill["timestamp"])
        if ts not in by_ts.index:
            unknown += 1
            continue
        row = by_ts.loc[ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        price = float(fill["price"])

        bid = float(row["bid_price_1"]) if pd.notna(row["bid_price_1"]) else None
        ask = float(row["ask_price_1"]) if pd.notna(row["ask_price_1"]) else None

        if fill.get("buyer") == "SUBMISSION":
            if ask is not None and price >= ask:
                aggressive += 1
            else:
                passive += 1
        elif fill.get("seller") == "SUBMISSION":
            if bid is not None and price <= bid:
                aggressive += 1
            else:
                passive += 1
        else:
            unknown += 1

    print("\n=== FILL TYPE VS VISIBLE L1 ===")
    print(f"aggressive_like: {aggressive}")
    print(f"passive_like: {passive}")
    print(f"unknown: {unknown}")


if __name__ == "__main__":
    main()
