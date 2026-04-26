"""
Offline analysis for the Phase 1 hold-1-unit ASH_COATED_OSMIUM probe.

This file is intentionally NOT submission code.

Usage:
  python3 analyze_hold_1_osmium.py /path/to/submission.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import io
import json


PRODUCT = "ASH_COATED_OSMIUM"


def _available_asks(row) -> List[float]:
    asks: List[float] = []
    for col in ("ask_price_1", "ask_price_2", "ask_price_3"):
        value = row.get(col)
        if value == value:
            asks.append(float(value))
    return asks


def _infer_buy_price(osmium_df) -> tuple[int, float]:
    entry_timestamp = int(osmium_df["timestamp"].min())
    entry_rows = osmium_df[osmium_df["timestamp"] == entry_timestamp]
    if entry_rows.empty:
        raise ValueError("Could not find an entry timestamp for osmium rows.")

    asks = _available_asks(entry_rows.iloc[0])
    if not asks:
        raise ValueError("No visible ask found on the entry timestamp; cannot infer buy price.")

    buy_price = min(asks)
    return entry_timestamp, float(buy_price)


def analyze_submission_log(log_file: Path) -> None:
    import os
    import tempfile
    import numpy as np
    import pandas as pd

    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with log_file.open("r") as f:
        data = json.load(f)

    csv_data = data["activitiesLog"]
    df = pd.read_csv(io.StringIO(csv_data), sep=";")

    osmium = df[df["product"] == PRODUCT].copy()
    osmium.sort_values(by="timestamp", inplace=True)
    osmium.reset_index(drop=True, inplace=True)
    if osmium.empty:
        raise ValueError(f"No {PRODUCT} rows found in activitiesLog.")

    entry_timestamp, buy_price = _infer_buy_price(osmium)

    osmium["true_fv"] = np.where(
        osmium["timestamp"] > entry_timestamp,
        osmium["profit_and_loss"] + buy_price,
        np.nan,
    )

    calibrated = osmium.dropna(subset=["true_fv"]).copy()
    if calibrated.empty:
        raise ValueError("No post-entry ticks available to reconstruct true_fv.")

    fv = calibrated["true_fv"].to_numpy(dtype=float)
    returns = np.diff(fv)

    print("=== HOLD-1 OSMIUM PHASE 1 ANALYSIS ===")
    print(f"log_file: {log_file}")
    print(f"entry_timestamp: {entry_timestamp}")
    print(f"inferred_buy_price: {buy_price:.4f}")
    print(f"n_post_entry_ticks: {len(calibrated)}")
    print()
    print("=== TRUE FV / SERVER MARK STATISTICS ===")
    print(f"mean: {np.mean(fv):.6f}")
    print(f"std:  {np.std(fv):.6f}")
    print(f"min:  {np.min(fv):.6f}")
    print(f"max:  {np.max(fv):.6f}")
    if len(fv) >= 2:
        print(f"lag1_level_autocorr: {np.corrcoef(fv[:-1], fv[1:])[0, 1]:.6f}")
    if len(fv) >= 11:
        print(f"lag10_level_autocorr: {np.corrcoef(fv[:-10], fv[10:])[0, 1]:.6f}")
    if len(returns) >= 2:
        print(f"return_mean: {np.mean(returns):.6f}")
        print(f"return_std:  {np.std(returns):.6f}")
        print(f"lag1_return_autocorr: {np.corrcoef(returns[:-1], returns[1:])[0, 1]:.6f}")

    output_dir = log_file.parent
    plot_path = output_dir / f"{log_file.stem}_osmium_true_fv.png"

    plt.figure(figsize=(12, 7))
    plt.plot(calibrated["timestamp"], calibrated["true_fv"], label="Implied Server Mark / FV", color="blue", alpha=0.8)
    plt.plot(calibrated["timestamp"], calibrated["mid_price"], label="Visible Mid Price", color="red", alpha=0.55, linestyle="--")
    plt.axhline(10000.0, color="black", linestyle=":", label="Reference Level 10000")
    plt.title("ASH_COATED_OSMIUM Hold-1 Probe: Implied Server Mark vs Visible Mid")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print()
    print(f"plot_saved_to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a downloaded hold-1-unit osmium submission log."
    )
    parser.add_argument(
        "log_file",
        help="Path to the downloaded submission JSON/log file.",
    )
    args = parser.parse_args()

    analyze_submission_log(Path(args.log_file).expanduser().resolve())


if __name__ == "__main__":
    main()
