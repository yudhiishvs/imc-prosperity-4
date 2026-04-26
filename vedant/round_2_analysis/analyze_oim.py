from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def _compute_oim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    b1 = out["bid_volume_1"].fillna(0)
    a1 = out["ask_volume_1"].fillna(0)
    b2 = out["bid_volume_2"].fillna(0)
    a2 = out["ask_volume_2"].fillna(0)
    b3 = out["bid_volume_3"].fillna(0)
    a3 = out["ask_volume_3"].fillna(0)

    out["l1_oim"] = (b1 - a1) / (b1 + a1).replace(0, np.nan)
    out["l2_oim"] = ((b1 + b2) - (a1 + a2)) / ((b1 + b2) + (a1 + a2)).replace(0, np.nan)
    out["l3_oim"] = ((b1 + b2 + b3) - (a1 + a2 + a3)) / ((b1 + b2 + b3) + (a1 + a2 + a3)).replace(0, np.nan)
    return out


def _print_metrics(sub: pd.DataFrame, product: str, day: int) -> None:
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    # Use the dataset mid_price directly so we keep the original tick sequence.
    # Rebuilding mid from partial book levels can introduce selection bias if we
    # drop one-sided rows and then correlate against non-adjacent future rows.
    sub["mid_price"] = sub["mid_price"].astype(float)
    sub = _compute_oim(sub)

    for h in (1, 2, 5):
        sub[f"mid_fwd_{h}"] = sub["mid_price"].shift(-h) - sub["mid_price"]
        sub[f"dt_fwd_{h}"] = sub["timestamp"].shift(-h) - sub["timestamp"]

    print(f"\n=== OIM RESULT product={product} day={day} n={len(sub)} ===")
    for level in ("l1", "l2", "l3"):
        col = f"{level}_oim"
        print(f"\n--- {level.upper()} ---")
        for h in (1, 2, 5):
            valid = sub[(sub[col].notna()) & (sub[f"mid_fwd_{h}"].notna()) & (sub[f"dt_fwd_{h}"] == 100 * h)]
            corr = valid[col].corr(valid[f"mid_fwd_{h}"])
            print(f"corr_fwd_{h}: {corr:.6f}")

        for h in (1, 2, 5):
            valid = sub[(sub[col].notna()) & (sub[f"mid_fwd_{h}"].notna()) & (sub[f"dt_fwd_{h}"] == 100 * h)]
            long_sig = valid[col] > 0.5
            short_sig = valid[col] < -0.5
            hits = (valid.loc[long_sig, f"mid_fwd_{h}"] > 0).sum() + (valid.loc[short_sig, f"mid_fwd_{h}"] < 0).sum()
            n = int(long_sig.sum() + short_sig.sum())
            rate = (hits / n) if n > 0 else np.nan
            print(f"hit_rate_threshold_0.5_fwd_{h}: {rate:.6f} n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OIM predictive power on Round 2 price CSVs.")
    parser.add_argument(
        "--data-dir",
        default="imc-prosperity-4/data/ROUND_2",
        help="Directory containing prices_round_2_day_*.csv",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"],
    )
    parser.add_argument("--days", nargs="+", type=int, default=[-1, 0, 1])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    for day in args.days:
        path = data_dir / f"prices_round_2_day_{day}.csv"
        df = _load_prices(path)
        for product in args.products:
            sub = df[df["product"] == product].copy()
            if sub.empty:
                print(f"\n=== OIM RESULT product={product} day={day} n=0 ===")
                continue
            _print_metrics(sub, product, day)


if __name__ == "__main__":
    main()
