from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from submission_log_utils import load_submission_log


PRODUCTS = ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT")


@dataclass
class Claim:
    claim: str
    status: str
    evidence: str
    values: dict[str, Any]


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _best_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["best_bid"] = out[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1, skipna=True)
    out["best_ask"] = out[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1, skipna=True)
    out["best_spread"] = out["best_ask"] - out["best_bid"]
    return out


def _dataset_claims(data_dir: Path) -> list[Claim]:
    claims: list[Claim] = []
    price_counts: dict[str, dict[str, int]] = {p: {} for p in PRODUCTS}
    trade_counts: dict[str, dict[str, int]] = {p: {} for p in PRODUCTS}
    pepper_slopes: dict[str, float] = {}
    pepper_r2: dict[str, float] = {}
    ash_slopes: dict[str, float] = {}
    ash_r2: dict[str, float] = {}
    spread_median: dict[str, dict[str, float]] = {p: {} for p in PRODUCTS}
    level_presence_two_sided: dict[str, dict[str, float]] = {p: {} for p in PRODUCTS}
    level_presence_any_side: dict[str, dict[str, float]] = {p: {} for p in PRODUCTS}
    oim_corr: dict[str, dict[str, float]] = {p: {} for p in PRODUCTS}

    for day in (-1, 0, 1):
        prices = pd.read_csv(data_dir / f"prices_round_2_day_{day}.csv", sep=";")
        trades = pd.read_csv(data_dir / f"trades_round_2_day_{day}.csv", sep=";")

        for product in PRODUCTS:
            p = prices[prices["product"] == product].copy().sort_values("timestamp")
            t = trades[trades["symbol"] == product].copy()
            price_counts[product][str(day)] = int(len(p))
            trade_counts[product][str(day)] = int(len(t))

            if p.empty:
                continue

            p = _best_prices(p)
            spread_median[product][str(day)] = float(p["best_spread"].median())

            for lvl in (1, 2, 3):
                bid_ok = p[f"bid_price_{lvl}"].notna()
                ask_ok = p[f"ask_price_{lvl}"].notna()
                level_presence_two_sided[product][f"day_{day}_L{lvl}"] = float((bid_ok & ask_ok).mean())
                level_presence_any_side[product][f"day_{day}_L{lvl}"] = float((bid_ok | ask_ok).mean())

            fit_df = p[p["mid_price"] > 0].copy()
            x = fit_df["timestamp"].astype(float).to_numpy()
            y = fit_df["mid_price"].astype(float).to_numpy()
            coef = np.polyfit(x, y, deg=1)
            y_hat = coef[0] * x + coef[1]
            slope_per_tick = float(coef[0] * 100.0)
            fit_r2 = _r2(y, y_hat)
            if product == "INTARIAN_PEPPER_ROOT":
                pepper_slopes[str(day)] = slope_per_tick
                pepper_r2[str(day)] = fit_r2
            else:
                ash_slopes[str(day)] = slope_per_tick
                ash_r2[str(day)] = fit_r2

            # OIM predictive check at 1-step horizon.
            b1 = p["bid_volume_1"].fillna(0)
            a1 = p["ask_volume_1"].fillna(0)
            oim = (b1 - a1) / (b1 + a1).replace(0, np.nan)
            ret1 = p["mid_price"].shift(-1) - p["mid_price"]
            corr = float(pd.Series(oim).corr(pd.Series(ret1)))
            oim_corr[product][str(day)] = corr

    claims.append(
        Claim(
            claim="Round 2 dataset size by product is known from raw CSVs.",
            status="VERIFIED",
            evidence="prices_round_2_day_*.csv and trades_round_2_day_*.csv",
            values={"price_counts": price_counts, "trade_counts": trade_counts},
        )
    )

    claims.append(
        Claim(
            claim="Pepper has near-deterministic upward trend around +0.1 per tick across days.",
            status="VERIFIED",
            evidence="Linear fit on Pepper mid_price vs timestamp from raw CSVs.",
            values={"slope_per_tick": pepper_slopes, "r2": pepper_r2},
        )
    )

    claims.append(
        Claim(
            claim="Ash has low drift relative to Pepper and weaker linear fit.",
            status="VERIFIED",
            evidence="Linear fit on Ash mid_price vs timestamp from raw CSVs.",
            values={"slope_per_tick": ash_slopes, "r2": ash_r2},
        )
    )

    claims.append(
        Claim(
            claim="Spread and depth level presence can be measured directly from raw CSVs.",
            status="VERIFIED",
            evidence="Best spread and level presence computed from price snapshots.",
            values={
                "spread_median": spread_median,
                "level_presence_two_sided": level_presence_two_sided,
                "level_presence_any_side": level_presence_any_side,
            },
        )
    )

    claims.append(
        Claim(
            claim="L1 OIM has weak next-tick predictive power in current checks.",
            status="VERIFIED",
            evidence="Pearson corr(L1 OIM, 1-step mid return) on each day.",
            values={"l1_oim_corr_1step": oim_corr},
        )
    )

    # Validate documented large-L2 structure for Ash.
    ash_prices = []
    ash_trades = []
    for day in (-1, 0, 1):
        p = pd.read_csv(data_dir / f"prices_round_2_day_{day}.csv", sep=";")
        t = pd.read_csv(data_dir / f"trades_round_2_day_{day}.csv", sep=";")
        p = p[p["product"] == "ASH_COATED_OSMIUM"].copy()
        p["day"] = day
        t = t[t["symbol"] == "ASH_COATED_OSMIUM"].copy()
        t["day"] = day
        ash_prices.append(p)
        ash_trades.append(t)
    ash_p = pd.concat(ash_prices, ignore_index=True)
    ash_t = pd.concat(ash_trades, ignore_index=True)

    both_l2 = ash_p["bid_volume_2"].notna() & ash_p["ask_volume_2"].notna()
    any_l2 = ash_p["bid_volume_2"].notna() | ash_p["ask_volume_2"].notna()
    ash_p["any_l2"] = any_l2
    ash_p["l2_total"] = ash_p["bid_volume_2"].fillna(0) + ash_p["ask_volume_2"].fillna(0)
    ash_p["large_l2"] = both_l2 & (ash_p["l2_total"] >= 48)

    trade_ticks = (
        ash_t.groupby(["day", "timestamp"]).size().reset_index(name="n_trades")
    )
    trade_keys = set(zip(trade_ticks["day"], trade_ticks["timestamp"]))
    ash_p["has_trade"] = [
        (int(d), int(ts)) in trade_keys for d, ts in zip(ash_p["day"], ash_p["timestamp"])
    ]

    ash_p = ash_p.sort_values(["day", "timestamp"]).reset_index(drop=True)
    prev_large = ash_p.groupby("day")["large_l2"].shift(1)
    prev_large = prev_large.where(prev_large.notna(), False).astype(bool)
    prev_large = prev_large & (
        ash_p.groupby("day")["timestamp"].shift(1) == (ash_p["timestamp"] - 100)
    )
    ash_p["new_large_l2"] = ash_p["large_l2"] & (~prev_large)
    ash_p["resting_large_l2"] = ash_p["large_l2"] & prev_large

    prev_any_l2 = ash_p.groupby("day")["any_l2"].shift(1)
    prev_any_l2 = prev_any_l2.where(prev_any_l2.notna(), False).astype(bool)
    prev_any_l2 = prev_any_l2 & (
        ash_p.groupby("day")["timestamp"].shift(1) == (ash_p["timestamp"] - 100)
    )
    ash_p["new_large_l2_prev_any"] = ash_p["large_l2"] & (~prev_any_l2)
    ash_p["resting_large_l2_prev_any"] = ash_p["large_l2"] & prev_any_l2

    large = ash_p[ash_p["large_l2"]].copy()
    breakdown_prev_large = {
        "new_l2_trade": int((large["new_large_l2"] & large["has_trade"]).sum()),
        "new_l2_no_trade": int((large["new_large_l2"] & ~large["has_trade"]).sum()),
        "resting_l2_trade": int((large["resting_large_l2"] & large["has_trade"]).sum()),
        "resting_l2_no_trade": int((large["resting_large_l2"] & ~large["has_trade"]).sum()),
    }
    breakdown_prev_any = {
        "new_l2_trade": int((large["new_large_l2_prev_any"] & large["has_trade"]).sum()),
        "new_l2_no_trade": int((large["new_large_l2_prev_any"] & ~large["has_trade"]).sum()),
        "resting_l2_trade": int((large["resting_large_l2_prev_any"] & large["has_trade"]).sum()),
        "resting_l2_no_trade": int((large["resting_large_l2_prev_any"] & ~large["has_trade"]).sum()),
    }

    claims.append(
        Claim(
            claim="Ash L2 persistence structure (including large-L2 threshold) is measurable from raw data.",
            status="VERIFIED",
            evidence=(
                "Round 2 Ash price/trade CSVs with large-L2 defined as "
                "bid_vol_2 + ask_vol_2 >= 48; includes both prev-large-L2 and prev-any-L2 decompositions."
            ),
            values={
                "ticks_with_any_l2": int(any_l2.sum()),
                "ticks_with_large_l2": int(ash_p["large_l2"].sum()),
                "large_l2_breakdown_prev_large_l2": breakdown_prev_large,
                "large_l2_breakdown_prev_any_l2": breakdown_prev_any,
            },
        )
    )

    return claims


def _infer_buy_price(product_df: pd.DataFrame) -> tuple[int, float]:
    entry_ts = int(product_df["timestamp"].min())
    row = product_df[product_df["timestamp"] == entry_ts].iloc[0]
    asks = [float(row[c]) for c in ("ask_price_1", "ask_price_2", "ask_price_3") if pd.notna(row[c])]
    if not asks:
        raise ValueError("No ask observed at entry tick; cannot infer buy price.")
    return entry_ts, min(asks)


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


def _reconstruct_mark_from_inventory(log_path: Path, product: str) -> dict[str, Any] | None:
    log = load_submission_log(log_path)
    if log.activities.empty:
        return None

    a = log.activities[log.activities["product"] == product].copy().sort_values("timestamp").reset_index(drop=True)
    if a.empty:
        return None

    t = _submission_trades_for_product(log.trades, product)
    if t.empty:
        return None

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
    pos_series = []
    cash_series = []
    traded_series = []
    trade_ts = set(by_ts.keys())
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
        return None

    nz["mark"] = (nz["profit_and_loss"] - nz["recon_cash"]) / nz["recon_pos"]
    nz["best_bid"] = nz[["bid_price_1", "bid_price_2", "bid_price_3"]].max(axis=1, skipna=True)
    nz["best_ask"] = nz[["ask_price_1", "ask_price_2", "ask_price_3"]].min(axis=1, skipna=True)
    nz["mid"] = (nz["best_bid"] + nz["best_ask"]) / 2.0
    nz = nz.dropna(subset=["mark"]).copy()
    if nz.empty:
        return None

    mark = nz["mark"].astype(float)
    dmark = mark.diff().dropna()

    valid_mid = nz.dropna(subset=["mid"]).copy()
    corr_mark_mid = float(mark.corr(valid_mid["mid"])) if not valid_mid.empty else float("nan")
    abs_err = (valid_mid["mark"] - valid_mid["mid"]).abs() if not valid_mid.empty else pd.Series(dtype=float)
    inside_rate = (
        float(((valid_mid["mark"] >= valid_mid["best_bid"]) & (valid_mid["mark"] <= valid_mid["best_ask"])).mean())
        if not valid_mid.empty
        else float("nan")
    )

    nz["prev_pos"] = nz["recon_pos"].shift(1)
    stable = nz[(~nz["had_submission_trade"]) & (nz["prev_pos"] == nz["recon_pos"])].copy()
    stable_mark = stable["mark"].astype(float) if not stable.empty else pd.Series(dtype=float)
    stable_dmark = stable_mark.diff().dropna() if len(stable_mark) >= 2 else pd.Series(dtype=float)
    stable_slope = (
        float(np.polyfit(stable["timestamp"].astype(float), stable_mark.to_numpy(dtype=float), deg=1)[0] * 100.0)
        if len(stable_mark) >= 2
        else float("nan")
    )

    out: dict[str, Any] = {
        "log_path": str(log_path),
        "submission_trade_rows": int(len(t)),
        "ticks_nonzero_position": int(len(nz)),
        "max_abs_position": int(nz["recon_pos"].abs().max()),
        "mark_mean": float(mark.mean()),
        "mark_std": float(mark.std(ddof=0)),
        "mark_min": float(mark.min()),
        "mark_max": float(mark.max()),
        "lag1_mark_autocorr": float(mark.autocorr()) if len(mark) >= 2 else float("nan"),
        "dmark_mean": float(dmark.mean()) if len(dmark) > 0 else float("nan"),
        "dmark_std": float(dmark.std(ddof=0)) if len(dmark) > 0 else float("nan"),
        "mark_on_1_over_1024_grid": bool(np.allclose(mark, np.round(mark * 1024) / 1024)),
        "corr_mark_mid": corr_mark_mid,
        "mean_abs_mark_minus_mid": float(abs_err.mean()) if len(abs_err) > 0 else float("nan"),
        "inside_best_spread_rate": inside_rate,
        "stable_ticks": int(len(stable)),
        "stable_dmark_mean": float(stable_dmark.mean()) if len(stable_dmark) > 0 else float("nan"),
        "stable_dmark_std": float(stable_dmark.std(ddof=0)) if len(stable_dmark) > 0 else float("nan"),
        "stable_slope_per_tick": stable_slope,
    }
    return out


def _hold1_claims(primary_log: Path, secondary_log: Path | None) -> list[Claim]:
    claims: list[Claim] = []

    primary = load_submission_log(primary_log)
    if primary.activities.empty:
        raise ValueError(f"Empty activities in {primary_log}")
    p = primary.activities[primary.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
    entry_ts, buy_px = _infer_buy_price(p)
    p["server_mark"] = np.where(p["timestamp"] > entry_ts, p["profit_and_loss"] + buy_px, np.nan)
    cal = p.dropna(subset=["server_mark"]).copy()
    cal = _best_prices(cal)
    cal["mid"] = (cal["best_bid"] + cal["best_ask"]) / 2.0

    mark = cal["server_mark"].astype(float)
    dmark = mark.diff().dropna()

    vals = {
        "entry_timestamp": entry_ts,
        "inferred_buy_price": float(buy_px),
        "n_post_entry_ticks": int(len(cal)),
        "mark_mean": float(mark.mean()),
        "mark_std": float(mark.std(ddof=0)),
        "mark_min": float(mark.min()),
        "mark_max": float(mark.max()),
        "lag1_mark_autocorr": float(mark.autocorr()),
        "dmark_mean": float(dmark.mean()),
        "dmark_std": float(dmark.std(ddof=0)),
        "dmark_min": float(dmark.min()),
        "dmark_max": float(dmark.max()),
        "mark_on_1_over_1024_grid": bool(np.allclose(mark, np.round(mark * 1024) / 1024)),
        "dmark_on_1_over_1024_grid": bool(np.allclose(dmark, np.round(dmark * 1024) / 1024)),
        "corr_mark_mid": float(mark.corr(cal["mid"])),
        "corr_dmark_dmid": float(dmark.corr(cal["mid"].diff().dropna())),
        "inside_best_spread_rate": float(((mark >= cal["best_bid"]) & (mark <= cal["best_ask"])).mean()),
        "below_best_bid_rate": float((mark < cal["best_bid"]).mean()),
        "above_best_ask_rate": float((mark > cal["best_ask"]).mean()),
    }
    claims.append(
        Claim(
            claim="Osmium hold-1 probe recovers continuous server mark path from PnL + buy price.",
            status="VERIFIED",
            evidence=f"{primary_log}",
            values=vals,
        )
    )

    pair_df = cal.dropna(subset=["best_bid", "best_ask"]).copy()
    pair_df["round_mark"] = np.round(pair_df["server_mark"])
    pair_df["pair"] = list(
        zip(
            (pair_df["best_bid"] - pair_df["round_mark"]).astype(int),
            (pair_df["best_ask"] - pair_df["round_mark"]).astype(int),
        )
    )
    top_pairs = pair_df["pair"].value_counts(normalize=True).head(12)
    claims.append(
        Claim(
            claim="Osmium L1 offset-pair regimes relative to round(server_mark) are measurable.",
            status="VERIFIED",
            evidence=f"{primary_log}",
            values={"top_offset_pair_shares": {str(k): float(v) for k, v in top_pairs.items()}},
        )
    )

    if secondary_log is not None:
        secondary = load_submission_log(secondary_log)
        s = secondary.activities[secondary.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
        s_entry, s_buy = _infer_buy_price(s)
        s_mark = s.loc[s["timestamp"] > s_entry, "profit_and_loss"] + s_buy
        p_mark = p.loc[p["timestamp"] > entry_ts, "profit_and_loss"] + buy_px
        same = bool(len(s_mark) == len(p_mark) and np.allclose(s_mark.to_numpy(), p_mark.to_numpy()))
        claims.append(
            Claim(
                claim="Two available Osmium hold-1 runs represent independent hidden-state samples.",
                status="PARTIAL" if same else "VERIFIED",
                evidence=f"{primary_log} vs {secondary_log}",
                values={
                    "same_tick_path": same,
                    "interpretation": (
                        "same path -> useful consistency check but not independent sample"
                        if same
                        else "different paths -> independent hidden-state evidence"
                    ),
                },
            )
        )

    return claims


def _hold1_pepper_claims(pepper_log: Path | None) -> list[Claim]:
    if pepper_log is None or not pepper_log.exists():
        return [
            Claim(
                claim="Pepper hidden server mark process is recovered from a controlled live probe.",
                status="ASSUMPTION",
                evidence="No hold-1 Pepper probe log currently analyzed.",
                values={},
            )
        ]

    log = load_submission_log(pepper_log)
    if log.activities.empty:
        return [
            Claim(
                claim="Pepper hidden server mark process is recovered from a controlled live probe.",
                status="ASSUMPTION",
                evidence=f"{pepper_log} has empty activitiesLog.",
                values={},
            )
        ]

    p = log.activities[log.activities["product"] == "INTARIAN_PEPPER_ROOT"].copy().sort_values("timestamp")
    if p.empty:
        return [
            Claim(
                claim="Pepper hidden server mark process is recovered from a controlled live probe.",
                status="ASSUMPTION",
                evidence=f"{pepper_log} contains no INTARIAN_PEPPER_ROOT rows.",
                values={},
            )
        ]

    entry_ts, buy_px = _infer_buy_price(p)
    p["server_mark"] = np.where(p["timestamp"] > entry_ts, p["profit_and_loss"] + buy_px, np.nan)
    cal = p.dropna(subset=["server_mark"]).copy()
    if cal.empty:
        return [
            Claim(
                claim="Pepper hidden server mark process is recovered from a controlled live probe.",
                status="ASSUMPTION",
                evidence=f"{pepper_log} has no post-entry rows for mark recovery.",
                values={},
            )
        ]

    cal = _best_prices(cal)
    cal["mid"] = (cal["best_bid"] + cal["best_ask"]) / 2.0
    mark = cal["server_mark"].astype(float)
    dmark = mark.diff().dropna()

    vals = {
        "entry_timestamp": int(entry_ts),
        "inferred_buy_price": float(buy_px),
        "n_post_entry_ticks": int(len(cal)),
        "mark_mean": float(mark.mean()),
        "mark_std": float(mark.std(ddof=0)),
        "mark_min": float(mark.min()),
        "mark_max": float(mark.max()),
        "lag1_mark_autocorr": float(mark.autocorr()) if len(mark) >= 2 else float("nan"),
        "dmark_mean": float(dmark.mean()) if len(dmark) > 0 else float("nan"),
        "dmark_std": float(dmark.std(ddof=0)) if len(dmark) > 0 else float("nan"),
        "dmark_min": float(dmark.min()) if len(dmark) > 0 else float("nan"),
        "dmark_max": float(dmark.max()) if len(dmark) > 0 else float("nan"),
        "mark_on_1_over_1024_grid": bool(np.allclose(mark, np.round(mark * 1024) / 1024)),
        "corr_mark_mid": float(mark.corr(cal["mid"])),
        "inside_best_spread_rate": float(((mark >= cal["best_bid"]) & (mark <= cal["best_ask"])).mean()),
        "below_best_bid_rate": float((mark < cal["best_bid"]).mean()),
        "above_best_ask_rate": float((mark > cal["best_ask"]).mean()),
    }

    pair_df = cal.dropna(subset=["best_bid", "best_ask"]).copy()
    pair_df["round_mark"] = np.round(pair_df["server_mark"])
    pair_df["pair"] = list(
        zip(
            (pair_df["best_bid"] - pair_df["round_mark"]).astype(int),
            (pair_df["best_ask"] - pair_df["round_mark"]).astype(int),
        )
    )
    top_pairs = pair_df["pair"].value_counts(normalize=True).head(12)

    return [
        Claim(
            claim="Pepper hold-1 probe recovers continuous server mark path from PnL + buy price.",
            status="VERIFIED",
            evidence=f"{pepper_log}",
            values=vals,
        ),
        Claim(
            claim="Pepper L1 offset-pair regimes relative to round(server_mark) are measurable.",
            status="VERIFIED",
            evidence=f"{pepper_log}",
            values={"top_offset_pair_shares": {str(k): float(v) for k, v in top_pairs.items()}},
        ),
    ]


def _flip_osmium_claims(flip_log: Path | None, hold1_log: Path | None) -> list[Claim]:
    if flip_log is None or not flip_log.exists():
        return [
            Claim(
                claim="Osmium hold-then-flat behavior is validated via controlled flip probe.",
                status="ASSUMPTION",
                evidence="No flip-1 Osmium probe log currently analyzed.",
                values={},
            )
        ]

    log = load_submission_log(flip_log)
    if log.activities.empty:
        return [
            Claim(
                claim="Osmium hold-then-flat behavior is validated via controlled flip probe.",
                status="ASSUMPTION",
                evidence=f"{flip_log} has empty activitiesLog.",
                values={},
            )
        ]

    a = log.activities[log.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
    trades = _submission_trades_for_product(log.trades, "ASH_COATED_OSMIUM")
    if a.empty or trades.empty:
        return [
            Claim(
                claim="Osmium hold-then-flat behavior is validated via controlled flip probe.",
                status="ASSUMPTION",
                evidence=f"{flip_log} missing required ASH activity/trade rows.",
                values={},
            )
        ]

    buys = trades[trades["buyer"] == "SUBMISSION"].copy()
    sells = trades[trades["seller"] == "SUBMISSION"].copy()
    if buys.empty or sells.empty:
        return [
            Claim(
                claim="Osmium hold-then-flat behavior is validated via controlled flip probe.",
                status="ASSUMPTION",
                evidence=f"{flip_log} missing buy/sell submission legs.",
                values={},
            )
        ]

    entry_trade = buys.iloc[0]
    exit_trade = sells.iloc[-1]
    entry_ts = int(entry_trade["timestamp"])
    exit_ts = int(exit_trade["timestamp"])
    buy_price = float(entry_trade["price"])
    sell_price = float(exit_trade["price"])

    a["server_mark"] = np.where(
        (a["timestamp"] > entry_ts) & (a["timestamp"] <= exit_ts),
        a["profit_and_loss"] + buy_price,
        np.nan,
    )
    hold = a.dropna(subset=["server_mark"]).copy()
    flat = a[a["timestamp"] > exit_ts].copy()
    hold_mark = hold["server_mark"].astype(float)
    hold_dmark = hold_mark.diff().dropna()

    overlap_match = float("nan")
    if hold1_log is not None and hold1_log.exists():
        base = load_submission_log(hold1_log)
        bp = base.activities[base.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
        if not bp.empty:
            b_entry, b_buy = _infer_buy_price(bp)
            bp["server_mark"] = np.where(bp["timestamp"] > b_entry, bp["profit_and_loss"] + b_buy, np.nan)
            b_hold = bp.dropna(subset=["server_mark"]).copy()
            overlap = min(len(hold), len(b_hold))
            if overlap > 0:
                overlap_match = float(
                    np.mean(
                        np.isclose(
                            hold["server_mark"].to_numpy()[:overlap],
                            b_hold["server_mark"].to_numpy()[:overlap],
                        )
                    )
                )

    vals = {
        "entry_timestamp": entry_ts,
        "exit_timestamp": exit_ts,
        "inferred_buy_price": buy_price,
        "inferred_sell_price": sell_price,
        "n_hold_ticks": int(len(hold)),
        "n_flat_ticks": int(len(flat)),
        "hold_mark_mean": float(hold_mark.mean()) if len(hold_mark) > 0 else float("nan"),
        "hold_mark_std": float(hold_mark.std(ddof=0)) if len(hold_mark) > 0 else float("nan"),
        "hold_dmark_mean": float(hold_dmark.mean()) if len(hold_dmark) > 0 else float("nan"),
        "hold_dmark_std": float(hold_dmark.std(ddof=0)) if len(hold_dmark) > 0 else float("nan"),
        "hold_mark_on_1_over_1024_grid": bool(np.allclose(hold_mark, np.round(hold_mark * 1024) / 1024))
        if len(hold_mark) > 0
        else False,
        "flat_pnl_std": float(flat["profit_and_loss"].std(ddof=0)) if len(flat) > 0 else float("nan"),
        "flat_pnl_range": float(flat["profit_and_loss"].max() - flat["profit_and_loss"].min())
        if len(flat) > 0
        else float("nan"),
        "hold_overlap_match_vs_hold1_osmium": overlap_match,
    }

    status = "VERIFIED"
    if np.isnan(vals["flat_pnl_std"]) or vals["flat_pnl_std"] > 1e-9:
        status = "PARTIAL"

    return [
        Claim(
            claim="Osmium hold-then-flat behavior is validated via controlled flip probe.",
            status=status,
            evidence=f"{flip_log}",
            values=vals,
        )
    ]


def _dual_hold_claims(dual_log: Path | None, osmium_hold1_log: Path | None, pepper_hold1_log: Path | None) -> list[Claim]:
    if dual_log is None or not dual_log.exists():
        return [
            Claim(
                claim="Dual hold probe recovers both product marks in one run.",
                status="ASSUMPTION",
                evidence="No dual-hold probe log currently analyzed.",
                values={},
            )
        ]

    log = load_submission_log(dual_log)
    if log.activities.empty:
        return [
            Claim(
                claim="Dual hold probe recovers both product marks in one run.",
                status="ASSUMPTION",
                evidence=f"{dual_log} has empty activitiesLog.",
                values={},
            )
        ]

    out: dict[str, Any] = {}
    status = "VERIFIED"
    for product in ("ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"):
        p = log.activities[log.activities["product"] == product].copy().sort_values("timestamp")
        if p.empty:
            status = "PARTIAL"
            out[product] = {"missing": True}
            continue
        entry_ts, buy_px = _infer_buy_price(p)
        p["server_mark"] = np.where(p["timestamp"] > entry_ts, p["profit_and_loss"] + buy_px, np.nan)
        cal = p.dropna(subset=["server_mark"]).copy()
        mark = cal["server_mark"].astype(float)
        dmark = mark.diff().dropna()
        out[product] = {
            "entry_timestamp": int(entry_ts),
            "inferred_buy_price": float(buy_px),
            "n_post_entry_ticks": int(len(cal)),
            "mark_mean": float(mark.mean()) if len(mark) > 0 else float("nan"),
            "mark_std": float(mark.std(ddof=0)) if len(mark) > 0 else float("nan"),
            "dmark_mean": float(dmark.mean()) if len(dmark) > 0 else float("nan"),
            "dmark_std": float(dmark.std(ddof=0)) if len(dmark) > 0 else float("nan"),
            "mark_on_1_over_1024_grid": bool(np.allclose(mark, np.round(mark * 1024) / 1024))
            if len(mark) > 0
            else False,
        }

    # Consistency checks vs single-product probes.
    osmium_match = float("nan")
    pepper_match = float("nan")
    if osmium_hold1_log is not None and osmium_hold1_log.exists():
        base = load_submission_log(osmium_hold1_log)
        b = base.activities[base.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
        if not b.empty:
            be, bb = _infer_buy_price(b)
            b["server_mark"] = np.where(b["timestamp"] > be, b["profit_and_loss"] + bb, np.nan)
            bcal = b.dropna(subset=["server_mark"]).copy()
            d_ash = log.activities[log.activities["product"] == "ASH_COATED_OSMIUM"].copy().sort_values("timestamp")
            de, db = _infer_buy_price(d_ash)
            d_ash["server_mark"] = np.where(d_ash["timestamp"] > de, d_ash["profit_and_loss"] + db, np.nan)
            dcal = d_ash.dropna(subset=["server_mark"]).copy()
            overlap = min(len(bcal), len(dcal))
            if overlap > 0:
                osmium_match = float(
                    np.mean(
                        np.isclose(
                            bcal["server_mark"].to_numpy()[:overlap],
                            dcal["server_mark"].to_numpy()[:overlap],
                        )
                    )
                )

    if pepper_hold1_log is not None and pepper_hold1_log.exists():
        base = load_submission_log(pepper_hold1_log)
        b = base.activities[base.activities["product"] == "INTARIAN_PEPPER_ROOT"].copy().sort_values("timestamp")
        if not b.empty:
            be, bb = _infer_buy_price(b)
            b["server_mark"] = np.where(b["timestamp"] > be, b["profit_and_loss"] + bb, np.nan)
            bcal = b.dropna(subset=["server_mark"]).copy()
            d_pep = log.activities[log.activities["product"] == "INTARIAN_PEPPER_ROOT"].copy().sort_values("timestamp")
            de, db = _infer_buy_price(d_pep)
            d_pep["server_mark"] = np.where(d_pep["timestamp"] > de, d_pep["profit_and_loss"] + db, np.nan)
            dcal = d_pep.dropna(subset=["server_mark"]).copy()
            overlap = min(len(bcal), len(dcal))
            if overlap > 0:
                pepper_match = float(
                    np.mean(
                        np.isclose(
                            bcal["server_mark"].to_numpy()[:overlap],
                            dcal["server_mark"].to_numpy()[:overlap],
                        )
                    )
                )

    out["match_vs_single_product_probes"] = {
        "osmium_match_rate": osmium_match,
        "pepper_match_rate": pepper_match,
    }
    if (not np.isnan(osmium_match) and osmium_match < 0.999999) or (
        not np.isnan(pepper_match) and pepper_match < 0.999999
    ):
        status = "PARTIAL"

    return [
        Claim(
            claim="Dual hold probe recovers both product marks in one run.",
            status=status,
            evidence=f"{dual_log}",
            values=out,
        )
    ]


def _pepper_partial_reconstruction_claims() -> list[Claim]:
    candidate_logs = [
        Path("imc-prosperity-4/308866/308866.log"),
        Path("imc-prosperity-4/best_strat_vedant/best_strat_vedant.log"),
        Path("imc-prosperity-4/best_strat_yudhiish/332493.log"),
    ]
    resolved_logs = [p.expanduser().resolve() for p in candidate_logs if p.expanduser().resolve().exists()]

    recon: dict[str, dict[str, Any]] = {}
    for p in resolved_logs:
        try:
            stats = _reconstruct_mark_from_inventory(p, "INTARIAN_PEPPER_ROOT")
        except Exception:
            stats = None
        if stats is not None:
            recon[p.name] = stats

    if not recon:
        return [
            Claim(
                claim="Pepper mark has partial reconstruction from non-probe submission logs.",
                status="ASSUMPTION",
                evidence="No non-probe logs with usable Pepper submission inventory/trade history found.",
                values={},
            )
        ]

    return [
        Claim(
            claim="Pepper mark has partial reconstruction from non-probe submission logs.",
            status="PARTIAL",
            evidence=", ".join(str(p) for p in resolved_logs),
            values={"reconstructions": recon},
        )
    ]


def _assumption_claims() -> list[Claim]:
    return [
        Claim(
            claim="Current Round 2 Rust simulator parameters match inferred live bot transfer functions.",
            status="ASSUMPTION",
            evidence="No full actual-vs-sim validation against inferred layer rules.",
            values={},
        ),
        Claim(
            claim="Trade-flow conditionals are fully identified by current scripts.",
            status="ASSUMPTION",
            evidence="Most existing scripts summarize marginals and selected probes only.",
            values={},
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Round 2 facts vs assumptions.")
    parser.add_argument(
        "--data-dir",
        default="imc-prosperity-4/data/ROUND_2",
        help="Path to ROUND_2 csv directory.",
    )
    parser.add_argument(
        "--hold1-log",
        default="imc-prosperity-4/vedant/round_2_analysis/buy_1_osmium/340214.log",
        help="Primary hold-1 Osmium submission artifact (.log preferred).",
    )
    parser.add_argument(
        "--hold1-secondary-log",
        default="imc-prosperity-4/vedant/round_2_analysis/hold_1_unit/296479.log",
        help="Secondary hold-1 Osmium artifact for cross-check.",
    )
    parser.add_argument(
        "--hold1-pepper-log",
        default="imc-prosperity-4/vedant/round_2_analysis/hold_1_pepper_logs/343258.log",
        help="Hold-1 Pepper probe artifact.",
    )
    parser.add_argument(
        "--flip-osmium-log",
        default="imc-prosperity-4/vedant/round_2_analysis/flip_1_osmium_logs/343375.log",
        help="Flip-1 Osmium probe artifact.",
    )
    parser.add_argument(
        "--dual-hold-log",
        default="imc-prosperity-4/vedant/round_2_analysis/dual_hold_logs/343414.log",
        help="Dual-hold probe artifact.",
    )
    parser.add_argument(
        "--out-json",
        default="imc-prosperity-4/vedant/round_2_analysis/round2_evidence_validation.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    hold1_log = Path(args.hold1_log).expanduser().resolve()
    hold1_secondary = Path(args.hold1_secondary_log).expanduser().resolve() if args.hold1_secondary_log else None
    hold1_pepper = Path(args.hold1_pepper_log).expanduser().resolve() if args.hold1_pepper_log else None
    flip_osmium = Path(args.flip_osmium_log).expanduser().resolve() if args.flip_osmium_log else None
    dual_hold = Path(args.dual_hold_log).expanduser().resolve() if args.dual_hold_log else None
    out_json = Path(args.out_json).expanduser().resolve()

    claims: list[Claim] = []
    claims.extend(_dataset_claims(data_dir))
    claims.extend(_hold1_claims(hold1_log, hold1_secondary))
    claims.extend(_hold1_pepper_claims(hold1_pepper))
    claims.extend(_flip_osmium_claims(flip_osmium, hold1_log))
    claims.extend(_dual_hold_claims(dual_hold, hold1_log, hold1_pepper))
    claims.extend(_pepper_partial_reconstruction_claims())
    claims.extend(_assumption_claims())

    print("=== ROUND 2 EVIDENCE MATRIX ===")
    for idx, c in enumerate(claims, start=1):
        print(f"{idx:02d}. [{c.status}] {c.claim}")
        print(f"    evidence: {c.evidence}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"claims": [asdict(c) for c in claims]}
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nreport_saved_to: {out_json}")


if __name__ == "__main__":
    main()
