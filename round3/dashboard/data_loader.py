"""
Data loading and preprocessing for IMC Prosperity 4 Round 3 dashboard.
"""

from pathlib import Path
import io
import json

import numpy as np
import pandas as pd

from options_math import implied_vol, log_moneyness, fit_smile_parabola, smile_iv_from_coeffs, bs_call, bs_greeks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "ROUND_3"
ROUND3_DAYS = (0, 1, 2)

UNDERLYING = "VELVETFRUIT_EXTRACT"
VEV_STRIKES: dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
ALL_PRODUCTS = [UNDERLYING, "HYDROGEL_PACK"] + list(VEV_STRIKES.keys())

# Each "Solvenarian day" in the data = timestamps 0 … 999900 (step 100).
# TTE decreases linearly within a day.
MAX_TS_PER_DAY = 999_900  # last timestamp in a day


# ---------------------------------------------------------------------------
# Price / trade CSV loaders
# ---------------------------------------------------------------------------

_PRICE_NUM_COLS = [
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3", "ask_price_1", "ask_volume_1",
    "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]


def _to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_prices(data_dir=None, days=ROUND3_DAYS) -> pd.DataFrame:
    """
    Load prices CSVs for the specified days.

    Adds column `global_ts = day * 1_000_000 + timestamp` for cross-day ordering.
    """
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    frames = []
    for day in days:
        path = base / f"prices_round_3_day_{day}.csv"
        if path.exists():
            df = pd.read_csv(path, sep=";")
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    _to_numeric_inplace(out, _price_num_cols := _PRICE_NUM_COLS)
    out["global_ts"] = out["day"] * 1_000_000 + out["timestamp"]
    return out


def load_trades(data_dir=None, days=ROUND3_DAYS) -> pd.DataFrame:
    """
    Load trades CSVs for the specified days.

    Adds `day` and `global_ts` columns.
    """
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    frames = []
    for day in days:
        path = base / f"trades_round_3_day_{day}.csv"
        if path.exists():
            df = pd.read_csv(path, sep=";")
            df["day"] = day
            df["global_ts"] = day * 1_000_000 + df["timestamp"]
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Log file parser
# ---------------------------------------------------------------------------

def parse_log(content: str | bytes) -> dict | None:
    """
    Parse a Prosperity backtester / website log file.

    Returns dict with optional keys:
      activities_df : pd.DataFrame  (order-book data from activitiesLog)
      logs_df       : pd.DataFrame  (algorithm print output per timestamp)
      own_trades_df : pd.DataFrame  (tradeHistory where buyer/seller='SUBMISSION')
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    result: dict = {}

    if data.get("activitiesLog"):
        try:
            df = pd.read_csv(io.StringIO(data["activitiesLog"]), sep=";")
            _to_numeric_inplace(df, _PRICE_NUM_COLS)
            df["global_ts"] = df["day"] * 1_000_000 + df["timestamp"]
            result["activities_df"] = df
        except Exception:
            pass

    if data.get("logs"):
        result["logs_df"] = pd.DataFrame(data["logs"])

    if data.get("tradeHistory"):
        # tradeHistory always contains *our* fills regardless of buyer/seller content.
        # The backtester stamps buyer/seller='SUBMISSION'; the IMC website leaves them blank.
        trades_df = pd.DataFrame(data["tradeHistory"])
        trades_df["price"]    = pd.to_numeric(trades_df["price"], errors="coerce")
        trades_df["quantity"] = pd.to_numeric(trades_df["quantity"], errors="coerce")
        trades_df["timestamp"] = pd.to_numeric(trades_df["timestamp"], errors="coerce")

        # Infer buy/sell direction:
        #   1. If SUBMISSION appears → use it directly.
        #   2. Otherwise (website format) → infer from price vs activities mid price.
        buyer  = trades_df.get("buyer",  pd.Series("", index=trades_df.index)).fillna("")
        seller = trades_df.get("seller", pd.Series("", index=trades_df.index)).fillna("")
        has_submission = ((buyer.str.upper() == "SUBMISSION") |
                          (seller.str.upper() == "SUBMISSION")).any()

        if has_submission:
            trades_df["side"] = "BUY"
            trades_df.loc[seller.str.upper() == "SUBMISSION", "side"] = "SELL"
        else:
            # Infer from activitiesLog mid price at same (symbol, timestamp)
            trades_df["side"] = "UNKNOWN"
            if "activities_df" in result:
                acts = result["activities_df"][["product", "timestamp", "mid_price"]]
                merged = trades_df.merge(
                    acts.rename(columns={"product": "symbol"}),
                    on=["symbol", "timestamp"], how="left",
                )
                trades_df["side"] = np.where(
                    merged["mid_price"].isna(), "UNKNOWN",
                    np.where(merged["price"] >= merged["mid_price"], "BUY", "SELL"),
                )

        result["own_trades_df"] = trades_df

    return result


# ---------------------------------------------------------------------------
# Options analytics
# ---------------------------------------------------------------------------

def compute_tte(day: np.ndarray, timestamp: np.ndarray, tte_start: float = 7.0) -> np.ndarray:
    """
    Compute time-to-expiry (Solvenarian days) for each (day, timestamp) pair.

    At day=0, ts=0 → TTE = tte_start.
    Each day decrements TTE by 1; each timestamp fraction within a day
    decrements proportionally (ts / MAX_TS_PER_DAY).
    """
    return tte_start - day - timestamp / MAX_TS_PER_DAY


def build_options_df(prices_df: pd.DataFrame, tte_start: float = 7.0) -> pd.DataFrame:
    """
    Build a combined options analysis DataFrame.

    For each VEV product and each timestamp, computes:
      - spot price of the underlying
      - tte
      - market mid price of option
      - implied vol (iv)
      - log-moneyness (moneyness)
      - BS theoretical price using smile-fitted IV (theoretical_price)
      - IV deviation from fitted smile (iv_dev)
      - price deviation from theoretical (price_dev)
      - greeks: delta, gamma, vega

    Returns a long-format DataFrame with one row per (product, day, timestamp).
    """
    if prices_df.empty:
        return pd.DataFrame()

    # Extract underlying prices
    spot_df = (
        prices_df[prices_df["product"] == UNDERLYING][["day", "timestamp", "mid_price"]]
        .rename(columns={"mid_price": "spot"})
        .drop_duplicates(["day", "timestamp"])
    )

    rows = []
    for product, strike in VEV_STRIKES.items():
        vev_df = (
            prices_df[prices_df["product"] == product][
                ["day", "timestamp", "mid_price", "bid_price_1", "ask_price_1"]
            ]
            .drop_duplicates(["day", "timestamp"])
            .copy()
        )
        if vev_df.empty:
            continue

        merged = vev_df.merge(spot_df, on=["day", "timestamp"], how="inner")
        if merged.empty:
            continue

        # TTE
        merged["tte"] = compute_tte(merged["day"].values, merged["timestamp"].values, tte_start)
        merged["tte"] = merged["tte"].clip(lower=0.0)

        S = merged["spot"].values
        T = merged["tte"].values
        C = merged["mid_price"].values

        # Implied vol (vectorised) — mid, bid, ask
        merged["iv"]     = implied_vol(C, S, strike, T)
        merged["bid_iv"] = implied_vol(merged["bid_price_1"].values, S, strike, T)
        merged["ask_iv"] = implied_vol(merged["ask_price_1"].values, S, strike, T)

        # Log-moneyness: log(S/K)/sqrt(T)  (positive = ITM)
        merged["moneyness"] = log_moneyness(S, strike, T)

        merged["product"] = product
        merged["strike"] = strike
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    opts = pd.concat(rows, ignore_index=True)

    # ---------------------------------------------------------------------------
    # Fit vol smile at each timestamp → iv_hat, iv_dev, theoretical_price, price_dev
    # ---------------------------------------------------------------------------
    opts = _attach_smile_metrics(opts)

    return opts


def _attach_smile_metrics(opts: pd.DataFrame) -> pd.DataFrame:
    """
    For each (day, timestamp), fit a parabola to (moneyness, iv).
    Attach iv_hat, iv_dev, theoretical_price, price_dev, and greeks.
    """
    iv_hat_all = np.full(len(opts), np.nan)
    iv_dev_all = np.full(len(opts), np.nan)
    theo_all = np.full(len(opts), np.nan)
    price_dev_all = np.full(len(opts), np.nan)

    delta_all = np.full(len(opts), np.nan)
    gamma_all = np.full(len(opts), np.nan)
    vega_all = np.full(len(opts), np.nan)

    # Group by timestamp (global_ts is not yet built here; use day+timestamp)
    groups = opts.groupby(["day", "timestamp"])

    for (day, ts), grp in groups:
        idx = grp.index
        m = grp["moneyness"].values
        iv = grp["iv"].values
        S = grp["spot"].values
        T = grp["tte"].values
        strikes = grp["strike"].values

        # Fit smile
        coeffs = fit_smile_parabola(m, iv)
        iv_hat = smile_iv_from_coeffs(m, coeffs)
        iv_hat_all[idx] = iv_hat
        iv_dev_all[idx] = iv - iv_hat

        # Theoretical price (vectorised): use smile-implied IV where available
        iv_for_theo = np.where(np.isnan(iv_hat), 0.0, iv_hat)
        has_iv = ~np.isnan(iv_hat) & (T > 0)
        theo = np.where(S > strikes, S - strikes, 0.0).astype(float)
        if has_iv.any():
            theo[has_iv] = bs_call(S[has_iv], strikes[has_iv], T[has_iv], iv_for_theo[has_iv])
        theo_all[idx] = theo
        mid_prices = opts.loc[idx, "mid_price"].values
        price_dev_all[idx] = mid_prices - theo

        # Greeks using smile IV (or actual IV if smile fails)
        iv_for_greeks = np.where(np.isnan(iv_hat), grp["iv"].values, iv_hat)
        g = bs_greeks(S, strikes, T, iv_for_greeks)
        delta_all[idx] = g["delta"]
        gamma_all[idx] = g["gamma"]
        vega_all[idx] = g["vega"]

    opts = opts.copy()
    opts["iv_hat"] = iv_hat_all
    opts["iv_dev"] = iv_dev_all
    opts["theoretical_price"] = theo_all
    opts["price_dev"] = price_dev_all
    opts["delta"] = delta_all
    opts["gamma"] = gamma_all
    opts["vega"] = vega_all
    opts["global_ts"] = opts["day"] * 1_000_000 + opts["timestamp"]

    return opts
