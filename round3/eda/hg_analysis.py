"""
hg_analysis.py — Deep-dive on HYDROGEL_PACK strategy (v14 parameters)
======================================================================
Questions:
  1. Where does the strategy gain PnL? Where does it lose?
  2. Is the dual-EMA trend filter helping or hurting?
  3. How does position skew / quote tick affect outcomes?
  4. Imbalance signal: does it help HG quoting?
  5. Where is the EMA tracking poorly vs actual price?
  6. Regime map: flat / up-trend / down-trend — PnL per regime
"""

import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
plt.style.use("default")

DATA_DIR = Path(__file__).parent.parent / "data" / "ROUND_3"
OUT_DIR  = Path(__file__).parent / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── V14 HG parameters ──────────────────────────────────────────────────────────
HG_EMA_ALPHA   = 0.050
HG_TREND_ALPHA = 0.003
HG_TREND_GAP   = 8
HG_TAKE_EDGE   = 20
HG_TAKER_SIZE  = 8
HG_QUOTE_TICK  = 2
HG_SKEW_TICKS  = 5.0
HG_QUOTE_SIZE  = 15
HG_SOFT_LIMIT  = 60
HG_HARD_LIMIT  = 190
HG_UNWIND_SIZE = 30
HG_VOL_ALPHA   = 0.10
HG_VOL_THRESH  = 2.5
HG_VOL_SIZE    = 8
HG_EOD_TS      = 950_000
HG_LIMIT       = 200

DAY_COLORS = ["#1565C0", "#E65100", "#2E7D32"]


def ewma(prev, val, alpha):
    return float(val) if prev is None else (1.0 - alpha) * float(prev) + alpha * float(val)


def load_hg():
    frames = []
    for day in [0, 1, 2]:
        df = pd.read_csv(DATA_DIR / f"prices_round_3_day_{day}.csv", sep=";")
        df = df[df["product"] == "HYDROGEL_PACK"].copy()
        df["data_day"] = day
        df["global_ts"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("global_ts").reset_index(drop=True)


def simulate_hg(df):
    ema_val   = None
    trend_val = None
    vol_val   = None
    prev_mid  = None
    pos       = 0
    cash      = 0.0
    rows      = []

    for _, row in df.iterrows():
        ts  = row["timestamp"]
        bb  = row["bid_price_1"]
        bbv = row["bid_volume_1"]
        ba  = row["ask_price_1"]
        bav = row["ask_volume_1"]

        if pd.isna(bb) or pd.isna(ba) or bb <= 0 or ba <= 0 or bb >= ba:
            rows.append({**row.to_dict(),
                         "wmid": np.nan, "ema": ema_val, "trend_ema": trend_val,
                         "vol": vol_val, "high_vol": False, "regime": "flat",
                         "pos": pos, "cash": cash, "pnl": np.nan,
                         "fill_price": None, "fill_qty": 0, "trade_type": None})
            continue

        wmid = (bb * bav + ba * bbv) / (bbv + bav)

        if prev_mid is not None:
            vol_val = ewma(vol_val, abs(wmid - prev_mid), HG_VOL_ALPHA)
        if vol_val is None:
            vol_val = 0.0
        prev_mid = wmid

        high_vol = vol_val > HG_VOL_THRESH
        q_size   = HG_VOL_SIZE if high_vol else HG_QUOTE_SIZE

        ema_val   = ewma(ema_val,   wmid, HG_EMA_ALPHA)
        trend_val = ewma(trend_val, wmid, HG_TREND_ALPHA)

        gap          = ema_val - trend_val
        in_downtrend = gap < -HG_TREND_GAP
        in_uptrend   = gap >  HG_TREND_GAP
        regime = "DOWN" if in_downtrend else ("UP" if in_uptrend else "flat")

        fill_price = None
        fill_qty   = 0
        trade_type = None

        if ts > HG_EOD_TS and pos != 0:
            if pos > 0:
                qty = min(pos, int(bbv), max(0, HG_LIMIT - pos))
                if qty > 0:
                    cash += qty * bb; pos -= qty
                    fill_price = bb; fill_qty = -qty; trade_type = "EOD_SELL"
            else:
                qty = min(-pos, int(bav), max(0, HG_LIMIT + pos))
                if qty > 0:
                    cash -= qty * ba; pos += qty
                    fill_price = ba; fill_qty = qty; trade_type = "EOD_BUY"

        elif pos >= HG_HARD_LIMIT:
            qty = min(HG_UNWIND_SIZE, int(bbv), max(0, HG_LIMIT - pos))
            if qty > 0:
                cash += qty * bb; pos -= qty
                fill_price = bb; fill_qty = -qty; trade_type = "UNWIND_SELL"
        elif pos <= -HG_HARD_LIMIT:
            qty = min(HG_UNWIND_SIZE, int(bav), max(0, HG_LIMIT + pos))
            if qty > 0:
                cash -= qty * ba; pos += qty
                fill_price = ba; fill_qty = qty; trade_type = "UNWIND_BUY"

        elif bb >= ema_val + HG_TAKE_EDGE:
            qty = min(HG_TAKER_SIZE, int(bbv), max(0, HG_LIMIT - pos))
            if qty > 0:
                cash += qty * bb; pos -= qty
                fill_price = bb; fill_qty = -qty; trade_type = "TAKE_SELL"
        elif ba <= ema_val - HG_TAKE_EDGE:
            qty = min(HG_TAKER_SIZE, int(bav), max(0, HG_LIMIT + pos))
            if qty > 0:
                cash -= qty * ba; pos += qty
                fill_price = ba; fill_qty = qty; trade_type = "TAKE_BUY"

        rows.append({
            **row.to_dict(),
            "wmid":       wmid,
            "ema":        ema_val,
            "trend_ema":  trend_val,
            "ema_gap":    gap,
            "vol":        vol_val,
            "high_vol":   high_vol,
            "regime":     regime,
            "pos":        pos,
            "cash":       cash,
            "pnl":        cash + pos * wmid,
            "fill_price": fill_price,
            "fill_qty":   fill_qty,
            "trade_type": trade_type,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading HG price data …")
    df_raw = load_hg()
    print(f"  {len(df_raw):,} rows across 3 days")

    print("Simulating v14 HG strategy …")
    df = simulate_hg(df_raw)

    # ── PLOT A: Price + EMA + Regime ───────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))
    fig.suptitle("HG Plot A — Price, EMA, Regime Shading (v14 params)", fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub = df[df["data_day"] == day]
        ax  = axes[i]
        ax.plot(sub["timestamp"], sub["wmid"], color=DAY_COLORS[i], lw=0.9, label="wmid", alpha=0.8)
        ax.plot(sub["timestamp"], sub["ema"],  color="black", lw=1.3, label="fast EMA (α=0.05)")
        ax.plot(sub["timestamp"], sub["trend_ema"], color="purple", lw=1.0, ls="--",
                label="trend EMA (α=0.003)")

        up   = sub[sub["regime"] == "UP"]
        down = sub[sub["regime"] == "DOWN"]
        if len(up):
            for ts_v in up["timestamp"].values:
                ax.axvspan(ts_v - 50, ts_v + 50, alpha=0.08, color="green", lw=0)
        if len(down):
            for ts_v in down["timestamp"].values:
                ax.axvspan(ts_v - 50, ts_v + 50, alpha=0.08, color="red", lw=0)

        fills = sub[sub["trade_type"].notna()]
        buys  = fills[fills["fill_qty"] > 0]
        sells = fills[fills["fill_qty"] < 0]
        ax.scatter(buys["timestamp"],  buys["fill_price"],  color="blue",  s=25, zorder=5, label="BUY")
        ax.scatter(sells["timestamp"], sells["fill_price"], color="red",   s=25, zorder=5, label="SELL")

        ax.set_ylabel("Price"); ax.set_title(f"Day {day} — green=UP trend, red=DOWN trend")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Vol
    for day in [0, 1, 2]:
        sub = df[df["data_day"] == day]
        axes[3].plot(sub["timestamp"] + day * 1e6, sub["vol"],
                     color=DAY_COLORS[day], lw=0.8, label=f"Day {day}")
    axes[3].axhline(HG_VOL_THRESH, color="red", ls="--", lw=1.5, label=f"vol thresh={HG_VOL_THRESH}")
    axes[3].set_ylabel("EWMA vol"); axes[3].set_xlabel("Global timestamp")
    axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_A_price_ema_regime.png", dpi=130)
    plt.close()
    print("  [saved] hg_A_price_ema_regime.png")

    # ── PLOT B: Spread structure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("HG Plot B — Bid-Ask Spread Structure", fontsize=14, fontweight="bold")

    all_spreads = []
    for day in [0, 1, 2]:
        sub = df[df["data_day"] == day]
        sp  = (sub["ask_price_1"] - sub["bid_price_1"]).dropna()
        all_spreads.append(sp)
    combined = pd.concat(all_spreads)

    axes[0].hist(combined, bins=40, color="#1565C0", alpha=0.8, edgecolor="white")
    axes[0].axvline(combined.median(), color="red",    lw=2, label=f"median={combined.median():.0f}")
    axes[0].axvline(combined.mean(),   color="orange", lw=1.5, ls="--", label=f"mean={combined.mean():.1f}")
    axes[0].axvline(HG_QUOTE_TICK * 2 + 1, color="green", lw=1.5, ls=":",
                    label=f"min for inside (spread>{HG_QUOTE_TICK*2})")
    axes[0].set_xlabel("Bid-ask spread (ticks)"); axes[0].set_ylabel("Count")
    axes[0].set_title("All days — spread distribution"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for day in [0, 1, 2]:
        sub = df[df["data_day"] == day]
        sp  = (sub["ask_price_1"] - sub["bid_price_1"]).dropna()
        axes[1].plot(sub["timestamp"].iloc[:len(sp)], sp.values, color=DAY_COLORS[day],
                     lw=0.7, alpha=0.9, label=f"Day {day}")
    axes[1].axhline(combined.median(), color="red", lw=1.5, ls="--",
                    label=f"median={combined.median():.0f}")
    axes[1].set_xlabel("Timestamp"); axes[1].set_ylabel("Spread (ticks)")
    axes[1].set_title("Spread over time"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_B_spread.png", dpi=130)
    plt.close()
    print("  [saved] hg_B_spread.png")

    # ── PLOT C: EMA tracking error ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HG Plot C — EMA Tracking Error (wmid - fast EMA)", fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub = df[df["data_day"] == day].dropna(subset=["ema"])
        err = sub["wmid"] - sub["ema"]

        ax = axes[0][i]
        ax.plot(sub["timestamp"], err, color=DAY_COLORS[i], lw=0.8, alpha=0.9)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axhline( HG_TAKE_EDGE, color="red",   lw=1.5, ls=":", label=f"+{HG_TAKE_EDGE} TAKE sell")
        ax.axhline(-HG_TAKE_EDGE, color="green", lw=1.5, ls=":", label=f"-{HG_TAKE_EDGE} TAKE buy")
        ax.set_title(f"Day {day}: tracking error over time")
        ax.set_ylabel("Ticks error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[1][i]
        ax2.hist(err, bins=60, color=DAY_COLORS[i], alpha=0.8, density=True, edgecolor="none")
        ax2.axvline(0, color="black", lw=1, ls="--")
        ax2.axvline( HG_TAKE_EDGE, color="red",   lw=1.5, ls=":", label=f"+{HG_TAKE_EDGE} TAKE sell")
        ax2.axvline(-HG_TAKE_EDGE, color="green", lw=1.5, ls=":", label=f"-{HG_TAKE_EDGE} TAKE buy")
        pct = (abs(err) > HG_TAKE_EDGE).mean() * 100
        ax2.set_title(f"Day {day}: distribution  ({pct:.1f}% > TAKE edge)")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_C_ema_tracking.png", dpi=130)
    plt.close()
    print("  [saved] hg_C_ema_tracking.png")

    # ── PLOT D: Trend filter gap distribution ───────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HG Plot D — Trend Filter: EMA Gap Distribution", fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub = df[df["data_day"] == day].dropna(subset=["ema", "trend_ema"])
        gap = sub["ema_gap"]
        pct_up   = (gap >  HG_TREND_GAP).mean() * 100
        pct_down = (gap < -HG_TREND_GAP).mean() * 100
        pct_flat = 100 - pct_up - pct_down

        ax = axes[0][i]
        ax.plot(sub["timestamp"], gap, color=DAY_COLORS[i], lw=0.8)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axhline( HG_TREND_GAP, color="green", lw=2, ls="--",
                   label=f"UP thresh → {pct_up:.1f}% of ticks")
        ax.axhline(-HG_TREND_GAP, color="red",   lw=2, ls="--",
                   label=f"DOWN thresh → {pct_down:.1f}% of ticks")
        ax.set_title(f"Day {day}: fast_EMA - trend_EMA gap")
        ax.set_ylabel("Gap (ticks)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[1][i]
        ax2.hist(gap, bins=60, color=DAY_COLORS[i], alpha=0.8, density=True, edgecolor="none")
        ax2.axvline( HG_TREND_GAP, color="green", lw=2, ls="--")
        ax2.axvline(-HG_TREND_GAP, color="red",   lw=2, ls="--")
        ax2.axvline(0, color="black", lw=1, ls="--")
        ax2.set_title(f"Day {day}: flat={pct_flat:.0f}%  UP={pct_up:.0f}%  DOWN={pct_down:.0f}%")
        ax2.set_xlabel("Gap (ticks)"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_D_trend_filter.png", dpi=130)
    plt.close()
    print("  [saved] hg_D_trend_filter.png")

    # ── PLOT E: Imbalance → fwd return ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HG Plot E — Order Imbalance → 50-tick Forward Return", fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub = df[df["data_day"] == day].copy()
        bbv_s = sub["bid_volume_1"].fillna(0)
        bav_s = sub["ask_volume_1"].fillna(0)
        total = bbv_s + bav_s
        imb   = (bbv_s - bav_s) / total.replace(0, np.nan)
        fwd   = (sub["wmid"].shift(-50) / sub["wmid"] - 1) * 10000  # bps

        mask = imb.notna() & fwd.notna()
        x, y = imb[mask].values, fwd[mask].values

        ax = axes[i]
        ax.scatter(x, y, alpha=0.04, s=2, color=DAY_COLORS[i])
        if len(x) > 10:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(-1, 1, 100)
            ax.plot(xs, m * xs + b, color="red", lw=2.5, label=f"slope={m:.3f} bps")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Imbalance"); ax.set_ylabel("Fwd 50-tick return (bps)")
        ax.set_title(f"Day {day}"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_E_imbalance.png", dpi=130)
    plt.close()
    print("  [saved] hg_E_imbalance.png")

    # ── PLOT F: Quote placement — where do we sit in the spread? ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HG Plot F — Our Quote vs BBO (where in spread do we sit?)",
                 fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub  = df[df["data_day"] == day].dropna(subset=["ema"]).copy()
        skew  = (sub["pos"] / HG_LIMIT) * HG_SKEW_TICKS
        fair_q = sub["ema"] - skew
        bid_q  = np.floor(fair_q - HG_QUOTE_TICK)
        ask_q  = np.ceil(fair_q  + HG_QUOTE_TICK)
        bid_q  = np.minimum(bid_q, sub["ask_price_1"] - 1)
        ask_q  = np.maximum(ask_q, sub["bid_price_1"] + 1)

        bid_above_bb = bid_q - sub["bid_price_1"]   # >0 means inside spread
        ask_below_ba = sub["ask_price_1"] - ask_q   # >0 means inside spread

        ax = axes[i]
        ax.hist(bid_above_bb.clip(-2, 12), bins=20, alpha=0.7,
                color="#1565C0", label="Bid: quote - BB (>0=inside)")
        ax.hist(ask_below_ba.clip(-2, 12), bins=20, alpha=0.7,
                color="#E65100", label="Ask: BA - quote (>0=inside)")
        ax.axvline(0, color="black", lw=1.5, ls="--", label="at market")
        ax.axvline(1, color="green", lw=1.5, ls=":", label="1-tick inside")
        ax.set_xlabel("Ticks inside spread"); ax.set_title(f"Day {day}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_F_quote_placement.png", dpi=130)
    plt.close()
    print("  [saved] hg_F_quote_placement.png")

    # ── PLOT G: Large adverse move exposure ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HG Plot G — Our Position Before/After Large Price Moves",
                 fontsize=14, fontweight="bold")

    for i, day in enumerate([0, 1, 2]):
        sub = df[df["data_day"] == day].copy()
        sub["fwd50"] = sub["wmid"].shift(-50) - sub["wmid"]

        large_adv = sub[sub["fwd50"] < -15]   # price falls 15+ ticks: bad if long
        large_fav = sub[sub["fwd50"] >  15]   # price rises 15+ ticks: bad if short

        ax = axes[0][i]
        ax.scatter(sub["timestamp"], sub["wmid"], s=1, color="grey", alpha=0.3, label="price")
        if len(large_adv):
            ax.scatter(large_adv["timestamp"], large_adv["wmid"], s=12,
                       color="red", alpha=0.5, label=f"fwd -15+ ({len(large_adv)})")
        if len(large_fav):
            ax.scatter(large_fav["timestamp"], large_fav["wmid"], s=12,
                       color="green", alpha=0.5, label=f"fwd +15+ ({len(large_fav)})")
        ax.set_title(f"Day {day}: large move events on price"); ax.set_ylabel("Price")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax2 = axes[1][i]
        if len(large_adv):
            ax2.hist(large_adv["pos"], bins=20, alpha=0.7, color="red",
                     label=f"Pos before -15 fall (n={len(large_adv)})")
        if len(large_fav):
            ax2.hist(large_fav["pos"], bins=20, alpha=0.7, color="green",
                     label=f"Pos before +15 rise (n={len(large_fav)})")
        ax2.axvline(0, color="black", lw=1.5, ls="--")
        ax2.set_xlabel("Our position"); ax2.set_title(f"Day {day}: position at large moves")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "hg_G_large_moves.png", dpi=130)
    plt.close()
    print("  [saved] hg_G_large_moves.png")

    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HG KEY STATISTICS")
    print("=" * 60)
    for day in [0, 1, 2]:
        sub    = df[df["data_day"] == day]
        spread = sub["ask_price_1"] - sub["bid_price_1"]
        gap    = sub["ema_gap"].dropna()
        pct_up   = (gap >  HG_TREND_GAP).mean() * 100
        pct_down = (gap < -HG_TREND_GAP).mean() * 100
        pct_flat = 100 - pct_up - pct_down
        print(f"\nDay {day}:")
        print(f"  Spread: median={spread.median():.0f}  mean={spread.mean():.1f}  "
              f"min={spread.min():.0f}  max={spread.max():.0f}")
        print(f"  Regime: flat={pct_flat:.1f}%  UP={pct_up:.1f}%  DOWN={pct_down:.1f}%")
        print(f"  Price range: [{sub['wmid'].min():.0f}, {sub['wmid'].max():.0f}]  "
              f"std={sub['wmid'].std():.1f}")
        err = (sub["wmid"] - sub["ema"]).dropna()
        print(f"  EMA tracking error: std={err.std():.1f}  "
              f"pct > TAKE_EDGE({HG_TAKE_EDGE}): {(abs(err) > HG_TAKE_EDGE).mean()*100:.1f}%")

    # Quote placement summary
    print("\nQuote placement (all days):")
    all_bid_above = []
    for day in [0, 1, 2]:
        sub = df[df["data_day"] == day].dropna(subset=["ema"])
        skew   = (sub["pos"] / HG_LIMIT) * HG_SKEW_TICKS
        fair_q = sub["ema"] - skew
        bid_q  = np.floor(fair_q - HG_QUOTE_TICK)
        bid_q  = np.minimum(bid_q, sub["ask_price_1"] - 1)
        all_bid_above.append(bid_q - sub["bid_price_1"])
    bab = pd.concat(all_bid_above)
    print(f"  Bid ticks above BB: mean={bab.mean():.1f}  "
          f"pct inside(>0): {(bab > 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
