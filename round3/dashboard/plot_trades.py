"""
plot_trades.py — visualise order book + own fills + PnL for any backtest log.

Usage:
    python plot_trades.py [path/to/xxx.log]   # defaults to latest v9 log

Produces two figures:
  1. HYDROGEL_PACK  — price timeline, bid/ask bands, fill dots, PnL curve
  2. VEV options    — per-strike fill timeline + total options PnL

Run from the dashboard/ directory.
"""

import io
import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Load log file ─────────────────────────────────────────────────────────────

DEFAULT_LOG = Path(__file__).parent.parent / "logs" / "v9_logs" / "457353.log"

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
print(f"Loading {log_path} ...", flush=True)

with open(log_path) as f:
    data = json.load(f)

acts = pd.read_csv(io.StringIO(data["activitiesLog"]), sep=";")
acts["global_ts"] = acts["day"] * 1_000_000 + acts["timestamp"]
for col in ["bid_price_1", "ask_price_1", "mid_price", "profit_and_loss"]:
    acts[col] = pd.to_numeric(acts[col], errors="coerce")

trades_raw = pd.DataFrame(data.get("tradeHistory", []))
if not trades_raw.empty:
    trades_raw["price"]    = pd.to_numeric(trades_raw["price"],    errors="coerce")
    trades_raw["quantity"] = pd.to_numeric(trades_raw["quantity"], errors="coerce")
    trades_raw["timestamp"] = pd.to_numeric(trades_raw["timestamp"], errors="coerce")
    # day not in tradeHistory — infer from timestamp value
    # timestamps in tradeHistory are raw (0-999900); day baked into acts
    # Use activitiesLog day context (all trades are on the same day in single-day runs)
    day_val = acts["day"].iloc[0] if not acts.empty else 2
    trades_raw["global_ts"] = day_val * 1_000_000 + trades_raw["timestamp"]

# Separate own trades (SUBMISSION) from market trades
def own_trades(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty or "buyer" not in df.columns:
        return pd.DataFrame()
    s = df[df["symbol"] == symbol].copy()
    own = s[(s["buyer"] == "SUBMISSION") | (s["seller"] == "SUBMISSION")]
    own = own.copy()
    own["side"] = own.apply(
        lambda r: "BUY" if r.get("buyer") == "SUBMISSION" else "SELL", axis=1
    )
    return own.reset_index(drop=True)

# ── Compute per-trade realized P&L (FIFO) ────────────────────────────────────

def fifo_pnl(trades: pd.DataFrame) -> pd.Series:
    """
    For each trade, estimate the realized P&L using FIFO matching.
    Buys get NaN until matched with a subsequent sell (and vice-versa).
    Returns a Series aligned to trades index.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    pnl = pd.Series(np.nan, index=trades.index)
    inventory: list[tuple[float, int]] = []   # (price, qty)
    for i, row in trades.iterrows():
        qty = int(row["quantity"])
        px  = float(row["price"])
        if row["side"] == "BUY":
            inventory.append((px, qty))
        else:
            # Match sell against inventory (FIFO)
            remaining = qty
            realized  = 0.0
            new_inv   = []
            for (bp, bq) in inventory:
                if remaining <= 0:
                    new_inv.append((bp, bq))
                    continue
                matched  = min(bq, remaining)
                realized += matched * (px - bp)
                remaining -= matched
                if bq > matched:
                    new_inv.append((bp, bq - matched))
            inventory = new_inv
            pnl[i] = realized  # positive = profit
    return pnl


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: HYDROGEL_PACK
# ═══════════════════════════════════════════════════════════════════════════════

BG  = "#131722"
AX  = "#1e222d"
GRD = "#2a2e39"
TXT = "#d1d4dc"
DIM = "#787b86"

hg_acts = acts[acts["product"] == "HYDROGEL_PACK"].sort_values("global_ts").copy()
hg_ts   = hg_acts["global_ts"].values
hg_mid  = hg_acts["mid_price"].values
hg_bid  = hg_acts["bid_price_1"].values
hg_ask  = hg_acts["ask_price_1"].values
hg_pnl  = hg_acts["profit_and_loss"].values

hg_own  = own_trades(trades_raw, "HYDROGEL_PACK")
hg_pnl_series = fifo_pnl(hg_own) if not hg_own.empty else pd.Series(dtype=float)

fig1, (ax_price, ax_pnl) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
fig1.patch.set_facecolor(BG)
for ax in (ax_price, ax_pnl):
    ax.set_facecolor(AX)
    ax.grid(True, color=GRD, linewidth=0.5)
    ax.tick_params(colors=DIM)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRD)

# Price: mid line + bid/ask band
ax_price.plot(hg_ts, hg_mid, color="#d1d4dc", linewidth=1.2, label="Mid", zorder=3)
ax_price.fill_between(hg_ts, hg_bid, hg_ask, color="#2196f3", alpha=0.15, label="Bid/Ask band")
ax_price.plot(hg_ts, hg_bid, color="#2196f3", linewidth=0.6, alpha=0.6)
ax_price.plot(hg_ts, hg_ask, color="#ef5350", linewidth=0.6, alpha=0.6)

# Own fills
if not hg_own.empty:
    buys  = hg_own[hg_own["side"] == "BUY"]
    sells = hg_own[hg_own["side"] == "SELL"]

    # Color by realized P&L: green if profitable, red if loss, grey if unmatched
    def fill_colors(sub: pd.DataFrame, pnl_s: pd.Series) -> list:
        cols = []
        for i in sub.index:
            v = pnl_s.get(i, np.nan)
            if np.isnan(v):
                cols.append("#aaaaaa")
            elif v >= 0:
                cols.append("#00e676")
            else:
                cols.append("#ff1744")
        return cols

    buy_cols  = fill_colors(buys,  hg_pnl_series)
    sell_cols = fill_colors(sells, hg_pnl_series)

    ax_price.scatter(
        buys["global_ts"], buys["price"],
        marker="^", s=60, c=buy_cols, zorder=5, edgecolors="#ffffff33", linewidths=0.4,
    )
    ax_price.scatter(
        sells["global_ts"], sells["price"],
        marker="v", s=60, c=sell_cols, zorder=5, edgecolors="#ffffff33", linewidths=0.4,
    )

    # Annotate fill prices with quantity
    for _, row in pd.concat([buys, sells]).sort_values("global_ts").iterrows():
        side_sym = "▲" if row["side"] == "BUY" else "▼"
        ax_price.annotate(
            f"{side_sym}{int(row['quantity'])}@{int(row['price'])}",
            xy=(row["global_ts"], row["price"]),
            fontsize=5.5, color=TXT, alpha=0.75,
            xytext=(0, 8 if row["side"] == "BUY" else -12),
            textcoords="offset points",
        )

ax_price.set_ylabel("Price", color=DIM, fontsize=10)
ax_price.set_title(
    f"HYDROGEL_PACK  ·  {log_path.stem}  ·  final PnL = {hg_pnl[-1]:+,.0f}",
    color=TXT, fontsize=12, pad=8,
)

legend_elements = [
    Line2D([0], [0], color="#d1d4dc", label="Mid price"),
    Line2D([0], [0], color="#2196f3", alpha=0.6, label="Market bid"),
    Line2D([0], [0], color="#ef5350", alpha=0.6, label="Market ask"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#00e676", markersize=8, label="BUY fill (profit)", linestyle="None"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#ff1744", markersize=8, label="BUY fill (loss)",   linestyle="None"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#00e676", markersize=8, label="SELL fill (profit)", linestyle="None"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff1744", markersize=8, label="SELL fill (loss)",  linestyle="None"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",  markersize=8, label="Fill (open/unmatched)", linestyle="None"),
]
ax_price.legend(handles=legend_elements, facecolor=AX, edgecolor=GRD, labelcolor=TXT, fontsize=7.5, ncol=2)

# PnL subplot
ax_pnl.plot(hg_ts, hg_pnl, color="#ff9800", linewidth=1.4)
ax_pnl.axhline(0, color=GRD, linewidth=0.8, linestyle="--")
ax_pnl.fill_between(hg_ts, hg_pnl, 0,
    where=(np.array(hg_pnl) >= 0), color="#00e676", alpha=0.2)
ax_pnl.fill_between(hg_ts, hg_pnl, 0,
    where=(np.array(hg_pnl) < 0),  color="#ff1744", alpha=0.2)
ax_pnl.set_ylabel("P&L", color=DIM, fontsize=9)
ax_pnl.set_xlabel("Global timestamp (day×1M + ts)", color=DIM, fontsize=9)

plt.tight_layout()
out1 = log_path.parent / f"{log_path.stem}_hg_trades.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out1}")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: VEV Options fills
# ═══════════════════════════════════════════════════════════════════════════════

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
ACTIVE = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]

# Collect per-strike own trades
opt_fills = []
for sym in ACTIVE:
    sym_own = own_trades(trades_raw, sym)
    if sym_own.empty:
        continue
    pnl_s = fifo_pnl(sym_own)
    for i, row in sym_own.iterrows():
        opt_fills.append({
            "sym": sym,
            "strike": VEV_STRIKES[sym],
            "global_ts": row["global_ts"],
            "price": row["price"],
            "quantity": row["quantity"],
            "side": row["side"],
            "realized_pnl": pnl_s.get(i, np.nan),
        })

opt_df = pd.DataFrame(opt_fills)

# Overall options PnL per symbol
opt_pnl_by_sym = {}
for sym in VEV_STRIKES:
    a = acts[acts["product"] == sym]
    if not a.empty:
        opt_pnl_by_sym[sym] = a["profit_and_loss"].iloc[-1]

fig2, axes = plt.subplots(
    len(ACTIVE) + 1, 1, figsize=(14, 2.5 * (len(ACTIVE) + 1)),
    sharex=True,
)
fig2.patch.set_facecolor(BG)

colors_strike = plt.cm.tab10(np.linspace(0, 1, len(ACTIVE)))

for ax_i, (sym, col) in enumerate(zip(ACTIVE, colors_strike)):
    ax = axes[ax_i]
    ax.set_facecolor(AX)
    ax.grid(True, color=GRD, linewidth=0.5)
    ax.tick_params(colors=DIM)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRD)

    sym_acts = acts[acts["product"] == sym].sort_values("global_ts")
    if not sym_acts.empty:
        ax.plot(sym_acts["global_ts"], sym_acts["mid_price"],
                color=col, linewidth=1.0, alpha=0.8)
        ax.fill_between(sym_acts["global_ts"],
                        sym_acts["bid_price_1"], sym_acts["ask_price_1"],
                        color=col, alpha=0.12)

    sym_fills = opt_df[opt_df["sym"] == sym] if not opt_df.empty else pd.DataFrame()
    final_pnl = opt_pnl_by_sym.get(sym, 0.0)

    if not sym_fills.empty:
        for _, row in sym_fills.iterrows():
            marker = "^" if row["side"] == "BUY" else "v"
            v = row["realized_pnl"]
            fc = "#aaaaaa" if np.isnan(v) else ("#00e676" if v >= 0 else "#ff1744")
            ax.scatter(row["global_ts"], row["price"],
                       marker=marker, s=80, color=fc, zorder=5, edgecolors="#fff3")
            ax.annotate(
                f"{'▲' if row['side']=='BUY' else '▼'}{int(row['quantity'])}@{int(row['price'])}",
                xy=(row["global_ts"], row["price"]),
                fontsize=5.5, color=TXT, alpha=0.8,
                xytext=(0, 8 if row["side"] == "BUY" else -12),
                textcoords="offset points",
            )

    n_fills = len(sym_fills) if not sym_fills.empty else 0
    ax.set_ylabel(f"{sym}\n(pnl={final_pnl:+.0f}, {n_fills} fills)", color=col, fontsize=8)

# Bottom subplot: total options PnL
ax_opt_pnl = axes[-1]
ax_opt_pnl.set_facecolor(AX)
ax_opt_pnl.grid(True, color=GRD, linewidth=0.5)
ax_opt_pnl.tick_params(colors=DIM)
for sp in ax_opt_pnl.spines.values():
    sp.set_edgecolor(GRD)

total_opt_pnl = sum(opt_pnl_by_sym.get(s, 0) for s in ACTIVE)
# Plot cumulative options PnL from activitiesLog
opt_pnl_ts = None
for sym in ACTIVE:
    a = acts[acts["product"] == sym].sort_values("global_ts")[["global_ts", "profit_and_loss"]]
    if a.empty:
        continue
    if opt_pnl_ts is None:
        opt_pnl_ts = a.set_index("global_ts").rename(columns={"profit_and_loss": sym})
    else:
        opt_pnl_ts = opt_pnl_ts.join(a.set_index("global_ts").rename(columns={"profit_and_loss": sym}), how="outer")

if opt_pnl_ts is not None:
    opt_pnl_ts = opt_pnl_ts.ffill().fillna(0)
    opt_pnl_ts["total"] = opt_pnl_ts.sum(axis=1)
    ax_opt_pnl.plot(opt_pnl_ts.index, opt_pnl_ts["total"],
                    color="#ff9800", linewidth=1.4, label=f"Options total P&L (final={total_opt_pnl:+.0f})")
    ax_opt_pnl.axhline(0, color=GRD, linewidth=0.8, linestyle="--")
    ax_opt_pnl.fill_between(opt_pnl_ts.index, opt_pnl_ts["total"], 0,
        where=(opt_pnl_ts["total"].values >= 0), color="#00e676", alpha=0.2)
    ax_opt_pnl.fill_between(opt_pnl_ts.index, opt_pnl_ts["total"], 0,
        where=(opt_pnl_ts["total"].values < 0),  color="#ff1744", alpha=0.2)
ax_opt_pnl.set_ylabel(f"Options PnL", color=DIM, fontsize=8)
ax_opt_pnl.set_xlabel("Global timestamp", color=DIM, fontsize=9)
ax_opt_pnl.legend(facecolor=AX, edgecolor=GRD, labelcolor=TXT, fontsize=8)

fig2.suptitle(f"VEV Options Fills  ·  {log_path.stem}", color=TXT, fontsize=12, y=1.002)
plt.tight_layout()
out2 = log_path.parent / f"{log_path.stem}_opt_trades.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out2}")
plt.show()
