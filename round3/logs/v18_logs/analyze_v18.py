"""
v18 log analysis — trades vs market prices, PnL by product, missed edge.
"""
import json, csv, io, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/v18_logs/478449.log"
OUT = "/Users/visheshng/Documents/Code/UMDClubs/Apex/IMC-Prosperity/prosperity4/imc-prosperity-4/round3/logs/v18_logs/"

with open(LOG) as f:
    d = json.load(f)

# ── Activities log: mid_price + PnL timeseries ────────────────────────────────
acts = list(csv.DictReader(io.StringIO(d["activitiesLog"]), delimiter=";"))
print(f"Activity rows: {len(acts)}")

mid_ts   = defaultdict(list)   # product → [(ts, mid)]
pnl_ts   = defaultdict(list)   # product → [(ts, pnl)]
spread_ts= defaultdict(list)   # product → [(ts, spread)]

for row in acts:
    prod = row["product"].strip()
    ts   = int(row["timestamp"])
    mid  = float(row["mid_price"])
    pnl  = float(row["profit_and_loss"])
    try:
        b1 = float(row["bid_price_1"]) if row["bid_price_1"].strip() else None
        a1 = float(row["ask_price_1"]) if row["ask_price_1"].strip() else None
        spd = (a1 - b1) if (b1 and a1) else None
    except (ValueError, KeyError):
        spd = None
    mid_ts[prod].append((ts, mid))
    pnl_ts[prod].append((ts, pnl))
    if spd is not None:
        spread_ts[prod].append((ts, spd))

products = sorted(mid_ts.keys())
print("Products:", products)

# ── Trade history ─────────────────────────────────────────────────────────────
trades = d.get("tradeHistory", [])
our_trades  = [t for t in trades if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]
mkt_trades  = [t for t in trades if t.get("buyer") != "SUBMISSION" and t.get("seller") != "SUBMISSION"]

print(f"Total trades: {len(trades)} | Ours: {len(our_trades)} | Market: {len(mkt_trades)}")

# Group by symbol
our_by_sym = defaultdict(list)
mkt_by_sym = defaultdict(list)
for t in our_trades:
    our_by_sym[t["symbol"]].append(t)
for t in mkt_trades:
    mkt_by_sym[t["symbol"]].append(t)

# ── Lambda logs: extract algo print lines ─────────────────────────────────────
algo_lines = []   # (ts, line)
for entry in d.get("logs", []):
    ts  = entry.get("timestamp", 0)
    raw = entry.get("lambdaLog", "")
    if not raw:
        continue
    try:
        parsed = json.loads(raw)
        # lambdaLog is a JSON list; last string element is algo stdout
        algo_str = ""
        for elem in parsed:
            if isinstance(elem, str) and ("[HG" in elem or "[OPT" in elem or "[VF" in elem):
                algo_str = elem
        if algo_str:
            for line in algo_str.split("\n"):
                line = line.strip()
                if line:
                    algo_lines.append((ts, line))
    except Exception:
        pass

print(f"Algo log lines parsed: {len(algo_lines)}")

# Count tag types
tag_counts = defaultdict(int)
for _, line in algo_lines:
    m = re.match(r'\[(\w[\w ]*?)\]', line)
    if m:
        tag_counts[m.group(1)] += 1
print("Tag distribution:", dict(tag_counts))

# ── Final PnL per product ─────────────────────────────────────────────────────
print("\n── PnL at final tick ──")
total_pnl = 0.0
for prod in sorted(pnl_ts.keys()):
    final = pnl_ts[prod][-1][1]
    n_trades = len(our_by_sym.get(prod, []))
    print(f"  {prod:30s}  PnL={final:+9.1f}   our_trades={n_trades}")
    total_pnl += final
print(f"  {'TOTAL':30s}  PnL={total_pnl:+9.1f}")

# ── Trade fill quality: our fill price vs mid at same timestamp ───────────────
print("\n── Fill quality (fill_price vs mid_price at same tick) ──")
fill_edge = defaultdict(list)
for sym, tlist in our_by_sym.items():
    mid_lookup = {ts: mid for ts, mid in mid_ts.get(sym, [])}
    for t in tlist:
        ts    = t["timestamp"]
        price = t["price"]
        qty   = t["quantity"]
        side  = "BUY" if t["buyer"] == "SUBMISSION" else "SELL"
        mid   = mid_lookup.get(ts)
        if mid is None:
            # find nearest
            arr = mid_ts.get(sym, [])
            if arr:
                closest = min(arr, key=lambda x: abs(x[0]-ts))
                mid = closest[1]
        if mid is not None:
            edge = (mid - price) if side == "BUY" else (price - mid)
            fill_edge[sym].append((ts, edge, side, price, mid, qty))

for sym in sorted(fill_edge.keys()):
    edges = [e[1] for e in fill_edge[sym]]
    print(f"  {sym:25s}  fills={len(edges):3d}  avg_edge={np.mean(edges):+.2f}  "
          f"min={min(edges):+.2f}  max={max(edges):+.2f}")

# ── Market trades we DIDN'T participate in (missed edge) ──────────────────────
# A market trade at ts where we had no fill = potential missed edge
print("\n── Market trades we missed (bot vs bot, we were passive) ──")
for sym in sorted(mkt_by_sym.keys()):
    our_ts = {t["timestamp"] for t in our_by_sym.get(sym, [])}
    missed = [t for t in mkt_by_sym[sym] if t["timestamp"] not in our_ts]
    if missed:
        mid_lookup = dict(mid_ts.get(sym, []))
        edges = []
        for t in missed:
            mid = mid_lookup.get(t["timestamp"])
            if mid:
                edges.append(abs(t["price"] - mid))
        print(f"  {sym:25s}  missed={len(missed):3d}  "
              f"avg_spread_from_mid={np.mean(edges):.2f}" if edges else
              f"  {sym:25s}  missed={len(missed):3d}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
VEV_STRIKES = ["VEV_4500","VEV_5000","VEV_5100","VEV_5200","VEV_5300","VEV_5400","VEV_5500"]
VEV_COLORS  = {"VEV_4500":"#9467bd","VEV_5000":"#8c564b","VEV_5100":"#e377c2",
               "VEV_5200":"#7f7f7f","VEV_5300":"#bcbd22","VEV_5400":"#17becf",
               "VEV_5500":"#aec7e8"}

# ── Plot 1: PnL timeseries ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=False)
fig.suptitle("v18 — PnL Timeseries by Product Group", fontsize=13, fontweight="bold")

# HG
ax = axes[0]
ts_arr, pnl_arr = zip(*pnl_ts["HYDROGEL_PACK"]) if "HYDROGEL_PACK" in pnl_ts else ([],[])
ax.plot(ts_arr, pnl_arr, color="steelblue", lw=1.5)
ax.set_title(f"HYDROGEL_PACK   final={pnl_arr[-1] if pnl_arr else 0:+.0f}")
ax.grid(alpha=0.3); ax.set_ylabel("PnL")

# VF Extract
ax = axes[1]
ts_arr, pnl_arr = zip(*pnl_ts.get("VELVETFRUIT_EXTRACT",[(0,0)]))
ax.plot(ts_arr, pnl_arr, color="crimson", lw=1.5)
ax.set_title(f"VELVETFRUIT_EXTRACT   final={pnl_arr[-1]:+.0f}")
ax.grid(alpha=0.3); ax.set_ylabel("PnL")

# Options combined
ax = axes[2]
opt_pnl = defaultdict(float)
for prod, rows in pnl_ts.items():
    if prod.startswith("VEV_"):
        for ts, pnl in rows:
            opt_pnl[ts] += pnl
if opt_pnl:
    ts_s = sorted(opt_pnl.keys())
    ax.plot(ts_s, [opt_pnl[t] for t in ts_s], color="darkorange", lw=1.5)
    ax.set_title(f"Options combined   final={opt_pnl[ts_s[-1]]:+.0f}")
ax.grid(alpha=0.3); ax.set_ylabel("PnL"); ax.set_xlabel("Timestamp")

fig.tight_layout()
fig.savefig(OUT + "01_pnl_timeseries.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nSaved 01_pnl_timeseries.png")

# ── Plot 2: Our trades vs mid-price for each option strike ────────────────────
n_cols = 4
n_rows = (len(VEV_STRIKES) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()
fig.suptitle("v18 — Our Fills vs Mid-Price (per option strike)", fontsize=13, fontweight="bold")

for i, sym in enumerate(VEV_STRIKES):
    ax = axes[i]
    color = VEV_COLORS.get(sym, "gray")

    # Mid price line
    if sym in mid_ts:
        ts_m, mid_m = zip(*mid_ts[sym])
        ax.plot(ts_m, mid_m, color=color, lw=1.2, alpha=0.7, label="Mid")

    # Our fills
    buys  = [(t["timestamp"], t["price"], t["quantity"]) for t in our_by_sym.get(sym, []) if t["buyer"] == "SUBMISSION"]
    sells = [(t["timestamp"], t["price"], t["quantity"]) for t in our_by_sym.get(sym, []) if t["seller"] == "SUBMISSION"]
    if buys:
        bts, bpx, bqt = zip(*buys)
        ax.scatter(bts, bpx, marker="^", color="green", s=[q*4 for q in bqt], zorder=5, label=f"Buy ({len(buys)})")
    if sells:
        sts, spx, sqt = zip(*sells)
        ax.scatter(sts, spx, marker="v", color="red", s=[q*4 for q in sqt], zorder=5, label=f"Sell ({len(sells)})")

    # Market (bot) trades we missed
    missed = [t for t in mkt_by_sym.get(sym, [])]
    if missed:
        mts = [t["timestamp"] for t in missed]
        mpx = [t["price"] for t in missed]
        ax.scatter(mts, mpx, marker="x", color="gray", s=15, alpha=0.4, zorder=3, label=f"Mkt ({len(missed)})")

    final_pnl = pnl_ts[sym][-1][1] if sym in pnl_ts else 0
    ax.set_title(f"{sym}  PnL={final_pnl:+.0f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Timestamp", fontsize=7)
    ax.set_ylabel("Price", fontsize=7)

# Hide unused axes
for j in range(len(VEV_STRIKES), len(axes)):
    axes[j].set_visible(False)

fig.tight_layout()
fig.savefig(OUT + "02_trades_vs_mid_options.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 02_trades_vs_mid_options.png")

# ── Plot 3: HG — mid-price + our fills ───────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 9))
fig.suptitle("v18 — HYDROGEL_PACK: Price + Fills + PnL", fontsize=13, fontweight="bold")

ax = axes[0]
sym = "HYDROGEL_PACK"
if sym in mid_ts:
    ts_m, mid_m = zip(*mid_ts[sym])
    ax.plot(ts_m, mid_m, color="steelblue", lw=1.2, label="Mid")
buys  = [(t["timestamp"], t["price"], t["quantity"]) for t in our_by_sym.get(sym, []) if t["buyer"] == "SUBMISSION"]
sells = [(t["timestamp"], t["price"], t["quantity"]) for t in our_by_sym.get(sym, []) if t["seller"] == "SUBMISSION"]
if buys:
    bts, bpx, bqt = zip(*buys)
    ax.scatter(bts, bpx, marker="^", color="green", s=[q*5 for q in bqt], zorder=5, label=f"Buy ({len(buys)})")
if sells:
    sts, spx, sqt = zip(*sells)
    ax.scatter(sts, spx, marker="v", color="red", s=[q*5 for q in sqt], zorder=5, label=f"Sell ({len(sells)})")
ax.set_ylabel("Price"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
if sym in pnl_ts:
    ts_p, pnl_p = zip(*pnl_ts[sym])
    ax.plot(ts_p, pnl_p, color="steelblue", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("PnL"); ax.set_xlabel("Timestamp"); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT + "03_hg_trades.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 03_hg_trades.png")

# ── Plot 4: spread distribution per option ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("v18 — Option Spread Distribution (how much edge is available)", fontsize=12)

all_spreads = {}
for sym in VEV_STRIKES:
    if sym in spread_ts:
        all_spreads[sym] = [s for _, s in spread_ts[sym]]

labels = list(all_spreads.keys())
data   = [all_spreads[l] for l in labels]
bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
for patch, sym in zip(bp["boxes"], labels):
    patch.set_facecolor(VEV_COLORS.get(sym, "gray"))
    patch.set_alpha(0.7)
ax.set_ylabel("Bid-ask spread (ticks)")
ax.axhline(3, color="red", ls="--", lw=1.5, label="Min spread=3 (quote threshold)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT + "04_spread_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 04_spread_distribution.png")

# ── Plot 5: fill edge distribution per product ────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle("v18 — Fill Edge (fill_price - mid at fill time) per product", fontsize=12)
all_edges = {sym: [e[1] for e in edges] for sym, edges in fill_edge.items() if edges}
if all_edges:
    labels = sorted(all_edges.keys())
    bp = ax.boxplot([all_edges[l] for l in labels], labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.7)
    ax.axhline(0, color="red", ls="--", lw=1.5)
    ax.set_ylabel("Edge (positive = bought below mid / sold above mid)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT + "05_fill_edge.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved 05_fill_edge.png")

print("\n=== DONE ===")
