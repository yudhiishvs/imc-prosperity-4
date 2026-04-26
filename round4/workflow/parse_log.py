"""
Pure Python log parser — no LLM needed.
Reads IMC Prosperity submission .json logs into structured metrics.
"""
import io
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_log(path: str) -> dict:
    """Load a .json or .log submission file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    # Raw .log file — wrap it as if it were the activitiesLog field
    return {"activitiesLog": text, "profit": None}


def parse_submission_log(raw: dict) -> dict:
    """
    Parse IMC Prosperity submission JSON into structured metrics.

    Returns a dict with:
      total_pnl, products, per-product stats, trade stats,
      daily breakdown, and compact timeseries for plotting.
    """
    activities_csv = raw.get("activitiesLog", "")

    # ── Parse activities CSV ────────────────────────────────────────────────
    rows = []
    reader = csv.DictReader(io.StringIO(activities_csv), delimiter=";")
    for row in reader:
        rows.append(row)

    if not rows:
        return {"error": "No activity data in log"}

    products = sorted(set(r["product"] for r in rows if r.get("product")))
    days = sorted(set(int(r["day"]) for r in rows if r.get("day")))

    # Build per-product timeseries keyed by (product, day)
    series: dict[tuple, list] = defaultdict(list)
    for row in rows:
        try:
            series[(row["product"], int(row["day"]))].append({
                "ts":  int(row["timestamp"]),
                "pnl": float(row["profit_and_loss"]) if row.get("profit_and_loss") else 0.0,
                "mid": float(row["mid_price"]) if row.get("mid_price") else None,
                "b1":  float(row["bid_price_1"]) if row.get("bid_price_1") else None,
                "a1":  float(row["ask_price_1"]) if row.get("ask_price_1") else None,
                "bv1": float(row["bid_volume_1"]) if row.get("bid_volume_1") else None,
                "av1": float(row["ask_volume_1"]) if row.get("ask_volume_1") else None,
            })
        except (ValueError, KeyError):
            continue

    # ── Per-product stats ───────────────────────────────────────────────────
    product_stats = {}
    computed_total = 0.0

    for product in products:
        daily = {}
        prod_pnl = 0.0

        for day in days:
            ticks = series.get((product, day), [])
            if not ticks:
                continue
            end_pnl = ticks[-1]["pnl"]
            prod_pnl += end_pnl

            mids = [t["mid"] for t in ticks if t["mid"] is not None]
            spreads = [
                (t["a1"] - t["b1"])
                for t in ticks
                if t["a1"] is not None and t["b1"] is not None
            ]
            pnl_increments = [
                ticks[i]["pnl"] - ticks[i - 1]["pnl"]
                for i in range(1, len(ticks))
            ]

            daily[day] = {
                "end_pnl":    round(end_pnl, 2),
                "n_ticks":    len(ticks),
                "mid_min":    round(min(mids), 2) if mids else None,
                "mid_max":    round(max(mids), 2) if mids else None,
                "mid_range":  round(max(mids) - min(mids), 2) if mids else None,
                "avg_spread": round(sum(spreads) / len(spreads), 2) if spreads else None,
                "max_pnl_drawdown": _max_drawdown(pnl_increments),
            }

        product_stats[product] = {
            "total_pnl": round(prod_pnl, 2),
            "daily": daily,
        }
        computed_total += prod_pnl

    # ── Trade / fill stats from sandbox log ────────────────────────────────
    trade_stats = _parse_sandbox_log(raw.get("tradingLog", ""))
    position_stats = _extract_position_stats(raw.get("tradingLog", ""), products)

    # ── Compact timeseries (target ~150 pts/product) for plotting ──────────
    timeseries = {}
    for product in products:
        pts = []
        for day in days:
            ticks = series.get((product, day), [])
            # Adaptive step: aim for ≤150 points per (product, day)
            step = max(1, len(ticks) // 150)
            sampled = ticks[::step]
            for t in sampled:
                pts.append({"day": day, **t})
        timeseries[product] = pts

    return {
        "total_pnl":      round(raw.get("profit") or computed_total, 2),
        "products":       products,
        "days":           days,
        "n_products":     len(products),
        "product_stats":  product_stats,
        "trade_stats":    trade_stats,
        "position_stats": position_stats,
        "timeseries":     timeseries,
    }


def _max_drawdown(increments: list[float]) -> float:
    if not increments:
        return 0.0
    peak = 0.0
    running = 0.0
    max_dd = 0.0
    for x in increments:
        running += x
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def _parse_sandbox_log(log_str: str) -> dict:
    """
    The tradingLog is a JSON array of sandbox events from the submission.
    Each element can be:  {"sandboxLog": "...", "lambdaLog": "...", "timestamp": N, "state": {...}}
    We look inside lambdaLog for lines starting with our print() calls.
    """
    if not log_str:
        return {}

    fills_by_product: dict[str, list] = defaultdict(list)
    try:
        entries = json.loads(log_str)
        if not isinstance(entries, list):
            return {}
        for entry in entries:
            lambda_log = entry.get("lambdaLog", "")
            for line in lambda_log.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Try to parse JSON lines (structured fill logs from trader)
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "product" in obj:
                        fills_by_product[obj["product"]].append(obj)
                except json.JSONDecodeError:
                    pass
    except (json.JSONDecodeError, TypeError):
        return {}

    stats = {}
    for product, fills in fills_by_product.items():
        buys = [f for f in fills if f.get("qty", f.get("quantity", 0)) > 0]
        sells = [f for f in fills if f.get("qty", f.get("quantity", 0)) < 0]
        stats[product] = {
            "n_fills":    len(fills),
            "buy_fills":  len(buys),
            "sell_fills": len(sells),
        }
    return stats


def _extract_position_stats(log_str: str, products: list[str]) -> dict:
    """Extract max/min positions from sandbox log state snapshots."""
    if not log_str:
        return {p: {"max_long": None, "max_short": None} for p in products}

    pos_by_product: dict[str, list] = defaultdict(list)
    try:
        entries = json.loads(log_str)
        if not isinstance(entries, list):
            return {}
        for entry in entries:
            state = entry.get("state", {})
            positions = state.get("position", {})
            if isinstance(positions, dict):
                for product, pos in positions.items():
                    pos_by_product[product].append(int(pos))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    stats = {}
    for product in products:
        positions = pos_by_product.get(product, [])
        if positions:
            stats[product] = {
                "max_long":  max(positions),
                "max_short": min(positions),
                "avg_abs":   round(sum(abs(p) for p in positions) / len(positions), 1),
            }
        else:
            stats[product] = {"max_long": None, "max_short": None, "avg_abs": None}
    return stats
