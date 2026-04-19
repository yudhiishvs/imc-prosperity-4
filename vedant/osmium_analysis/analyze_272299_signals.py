"""
analyze_272299_signals.py — Compare OIM vs Last-Tick-Change as predictive signals
for Osmium price movements. Uses 272299 live round data.

Goal: Determine which signal (or combination) is most predictive for fading.
"""

import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "272299" / "272299.log"

def load_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_activities(raw_csv, product):
    rows = raw_csv.strip().split("\n")
    header = rows[0].split(";")
    records = []
    for row in rows[1:]:
        parts = row.split(";")
        if len(parts) < len(header):
            continue
        r = dict(zip(header, parts))
        if r.get("product", "").strip() != product:
            continue
        def fv(k):
            v = r.get(k, "").strip()
            return float(v) if v else None
        records.append({
            "ts": int(r.get("timestamp", 0)),
            "bid1": fv("bid_price_1"), "bid1v": fv("bid_volume_1"),
            "ask1": fv("ask_price_1"), "ask1v": fv("ask_volume_1"),
            "mid": fv("mid_price"),
        })
    return records


def section(title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def pct(n, d):
    return f"{100*n/max(1,d):.1f}%"


def main():
    data = load_log(LOG_PATH)
    acts = parse_activities(data["activitiesLog"], "ASH_COATED_OSMIUM")

    # Only use two-sided ticks for reliable analysis
    clean = [r for r in acts if r["bid1"] is not None and r["ask1"] is not None and r["mid"] is not None]
    print(f"Clean two-sided ticks: {len(clean)}")

    # ─────────────────────────────────────────────────────────────
    # SIGNAL 1: LAST-TICK CHANGE (lagging)
    # ─────────────────────────────────────────────────────────────
    section("SIGNAL 1: LAST-TICK CHANGE (Lagging Momentum Fade)")

    for horizon in [1, 2, 3, 5, 10]:
        correct = wrong = neither = 0
        returns = []
        for i in range(1, len(clean)):
            change = clean[i]["mid"] - clean[i-1]["mid"]
            if change == 0:
                continue
            fade_dir = -1 if change > 0 else 1  # fade = expect reversal

            if i + horizon < len(clean):
                future_move = clean[i + horizon]["mid"] - clean[i]["mid"]
                returns.append(future_move * fade_dir)  # positive = fade was right
                if future_move * fade_dir > 0:
                    correct += 1
                elif future_move * fade_dir < 0:
                    wrong += 1
                else:
                    neither += 1

        total = correct + wrong
        avg_ret = sum(returns) / len(returns) if returns else 0
        print(f"\n  Horizon={horizon} ticks:")
        print(f"    Signals: {total} (non-zero change)")
        print(f"    Correct: {correct} ({pct(correct, total)})  Wrong: {wrong} ({pct(wrong, total)})")
        print(f"    Avg fade return: {avg_ret:+.4f} ticks")
        print(f"    {'✅ Profitable fade' if avg_ret > 0 else '❌ Unprofitable fade'}")

    # ─────────────────────────────────────────────────────────────
    # SIGNAL 2: OIM (Leading)
    # ─────────────────────────────────────────────────────────────
    section("SIGNAL 2: ORDER IMBALANCE (OIM) — Leading Indicator")

    for horizon in [1, 2, 3, 5, 10]:
        correct = wrong = neither = 0
        returns = []
        oim_magnitude_returns = []

        for i in range(len(clean)):
            r = clean[i]
            bid_vol = r["bid1v"] or 0
            ask_vol = r["ask1v"] or 0
            total_vol = bid_vol + ask_vol
            if total_vol == 0:
                continue
            oim = (bid_vol - ask_vol) / total_vol  # +1 = all bids, -1 = all asks

            if abs(oim) < 0.05:  # filter noise
                continue

            # OIM > 0 (bid heavy) → expect price to move UP → we should BUY (or fade asks)
            # OIM < 0 (ask heavy) → expect price to move DOWN → we should SELL (or fade bids)
            expected_dir = 1 if oim > 0 else -1

            if i + horizon < len(clean):
                future_move = clean[i + horizon]["mid"] - clean[i]["mid"]
                aligned_return = future_move * expected_dir  # positive = OIM was right
                returns.append(aligned_return)
                oim_magnitude_returns.append((abs(oim), aligned_return))

                if aligned_return > 0:
                    correct += 1
                elif aligned_return < 0:
                    wrong += 1
                else:
                    neither += 1

        total = correct + wrong
        avg_ret = sum(returns) / len(returns) if returns else 0
        print(f"\n  Horizon={horizon} ticks:")
        print(f"    Signals: {total} (|OIM| > 0.05)")
        print(f"    Correct: {correct} ({pct(correct, total)})  Wrong: {wrong} ({pct(wrong, total)})")
        print(f"    Avg OIM-aligned return: {avg_ret:+.4f} ticks")
        print(f"    {'✅ OIM is predictive' if avg_ret > 0 else '❌ OIM is NOT predictive (fade OIM instead?)'}")

        # Breakdown by OIM strength
        if horizon == 1:
            print(f"\n    OIM strength breakdown (horizon=1):")
            buckets = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            for lo, hi in buckets:
                subset = [r for oim_abs, r in oim_magnitude_returns if lo <= oim_abs < hi]
                if subset:
                    avg = sum(subset) / len(subset)
                    pos = sum(1 for r in subset if r > 0)
                    print(f"      |OIM| [{lo:.2f}, {hi:.2f}): n={len(subset):>5}  avg={avg:+.4f}  correct={pct(pos, len(subset))}")

    # ─────────────────────────────────────────────────────────────
    # SIGNAL 3: OIM as FADE signal (inverse: sell into bid-heavy)
    # ─────────────────────────────────────────────────────────────
    section("SIGNAL 2b: OIM as CONTRARIAN/FADE signal")
    print("  (What if we FADE OIM instead of following it?)")

    for horizon in [1, 2, 3, 5, 10]:
        returns = []
        correct = wrong = 0
        for i in range(len(clean)):
            r = clean[i]
            bid_vol = r["bid1v"] or 0
            ask_vol = r["ask1v"] or 0
            total_vol = bid_vol + ask_vol
            if total_vol == 0:
                continue
            oim = (bid_vol - ask_vol) / total_vol
            if abs(oim) < 0.05:
                continue

            # FADE: OIM > 0 → expect reversion DOWN, OIM < 0 → expect UP
            fade_dir = -1 if oim > 0 else 1

            if i + horizon < len(clean):
                future_move = clean[i + horizon]["mid"] - clean[i]["mid"]
                returns.append(future_move * fade_dir)
                if future_move * fade_dir > 0:
                    correct += 1
                elif future_move * fade_dir < 0:
                    wrong += 1

        total = correct + wrong
        avg_ret = sum(returns) / len(returns) if returns else 0
        print(f"\n  Horizon={horizon}: avg fade return={avg_ret:+.4f}  correct={pct(correct, total)}  n={total}")

    # ─────────────────────────────────────────────────────────────
    # SIGNAL 3: COMBINED (OIM + Last-Tick Change)
    # ─────────────────────────────────────────────────────────────
    section("SIGNAL 3: COMBINED — OIM + Last-Tick-Change")

    for horizon in [1, 3, 5]:
        # Case A: Both signals agree (OIM momentum + tick momentum fade → same direction)
        # This means: OIM says price going UP, but last tick went UP too → fade says go DOWN
        # → signals DISAGREE
        # 
        # Actually let's test: when OIM direction MATCHES fade direction vs when they CONFLICT

        agree_returns = []
        conflict_returns = []
        oim_only_returns = []
        fade_only_returns = []

        for i in range(1, len(clean)):
            r = clean[i]
            bid_vol = r["bid1v"] or 0
            ask_vol = r["ask1v"] or 0
            total_vol = bid_vol + ask_vol
            change = r["mid"] - clean[i-1]["mid"]

            if total_vol == 0 or change == 0:
                continue

            oim = (bid_vol - ask_vol) / total_vol
            if abs(oim) < 0.05:
                continue

            oim_dir = 1 if oim > 0 else -1      # OIM says: follow this direction
            fade_dir = -1 if change > 0 else 1   # Fade says: go opposite to change

            if i + horizon >= len(clean):
                continue
            future_move = clean[i + horizon]["mid"] - r["mid"]

            # Which signal was right?
            oim_right = future_move * oim_dir
            fade_right = future_move * fade_dir

            oim_only_returns.append(oim_right)
            fade_only_returns.append(fade_right)

            if oim_dir == fade_dir:
                # Both agree
                agree_returns.append(future_move * oim_dir)
            else:
                # They conflict - which wins?
                conflict_returns.append({"oim": oim_right, "fade": fade_right, "oim_dir": oim_dir, "fade_dir": fade_dir})

        print(f"\n  Horizon={horizon}:")
        print(f"    Total signals (both non-zero): {len(oim_only_returns)}")

        if oim_only_returns:
            oim_avg = sum(oim_only_returns) / len(oim_only_returns)
            fade_avg = sum(fade_only_returns) / len(fade_only_returns)
            print(f"    OIM-follow avg return:  {oim_avg:+.4f}")
            print(f"    Fade avg return:        {fade_avg:+.4f}")

        if agree_returns:
            agree_avg = sum(agree_returns) / len(agree_returns)
            agree_correct = sum(1 for r in agree_returns if r > 0)
            print(f"\n    AGREE (OIM + fade same direction): n={len(agree_returns)}")
            print(f"      Avg return: {agree_avg:+.4f}  Correct: {pct(agree_correct, len(agree_returns))}")

        if conflict_returns:
            oim_wins = sum(1 for c in conflict_returns if c["oim"] > 0 and c["fade"] < 0)
            fade_wins = sum(1 for c in conflict_returns if c["fade"] > 0 and c["oim"] < 0)
            both_wrong = sum(1 for c in conflict_returns if c["fade"] <= 0 and c["oim"] <= 0)
            both_right = sum(1 for c in conflict_returns if c["fade"] > 0 and c["oim"] > 0)
            oim_avg_c = sum(c["oim"] for c in conflict_returns) / len(conflict_returns)
            fade_avg_c = sum(c["fade"] for c in conflict_returns) / len(conflict_returns)
            print(f"\n    CONFLICT (OIM vs fade disagree): n={len(conflict_returns)}")
            print(f"      OIM wins: {oim_wins} ({pct(oim_wins, len(conflict_returns))})")
            print(f"      Fade wins: {fade_wins} ({pct(fade_wins, len(conflict_returns))})")
            print(f"      Both wrong: {both_wrong}  Both right: {both_right}")
            print(f"      OIM avg return in conflict: {oim_avg_c:+.4f}")
            print(f"      Fade avg return in conflict: {fade_avg_c:+.4f}")

    # ─────────────────────────────────────────────────────────────
    # EXTRA: OIM shifted by 1 tick (does OIM at t predict move from t+1 to t+2?)
    # ─────────────────────────────────────────────────────────────
    section("SIGNAL 4: SHIFTED OIM — Does OIM at t-1 predict move from t to t+N?")
    print("  (Leading indicator test: does OIM at the PREVIOUS tick predict the NEXT move?)")

    for horizon in [1, 2, 3, 5]:
        returns = []
        correct = wrong = 0
        for i in range(1, len(clean)):
            # Use OIM from PREVIOUS tick
            r_prev = clean[i-1]
            bid_vol = r_prev["bid1v"] or 0
            ask_vol = r_prev["ask1v"] or 0
            total_vol = bid_vol + ask_vol
            if total_vol == 0:
                continue
            oim = (bid_vol - ask_vol) / total_vol
            if abs(oim) < 0.05:
                continue

            oim_dir = 1 if oim > 0 else -1

            if i + horizon < len(clean):
                future_move = clean[i + horizon]["mid"] - clean[i]["mid"]
                ret = future_move * oim_dir
                returns.append(ret)
                if ret > 0:
                    correct += 1
                elif ret < 0:
                    wrong += 1

        total = correct + wrong
        avg = sum(returns) / len(returns) if returns else 0
        print(f"\n  OIM(t-1) → move(t, t+{horizon}): avg={avg:+.4f}  correct={pct(correct, total)}  n={total}")
        if avg > 0:
            print(f"    ✅ Previous-tick OIM IS a leading indicator at {horizon}-tick horizon")
        else:
            print(f"    ❌ Previous-tick OIM is NOT a leading indicator at {horizon}-tick horizon")


if __name__ == "__main__":
    main()
