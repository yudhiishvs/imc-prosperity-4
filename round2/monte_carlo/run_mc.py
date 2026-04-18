"""
run_mc.py  (Round 2)
---------------------
Monte Carlo runner: generates N synthetic days, runs prosperity4btx on each,
collects PnL and identifies adversarial market regimes.

Adversarial detection
---------------------
After all paths, the script correlates per-day regime metadata with PnL:
  - High liquidator activity  → should HELP PnL (we capture spread)
  - High reverter activity    → can HURT PnL if our quotes lag fair value
  - High imbalance            → directional risk, hurts market-maker if uncorrected
  - High deviation from fair  → adversarial for ASH if we're slow to mean-revert

Usage:
    python run_mc.py <path_to_algo.py> [--n-days 1000] [--seed 42] [--regen]

Example:
    python run_mc.py ../../algos/v1_round2.py --n-days 1000 --seed 42
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MC_DATA     = os.path.join(SCRIPT_DIR, "mc_data")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "params.json")
BACKTESTER  = "/opt/homebrew/bin/prosperity4btx"
ROUND_NUM   = 99

PRODUCT_RE = re.compile(r"^([\w_]+):\s+([\-\d,]+)$")
TOTAL_RE   = re.compile(r"^Total profit:\s+([\-\d,]+)$")

PRODUCTS   = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]


# ---------------------------------------------------------------------------
# Backtester runner
# ---------------------------------------------------------------------------

def parse_pnl(line):
    m = PRODUCT_RE.match(line.strip())
    if m:
        return m.group(1), int(m.group(2).replace(",", ""))
    m = TOTAL_RE.match(line.strip())
    if m:
        return "Total", int(m.group(1).replace(",", ""))
    return None


def run_day(algo_path, day):
    cmd = [
        BACKTESTER, algo_path,
        f"{ROUND_NUM}-{day}",
        "--data", MC_DATA,
        "--no-out",
        "--no-progress",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    pnls = {}
    for line in result.stdout.splitlines():
        parsed = parse_pnl(line)
        if parsed:
            pnls[parsed[0]] = parsed[1]
    return pnls


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def stats(values):
    if not values:
        return {}
    n    = len(values)
    mean = sum(values) / n
    var  = sum((v - mean)**2 for v in values) / max(1, n - 1)
    std  = var**0.5
    srt  = sorted(values)
    return {
        "n":      n,
        "mean":   mean,
        "std":    std,
        "sharpe": mean / std if std > 0 else 0.0,
        "min":    srt[0],
        "p5":     srt[max(0, int(0.05 * n))],
        "median": srt[n // 2],
        "p95":    srt[min(n-1, int(0.95 * n))],
        "max":    srt[-1],
    }


def ascii_histogram(values, bins=20, width=50):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  All values = {lo:,.0f}"
    bin_size = (hi - lo) / bins
    counts   = [0] * bins
    for v in values:
        counts[min(int((v - lo) / bin_size), bins - 1)] += 1
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(c / max_c * width)
        lines.append(f"  {lo + i*bin_size:>10,.0f} | {bar:<{width}} {c}")
    return "\n".join(lines)


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx  = sum((x - mx)**2 for x in xs)**0.5
    vy  = sum((y - my)**2 for y in ys)**0.5
    return cov / (vx * vy) if vx * vy > 0 else 0.0


# ---------------------------------------------------------------------------
# Adversarial path analysis
# ---------------------------------------------------------------------------

def analyze_adversarial(all_results, n_days):
    """
    Correlate regime metadata (from _meta.json sidescar) with PnL outcomes.
    Prints a ranked table of which market conditions hurt PnL most.
    """
    print("\n" + "=" * 70)
    print("ADVERSARIAL PATH ANALYSIS")
    print("=" * 70)

    meta_dir = os.path.join(MC_DATA, "round99")
    totals   = [r.get("Total", 0) for r in all_results]

    features = {
        "ASH liq_total":    [],
        "ASH rev_total":    [],
        "ASH imb_mean":     [],
        "ASH imb_std":      [],
        "ASH dev_std":      [],
        "ASH dev_abs_max":  [],
        "PEP liq_total":    [],
        "PEP rev_total":    [],
        "PEP imb_mean":     [],
        "PEP dev_std":      [],
        "PEP dev_abs_max":  [],
    }

    for d in range(n_days):
        meta_path = os.path.join(meta_dir, f"{d}_meta.json")
        if not os.path.exists(meta_path):
            # fill with zeros so lengths stay aligned with totals
            for k in features:
                features[k].append(0.0)
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        ash = meta["products"].get("ASH_COATED_OSMIUM",    {})
        pep = meta["products"].get("INTARIAN_PEPPER_ROOT", {})
        features["ASH liq_total"].append(ash.get("liq_total",   0))
        features["ASH rev_total"].append(ash.get("rev_total",   0))
        features["ASH imb_mean"].append( ash.get("imb_mean",    0))
        features["ASH imb_std"].append(  ash.get("imb_std",     0))
        features["ASH dev_std"].append(  ash.get("dev_std",     0))
        features["ASH dev_abs_max"].append(ash.get("dev_abs_max", 0))
        features["PEP liq_total"].append(pep.get("liq_total",   0))
        features["PEP rev_total"].append(pep.get("rev_total",   0))
        features["PEP imb_mean"].append( pep.get("imb_mean",    0))
        features["PEP dev_std"].append(  pep.get("dev_std",     0))
        features["PEP dev_abs_max"].append(pep.get("dev_abs_max", 0))

    print(f"\n{'Feature':<25}  {'Corr w/ Total PnL':>18}  {'Interpretation'}")
    print("-" * 70)
    correlations = []
    for feat, vals in features.items():
        if len(vals) != len(totals):
            continue
        r = pearson_r(vals, totals)
        correlations.append((feat, r))

    correlations.sort(key=lambda x: x[1])  # most negative first = most adversarial

    for feat, r in correlations:
        if r < -0.05:
            interp = "ADVERSARIAL — hurts PnL"
        elif r > 0.05:
            interp = "beneficial — helps PnL"
        else:
            interp = "neutral"
        print(f"  {feat:<23}  {r:>+18.3f}  {interp}")

    # Identify the worst paths
    print("\n--- Worst 10 paths (by Total PnL) ---")
    indexed = sorted(enumerate(totals), key=lambda x: x[1])[:10]
    for d, pnl in indexed:
        meta_path = os.path.join(meta_dir, f"{d}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            ash = meta["products"].get("ASH_COATED_OSMIUM",    {})
            pep = meta["products"].get("INTARIAN_PEPPER_ROOT", {})
            print(f"  Day {d:>4}: PnL={pnl:>10,.0f} | "
                  f"ASH liq={ash.get('liq_total','?'):>4}  rev={ash.get('rev_total','?'):>4}  "
                  f"dev_max={ash.get('dev_abs_max',0):.1f} | "
                  f"PEP liq={pep.get('liq_total','?'):>4}  rev={pep.get('rev_total','?'):>4}  "
                  f"dev_max={pep.get('dev_abs_max',0):.1f}")
        else:
            print(f"  Day {d:>4}: PnL={pnl:>10,.0f}  (no meta)")

    # Regime split: high vs low adversarial days
    if features["ASH rev_total"]:
        rev_vals = features["ASH rev_total"]
        median_rev = sorted(rev_vals)[len(rev_vals) // 2]
        high_rev_pnl = [totals[i] for i, v in enumerate(rev_vals) if v > median_rev]
        low_rev_pnl  = [totals[i] for i, v in enumerate(rev_vals) if v <= median_rev]
        s_hi = stats(high_rev_pnl)
        s_lo = stats(low_rev_pnl)
        print(f"\n--- PnL split by ASH reverter activity (median={median_rev:.0f}) ---")
        print(f"  High reverter days ({len(high_rev_pnl)}): "
              f"mean={s_hi.get('mean',0):,.0f}  std={s_hi.get('std',0):,.0f}  "
              f"sharpe={s_hi.get('sharpe',0):.3f}")
        print(f"  Low  reverter days ({len(low_rev_pnl)}): "
              f"mean={s_lo.get('mean',0):,.0f}  std={s_lo.get('std',0):,.0f}  "
              f"sharpe={s_lo.get('sharpe',0):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", help="Path to algorithm .py file")
    parser.add_argument("--n-days", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--regen",  action="store_true",
                        help="Re-estimate params and regenerate data even if cached")
    args = parser.parse_args()

    algo_path = os.path.abspath(args.algo)
    if not os.path.exists(algo_path):
        print(f"Error: algo file not found: {algo_path}")
        sys.exit(1)

    # Step 1: estimate parameters
    if args.regen or not os.path.exists(PARAMS_FILE):
        print("Estimating parameters from real Round-2 data...")
        subprocess.run([sys.executable,
                        os.path.join(SCRIPT_DIR, "estimate_params.py")], check=True)
    else:
        print(f"Using cached params: {PARAMS_FILE}")

    # Step 2: generate synthetic data
    last_csv = os.path.join(MC_DATA, "round99",
                            f"prices_round_99_day_{args.n_days - 1}.csv")
    if args.regen or not os.path.exists(last_csv):
        print(f"Generating {args.n_days} synthetic days (seed={args.seed})...")
        subprocess.run([
            sys.executable,
            os.path.join(SCRIPT_DIR, "generate_data.py"),
            "--n-days", str(args.n_days),
            "--seed",   str(args.seed),
        ], check=True)
    else:
        print(f"Using existing synthetic data ({args.n_days} days).")

    # Step 3: run backtester
    print(f"\nRunning backtester on {args.n_days} synthetic days...\n")
    all_results = []
    for d in range(args.n_days):
        pnls = run_day(algo_path, d)
        all_results.append(pnls)
        total = pnls.get("Total", 0)
        products_str = "  ".join(
            f"{k}={v:>8,.0f}" for k, v in pnls.items() if k != "Total"
        )
        print(f"  Day {d:>4}: Total={total:>10,.0f}  {products_str}")

    # Step 4: PnL summary statistics
    print("\n" + "=" * 70)
    print("MONTE CARLO SUMMARY")
    print("=" * 70)
    all_keys = sorted(set(k for r in all_results for k in r if k != "Total"))
    for key in all_keys + ["Total"]:
        vals = [r[key] for r in all_results if key in r]
        if not vals:
            continue
        s = stats(vals)
        print(f"\n{key}:")
        print(f"  Mean    : {s['mean']:>10,.0f}")
        print(f"  Std     : {s['std']:>10,.0f}")
        print(f"  Sharpe  : {s['sharpe']:>10.3f}")
        print(f"  Min     : {s['min']:>10,.0f}")
        print(f"  P5      : {s['p5']:>10,.0f}")
        print(f"  Median  : {s['median']:>10,.0f}")
        print(f"  P95     : {s['p95']:>10,.0f}")
        print(f"  Max     : {s['max']:>10,.0f}")

    totals = [r["Total"] for r in all_results if "Total" in r]
    if totals:
        print(f"\nTotal PnL Distribution ({len(totals)} paths):")
        print(ascii_histogram(totals))

    # Step 5: adversarial analysis
    analyze_adversarial(all_results, args.n_days)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
