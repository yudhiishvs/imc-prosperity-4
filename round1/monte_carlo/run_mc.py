"""
run_mc.py
---------
Monte Carlo backtester runner.
Generates N synthetic days, runs the backtester on each, collects PnL statistics.

Usage:
    python run_mc.py <path_to_algo.py> [--n-days 50] [--seed 42] [--regen]

Example:
    python run_mc.py ../imc-prosperity-4/v35_round1_prosperity.py --n-days 100 --seed 42
"""

import argparse
import os
import re
import subprocess
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MC_DATA      = os.path.join(SCRIPT_DIR, "mc_data")
PARAMS_FILE  = os.path.join(SCRIPT_DIR, "params.json")
BACKTESTER   = "/opt/homebrew/bin/prosperity4btx"
ROUND_NUM    = 99

# Regex to parse per-product PnL lines from backtester stdout
# e.g.  "ASH_COATED_OSMIUM: 16,473"
PRODUCT_RE = re.compile(r"^([\w_]+):\s+([\-\d,]+)$")
TOTAL_RE   = re.compile(r"^Total profit:\s+([\-\d,]+)$")


def parse_pnl(line: str):
    """Parse 'PRODUCT: X,XXX' or 'Total profit: X,XXX'. Returns (name, value) or None."""
    m = PRODUCT_RE.match(line.strip())
    if m:
        return m.group(1), int(m.group(2).replace(",", ""))
    m = TOTAL_RE.match(line.strip())
    if m:
        return "Total", int(m.group(1).replace(",", ""))
    return None


def run_day(algo_path: str, day: int) -> dict:
    """
    Run backtester on one synthetic day. Returns {product: pnl, 'Total': pnl}.
    """
    cmd = [
        BACKTESTER,
        algo_path,
        f"{ROUND_NUM}-{day}",
        "--data", MC_DATA,
        "--no-out",
        "--no-progress",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout

    pnls = {}
    for line in output.splitlines():
        parsed = parse_pnl(line)
        if parsed:
            pnls[parsed[0]] = parsed[1]

    return pnls


def stats(values: list) -> dict:
    if not values:
        return {}
    n     = len(values)
    mean  = sum(values) / n
    var   = sum((v - mean)**2 for v in values) / (n - 1) if n > 1 else 0
    std   = var**0.5
    srt   = sorted(values)
    p5    = srt[max(0, int(0.05 * n))]
    p95   = srt[min(n-1, int(0.95 * n))]
    sharpe = mean / std if std > 0 else 0.0
    return {
        "n":      n,
        "mean":   mean,
        "std":    std,
        "sharpe": sharpe,
        "min":    srt[0],
        "p5":     p5,
        "median": srt[n // 2],
        "p95":    p95,
        "max":    srt[-1],
    }


def ascii_histogram(values: list, bins: int = 20, width: int = 50) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"  All values = {lo:,.0f}"
    bin_size = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / bin_size), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar_lo = lo + i * bin_size
        bar = "#" * int(c / max_count * width)
        lines.append(f"  {bar_lo:>10,.0f} | {bar:<{width}} {c}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", help="Path to algorithm .py file")
    parser.add_argument("--n-days", type=int, default=50)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--regen",  action="store_true",
                        help="Re-run estimate_params + generate_data even if files exist")
    args = parser.parse_args()

    algo_path = os.path.abspath(args.algo)
    if not os.path.exists(algo_path):
        print(f"Error: algo file not found: {algo_path}")
        sys.exit(1)

    # Step 1: estimate parameters (once)
    if args.regen or not os.path.exists(PARAMS_FILE):
        print("Estimating parameters from real data...")
        subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "estimate_params.py")], check=True)
    else:
        print(f"Using cached params: {PARAMS_FILE}")

    # Step 2: generate synthetic data
    if args.regen or not os.path.exists(os.path.join(MC_DATA, "round99",
                                                       f"prices_round_99_day_{args.n_days - 1}.csv")):
        print(f"Generating {args.n_days} synthetic days (seed={args.seed})...")
        subprocess.run([
            sys.executable, os.path.join(SCRIPT_DIR, "generate_data.py"),
            "--n-days", str(args.n_days),
            "--seed",   str(args.seed),
        ], check=True)
    else:
        print(f"Using existing synthetic data ({args.n_days} days).")

    # Step 3: run backtester on each day
    print(f"\nRunning backtester on {args.n_days} synthetic days...\n")

    all_results = []   # list of {product: pnl, ...}
    for d in range(args.n_days):
        pnls = run_day(algo_path, d)
        all_results.append(pnls)
        total = pnls.get("Total", 0)
        print(f"  Day {d:>3}: Total={total:>10,.0f}  "
              + "  ".join(f"{k}={v:>8,.0f}" for k, v in pnls.items() if k != "Total"))

    # Step 4: summary statistics
    print("\n" + "="*70)
    print("MONTE CARLO SUMMARY")
    print("="*70)

    products = sorted(set(k for r in all_results for k in r if k != "Total"))
    for key in products + ["Total"]:
        vals = [r[key] for r in all_results if key in r]
        if not vals:
            continue
        s = stats(vals)
        print(f"\n{key}:")
        print(f"  Mean:    {s['mean']:>10,.0f}")
        print(f"  Std:     {s['std']:>10,.0f}")
        print(f"  Sharpe:  {s['sharpe']:>10.3f}  (mean/std)")
        print(f"  Min:     {s['min']:>10,.0f}")
        print(f"  P5:      {s['p5']:>10,.0f}")
        print(f"  Median:  {s['median']:>10,.0f}")
        print(f"  P95:     {s['p95']:>10,.0f}")
        print(f"  Max:     {s['max']:>10,.0f}")

    # Histogram of Total PnL
    totals = [r["Total"] for r in all_results if "Total" in r]
    if totals:
        print(f"\nTotal PnL Distribution ({len(totals)} days):")
        print(ascii_histogram(totals))

    print("="*70)


if __name__ == "__main__":
    main()
