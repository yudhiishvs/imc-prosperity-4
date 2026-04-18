"""
optimizer.py
------------
Hyperparameter optimizer for the IMC Prosperity Trader.

Modes
-----
grid      — fast 2-D / 3-D grid sweep on 3 real days
optuna    — Bayesian TPE search on 3 real days  (requires: pip install optuna)
mc-grid   — grid sweep on 50 Monte Carlo days
mc-optuna — Bayesian search on 50 Monte Carlo days

Usage
-----
    # Fast grid search (ASH skew × size, PEPPER ceiling)
    python optimizer.py --mode grid --n-trials 200

    # Bayesian search, all params
    python optimizer.py --mode optuna --n-trials 400

    # Monte Carlo Bayesian (higher-fidelity signal)
    python optimizer.py --mode mc-optuna --n-trials 200

    # Resume a saved Optuna study
    python optimizer.py --mode optuna --study existing-study-name

Output
------
    results/grid_results.csv        — every trial + score
    results/best_params.json        — best parameter vector found
    results/optuna_study.db         — Optuna SQLite storage (for mc-optuna / optuna modes)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── path setup ──────────────────────────────────────────────────────────────
OPTIMIZER_DIR = Path(__file__).parent
PROSPERITY4   = OPTIMIZER_DIR.parent
ALGO_DIR      = PROSPERITY4 / "imc-prosperity-4"
BACKTESTER_DIR = PROSPERITY4 / "imc-prosperity-4-backtester"
MC_DATA_DIR   = PROSPERITY4 / "monte_carlo" / "mc_data"
RESULTS_DIR   = OPTIMIZER_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Make the backtester and algo importable
sys.path.insert(0, str(BACKTESTER_DIR))
sys.path.insert(0, str(ALGO_DIR))

from prosperity4bt.file_reader import FileSystemReader, PackageResourcesReader
from prosperity4bt.models import TradeMatchingMode
from prosperity4bt.runner import run_backtest

from trader_template import DEFAULT_PARAMS, PARAM_BOUNDS, ParameterizedTrader

# ── data readers ────────────────────────────────────────────────────────────
REAL_DAYS = [(1, -2), (1, -1), (1, 0)]
# FileSystemReader needs root that contains round1/ subdirectory
# imc-prosperity-4/round1 is a symlink to ROUND1/
REAL_READER = FileSystemReader(ALGO_DIR)

def mc_reader(n_days: int = 50):
    return FileSystemReader(MC_DATA_DIR), [(99, d) for d in range(n_days)]


# ── PnL extraction ──────────────────────────────────────────────────────────

def extract_day_pnl(result) -> dict:
    """Return {product: final_pnl} for one BacktestResult."""
    last_ts = result.activity_logs[-1].timestamp
    pnls = {}
    for row in reversed(result.activity_logs):
        if row.timestamp != last_ts:
            break
        pnls[row.columns[2]] = row.columns[-1]
    return pnls


def evaluate(params: dict,
             file_reader,
             days: list,
             ) -> dict:
    """
    Run backtest across all days. Returns:
        {
          "total_pnl":   float,
          "ash_pnl":     float,
          "pepper_pnl":  float,
          "sharpe":      float,   # daily PnL mean / std  (0 if 1 day)
          "min_day":     float,
          "per_day":     [float, ...]
        }
    """
    trader = ParameterizedTrader(params)
    daily_totals = []
    ash_sum = pepper_sum = 0.0

    for round_num, day_num in days:
        try:
            result = run_backtest(
                trader,
                file_reader,
                round_num,
                day_num,
                print_output=False,
                trade_matching_mode=TradeMatchingMode.all,
                no_names=True,
                show_progress_bar=False,
            )
        except Exception as e:
            # Parameter combination caused a crash → penalize
            return {"total_pnl": -1e9, "ash_pnl": 0, "pepper_pnl": 0,
                    "sharpe": -999, "min_day": -1e9, "per_day": [], "error": str(e)}

        pnls = extract_day_pnl(result)
        day_total = sum(pnls.values())
        daily_totals.append(day_total)
        ash_sum    += pnls.get("ASH_COATED_OSMIUM", 0)
        pepper_sum += pnls.get("INTARIAN_PEPPER_ROOT", 0)

    total = sum(daily_totals)
    n     = len(daily_totals)
    mean  = total / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in daily_totals) / (n - 1)
        std = variance ** 0.5
        sharpe = mean / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_pnl":  total,
        "ash_pnl":    ash_sum,
        "pepper_pnl": pepper_sum,
        "sharpe":     sharpe,
        "min_day":    min(daily_totals),
        "per_day":    daily_totals,
    }


# ── Objective function ──────────────────────────────────────────────────────

def objective(metrics: dict,
              pnl_weight: float = 1.0,
              sharpe_weight: float = 0.0) -> float:
    """
    Scalar score to maximize.

    For 3 real days:   pnl_weight=1.0 dominates (sharpe is noisy on 3 samples)
    For 50 MC days:    sharpe_weight=0.3 adds meaningful signal

    score = pnl_weight * total_pnl + sharpe_weight * (sharpe * 10_000)
    The 10_000 scaling makes Sharpe numerically comparable to PnL magnitude.
    """
    return pnl_weight * metrics["total_pnl"] + sharpe_weight * metrics["sharpe"] * 10_000


# ── Grid search ─────────────────────────────────────────────────────────────

def linspace(lo, hi, n, typ):
    """n evenly-spaced values from lo to hi inclusive."""
    if n == 1:
        return [typ(lo)]
    step = (hi - lo) / (n - 1)
    vals = [typ(lo + i * step) for i in range(n)]
    if typ is int:
        vals = sorted(set(vals))
    return vals


# Grid axes: define which params to sweep and how many steps
GRID_AXES = {
    # ASH
    "ash_base_quote_size":     (10, 60, 6),
    "ash_volume_skew":         (0.5, 3.0, 6),
    "ash_emergency_threshold": (40, 70, 4),
    "ash_kill_switch":         (60, 80, 3),
    # PEPPER
    "pepper_buy_ceiling":      (4, 14, 11),
    "pepper_passive_bid_1":    (0, 5, 6),
}


def run_grid(file_reader, days, n_trials=None,
             pnl_weight=1.0, sharpe_weight=0.0,
             axes=None):
    """
    Cartesian grid search. n_trials caps total evaluations (random subsample
    of the full grid if the grid is larger).
    """
    import itertools, random

    axes = axes or GRID_AXES
    lo, hi, typ = PARAM_BOUNDS["ash_emergency_threshold"]

    # Build full grid as list of param-dicts
    axis_vals = {}
    for key, (lo, hi, steps) in axes.items():
        lo_b, hi_b, typ = PARAM_BOUNDS[key]
        axis_vals[key] = linspace(max(lo, lo_b), min(hi, hi_b), steps, typ)

    keys   = list(axis_vals.keys())
    combos = list(itertools.product(*[axis_vals[k] for k in keys]))

    if n_trials is not None and len(combos) > n_trials:
        random.shuffle(combos)
        combos = combos[:n_trials]

    print(f"Grid search: {len(combos)} trials over {keys}")

    results = []
    best_score = -1e18
    best_params = dict(DEFAULT_PARAMS)

    csv_path = RESULTS_DIR / "grid_results.csv"
    fieldnames = ["trial", "score", "total_pnl", "ash_pnl", "pepper_pnl", "sharpe"] + keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos):
            params = dict(DEFAULT_PARAMS)
            for k, v in zip(keys, combo):
                params[k] = v

            # Enforce: emergency_target < emergency_threshold
            if params.get("ash_emergency_target", 30) >= params.get("ash_emergency_threshold", 60):
                params["ash_emergency_target"] = params["ash_emergency_threshold"] - 10

            metrics = evaluate(params, file_reader, days)
            score   = objective(metrics, pnl_weight, sharpe_weight)

            row = {"trial": i, "score": score,
                   "total_pnl": metrics["total_pnl"],
                   "ash_pnl":   metrics["ash_pnl"],
                   "pepper_pnl":metrics["pepper_pnl"],
                   "sharpe":    metrics["sharpe"]}
            for k, v in zip(keys, combo):
                row[k] = v
            writer.writerow(row)
            f.flush()

            if score > best_score:
                best_score  = score
                best_params = dict(params)
                print(f"  ★ Trial {i:4d} | score={score:,.0f} | "
                      f"PnL={metrics['total_pnl']:,.0f} | "
                      f"Sharpe={metrics['sharpe']:.3f} | "
                      + " ".join(f"{k}={v}" for k, v in zip(keys, combo)))

            elif i % 20 == 0:
                print(f"    Trial {i:4d} | score={score:,.0f}")

    print(f"\nGrid search done. Best score={best_score:,.0f}")
    return best_params, best_score


# ── Optuna (Bayesian) search ─────────────────────────────────────────────────

def run_optuna(file_reader, days, n_trials=300,
               study_name="prosperity_opt",
               pnl_weight=1.0, sharpe_weight=0.0):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    storage = f"sqlite:///{RESULTS_DIR}/optuna_study.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def trial_fn(trial):
        params = {}
        for key, (lo, hi, typ) in PARAM_BOUNDS.items():
            if typ is int:
                params[key] = trial.suggest_int(key, lo, hi)
            else:
                params[key] = trial.suggest_float(key, lo, hi)

        # Constraint: emergency_target must be below emergency_threshold
        if params["ash_emergency_target"] >= params["ash_emergency_threshold"]:
            params["ash_emergency_target"] = params["ash_emergency_threshold"] - 5

        metrics = evaluate(params, file_reader, days)
        # Store extra metrics as user attributes for analysis
        trial.set_user_attr("total_pnl",   metrics["total_pnl"])
        trial.set_user_attr("ash_pnl",     metrics["ash_pnl"])
        trial.set_user_attr("pepper_pnl",  metrics["pepper_pnl"])
        trial.set_user_attr("sharpe",      metrics["sharpe"])
        trial.set_user_attr("min_day",     metrics["min_day"])
        return objective(metrics, pnl_weight, sharpe_weight)

    completed = len(study.trials)
    remaining = max(0, n_trials - completed)
    print(f"Optuna study '{study_name}': {completed} trials done, running {remaining} more")

    study.optimize(trial_fn, n_trials=remaining, show_progress_bar=True)

    best = study.best_trial
    best_params = {**DEFAULT_PARAMS, **{k: best.params[k] for k in PARAM_BOUNDS}}
    print(f"\nBest score: {best.value:,.0f}")
    print(f"  total_pnl  = {best.user_attrs['total_pnl']:,.0f}")
    print(f"  sharpe     = {best.user_attrs['sharpe']:.3f}")
    return best_params, best.value


# ── Save / load ──────────────────────────────────────────────────────────────

def save_best(params: dict, score: float, mode: str):
    path = RESULTS_DIR / "best_params.json"
    record = {"score": score, "mode": mode, "params": params}
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved best params → {path}")


def load_best() -> dict:
    path = RESULTS_DIR / "best_params.json"
    if not path.exists():
        return dict(DEFAULT_PARAMS)
    with open(path) as f:
        return json.load(f)["params"]


# ── Validation: compare best vs default on real data ─────────────────────────

def validate(params: dict, label: str = "best"):
    print(f"\n{'='*60}")
    print(f"Validating '{label}' on real 3-day data")
    print(f"{'='*60}")
    metrics = evaluate(params, REAL_READER, REAL_DAYS)
    print(f"  Total PnL : {metrics['total_pnl']:>12,.0f}")
    print(f"  ASH PnL   : {metrics['ash_pnl']:>12,.0f}")
    print(f"  PEPPER PnL: {metrics['pepper_pnl']:>12,.0f}")
    for i, d in enumerate(metrics["per_day"]):
        print(f"  Day {REAL_DAYS[i][1]:+d}   : {d:>12,.0f}")

    print(f"\nBaseline (DEFAULT_PARAMS) on real 3-day data:")
    base = evaluate(DEFAULT_PARAMS, REAL_READER, REAL_DAYS)
    print(f"  Total PnL : {base['total_pnl']:>12,.0f}")

    delta = metrics["total_pnl"] - base["total_pnl"]
    print(f"\n  Delta vs baseline: {delta:+,.0f}")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity hyperparameter optimizer")
    parser.add_argument("--mode", choices=["grid", "optuna", "mc-grid", "mc-optuna"],
                        default="grid")
    parser.add_argument("--n-trials",    type=int,   default=None,
                        help="Number of trials (default: auto per mode)")
    parser.add_argument("--n-mc-days",  type=int,   default=50,
                        help="MC days for mc-* modes")
    parser.add_argument("--pnl-weight", type=float, default=1.0)
    parser.add_argument("--sharpe-weight", type=float, default=0.0,
                        help="Non-zero adds Sharpe to objective (useful for mc modes)")
    parser.add_argument("--study",      type=str,   default="prosperity_opt",
                        help="Optuna study name (set to resume existing)")
    parser.add_argument("--validate",   action="store_true",
                        help="After optimization, validate best params on real 3-day data")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip optimization, just validate saved best_params.json")
    args = parser.parse_args()

    if args.validate_only:
        params = load_best()
        validate(params, "saved best")
        return

    t0 = time.time()

    # ── choose data source ──────────────────────────────────────────────────
    if args.mode in ("grid", "optuna"):
        file_reader = REAL_READER
        days        = REAL_DAYS
        default_trials = {"grid": 300, "optuna": 400}
    else:
        file_reader, days = mc_reader(args.n_mc_days)
        default_trials    = {"mc-grid": 200, "mc-optuna": 300}
        # MC modes benefit from Sharpe weighting
        if args.sharpe_weight == 0.0:
            args.sharpe_weight = 0.3

    n_trials = args.n_trials or default_trials[args.mode]

    # ── run chosen optimizer ────────────────────────────────────────────────
    if args.mode in ("grid", "mc-grid"):
        best_params, best_score = run_grid(
            file_reader, days, n_trials=n_trials,
            pnl_weight=args.pnl_weight, sharpe_weight=args.sharpe_weight,
        )
    else:
        best_params, best_score = run_optuna(
            file_reader, days, n_trials=n_trials,
            study_name=args.study,
            pnl_weight=args.pnl_weight, sharpe_weight=args.sharpe_weight,
        )

    save_best(best_params, best_score, args.mode)

    elapsed = time.time() - t0
    print(f"\nOptimization finished in {elapsed:.1f}s")

    # ── print best params vs defaults ───────────────────────────────────────
    print("\nBest params (changes from default):")
    for k, v in best_params.items():
        if v != DEFAULT_PARAMS.get(k):
            print(f"  {k:35s}: {DEFAULT_PARAMS.get(k)} → {v}")

    if args.validate or args.mode in ("grid", "mc-grid"):
        validate(best_params)


if __name__ == "__main__":
    main()
