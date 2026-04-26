"""
optimizer.py  (Round 2)
-----------------------
Parameter sweep for ASH and PEPPER strategies, including MAF analysis.

Sweep modes:
  grid        — grid search on MC data
  optuna      — Bayesian TPE search on MC data  (requires: pip install optuna)
  real-grid   — grid search on the 3 real round-2 days (noisier, use to validate)

Per-asset isolation:
  Phase 1: Sweep ASH params only (PEPPER fixed at defaults)
  Phase 2: Sweep PEPPER params only (ASH fixed at defaults)
  Phase 3: Combine best params + MAF break-even analysis

MAF analysis:
  For each best-param combo, the optimizer runs both pos_limit=80 (no MAF)
  and pos_limit=100 (MAF won = 25% extra capacity).
  PnL delta per day = value of the extra capacity.
  Recommended MAF bid = delta * competitive_factor (default 0.65).
  Only bid if expected value is positive.

Usage:
    # Full sweep on 50 MC days (recommended)
    uv run python optimizer.py --mode grid --n-trials 200 --n-mc-days 50

    # Bayesian search (better quality, slower)
    uv run python optimizer.py --mode optuna --n-trials 300 --n-mc-days 100

    # Validate on real 3-day data
    uv run python optimizer.py --mode real-grid --n-trials 50

    # Skip re-sweep, just rerun MAF analysis on saved best params
    uv run python optimizer.py --maf-only
"""

import argparse
import csv
import json
import os
import sys
import itertools
import random
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SWEEP_DIR      = Path(__file__).parent
ROUND2_DIR     = SWEEP_DIR.parent
REPO_ROOT      = ROUND2_DIR.parent.parent          # prosperity4/
BACKTESTER_DIR = REPO_ROOT / "imc-prosperity-4-backtester"
MC_DATA_DIR    = ROUND2_DIR / "monte_carlo" / "mc_data"
REAL_DATA_DIR  = ROUND2_DIR / "data"
RESULTS_DIR    = SWEEP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BACKTESTER_DIR))
sys.path.insert(0, str(SWEEP_DIR))

from prosperity4bt.file_reader import FileSystemReader, FileReader
from prosperity4bt.file_reader import wrap_in_context_manager
from prosperity4bt.models import TradeMatchingMode
from prosperity4bt.runner import run_backtest

from trader_template import (
    ParameterizedTrader, DEFAULT_PARAMS, PARAM_BOUNDS,
    ASH_PARAMS, PEPPER_PARAMS,
)


# ── data readers ──────────────────────────────────────────────────────────────

MC_READER  = FileSystemReader(MC_DATA_DIR)          # round 99, days 0..N
MC_ROUND   = 99


class Round2DataReader(FileReader):
    """Redirects round2 lookups to imc-prosperity-4/round2/data/ where we keep CSVs."""
    def __init__(self):
        self._base = REAL_DATA_DIR

    def file(self, path_parts):
        # path_parts = ["round2", "prices_round_2_day_-1.csv"] etc.
        filename = path_parts[-1]
        full = self._base / filename
        if not full.is_file():
            return wrap_in_context_manager(None)
        return wrap_in_context_manager(full)


REAL_READER = Round2DataReader()
REAL_DAYS   = [(2, -1), (2, 0), (2, 1)]


def mc_days(n: int):
    return [(MC_ROUND, d) for d in range(n)]


# ── PnL extraction ────────────────────────────────────────────────────────────

ASH    = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"


def extract_pnl(result) -> dict:
    """Return {product: final_pnl, 'total': float} from one BacktestResult."""
    last_ts = result.activity_logs[-1].timestamp
    pnls = {}
    for row in reversed(result.activity_logs):
        if row.timestamp != last_ts:
            break
        pnls[row.columns[2]] = float(row.columns[-1])
    pnls["total"] = sum(pnls.values())
    return pnls


def evaluate(params: dict, file_reader, days: list) -> dict:
    """
    Run backtest across all days.
    Returns {total_pnl, ash_pnl, pepper_pnl, sharpe, min_day, per_day}.
    """
    trader       = ParameterizedTrader(params)
    daily_totals = []
    ash_sum = pepper_sum = 0.0

    for round_num, day_num in days:
        try:
            result = run_backtest(
                trader, file_reader, round_num, day_num,
                print_output=False,
                trade_matching_mode=TradeMatchingMode.all,
                no_names=True,
                show_progress_bar=False,
            )
        except Exception as e:
            return {"total_pnl": -1e9, "ash_pnl": 0.0, "pepper_pnl": 0.0,
                    "sharpe": -999.0, "min_day": -1e9, "per_day": [], "error": str(e)}

        pnls = extract_pnl(result)
        day_total   = pnls["total"]
        daily_totals.append(day_total)
        ash_sum    += pnls.get(ASH, 0.0)
        pepper_sum += pnls.get(PEPPER, 0.0)

    n    = len(daily_totals)
    total = sum(daily_totals)
    mean  = total / n
    if n > 1:
        var    = sum((v - mean) ** 2 for v in daily_totals) / (n - 1)
        sharpe = mean / var ** 0.5 if var > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_pnl":   total,
        "ash_pnl":     ash_sum,
        "pepper_pnl":  pepper_sum,
        "sharpe":      sharpe,
        "min_day":     min(daily_totals),
        "per_day":     daily_totals,
    }


def score(metrics: dict, pnl_w=1.0, sharpe_w=0.3) -> float:
    return pnl_w * metrics["total_pnl"] + sharpe_w * metrics["sharpe"] * 10_000


# ── Grid search ───────────────────────────────────────────────────────────────

def linspace(lo, hi, n, typ):
    if n == 1:
        return [typ(lo)]
    step = (hi - lo) / (n - 1)
    vals = [typ(lo + i * step) for i in range(n)]
    if typ is int:
        vals = sorted(set(vals))
    return vals


def _build_grid(param_names: list, steps_per_param=4) -> list:
    axis_vals = {}
    for k in param_names:
        lo, hi, typ = PARAM_BOUNDS[k]
        axis_vals[k] = linspace(lo, hi, steps_per_param, typ)
    combos = list(itertools.product(*[axis_vals[k] for k in param_names]))
    return param_names, combos


def run_grid(sweep_name: str,
             sweep_params: list,
             fixed_params: dict,
             file_reader, days: list,
             n_trials: int,
             pnl_w: float, sharpe_w: float,
             steps_per_param: int = 4) -> dict:
    """
    Grid search over sweep_params. Fixed_params merged with defaults.
    Returns best_params dict (full, merged with defaults).
    """
    keys, combos = _build_grid(sweep_params, steps_per_param)

    if len(combos) > n_trials:
        random.shuffle(combos)
        combos = combos[:n_trials]

    print(f"\n{'='*60}")
    print(f"Phase: {sweep_name}  |  {len(combos)} trials  |  {len(days)} days each")
    print(f"Sweeping: {keys}")

    best_score  = -1e18
    best_params = {**DEFAULT_PARAMS, **fixed_params}
    csv_path    = RESULTS_DIR / f"{sweep_name}_grid.csv"
    fieldnames  = ["trial", "score", "total_pnl", "ash_pnl", "pepper_pnl", "sharpe"] + keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, combo in enumerate(combos):
            params = {**DEFAULT_PARAMS, **fixed_params}
            for k, v in zip(keys, combo):
                params[k] = v

            # Enforce hard_limit > soft_limit (ASH)
            if "ash_hard_limit" in params and "ash_soft_limit" in params:
                if params["ash_hard_limit"] <= params["ash_soft_limit"]:
                    params["ash_hard_limit"] = params["ash_soft_limit"] + 10

            metrics = evaluate(params, file_reader, days)
            sc      = score(metrics, pnl_w, sharpe_w)

            row = {"trial": i, "score": sc,
                   "total_pnl": metrics["total_pnl"],
                   "ash_pnl":   metrics["ash_pnl"],
                   "pepper_pnl":metrics["pepper_pnl"],
                   "sharpe":    metrics["sharpe"]}
            for k, v in zip(keys, combo):
                row[k] = v
            writer.writerow(row)
            f.flush()

            if sc > best_score:
                best_score  = sc
                best_params = dict(params)
                print(f"  ★ Trial {i:4d} | score={sc:>12,.0f} | "
                      f"pnl={metrics['total_pnl']:>12,.0f} | sharpe={metrics['sharpe']:.3f} | "
                      + " ".join(f"{k}={v}" for k, v in zip(keys, combo)))
            elif i % 50 == 0:
                print(f"    Trial {i:4d} | score={sc:>12,.0f}")

    print(f"  Best score: {best_score:,.0f}")
    print(f"  Results → {csv_path}")
    return best_params, best_score


# ── Optuna (Bayesian) search ──────────────────────────────────────────────────

def run_optuna(sweep_name: str,
               sweep_params: list,
               fixed_params: dict,
               file_reader, days: list,
               n_trials: int,
               pnl_w: float, sharpe_w: float) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    storage    = f"sqlite:///{RESULTS_DIR}/optuna_{sweep_name}.db"
    study_name = f"r2_{sweep_name}"
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="maximize", load_if_exists=True,
    )

    def trial_fn(trial):
        params = {**DEFAULT_PARAMS, **fixed_params}
        for k in sweep_params:
            lo, hi, typ = PARAM_BOUNDS[k]
            if typ is int:
                params[k] = trial.suggest_int(k, lo, hi)
            else:
                params[k] = trial.suggest_float(k, lo, hi)

        if "ash_hard_limit" in params and "ash_soft_limit" in params:
            if params["ash_hard_limit"] <= params["ash_soft_limit"]:
                params["ash_hard_limit"] = params["ash_soft_limit"] + 10

        metrics = evaluate(params, file_reader, days)
        for attr in ["total_pnl", "ash_pnl", "pepper_pnl", "sharpe", "min_day"]:
            trial.set_user_attr(attr, metrics[attr])
        return score(metrics, pnl_w, sharpe_w)

    done      = len(study.trials)
    remaining = max(0, n_trials - done)
    print(f"\nOptuna '{study_name}': {done} done, running {remaining} more ({len(days)} days each)")
    study.optimize(trial_fn, n_trials=remaining, show_progress_bar=True)

    best        = study.best_trial
    best_params = {**DEFAULT_PARAMS, **fixed_params,
                   **{k: best.params[k] for k in sweep_params if k in best.params}}
    print(f"  Best score: {best.value:,.0f} | pnl={best.user_attrs['total_pnl']:,.0f}")
    return best_params, best.value


# ── MAF analysis ──────────────────────────────────────────────────────────────

def maf_analysis(best_ash_params: dict,
                 best_pepper_params: dict,
                 file_reader, days: list,
                 competitive_factor: float = 0.65) -> dict:
    """
    Computes the value of the MAF contract by running combined best params
    with pos_limit=80 vs pos_limit=100 across all days.

    Returns recommended MAF bid and per-product break-even breakdown.
    """
    print(f"\n{'='*60}")
    print("MAF Analysis: pos_limit 80 vs 100 on combined best params")
    print(f"{'='*60}")

    combined = {**DEFAULT_PARAMS, **best_ash_params, **best_pepper_params}

    # Remove MAF-related keys that might differ — we set pos_limit explicitly
    for limit, label in [(80, "no_maf"), (100, "maf")]:
        params = {**combined, "pos_limit": limit}
        m = evaluate(params, file_reader, days)
        print(f"\n  pos_limit={limit} ({label})")
        print(f"    total_pnl = {m['total_pnl']:>12,.1f}")
        print(f"    ash_pnl   = {m['ash_pnl']:>12,.1f}")
        print(f"    pepper_pnl= {m['pepper_pnl']:>12,.1f}")
        print(f"    sharpe    = {m['sharpe']:>8.3f}")
        if label == "no_maf":
            base = m
        else:
            maf_m = m

    n_days   = len(days)
    delta_total  = maf_m["total_pnl"]  - base["total_pnl"]
    delta_ash    = maf_m["ash_pnl"]    - base["ash_pnl"]
    delta_pepper = maf_m["pepper_pnl"] - base["pepper_pnl"]

    per_day_total  = delta_total  / n_days
    per_day_ash    = delta_ash    / n_days
    per_day_pepper = delta_pepper / n_days

    rec_maf = max(0.0, per_day_total * competitive_factor)

    print(f"\n  PnL delta from MAF (over {n_days} days):")
    print(f"    Total   delta: {delta_total:>10,.1f}  ({per_day_total:>8,.1f} / day)")
    print(f"    ASH     delta: {delta_ash:>10,.1f}  ({per_day_ash:>8,.1f} / day)")
    print(f"    PEPPER  delta: {delta_pepper:>10,.1f}  ({per_day_pepper:>8,.1f} / day)")
    print(f"\n  Recommended MAF bid: {rec_maf:,.0f}  "
          f"(= {per_day_total:.1f} × {competitive_factor} competitive factor)")
    if per_day_total <= 0:
        print("  ⚠  MAF is NOT worth it under these params — do NOT pay the MAF.")
    else:
        print(f"  → Set MAF return value to ~{rec_maf:,.0f} in the algo.")
        print(f"    Minimum competitive bid: {per_day_total * 0.5:,.0f}  "
              f"(50th pct) — anything above this wins the contract.")

    return {
        "delta_total":    delta_total,
        "delta_ash":      delta_ash,
        "delta_pepper":   delta_pepper,
        "per_day_total":  per_day_total,
        "per_day_ash":    per_day_ash,
        "per_day_pepper": per_day_pepper,
        "rec_maf":        rec_maf,
        "use_maf":        per_day_total > 0,
    }


# ── Save / load ───────────────────────────────────────────────────────────────

def save_results(best_ash: dict, best_pepper: dict, maf: dict):
    out = {
        "best_ash_params":    best_ash,
        "best_pepper_params": best_pepper,
        "maf_analysis":       maf,
        "combined_params":    {**DEFAULT_PARAMS, **best_ash, **best_pepper,
                               "pos_limit": 100 if maf["use_maf"] else 80},
    }
    path = RESULTS_DIR / "best_params.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")
    return out


def load_best() -> dict:
    path = RESULTS_DIR / "best_params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Print summary of best vs default ─────────────────────────────────────────

def print_param_diff(best: dict, label: str):
    print(f"\n{label} — changes from DEFAULT_PARAMS:")
    changed = False
    for k in sorted(DEFAULT_PARAMS):
        if k == "pos_limit":
            continue
        dv, bv = DEFAULT_PARAMS[k], best.get(k, DEFAULT_PARAMS[k])
        if dv != bv:
            print(f"  {k:25s}: {dv} → {bv}")
            changed = True
    if not changed:
        print("  (no changes — default params are optimal)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Round 2 parameter sweep + MAF analysis")
    parser.add_argument("--mode",          choices=["grid", "optuna", "real-grid"],
                        default="grid")
    parser.add_argument("--n-trials",      type=int,   default=None)
    parser.add_argument("--n-mc-days",     type=int,   default=50,
                        help="MC days to use (default 50; use 100+ for higher fidelity)")
    parser.add_argument("--pnl-weight",    type=float, default=1.0)
    parser.add_argument("--sharpe-weight", type=float, default=0.3)
    parser.add_argument("--comp-factor",   type=float, default=0.65,
                        help="Competitive factor for MAF bid (0.65 = beat ~65th pct)")
    parser.add_argument("--maf-days",      type=int,   default=None,
                        help="Days for MAF analysis (default: same as sweep days)")
    parser.add_argument("--maf-only",      action="store_true",
                        help="Skip param sweep, only rerun MAF analysis on saved best params")
    parser.add_argument("--study-suffix",  type=str,   default="",
                        help="Suffix for Optuna study names (to avoid conflicts)")
    args = parser.parse_args()

    # ── choose data reader + days ─────────────────────────────────────────────
    if args.mode == "real-grid":
        file_reader = REAL_READER
        days        = REAL_DAYS
        default_n   = 30
        mode_label  = "real-grid"
    else:
        file_reader = MC_READER
        days        = mc_days(args.n_mc_days)
        default_n   = {"grid": 200, "optuna": 300}[args.mode]
        mode_label  = args.mode

    n_trials = args.n_trials or default_n
    maf_days = mc_days(args.maf_days) if args.maf_days else days
    pnl_w, sharpe_w = args.pnl_weight, args.sharpe_weight
    sfx = args.study_suffix

    search_fn = run_optuna if args.mode == "optuna" else run_grid

    def _sweep(name, sweep_params, fixed_params):
        kwargs = dict(
            sweep_name=name + sfx,
            sweep_params=sweep_params,
            fixed_params=fixed_params,
            file_reader=file_reader,
            days=days,
            n_trials=n_trials,
            pnl_w=pnl_w,
            sharpe_w=sharpe_w,
        )
        if args.mode != "optuna":
            # Grid: auto-compute steps from n_trials
            n_dims = len(sweep_params)
            steps  = max(2, int(n_trials ** (1 / n_dims)))
            kwargs["steps_per_param"] = steps
        return search_fn(**kwargs)

    # ── load or sweep ──────────────────────────────────────────────────────────
    if args.maf_only:
        saved = load_best()
        if not saved:
            print("No saved best_params.json found. Run a sweep first.")
            sys.exit(1)
        best_ash    = saved.get("best_ash_params",    DEFAULT_PARAMS)
        best_pepper = saved.get("best_pepper_params", DEFAULT_PARAMS)
        print("Loaded saved best params.")
    else:
        t0 = time.time()

        # Phase 1: ASH sweep (PEPPER fixed at defaults)
        print("\n" + "═"*60)
        print("PHASE 1 — ASH parameter sweep")
        print("═"*60)
        best_ash, ash_score = _sweep(
            name="ash",
            sweep_params=ASH_PARAMS,
            fixed_params={k: DEFAULT_PARAMS[k] for k in PEPPER_PARAMS},
        )

        # Phase 2: PEPPER sweep (ASH fixed at best found above)
        print("\n" + "═"*60)
        print("PHASE 2 — PEPPER parameter sweep")
        print("═"*60)
        best_pepper, pepper_score = _sweep(
            name="pepper",
            sweep_params=PEPPER_PARAMS,
            fixed_params={k: best_ash.get(k, DEFAULT_PARAMS[k]) for k in ASH_PARAMS},
        )

        elapsed = time.time() - t0
        print(f"\n  Sweep complete in {elapsed:.1f}s")

        print_param_diff(best_ash,    "Best ASH params")
        print_param_diff(best_pepper, "Best PEPPER params")

    # Phase 3: MAF analysis on combined best params
    maf = maf_analysis(
        best_ash, best_pepper,
        file_reader if not args.maf_days else MC_READER,
        maf_days,
        competitive_factor=args.comp_factor,
    )

    save_results(best_ash, best_pepper, maf)

    # ── Final validation on real 3-day data ───────────────────────────────────
    print(f"\n{'='*60}")
    print("Real 3-day validation: combined best params")
    print(f"{'='*60}")
    combined = {**DEFAULT_PARAMS, **best_ash, **best_pepper,
                "pos_limit": 100 if maf["use_maf"] else 80}
    real_m   = evaluate(combined, REAL_READER, REAL_DAYS)
    base_m   = evaluate(DEFAULT_PARAMS, REAL_READER, REAL_DAYS)

    print(f"  Default  total_pnl: {base_m['total_pnl']:>12,.1f}")
    print(f"  Best     total_pnl: {real_m['total_pnl']:>12,.1f}")
    print(f"  Delta:              {real_m['total_pnl'] - base_m['total_pnl']:>+12,.1f}")
    print(f"  Best ASH PnL:       {real_m['ash_pnl']:>12,.1f}")
    print(f"  Best PEPPER PnL:    {real_m['pepper_pnl']:>12,.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
