"""
walk_search_osmium.py — Adaptive Gradient Ascent for ASH_COATED_OSMIUM params.

Hill-climbing with adaptive step sizes:
- Steps shrink as improvements get smaller (fine-tune near the top)
- After 10 steps with no improvement, jitter rate increases to escape plateaus
- After 5 jitter steps with no improvement, declares local maximum
- All values capped at 3 decimal places to avoid overfitting

Usage:
    python3 vedant/walk_search_osmium.py
    python3 vedant/walk_search_osmium.py --rounds 1   # R1 only
    python3 vedant/walk_search_osmium.py --rounds 2   # R2 only
    python3 vedant/walk_search_osmium.py --rounds 1 2  # both rounds
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import copy
import math
import random
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = REPO_ROOT / "vedant" / "strategy.py"
WORKER_DIR = REPO_ROOT / ".osmium_worker_pool"

ROUND_DAYS = {
    1: ["1--2", "1--1", "1-0"],
    2: ["2--1", "2-0", "2-1"],
}

# ── Configurable Global Variables ──────────────────────────────────────
ITERATIONS = 20
MAX_NO_IMPROVE = 20          # Steps without improvement before switching to jitter mode
MAX_JITTER_NO_IMPROVE = 30   # Jitter steps without improvement before declaring a local max
JITTER_STEP_SCALE = 1.0      # Scale factor applied to steps during plateau escape
FINE_TUNE_STEP_SCALE = 0.5   # Scale factor after finding a jitter improvement to fine tune
SHRINK_RATE = 0.95           # Shrink rate for consecutive failing steps
ADAPT_SHRINK_RATE = 0.8      # Shrink rate when improvement is small (< 50% of last improvement)

# The following params will be tuned by gradient ascent
# format: ParamDef("NAME", is_int, default_val, base_step, min_step, (min_bound, max_bound))
def get_params() -> List[ParamDef]:
    return [
        ParamDef("OSMIUM_INNER_OFFSET",           True,   3,    1,    1,    (0, 5)),
        ParamDef("OSMIUM_OUTER_OFFSET",           True,   26,   2,    1,    (8, 30)),
        ParamDef("OSMIUM_VOLUME_SKEW_AGGRESSION", False,  0.59, 0.2,  0.05, (0.0, 3.0)),
        ParamDef("OSMIUM_OIM_SHIFT",              True,   1,    1,    1,    (0, 3)),
        ParamDef("OSMIUM_BASE_QUOTE_SIZE",        True,   21,   5,    1,    (10, 80)),
        ParamDef("OSMIUM_KILL_SWITCH_THRESHOLD",  True,   80,   5,    1,    (40, 80)),
        ParamDef("OSMIUM_OIM_THRESHOLD",          False,  0.9,  0.05, 0.01, (0.01, 0.99)),
        ParamDef("OSMIUM_OIM_FADE_SCALE",         False,  1.0,  0.1,  0.05, (0.0, 1.5)),
        ParamDef("OSMIUM_OIM_EDGE_SCALE",         False,  1.0,  0.2,  0.05, (1.0, 5.0)),
        ParamDef("OSMIUM_OIM_TAKE_SCALE",         False,  1.0,  0.2,  0.05, (0.0, 5.0)),
        ParamDef("OSMIUM_FV_TETHER_SCALE",        False,  0.05, 0.01, 0.005, (0.0, 0.5)),
    ]


# ── Parameter Definitions ─────────────────────────────────────

@dataclass
class ParamDef:
    name: str
    is_int: bool
    default: float
    base_step: float      # initial step size for gradient walk
    min_step: float       # smallest step we'll take (precision floor)
    bounds: Tuple[float, float]
    curr_val: float = None

    def __post_init__(self):
        if self.curr_val is None:
            self.randomize()

    def randomize(self):
        if self.is_int:
            lo, hi = int(self.bounds[0]), int(self.bounds[1])
            self.curr_val = float(random.randint(lo, hi))
        else:
            self.curr_val = round(random.uniform(self.bounds[0], self.bounds[1]), 3)

    def perturb(self, step_scale: float = 1.0, jitter: bool = False) -> float:
        """Generate a neighbor value. step_scale shrinks near hilltop. jitter = large random jump."""
        if jitter:
            # Large random jump to escape plateau
            if self.is_int:
                lo, hi = int(self.bounds[0]), int(self.bounds[1])
                return float(random.randint(lo, hi))
            else:
                return round(random.uniform(self.bounds[0], self.bounds[1]), 3)

        step = max(self.min_step, self.base_step * step_scale)

        if self.is_int:
            delta = max(1, int(round(step)))
            v = self.curr_val + random.choice([-delta, delta])
            v = max(self.bounds[0], min(self.bounds[1], round(v)))
            return float(v)
        else:
            delta = random.uniform(-step, step)
            if abs(delta) < self.min_step:
                delta = math.copysign(self.min_step, delta)
            v = self.curr_val + delta
            v = max(self.bounds[0], min(self.bounds[1], v))
            return round(v, 3)  # cap at 3 decimal places

    def set(self, val: float):
        if self.is_int:
            self.curr_val = float(round(val))
        else:
            self.curr_val = round(val, 3)


# ── Strategy Patching ─────────────────────────────────────────

def _load_strategy_src() -> str:
    return STRATEGY_PATH.read_text(encoding="utf-8")


def _patch_strategy(src: str, param_dict: Dict[str, float], params: List[ParamDef]) -> str:
    out = src
    for name, value in param_dict.items():
        p = next(pd for pd in params if pd.name == name)
        if p.is_int:
            pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d[\d_]*)(.*?)$"
            repl = rf"\g<1>{int(value)}\3"
        else:
            pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+(?:\.\d+)?)(.*?)$"
            repl = rf"\g<1>{value:.3f}\3"
        out, n = re.subn(pattern, repl, out, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Failed to patch {name}: matched {n} times")
    return out


# ── Backtesting ───────────────────────────────────────────────

CACHE: Dict[tuple, float] = {}


def _cache_key(param_dict: Dict[str, float], days: List[str]) -> tuple:
    items = tuple(sorted((k, round(v, 3)) for k, v in param_dict.items()))
    return (*items, tuple(days))


def _run_backtest(src: str, param_dict: Dict[str, float], days: List[str], params: List[ParamDef]) -> float:
    key = _cache_key(param_dict, days)
    if key in CACHE:
        return CACHE[key]

    WORKER_DIR.mkdir(exist_ok=True)
    worker_file = WORKER_DIR / "walker.py"
    worker_file.write_text(_patch_strategy(src, param_dict, params), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m", "prosperity4bt",
        str(worker_file),
        *days,
        "--data", str(REPO_ROOT / "data"),
        "--no-progress", "--no-out",
        "--limit", "INTARIAN_PEPPER_ROOT:80",
        "--limit", "ASH_COATED_OSMIUM:80",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)

    if proc.returncode != 0:
        print(f"  ⚠ Backtest FAILED: {proc.stderr[:200]}")
        CACHE[key] = -999999
        return -999999

    matches = re.findall(r"^ASH_COATED_OSMIUM:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if len(matches) != len(days):
        print(f"  ⚠ Parse error: expected {len(days)} PnLs, got {len(matches)}")
        CACHE[key] = -999999
        return -999999

    total_pnl = sum(int(m.replace(",", "")) for m in matches)
    CACHE[key] = total_pnl
    return total_pnl


# ── Gradient Ascent Walker ────────────────────────────────────

def walk(rounds: List[int]):
    """Adaptive gradient ascent with plateau detection and jitter escape."""
    params = get_params()
    src = _load_strategy_src()
    days = []
    for r in rounds:
        days.extend(ROUND_DAYS[r])

    print(f"\n{'='*70}")
    print(f"  OSMIUM GRADIENT ASCENT WALKER")
    print(f"  Rounds: {rounds}  Days: {days}")
    print(f"  Runs (Iterations): {ITERATIONS}")
    print(f"  Steps to plateua (Max No Improve): {MAX_NO_IMPROVE}")
    print(f"  Steps to local diff max (Max Jitter No Improve): {MAX_JITTER_NO_IMPROVE}")
    print(f"  Active Parameters: {len(params)}")
    for p in params:
        print(f"    - {p.name}: bounds {p.bounds}, base_step {p.base_step}, min_step {p.min_step}")
    print(f"{'='*70}")

    global_best_pnl = -float('inf')
    global_best_dict = {}
    total_steps_all_runs = 0

    for run_idx in range(ITERATIONS):
        # Randomize for this run
        for p in params:
            p.randomize()

        current_dict = {p.name: p.curr_val for p in params}
        current_pnl = _run_backtest(src, current_dict, days, params)

        print(f"\n{'='*70}")
        print(f"  RUN {run_idx + 1} / {ITERATIONS} (Random Seed)")
        print(f"{'='*70}")
        print(f"  STARTING PnL: {current_pnl:+,.0f}")
        for p in params:
            print(f"    {p.name}: {p.curr_val}")

        best_pnl = current_pnl
        best_dict = dict(current_dict)
        step_num = 0

        # Adaptive state
        no_improve_count = 0         # consecutive steps with no improvement
        jitter_no_improve_count = 0  # consecutive jitter steps with no improvement
        jitter_mode = False          # whether we're in jitter escape mode
        step_scale = 1.0             # shrinks as we approach hilltop
        last_improvement_size = None # tracks how much the last improvement was

        while True:
            step_num += 1

            if jitter_mode and jitter_no_improve_count >= MAX_JITTER_NO_IMPROVE:
                print(f"\n  🏔 LOCAL MAXIMUM FOUND after {step_num - 1} steps for this run.")
                break

            if not jitter_mode and no_improve_count >= MAX_NO_IMPROVE:
                print(f"\n  📈 Plateau detected at step {step_num}. Switching to JITTER mode.")
                jitter_mode = True
                jitter_no_improve_count = 0
                step_scale = 1.0  # reset step scale for jitter

            # Generate a neighbor: perturb 1-3 random parameters
            n_perturb = random.randint(1, min(3, len(params)))
            chosen = random.sample(params, n_perturb)

            candidate_dict = dict(current_dict)
            for p in chosen:
                candidate_dict[p.name] = p.perturb(
                    step_scale=step_scale,
                    jitter=jitter_mode,
                )

            # Evaluate
            candidate_pnl = _run_backtest(src, candidate_dict, days, params)

            improved = candidate_pnl > current_pnl
            new_best = candidate_pnl > best_pnl

            # Status line
            status = ""
            if new_best:
                status = "⭐ NEW BEST"
            elif improved:
                status = "✅ improved"
            else:
                status = "  ─"

            mode_tag = "JITTER" if jitter_mode else f"scale={step_scale:.2f}"
            changed_str = ", ".join(f"{p.name}={candidate_dict[p.name]}" for p in chosen)
            print(f"  [{step_num:>4}] {mode_tag:>12}  PnL={candidate_pnl:>+10,.0f}  Δ={candidate_pnl - current_pnl:>+8,.0f}  {status}  | {changed_str}")

            if improved:
                improvement = candidate_pnl - current_pnl
                current_pnl = candidate_pnl
                current_dict = dict(candidate_dict)
                for p in params:
                    p.set(current_dict[p.name])

                # Adaptive step reduction: as improvements get smaller, reduce step size
                if last_improvement_size is not None and improvement < last_improvement_size * 0.5:
                    step_scale = max(0.1, step_scale * 0.8)  # shrink steps
                elif improvement > 500:
                    step_scale = min(2.0, step_scale * 1.2)  # we're far from top, keep exploring

                last_improvement_size = improvement
                no_improve_count = 0
                jitter_no_improve_count = 0

                if new_best:
                    best_pnl = candidate_pnl
                    best_dict = dict(candidate_dict)

                if jitter_mode:
                    # Found something good during jitter — switch back to fine-tuning
                    print(f"    → Jitter found improvement! Returning to fine-tune mode.")
                    jitter_mode = False
                    step_scale = 0.5  # start fine-tuning from smaller steps
                    no_improve_count = 0
            else:
                if jitter_mode:
                    jitter_no_improve_count += 1
                else:
                    no_improve_count += 1
                    # Each consecutive failure slightly reduces step size
                    step_scale = max(0.1, step_scale * 0.95)
        
        # ── Intermediate Run Report ──
        print(f"\n  RUN {run_idx + 1} COMPLETE")
        print(f"  Best Run PnL: {best_pnl:>+10,.0f}")
        print(f"  Best Run Params:")
        for name, val in best_dict.items():
            print(f"    {name:<35} = {val:>10}")
        
        total_steps_all_runs += step_num
        
        if best_pnl > global_best_pnl:
            global_best_pnl = best_pnl
            global_best_dict = dict(best_dict)

    # ── Final Report ──
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS ACROSS ALL {ITERATIONS} RUNS")
    print(f"{'='*70}")
    print(f"  Best Global PnL:  {global_best_pnl:>+10,.0f}")
    print(f"  Total steps:      {total_steps_all_runs}")
    print(f"  Cache hits:       {sum(1 for v in CACHE.values() if v != -999999)}")
    print(f"\n  Optimal parameters (from globally best run):")
    for p in params:
        best_val = global_best_dict[p.name]
        print(f"    {p.name:<35} = {best_val:>10}")

    # Print ready-to-paste Python
    print(f"\n  ── Copy-paste into strategy.py ──")
    for p in params:
        v = global_best_dict[p.name]
        if p.is_int:
            print(f"    {p.name} = {int(v)}")
        else:
            print(f"    {p.name} = {v:.3f}")

    # Per-day breakdown of best
    print(f"\n  Per-day PnL breakdown (global best params):")
    for r in rounds:
        r_days = ROUND_DAYS[r]
        r_pnl = _run_backtest(src, global_best_dict, r_days, params)
        print(f"    Round {r}: {r_pnl:>+10,.0f} (days: {', '.join(r_days)})")

        # Individual days
        for d in r_days:
            d_pnl = _run_backtest(src, global_best_dict, [d], params)
            print(f"      {d}: {d_pnl:>+8,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Osmium Gradient Ascent Walker")
    parser.add_argument("--rounds", nargs="+", type=int, default=[2],
                        help="Which rounds to evaluate on (default: 2)")
    args = parser.parse_args()

    for r in args.rounds:
        if r not in ROUND_DAYS:
            print(f"Error: Round {r} not available. Options: {list(ROUND_DAYS.keys())}")
            sys.exit(1)

    walk(args.rounds)


if __name__ == "__main__":
    main()
