"""
optimize_osmium_mc.py — Stochastic Gradient Ascent for ASH_COATED_OSMIUM params.

Optimizes for Risk-Adjusted Return (Sharpe Ratio) using Monte Carlo simulations.
Minimizes variance (StdDev) while maximizing Mean PnL to find robust bot interaction.

Usage:
    python3 vedant/optimize_osmium_mc.py --sessions 100
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
WORKER_DIR = REPO_ROOT / ".osmium_mc_optimizer"

# ── Configurable Global Variables ──────────────────────────────────────
ITERATIONS = 20
MAX_NO_IMPROVE = 20          # Steps without improvement before switching to jitter mode
MAX_JITTER_NO_IMPROVE = 30   # Jitter steps without improvement before declaring a local max
SHRINK_RATE = 0.95           # Shrink rate for consecutive failing steps
ADAPT_SHRINK_RATE = 0.8      # Shrink rate when improvement is small

# ── Parameter Definitions ─────────────────────────────────────

@dataclass
class ParamDef:
    name: str
    is_int: bool
    default: float
    base_step: float      # initial step size for gradient walk
    min_step: float       # smallest step we'll take
    bounds: Tuple[float, float]
    curr_val: float = None

    def __post_init__(self):
        if self.curr_val is None:
            self.curr_val = self.default

    def randomize(self):
        if self.is_int:
            lo, hi = int(self.bounds[0]), int(self.bounds[1])
            self.curr_val = float(random.randint(lo, hi))
        else:
            self.curr_val = round(random.uniform(self.bounds[0], self.bounds[1]), 3)

    def perturb(self, step_scale: float = 1.0, jitter: bool = False) -> float:
        if jitter:
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
            return round(v, 3)

    def set(self, val: float):
        if self.is_int:
            self.curr_val = float(round(val))
        else:
            self.curr_val = round(val, 3)

def get_params() -> List[ParamDef]:
    return [
        ParamDef("OSMIUM_EMA_ALPHA",             False,  0.001, 0.001, 0.0001, (0.0001, 0.1)),
        ParamDef("OSMIUM_INNER_OFFSET",           True,   10,    1,    1,    (0, 10)),
        ParamDef("OSMIUM_OUTER_OFFSET",           True,   10,    2,    1,    (0, 30)),
        ParamDef("OSMIUM_VOLUME_SKEW_AGGRESSION", False,  1.5,   0.2,  0.05, (0.0, 3.0)),
        ParamDef("OSMIUM_OIM_SHIFT",              True,   0,     1,    1,    (0, 5)),
        ParamDef("OSMIUM_BASE_QUOTE_SIZE",        True,   20,    5,    1,    (10, 80)),
        ParamDef("OSMIUM_L2_QUOTE_SIZE",          True,   40,    5,    1,    (10, 80)),
        ParamDef("OSMIUM_KILL_SWITCH_THRESHOLD",  True,   80,    5,    1,    (40, 80)),
        ParamDef("OSMIUM_OIM_THRESHOLD",          False,  0.9,   0.05, 0.01, (0.0, 0.99)),
        ParamDef("OSMIUM_OIM_FADE_SCALE",         False,  0.0,   0.1,  0.05, (0.0, 1.5)),
        ParamDef("OSMIUM_OIM_EDGE_SCALE",         False,  1.0,   0.2,  0.1,  (0.0, 5.0)),
        ParamDef("OSMIUM_OIM_TAKE_SCALE",         False,  1.0,   0.2,  0.1,  (0.0, 8.0)),
        ParamDef("OSMIUM_FV_TETHER_SCALE",        False,  0.05,  0.01, 0.005, (0.0, 0.5)),
    ]

# ── Strategy Patching ─────────────────────────────────────────

def _load_strategy_src() -> str:
    return STRATEGY_PATH.read_text(encoding="utf-8")

def _patch_strategy(src: str, param_dict: Dict[str, float], params: List[ParamDef]) -> str:
    out = src
    for name, value in param_dict.items():
        p = next(pd for pd in params if pd.name == name)
        # Match class-level attribute assignment with possible indentation
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

CACHE: Dict[tuple, Tuple[float, float]] = {}

# Autodetect paths
BACKTESTER_BIN = (REPO_ROOT.parent / "prosperity4mcbt" / "rust_simulator" / "target" / "release" / "rust_simulator").resolve()
PYTHON_BIN = (REPO_ROOT.parent / "prosperity4mcbt" / "backtester" / ".venv" / "bin" / "python3").resolve()

if not BACKTESTER_BIN.exists():
    raise RuntimeError(f"Could not find backtester binary at {BACKTESTER_BIN}")
if not PYTHON_BIN.exists():
    PYTHON_BIN = sys.executable # Fallback

def calculate_score(mean: float, std: float) -> float:
    """Maximizes Risk-Adjusted Reward (Mean / Std)."""
    if mean <= 0:
        return mean # Linear penalty for losing paths
    if std <= 0.01:
        return mean # Avoid div by zero, treat zero-variance as raw PnL
    return mean / std

def _run_backtest(src: str, param_dict: Dict[str, float], params: List[ParamDef], sessions: int) -> Tuple[float, float]:
    key = tuple(sorted(param_dict.items()))
    if key in CACHE:
        return CACHE[key]

    BACKTESTER_ROOT = REPO_ROOT.parent / "prosperity4mcbt"
    WORKER_DIR.mkdir(exist_ok=True)
    worker_file = (WORKER_DIR / "mc_strategy.py").resolve()
    worker_file.write_text(_patch_strategy(src, param_dict, params), encoding="utf-8")
    
    output_dir = (WORKER_DIR / "output").resolve()
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call simulator directly (bypass CLI/Dashboard overhead)
    cmd = [
        str(BACKTESTER_BIN),
        "--strategy", str(worker_file),
        "--sessions", str(sessions),
        "--output", str(output_dir),
        "--python-bin", str(PYTHON_BIN),
        "--actual-dir", str(BACKTESTER_ROOT / "data" / "round0"),
        "--write-session-limit", "0", 
        "--fv-mode", "simulate",
        "--trade-mode", "simulate",
        "--ticks-per-day", "10000",
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PROSPERITY4MCBT_ROOT"] = str(BACKTESTER_ROOT)

    proc = subprocess.run(cmd, cwd=str(BACKTESTER_ROOT), text=True, capture_output=True, env=env)

    if proc.returncode != 0:
        print(f"  ⚠ Simulator FAILED (code {proc.returncode})")
        print(f"  STDERR: {proc.stderr}")
        return -999999.0, 1.0

    # Parse run_summary.csv
    summary_file = output_dir / "run_summary.csv"
    if not summary_file.exists():
        print(f"  ⚠ run_summary.csv missing.")
        return -999999.0, 1.0

    import csv
    with open(summary_file, "r") as f:
        reader = csv.DictReader(f)
        pnls = [float(row["ash_coated_osmium_pnl"]) for row in reader]

    if not pnls:
        return -999999.0, 1.0

    import statistics
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls) if len(pnls) > 1 else 1.0
    
    CACHE[key] = (mean, std)
    return mean, std

def _print_params(params: List[ParamDef], param_dict: Dict[str, float], title: str):
    print(f"\n  ── {title} ──")
    snippet = ""
    for p in params:
        val = param_dict[p.name]
        line = f"    {p.name:<35} = "
        if p.is_int:
            line += f"{int(val):>10}"
        else:
            line += f"{val:>10.3f}"
        print(line)
        snippet += line + "\n"
    print("")
    
    # Persist the global best to a file
    if "GLOBAL BEST" in title:
        best_file = REPO_ROOT / "vedant" / "osmium_best_params.py"
        best_file.write_text(f"# {title}\n" + snippet, encoding="utf-8")

# ── Gradient Ascent Walker ────────────────────────────────────

def walk(sessions: int, iterations: int):
    params = get_params()
    src = _load_strategy_src()

    print(f"\n{'='*70}")
    print(f"  MC RISK-ADJUSTED OPTIMIZER (OSMIUM)")
    print(f"  Sessions per step: {sessions}")
    print(f"  Iterations (starts): {iterations}")
    print(f"  Objective: Maximize Mean / StdDev")
    print(f"{'='*70}\n")

    global_best_score = -float('inf')
    global_best_dict = {}
    
    for run_idx in range(iterations):
        if run_idx > 0:
            for p in params: p.randomize()
        
        current_dict = {p.name: p.curr_val for p in params}
        mean, std = _run_backtest(src, current_dict, params, sessions)
        current_score = calculate_score(mean, std)

        print(f"\n  RUN {run_idx + 1} START | Score: {current_score:.4f} (Mean: {mean:,.0f}, Std: {std:,.0f})")
        
        best_score = current_score
        best_dict = dict(current_dict)
        no_improve_count = 0
        jitter_mode = False
        step_scale = 1.0

        for step in range(100): # Limit steps per random start
            if jitter_mode and no_improve_count >= MAX_JITTER_NO_IMPROVE:
                break
            if not jitter_mode and no_improve_count >= MAX_NO_IMPROVE:
                jitter_mode = True
                no_improve_count = 0
                step_scale = 1.0

            # Perturb 1-3 params
            n_perturb = random.randint(1, 3)
            chosen = random.sample(params, n_perturb)
            candidate_dict = dict(current_dict)
            for p in chosen:
                candidate_dict[p.name] = p.perturb(step_scale=step_scale, jitter=jitter_mode)

            c_mean, c_std = _run_backtest(src, candidate_dict, params, sessions)
            c_score = calculate_score(c_mean, c_std)

            if c_score > current_score:
                # Calculate improvement over the previous CURRENT score, not best
                # current_score is what we walk on
                current_score = c_score
                current_dict = dict(candidate_dict)
                for p in params: p.set(current_dict[p.name])
                no_improve_count = 0
                if jitter_mode:
                    jitter_mode = False
                    step_scale = 0.5
                
                if c_score > best_score:
                    best_score = c_score
                    best_dict = dict(candidate_dict)
                    if c_score > global_best_score:
                        _print_params(params, best_dict, "NEW GLOBAL BEST FOUND")
                
                status = "⭐ NEW BEST" if c_score > best_score else "✅ improved"
                print(f"  [{step:>3}] Score: {c_score:.4f} | {status}")
            else:
                no_improve_count += 1
                step_scale = max(0.1, step_scale * SHRINK_RATE)
                sys.stdout.write(".")
                sys.stdout.flush()

        if best_score > global_best_score:
            global_best_score = best_score
            global_best_dict = dict(best_dict)
        
        _print_params(params, best_dict, f"BEST FOR RUN {run_idx + 1}")

    print(f"\n{'='*70}")
    print(f"  GLOBAL BEST RESULTS")
    print(f"{'='*70}")
    print(f"  Score (Sharpe): {global_best_score:.4f}")
    
    print(f"\n  ── Snippet for strategy.py ──")
    for p in params:
        val = global_best_dict[p.name]
        if p.is_int:
            print(f"    {p.name} = {int(val)}")
        else:
            print(f"    {p.name} = {val:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    
    walk(args.sessions, args.iterations)

if __name__ == "__main__":
    main()
