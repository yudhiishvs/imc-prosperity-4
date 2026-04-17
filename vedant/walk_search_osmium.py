"""
Coordinate Ascent (Walker) Optimization for ASH_COATED_OSMIUM parameters.
Optimizes for high mean PnL with low variance across days.

Run from repo root:
  python3 -u ./vedant/walk_search_osmium.py
"""

from __future__ import annotations

import os
import re
import sys
import copy
import subprocess
import tempfile
import numpy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = REPO_ROOT / "vedant" / "updated_osmium.py"
DAYS = ["1--2", "1--1", "1-0"]

# Score = Mean PnL - PENALTY * StdDev PnL
PNL_VARIANCE_PENALTY = 0.5  

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)

@dataclass
class ParamDef:
    name: str 
    is_int: bool
    curr_val: float
    step: float
    min_step: float
    bounds: tuple[float, float]

    def get_neighbors(self) -> List[float]:
        if self.step <= 0:
            return []
        
        v1 = self.curr_val - self.step
        v2 = self.curr_val + self.step
        
        if self.is_int:
            v1 = round(v1)
            v2 = round(v2)
            
        v1 = max(self.bounds[0], min(self.bounds[1], v1))
        v2 = max(self.bounds[0], min(self.bounds[1], v2))
        
        neighbors = set()
        if abs(v1 - self.curr_val) > 1e-6:
            neighbors.add(v1)
        if abs(v2 - self.curr_val) > 1e-6:
            neighbors.add(v2)
        return list(neighbors)

    def shrink_step(self) -> bool:
        """Shrink step size. Returns True if step is still valid, False if stopped."""
        if self.step <= 0:
            return False
            
        if self.is_int:
            new_step = self.step * 0.5
            if new_step < 1.0:
                self.step = 0
            else:
                self.step = max(1, int(new_step))
        else:
            self.step = self.step * 0.5
            if self.step < self.min_step:
                self.step = 0
                
        return self.step > 0

# Master definitions
PARAM_DEFINITIONS = [
    ParamDef("OSMIUM_EMA_ALPHA", False, 0.30, 0.05, 0.01, (0.01, 1.00)),
    ParamDef("OSMIUM_INVENTORY_SKEW", False, 0.06, 0.02, 0.005, (0.00, 0.50)),
    ParamDef("OSMIUM_SKEW_POWER", False, 2.00, 0.50, 0.10, (0.50, 5.00)),
    ParamDef("OSMIUM_ACCUM_FLOOR", False, 0.00, 0.05, 0.01, (0.00, 1.00)),
    ParamDef("OSMIUM_UNWIND_CEILING", False, 2.00, 0.20, 0.05, (1.00, 5.00)),
    ParamDef("OSMIUM_TAKE_UNWIND_WIDTH", True, 1, 1, 1, (0, 5)),
    ParamDef("OSMIUM_TAKE_ACCUM_WIDTH", True, 0, 1, 1, (-3, 3)),
    ParamDef("OSMIUM_SYMMETRIC_ZONE", True, 15, 5, 1, (0, 40)),
    ParamDef("OSMIUM_INNER_QUOTE_OFFSET", True, 0, 1, 1, (-2, 5)),
    ParamDef("OSMIUM_OUTER_QUOTE_OFFSET", True, 1, 1, 1, (0, 8)),
    ParamDef("OSMIUM_INNER_QTY_RATIO", False, 0.90, 0.10, 0.02, (0.00, 1.00)),
    ParamDef("OSMIUM_MOMENTUM_QUOTE_SHIFT", True, 4, 1, 1, (0, 10)),
    ParamDef("OSMIUM_MOMENTUM_AGRESS_SCALE", False, 1.70, 0.20, 0.05, (0.50, 4.00)),
    ParamDef("OSMIUM_MOMENTUM_DEFENSE_SCALE", False, 1.20, 0.20, 0.05, (0.50, 4.00)),
]

def _patch_strategy_text(src: str, param_dict: Dict[str, float]) -> str:
    out = src
    for name, value in param_dict.items():
        is_int_param = next(p.is_int for p in PARAM_DEFINITIONS if p.name == name)
        if is_int_param:
            pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+)(\s*#.*)?$"
            out, n = re.subn(pattern, rf"\g<1>{int(value)}\g<3>", out, flags=re.MULTILINE)
        else:
            pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+(?:\.\d+)?)(\s*#.*)?$"
            out, n = re.subn(pattern, rf"\g<1>{value:.3f}\g<3>", out, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Expected exactly 1 match for {name}, found {n}")
    return out

def _run_backtest_eval(algo_path: Path, days: list[str]) -> Tuple[float, List[int]]:
    cmd = [
        sys.executable,
        "-m", "prosperity4bt",
        str(algo_path),
        *days,
        "--data", str(REPO_ROOT / "data"),
        "--no-progress", "--no-out",
        "--limit", "INTARIAN_PEPPER_ROOT:80",
        "--limit", "ASH_COATED_OSMIUM:80",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "backtest failed")

    matches = re.findall(r"^ASH_COATED_OSMIUM:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if len(matches) != len(days):
        raise RuntimeError(f"Expected {len(days)} PnLs for Osmium, got {len(matches)}.\n{proc.stdout[-1500:]}")
    
    pnls = [int(m.replace(",", "")) for m in matches]
    mean_pnl = sum(pnls) / len(pnls)
    std_pnl = float(numpy.std(pnls))
    score = mean_pnl - (PNL_VARIANCE_PENALTY * std_pnl)
    
    return float(score), pnls

def _eval_one(param_dict: Dict[str, float], base_src: str) -> Tuple[float, List[int], Dict[str, float]]:
    with tempfile.TemporaryDirectory(prefix="osmium_walker_") as tmpdir:
        algo = Path(tmpdir) / "strategy_variant.py"
        algo.write_text(_patch_strategy_text(base_src, param_dict), encoding="utf-8")
        score, pnls = _run_backtest_eval(algo, DAYS)
    return score, pnls, param_dict

def dict_from_defs(defs: List[ParamDef]) -> Dict[str, float]:
    return {p.name: p.curr_val for p in defs}

def main() -> int:
    base_src = STRATEGY_PATH.read_text(encoding="utf-8")
    current_params = copy.deepcopy(PARAM_DEFINITIONS)
    
    print(f"Starting Coordinate Ascent Walker Optimization")
    print(f"Workers: {DEFAULT_WORKERS} | Penalty Weight: {PNL_VARIANCE_PENALTY}")
    
    # Eval initial configuration
    init_dict = dict_from_defs(current_params)
    current_score, current_pnls, _ = _eval_one(init_dict, base_src)
    
    def log_score(score, pnls):
        return f"{score:>8.1f} (Days: {pnls})"
        
    print(f"\nInitial Score: {log_score(current_score, current_pnls)}")
    
    step_phase = 1
    
    while True:
        active_params = [p for p in current_params if p.step > 0]
        if not active_params:
            print("\nAll parameters have reached their minimum step sizes. Optimization Concluded.")
            break
            
        print(f"\n=== Step Phase {step_phase} (Active params: {len(active_params)}) ===")
        
        improvement_in_phase = False
        
        # Keep cycling to climb until no improvements are found at THIS step step resolution
        epoch = 1
        while True:
            improvement_in_epoch = False
            
            for param in current_params:
                if param.step <= 0:
                    continue
                    
                neighbors = param.get_neighbors()
                if not neighbors:
                    continue
                    
                # Prepare dictionaries for parallel eval
                eval_dicts = []
                for val in neighbors:
                    d = dict_from_defs(current_params)
                    d[param.name] = val
                    eval_dicts.append(d)
                    
                # Run parallel eval
                results = []
                with ProcessPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
                    futures = [pool.submit(_eval_one, d, base_src) for d in eval_dicts]
                    for fut in as_completed(futures):
                        results.append(fut.result())
                
                # Check for strictly better score
                best_neighbor_score = current_score
                best_neighbor_val = param.curr_val
                best_neighbor_pnls = current_pnls
                
                for r_score, r_pnls, r_dict in results:
                    if r_score > best_neighbor_score:
                        best_neighbor_score = r_score
                        best_neighbor_val = r_dict[param.name]
                        best_neighbor_pnls = r_pnls
                        
                if best_neighbor_score > current_score:
                    print(f"  [Epoch {epoch}] {param.name}: {param.curr_val} -> {best_neighbor_val} | Score {current_score:.1f} -> {log_score(best_neighbor_score, best_neighbor_pnls)}")
                    param.curr_val = best_neighbor_val
                    current_score = best_neighbor_score
                    current_pnls = best_neighbor_pnls
                    improvement_in_epoch = True
                    improvement_in_phase = True
                    
            if not improvement_in_epoch:
                print(f"  [Epoch {epoch}] Local maximum reached at current step resolution.")
                break
                
            epoch += 1
            
        # Shrink step sizes for the next phase
        print(f"Shrinking step sizes...")
        for p in current_params:
            p.shrink_step()
            
        step_phase += 1
        
    print("\n=== FINAL OPTIMIZED PARAMETERS ===")
    print(f"Final Score: {log_score(current_score, current_pnls)}")
    for p in current_params:
        if p.is_int:
            print(f"{p.name} = {int(p.curr_val)}")
        else:
            print(f"{p.name} = {p.curr_val:.3f}")
            
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
