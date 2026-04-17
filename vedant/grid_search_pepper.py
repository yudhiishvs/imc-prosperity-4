"""
Parallel stage-2 coarse grid search for INTARIAN_PEPPER_ROOT parameters in vedant/strategy.py.

This version includes BOTH:
  1) Original pepper knobs (acc/scalp/max_vol/recoup)
  2) New post-reach MM knobs (mm size / bid weight / min long position)

Run from repo root:
  python3 -u ./vedant/grid_search_pepper.py
"""

from __future__ import annotations

import itertools
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = REPO_ROOT / "vedant" / "strategy.py"

# --- stage-2 coarse search config ---
# Evaluate all 3 days to reduce overfitting to any single day.
COARSE_DAYS = ["1--2", "1--1", "1-0"]

# Ultra-Fine Search centered on the absolute peak of the Top 100
ACC_THRESH_VALUES = [8]
SCALP_MIN_MARGIN_VALUES = [4]
MAX_SCALP_VOLUME_VALUES = [3]
RECOUP_MAX_MARGIN_VALUES = [-2]

MM_BASE_QTY_VALUES = [14, 15, 16, 17]
MM_BID_WEIGHT_VALUES = [0.40, 0.45, 0.50]
MM_MIN_LONG_POS_VALUES = [50, 55, 60, 65]

MM_L2_MAX_BID_GAP_VALUES = [5, 6, 7, 8]
MM_L2_MAX_ASK_GAP_VALUES = [5]

# OIM configurations
OIM_BASE_THRESH_VALUES = [0.0, 0.15, 0.3, 0.45]
OIM_MAX_SHIFT_VALUES = [1, 2, 3]

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)


@dataclass(frozen=True)
class PepperParams:
    acc_thresh: int
    scalp_min_margin: int
    max_scalp_volume: int
    recoup_max_margin: int
    mm_base_qty: int
    mm_bid_weight: float
    mm_min_long_pos: int
    mm_l2_max_bid_gap: int
    mm_l2_max_ask_gap: int
    oim_base_thresh: float
    oim_max_shift: int


def _patch_strategy_text(src: str, p: PepperParams) -> str:
    def sub_int(name: str, value: int, text: str) -> str:
        pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+)(\s*#.*)?$"
        out, n = re.subn(pattern, rf"\g<1>{value}\g<3>", text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Expected exactly 1 match for {name}, found {n}")
        return out

    def sub_float(name: str, value: float, text: str) -> str:
        pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+(?:\.\d+)?)(\s*#.*)?$"
        out, n = re.subn(pattern, rf"\g<1>{value:.2f}\g<3>", text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Expected exactly 1 match for {name}, found {n}")
        return out

    out = src
    out = sub_int("PEPPER_INITIAL_ACC_THRESH", p.acc_thresh, out)
    out = sub_int("PEPPER_SCALP_MIN_MARGIN", p.scalp_min_margin, out)
    out = sub_int("PEPPER_MAX_SCALP_VOLUME", p.max_scalp_volume, out)
    out = sub_int("PEPPER_RECOUP_MAX_MARGIN", p.recoup_max_margin, out)
    out = sub_int("PEPPER_MM_BASE_QUOTE_SIZE", p.mm_base_qty, out)
    out = sub_float("PEPPER_MM_BID_WEIGHT", p.mm_bid_weight, out)
    out = sub_int("PEPPER_MM_MIN_LONG_POSITION", p.mm_min_long_pos, out)
    out = sub_int("PEPPER_MM_L2_MAX_BID_GAP", p.mm_l2_max_bid_gap, out)
    out = sub_int("PEPPER_MM_L2_MAX_ASK_GAP", p.mm_l2_max_ask_gap, out)
    out = sub_float("PEPPER_OIM_BASE_THRESHOLD", p.oim_base_thresh, out)
    out = sub_int("PEPPER_OIM_MAX_SHIFT", p.oim_max_shift, out)
    return out


def _run_backtest(algo_path: Path, days: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "prosperity4bt",
        str(algo_path),
        *days,
        "--data",
        str(REPO_ROOT / "data"),
        "--no-progress",
        "--no-out",
        "--limit",
        "INTARIAN_PEPPER_ROOT:80",
        "--limit",
        "ASH_COATED_OSMIUM:80",
    ]
    if len(days) > 1:
        cmd.append("--merge-pnl")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "backtest failed")

    matches = re.findall(r"^INTARIAN_PEPPER_ROOT:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError("Could not find INTARIAN_PEPPER_ROOT PnL in output:\n" + proc.stdout[-1500:])
    
    total_pnl = sum(int(m.replace(",", "")) for m in matches)
    return total_pnl


def _iter_params():
    seen = set()
    for (
        acc_thresh, scalp_min, max_vol, recoup_max, mm_qty, mm_w, mm_min_pos, l2_bid_gap, l2_ask_gap, oim_thresh, oim_max
    ) in itertools.product(
        ACC_THRESH_VALUES,
        SCALP_MIN_MARGIN_VALUES,
        MAX_SCALP_VOLUME_VALUES,
        RECOUP_MAX_MARGIN_VALUES,
        MM_BASE_QTY_VALUES,
        MM_BID_WEIGHT_VALUES,
        MM_MIN_LONG_POS_VALUES,
        MM_L2_MAX_BID_GAP_VALUES,
        MM_L2_MAX_ASK_GAP_VALUES,
        OIM_BASE_THRESH_VALUES,
        OIM_MAX_SHIFT_VALUES,
    ):
        # Deduplicate: if bid weight is 1.0, we never post asks, so min_long_pos has no effect
        if mm_w == 1.0:
            mm_min_pos = 80
            
        p = PepperParams(
            acc_thresh, scalp_min, max_vol, recoup_max, mm_qty, mm_w, mm_min_pos, l2_bid_gap, l2_ask_gap, oim_thresh, oim_max
        )
        seen.add(p)
        
    unique_list = sorted(list(seen), key=lambda x: str(x))
    
    # 3^7 = 2187 combinations. To stay under 1000 without sacrificing resolution, we randomly subsample 900.
    if len(unique_list) > 1000:
        rng = random.Random(42)
        unique_list = rng.sample(unique_list, 1000)
        
    for p in unique_list:
        yield p


def _eval_one(p: PepperParams, base_src: str, days: list[str]) -> tuple[int, PepperParams]:
    with tempfile.TemporaryDirectory(prefix="pepper_grid_worker_") as tmpdir:
        algo = Path(tmpdir) / "strategy_variant.py"
        algo.write_text(_patch_strategy_text(base_src, p), encoding="utf-8")
        pnl = _run_backtest(algo, days)
    return pnl, p


def main() -> int:
    base_src = STRATEGY_PATH.read_text(encoding="utf-8")

    params = list(_iter_params())
    total = len(params)
    workers = DEFAULT_WORKERS
    print(f"Coarse days: {COARSE_DAYS}", flush=True)
    print(f"Total grid points: {total} | workers: {workers}", flush=True)

    results: list[tuple[int, PepperParams]] = []
    done = 0

    def format_log(p: PepperParams, pnl: int) -> str:
        return (
            f"pnl={pnl:>8}  "
            f"PEPPER_INITIAL_ACC_THRESH={p.acc_thresh} "
            f"PEPPER_SCALP_MIN_MARGIN={p.scalp_min_margin} "
            f"PEPPER_MAX_SCALP_VOLUME={p.max_scalp_volume} "
            f"PEPPER_RECOUP_MAX_MARGIN={p.recoup_max_margin} "
            f"PEPPER_MM_BASE_QUOTE_SIZE={p.mm_base_qty} "
            f"PEPPER_MM_BID_WEIGHT={p.mm_bid_weight:.2f} "
            f"PEPPER_MM_MIN_LONG_POSITION={p.mm_min_long_pos} "
            f"PEPPER_MM_L2_MAX_BID_GAP={p.mm_l2_max_bid_gap} "
            f"PEPPER_MM_L2_MAX_ASK_GAP={p.mm_l2_max_ask_gap} "
            f"PEPPER_OIM_BASE_THRESHOLD={p.oim_base_thresh:.2f} "
            f"PEPPER_OIM_MAX_SHIFT={p.oim_max_shift}"
        )

    print(f"\nStarting evaluation of {total} parameters randomly sampled from the total permutation space using {DEFAULT_WORKERS} workers.\n", flush=True)

    with ProcessPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
        futures = {
            pool.submit(_eval_one, p, base_src, COARSE_DAYS): p
            for p in params
        }
        for fut in as_completed(futures):
            pnl, p = fut.result()
            done += 1
            results.append((pnl, p))
            print(f"[{done:>4}/{total}] {format_log(p, pnl)}", flush=True)

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n=== BEST (coarse) ===", flush=True)
    best_pnl, best_p = results[0]
    print(format_log(best_p, best_pnl), flush=True)

    print("\n=== TOP 100 (coarse) ===", flush=True)
    for rank, (pnl, p) in enumerate(results[:100], start=1):
        print(f"{rank:>2}. {format_log(p, pnl)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
