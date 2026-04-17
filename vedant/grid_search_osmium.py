"""
Parallel coarse grid search for ASH_COATED_OSMIUM parameters in vedant/updated_osmium.py.

Run from repo root:
  python3 -u ./vedant/grid_search_osmium.py
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
STRATEGY_PATH = REPO_ROOT / "vedant" / "updated_osmium.py"

# --- Grid Config ---
COARSE_DAYS = ["1--2", "1--1", "1-0"]

# Fair Value
EMA_ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4]

# A-S Inventory Cost (reservation = ema - pos * skew)
INVENTORY_SKEW_VALUES = [0.02, 0.04, 0.06, 0.08, 0.12]

# Volume Scaling (power curve)
SKEW_POWER_VALUES = [1.0, 1.5, 2.0, 3.0]
ACCUM_FLOOR_VALUES = [0.0, 0.05, 0.15]
UNWIND_CEILING_VALUES = [1.5, 2.0, 2.5]

# Taking Policy
TAKE_UNWIND_WIDTH_VALUES = [0, 1, 2, 3]
TAKE_ACCUM_WIDTH_VALUES = [0, 1, 2]
SYMMETRIC_ZONE_VALUES = [10, 15, 25]

# Quote Structure
INNER_QUOTE_OFFSET_VALUES = [0, 2, 4]
OUTER_QUOTE_OFFSET_VALUES = [1, 3, 5]
INNER_QTY_RATIO_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Momentum Fade
MOMENTUM_QUOTE_SHIFT_VALUES = [0, 2, 4]
MOMENTUM_AGRESS_SCALE_VALUES = [1.0, 1.35, 1.7]
MOMENTUM_DEFENSE_SCALE_VALUES = [0.8, 1.0, 1.2]

DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)


@dataclass(frozen=True)
class OsmiumParams:
    ema_alpha: float
    inventory_skew: float
    skew_power: float
    accum_floor: float
    unwind_ceiling: float
    take_unwind_width: int
    take_accum_width: int
    symmetric_zone: int
    inner_offset: int
    outer_offset: int
    inner_qty_ratio: float
    mom_quote_shift: int
    mom_agress: float
    mom_defense: float


def _patch_strategy_text(src: str, p: OsmiumParams) -> str:
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
    out = sub_float("OSMIUM_EMA_ALPHA", p.ema_alpha, out)
    out = sub_float("OSMIUM_INVENTORY_SKEW", p.inventory_skew, out)
    out = sub_float("OSMIUM_SKEW_POWER", p.skew_power, out)
    out = sub_float("OSMIUM_ACCUM_FLOOR", p.accum_floor, out)
    out = sub_float("OSMIUM_UNWIND_CEILING", p.unwind_ceiling, out)
    out = sub_int("OSMIUM_TAKE_UNWIND_WIDTH", p.take_unwind_width, out)
    out = sub_int("OSMIUM_TAKE_ACCUM_WIDTH", p.take_accum_width, out)
    out = sub_int("OSMIUM_SYMMETRIC_ZONE", p.symmetric_zone, out)
    out = sub_int("OSMIUM_INNER_QUOTE_OFFSET", p.inner_offset, out)
    out = sub_int("OSMIUM_OUTER_QUOTE_OFFSET", p.outer_offset, out)
    out = sub_float("OSMIUM_INNER_QTY_RATIO", p.inner_qty_ratio, out)
    out = sub_int("OSMIUM_MOMENTUM_QUOTE_SHIFT", p.mom_quote_shift, out)
    out = sub_float("OSMIUM_MOMENTUM_AGRESS_SCALE", p.mom_agress, out)
    out = sub_float("OSMIUM_MOMENTUM_DEFENSE_SCALE", p.mom_defense, out)
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

    matches = re.findall(r"^ASH_COATED_OSMIUM:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError("Could not find ASH_COATED_OSMIUM PnL in output:\n" + proc.stdout[-1500:])
    
    total_pnl = sum(int(m.replace(",", "")) for m in matches)
    return total_pnl


def _iter_params():
    seen = set()
    for (
        e_alpha, inv_skew, s_power, a_floor, u_ceil,
        t_unwind, t_accum, sym_zone,
        in_off, out_off, in_ratio,
        m_shift, m_agr, m_def
    ) in itertools.product(
        EMA_ALPHA_VALUES,
        INVENTORY_SKEW_VALUES,
        SKEW_POWER_VALUES,
        ACCUM_FLOOR_VALUES,
        UNWIND_CEILING_VALUES,
        TAKE_UNWIND_WIDTH_VALUES,
        TAKE_ACCUM_WIDTH_VALUES,
        SYMMETRIC_ZONE_VALUES,
        INNER_QUOTE_OFFSET_VALUES,
        OUTER_QUOTE_OFFSET_VALUES,
        INNER_QTY_RATIO_VALUES,
        MOMENTUM_QUOTE_SHIFT_VALUES,
        MOMENTUM_AGRESS_SCALE_VALUES,
        MOMENTUM_DEFENSE_SCALE_VALUES,
    ):
        # No invalid-combination constraints for the new params (they're all independent)
        p = OsmiumParams(
            e_alpha, inv_skew, s_power, a_floor, u_ceil,
            t_unwind, t_accum, sym_zone,
            in_off, out_off, in_ratio,
            m_shift, m_agr, m_def
        )
        seen.add(p)
        
    unique_list = sorted(list(seen), key=lambda x: str(x))
    
    if len(unique_list) > 1000:
        rng = random.Random(42)
        unique_list = rng.sample(unique_list, 1000)
        
    for p in unique_list:
        yield p


def _eval_one(p: OsmiumParams, base_src: str, days: list[str]) -> tuple[int, OsmiumParams]:
    with tempfile.TemporaryDirectory(prefix="osmium_grid_worker_") as tmpdir:
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
    print(f"Total grid points (random subset): {total} | workers: {workers}", flush=True)

    results: list[tuple[int, OsmiumParams]] = []
    done = 0

    def format_log(p: OsmiumParams, pnl: int) -> str:
        return (
            f"pnl={pnl:>8}  "
            f"EMA={p.ema_alpha:.2f} "
            f"SKEW={p.inventory_skew:.2f} "
            f"POW={p.skew_power:.1f} "
            f"FLOOR={p.accum_floor:.2f} "
            f"CEIL={p.unwind_ceiling:.1f} "
            f"TUW={p.take_unwind_width} "
            f"TAW={p.take_accum_width} "
            f"SZ={p.symmetric_zone} "
            f"IQO={p.inner_offset} "
            f"OQO={p.outer_offset} "
            f"IQR={p.inner_qty_ratio:.2f} "
            f"MQS={p.mom_quote_shift} "
            f"MAG={p.mom_agress:.2f} "
            f"MDF={p.mom_defense:.2f}"
        )

    print(f"\nStarting evaluation of {total} parameters randomly sampled from the space using {DEFAULT_WORKERS} workers.\n", flush=True)

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
