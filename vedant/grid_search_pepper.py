import itertools
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_PATH = REPO_ROOT / "vedant" / "strategy.py"


@dataclass(frozen=True)
class PepperParams:
    initial_acc_thresh: int
    scalp_min_margin: int
    recoup_max_margin: int


def _patch_strategy_text(src: str, p: PepperParams) -> str:
    def sub_int(name: str, value: int, text: str) -> str:
        # Keep comments/spacing; replace only the integer literal after '='.
        pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+)(\s*#.*)?$"
        out, n = re.subn(pattern, rf"\g<1>{value}\g<3>", text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Expected exactly 1 match for {name}, found {n}")
        return out

    out = src
    out = sub_int("PEPPER_INITIAL_ACC_THRESH", p.initial_acc_thresh, out)
    out = sub_int("PEPPER_SCALP_MIN_MARGIN", p.scalp_min_margin, out)
    out = sub_int("PEPPER_RECOUP_MAX_MARGIN", p.recoup_max_margin, out)
    return out


def _run_backtest(algo_path: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "prosperity4bt",
        str(algo_path),
        "1--2",
        "1--1",
        "1-0",
        "--data",
        str(REPO_ROOT / "data"),
        "--no-progress",
        "--no-out",
        "--merge-pnl",
        "--limit",
        "INTARIAN_PEPPER_ROOT:80",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "backtest failed")

    m = re.search(r"^INTARIAN_PEPPER_ROOT:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find INTARIAN_PEPPER_ROOT PnL in output:\n" + proc.stdout[-1000:])
    return int(m.group(1).replace(",", ""))


def _grid(values_initial, values_scalp, values_recoup):
    for a, s, r in itertools.product(values_initial, values_scalp, values_recoup):
        yield PepperParams(a, s, r)


def main() -> int:
    base_src = STRATEGY_PATH.read_text(encoding="utf-8")

    # Coarse grid (125 runs): good speed/coverage tradeoff.
    coarse_initial = [7, 8, 9, 10, 11]
    coarse_scalp = [3, 4, 5, 6, 7]
    coarse_recoup = [1, 3, 5, 7, 9]

    results: list[tuple[int, PepperParams]] = []

    with tempfile.TemporaryDirectory(prefix="pepper_grid_") as tmpdir:
        tmpdir = Path(tmpdir)
        algo = tmpdir / "strategy_variant.py"

        total = len(coarse_initial) * len(coarse_scalp) * len(coarse_recoup)
        i = 0
        for p in _grid(coarse_initial, coarse_scalp, coarse_recoup):
            i += 1
            algo.write_text(_patch_strategy_text(base_src, p), encoding="utf-8")
            pnl = _run_backtest(algo)
            results.append((pnl, p))
            print(f"[coarse {i:>3}/{total}] pnl={pnl:>8}  acc={p.initial_acc_thresh} scalp={p.scalp_min_margin} recoup={p.recoup_max_margin}")

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:10]

    print("\n=== TOP 10 (coarse) ===")
    for rank, (pnl, p) in enumerate(top, start=1):
        print(f"{rank:>2}. pnl={pnl:>8}  acc={p.initial_acc_thresh} scalp={p.scalp_min_margin} recoup={p.recoup_max_margin}")

    # Refine around best coarse point (+/-2 acc, +/-2 scalp, +/-2 recoup) with step 1.
    best_pnl, best_p = results[0]
    refine_initial = sorted({x for x in range(best_p.initial_acc_thresh - 2, best_p.initial_acc_thresh + 3) if 3 <= x <= 15})
    refine_scalp = sorted({x for x in range(best_p.scalp_min_margin - 2, best_p.scalp_min_margin + 3) if 1 <= x <= 12})
    refine_recoup = sorted({x for x in range(best_p.recoup_max_margin - 2, best_p.recoup_max_margin + 3) if 0 <= x <= 15})

    refine_results: list[tuple[int, PepperParams]] = []
    with tempfile.TemporaryDirectory(prefix="pepper_refine_") as tmpdir:
        tmpdir = Path(tmpdir)
        algo = tmpdir / "strategy_variant.py"
        total = len(refine_initial) * len(refine_scalp) * len(refine_recoup)
        i = 0
        for p in _grid(refine_initial, refine_scalp, refine_recoup):
            i += 1
            algo.write_text(_patch_strategy_text(base_src, p), encoding="utf-8")
            pnl = _run_backtest(algo)
            refine_results.append((pnl, p))
            print(f"[refine {i:>3}/{total}] pnl={pnl:>8}  acc={p.initial_acc_thresh} scalp={p.scalp_min_margin} recoup={p.recoup_max_margin}")

    refine_results.sort(key=lambda x: x[0], reverse=True)
    best_ref_pnl, best_ref_p = refine_results[0]

    print("\n=== BEST (refine) ===")
    print(f"pnl={best_ref_pnl}  acc={best_ref_p.initial_acc_thresh} scalp={best_ref_p.scalp_min_margin} recoup={best_ref_p.recoup_max_margin}")

    print("\n=== TOP 10 (refine) ===")
    for rank, (pnl, p) in enumerate(refine_results[:10], start=1):
        print(f"{rank:>2}. pnl={pnl:>8}  acc={p.initial_acc_thresh} scalp={p.scalp_min_margin} recoup={p.recoup_max_margin}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

