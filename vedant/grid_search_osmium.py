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
class OsmiumParams:
    oim_multiplier: float
    base_quote_size: int
    volume_skew_aggression: float
    emergency_threshold: int
    emergency_target: int


def _sub_number(name: str, value: float | int, text: str) -> str:
    # Replace only the numeric literal after '=' on the constant line.
    pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+(?:\.\d+)?)(\s*#.*)?$"
    repl = rf"\g<1>{value}\g<3>"
    out, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Expected exactly 1 match for {name}, found {n}")
    return out


def _patch_strategy_text(src: str, p: OsmiumParams) -> str:
    out = src
    out = _sub_number("OSMIUM_OIM_MULTIPLIER", p.oim_multiplier, out)
    out = _sub_number("OSMIUM_BASE_QUOTE_SIZE", p.base_quote_size, out)
    out = _sub_number("OSMIUM_VOLUME_SKEW_AGGRESSION", p.volume_skew_aggression, out)
    out = _sub_number("OSMIUM_EMERGENCY_THRESHOLD", p.emergency_threshold, out)
    out = _sub_number("OSMIUM_EMERGENCY_TARGET", p.emergency_target, out)
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
        "ASH_COATED_OSMIUM:80",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "backtest failed")

    m = re.search(r"^ASH_COATED_OSMIUM:\s*([-0-9,]+)\s*$", proc.stdout, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find ASH_COATED_OSMIUM PnL in output:\n" + proc.stdout[-1500:])
    return int(m.group(1).replace(",", ""))


def _grid(values_mult, values_base, values_aggr, values_thresh, values_target):
    for mult, base, aggr, thresh, tgt in itertools.product(
        values_mult, values_base, values_aggr, values_thresh, values_target
    ):
        if tgt >= thresh:
            continue
        yield OsmiumParams(mult, base, aggr, thresh, tgt)


def main() -> int:
    base_src = STRATEGY_PATH.read_text(encoding="utf-8")

    # Coarse grid (kept modest so you can run locally without waiting forever).
    # You can expand these lists once you see where the optimum region is.
    mults = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    bases = [5, 9, 15, 20, 25]
    aggrs = [0.5, 1.0, 1.5, 2.0]
    thresholds = [60, 65, 70, 75]
    targets = [30, 40, 50, 55]

    params = list(_grid(mults, bases, aggrs, thresholds, targets))
    total = len(params)

    results: list[tuple[int, OsmiumParams]] = []

    with tempfile.TemporaryDirectory(prefix="osmium_grid_") as tmpdir:
        tmpdir = Path(tmpdir)
        algo = tmpdir / "strategy_variant.py"

        for i, p in enumerate(params, start=1):
            algo.write_text(_patch_strategy_text(base_src, p), encoding="utf-8")
            pnl = _run_backtest(algo)
            results.append((pnl, p))
            print(
                f"[{i:>4}/{total}] pnl={pnl:>7}  "
                f"mult={p.oim_multiplier:<4} base={p.base_quote_size:<2} aggr={p.volume_skew_aggression:<3} "
                f"thr={p.emergency_threshold:<2} tgt={p.emergency_target:<2}"
            )

    results.sort(key=lambda x: x[0], reverse=True)

    print("\n=== BEST (coarse) ===")
    best_pnl, best_p = results[0]
    print(
        f"pnl={best_pnl}  mult={best_p.oim_multiplier} base={best_p.base_quote_size} "
        f"aggr={best_p.volume_skew_aggression} thr={best_p.emergency_threshold} tgt={best_p.emergency_target}"
    )

    print("\n=== TOP 15 (coarse) ===")
    for rank, (pnl, p) in enumerate(results[:15], start=1):
        print(
            f"{rank:>2}. pnl={pnl:>7}  mult={p.oim_multiplier:<4} base={p.base_quote_size:<2} "
            f"aggr={p.volume_skew_aggression:<3} thr={p.emergency_threshold:<2} tgt={p.emergency_target:<2}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

