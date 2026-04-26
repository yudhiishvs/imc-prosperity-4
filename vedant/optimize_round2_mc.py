from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


THIS_FILE = Path(__file__).resolve()
IMC_REPO_ROOT = THIS_FILE.parents[1]
WORKSPACE_ROOT = IMC_REPO_ROOT.parent
PROSPERITY4MCBT_ROOT = WORKSPACE_ROOT / "prosperity4mcbt"
ROUND2_MC_RUNNER = PROSPERITY4MCBT_ROOT / "round2" / "round2_monte_carlo.py"
DEFAULT_STRATEGY_PATH = IMC_REPO_ROOT / "vedant" / "strategy.py"
WORKER_DIR = IMC_REPO_ROOT / "vedant" / ".round2_mc_optimizer"


@dataclass
class ParamDef:
    name: str
    is_int: bool
    default: float
    jump: float
    min_jump: float
    bounds: tuple[float, float]


def default_params() -> list[ParamDef]:
    return [
        ParamDef("OSMIUM_EMA_ALPHA", False, 0.081, 0.008, 0.001, (0.0001, 0.2)),
        ParamDef("OSMIUM_INNER_OFFSET", True, 9.0, 1.0, 1.0, (0.0, 15.0)),
        ParamDef("OSMIUM_OUTER_OFFSET", True, 6.0, 1.0, 1.0, (0.0, 40.0)),
        ParamDef("OSMIUM_VOLUME_SKEW_AGGRESSION", False, 1.1, 0.2, 0.05, (0.0, 5.0)),
        ParamDef("OSMIUM_OIM_SHIFT", True, 2.0, 1.0, 1.0, (0.0, 10.0)),
        ParamDef("OSMIUM_BASE_QUOTE_SIZE", True, 70.0, 5.0, 1.0, (10.0, 120.0)),
        ParamDef("OSMIUM_L2_QUOTE_SIZE", True, 35.0, 4.0, 1.0, (5.0, 120.0)),
        ParamDef("OSMIUM_KILL_SWITCH_THRESHOLD", True, 80.0, 3.0, 1.0, (40.0, 100.0)),
        ParamDef("OSMIUM_OIM_THRESHOLD", False, 0.031, 0.03, 0.005, (0.0, 0.99)),
        ParamDef("OSMIUM_OIM_FADE_SCALE", False, 0.341, 0.1, 0.02, (0.0, 8.0)),
        ParamDef("OSMIUM_OIM_EDGE_SCALE", False, 3.389, 0.2, 0.05, (0.0, 12.0)),
        ParamDef("OSMIUM_OIM_TAKE_SCALE", False, 6.8, 0.3, 0.05, (0.0, 20.0)),
        ParamDef("OSMIUM_FV_TETHER_SCALE", False, 0.053, 0.02, 0.005, (0.0, 0.7)),
    ]


def default_pepper_params() -> list[ParamDef]:
    return [
        ParamDef("PEPPER_SLOPE", False, 0.001, 0.0004, 0.0001, (-0.02, 0.02)),
        ParamDef("PEPPER_INITIAL_ACC_THRESH", True, 8.0, 1.0, 1.0, (0.0, 20.0)),
        ParamDef("PEPPER_SCALP_MIN_MARGIN", True, 4.0, 1.0, 1.0, (0.0, 15.0)),
        ParamDef("PEPPER_MAX_SCALP_VOLUME", True, 3.0, 1.0, 1.0, (1.0, 25.0)),
        ParamDef("PEPPER_RECOUP_MAX_MARGIN", True, -2.0, 1.0, 1.0, (-20.0, 10.0)),
        ParamDef("PEPPER_MM_BASE_QUOTE_SIZE", True, 24.0, 2.0, 1.0, (5.0, 120.0)),
        ParamDef("PEPPER_MM_BID_WEIGHT", False, 0.45, 0.04, 0.01, (0.0, 1.0)),
        ParamDef("PEPPER_MM_MIN_LONG_POSITION", True, 60.0, 2.0, 1.0, (0.0, 80.0)),
        ParamDef("PEPPER_MM_L2_MAX_BID_GAP", True, 6.0, 1.0, 1.0, (0.0, 20.0)),
        ParamDef("PEPPER_MM_L2_MAX_ASK_GAP", True, 5.0, 1.0, 1.0, (0.0, 20.0)),
        ParamDef("PEPPER_OIM_BASE_THRESHOLD", False, 0.1, 0.03, 0.01, (0.0, 1.0)),
        ParamDef("PEPPER_OIM_MAX_SHIFT", True, 2.0, 1.0, 1.0, (0.0, 10.0)),
        ParamDef("PEPPER_BASE_UPDATE_ALPHA", False, 0.025, 0.01, 0.002, (0.001, 0.4)),
        ParamDef("PEPPER_RESID_EMA_ALPHA", False, 0.035, 0.01, 0.002, (0.001, 0.4)),
        ParamDef("PEPPER_TREND_EMA_ALPHA", False, 0.07, 0.02, 0.002, (0.001, 0.5)),
        ParamDef("PEPPER_SPREAD_EMA_ALPHA", False, 0.06, 0.02, 0.002, (0.001, 0.5)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient ascent optimizer for Round 2 Monte Carlo strategy parameters")
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY_PATH, help="Path to strategy.py with Trader class")
    parser.add_argument("--sessions", type=int, default=30, help="Monte Carlo sessions per objective evaluation")
    parser.add_argument("--ticks-per-day", type=int, default=4000, help="Ticks per simulated day")
    parser.add_argument("--days-per-session", type=int, default=1, help="Days simulated per session")
    parser.add_argument("--runs", type=int, default=3, help="Number of gradient-ascent restarts")
    parser.add_argument("--product", choices=["osmium", "pepper"], default="osmium", help="Product parameter family to optimize")
    parser.add_argument("--sample-sessions", type=int, default=0, help="Sample sessions written by MC runner")
    parser.add_argument("--seed", type=int, default=20260420, help="Base seed for MC runs")
    parser.add_argument(
        "--objective",
        choices=["auto", "total_mean", "total_sharpe", "osmium_mean", "pepper_mean"],
        default="auto",
        help="Objective to maximize (auto maps to osmium_mean or pepper_mean by --product)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Patience threshold for both normal and jitter phases (normal->jitter trigger, jitter->run stop)",
    )
    parser.add_argument("--step-growth", type=float, default=1.15, help="Multiply jumps after successful steps")
    parser.add_argument("--step-shrink", type=float, default=0.65, help="Multiply jumps after unsuccessful steps")
    parser.add_argument(
        "--jitter-multiplier",
        type=float,
        default=2.5,
        help="When normal mode stalls for --patience steps, multiply jump sizes by this factor and enter jitter mode",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=None,
        help="Optional subset of parameter names to optimize. Default: all configured params.",
    )
    parser.add_argument(
        "--init-json",
        type=Path,
        default=None,
        help="Optional JSON object with initial parameter values to override defaults",
    )
    parser.add_argument(
        "--fused-params",
        type=Path,
        default=PROSPERITY4MCBT_ROOT / "round2" / "calibration" / "fused_parameters" / "round2_fused_parameters.json",
        help="Fused parameter JSON for round2_monte_carlo.py",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep per-evaluation output directories for inspection (default cleans intermediate runs)",
    )
    return parser.parse_args()


def normalize_value(p: ParamDef, value: float) -> float:
    lo, hi = p.bounds
    clipped = max(lo, min(hi, value))
    if p.is_int:
        return float(int(round(clipped)))
    return round(float(clipped), 6)


def load_strategy_src(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"strategy file not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_assigned_values(src: str, param_defs: dict[str, ParamDef]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for name, p in param_defs.items():
        pattern = rf"^\s*{re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:#.*)?$"
        match = re.search(pattern, src, flags=re.MULTILINE)
        if match is None:
            continue
        value = float(match.group(1))
        extracted[name] = normalize_value(p, value)
    return extracted


def patch_strategy(src: str, params: dict[str, float], param_defs: dict[str, ParamDef]) -> str:
    out = src
    for name, value in params.items():
        p = param_defs[name]
        if p.is_int:
            replacement = str(int(round(value)))
        else:
            replacement = f"{float(value):.6f}".rstrip("0").rstrip(".")

        pattern = rf"(^\s*{re.escape(name)}\s*=\s*)(-?\d+(?:\.\d+)?)(\s*(?:#.*)?)$"
        out, n = re.subn(pattern, rf"\g<1>{replacement}\3", out, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Failed to patch param {name}; matches={n}")
    return out


def hash_params(params: dict[str, float], precision: int = 6) -> str:
    stable = [(k, round(float(v), precision)) for k, v in sorted(params.items())]
    return hashlib.sha256(json.dumps(stable, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def read_session_summary(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "total_pnl": float(row["total_pnl"]),
                    "ash_coated_osmium_pnl": float(row["ash_coated_osmium_pnl"]),
                    "intarian_pepper_root_pnl": float(row["intarian_pepper_root_pnl"]),
                }
            )
    return rows


def compute_objective(rows: list[dict[str, float]], objective: str) -> tuple[float, dict[str, float]]:
    if not rows:
        return -1e18, {"mean": -1e18, "std": 0.0, "sharpe": -1e18}

    total = [r["total_pnl"] for r in rows]
    osmium = [r["ash_coated_osmium_pnl"] for r in rows]
    pepper = [r["intarian_pepper_root_pnl"] for r in rows]

    mean_total = statistics.mean(total)
    std_total = statistics.stdev(total) if len(total) > 1 else 0.0
    sharpe_total = mean_total / std_total if std_total > 1e-9 else mean_total

    metrics = {
        "mean_total": mean_total,
        "std_total": std_total,
        "sharpe_total": sharpe_total,
        "mean_osmium": statistics.mean(osmium),
        "mean_pepper": statistics.mean(pepper),
    }

    if objective == "total_mean":
        return metrics["mean_total"], metrics
    if objective == "total_sharpe":
        return metrics["sharpe_total"], metrics
    if objective == "osmium_mean":
        return metrics["mean_osmium"], metrics
    if objective == "pepper_mean":
        return metrics["mean_pepper"], metrics
    raise ValueError(f"Unsupported objective: {objective}")


def format_params(params: dict[str, float], param_defs: dict[str, ParamDef]) -> str:
    lines = []
    for name in sorted(params.keys()):
        p = param_defs[name]
        val = params[name]
        if p.is_int:
            lines.append(f"{name} = {int(round(val))}")
        else:
            lines.append(f"{name} = {val:.6f}".rstrip("0").rstrip("."))
    return "\n".join(lines)


def resolve_objective(product: str, objective: str) -> str:
    if objective != "auto":
        return objective
    return "osmium_mean" if product == "osmium" else "pepper_mean"


def randomize_params(base: dict[str, float], selected_params: list[ParamDef]) -> dict[str, float]:
    out = dict(base)
    for p in selected_params:
        lo, hi = p.bounds
        if p.is_int:
            out[p.name] = float(random.randint(int(lo), int(hi)))
        else:
            out[p.name] = normalize_value(p, random.uniform(lo, hi))
    return out


def print_new_global_max(score: float, metrics: dict[str, float], params: dict[str, float], param_defs: dict[str, ParamDef], run_idx: int, iteration: int) -> None:
    print("\n" + "!" * 80)
    print(f"NEW GLOBAL MAX FOUND | run={run_idx} iter={iteration} | score={score:.6f}")
    print(f"metrics={json.dumps(metrics, indent=2)}")
    print("params:")
    print(format_params(params, param_defs))
    print("!" * 80)


class Evaluator:
    def __init__(
        self,
        strategy_path: Path,
        fused_params: Path,
        sessions: int,
        ticks_per_day: int,
        days_per_session: int,
        sample_sessions: int,
        objective: str,
        seed: int,
        param_defs: dict[str, ParamDef],
        keep_output: bool,
    ):
        self.strategy_path = strategy_path.resolve()
        self.fused_params = fused_params.resolve()
        self.sessions = sessions
        self.ticks_per_day = ticks_per_day
        self.days_per_session = days_per_session
        self.sample_sessions = sample_sessions
        self.objective = objective
        self.seed = seed
        self.param_defs = param_defs
        self.keep_output = keep_output

        self.strategy_src = load_strategy_src(self.strategy_path)
        self.cache: dict[tuple[tuple[str, float], ...], tuple[float, dict[str, float]]] = {}

        WORKER_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, params: dict[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple((k, round(float(v), 6)) for k, v in sorted(params.items()))

    def evaluate(self, params: dict[str, float]) -> tuple[float, dict[str, float]]:
        key = self._cache_key(params)
        if key in self.cache:
            return self.cache[key]

        run_tag = hash_params(params)
        run_dir = WORKER_DIR / f"eval_{run_tag}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        worker_strategy = run_dir / "strategy_eval.py"
        worker_strategy.write_text(patch_strategy(self.strategy_src, params, self.param_defs), encoding="utf-8")

        dashboard_out = run_dir / "dashboard.json"

        cmd = [
            sys.executable,
            str(ROUND2_MC_RUNNER),
            str(worker_strategy),
            "--fused-params",
            str(self.fused_params),
            "--sessions",
            str(self.sessions),
            "--days-per-session",
            str(self.days_per_session),
            "--ticks-per-day",
            str(self.ticks_per_day),
            "--sample-sessions",
            str(self.sample_sessions),
            "--seed",
            str(self.seed),
            "--out",
            str(dashboard_out),
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(PROSPERITY4MCBT_ROOT),
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": (
                    str(WORKSPACE_ROOT)
                    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
                ),
            },
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "Round2 Monte Carlo run failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        summary_path = run_dir / "session_summary.csv"
        if not summary_path.exists():
            raise RuntimeError(f"Missing session_summary.csv at {summary_path}")

        rows = read_session_summary(summary_path)
        score, metrics = compute_objective(rows, self.objective)

        if not self.keep_output:
            for child in run_dir.iterdir():
                if child.name not in {"session_summary.csv", "run_summary.csv", "run.log"}:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)

        self.cache[key] = (score, metrics)
        return score, metrics


def apply_delta(curr: dict[str, float], p: ParamDef, direction: int, jump: float) -> dict[str, float]:
    nxt = dict(curr)
    proposed = curr[p.name] + direction * jump
    nxt[p.name] = normalize_value(p, proposed)
    return nxt


def select_params(params: list[ParamDef], names: Iterable[str] | None) -> list[ParamDef]:
    if not names:
        return params
    name_set = set(names)
    selected = [p for p in params if p.name in name_set]
    missing = sorted(name_set - {p.name for p in selected})
    if missing:
        raise ValueError(f"Unknown params requested: {', '.join(missing)}")
    return selected


def load_init_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--init-json must be a JSON object mapping param_name -> value")
    out: dict[str, float] = {}
    for k, v in payload.items():
        out[str(k)] = float(v)
    return out


def walk(args: argparse.Namespace) -> None:
    all_params = default_params() if args.product == "osmium" else default_pepper_params()
    selected_params = select_params(all_params, args.params)
    param_defs = {p.name: p for p in selected_params}
    effective_objective = resolve_objective(args.product, args.objective)

    if not ROUND2_MC_RUNNER.exists():
        raise FileNotFoundError(f"Round2 MC runner not found: {ROUND2_MC_RUNNER}")

    init_overrides = load_init_overrides(args.init_json)
    strategy_src = load_strategy_src(args.strategy.resolve())
    strategy_defaults = extract_assigned_values(strategy_src, param_defs)
    base_start = {
        p.name: normalize_value(p, init_overrides.get(p.name, strategy_defaults.get(p.name, p.default)))
        for p in selected_params
    }

    evaluator = Evaluator(
        strategy_path=args.strategy,
        fused_params=args.fused_params,
        sessions=args.sessions,
        ticks_per_day=args.ticks_per_day,
        days_per_session=args.days_per_session,
        sample_sessions=args.sample_sessions,
        objective=effective_objective,
        seed=args.seed,
        param_defs=param_defs,
        keep_output=args.keep_output,
    )

    print("=" * 72)
    print("Round 2 Gradient-Ascent Walker")
    print(f"Product: {args.product}")
    print(f"Objective: {effective_objective}")
    print(f"Runs: {args.runs} | Stop mode: patience -> jitter -> patience")
    print(f"Sessions/eval: {args.sessions}, ticks/day: {args.ticks_per_day}, base_seed: {args.seed}")
    print("=" * 72)

    global_best: dict[str, float] | None = None
    global_best_score = -1e18
    global_best_metrics: dict[str, float] = {}
    all_runs: list[dict[str, object]] = []

    for run_idx in range(1, args.runs + 1):
        current = dict(base_start) if run_idx == 1 else randomize_params(base_start, selected_params)
        jumps = {p.name: p.jump for p in selected_params}
        run_seed = args.seed + (run_idx - 1) * 100003
        evaluator.seed = run_seed

        run_best = dict(current)
        run_best_score, run_best_metrics = evaluator.evaluate(run_best)
        current_score, current_metrics = run_best_score, run_best_metrics
        if run_best_score > global_best_score:
            global_best_score = run_best_score
            global_best = dict(run_best)
            global_best_metrics = dict(run_best_metrics)
            print_new_global_max(global_best_score, global_best_metrics, global_best, param_defs, run_idx, 0)

        print(f"\n{'-' * 72}")
        print(f"RUN {run_idx}/{args.runs} | seed={run_seed}")
        print(f"Start score: {current_score:.6f}")
        print("Start params:")
        print(format_params(current, param_defs))

        mode = "normal"
        no_improve = 0
        run_history: list[dict[str, float | int | str]] = []
        itr = 0

        while True:
            itr += 1
            gains: dict[str, float] = {}
            preferred_dir: dict[str, int] = {}

            print(f"\n[Run {run_idx} | Iter {itr} | mode={mode}] base score={current_score:.6f}")

            for p in selected_params:
                h = max(p.min_jump, jumps[p.name])
                plus_cfg = apply_delta(current, p, +1, h)
                minus_cfg = apply_delta(current, p, -1, h)

                plus_score, _ = evaluator.evaluate(plus_cfg)
                minus_score, _ = evaluator.evaluate(minus_cfg)

                delta_denom = plus_cfg[p.name] - minus_cfg[p.name]
                grad = 0.0 if abs(delta_denom) < 1e-12 else (plus_score - minus_score) / delta_denom

                if plus_score > minus_score and plus_score > current_score:
                    preferred_dir[p.name] = 1
                    gains[p.name] = plus_score - current_score
                elif minus_score > plus_score and minus_score > current_score:
                    preferred_dir[p.name] = -1
                    gains[p.name] = minus_score - current_score
                else:
                    preferred_dir[p.name] = 0
                    gains[p.name] = 0.0

                print(
                    f"  {p.name:<33} h={h:<7.4f} plus={plus_score:>10.4f} "
                    f"minus={minus_score:>10.4f} grad={grad:>10.4f}"
                )

            candidate = dict(current)
            moved = []
            for p in selected_params:
                direction = preferred_dir[p.name]
                if direction == 0:
                    continue
                step = max(p.min_jump, jumps[p.name])
                nxt = normalize_value(p, candidate[p.name] + direction * step)
                if math.isclose(nxt, candidate[p.name], rel_tol=0, abs_tol=1e-12):
                    continue
                candidate[p.name] = nxt
                moved.append((p.name, direction, step))

            improved = False
            if moved:
                cand_score, cand_metrics = evaluator.evaluate(candidate)
                print(f"  Candidate(all improving dims) score={cand_score:.6f}")
                if cand_score > current_score:
                    improved = True
                    current, current_score, current_metrics = candidate, cand_score, cand_metrics

            if not improved:
                best_single: tuple[str, dict[str, float], float, dict[str, float]] | None = None
                sorted_dims = sorted(selected_params, key=lambda pd: gains[pd.name], reverse=True)
                for p in sorted_dims:
                    if gains[p.name] <= 0:
                        break
                    step = max(p.min_jump, jumps[p.name])
                    direction = preferred_dir[p.name]
                    trial = apply_delta(current, p, direction, step)
                    trial_score, trial_metrics = evaluator.evaluate(trial)
                    if best_single is None or trial_score > best_single[2]:
                        best_single = (p.name, trial, trial_score, trial_metrics)

                if best_single is not None and best_single[2] > current_score:
                    improved = True
                    chosen = best_single[0]
                    current = best_single[1]
                    current_score = best_single[2]
                    current_metrics = best_single[3]
                    print(f"  Accepted single-dim move on {chosen} -> score={current_score:.6f}")

            if improved:
                no_improve = 0
                for p in selected_params:
                    jumps[p.name] = min(p.bounds[1] - p.bounds[0], max(p.min_jump, jumps[p.name] * args.step_growth))
                print("  Accepted. Expanded jump sizes.")
            else:
                no_improve += 1
                for p in selected_params:
                    jumps[p.name] = max(p.min_jump, jumps[p.name] * args.step_shrink)
                print("  No improvement. Shrunk jump sizes.")

            if current_score > run_best_score:
                run_best = dict(current)
                run_best_score = current_score
                run_best_metrics = dict(current_metrics)
                if run_best_score > global_best_score:
                    global_best_score = run_best_score
                    global_best = dict(run_best)
                    global_best_metrics = dict(run_best_metrics)
                    print_new_global_max(global_best_score, global_best_metrics, global_best, param_defs, run_idx, itr)

            run_history.append(
                {
                    "iteration": itr,
                    "score": current_score,
                    "best_score": run_best_score,
                    "no_improve": no_improve,
                    "mode": mode,
                }
            )

            print(f"  Current score={current_score:.6f} | Run-best score={run_best_score:.6f}")
            if mode == "normal" and no_improve >= args.patience:
                mode = "jitter"
                no_improve = 0
                for p in selected_params:
                    jumps[p.name] = min(p.bounds[1] - p.bounds[0], max(p.min_jump, jumps[p.name] * args.jitter_multiplier))
                print(f"Entering jitter mode (step sizes multiplied by {args.jitter_multiplier})")
                continue

            if mode == "jitter" and no_improve >= args.patience:
                print(f"Ending run: no improvements for {no_improve} consecutive jitter steps")
                break

        print(f"\nRUN {run_idx} BEST SUMMARY")
        print(f"best_score={run_best_score:.6f}")
        print(f"best_metrics={json.dumps(run_best_metrics, indent=2)}")
        print("best_params:")
        print(format_params(run_best, param_defs))

        all_runs.append(
            {
                "run_index": run_idx,
                "seed": run_seed,
                "best_score": run_best_score,
                "best_metrics": run_best_metrics,
                "best_params": run_best,
                "history": run_history,
            }
        )

    result = {
        "product": args.product,
        "objective": effective_objective,
        "best_score": global_best_score,
        "best_metrics": global_best_metrics,
        "best_params": global_best or {},
        "runs": all_runs,
        "config": {
            "sessions": args.sessions,
            "ticks_per_day": args.ticks_per_day,
            "days_per_session": args.days_per_session,
            "runs": args.runs,
            "patience": args.patience,
            "jitter_multiplier": args.jitter_multiplier,
            "seed": args.seed,
        },
    }

    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    result_file = WORKER_DIR / "best_result.json"
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Optimization finished")
    print(f"Global best score: {global_best_score:.6f}")
    print(f"Global best metrics: {json.dumps(global_best_metrics, indent=2)}")
    print("Global best params snippet:")
    if global_best is not None:
        print(format_params(global_best, param_defs))
    print(f"Saved result JSON: {result_file}")


def main() -> None:
    args = parse_args()
    walk(args)


if __name__ == "__main__":
    main()
