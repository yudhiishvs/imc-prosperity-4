"""
Async orchestrator for the Prosperity 4 strategy improvement pipeline.

Pipeline:
  Phase 1 (sync, fast)  — parse log + locate current algo
  Phase 2 (parallel)    — plot_agent  ||  analyst_agent
  Phase 3 (sync)        — coder_agent → new strategy .py
"""
import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from parse_log import load_log, parse_submission_log
from agents import analyst_agent, coder_agent, plot_agent


# ── Version helpers ──────────────────────────────────────────────────────────

def _extract_version(path: Path | None) -> str:
    if not path:
        return "v0"
    m = re.match(r"(v\d+)", path.name)
    return m.group(1) if m else "v0"


def _next_version(version: str) -> str:
    m = re.match(r"v(\d+)", version)
    n = int(m.group(1)) if m else 0
    return f"v{n + 1}"


def _version_key(p: Path) -> int:
    """Natural sort key: extract the version number from vN_round*.py."""
    m = re.match(r"v(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _find_latest_algo(log_file: str, round_num: int = 4) -> Path | None:
    """
    Search heuristically for the latest strategy .py:
      1. Same directory as the log file (pattern from existing log dirs)
      2. ../algos/ relative to log file
      3. round{N}/algos/ from the project root
    """
    log_dir = Path(log_file).resolve().parent

    # 1. .py in the same dir as the log
    py_files = sorted(log_dir.glob("v*_round*.py"), key=_version_key)
    if py_files:
        return py_files[-1]

    # 2. sibling algos/ dir
    algos_dir = log_dir.parent / "algos"
    if algos_dir.exists():
        py_files = sorted(algos_dir.glob("v*_round*.py"), key=_version_key)
        if py_files:
            return py_files[-1]

    # 3. walk up looking for round{N}/algos/
    for parent in log_dir.parents:
        candidate = parent / f"round{round_num}" / "algos"
        if candidate.exists():
            py_files = sorted(candidate.glob("v*_round*.py"), key=_version_key)
            if py_files:
                return py_files[-1]

    return None


# ── Main workflow ────────────────────────────────────────────────────────────

async def run_workflow(
    log_file: str,
    algo_file: str | None,
    output_dir: str,
    round_num: int = 4,
    skip_plots: bool = False,
    skip_code: bool = False,
) -> dict:
    """
    Run the full three-phase workflow.
    Returns a dict with paths to all outputs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  PROSPERITY 4 — STRATEGY IMPROVEMENT WORKFLOW")
    print("=" * 60)

    # ── Phase 1: Parse ───────────────────────────────────────────────────────
    print("\n[1/3] Parsing log…")

    raw = load_log(log_file)
    metrics = parse_submission_log(raw)

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        sys.exit(1)

    print(f"  Total PnL : {metrics['total_pnl']}")
    print(f"  Products  : {', '.join(metrics['products'])}")
    print(f"  Days      : {metrics['days']}")

    # Locate strategy code
    algo_path = Path(algo_file) if algo_file else _find_latest_algo(log_file, round_num)
    code = algo_path.read_text() if algo_path and algo_path.exists() else ""
    current_version = _extract_version(algo_path)
    new_version = _next_version(current_version)

    if algo_path:
        print(f"  Strategy  : {algo_path.name}  ({current_version} → {new_version})")
    else:
        print("  Strategy  : not found (will skip code generation)")

    # ── Phase 2: Parallel analysis ───────────────────────────────────────────
    print("\n[2/3] Running plot_agent + analyst_agent in parallel…")

    tasks = {}

    if not skip_plots:
        tasks["plots"] = asyncio.create_task(
            plot_agent(metrics, str(out)), name="plot_agent"
        )

    tasks["hypotheses"] = asyncio.create_task(
        analyst_agent(metrics, code, current_version), name="analyst_agent"
    )

    # Await all phase-2 tasks, collecting results and errors separately
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    phase2 = dict(zip(tasks.keys(), results))

    # Handle plots result
    plots_dir = None
    if "plots" in phase2:
        if isinstance(phase2["plots"], Exception):
            print(f"  [plot_agent]    FAILED — {phase2['plots']}")
        else:
            plots_dir = phase2["plots"]
            n_plots = len(list(plots_dir.glob("*.png")))
            print(f"  [plot_agent]    OK — {n_plots} plot(s) in {plots_dir}/")

    # Handle hypotheses result
    hypotheses = ""
    if isinstance(phase2["hypotheses"], Exception):
        print(f"  [analyst_agent] FAILED — {phase2['hypotheses']}")
    else:
        hypotheses = phase2["hypotheses"]
        hyp_file = out / f"hypotheses_{new_version}_{stamp}.md"
        hyp_file.write_text(hypotheses)
        print(f"  [analyst_agent] OK — {hyp_file.name}")

    # ── Phase 3: Code generation ─────────────────────────────────────────────
    new_algo_file = None

    if skip_code or not code or not hypotheses:
        if not code:
            print("\n[3/3] Skipping code generation (no strategy file found)")
        elif not hypotheses:
            print("\n[3/3] Skipping code generation (analyst failed)")
        else:
            print("\n[3/3] Skipping code generation (--skip-code)")
    else:
        print(f"\n[3/3] coder_agent writing {new_version}…")
        try:
            new_code = await coder_agent(code, hypotheses, new_version)

            # Determine output path: same dir as source algo, or out/algos/
            if algo_path:
                new_algo_file = algo_path.parent / f"{new_version}_round{round_num}_prosperity.py"
            else:
                (out / "algos").mkdir(exist_ok=True)
                new_algo_file = out / "algos" / f"{new_version}_round{round_num}_prosperity.py"

            new_algo_file.write_text(new_code)
            print(f"  [coder_agent]   OK — {new_algo_file}")
        except Exception as e:
            print(f"  [coder_agent]   FAILED — {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    if plots_dir:
        print(f"  Plots      → {plots_dir}/")
    if hypotheses:
        print(f"  Hypotheses → {hyp_file}")
    if new_algo_file:
        print(f"  New algo   → {new_algo_file}")
        print(f"\n  Next step: prosperity4btx {new_algo_file} {round_num}")
    print()

    return {
        "plots_dir":    str(plots_dir) if plots_dir else None,
        "hyp_file":     str(hyp_file) if hypotheses else None,
        "new_algo":     str(new_algo_file) if new_algo_file else None,
        "metrics":      metrics,
        "hypotheses":   hypotheses,
    }
