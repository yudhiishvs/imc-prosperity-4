#!/usr/bin/env python3
"""
Prosperity 4 — Strategy Improvement Workflow
=============================================

Usage:
  python run.py <log.json>
  python run.py <log.json> --algo ../algos/v5_round4_prosperity.py
  python run.py <log.json> --skip-code        # analysis + plots only
  python run.py <log.json> --skip-plots       # analysis + code only
  python run.py <log.json> --out ./my_output  # custom output dir

The workflow runs three phases:
  [1] Parse   — pure Python, extracts metrics from the log
  [2] Analyze — plot_agent + analyst_agent run in PARALLEL via Claude API
  [3] Code    — coder_agent writes the next strategy version

Requires:  pip install anthropic matplotlib pandas numpy
Env var:   ANTHROPIC_API_KEY must be set
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Make sure the workflow dir is on the path when called from elsewhere
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import run_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Prosperity 4 strategy improvement workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "log_file",
        help="Path to submission .json (or raw .log) file",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help="Path to current strategy .py (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <log_dir>/workflow_output/)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Competition round number (default: 4)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (faster, analysis + code only)",
    )
    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Skip code generation (analysis + plots only)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    output_dir = args.out or str(log_path.parent / "workflow_output")

    asyncio.run(
        run_workflow(
            log_file=str(log_path),
            algo_file=args.algo,
            output_dir=output_dir,
            round_num=args.round,
            skip_plots=args.skip_plots,
            skip_code=args.skip_code,
        )
    )


if __name__ == "__main__":
    main()
