"""
Three Claude agents for the Prosperity 4 strategy improvement workflow.

  plot_agent    — generates matplotlib code from log data, executes it
  analyst_agent — reads metrics + code, produces ranked hypotheses
  coder_agent   — implements the top hypothesis as a new strategy version

All agents are async and can be awaited in parallel.
"""
import asyncio
import json
import subprocess
import textwrap
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-6"

_client = anthropic.AsyncAnthropic()


async def _call(system: str, user: str, max_tokens: int = 8096) -> str:
    resp = await _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


# ── Agent 1: Plot Generator ─────────────────────────────────────────────────

_PLOT_SYSTEM = """\
You are a data visualization expert for algorithmic trading.
Given structured metrics from an IMC Prosperity trading log, write complete
self-contained matplotlib Python code that:

1. Hardcodes the provided data inline (do NOT read from files).
2. Creates 4–6 plots covering:
   - Cumulative PnL per product over time (all days overlaid or stitched)
   - Position over time per product (from position_stats if available; else skip)
   - Per-day PnL bar chart (product breakdown stacked)
   - Spread over time or mid-price distribution for each product
   - Any option-specific plot if options products are present (e.g. IV smile)
3. Saves each figure as a separate PNG to the PLOTS_DIR variable defined at top.
4. Uses tight_layout(), clear titles, labelled axes, and a legend.
5. Does NOT call plt.show().

Return ONLY the Python code block — no explanation, no markdown fences.\
"""


async def plot_agent(metrics: dict, output_dir: str) -> Path:
    """
    Ask Claude to generate matplotlib code, then execute it.
    Returns path to the plots directory.
    """
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Trim timeseries to keep prompt lean (at most 300 points per product)
    lean_metrics = _trim_for_prompt(metrics)

    prompt = (
        f"PLOTS_DIR = {str(plots_dir)!r}\n\n"
        f"Metrics:\n{json.dumps(lean_metrics, indent=2)}"
    )

    code = await _call(_PLOT_SYSTEM, prompt, max_tokens=4096)

    # Strip accidental markdown fences
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()

    # Inject the PLOTS_DIR definition at the top so the code can use it
    code = f"PLOTS_DIR = {str(plots_dir)!r}\n" + code

    script_path = Path(output_dir) / "_generated_plots.py"
    script_path.write_text(code)

    try:
        subprocess.run(
            ["python3", str(script_path)],
            check=True,
            timeout=90,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [plot_agent] stderr: {e.stderr[-500:]}")
        raise RuntimeError(f"Plot script failed: {e.returncode}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("Plot script timed out")

    return plots_dir


# ── Agent 2: Strategy Analyst ───────────────────────────────────────────────

_ANALYST_SYSTEM = """\
You are an expert quant analyst for IMC Prosperity algorithmic trading competitions.

Given:
  - Parsed trading metrics from the latest submission
  - The current strategy's source code

Produce a concise hypothesis log with 3–5 ranked improvement ideas.

Use this format exactly:

## Hypothesis Log
Base: {version}

### H1: [short name] — [HIGH / MED / LOW]
**Observation:** what the data reveals
**Hypothesis:** the exact change to make
**Expected ΔPnL:** +/- estimate with reasoning
**Implementation:** specific parameter name, line, or logic block to change
**Risk:** what could go wrong

(repeat H2 … H5)

Rules:
- Rank by expected impact × confidence
- Each hypothesis changes EXACTLY ONE thing
- Be specific: name variables, thresholds, and expected direction
- Do NOT suggest adding logging, comments, or refactors\
"""


async def analyst_agent(metrics: dict, code: str, version: str) -> str:
    """Returns a markdown string of ranked hypotheses."""
    system = _ANALYST_SYSTEM.replace("{version}", version)

    # Summarise metrics without the full timeseries to keep context lean
    summary = _metrics_summary(metrics)

    prompt = textwrap.dedent(f"""\
        === METRICS SUMMARY ===
        {summary}

        === STRATEGY CODE ===
        {code}
    """)

    return await _call(system, prompt, max_tokens=4096)


# ── Agent 3: Code Writer ─────────────────────────────────────────────────────

_CODER_SYSTEM = """\
You are an expert Python developer implementing algorithmic trading strategies
for IMC Prosperity. Given a current strategy and a hypothesis log, implement
the #1 hypothesis as a new strategy version.

Rules:
  1. Make EXACTLY ONE change — the top-ranked hypothesis only.
  2. Preserve all other logic, variable names, formatting, and imports exactly.
  3. Add a single comment immediately before or after your change:
       # [{new_version} change]: <one-line explanation>
  4. Do NOT clean up, refactor, add docstrings, or touch anything else.
  5. Return the COMPLETE Python file.\
"""


async def coder_agent(code: str, hypotheses: str, new_version: str) -> str:
    """Returns the complete source of the new strategy version."""
    system = _CODER_SYSTEM.replace("{new_version}", new_version)

    prompt = textwrap.dedent(f"""\
        New version: {new_version}

        === HYPOTHESES (implement #1 only) ===
        {hypotheses}

        === CURRENT STRATEGY CODE ===
        {code}
    """)

    return await _call(system, prompt, max_tokens=8096)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _trim_for_prompt(metrics: dict) -> dict:
    """Return metrics with timeseries capped at 300 points per product."""
    m = {k: v for k, v in metrics.items() if k != "timeseries"}
    m["timeseries"] = {
        product: pts[:300]
        for product, pts in metrics.get("timeseries", {}).items()
    }
    return m


def _metrics_summary(metrics: dict) -> str:
    """Human-readable metrics summary (no timeseries data)."""
    lines = [
        f"Total PnL: {metrics['total_pnl']}",
        f"Products:  {', '.join(metrics['products'])}",
        f"Days:      {metrics['days']}",
        "",
        "Per-product PnL:",
    ]
    for product, stats in metrics.get("product_stats", {}).items():
        lines.append(f"  {product}: {stats['total_pnl']}")
        for day, d in stats.get("daily", {}).items():
            lines.append(
                f"    day {day}: pnl={d['end_pnl']}, "
                f"spread={d.get('avg_spread')}, "
                f"mid_range={d.get('mid_range')}, "
                f"drawdown={d.get('max_pnl_drawdown')}"
            )
    lines.append("")
    lines.append("Position stats:")
    for product, ps in metrics.get("position_stats", {}).items():
        lines.append(
            f"  {product}: max_long={ps.get('max_long')}, "
            f"max_short={ps.get('max_short')}, "
            f"avg_abs={ps.get('avg_abs')}"
        )
    lines.append("")
    lines.append("Trade stats (from lambda log):")
    for product, ts in metrics.get("trade_stats", {}).items():
        lines.append(
            f"  {product}: fills={ts.get('n_fills')}, "
            f"buys={ts.get('buy_fills')}, sells={ts.get('sell_fills')}"
        )
    return "\n".join(lines)
