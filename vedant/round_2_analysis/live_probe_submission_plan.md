# Live Probe Submission Plan (Round 2)

This plan targets unresolved assumptions with controlled submissions.

## 1) Pepper hidden-mark recovery

- Submit:
  - `imc-prosperity-4/vedant/round_2_analysis/hold_1_pepper_probe.py`
- Download artifact `.log` from live site.
- Analyze:
  - `python3 imc-prosperity-4/vedant/round_2_analysis/analyze_hold_1_pepper.py /path/to/submission.log`
- Confirms:
  - Pepper server mark reconstruction
  - Pepper mark grid and trend behavior in live scoring

## 2) Osmium hold-then-flat mark check

- Submit:
  - `imc-prosperity-4/vedant/round_2_analysis/flip_1_osmium_probe.py`
- Download artifact `.log`.
- Analyze:
  - `python3 imc-prosperity-4/vedant/round_2_analysis/analyze_flip_1_osmium.py /path/to/submission.log`
- Confirms:
  - mark reconstruction during hold segment
  - PnL behavior after flatting inventory

## 3) Combined dual-product mark run

- Submit:
  - `imc-prosperity-4/vedant/round_2_analysis/dual_hold_probe.py`
- Download artifact `.log`.
- Analyze:
  - `python3 imc-prosperity-4/vedant/round_2_analysis/analyze_dual_hold_probe.py /path/to/submission.log`
- Confirms:
  - both product mark paths in one run
  - consistency of per-product PnL decomposition

## 4) Recompute evidence matrix after each probe

- Run:
  - `python3 imc-prosperity-4/vedant/round_2_analysis/validate_round2_evidence.py`
- Optional override:
  - `--hold1-log /path/to/new_osmium_hold.log`

