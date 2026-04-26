# Round 2 Monte Carlo Reconstruction State (Current)

This note tracks only what matters for rebuilding a truthful Round 2 Monte Carlo
backtester under `prosperity4mcbt/round2`.

## Scope

- `imc-prosperity-4` is the data/probe truth source:
  - raw Round 2 CSVs
  - downloaded live submission artifacts
  - probe strategies and analyzers in `vedant/round_2_analysis`
- `prosperity4mcbt` is implementation space:
  - tutorial calibration in `calibration/` is methodology reference
  - new Round 2 build target is `round2/`
  - current `rust_simulator/src/main.rs` is not trusted Round 2 truth

## Phase 0 -> Result (Workspace Separation)

Goal: do not contaminate tutorial inference with Round 2 assumptions.

Result:
- Round 2 implementation target is `prosperity4mcbt/round2`.
- Tutorial artifacts remain reference only.
- `prosperity4mcbt/rust_simulator/src/main.rs` still has legacy mismatches:
  - `const DAYS: [-2, -1]`
  - `load_price_rows/load_trade_rows` read `round_0` filenames
  - replay filtering still includes `TOMATOES`
  - ASH replay path still calls simulated fair generator

## Phase 1 -> Result (Hidden State Recovery)

Goal: recover hidden server mark before inferring quote mechanics.

### Verified from live probes

Primary probe artifact:
- `imc-prosperity-4/vedant/round_2_analysis/buy_1_osmium/340214.log`

Cross-check artifact:
- `imc-prosperity-4/vedant/round_2_analysis/hold_1_unit/296479.log`

Recovered Osmium mark (`server_mark = pnl + buy_price`):
- entry timestamp: `0`
- inferred buy price: `10016.0`
- post-entry ticks: `999`
- mean/std/min/max: `10004.1362 / 2.7773 / 9998.6611 / 10010.8027`
- lag-1 autocorr: `0.99347`
- delta mean/std: `-0.00535 / 0.31731`
- mark and delta both lie exactly on a `1/1024` grid

Cross-run independence check:
- `340214` and `296479` reconstruct the same tick path
- status: consistency verified, independence not verified

Controlled Pepper hold-1 artifact:
- `imc-prosperity-4/vedant/round_2_analysis/hold_1_pepper_logs/343258.log`

Recovered Pepper mark (`server_mark = pnl + buy_price`):
- entry timestamp: `0`
- inferred buy price: `13007.0`
- post-entry ticks: `999`
- mean/std/min/max: `13050.0000 / 28.8386 / 13000.0996 / 13099.9004`
- lag-1 autocorr: `~1.0`
- delta mean/std: `0.1000008 / 0.0004786`
- delta support is effectively two-point on the `1/1024` grid:
  - `0.099609` (~60%)
  - `0.100586` (~40%)
- mark lies on a `1/1024` grid.

Controlled Osmium flip artifact:
- `imc-prosperity-4/vedant/round_2_analysis/flip_1_osmium_logs/343375.log`

Flip result:
- hold segment recovered cleanly (`n_hold_ticks = 200`, `1/1024` grid)
- flat segment locks PnL exactly (`flat_pnl_std = 0`, `flat_pnl_range = 0`)
- hold segment matches the first 200 ticks of the hold-1 Osmium mark path exactly.

Controlled dual-hold artifact:
- `imc-prosperity-4/vedant/round_2_analysis/dual_hold_logs/343414.log`

Dual result:
- both products recover marks cleanly in one run
- dual Osmium path matches hold-1 Osmium path exactly
- dual Pepper path matches hold-1 Pepper path exactly
- this validates per-product PnL decomposition and no cross-product contamination.

### Verified from raw CSVs

From `imc-prosperity-4/data/ROUND_2`:
- per-product price rows: `10000` per day (`-1, 0, 1`)
- trade rows:
  - ASH: `459, 471, 465`
  - PEPPER: `331, 332, 333`
- Pepper mid has near-deterministic drift:
  - slope per tick approx `+0.1` each day
  - `R^2` approx `0.9999`
- Ash drift is weak and far less linear than Pepper.

## Phase 2 -> Result (Layer/Regime Inference)

Goal: infer quote layers against hidden state, not visible mid.

### Verified for Osmium (from `340214.log`)

Top L1 offset pairs vs `round(server_mark)`:
- `(-8, 8): 62.13%`
- `(-10, 8): 7.23%`
- `(-11, 8): 7.02%`
- `(-8, 10): 6.06%`
- `(-8, 11): 5.96%`
- rare narrow/one-sided states also present (`(-8, 2)`, `(-8, -2)`, `(-3, 8)`, `(-2, 8)`).

Mark vs visible L1 relationship:
- inside spread: `91.19%`
- below bid: `0.90%`
- above ask: `2.10%`

Implication:
- one spread PMF is insufficient; regime/state model is required.

### Verified for Pepper (from `343258.log`)

Top L1 offset pairs vs `round(server_mark)`:
- `(-7, 7): 59.83%`
- `(-7, 10): 9.99%`
- `(-10, 7): 6.30%`
- `(-7, 11): 4.23%`
- `(-11, 7): 3.69%`
- `(-7, 8): 3.37%`
- `(-8, 7): 3.04%`
- `(-10, 11): 1.09%`

Mark vs visible L1 relationship:
- inside spread: `88.59%`
- below bid: `2.00%`
- above ask: `1.70%`

Implication:
- Pepper also has discrete structural quote regimes, centered near `round(mark)` but
  with frequent asymmetric widening and occasional one-sided/narrow anomalies.
- Pepper can now be modeled with the same regime-first process used for Osmium.

### Partially verified from raw CSVs (both products)

- L1/L2 presence and spread medians are measurable directly.
- L1 OIM next-tick correlation is weak (close to zero).
- Ash large-L2 structure is measurable and persistent:
  - ticks with any L2: `26399`
  - large L2 ticks (`bid_vol_2 + ask_vol_2 >= 48`): `7466`
  - mostly resting/no-trade behavior

Still missing:
- full per-product transition matrix between regime states
- layer-size and persistence conditionals by regime
- trade arrival/side/size conditionals by regime and mark context

## Phase 3 -> Result (Conditional Flow Model)

Goal: fit conditional arrival/side/size/persistence by structural state.

Current status:
- descriptive scripts exist, but full conditional model is not identified.
- this remains an assumption area until state-conditioned fits and validation are added.

## Phase 4 -> Result (Validation Harness)

Goal: establish actual-vs-sim acceptance tests before trusting Round 2 simulator.

Current status:
- no accepted Round 2 validation harness yet.
- current Rust path is still legacy/heuristic, so outputs are not calibration truth.

## Evidence Pipeline (Current)

Core validator:
- `python3 imc-prosperity-4/vedant/round_2_analysis/validate_round2_evidence.py`
- output:
  - `imc-prosperity-4/vedant/round_2_analysis/round2_evidence_validation.json`

Probe/analyzer utilities now in place:
- log parsing:
  - `submission_log_utils.py`
- new probes:
  - `hold_1_pepper_probe.py`
  - `flip_1_osmium_probe.py`
  - `dual_hold_probe.py`
- matching analyzers:
  - `analyze_hold_1_pepper.py`
  - `analyze_flip_1_osmium.py`
  - `analyze_dual_hold_probe.py`
  - `analyze_submission_mark_reconstruction.py`
- runbook:
  - `live_probe_submission_plan.md`

Documentation conflict resolutions (confirmed):
- Keep both L2 persistence decompositions:
  - `prev_large_l2` (continuity of large resting walls)
  - `prev_any_l2` (continuity of any resting L2)
- Keep both level presence metrics:
  - `any-side` presence
  - `two-sided` presence
- Treat tutorial docs as methodology truth only; Round 2 probe/CSV evidence is implementation truth.

## Blocking Inputs Needed From Live Site

All currently planned controlled probes have now been submitted and analyzed:
- `hold_1_pepper_logs/343258.log`
- `flip_1_osmium_logs/343375.log`
- `dual_hold_logs/343414.log`

The bottleneck is no longer data collection; it is model fitting and simulator implementation.

## Round2 Calibration Artifacts (Implemented)

The following scripts now exist and run end-to-end under `prosperity4mcbt/round2/scripts`:

1. `extract_probe_hidden_state_tables.py`
   - outputs:
     - `round2/calibration/probe_tables/osmium_hold1_labeled.csv`
     - `round2/calibration/probe_tables/pepper_hold1_labeled.csv`
     - `round2/calibration/probe_tables/probe_hidden_state_summary.json`
2. `fit_regime_transition_models.py`
   - outputs:
     - `round2/calibration/regime_models/*transition*.csv`
     - `round2/calibration/regime_models/*run_lengths*.csv`
     - `round2/calibration/regime_models/regime_model_summary.json`
3. `fit_state_conditionals_from_probe_tables.py`
   - outputs:
     - `round2/calibration/flow_models/*by_regime*.csv`
     - `round2/calibration/flow_models/state_conditionals_summary.json`
4. `fit_observed_state_models_from_raw_csv.py`
   - outputs:
     - `round2/calibration/raw_state_models/*obs_state*.csv`
     - `round2/calibration/raw_state_models/raw_state_model_summary.json`
5. `build_round2_fused_parameters.py`
   - outputs:
     - `round2/calibration/fused_parameters/round2_fused_parameters.json`

Interpretation:
- probe-conditioned models provide hidden-state-grounded regime structure.
- raw-CSV observable-state models provide high-sample transition/flow estimates.
- fused parameter tables are now generated and ready for simulator integration.

## Next Steps (Distribution Backout -> Simulator)

1. Build labeled regime tables in `prosperity4mcbt/round2/scripts` for both products:
   - hidden mark
   - `round(mark)`
   - L1/L2/L3 offsets
   - regime label per tick
2. Fit regime transition models:
   - `P(regime_t | regime_{t-1}, mark_context)`
   - separate fits for Osmium and Pepper
3. Fit conditional layer models:
   - presence by layer and regime
   - size PMFs by layer and regime
   - run-length / persistence for non-base states
4. Fit conditional trade-flow models:
  - arrival probability by regime/layer state
  - taker side by state
  - taker size by state
5. Implement fused parameter tables (probe-grounded + raw-count stabilized) in `prosperity4mcbt/round2` generator code.
6. Add actual-vs-sim validation suite before using Monte Carlo for strategy search.

## Build Gate For `prosperity4mcbt/round2`

Do not treat Round 2 simulator as calibrated until:
1. Regime and conditional models are fit from reconstructed hidden-state tables for both products.
2. Osmium + Pepper regime tables are fit against reconstructed hidden state.
3. Conditional flow model is state-conditioned (arrival, side, size, persistence).
4. Actual-vs-sim checks pass on key distributions before strategy research use.

## Phase 5 -> Result (Round 2 Monte Carlo Backtester Implemented)

Goal: run real `Trader.run(state)` against a Round 2 simulator grounded in fused
probe + raw-CSV calibration outputs, then inspect outcomes in the existing dashboard.

Implemented:
- `prosperity4mcbt/round2/round2_monte_carlo.py`
- docs updated:
  - `prosperity4mcbt/round2/README.md`
  - `prosperity4mcbt/round2/scripts/README.md`

Core runtime behavior:
- loads fused parameters from:
  - `prosperity4mcbt/round2/calibration/fused_parameters/round2_fused_parameters.json`
- simulates products:
  - `ASH_COATED_OSMIUM`
  - `INTARIAN_PEPPER_ROOT`
- calls strategy `Trader.run(state)` each tick (real strategy execution path, not dummy PnL replay)
- writes dashboard bundle and supporting artifacts:
  - `dashboard.json`
  - `session_summary.csv`
  - `run_summary.csv`
  - sampled session CSVs under `sessions/`
  - `run.log`

CLI surface (Round 2 runner):
- `--quick` preset: `sessions=20`, `sample_sessions=5`, `ticks_per_day=4000`
- `--heavy` preset: `sessions=300`, `sample_sessions=30`, `ticks_per_day=10000`
- explicit options for:
  - sessions
  - days per session
  - ticks per day
  - RNG seed
  - fused parameter path
  - output dashboard path
  - optional visualizer launch (`--vis`)

Observed reproducibility check:
- command:
  - `python3 round2/round2_monte_carlo.py ../imc-prosperity-4/vedant/strategy.py --quick --out backtests/round2_codex_check/dashboard.json`
- current run output:
  - sessions: `20`
  - mean total PnL: `37,838.43`
  - std total PnL: `827.54`
  - median total PnL: `37,980.11`
  - 5%-95%: `36,148.41` to `38,836.97`

## Phase 5A -> Result (Visualizer Port Flow + Server Startup Fix)

Issue addressed:
- visualizer on `localhost:5555` was up, but dashboard data server on `localhost:8001`
  could fail to start in some invocation contexts.

Implemented fix:
- updated:
  - `prosperity4mcbt/backtester/prosperity4mcbt/dashboard_server.py`
- `ensure_dashboard_server(...)` now launches using the script file path:
  - `python <...>/dashboard_server.py <root> <port>`
  instead of module launch assumptions that could break with import context.
- startup diagnostics improved:
  - server logs captured at `~/.prosperity4mcbt/dashboard_server.log`
  - readiness failure now surfaces tail log lines.

Operational note:
- local state files used by dashboard server:
  - `~/.prosperity4mcbt/dashboard_root.txt`
  - `~/.prosperity4mcbt/dashboard_server.pid`
  - `~/.prosperity4mcbt/dashboard_server.log`

## Round 2 Runbook (Current)

From `prosperity4mcbt/`:

1. Build/refresh calibration artifacts:
   - `python3 round2/scripts/extract_probe_hidden_state_tables.py`
   - `python3 round2/scripts/fit_regime_transition_models.py`
   - `python3 round2/scripts/fit_state_conditionals_from_probe_tables.py`
   - `python3 round2/scripts/fit_observed_state_models_from_raw_csv.py`
   - `python3 round2/scripts/build_round2_fused_parameters.py`
2. Run Round 2 Monte Carlo:
   - `python3 round2/round2_monte_carlo.py ../imc-prosperity-4/vedant/strategy.py --out backtests/round2_main/dashboard.json`
3. Optional quick dev loop:
   - `python3 round2/round2_monte_carlo.py ../imc-prosperity-4/vedant/strategy.py --quick --out backtests/round2_quick/dashboard.json`
4. Optional visualizer:
   - start frontend once (if not running): `cd visualizer && npm run dev`
   - run with auto-open: `python3 round2/round2_monte_carlo.py ../imc-prosperity-4/vedant/strategy.py --quick --vis --out backtests/round2_quick/dashboard.json`

## Remaining Work (Post-Implementation)

The simulator now runs end-to-end, but calibration trust still depends on validation depth.
Highest-priority next steps:

1. Add actual-vs-sim acceptance harness for Round 2 keys:
   - L1 offset-pair frequencies
   - spread and layer-presence distributions
   - trade arrival/side/size marginals and conditionals
   - run-length distributions for non-base regimes
2. Add report card with pass/fail thresholds and confidence intervals.
3. Tighten fallback behavior for sparse states in fused tables (document exact fallback cascade).
4. Run seed sweeps and stability checks before strategy-search usage.

## Phase 6 -> Result (Round 2 Gradient-Ascent Parameter Walker)

Goal: optimize strategy parameters against the Round 2 Monte Carlo environment by
probing local plus/minus jumps and walking in the direction that improves objective value.

Implemented:
- optimizer script:
  - `imc-prosperity-4/vedant/optimize_round2_mc.py`
- optimizer uses:
  - patched temporary strategy copies (class-level constants changed per candidate)
  - `prosperity4mcbt/round2/round2_monte_carlo.py` as the evaluation engine
  - finite-difference directional checks (`+h` / `-h`) per parameter
  - adaptive jump scaling and restart-based search

### Optimization Behavior

Per run:
1. evaluate current parameter vector
2. for each parameter, evaluate `+jump` and `-jump`
3. compute local directionality and propose multi-parameter uphill move
4. fallback to best single-dimension move if joint move fails
5. grow jumps after success, shrink jumps after failure
6. if no improvement for `--patience` consecutive steps in normal mode:
   - switch to jitter mode
   - multiply step sizes by `--jitter-multiplier`
7. continue stepping in jitter mode
8. stop the run only after jitter mode also reaches `--patience` consecutive non-improving steps

Across runs (`--runs`):
- run 1 starts from current `strategy.py` values (or `--init-json` overrides)
- later runs random-restart within parameter bounds
- global best is tracked across all runs

Console guarantees:
- every time a new global maximum is found, a high-visibility banner prints:
  - run/iteration
  - score
  - metrics
  - full parameter set
- after each run, best score/metrics/params for that run are printed explicitly.

### Objective Modes

Available objectives:
- `auto`
- `total_mean`
- `total_sharpe`
- `osmium_mean`
- `pepper_mean`

`auto` mapping:
- with `--product osmium` => `osmium_mean`
- with `--product pepper` => `pepper_mean`

### Product Modes

`--product osmium`:
- optimizes Osmium parameter family (`OSMIUM_*` definitions in script).

`--product pepper`:
- optimizes Pepper parameter family (`PEPPER_*` definitions in script).

### Full CLI Flag Breakdown

Core search controls:
- `--product {osmium,pepper}`
- `--runs <int>`: number of restart runs
- `--patience <int>`: no-improvement threshold used for both normal and jitter phases
- `--step-growth <float>`: jump multiplier after improving step
- `--step-shrink <float>`: jump multiplier after non-improving step
- `--jitter-multiplier <float>`: step-size multiplier applied when switching from normal mode to jitter mode

Monte Carlo evaluation controls:
- `--sessions <int>`: MC sessions per evaluation
- `--ticks-per-day <int>`
- `--days-per-session <int>`
- `--sample-sessions <int>`: forwarded to MC runner
- `--seed <int>`: base seed (per-run seeds derived from this)
- `--fused-params <path>`: fused round2 parameter JSON

Strategy and parameter controls:
- `--strategy <path>`: strategy file containing `Trader`
- `--params <NAMES...>`: optimize only listed params
- `--init-json <path>`: JSON overrides for initial values
- `--objective {auto,total_mean,total_sharpe,osmium_mean,pepper_mean}`

Output/debug controls:
- `--keep-output`: keep per-evaluation folders (instead of cleanup)

### Recommended Run Commands

From workspace root (`/Users/vedant/Quant/Prosperity4`):

Osmium focused (default objective auto -> `osmium_mean`):
- `python3 imc-prosperity-4/vedant/optimize_round2_mc.py --product osmium --runs 5 --patience 5 --jitter-multiplier 2.5 --sessions 40 --ticks-per-day 6000`

Pepper focused (default objective auto -> `pepper_mean`):
- `python3 imc-prosperity-4/vedant/optimize_round2_mc.py --product pepper --runs 5 --patience 5 --jitter-multiplier 2.5 --sessions 40 --ticks-per-day 6000`

Subset tuning (faster local search):
- `python3 imc-prosperity-4/vedant/optimize_round2_mc.py --product osmium --params OSMIUM_EMA_ALPHA OSMIUM_OIM_THRESHOLD OSMIUM_OIM_TAKE_SCALE --runs 4 --patience 4 --sessions 30`

Risk-adjusted objective:
- `python3 imc-prosperity-4/vedant/optimize_round2_mc.py --product pepper --objective total_sharpe --runs 4 --patience 5 --sessions 40`

Custom initialization:
- create JSON such as:
  - `{"OSMIUM_EMA_ALPHA": 0.09, "OSMIUM_OIM_THRESHOLD": 0.04}`
- run:
  - `python3 imc-prosperity-4/vedant/optimize_round2_mc.py --product osmium --init-json /abs/path/init.json --runs 4 --patience 5`

### Outputs and Artifacts

Primary result file:
- `imc-prosperity-4/vedant/.round2_mc_optimizer/best_result.json`

Contains:
- `product`
- `objective`
- global best score/metrics/params
- per-run records:
  - run seed
  - run best score/metrics/params
  - iteration history

Intermediate eval artifacts (if `--keep-output`):
- `imc-prosperity-4/vedant/.round2_mc_optimizer/eval_<hash>/...`

### Practical Notes

- Higher `--sessions` reduces objective noise but increases runtime substantially.
- For first-pass parameter discovery, start with:
  - moderate sessions (`20-40`)
  - fewer ticks (`3000-6000`)
  - subset params
- For final candidate confirmation, re-run best configuration with larger sessions and
  multiple seeds to validate stability before merging parameters into `strategy.py`.
