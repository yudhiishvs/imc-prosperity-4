# Round 3 Strategy Plan — IMC Prosperity 4
*Informed by Frankfurt Hedgehog (2nd globally, IMC Prosperity 3)*
*Cross-referenced with Round 3 EDA findings*

---

## 1. Direct Analogies to Hedgehog's Products

| Round 3 (P4) | Hedgehog Analogue | Key Characteristic |
|---|---|---|
| `HYDROGEL_PACK` | Rainforest Resin | ~Fixed fair value (~10k), near-stationary |
| `VELVETFRUIT_EXTRACT` | Volcanic Rock | Mean-reverting underlying with drift |
| `VEV_4000–VEV_6500` | Volcanic Rock Vouchers | Call options, TTE 7→5 days, IV smile exists |

---

## 2. What Hedgehog Did (and Why It Worked)

### 2.1 IV Scalping (Their Core Options Edge)

**What they did:**
1. Computed BS implied vol for each strike at each tick
2. Plotted IV vs moneyness (m = S/K) — observed a parabolic smile
3. **Fit a parabola** to IV(m): `v̂(m) = a·m² + b·m + c`
4. Computed residuals: `δ_v = v_observed - v̂(m)` — now moneyness-independent
5. Converted δ_v back to price space via BS: `δ_price = BS(σ=v̂+δ_v) - BS(σ=v̂)`
6. Traded when `|δ_price|` exceeded a threshold — buy if option cheap, sell if rich
7. Validated: tested **1-lag negative autocorrelation** in option mid-price returns → confirmed scalping signal was real

**Why it worked:**
The parabola fit removes the systematic moneyness-dependent component of IV, leaving only the mean-reverting noise around the smile. The signal is structurally grounded, not just a backtested pattern.

### 2.2 WallMid as Fair Value

**What they did:**
Identified "wall" quotes — deep liquidity at consistent prices from the designated market-maker bots — and averaged them (WallMid) rather than using raw mid-price.

**Why it mattered:**
Raw mid can be distorted by aggressive/passive quoting. Wall bids/asks represent the bot's estimate of true price and are much more stable.

**Our equivalent:** In the order book data, bids/asks that appear with large, consistent volume at the same level across many ticks are the "wall." Look at `bid_price_1` and `ask_price_1` where volumes are high and stable.

### 2.3 Mean Reversion on the Underlying

**What they did:**
- Tracked a **fast EMA** (not a fixed value, not a slow MA crossover)
- Traded deviations from EMA using **fixed thresholds** — NOT z-score normalized
- Validated with **lag-1 autocorrelation test** vs random baseline
- Also used the **deepest ITM option** (highest delta, ≈1.0) as a synthetic underlying proxy

**Why NOT z-score:**
If volatility isn't meaningfully time-varying (which it isn't in short competition windows), z-scoring introduces complexity without benefit and creates overfitting risk.

### 2.4 Market Making for Stable Products

**What they did (Rainforest Resin / Kelp):**
1. First, aggressively take any order that crosses fair value (positive expected edge)
2. Then, post passive quotes slightly better than existing liquidity (overbid, undercut)
3. Flatten inventory at fair value (zero edge) when position too large
4. Never over-complicate: no hedging needed for mean-reverting products

### 2.5 Parameter Selection Philosophy

- Run full **grid search** (2D or 3D)
- **Don't pick the PnL peak** — pick the flat/stable region of the landscape
- Prioritize **robustness over max backtest profit**
- Fewest parameters possible → minimize overfitting risk

### 2.6 Research Process They Applied

1. **Start with the data generation question**: "How could this price series have been produced?"
2. **Validate statistically** before trading: autocorrelation test, not just visual inspection
3. **Think from first principles**: only deploy a signal you can explain WHY it works
4. **Never blindly apply textbook techniques** (MA crossover, z-score) without justification
5. **Use normalized views**: plot deviations from fair value, not raw prices

---

## 3. Round 3 Implementation Plan

### Strategy A: HYDROGEL_PACK Market Making

**Analogue:** Rainforest Resin / Kelp

**Research findings:**
- ADF p=1.07e-5: stationary ✓
- KPSS suggests trend-stationary → use rolling fair value, not fixed
- Spread is narrow: ~2 ticks typical
- No bot IDs visible → can't identify "Olivia"-style informed trader in this round

**Implementation:**
```
1. FAIR VALUE = EMA of mid_price, span=200 ticks  (rolling anchor)
2. At each tick:
   a. TAKE: if any ask < FAIR - 1 → buy it; if any bid > FAIR + 1 → sell it
   b. MAKE: post bid at max(best_bid + 1, FAIR - 2), ask at min(best_ask - 1, FAIR + 2)
   c. SKEW: reduce bid/ask size proportional to current position
      (long → move ask closer to fair, reduce bid aggressiveness)
   d. FLATTEN: if |pos| > 40, post orders at FAIR to unwind
```

**Parameters to sweep:**
| Parameter | Range | Notes |
|---|---|---|
| EMA span | 50, 100, 200, 500 | How fast does fair value update? |
| Take threshold | 1, 2, 3 | Min edge to cross spread |
| Quote offset | 1, 2, 3 | Half-spread around fair value |
| Flatten threshold | 30, 40, 50 | Position at which to unwind |
| Max order size | 5, 10, 15 | Per-order volume |

**Validation test before deploying:**
- Compute WallMid (avg of deep bid/ask walls) vs EMA mid → if correlated, use WallMid
- Check ADF on per-day residuals (EMA deviation) → should be stationary
- Check 1-lag autocorrelation of HYDROGEL returns → negative = mean-reversion confirmed

---

### Strategy B: VELVETFRUIT_EXTRACT Market Making / Mean Reversion

**Analogue:** Volcanic Rock (Hedgehog), Kelp

**Research findings:**
- ADF p=4.7e-5: stationary intraday ✓
- ~8 tick upward drift per day (trend-stationary)
- Imbalance predicts future returns (p=2.2e-21) → use as quoting skew signal
- Returns non-normal (fat tails) → be more conservative near position limits

**Implementation:**
```
1. FAIR VALUE = fast EMA, span=100–200 ticks  (adapts to intraday drift)
2. IMBALANCE = (bid_vol1 - ask_vol1) / (bid_vol1 + ask_vol1)
3. SKEW_OFFSET = clip(imbalance * IMBALANCE_SCALE, -2, +2)
   (positive imbalance → lean ask down slightly, expect up move)
4. Quote: bid at FAIR - SPREAD + SKEW_OFFSET
          ask at FAIR + SPREAD - SKEW_OFFSET
5. Take: if ask < FAIR - TAKE_EDGE → buy; if bid > FAIR + TAKE_EDGE → sell
6. Flatten: at fair value if |pos| > 30
```

**Parameters to sweep:**
| Parameter | Range | Notes |
|---|---|---|
| EMA span | 50, 100, 200, 500 | Intraday vs inter-day adaptation |
| SPREAD | 1, 2, 3, 4 | Half-spread |
| TAKE_EDGE | 1, 2, 3 | Min edge to take liquidity |
| IMBALANCE_SCALE | 0, 1, 2, 3 | How much to lean on imbalance signal |
| MAX_POS | 20, 30, 50 | Position limit |

**Validation tests:**
- Measure realized PnL of imbalance-informed vs naive quoting in backtest
- ADF on EMA residuals per day

---

### Strategy C: IV Scalping (VEV Options) — Core Options Edge

**Analogue:** Volcanic Rock Vouchers IV scalping (Hedgehog's main alpha)

**Core idea:** Option prices fluctuate around a theoretically fair BS price. The fluctuations mean-revert. Trade them.

**Step-by-step implementation:**

#### C.1 Compute IV at every tick
```python
# For each (strike, tick):
S    = vf_extract_mid_price
K    = strike
TTE  = TTE_START[day] - timestamp / 1_000_000  # in Solvenarian days
C    = option_mid_price
IV   = bs_call_iv(C, S, K, TTE, r=0)
m    = np.log(K / S)  # log-moneyness (Hedgehog used S/K, log form more symmetric)
```

#### C.2 Fit parabola to IV smile
```python
# At each tick, have IV values for all active strikes
# Fit: IV = a * m^2 + b * m + c   where m = log(K/S)
# Use only options with:
#   - TTE > 0.5 (near expiry → IV unreliable)
#   - Option price > intrinsic + 0.5 (has meaningful time value)
#   - Neither deep OTM (VEV_6000, VEV_6500 excluded)

coeffs = np.polyfit(m_values, iv_values, deg=2)  # [a, b, c]
iv_hat = np.polyval(coeffs, m)   # "fair" IV given moneyness
```

#### C.3 Convert to price deviations
```python
# For each option:
fair_price = bs_call_price(S, K, TTE, sigma=iv_hat)
delta_price = option_mid_price - fair_price
# delta_price > 0 → option is RICH → sell
# delta_price < 0 → option is CHEAP → buy
```

#### C.4 Trade signal
```python
if delta_price > ENTRY_THRESHOLD:
    sell option (take bid)
elif delta_price < -ENTRY_THRESHOLD:
    buy option (take ask)
# Exit when delta_price crosses zero (or hits -EXIT_THRESHOLD)
```

**Parameters to sweep:**
| Parameter | Range | Notes |
|---|---|---|
| ENTRY_THRESHOLD | 1, 2, 3, 4, 5 | Min price deviation to enter |
| EXIT_THRESHOLD | 0, 0.5, 1 | Exit at zero vs slight opposite |
| Parabola update freq | every tick, every 10 ticks | Refit frequency |
| Min TTE | 0.2, 0.5, 1.0 | Exclude near-expiry |
| Min time value | 0.5, 1, 2 | Exclude near-worthless options |
| Strike exclusion | VEV_6000/6500 always | Deep OTM unreliable IV |

**Validation tests (MUST do before submitting):**
1. **1-lag autocorrelation of option mid-price returns** for each strike
   - Expected: negative autocorrelation → confirms scalping works
   - If not significant: this strike should not be scalped
2. **Backtest δ_price series** → check stationarity (ADF) of price deviations
3. **Check parabola stability**: refit every 100 ticks, plot coefficient evolution

#### C.5 Per-strike activation
Hedgehog dynamically activated/deactivated strikes. Do the same:
```python
# Activate strike for scalping if:
#   1. 1-lag ACF of option returns is negative and significant (p < 0.05)
#   2. Average |delta_price| > ENTRY_THRESHOLD in recent 500 ticks
#   3. TTE > 1.0 day (don't scalp in last day)
#   4. Option has meaningful time value (price > intrinsic + 1)
```

---

### Strategy D: Short Deep OTM Options (VEV_6000, VEV_6500)

**Analogue:** None in Hedgehog — this is unique to our data

**Rationale from EDA:**
- Both trade at floor (0.5 mid, bid=0, ask=1)
- Underlying S ≈ 5250; strike 6000/6500 are 14%/24% OTM
- Trade volume shows bots actively buying these at 0.0 → they eventually expire worthless
- Day-2 median S = 5255 → both expire OTM under any reasonable scenario

**Implementation:**
```python
# Each tick: sell VEV_6000 and VEV_6500 at ask (=1) whenever we can
# No delta hedge needed — delta ≈ 0, these are essentially free premium
# Max loss per unit if S → 6500: 6500 - 5250 = 1250 (but effectively impossible)
# Expected value per unit: ~0.5 (the floor premium we collect)

# RISK CONTROL:
# - Hard cap: SHORT_MAX = position_limit (typically 200)
# - Monitor underlying: if S > 5700, reduce short position
# - Track cumulative short gamma: if IV_6000 suddenly spikes → exit
```

**This is near-free alpha** — the highest priority trade to execute each tick.

---

### Strategy E: Deep ITM Options as Underlying Proxy (VEV_4000)

**Analogue:** Hedgehog used the deepest ITM voucher for mean reversion positioning

**Rationale:**
- VEV_4000: delta ≈ 1.0 (confirmed from Plot 19)
- Trades at ≈ S - 4000 (intrinsic + tiny time value)
- Spread: ~6 ticks vs VF Extract spread ~2 ticks → slightly more expensive
- Use for mean reversion when VF Extract position limit is saturated

**Implementation:**
```python
# When VF Extract position is at limit AND mean reversion signal fires:
# Trade VEV_4000 instead (equivalent economic exposure)
# 1 unit VEV_4000 ≈ 1 unit VF Extract (delta ≈ 1)
# Be aware: VEV_4000 has wider spread → only use when VF Extract is full
```

---

### Strategy F: Gamma Scalping (Secondary)

**Analogue:** Hedgehog's Gamma Scalping (stable but limited returns)

**Rationale:**
- Buy ATM option (e.g., VEV_5200 or VEV_5300)
- Continuously delta-hedge with VF Extract
- Gamma gains from underlying moves exceed theta decay if vol is high enough

**Check condition before deploying:**
```
Expected gamma PnL ≈ 0.5 * Gamma * (dS)^2 - Theta * dt

With:
- Realized vol σ ≈ 0.34 (annualized), daily σ = 0.34 / sqrt(252) ≈ 2.1%
- S ≈ 5250 → daily move ≈ 110 ticks
- Need: 0.5 * Gamma * 110^2 > Theta * 1_day

Compute Gamma and Theta from BS, verify condition holds for chosen strike.
If realized_vol > implied_vol → gamma scalping is positive EV.
```

---

## 4. Parameter Sweep Plan

### Phase 1: Vectorized Notebook Sweep (fast, before full backtest)

```python
# For each strategy, vary top 2 parameters over a grid
# Use 3-day price data, vectorized
# Output: 2D heatmap of PnL

# Example: Strategy A (HYDROGEL)
for ema_span in [50, 100, 200, 500]:
    for spread in [1, 2, 3, 4]:
        pnl = simulate_hydrogel_mm(prices, ema_span, spread)
        grid[ema_span, spread] = pnl

# PICK: stable flat region, not peak
```

### Phase 2: Full Backtester Sweep

Run using the prosperity4bt backtester for top-10 parameter combinations from Phase 1.

```bash
# Script structure:
# sweep/trader_template_r3.py  ← parameterized template
# sweep/optimizer_r3.py        ← reads template, injects params, runs backtester
# sweep/results/               ← JSON results
```

### Phase 3: Robustness Validation

For the top-3 final parameter sets:
1. Run on each day individually (Day 0, 1, 2 separately) → check consistency
2. Check PnL degradation if parameter moves ±20% from optimum
3. Run on shuffled data (permutation test) → ensure signal isn't spurious

---

## 5. Validation Tests Before Submission

| Test | Method | Accept if |
|---|---|---|
| HYDROGEL fair value quality | ADF on EMA residuals per day | p < 0.05 (stationary) |
| VF Extract mean reversion | 1-lag ACF of returns | Negative, p < 0.05 |
| Option scalping signal | 1-lag ACF of option mid returns | Negative per active strike |
| IV smile fit quality | R² of parabola fit | R² > 0.7 |
| δ_price stationarity | ADF on price deviation series | p < 0.05 |
| Gamma scalping condition | σ_realized > σ_implied | Check per strike per day |
| Parameter landscape | PnL std across ±20% param perturbation | Low sensitivity |

---

## 6. Key Lessons from Hedgehog (Don'ts)

| Don't | Why |
|---|---|
| Use z-score normalization | Volatility isn't time-varying enough to justify; adds overfit risk |
| Use MA crossover on mean-reverting series | No theoretical justification in this environment |
| Hedge baskets with constituents | Reduces expected value (spread costs) unless variance reduction is worth it |
| Optimize for website backtest peak | Overfits to simulation-specific noise |
| Pick peak PnL in parameter sweep | Fragile; pick the flat stable region |
| Apply any technique without knowing WHY | Every signal needs a structural explanation |
| Try to extract IV from VEV_6000/6500 | Zero time value → IV undefined/unreliable |
| Use fixed fair value for HYDROGEL | EDA shows trend-stationarity → need rolling EMA |

---

## 7. Priority Order for Implementation

```
1. [IMMEDIATE] Short VEV_6000 + VEV_6500 at every opportunity
   → Near-zero risk, free premium, ~0.5 per unit collected

2. [HIGH] HYDROGEL market making (Strategy A)
   → Hedgehog's Resin equivalent was worth ~39k/round, very reliable

3. [HIGH] IV Scalping (Strategy C)
   → Hedgehog's core edge, ~100-150k/round. Our options structure is similar.
   → But VALIDATE 1-lag ACF per strike first.

4. [MEDIUM] VF Extract market making with imbalance skew (Strategy B)
   → Imbalance signal is significant (p=2.2e-21) — use it

5. [MEDIUM] Deep ITM option as VF Extract proxy (Strategy E)
   → Only activate when VF Extract position-limited

6. [LOWER] Gamma scalping (Strategy F)
   → Hedgehog said returns were limited; only if time permits
```

---

## 8. Notes on Bot Behavior (Unique to Round 3)

Unlike Hedgehog's rounds where trader IDs were visible, **all buyer/seller fields are NaN** in Round 3 data. However:

- Trade inter-arrivals are non-Poisson (clustered): p=8.9e-25
- Bot actively trades VEV_6000/6500 at price 0 — likely a bot that systematically buys all OTM options as a "lottery ticket" strategy
- VEV_4000 has 464 trades vs VEV_5000 with only 1 trade → deep ITM liquidity is concentrated
- VEV_5400/5500 have significant volume (225/267 trades) → bots accumulate OTM options

**Hypothesis on bot behavior:**
- Bot A: Buys VEV_6000 + VEV_6500 systematically at 0 (lottery tickets)
- Bot B: Trades VF Extract and HYDROGEL as the "deep liquidity" market maker (the "wall")
- Bot C: Trades VEV_4000 as a delta-1 proxy for VF Extract
- No bot appears to run a proper delta-hedged options book (this is OUR opportunity)

**Watch for in live rounds:**
- Any systematic timing pattern in trades (regular intervals → exploitable)
- Large size trades in VF Extract that could be Olivia-equivalent → track daily min/max
- If IV suddenly spikes on a specific strike → someone is taking a position → follow

---

*Based on: Frankfurt Hedgehog IMC Prosperity 3 writeup + Round 3 EDA analysis*
*Created: 2026-04-25*
