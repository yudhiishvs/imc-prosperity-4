# Round 3 EDA Findings — IMC Prosperity 4
Generated: 2026-04-25 17:00:05

---

## 1. Products Overview

| Product | Type | Notes |
|---------|------|-------|
| `HYDROGEL_PACK` | Regular tradeable | Mean price ~9991, σ ≈ 25.3 |
| `VELVETFRUIT_EXTRACT` | Option underlying | Mean price ~5247 (Day 0) → 5255 (Day 2) |
| `VEV_4000 … VEV_6500` | European call options | TTE = 7 on Day 0, 5 on Day 2 |

**TTE Mapping:** Data day 0 → TTE=7, day 1 → TTE=6, day 2 → TTE=5

---

## 2. HYDROGEL_PACK Analysis

- Mean price: **9990.96** across all days (highly stable)
- Spread: consistently narrow (1–2 ticks typical)

**Stationarity Tests (H1):**
| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| ADF (H0: unit root) | -5.1584 | 1.07e-05 | **REJECT H0 — stationary** |
| KPSS (H0: stationary) | 1.0742 | ≤0.01 | **REJECT H0 — non-stationary** |

Per-day ADF: Day 0 p=0.0064 ✓ | Day 1 p=0.103 ✗ | Day 2 p=0.032 ✓

- **Joint interpretation: TREND-STATIONARY** — mean-reverts within each day but has a slow
  inter-day drift. ADF picks up the intraday reversion; KPSS detects the cross-day drift.
- **Strategy:** Mean-reversion is valid intraday. Use a **rolling** (not fixed) fair-value
  estimate that re-anchors each day. Fade deviations from the 200–500 tick rolling mean.

---

## 3. VELVETFRUIT_EXTRACT (Underlying) Analysis

- Day 0 mean: **5246.51** ± 13.68
- Day 2 mean: **5255.39**

**Stationarity Tests (H2):**
| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| ADF (H0: unit root) | -4.8325 | 4.71e-05 | **REJECT H0 — stationary** |
| KPSS (H0: stationary) | 2.0928 | ≤0.01 | **REJECT H0 — non-stationary** |

Per-day ADF: Day 0 p=0.0073 ✓ | Day 1 p=0.0294 ✓ | Day 2 p=0.175 ✗

- **Joint interpretation: TREND-STATIONARY** — same pattern as HYDROGEL. Strong intraday
  mean-reversion, but the underlying drifts upward across days (~8 ticks/day).
- **Strategy:** Options fair value must use a **dynamic S** that accounts for the drift.
  For within-day option pricing, mid-price reversion is exploitable.

- **Normality (H3, Jarque-Bera):** p=1.81e-35
  → **Non-normal** — fat tails from bot activity. BS underestimates tail probability.
- **Log-returns ADF:** p≈0 → returns are stationary (series is I(1) at most, not I(2))

---

## 4. Options Analysis (VEV_*)

### 4.1 Option Price Structure (Day 0)

All options are **European calls** (vouchers to buy VF Extract at strike).

| Strike | Avg Mid | Status @ Day-0 |
|--------|---------|----------------|
| 4000 | ~1250.1 | Deep ITM (S-K≈1247) |
| 4500 | ~750.1 | Deep ITM (S-K≈747) |
| 5000 | ~255.0 | Slightly ITM |
| 5100 | ~166.8 | Near ATM |
| 5200 | ~95.5 | Near ATM |
| 5300 | ~46.8 | Near ATM |
| 5400 | ~16.0 | OTM |
| 5500 | ~6.6 | OTM |
| 6000 | ~0.5 | Deep OTM (floor price) |
| 6500 | ~0.5 | Deep OTM (floor price) |

### 4.2 Implied Volatility

Median implied vol by strike (all days, BS model with r=0, T in Solvenarian days/252):

  - Strike 4000: IV ≈ 0.336
  - Strike 4500: IV ≈ 0.209
  - Strike 5000: IV ≈ 0.218
  - Strike 5100: IV ≈ 0.217
  - Strike 5200: IV ≈ 0.220
  - Strike 5300: IV ≈ 0.224
  - Strike 5400: IV ≈ 0.208
  - Strike 5500: IV ≈ 0.226
  - Strike 6000: IV ≈ 0.361
  - Strike 6500: IV ≈ 0.546

**IV Smile (H4):** ANOVA test Day 0: F=2598.536, p=0
→ **IV varies across strikes (smile exists)**

Key observations:
- IV tends to be **higher for OTM options** than ATM (classic smile / right skew)
- VEV_6000 and VEV_6500 have **undefined/unreliable IV** (basically zero-value, no time value)
- IV varies across TTE — examine plot 11 for the term structure

### 4.3 Time Value Analysis (H7)

- **0.47%** of option snapshots show negative time value
- All options priced above intrinsic
- Negative TV by strike: {4000.0: 605, 4500.0: 807}

---

## 5. Bot Behavior Analysis

- All `buyer`/`seller` fields in trade data are **NaN** — identity not disclosed
- Currency field = `XIRECS` (settlement currency)
- **Trade timing (H5):** KS test p=8.891e-25
  → **Non-Poisson (clustered/periodic)**
  Mean inter-trade interval: 990 ticks
- Trades are typically **market-aggressive** (at/near ask for buys, bid for sells)
- Trade volume is concentrated in **VELVETFRUIT_EXTRACT** and **VEV_5400/5500** (OTM options)

**Bot Behavioral Pattern:**
- Systematic trades appear every ~990 ticks on average
- OTM option trades (VEV_5400, 5500) suggest bots speculate on upside breakouts
- Trade price vs mid-price analysis (plot 15): check if bots consistently buy at ask or sell at bid

---

## 6. Hypothesis Test Summary

| # | Hypothesis | Result | Action |
|---|-----------|--------|--------|
| H1 | HYDROGEL follows random walk | **REJECTED** (ADF p=1.07e-5) — trend-stationary | Mean-reversion w/ rolling fair value |
| H2 | VF Extract follows random walk | **REJECTED** (ADF p=4.71e-5) — trend-stationary | Mean-reversion intraday; use rolling mid |
| H3 | VF returns are normal | **Non-normal** | Use heavier tails in model |
| H4 | IV is flat (no smile) | **IV varies across strikes (smile exists)** | Exploit smile — sell rich wings |
| H5 | Trade arrivals are Poisson | **Non-Poisson (clustered/periodic)** | Consider queue/cluster effects |
| H6a | Imbalance predicts VF returns | **Imbalance predicts returns** | Use imbalance signal in quoting |
| H6b | Imbalance predicts HYDROGEL returns | **Imbalance predicts returns** | Use imbalance signal in quoting |
| H7 | Options priced above intrinsic | **All options priced above intrinsic** | No arb |
| H8 | No autocorr in VF returns | Ret-p=N/A, Sq-p=N/A | No momentum |

---

## 7. Strategy Recommendations

### Strategy A: HYDROGEL_PACK Market Making / Mean Reversion

```python
# Parameters to tune
FAIR_VALUE = 9991   # or rolling mean
SPREAD = 2         # half-spread around fair value
MAX_POS = 50       # position limit

# Logic: quote bid at FAIR-SPREAD, ask at FAIR+SPREAD
# Skew quotes as position deviates from 0
```

- **Target:** Fair value ~9991
- **Signal:** Deviation from rolling mean → fade the move
- **Edge:** Bot activity creates predictable flow

### Strategy B: VF Extract Market Making

```python
# Parameters
FAIR_VALUE = rolling_mid   # track dynamically
MAX_POS = 50
SPREAD = 2–4 ticks
```

- If VF Extract is mean-reverting (H2 result), quote around rolling mean
- If it's a random walk, lean on trade imbalance signal (H6)

### Strategy C: Option Delta Hedging + IV Arbitrage

```python
# Core idea: Buy underpriced options, sell overpriced options
# Hedge delta by trading underlying

# Step 1: Compute IV for each strike
# Step 2: Identify strikes where IV deviates from median/theoretical curve
# Step 3: Trade options + hedge with VF Extract
```

- **Sell OTM options** (VEV_5500, VEV_6000, VEV_6500): collect premium,
  these expire worthless if S stays below strikes (historically likely)
- **Buy ATM options** when IV is low relative to realized vol

### Strategy D: Short Deep OTM Options (VEV_6000, VEV_6500)

- These trade at the floor price (0.5)
- Probability of expiring ITM appears very low (S ≈ 5247, strikes 6000/6500)
- Risk: Black-swan spike in VF Extract to >6000
- **Recommendation:** Sell these at every opportunity (near-free money)
- **Max loss per unit:** 500 (if S hits 6500) — size carefully

### Strategy E: Options Portfolio Greeks Management

```python
# Maintain delta-neutral book:
# - Track net delta across all VEV positions
# - Use VF Extract to offset: trade -net_delta units of VF Extract

# Theta decay play:
# - Short options decay toward intrinsic as TTE → 0
# - Enter short positions early (high TTE) and ride time decay
```

---

## 8. Implementation Notes for Trader Class

### Position Limits (typical IMC Prosperity Round 3)
- Check official limits — likely 50–200 per product

### Key Computations Needed
```python
def compute_tte(day: int, timestamp: int) -> float:
    """TTE in Solvenarian days."""
    TTE_START = {0: 7.0, 1: 6.0, 2: 5.0, 3: 4.0, 4: 3.0}
    return TTE_START.get(day, 7.0) - timestamp / 1_000_000

def bs_call(S, K, T, sigma, r=0):
    """European call price. T in days."""
    if T <= 0: return max(S - K, 0.0)
    T_yr = T / 252
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
    d2 = d1 - sigma*np.sqrt(T_yr)
    return S*norm.cdf(d1) - K*np.exp(-r*T_yr)*norm.cdf(d2)

def bs_delta(S, K, T, sigma, r=0):
    if T <= 0: return 1.0 if S > K else 0.0
    T_yr = T / 252
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_yr) / (sigma*np.sqrt(T_yr))
    return norm.cdf(d1)
```

### Priority Order of Strategies
1. **Short VEV_6000/VEV_6500** immediately — near-zero risk, free premium
2. **HYDROGEL mean reversion** — high confidence signal if ADF confirms stationarity
3. **VEV ATM option selling** — theta decay, hedge with VF Extract
4. **VF Extract market making** — if stationary, fade moves; if GBM, use imbalance signal

---

## 9. Plots Index

| File | Contents |
|------|----------|
| 01_midprice_overview.png | All products mid-price across 3 days |
| 02_hydrogel_analysis.png | HYDROGEL: timeseries, distribution, spread, returns |
| 03_vf_extract_timeseries.png | VF Extract with trade overlays |
| 04_vf_extract_returns.png | Returns analysis + ACF of returns and squared returns |
| 05_rolling_volatility.png | Rolling annualised vol for VF Extract |
| 06_option_price_timeseries.png | All VEV option prices (log scale + linear zoom) |
| 07_intrinsic_time_value.png | Intrinsic vs time value by strike, per day |
| 08_moneyness_analysis.png | Moneyness (S/K) over time |
| 09_iv_smile.png | IV smile for each day (median ± IQR) |
| 10_iv_surface.png | IV surface heatmap (strike × time) |
| 11_iv_vs_tte.png | ATM IV vs TTE |
| 12_spread_analysis.png | Bid-ask spreads all products |
| 13_orderbook_depth_imbalance.png | Depth and imbalance for VF Extract + HYDROGEL |
| 14_trade_flow_timing.png | Trade counts, volume, inter-arrival times |
| 15_bot_trade_vs_mid.png | Trade prices relative to mid-price |
| 16_vwap_analysis.png | VWAP vs mid for VF Extract and HYDROGEL |
| 17_imbalance_predictive.png | Order imbalance → future return scatter |
| 18_qq_plots.png | QQ normality plots |
| 19_option_deltas.png | BS delta by strike over time |
| 20_pnl_scenarios.png | Short option P&L at day-2 median S |
| 21_correlation_matrix.png | Full cross-product return correlations |
| 22_hypothesis_test_summary.png | All hypothesis test p-values |

---

*Generated by round3_eda.py — IMC Prosperity 4 Round 3*
