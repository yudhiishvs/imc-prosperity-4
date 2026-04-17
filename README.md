## ASH_COATED_OSMIUM — EMA-Based Market-Making with Momentum Fade

**Fair value estimation:** Uses an Exponential Moving Average (EMA) of mid-price (`α = 0.3`) as its dynamic fair value, falling back to a static `10,000` if no data exists yet.

**Execution pipeline** (sequential stages, short-circuiting on emergency):

1. **Take mispriced orders** — Sweeps any ask ≤ fair and any bid ≥ fair (inclusive on both sides), acting as an aggressive taker when the book is mis-priced relative to EMA.

2. **Flatten at fair** — If the algo has residual inventory after step 1, it tries to close toward flat by hitting resting orders exactly *at* the fair price.

3. **Emergency flatten** — If projected position exceeds `78` (long or short), it market-sells/buys aggressively at best bid/ask to pull position down to `30`. If triggered, **all further quoting is skipped** for that tick.

4. **Momentum-tilted layered quoting** — Posts a 2-tier quote (inner + outer) on both sides:
   - **Momentum fade logic:** If mid-price moved *up* since last tick, it shifts quotes *down* (sell more aggressively, buy more defensively) and vice-versa — a classic mean-reversion/fade approach.
   - **Inventory skew:** Quote sizes are skewed via a `position_ratio × 1.5` aggression factor — the more long you are, the smaller the bid qty and vice-versa.
   - **Kill switch:** At `±75` position, the refilling side's quotes are zeroed out entirely (e.g., at +75, no bids are posted).
   - **Outer quote anchoring:** Dynamically set one full spread-width away from the inner quote when the book has both sides.

**Key design choices:**
- The inner quote uses a `0` tick offset (i.e., penny-jumps at best), capturing `90%` of total quote size; the outer layer gets the remaining `10%`.
- It's a **mean-reverting** strategy that treats Osmium's volatility as noise around an adaptive fair value.

---

## INTARIAN_PEPPER_ROOT — Detrended Long Bias with Two-Phase State Machine

**Fair value estimation:** Models Pepper Root as a **linearly trending asset** with slope `0.001` per timestamp unit. On the first tick, it calibrates a `base_estimate` and then extrapolates: `fair = base + slope × t`.

**Two-phase state machine:**

### Phase 1: Accumulation (`reached_80 = False`)
- Aggressively buys up to the position limit (`80`):
  - Takes any asks up to `fair + 8` (very permissive, willing to overpay).
  - Posts remaining unfilled size as a resting bid at `fair`.
- Goal: reach max long position (`80`) as fast as possible to ride the uptrend.

### Phase 2: Post-Max Market-Making (`reached_80 = True`)
Once position has hit 80, the strategy transitions permanently and does two things:

1. **Scalp sells** — Sells into bids at `fair + 4` or higher, capped at `3` units per tick. This captures spikes above fair value without giving up much position.

2. **Long-biased penny-jump MM** — Places penny-jump bids and asks inside the spread, but heavily skewed toward the bid side:
   - Bid qty is inflated by the deficit from 80 (always trying to reload).
   - If position drops below `77`, asks are zeroed out entirely — only bids are placed.
   - **L2 quality gate**: Only activates when L1-to-L2 price gaps are small (bid gap ≤ 6, ask gap ≤ 5), filtering out illiquid or gappy book states.
   - **OIM (Order Imbalance) dynamic shifts**: Adjusts bid/ask prices based on L1 volume imbalance — fades heavy-bid/heavy-ask conditions.

3. **Recoup buying** — After scalping, any remaining deficit from 80 is refilled by taking asks up to `fair - 2` (only willing to *underpay* now, much tighter than phase 1's +8) and resting bids at that level.

**Key design choices:**
- The phase transition is **one-way** — once `reached_80 = True`, it never reverts.
- The strategy's entire thesis is that Pepper Root trends upward, so holding max long is optimal. Selling is only done opportunistically into spikes.

---

**State Persistence:** Both strategies serialize four values across ticks via `jsonpickle`: Pepper's base estimate and phase flag, and Osmium's EMA and last mid-price.
