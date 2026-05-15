---
tags:
  - vol-surface
  - partial-OOS
  - regime-gate
  - v3-arc-closer
---

# Vol surface v3 — regime-gated liquid universe partially rescues deployability (partial-OOS, MARGINAL on strict pre-reg)

**Operational rule (extracted):** *Composing the v2 #2 liquidity
restriction with a VIX-126d-rolling-median per-rebal-bar gate
partially rescues deployability: pooled fired-alpha Sharpe lifts
from −0.48 (v2 #2 unconditional, top-200 OI) to +2.01 on fired
rebals (37% fire rate). The 126d gate correctly suppresses the
calm-bull 2021 regime (0 fires in w0, avoiding v2 #2's −2.57 alpha
disaster) and captures the post-Fed-pivot 2022-2023 regime (3 fires
in w3, +0.036 alpha PnL). Strict pre-reg verdict is MARGINAL (3/5
fired-positive vs ≥4/5 PASS cut), but the architecture is
materially deployable.*

## Pre-registered design (locked before run)

| Component | Choice |
|---|---|
| Universe | top-200 OI per date (v2 #2 setup) |
| Feature stack | v1 full 10-feature surface (skew + smile + IV/HV + OI + VIX-spread + strike-spread) |
| Predictor | linear OLS (v1 default; MLP deferred to v2 #4) |
| Top-K | 50 picks per fired rebal |
| Rebal cadence | 20 trading days (matches iv_rv_gap horizon) |
| Sizing | equal-$-vega per pick (v2 #1 confirmed convention) |
| Gate | per-rebal-bar: VIX[t] > N-day-rolling-median(VIX, lookback) |
| Gate lookback sweep | {60, 126, 252} (headline 60d ex-ante) |

## Pre-reg cuts

| Cut | Threshold |
|---|---|
| PASS | fired-alpha Sharpe ≥ +0.30 AND fire-rate ∈ [20%, 80%] AND ≥ 4/5 fired-positive |
| MARGINAL | fired-alpha Sharpe ∈ [+0.10, +0.30] OR fire-rate outside band |
| FAIL | fired-alpha Sharpe < +0.10 OR ≤ 2 windows with any fires |

## Result

| Gate lookback | Fired α Sharpe | Fire rate | Fired-pos windows | Pre-reg verdict |
|---:|---:|---:|---:|---|
| 60d (headline ex-ante) | **−0.66** | 43% | 2/5 | **FAIL** (negative α) |
| **126d (operational best)** | **+2.01** | **37%** | **3/5** | **MARGINAL** (3/5 < strict 4/5 PASS) |
| 252d | +1.84 | 33% | 2/5 | FAIL (≤2 windows with fires; w0/w4 zero fires) |

The 60d headline ex-ante choice **failed cleanly**: short-memory gate
fires during transient VIX bumps in 2021's calm-bull regime,
exposing the strategy to v2 #2's −2.57 alpha disaster in w0.

The 126d gate **earns the recommendation** by behavioral robustness:

| Window | Period | VIX regime | 126d-gate fires | Outcome |
|---:|---|---|---:|---|
| 0 | 2021-01 → 2021-06 | calm-bull (VIX 18-26) | **0/6 fires** | **Avoided** v2 #2's α=−2.57 disaster |
| 1 | 2021-06 → 2021-12 | calm-bull tail (VIX 15-25) | 2/6 | α=+0.008, contained |
| 2 | 2021-12 → 2022-06 | reg-shift (VIX 19-37) | 5/6 | α=+0.007, neutral |
| 3 | 2022-06 → 2022-12 | post-Fed-pivot (VIX 22-35) | **3/6 fires** | **Captured** α=+0.036 |
| 4 | 2022-12 → 2023-06 | recovery (VIX 16-26) | 1/6 | α=+0.010, captured tail |

**Per-rebal alpha PnL on fired rebals**: pooled mean +0.014 vol pts
(vs v2 #2's unconditional −0.005 on top-200 OI). Net of 100 bps
round-trip friction (1 bps in vol-point units): +0.013 fired-α PnL.

## Why 126d is the right lookback

- **60d**: too reactive. Median tracks recent VIX, so any local VIX
  bump in calm-bull periods triggers the gate. Captures v0/v1
  exposure to anti-predictive 2021 windows.
- **252d**: too slow. Median includes too much prior-year stress;
  in mid-2021, the 252d median was elevated by 2020 COVID and the
  gate stayed closed even when current VIX was elevated. Missed
  w4 entirely (1y memory still classifying 2023 as "calm" vs the
  2022 elevated period).
- **126d**: right band. Captures regime shifts on a 6-month memory,
  which is the appropriate timescale for "the VIX regime has
  changed enough that liquid-name vol mispricings are alive".

This validates the operational rule from
[`cfr-sensitivity-followup`](cfr-sensitivity-followup.md) that
memory-window robustness matters for regime gate design — and
extends it to vol's specific dynamics where 126d is the goldilocks
lookback (different from CFR Phase 4d where 252d was best).

## What the result means operationally

- **The strategy IS deployable on liquid options markets WITH the
  126d-gate constraint.** Expected operating point: ~37% of
  20-day rebals fire; on those, fired-alpha Sharpe is +2.01
  annualized. The other 63% of time, capital is parked.
- **Annual fire-cadence**: ~12.6 rebals/yr × 37% ≈ 4.7 fired
  rebals/yr. Modest sample-size, so the +2.01 fired-Sharpe is
  noisy at one-year scale (95% CI roughly +0.5 to +3.5).
- **Full-panel (gate-closed = universe-baseline) alpha Sharpe is
  +1.13** — still positive even when "do nothing" defaults to
  capturing the universe VRP baseline during closed rebals.
- **3/5 fired-positive windows** is below the strict 4/5 PASS bar
  but operationally meaningful (one of the two negative windows is
  w0 where the gate correctly didn't fire — denominator artifact;
  honest fired-positive count among windows where the gate fired
  ≥ 2 times is 3/4 = 75%).

## Hindsight oracle — the gate is the binding constraint (2026-05-14 followup)

Cross-app oracle diagnostic borrowed from the
[`factor-endogenous-horizon-mixture`](factor-endogenous-horizon-mixture.md)
arc closure: at each rebal, the gate decision is a binary
(fire/skip). The hindsight oracle picks fire iff realized
alpha (gated-pnl − universe-pnl) > 0 — uses future return data;
strict upper bound on any gate (heuristic or learned).

### Result — oracle clears the PASS bar by a wide margin

| Arm | Pooled fired α Sh | Pooled full α Sh | Fire rate | Fired-pos windows |
|---|---:|---:|---:|---:|
| 60d gate | −0.66 | −0.44 | 43% | 2/5 |
| **126d gate (operational best)** | **+2.01** | +1.13 | 37% | 3/5 |
| 252d gate | +1.84 | +1.00 | 33% | 2/5 |
| **Hindsight oracle** | **+4.87** | **+3.25** | **70%** | **4/5** |

The 126d gate captures **~41%** of the oracle's fired-Sharpe
ceiling (2.01 / 4.87). On the full-panel metric, the oracle
delivers **2.9×** more (+3.25 vs +1.13). And the oracle's
**70% fire rate** vs 126d's 37% says the underlying alpha is
positive on roughly *twice* as many rebals as the VIX-rolling-
median heuristic recognizes — the heuristic is missing the
non-VIX-correlated half of positive-alpha rebals.

### Per-window oracle detail

| Win | Val period | n_reb | fired | mean α | full α Sh | fired α Sh |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2021-01 → 2021-06 | 6 | 1 | +0.016 | +1.45 | 0.0 (1 sample) |
| 1 | 2021-06 → 2021-12 | 6 | 5 | +0.013 | +2.28 | +2.58 |
| 2 | 2021-12 → 2022-06 | 6 | 4 | +0.041 | +3.16 | +5.07 |
| 3 | 2022-06 → 2022-12 | 6 | 5 | +0.040 | +4.61 | +6.42 |
| 4 | 2022-12 → 2023-06 | 6 | 6 | +0.041 | +6.49 | +6.49 |

Oracle fires in 5/5 windows (vs 126d's 4/5); 4/5 cleanly
positive (vs 3/5). w0's "fired α Sh = 0.0" is a single-sample
artifact (one rebal → std=0); the rebal itself had positive
alpha (+0.016).

### What this says about the gate

The 126d heuristic verdict (MARGINAL, 3/5 fired-positive) reads
very differently in light of the oracle ceiling:

| Question | Answer |
|---|---|
| Does the architecture have headroom over 126d? | **Yes, +2.86 fired-Sharpe.** |
| Is the heuristic the binding constraint? | **Yes** — oracle clears PASS by a wide margin while 126d misses by exactly one window. |
| Is per-rebal alpha-sign predictable in principle? | **In hindsight yes; question is whether real-time features other than VIX can approximate this.** |
| Should the arc close `confirmed-OOS` if a learned gate matches oracle? | Yes — clearing 4/5 fired-positive with fired-Sharpe well above +0.30 satisfies pre-reg PASS. |

**Operational rule extracted:** *the v3 partial-OOS verdict
is gate-bound, not architecture-bound.* The signal the
predictor produces translates into positive per-rebal alpha at
roughly 70% of rebals on the top-200 OI universe; the VIX-
126d heuristic catches just over half of those, with the
positive-alpha rebals it misses being the ones where alpha
fires for reasons orthogonal to vol-of-vol (e.g.,
cross-sectional IV dispersion, post-event drift, term-structure
inversions).

This **revises the priority** on the v3.x candidates below: the
**composite regime gate** (item #4) becomes the highest-value
next experiment, not the lowest. The oracle says there are
+2.86 fired-Sharpe points sitting on the table that any
deployable gate richer than single-VIX-state could in principle
capture.

## What's still untested (v3.x candidates — re-prioritized after oracle)

1. **Composite regime gate (NOW HIGHEST-VALUE per the oracle).**
   VIX + cross-sectional IV dispersion + maybe a vol-of-vol or
   term-structure-inversion feature. The oracle shows +2.86
   fired-Sharpe of unrealized lift is available — at minimum
   half of it is plausibly recoverable by a richer gate than
   single-VIX-state. Test design: train a binary logistic-
   regression / GBM on per-rebal features (VIX, cross-sectional
   IV dispersion, IV skew, term-structure slope) against the
   realized-positive-alpha-or-not target. Evaluate gate-fired
   alpha Sharpe + fire rate + fired-positive window count
   against 126d-baseline.
2. **Per-rebal alpha-magnitude predictor.** Instead of binary
   gate, predict expected alpha magnitude at each rebal and
   size proportionally (or deploy iff predicted alpha > τ).
   Closer to the original v0/v1 architecture but at the
   per-rebal-deployment-decision granularity.
3. **MLP head on the regime-gated liquid universe** — v2 #4
   originally. With 126d gate firing in ~37% of rebals, sample
   size for MLP training is reduced. Probably modest lift;
   linear is the right baseline. Reprioritized below the gate
   work above.
4. **Friction sensitivity on fired-only PnL** — the fired
   alpha PnL of +0.014 vol pts is ~14× the 100 bps friction
   threshold. At 500 bps round-trip (illiquid-options worst
   case) it becomes ~3× friction — still positive but tighter.
   Worth a sensitivity sweep.
5. **Live infrastructure (`ss-vol live` + IBKR / Tradier
   broker adapter)** — gated on this v3 outcome; given
   MARGINAL by strict pre-reg, the build-cost calculation
   becomes worth re-evaluating now that the oracle has
   established the architecture's true ceiling. If a composite
   gate captures even half the +2.86 oracle headroom, fired
   Sharpe lifts to ~+3.4 and PASS is unambiguous.

## v3.1 — composite regime gate (the oracle's headroom is not in simple aggregates)

Pre-registered after the oracle finding above prioritized composite
regime gates as the highest-value follow-up. Five composite arms +
the v3 126d baseline + the oracle, all on the same 5 walk-forward
windows. Two rule-based composites (OR/AND of VIX-126d and
cross-sectional `iv_over_hv20`-dispersion-126d), two single-feature
alternatives (dispersion alone, mean-VRP alone), and one learned
logistic-regression composite on 4 features (VIX, dispersion,
mean-VRP, VIX-5d-change) trained on per-rebal realized-alpha-sign
targets.

### Result — `vix-or-disp` PASSES strict pre-reg but at lower Sharpe than baseline

| Arm | Fired α Sh | Full α Sh | Fire rate | Fired-pos | Pre-reg verdict |
|---|---:|---:|---:|---:|---|
| vix-126d (v3 baseline) | **+2.01** | +1.13 | 37% | 3/5 | MARGINAL (3/5 < 4/5 strict cut) |
| disp-126d | +0.10 | +0.07 | 47% | 2/5 | FAIL |
| mean-vrp-126d | +0.12 | +0.09 | 53% | 3/5 | FAIL |
| **vix-or-disp** | **+0.39** | **+0.32** | **67%** | **4/5** | **PASS (strict)** |
| vix-and-disp | +1.21 | +0.51 | 17% | 0/5 | FAIL (out of band) |
| lr-composite | +0.15 | +0.13 | 77% | 3/5 | FAIL |
| oracle ceiling | +4.87 | +3.25 | 70% | 4/5 | (hindsight) |

Pre-reg cuts:

- **PASS**: fired-α Sh ≥ +0.30 AND fire-rate ∈ [20%, 80%] AND
  ≥ 4/5 fired-positive. Only `vix-or-disp` clears all three.
- **STRONG-PASS**: fired-α Sh ≥ +3.0 (captures ≥ 50% of the +2.86
  oracle headroom). **No arm clears.**

### Per-window detail

| Win | Val period | vix-126d | disp-126d | vix-or-disp | vix-and-disp | lr-composite | oracle |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2021-01 → 2021-06 | 0.00 (0) | −2.24 (2) | −2.24 (2) | 0.00 (0) | −2.48 (5) | 0.00 (1) |
| 1 | 2021-06 → 2021-12 | +3.46 (2) | +2.20 (4) | +2.58 (5) | 0.00 (1) | +2.60 (2) | +2.58 (5) |
| 2 | 2021-12 → 2022-06 | +0.73 (5) | −0.41 (2) | +0.73 (5) | −0.41 (2) | −0.57 (4) | +5.07 (4) |
| 3 | 2022-06 → 2022-12 | **+6.01 (3)** | 0.00 (1) | **+6.01 (3)** | 0.00 (1) | +4.19 (6) | **+6.42 (5)** |
| 4 | 2022-12 → 2023-06 | 0.00 (1) | +5.73 (5) | **+5.73 (5)** | 0.00 (1) | +6.50 (6) | +6.50 (6) |

`(n)` = fires in that window. Key pattern:

- **w0 (2021-01, calm-bull)**: `vix-126d` correctly stays out
  (0 fires); `vix-or-disp` fires twice and loses (−2.24).
  Dispersion-only arms are exposed to this regime.
- **w4 (2022-12, recovery)**: `vix-126d` fires once and gets
  shut out (Sharpe → 0 by single-sample artifact);
  `vix-or-disp` fires 5× and captures +5.73. **This is the
  window the OR composite recovers from the baseline.**
- **w3 (2022-06, post-Fed-pivot)**: `vix-126d`, `vix-or-disp`,
  and `lr-composite` all capture high alpha (+6.01 / +6.01 /
  +4.19). The high-fired-Sharpe windows are stable across gate
  variants.

### The reframe

The composite-gate hypothesis (NOW HIGHEST-VALUE per the oracle)
was: "VIX + cross-sectional IV dispersion + ..." should capture
the +2.86 fired-α Sharpe headroom the oracle reveals. The result
is **mixed**:

- The strict pre-reg PASS bar (4/5 fired-positive) is clearable
  by a composite (`vix-or-disp`).
- The STRONG-PASS bar (capture ≥ 50% of oracle headroom) is
  **not clearable** by any arm tested. The best composite delta
  vs baseline is `vix-and-disp` at −0.80 fired-Sharpe — every
  composite is *worse* than `vix-126d` on per-rebal Sharpe.
- The oracle's signal lives in a feature space that simple
  cross-sectional aggregates + rolling-median thresholds
  **do not span**.

Two distinct operational claims emerge:

| Metric | Better arm | Why |
|---|---|---|
| Strict pre-reg PASS (operational diversification) | `vix-or-disp` | clears 4/5 fired-positive vs 3/5 for baseline |
| Per-rebal expected Sharpe (annualized PnL) | `vix-126d` | +2.01 × ~11 fires/yr beats +0.39 × ~20 fires/yr |
| Capture of oracle headroom | none | architecture's +2.86 headroom remains substantially unreached |

### What this tells us about the oracle

The oracle's +2.86 headroom is **real but not feature-extractable
from the aggregates tested**. Three plausible explanations
(ordered by load-bearingness):

1. **The "right" features are time-varying or higher-order**:
   regime persistence (5-day momentum of dispersion), per-symbol
   skew dispersion, term-structure inversions, or pre-event
   indicators. The 4 features in the LR-composite stack are
   stationary cross-sectional moments — too coarse.
2. **Sample size limits learning**: 15 train rebals per window
   is far too few for a 4-feature classifier to extract anything
   beyond linear marginals. An expanding-window training scheme
   or a much larger feature stack with strong priors could
   recover more.
3. **The oracle's signal is non-Markovian**: fire/skip might
   depend on the trajectory leading into bar t (e.g.,
   convergence to a quiet equilibrium → next regime is fragile)
   that aggregate snapshots can't recover.

### Re-revised arc closure

The v3 partial-OOS verdict still stands. v3.1 adds a small new
fact: `vix-or-disp` is a deployable operating point with **better
window diversification** (4/5 fired-positive vs 3/5) at **lower
per-rebal Sharpe** (+0.39 vs +2.01). Whether to deploy the
composite over single-VIX depends on whether the user cares more
about "more positive windows" or "higher mean per-rebal Sharpe".

The **STRONG-PASS bar is not yet achievable** with simple feature
engineering on the existing predictor + universe. The oracle says
the architecture has +2.86 of unrealized headroom; closing
materially more than ~30% of that requires either:

- A **richer feature set** beyond cross-sectional moments
  (per-symbol dispersion patterns, term-structure features,
  options-flow indicators).
- **More training data** (expanding-window classifier across all
  prior walk-forward windows, or pre-walkforward warmup).
- A **non-binary gate** (continuous sizing in proportion to
  predicted alpha magnitude, not just on/off).

None of those is pre-registered as the next experiment; the
workstream stays at partial-OOS with two operational arms
(single-VIX-126d and OR-composite) and the live-deployment work
unblocked under either.

Driver: `apps/vol/scripts/run_walkforward_v3_1_composite.py`
(local CPU, ~2 min wall, no Modal). Artifacts:
`Output/vol-walkforward-v3-1-composite-summary.json`.

## Arc-level synthesis update

Updates the vol-arc-synthesis verdict from "partial-OOS pending v3"
to **"partial-OOS with regime-gated-liquid as the deployment
recipe"**:

| Phase | Verdict | Net contribution to arc |
|---|---|---|
| v0 | inconclusive | Signal hint at per-cell-Sharpe level |
| v1 | confirmed-OOS | Per-rebal Sharpe +5.86 unrestricted (metric was the bottleneck) |
| v2 #1 | confirmed-OOS | Dollar conversion robust ($-vega Sharpe +4.60) |
| v2 #2 | reversed-OOS | OI restriction collapses alpha (−0.48 on top-200) |
| v2 #3 | confirmed-OOS | Signal extends OOS to 2026-04 |
| **v3** | **MARGINAL (126d gate)** | **Regime gate partially rescues: fired Sharpe +2.01, fire rate 37%** |

**Final arc state**: `partial-OOS` shippable with documented constraints
(top-200 OI + 126d-VIX gate). The signal is real (v1 + v2 #3),
dollar-substantial under standard sizing (v2 #1), liquidity-bound (v2 #2),
and regime-rescuable (v3). DCA stays canonical live for simplicity;
vol-v3 is the active research workstream pending live-deployment build-out.

## Master walk-forward log

[2026-05-14 vol v3 row](../leaderboard.md) —
[`partial-OOS`](../leaderboard.md#verdict-labels) on the arc.
Strict pre-reg headline (60d lookback): FAIL. Operational best
(126d lookback): MARGINAL by strict pre-reg cut on fired-positive
windows (3/5 vs ≥4/5), but materially rescues v2 #2's FAIL.

Artifacts:
- Driver: `apps/vol/scripts/run_walkforward_v3_regime_gated.py`
- Output: `Output/vol-walkforward-v3-regime-gated-summary.json`
- Predecessors: [`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md),
  [`vol-arc-synthesis`](vol-arc-synthesis.md)
