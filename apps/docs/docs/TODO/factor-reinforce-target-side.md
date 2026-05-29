# `apps/factor` — target-side REINFORCE for π (train against Sharpe, not rank-IC)

## Closed 2026-05-15 — higher-β resolved; the endogenous-horizon rescue arc is benchmark-artifact-bound

The higher-β sweep (β ∈ {16, 32}, the `partial-OOS` default-next
from β=8) ran single-arm, provenance-verified. **β=32 Δ-fix +0.108
clears the literal +0.10 PASS cut** (β=16 +0.091 does not). But
the mandated window stratification is decisive:

> Against the **per-window** hindsight-best fixed horizon the
> mixture is **0/6 wins** at both β=16 and β=32 (3 ties, 3 losses).
> The learned π never beats committing to the single best horizon
> for a window. The +0.108 is `mean_endog − single_global_best_fixed`
> — a benchmark artifact of comparing a switching policy to one
> horizon fixed across heterogeneous windows, dominated by w0's
> structural negativity.

**Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels)**,
user-adjudicated *not* promoted to `confirmed-OOS` (a hollow PASS
would be the repo's 10th in 146 runs and corrupt the 9/146 base
rate the strategic frame rests on).

**This closes the workstream.** The retrospective implication is
arc-wide: every positive Δ-fix in the endogenous-horizon arc
(+0.048 baseline, β=8 +0.095, β=32 +0.108) is the same
single-global-fixed-benchmark artifact — the discrete
mixture-of-horizons-IC architecture **does not extract
state-conditional skill on factor-narrow**.

Superseded next-steps (do **not** run — pre-empted by the
stratification):

- **Higher-β still (β ∈ {64, 128})** — moot; the metric, not β, is
  the problem. More β cannot turn 0/6 per-window wins positive.
- **w0 regime gate** — moot; gating w0 mechanically lifts the
  cross-window mean without creating any per-window skill (a
  misleading rescue from the same artifact).

Live follow-ups, both *orthogonal* (neither pre-registered):

1. **Output-side restructure** — per-horizon specialised score
   heads, mixture *over* specialists. A genuinely different
   architecture, not another π-training intervention. Only worth a
   pre-reg if the mixture architecture is not retired outright.
2. **Novel-data arc (higher-EV, per the standing strategic frame
   and the research director's steer)** — borrow-stress
   conditioning (SEC FTD + FINRA daily short-volume + short
   interest) on the *liquid* vol universe. Orthogonal to
   cleverer-model-on-standard-data **at the prediction layer with
   forecast / rank-IC loss** (scope qualifier added 2026-05-29 —
   sizing-layer direct-Sharpe learners are a separate, partially-
   positive story per
   [`findings/learned-ensemble-beats-deterministic.md`](../findings/learned-ensemble-beats-deterministic.md)
   and
   [`notes.md#learner-layer-matters-more-than-learner-complexity`](../notes.md#learner-layer-matters-more-than-learner-complexity));
   see the new
   `TODO/vol-borrow-liquid-universe` pre-reg.

Finding: extended in
[`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
under "Target-side REINFORCE — higher-β sweep ... the stratification
correction that caps the arc". Leaderboard:
2026-05-15 target-side-REINFORCE higher-β row (`partial-OOS`).

---

## Resolved 2026-05-15 — `partial-OOS` (MARGINAL), first positive lever in the rescue arc

Sweep β ∈ {0, 0.5, 2.0, 8.0} ran on Modal T4 (single-arm jobs — a
Modal stdout-buffering bug drops all-but-last-arm artifacts on
multi-arm runs).

| β | mean endog | Δ-fix | verdict |
|---:|---:|---:|---|
| 0 (cached baseline) | +0.448 | +0.048 | partial-OOS |
| 0.5 | +0.435 | +0.039 | partial-OOS |
| 2.0 | +0.359 | −0.003 | confirmed-null |
| **8.0** | **+0.453** | **+0.095** | **partial-OOS (MARGINAL)** |

**β=8 is the first rescue in the four-attempt arc that lifts Δ-fix
above the +0.048 baseline** (entropy reg +0.011, bilevel −0.001,
horizon-aligned grid −0.036 all `confirmed-null`). MARGINAL per
pre-reg: Δ-fix +0.095 ≥ +0.07 with 5/6 positive windows, just below
the +0.10 PASS cut.

**Two turning-point findings**:

1. Non-monotonic phase transition in β: degrades through β≤2 (dips
   to −0.003 at β=2) then jumps at β=8. The Sharpe-residual signal
   needs β≈8 to fully escape the rank-IC h=60 attractor; gentle
   nudges are strictly worse than baseline.
2. First rescue that does NOT damage the w3 canary: w3 endog +0.855
   (vs default +0.84). The prior three all dropped w3 by −0.18 to
   −0.29. At w3 the mixture matches the best achievable fixed
   horizon (fixed-h40 +0.86) — π is correctly state-selecting.

The mean-centered + std-normalized Sharpe-residual is the first
intervention whose expected gradient direction is *orthogonal* to
rank-IC's (mean-centering kills the "raise-all-returns" component;
std-normalization is the Sharpe-shaping). The rank-IC-vs-Sharpe
misalignment the three prior failed rescues identified is real and
*addressable* by changing π's training-signal direction.

### Next-experiment chain (partial-OOS → stratify windows)

Workstream continues — partial-OOS is not a close. Per the
default-next-question for partial-OOS:

1. **Higher-β sweep β ∈ {16, 32}** (highest value, ~25 min Modal
   each, run single-arm). The response is monotone-up past the β=2
   dip; β=16/32 may clear the +0.10 PASS cut and convert
   `partial-OOS` → `confirmed-OOS`. **Pre-reg cuts**: PASS if best-β
   Δ-fix ≥ +0.10 AND ≥ 5/6 positive; STRONG-PASS if ≥ +0.12 AND 6/6
   positive; FAIL/plateau if Δ-fix ≤ +0.095 (β=8 was the peak).
2. **w0 regime gate**: w0 (2005-11) is the only negative window and
   is negative under *every* config + *every* fixed horizon —
   structurally hard, not a π failure. A window-level gate that
   flags w0-like regimes (candidate features: low cross-sectional
   dispersion, pre-GFC low-vol grind, VIX below trailing median)
   and de-risks would lift the mean by removing the −0.21 drag.
   **Pre-reg cuts**: PASS if gated-mean Δ-fix ≥ +0.12 AND the gate
   fires on w0 but not on ≥ 4 of the 5 positive windows.
3. **Output-side restructure (#2)**: per-horizon score heads, each
   trained on its own h-specific rank-IC, mixture *over* the
   specialized heads. The pre-reg flagged this as the next pre-reg
   if REINFORCE PASSed; at MARGINAL it's a candidate, not yet
   pre-registered (sequence it after the higher-β sweep — if β=16/32
   clears PASS, the architecture question is settled and output-side
   becomes the compounding lever; if it plateaus, output-side
   becomes the orthogonal next bet).

Priority order: (1) higher-β sweep first (cheapest, directly
resolves PASS-vs-MARGINAL), then (2) or (3) depending on its
outcome.

See the extended findings page
[`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
("Target-side REINFORCE" section) for the per-β table, w3-canary
comparison across all four rescues, mechanism, and the final
operational rule.

Leaderboard row: 2026-05-15 target-side-REINFORCE sweep
(`partial-OOS`, MARGINAL on β=8).

---

## Original pre-registration (preserved for the record)

**Status at time of writing**: pre-registered 2026-05-15, sweep not
yet run.

### Hypothesis

The horizon-aligned grid sweep (2026-05-15) closed `confirmed-null`
on the input-side rescue and produced the cleanest diagnosis of the
endogenous-horizon-mixture architecture's binding constraint:

**The score head can extract more short-horizon signal (fixed-h10
Sharpe lifted to +0.437 under horizon-aligned features) but π
can't capitalize because its training signal — rank-IC — over-rewards
h=60 due to highest per-bar IC SNR at long cumulative-return horizons.**

The remaining lever is changing **what π is trained to maximize**.
Currently:

```
L_π = -mean_t Σ_k π_t[k] · IC_k_t      ← rank-IC (the misaligned signal)
```

Target-side intervention swaps the supervision signal entirely.
Instead of teaching π to weight high-IC horizons, teach it to
weight horizons whose realized **Sharpe-contributions** are
positive, via REINFORCE-style policy gradient.

## Why per-bar Sharpe-residual REINFORCE (not full trajectory rollout)

Two REINFORCE forms could implement "train against realized Sharpe":

1. **Full trajectory REINFORCE**: sample a horizon trajectory per
   training step, run `simulate_irregular_daily_pnl` to compute
   episode Sharpe R, use R as the global reward in `∇log π · (R − baseline)`.
   Cleanest semantics but expensive: simulator is numpy (~1-5s per
   call); 200 steps × 6 windows × ~5 arms = ~10-15 minutes of
   pure simulator overhead per arm. Worth it ONLY if the per-bar
   variant fails AND the failure looks like Monte-Carlo-noise-bound.

2. **Per-bar Sharpe-residual REINFORCE** (this pre-reg): the
   trajectory's per-bar reward at the sampled action serves as a
   "per-bar Sharpe contribution"; mean-centering + std-normalizing
   across the trajectory gives the per-bar advantage. Cheap (no
   simulator call) and *structurally different from bilevel*: the
   z-score normalization removes the "raise all returns" signal
   that bilevel had, leaving only "this action improves trajectory
   consistency" — the Sharpe-shaping distinction.

The cheap variant is the right first cut. If it PASSes, ship it.
If it FAILs, escalate to full trajectory REINFORCE.

## Loss specification

```
L_total = L_IC_norm + β · L_REINFORCE
```

Where:

```
L_IC_norm = -mean_t Σ_k π_t[k] · IC_k_t * valid_k_t / std_detached(IC)   ← unchanged from bilevel; trains score head
```

And the REINFORCE term (per training step):

```
# 1. Sample one horizon index a_t ~ Categorical(π_t) at each bar (numpy)
# 2. Compute per-bar reward at sampled action, using detached scores:
#    ret_at_sampled[t] = (centered_scores_detached_t · fwd_log_return_{a_t}_t).mean / horizon_{a_t} − commission/horizon_{a_t}
# 3. Form per-bar advantage as the trajectory Sharpe-residual:
#    m = ret_at_sampled.detach().mean()
#    s = ret_at_sampled.detach().std() + ε
#    advantage_t = (ret_at_sampled_t.detach() − m) / s     ← z-score (mean 0, std 1)
# 4. Score-function gradient on log π_t at the sampled action:
#    log_pi_at_a = log(π[t, a_t] + 1e-12)
#    L_REINFORCE = -mean_t(log_pi_at_a · advantage_t)
```

**The advantage centering is the key difference vs bilevel.** Bilevel's
per-bar return signal pushes π toward any horizon with positive return.
Sharpe-residual mean-centers across the trajectory, so the gradient
only fires when an action's return is *above* the trajectory mean —
the Sharpe-shaping signal. Std-normalization further downweights
gradients on high-variance trajectories.

### Bilevel detach contract

The reward computation uses `scores.detach()`, so the score head's
gradient comes from L_IC only — same as bilevel. The π head's
gradient comes from both L_IC (rank-IC) and L_REINFORCE
(Sharpe-residual). The shared trunk's gradient combines both.

This is the same detach pattern as bilevel; the only thing that
changes is **what the π head's auxiliary loss optimizes for**.

## Sweep

β ∈ {0.0, 0.5, 2.0, 8.0} — 4 arms.

- β=0 reproduces the 2026-05-14 baseline (mean Δ-fix +0.048,
  `partial-OOS`).
- β=0.5 — light Sharpe-residual nudge.
- β=2.0 — Sharpe-residual term has comparable magnitude to rank-IC.
- β=8.0 — Sharpe-residual dominates.

Log-spaced because the right magnitude is unknown. The Sharpe-residual
advantage is already z-scored (mean 0, std 1), and the rank-IC term
is std-normalized, so β is dimensionless.

## Universe / windowing

Identical to 2026-05-14 entropy + 2026-05-15 bilevel + 2026-05-15
horizon-aligned sweeps:

- factor-narrow (297 stooq_us_long, `min_history_bars=6500`).
- 6-window walk-forward at `h_min=5` fine grid.
- Train 252 fine bars × val 156 × step 156.
- Horizons `(5, 10, 20, 40, 60)`.
- Commission 10 bps; temperature 1.0; AdamW; n_steps=200.
- **Default `IndicatorGridConfig`** (74-channel). NOT horizon-aligned
  — we already established that doesn't help and want to isolate the
  target-side intervention's effect on the baseline architecture.
- Seed 0.

## Pre-registered cuts

Apply per-arm; best-arm sets the sweep verdict:

- **STRONG-PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND 6/6
  positive windows.
- **PASS** (`confirmed-OOS`): mean Δ-fix ≥ +0.10 AND ≥ 5/6 positive.
- **MARGINAL** (`partial-OOS`): mean Δ-fix ≥ +0.07 AND ≥ 4/6 positive.
- **FAIL** (`confirmed-null`): otherwise — target-side intervention
  also fails. Architecture is structurally information-bounded at
  this dataset scale.

The cuts deliberately raise the Δ-fix threshold above the existing
+0.048 baseline. Tying is FAIL — we want the Sharpe-residual signal
to *lift* deployment, not just match it.

## Diagnostic to extract regardless of verdict

For each arm, report:

1. Per-window endog Sharpe + per-window Δ-fix (the standard table).
2. π argmax shares across the 5 horizons (does Sharpe-residual
   open up short-horizon mass like the oracle wanted?).
3. The **w3 (2015-06-30) check**: does this rescue ALSO damage w3
   like the prior three (entropy reg, bilevel, horizon-aligned
   grid)? If yes, the architecture's fragility is invariant across
   intervention types. If no — w3 survives or improves — that's
   the first sign of an intervention that doesn't break the working
   window.

## Why this might work where bilevel didn't

The bilevel objective added raw per-day return as π's auxiliary
supervision. Failed because:

1. Per-day return is noisier than rank-IC at this dataset scale.
2. The signal direction is the same as rank-IC's (both reward
   high score×return covariance); just at different magnitudes.
3. Adding noise without changing the gradient *direction* couldn't
   pull π off the h=60 attractor.

REINFORCE with Sharpe-residual changes the **direction**:

1. **Mean-centering**: only above-mean actions get positive
   advantage. Bilevel's gradient pulled toward all-positive-return
   horizons; Sharpe-residual pulls toward *better-than-average*
   horizons.
2. **Std-normalization**: high-variance trajectories produce
   small-magnitude advantages → less weight in the gradient.
   Bilevel had no variance penalty; Sharpe-residual implicitly
   penalizes noisy bars.
3. **Score-function gradient**: REINFORCE estimates the gradient
   via samples, which adds variance but doesn't change expected
   direction. If the expected direction is genuinely orthogonal
   to rank-IC's, REINFORCE can pull π off the attractor where
   pathwise gradients couldn't.

The directional differences are the load-bearing prediction. If
the result FAILs the same way bilevel did, that's strong evidence
the architecture is bound by something orthogonal to all of these
interventions.

## Honest acknowledgements before running

1. **Single-sample REINFORCE has high gradient variance.** Per-bar
   advantage z-scoring is the variance reducer; in expectation
   it's still single-trajectory. If convergence is unstable, the
   first remedy is multi-trajectory sampling (sample M actions per
   bar, average).
2. **Sharpe-residual is computed within-batch, not across batches.**
   So it normalizes per training step's trajectory, not a moving
   average across steps. Each step sees its own per-bar mean and
   std as constants. This is the cheapest, most stable variant
   but lacks the running-baseline variance reduction that classical
   REINFORCE uses.
3. **Sampling temperature is implicit at 1.0** (softmax direct).
   If π collapses fast and sampling becomes near-deterministic, the
   gradient variance drops but so does exploration. Worth checking
   π entropy across the sweep — if entropy crashes to <0.1 quickly,
   add exploration via temperature or entropy regularization.
4. **Same dataset as prior arc**. If this also fails on w3, the
   "fragile working window" pattern hardens to four independent
   replications — the architecture's limit isn't intervention-type-
   dependent; it's dataset/architecture-bound.

## Compute

```bash
uvx modal run apps/factor/scripts/modal/horizon_mixture.py \
    --reinforce-weights '0.0,0.5,2.0,8.0'
```

Modal T4, ~30 min wall (slightly longer than bilevel due to per-bar
sampling overhead), ~$0.20.

## Where to land the result

- One leaderboard row for the sweep.
- Extension to
  [`findings/factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
  under "Target-side REINFORCE" — with the per-arm + per-horizon-argmax
  + w3-canary tables.
- Update CLAUDE.md's apps/factor section IF the sweep PASSes
  (operational rule: "train π with target-side REINFORCE on
  Sharpe-residual; score head stays rank-IC").
- Close this TODO with verdict pointer.

## Concept link

If this works, it's the first intervention in the
[endogenous-horizon arc](../findings/factor-endogenous-horizon-mixture.md)
to genuinely close the rank-IC-vs-Sharpe-deployment misalignment.
**Output-side restructure (#2)** becomes the next pre-reg if PASS
to test whether per-horizon score heads compound the lift, OR moves
to the "open follow-ups" graveyard if FAIL.

If it fails — joining entropy reg, bilevel return, and horizon-aligned
grid as the fourth independent confirmed-null on this architecture —
the operational rule becomes: **the discrete mixture-of-horizons-IC
architecture's deployment ceiling at +0.448 on factor-narrow is
binding; the next experiment isn't an intervention on the mixture
but a different architecture entirely (per-horizon specialized score
heads, or end-to-end Sharpe-trained scorer)**.
