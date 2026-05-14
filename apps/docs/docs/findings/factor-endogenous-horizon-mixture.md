---
tags:
  - factor-narrow
  - partial-OOS
  - hypothesis-user
---

# Endogenous-horizon mixture — state-conditional rebal cadence beats random-π but not the best fixed-h baseline

A pre-registered test of the hypothesis that rebal cadence should be a
model output instead of a 20-day constant: emit per-bar scores **and** a
softmax `π_t` over horizon bins `{5, 10, 20, 40, 60}`, deploy at
`argmax(π_t)`, hold flat between rebals. The pre-registered
success criterion required `endog Sharpe ≥ best-fixed + 0.10 AND beats
random-π AND no π collapse`. Result: endog beats random-π convincingly
(+0.12) and beats every fixed-h baseline, but the lift over the best
fixed (h=60) is only **+0.048** — under the +0.10 threshold.

Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels) for the
state-conditional-horizon hypothesis. The signal **exists** (endog wins
on the 2/6 windows where π is mixed and ties or loses on the 4/6 windows
where π collapses to a single horizon), but the model finds the
degenerate "always pick h=60" attractor too eagerly. The operational
rule: **a per-bar mixture loss without entropy regularization will
collapse to the lowest-turnover horizon globally.** Next experiment is
an entropy-weight sweep (see
[`TODO/factor-horizon-entropy-reg`](../TODO/factor-horizon-entropy-reg.md))
to force the model to use the mixture before another window-level
verdict is taken.

## Setup

| | |
|---|---|
| Universe | `factor-narrow` (297 stooq_us_long names, `min_history_bars=6500`) |
| Windowing | 6 walk-forward windows at h_min=5 fine grid (train=252 fine bars ≈ 5y, val=156 ≈ 3y, step=156) |
| Horizons | `{5, 10, 20, 40, 60}` days (K=5) |
| Architecture | Shared MLP trunk (`hidden=32, n_layers=1`) on 74-channel `IndicatorGridConfig`; per-ticker score head + per-bar K-way horizon head pooled over the liquid cross-section |
| Loss | `-mean_t Σ_k π_t[k] · IC_k_t` — per-bar Pearson IC weighted by state-conditional π_t. No entropy regularization (the run we ran). |
| Eval | Daily PnL stream on the model-emitted irregular cadence, `Sharpe = mean / std × √252`, 10 bps one-sided turnover cost. |
| Baselines | Fixed-h ∈ {5,10,20,40,60} (same scores, fixed cadence); random-π (same scores, uniform horizon sampling). |

Pre-registered null hypotheses:

| # | Null | Threshold |
|---|---|---|
| N1 | π collapse (single-bin share > 90% globally) | argmax-bin global share ≤ 0.90 |
| N2 | Beats fixed h_max=60 | endog Sharpe > fixed-h=60 |
| N3 | Beats best fixed by ≥ 0.10 | delta ≥ +0.10 |
| N4 | Beats random-π | delta ≥ 0 |

Compute placement: Modal T4 cloud, ~4 min walk-forward wall +
feature/build time (~$0.07).

## Result (2026-05-14)

### Aggregate

| Arm | Mean val Sharpe |
|---|---:|
| **endog (argmax π)** | **+0.448** |
| fixed h=5 | +0.327 |
| fixed h=10 | +0.322 |
| fixed h=20 | +0.288 |
| fixed h=40 | +0.399 |
| **fixed h=60** (best fixed) | **+0.401** |
| random-π | +0.329 |

| Null | Observed | Verdict |
|---|---|---|
| N1 (no π collapse, share ≤ 0.90) | h=60 share = **0.80** | PASS (just) |
| N2 (beats fixed h_max=60) | +0.448 − +0.401 = **+0.047** | PASS |
| N3 (beats best fixed by ≥ 0.10) | delta **+0.048** | **FAIL** |
| N4 (beats random-π) | delta **+0.119** | PASS |

Pre-registered overall: `confirmed-OOS` iff all four pass.
**Three of four pass; N3 fails by 0.05.** Per verdict vocabulary this is
`partial-OOS`.

### Per-window stratification

The headline aggregate masks a bimodal per-window pattern: in 2/6
windows the model uses a real mixture and wins; in 4/6 it collapses to
h=60 and loses.

| Win | Val start | Endog | Best fixed (which) | Delta | π entropy | π argmax mode |
|---|---|---:|---:|---:|---:|---|
| w0 | 2005-11-15 | **−0.27** | +0.10 (h=10) | **−0.37** | 0.03 | h=60 (156/156) |
| w1 | 2008-12-24 | +0.57 | +0.57 (h=60) | 0.00 | 0.04 | h=60 (156/156) |
| w2 | 2012-03-22 | +0.82 | +0.88 (h=5) | −0.07 | 0.02 | h=60 (156/156) |
| w3 | 2015-06-22 | **+0.84** | +0.82 (h=40) | **+0.02** | 0.32 | h=40 (30) + h=60 (126) |
| w4 | 2018-08-20 | +0.45 | +0.46 (h=5) | −0.01 | 0.04 | h=60 (156/156) |
| w5 | 2021-09-28 | **+0.28** | +0.26 (h=40) | **+0.02** | 0.19 | h=10 (21) + h=40 (135) |

Aggregate is dragged down by **w0 (2005-11)**: the model confidently
picked h=60 (entropy 0.03) into the run-up to the GFC, and the long-
horizon position underperformed every alternative including the model's
own short-horizon arms (h=5/10 both positive at +0.075/+0.102 on this
window). Drop w0 and the remaining 5 windows show endog at +0.59 vs
best-fixed mean +0.59 — a tie, no longer a +0.05 loss.

**Both mixture windows (w3, w5) beat best-fixed by +0.02**, and both
are the only windows where the entropy gets meaningfully above 0.04.
The endogenous selection adds value *when the model actually uses the
mixture*. The problem is that without an entropy bonus, the loss
surface rewards collapse: at any state where h=60 has the highest
expected IC (the modal case), the per-bar loss pulls π toward a δ on
h=60 — and once there, the policy is locked in for the rest of the
training run.

![Per-window endog Sharpe vs fixed-h arms and π argmax-bin counts](images/factor-endogenous-horizon-mixture.png)

### π argmax distribution across all val bars

| Horizon | Share |
|---|---:|
| h=5  | 0.00 |
| h=10 | 0.02 |
| h=20 | 0.00 |
| h=40 | 0.18 |
| **h=60** | **0.80** |

Four of the five horizons are essentially unused. The horizon head
learned "long horizons are cheap" (lower turnover cost → easier to
register positive Sharpe) more than it learned "state predicts useful
IC half-life".

## Why endog still beats random-π by +0.12

Random-π samples uniformly from `{5, 10, 20, 40, 60}` per rebal, so
**60%** of its mass is on h≤20 (high-turnover, high-cost). The endog
policy concentrates 80% on h=60 — a 4× reduction in expected turnover
cost. The +0.12 lift over random-π is largely **cost-reduction**, not
state-conditional skill. This is consistent with N4 passing but N3
failing: beating random is easy; beating the best fixed-h baseline
requires picking *the right horizon for the state*, which the
collapsed policy cannot do.

## Mechanism — why π collapses

The training loss `-mean_t Σ_k π_t[k] · IC_k_t` rewards two things:

1. Per-bar correlation between scores and forward returns at horizon k
   (the IC signal).
2. *Choosing* horizons whose IC is large at that state (the π signal).

But Pearson IC at h=60 is structurally larger in magnitude than IC at
h=5 (longer horizon = more signal-to-noise in the per-bar correlation,
even before sign). On most bars, the best-IC horizon *is* h=60, so the
gradient on π pulls toward a δ on h=60. Once π_t is near δ, the
gradient on `π_t[k≠60]` is small (those terms barely contribute to
the loss), so the model gets stuck. There's no force balancing the
collapse — the loss has no entropy term.

The fix is to add `−α·H(π_t)` to the loss (already implemented as
`entropy_weight` in `objectives.horizon_mixture_loss`). The next
experiment is to sweep α and find the smallest value that keeps the
mixture-window pattern (w3, w5) alive across all 6 windows. If the
hypothesis is right, an `α` exists that makes all 6 windows look like
w3/w5 — and that arm clears N3.

## What this rules out — and what it doesn't

**Rules out**: state-conditional horizon selection via a naive per-bar
mixture-of-horizons IC loss, without regularization. Confirmed it
finds the degenerate "always-cheapest-horizon" attractor.

**Does not rule out**: state-conditional horizon selection in general.
The two windows where the model actually used the mixture beat the
best fixed baseline by +0.02 each, which is small but in the right
direction. The mechanism (random-π loses by costs alone; mixture beats
best-fixed when it fires) is consistent with the hypothesis having
real content underneath the collapse.

## Entropy-weight sweep (2026-05-14, followup)

Pre-registered in
[`TODO/factor-horizon-entropy-reg`](../TODO/factor-horizon-entropy-reg.md):
sweep `entropy_weight α ∈ {0.0, 0.05, 0.1, 0.2, 0.3}` to test whether
regularization rescues the partial-OOS verdict. Hypothesis was that
π collapse was the binding constraint and an α that keeps the mixture
alive would unlock the architecture.

**The hypothesis is falsified.** Entropy reg works exactly as
designed on π (mean H(π) climbs from 0.11 → 1.55 across the sweep —
the model uses all 5 horizons at α≥0.1), but the deployment Sharpe
does **not** improve at any α. The best result remains the
unregularized α=0 arm.

| α | mean endog | best-fixed | Δ-fix | Δ-rand | H(π) | argmax h=60 share |
|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | **+0.448** | +0.401 | **+0.048** | +0.119 | 0.11 | **0.80** |
| 0.05     | +0.408 | +0.397 | +0.011 | +0.121 | 1.10 | 0.78 |
| 0.10     | +0.422 | +0.395 | +0.028 | +0.135 | 1.40 | 0.71 |
| 0.20     | +0.420 | +0.397 | +0.023 | +0.136 | 1.52 | 0.66 |
| 0.30     | +0.401 | +0.392 | +0.009 | +0.119 | 1.55 | 0.60 |

Every arm lands `partial-OOS` (N1+N2+N4 pass, N3 fails). N3 delta
peaks at α=0 (+0.048) and falls toward +0.009 as α rises. **Δ-rand
stays roughly constant** at +0.12–0.14 across the sweep — endog
always beats random-π by a stable ~cost-savings margin, regardless
of π entropy.

### Per-window stratification across α

| Win | Val start | α=0 endog | α=0.05 | α=0.1 | α=0.2 | α=0.3 | α=0 best-fix |
|---|---|---:|---:|---:|---:|---:|---:|
| w0 | 2005-11 | **−0.27** | −0.27 | −0.25 | −0.25 | **−0.30** | +0.10 |
| w1 | 2008-12 | +0.57 | +0.52 | +0.56 | +0.56 | +0.56 | +0.57 |
| w2 | 2012-03 | +0.82 | +0.80 | +0.78 | +0.79 | +0.80 | +0.88 |
| w3 | 2015-06 | **+0.84** | +0.68 | +0.71 | +0.67 | +0.70 | +0.82 |
| w4 | 2018-08 | +0.45 | +0.45 | +0.50 | +0.49 | +0.42 | +0.46 |
| w5 | 2021-09 | +0.28 | +0.27 | +0.23 | +0.26 | +0.23 | +0.26 |

Two windows tell the whole story:

1. **w0 stays catastrophic regardless of α** (−0.25 to −0.30 across
   the sweep, vs best-fixed +0.07 to +0.10). Forcing the model to
   mix horizons does not rescue the 2005-11 pre-GFC window — none
   of the K horizons are correctly placed for that regime.
2. **w3 was the α=0 win and entropy reg breaks it** (+0.84 at α=0
   drops to +0.66–0.71 at α≥0.05). The unregularized model had
   already found a useful state-conditional mixture (30 bars at
   h=40 + 126 at h=60) at this window; forcing higher entropy
   replaces good selections with worse ones.

Entropy reg is doing what it's supposed to do mathematically —
preventing collapse — but the architecture is **not** bottlenecked
on collapse. It's bottlenecked on the fact that the score head's
information at horizon h=40 (or any other horizon) does not vary
meaningfully with market state in this universe + feature stack.

![α=0.1 sweep arm — per-window Sharpes, argmax bin stack, π entropy](images/factor-endogenous-horizon-mixture-alpha0p1.png)
![α=0.3 sweep arm — same panels at maximum regularization](images/factor-endogenous-horizon-mixture-alpha0p3.png)

### What the sweep rules out

| Hypothesis from prior partial-OOS verdict | Sweep verdict |
|---|---|
| π collapse is the binding constraint | **Falsified** — π expands at α≥0.05, endog Sharpe doesn't improve |
| The α=0 +0.05 lift is state-conditional skill on h=60 | **Reframed** — it's cost concentration; the lift disappears as the model uses h=60 less |
| An α exists that clears N3 by ≥ +0.10 | **Falsified within the swept range** — best Δ-fix is +0.048 at α=0 |
| Discrete mixture-of-horizons-IC has untapped state-conditional content | **Falsified** at this architecture / universe / feature stack |

### Verdict on the entropy-reg hypothesis

[`confirmed-null`](../leaderboard.md#verdict-labels) on entropy
regularization as a rescue for the architecture. The original
2026-05-14 partial-OOS row for the unregularized run stands; the
entropy-weight sweep does **not** supersede it.

### Operational rule

Where state-conditional behavior exists in a per-bar softmax model,
the collapse-to-cheapest-attractor is *also* the local optimum.
**Forcing higher entropy doesn't surface new content — it just
trades cost-concentration for noise.** Before adding entropy reg to
a softmax decision head, verify there's evidence of useful
state-conditional content underneath the collapse (e.g., a
held-out arm with manually-set state-dependent π that beats the
collapsed baseline). Without that evidence, the architecture is
already at its ceiling.

### What this leaves open (resolved by the regime-gated follow-up below)

After the entropy sweep, two threads remained: (1) was the +0.05 lift
state-conditional skill or cost-concentration? (2) would a different
*kind* of horizon-emission — specifically a hand-engineered macro
regime gate (VIX state) — work where the learned mixture didn't?
Both are answered by the regime-gated experiment in the next section.

## Regime-gated horizon selection (2026-05-14, closure)

Pre-registered as Option D in the design discussion: apply the
workspace's strongest operational rule —
**"regime filter > richer predictor"** from
[`prediction-problem-pivot-arc`](prediction-problem-pivot-arc.md) and
[`vol-surface-v3-regime-gated`](vol-surface-v3-regime-gated.md) — to
horizon choice instead of learning it. Use VIX vs 126d rolling
median as the regime variable. Two mapping arms test opposite
direction priors:

- **`inverted-vol`**: VIX high → h=5 (responsive); VIX low → h=60
  (low-turnover). "Stress = signal moves faster" prior.
- **`same-vol`**: opposite mapping (null-direction check).

Plus baselines: `fixed-h5`, `fixed-h60`, `random-h` (uniform random
horizon per bar). All arms eval on the same daily-PnL irregular-
cadence Sharpe; same universe, windowing, friction as the rest of
the arc. Score head: vanilla `train_scorer_indicators_walkforward`
linear rank-IC at `rebal_days=5`. Local-only, ~2 min wall.

### Result — confirmed-null on every pre-registered cut

| Arm | Mean val Sharpe | Mean holding days |
|---|---:|---:|
| **learned mixture α=0** | **+0.448** | (variable; argmax π) |
| fixed-h60 | +0.221 | 60.0 |
| same-vol (high→60, low→5) | +0.196 | 17.1 |
| random-h | +0.116 | 28.2 |
| fixed-h5 | −0.061 | 5.0 |
| **inverted-vol (high→5, low→60)** | **−0.123** | 20.0 |

| Null | Threshold | Observed | Verdict |
|---|---|---|---|
| N1: inverted-vol beats fixed-h60 by ≥ +0.10 | ≥ +0.10 | **−0.344** | **FAIL** |
| N2: inverted-vol beats same-vol by ≥ +0.05 | ≥ +0.05 | **−0.319** | **FAIL** |
| N3: inverted-vol beats learned mixture by ≥ +0.05 | ≥ +0.05 | **−0.572** | **FAIL** |
| N4: inverted-vol beats random-h sanity floor | > 0.00 | **−0.239** | **FAIL** |

All four nulls fail. The direction prior reverses (`same-vol` beats
`inverted-vol` by +0.32 — holding *longer* in stress is better than
shortening), but more decisive: every regime-gated arm loses to
plain `fixed-h60`. Even `random-h` beats `inverted-vol` by +0.24.
[`confirmed-null`](../leaderboard.md#verdict-labels) on the regime-
gating hypothesis at this universe and feature stack.

### Per-window detail

| Win | Val start | VIX-hi% | fixed-h5 | fixed-h60 | inverted-vol | same-vol | random-h |
|---|---|---:|---:|---:|---:|---:|---:|
| w0 | 2005-11-15 | 66% | −0.343 | −0.855 | −0.571 | −0.213 | −1.026 |
| w1 | 2008-12-24 | 44% | −0.488 | +0.517 | −0.439 | +0.468 | −0.007 |
| w2 | 2012-03-22 | 19% | +0.229 | +0.678 | −0.358 | +0.305 | +0.914 |
| w3 | 2015-06-22 | 40% | +0.084 | +0.448 | +0.211 | +0.203 | +0.307 |
| w4 | 2018-08-20 | 76% | +0.186 | +0.365 | +0.233 | +0.296 | +0.319 |
| w5 | 2021-09-28 | 32% | −0.032 | +0.173 | +0.183 | +0.115 | +0.187 |

`inverted-vol` loses in 5/6 windows. `same-vol` ties `fixed-h60` on
average but with materially higher variance window-to-window.
`random-h` is competitive — which mostly says that on this
universe and feature stack, the deployment Sharpe is approximately
*independent of horizon choice* given a reasonable mean holding.

### The reframe — what the regime-gated null tells us about the
mixture's +0.05 lift

Compare two equivalent quantities:

| | Score-head training | Deployment | Sharpe |
|---|---|---|---|
| vanilla rank-IC (this run) | per-bar IC at h=5 only | fixed h=60 | **+0.221** |
| learned mixture α=0 | per-bar IC × π (collapsed to ~80% h=60) | fixed h=60 (from prior row) | **+0.401** |

Same architecture (linear head on 74-channel
`IndicatorGridConfig`), same universe, same windowing, same
deployment cadence — different *score-head training horizon*.
The mixture's α=0 score head is implicitly trained at h≈60 (where
π puts its mass), and **scores +0.18 higher on the same
fixed-h60 deployment** than a vanilla h=5-trained head.

That ~+0.18 gap is the **score-head specialization effect**. It
recovers what was previously attributed to "the mixture chose
horizons cleverly". The mixture's +0.05 lift over its own
best-fixed baseline in the original
[partial-OOS row](../leaderboard.md#master-table) sits inside the
+0.18 specialization gap. Most of the architecture's value comes
from *training the score head for the deployment horizon*, not from
*picking horizons at deployment time*.

In other words: the endogenous-horizon arc was solving the wrong
problem. The right operational rule is "train the score head at
the horizon you intend to deploy at" — which is just the
existing factor convention (`rebal_days=20` baseline at val
Sharpe +0.44 in
[`factor-indicator-baseline`](factor-indicator-baseline.md)), with
the added datum that h=60 specialization beats h=20 on this
specific eval setup.

### Arc closure

The endogenous-horizon mixture-of-IC architecture is fully
explored on `factor-narrow`:

| Sub-hypothesis | Test | Verdict |
|---|---|---|
| State-conditional horizon selection unlocks a +0.10 Sharpe lift | Mixture trainer α=0 (2026-05-14, this page top) | `partial-OOS` (+0.048, under threshold) |
| Entropy regularization rescues the partial-OOS | α-sweep `{0.05, 0.1, 0.2, 0.3}` | `confirmed-null` — π expands cleanly but Sharpe doesn't follow |
| Hand-engineered VIX regime gate beats learned mixture | This section | `confirmed-null` — regime-gating loses to even random-h |
| Mixture's +0.05 lift is state-conditional horizon skill | Compared to vanilla rank-IC at h=5 → h=60 (this section's reframe) | **Falsified** — the lift is score-head specialization, not horizon choice |

The full arc closes
[`confirmed-null`](../leaderboard.md#verdict-labels) on the
"rebal cadence as model output" thesis at this universe and
feature stack. The orthogonal levers that remain (different
feature stack, continuous horizon head + REINFORCE, learned
value-function metric) are not pre-registered as next experiments
— the falsification of the score-head-specialization confound
makes the higher-priority next move "train factor at
`rebal_days=60` and re-evaluate" rather than continue the
horizon-emission research thread.

### Operational rule (closed-out)

**Before treating horizon as an output of the model, train the
score head at the deployment horizon and check whether the
single-horizon baseline is already at the architecture's ceiling.**
If specializing the score head explains the gap a "state-conditional
horizon" head was supposed to produce, the horizon-emission line of
inquiry is unwarranted. This is a corollary of the workspace's
"regime filter > richer predictor" rule applied at a different
granularity: the predictor's *training target* matters more than
the predictor's *output structure*.

Driver: `apps/factor/scripts/horizon_regime_gated.py` (local CPU,
~2 min wall). Artifacts:
`Output/horizon-regime-gated-windows.npz`. Reuses the existing
`packages/macro` FRED loader for VIX (cached at `.macro-cache/`).

## Master walk-forward log

See [Leaderboard](../leaderboard.md) rows dated 2026-05-14 — three
in the arc: the unregularized mixture (verdict
[`partial-OOS`](../leaderboard.md#verdict-labels)); the entropy-
weight sweep (verdict
[`confirmed-null`](../leaderboard.md#verdict-labels) on the
regularization rescue); and the regime-gated horizon closure
(verdict
[`confirmed-null`](../leaderboard.md#verdict-labels) on
hand-engineered VIX-state horizon selection, plus the
score-head-specialization reframe that closes the full arc).
Artifacts:
`Output/horizon-mixture-{a0,a0p05,a0p1,a0p2,a0p3}-{windows.npz, comparison.png}`,
`Output/horizon-mixture-sweep-summary.json`,
`Output/horizon-regime-gated-windows.npz`. Reproduce:

```bash
# Mixture + entropy sweep (Modal T4)
uvx --from modal modal run apps/factor/scripts/modal/horizon_mixture.py \\
    --entropy-weights '0.0,0.05,0.1,0.2,0.3'

# Regime-gated horizon (local CPU)
uv run python apps/factor/scripts/horizon_regime_gated.py
```
