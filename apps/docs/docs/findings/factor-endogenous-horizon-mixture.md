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

## Master walk-forward log

See [Leaderboard](../leaderboard.md) row dated 2026-05-14 — verdict
[`partial-OOS`](../leaderboard.md#verdict-labels). Artifacts:
`Output/horizon-mixture-{windows.npz, comparison.png}`. Reproduce via
`uvx --from modal modal run apps/factor/scripts/modal/horizon_mixture.py`.
