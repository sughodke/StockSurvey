# `apps/critic` — Φ-quality v0 (confirmed-null on the value-function-from-aggregate-data hypothesis)

## Operational rule

**Window-level state features (macro VIX/SPY scalars at val_start) +
cross-app action one-hot are insufficient signal to learn
deployment-Sharpe with the ~30-triple-per-app data we have on disk.**
A tiny MLP Φ trained on 134 window-level (app, action, sharpe) triples
fails to beat the per-action-mean baseline on any of the four apps in
the survey. The first-pass "high rank quality" result was a methodological
artifact: when the action vocabulary contains hindsight oracle arms, the
model trivially learns the oracle/non-oracle binary via the action one-hot
and inflates the apparent Spearman. Dropping oracle arms drops Spearman
from +0.65 (median) to **−0.59 (median)** — anti-correlated on held-out.

→ **Don't ship a window-level cross-app value function on the existing
walk-forward aggregates.** The natural follow-ups are (a) pair-level Φ
for pairs (300 training samples already on disk, richer per-pair
features), and (b) per-rebal triple emission across apps (requires
re-running walk-forwards with new emission code). The v0 result does
NOT close either follow-up — they test different prediction problems
at different granularities.

## Eval setup

| Knob | Value |
|---|---|
| Triples | 134 window-level: factor 60 (10 actions × 6 windows), gate 24 (4×6), pairs 30 (5×6), vol 20 (4×5) |
| State features | VIX-level, VIX-6m-change, VIX-1m-change, SPY 252d log-ret, SPY 63d log-ret, app one-hot (4 dims) — 9 dims total |
| Action features | One-hot over the 23-dim cross-app action vocab |
| Model | MLP (in_dim=32, hidden 16, 2 layers, ReLU); 817 params; tinygrad runtime |
| Loss | MSE + 1e-3 L2 on weights |
| Training | Adam, lr=1e-2, 300 steps per LOO fold |
| CV | Leave-one-(app, window)-out — 23 folds |
| Standardization | Train-set stats per fold (no test leakage) |
| Baselines | global-mean (no signal), per-app-mean (app effects only), per-action-mean (app + action effects, the toughest baseline) |

Pre-registered cut for App-PASS: `RMSE_Φ / RMSE_per-action-mean ≤ 0.75 AND Spearman r ≥ +0.20`.
App-MARGINAL: `RMSE ratio ≤ 0.90 AND r > 0`.

## Results: with oracle arms (deceptive)

| App | n_folds | n_eval | RMSE_Φ | RMSE_per-action | rel | Spearman r | Verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factor | 6 | 60 | 1.927 | 0.431 | 4.466 | −0.100 | **FAIL** |
| gate | 6 | 24 | 1.430 | 0.744 | 1.922 | **+0.754** | FAIL |
| pairs | 6 | 30 | 1.069 | 0.664 | 1.611 | **+0.645** | FAIL |
| vol | 5 | 20 | 7.423 | 3.049 | 2.435 | **+0.361** | FAIL |

Overall: **FAIL (confirmed-null)** by pre-reg.

Note the Spearman signature: three of four apps show high positive
rank correlation (gate +0.75 is particularly striking) despite RMSE
losing badly. The natural reading would be: "Φ is rank-quality positive
but mis-calibrated in magnitude" — useful for policy training even if
unfit for confidence estimation.

That reading is wrong.

## Diagnostic re-run: drop oracle arms from the vocab

Hindsight oracle arms (`gate:oracle-day`, `gate:oracle-dd`, `pairs:oracle-pos`,
`pairs:oracle-top-quartile`, `vol:v3:lookback-oracle`) have realized-data
leakage by construction. Including them in the action vocabulary lets Φ
learn the trivial classifier "this action_key contains 'oracle' → high
Sharpe" purely from the action one-hot dimension — no state-conditioning
required. Drop them:

| App | n_eval | RMSE_Φ | RMSE_per-action | rel | Spearman r | Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| factor | 60 | 1.695 | 0.431 | 3.929 | **−0.189** | FAIL |
| gate | 12 | 0.746 | 0.283 | 2.632 | **−0.853** | FAIL |
| pairs | 18 | 1.189 | 0.690 | 1.724 | **−0.510** | FAIL |
| vol | 15 | 3.483 | 3.024 | 1.152 | **−0.595** | FAIL |

Overall: **FAIL (confirmed-null)** — same verdict, but for a fundamentally
different reason than the first pass.

**The oracle-clean run is the honest result.** Spearman *flips negative*
on all four apps. The v0 "rank quality" of the first run was 100% the
model learning the binary oracle classifier — once the oracle/non-oracle
discriminator is removed, Φ has *no learnable state-conditional
ranking signal* and in fact ranks *worse than random* on held-out
windows. The remaining signal in the action one-hot is just the
per-action mean (which the baseline already captures); Φ's only marginal
contribution is overfit noise.

## Mechanism

Three things compound:

1. **The per-action-mean baseline is very strong.** Most apps have ~5
   actions across all windows; the train-set mean per action captures
   the bulk of variance. To beat it, Φ must extract *state-conditional*
   information beyond the action's marginal mean. With 9 state features
   (5 macro scalars + 4 app one-hot), the macro signal is too coarse
   to discriminate per-window deployment-Sharpe at this label noise.

2. **Off-policy distribution shift compounds the small-data problem.**
   The training data sampled a small set of hand-built policies (entropy
   weights, binary gate types, lookback values). Φ has roughly 4-5
   training observations per (app, action_key) cell. When a window
   is held out, the in-vocab actions that vary little across windows
   (per-action mean) match well; Φ's "smarter" predictions amount to
   adding noise.

3. **Cross-app pooling does NOT help.** The factor/vol/gate/pairs apps
   have completely different action vocabs and different label scales
   (vol Sharpe range −3.5 to +6.5; factor Sharpe range −0.6 to +0.9).
   The app one-hot is necessary but the cross-app gradient that
   pooling produces fights itself — each app's optimal feature use is
   different.

## Why this is `confirmed-null` and not `diagnostic`

The pre-registration pinned a falsifiable test: `RMSE_Φ ≤ 0.75 ×
RMSE_per-action-baseline AND Spearman r ≥ +0.20` for App-PASS, ≥ 3/4
apps to clear. The result fails the cut by wide margins on every app.
The diagnostic *interpretation* (high apparent Spearman = oracle
detection) reframes *why* it failed, but doesn't change the verdict.

The hypothesis being falsified: "Φ trained on cross-app window-level
triples beats per-action-mean baselines on deployment-Sharpe prediction."
That hypothesis is *confirmed false* at this dataset granularity.

The hypothesis NOT being falsified by this run:
- "Pair-level Φ on richer per-pair features beats LR on the same problem"
  — being tested next.
- "Per-rebal granularity Φ (with ~2-3k samples) beats window-level
  baselines" — requires per-rebal emission across apps; deferred.

## Mistake worth keeping

The first-pass result (Spearman +0.65 median) is the kind of false
positive that a less-careful eval would have published. The oracle
arms were included by default in the v0 because they were already in
the walk-forward outputs and dropping them required an extra flag. The
methodological lesson: **when the action vocabulary contains an action
that's perfectly correlated with the label by construction (hindsight
oracle = highest realized Sharpe by definition), any model with action
one-hot input can game it.** Either drop those actions from the
vocabulary, or evaluate the model only on the non-oracle subset, or
swap to a state-only baseline that doesn't see the action identity.

## v0.1 rescue — pair-level Φ for pairs (also confirmed-null)

After v0 closed confirmed-null on the cross-app window-level test, the
natural rescue was pair-level Φ for pairs: a finer-grained prediction
problem with 1200 per-pair training records (6 windows × 200 backtested
pairs) already produced by `apps/pairs/scripts/run_pair_predictor_walkforward.py`
(extended with `pairs-predictor-per-pair-records.npz` emission), 7
training-window pair features (`log_train_half_life`, `abs_corr`,
`log_eg_pvalue`, `abs_hedge_beta`, `train_sharpe`, `train_pct_in_trade`,
`log_train_n_trades`) optionally augmented with 5 macro features at
val_start.

### Pre-registered v0.1 cuts

- **PASS**: `RMSE_Φ / RMSE_LR ≤ 0.85` AND `Spearman r(Φ) ≥ +0.20`
- **STRONG-PASS**: `RMSE_Φ / RMSE_LR ≤ 0.70` AND `r ≥ +0.30` AND mean top-50 Sharpe (Φ) > (LR) + 0.5
- **MARGINAL**: `RMSE_Φ / RMSE_LR ≤ 0.95` AND `r > 0`
- **FAIL**: otherwise

LR baseline = ridge regression on the same features (the v1 LR-predictor
analog).

### Results

| Variant | n_features | RMSE_Φ | RMSE_LR | RMSE_window-mean | rel (Φ/LR) | r(Φ) | r(LR) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| with macro (12 features) | 12 | 0.945 | 0.944 | 0.576 | 1.001 | +0.054 | +0.103 | **FAIL** |
| pair-only (7 features) | 7 | 0.633 | 0.557 | 0.576 | 1.136 | +0.105 | +0.202 | **FAIL** |

Top-50 within-window selection (deployment proxy; mean realized
per-pair Sharpe across 6 held-out folds):

| Arm | Top-50 Sharpe | vs all-pairs baseline | % of oracle headroom |
|---|---:|---:|---:|
| All-pairs baseline | +0.102 | — | 0% |
| LR (with macro) | +0.141 | +0.039 | 5.5% |
| LR (no macro) | +0.141 | +0.039 | 5.5% |
| Φ (with macro) | +0.156 | +0.054 | 7.7% |
| Φ (no macro) | +0.105 | +0.003 | 0.4% |
| Within-window oracle | +0.804 | +0.702 | 100% |

### What the v0.1 numbers tell us

1. **Information is at the pair-feature level, but barely.** The
   pair-only LR captures r = +0.20 Spearman on held-out pairs and
   beats the predict-window-mean baseline by ~3% RMSE. That's a
   small-but-real linear signal — and *all* of it is captured by LR.
2. **Macro features are noise at this granularity.** Adding the 5
   macro features 5x balloons LR RMSE (0.557 → 0.944) because they
   add variance without information; the same row has the same 5
   macro values, so they amount to a per-window offset that the
   window-mean baseline already captures and that LOO-by-window
   sees as out-of-support at deployment time.
3. **Non-linearity hurts.** Φ (a 2-layer MLP) is *worse* than LR on
   every metric: in the no-macro variant Φ even underperforms the
   predict-window-mean baseline. With only 1000 train samples and
   7 features that carry a r=+0.20 Spearman signal, a non-linear
   model overfits the train distribution and adds noise on test.
4. **Top-K is more forgiving than RMSE.** Φ (with macro) does
   eke out a 7.7% capture of oracle headroom on top-50 — slightly
   better than LR's 5.5%. That's a 40% relative lift in argmax
   precision, but it's tiny in absolute Sharpe terms (+0.015) and
   does not clear the pre-reg cuts. The argmax view also matches
   the pairs v1 LR result on the deployment script (predictor-thr-0.5
   captured 5.4% of oracle headroom on the v1 portfolio Sharpe
   metric).

### Why this hardens the v0 verdict instead of changing it

The v0 confirmed-null was about whether *window-level* Φ adds signal
beyond per-action means. The v0.1 confirmed-null is about whether
*pair-level* Φ adds signal beyond per-feature LR. **Both fail at
their natural granularity for the same underlying reason: the
predictive information density of the available features is too low
for any model to lift held-out performance materially.** Adding
non-linearity, adding macro context, swapping classifier — none of
those move the needle once LR has done its job.

This generalizes the existing closing diagnostic in
[`pairs-eg-gate-falsified.md`](pairs-eg-gate-falsified.md):
the v1 LR's 5.4% capture rate is approximately a **ceiling at this
feature space**, not a floor. To lift pair-selection above 5-8%
oracle capture would require a *different feature representation*
— cross-pair rank within window, half-life trajectory across
training history, sector relations, or a learned representation
end-to-end against PnL. Not a different classifier on these same 7
scalars.

### Day-2 (policy training against -Φ) is NOT attempted

The pre-registration's exit conditions are explicit: day-1 FAIL closes
the arc as `confirmed-null`. Φ does not add information beyond the
existing baselines, so a policy trained against -Φ would amount to
a policy trained against a noisy version of the LR baseline — a
strictly worse experiment than just running the LR predictor in
deployment, which we already did (pairs v1, 5.4% capture).

The policy-training step rejoins the backlog as a *conditional*
follow-up: it gets re-prioritized if and only if a future Φ-quality
result PASSes. The natural candidates that could change the day-1
verdict are:

1. **Per-rebal emission + retry**: extend factor/vol/gate walk-forwards
   to dump per-rebal triples (~3-6 hour workstream) and retrain a
   per-rebal Φ. The window-level information loss in v0 was a
   real degradation; per-rebal gives ~10× more samples with per-bar
   labels.
2. **Richer pair features**: half-life trajectory across training
   bars, cross-pair rank within window, sector co-membership, term-
   structure of correlation. Adds genuinely new information not in
   the v1 7-feature stack.
3. **Different decision-structure**: skip the explicit value-function
   entirely and copy the factor mixture pattern — train the action
   distribution end-to-end against PnL. That was the only approach
   that crossed meaningful oracle headroom in the original survey
   (~42% capture on factor).

None of these are pre-registered as next experiments.

## Arc closure

The Φ-value-function arc closes with the same shape as the
prediction-problem-pivot arc that preceded it: **a clean negative
result with a precise diagnostic that re-frames the open question**.
The user's framing in the pre-reg ("learn a value function offline
from existing walk-forward triples") is falsified at the available
data granularity. The framing isn't dead — per-rebal data and
richer features each remain orthogonal levers — but the cheap
version of the idea (use what's on disk, train a small NN) was
adequately tested and adequately falsified.

## Master walk-forward log pointer

- v0 (cross-app window-level): [`confirmed-null`](../leaderboard.md#verdict-labels) row 2026-05-15.
- v0.1 (pair-level for pairs): [`confirmed-null`](../leaderboard.md#verdict-labels) row 2026-05-15.

Pre-registration: [`TODO/critic-phi-value-function`](../TODO/critic-phi-value-function.md).

Implementation:
- `apps/critic/src/critic/{dataset, features, model, eval, pairs_eval}.py`
- `apps/critic/scripts/{run_phi_quality_walkforward, run_pair_phi_quality}.py`
- `apps/pairs/scripts/run_pair_predictor_walkforward.py` (extended with per-pair-records emission)

Artifacts:
- `Output/critic-phi-quality-summary.json` (v0 with oracles)
- `Output/critic-phi-quality-no-oracle-summary.json` (v0 oracle-clean diagnostic)
- `Output/critic-pair-phi-quality-summary.json` (v0.1 with macro)
- `Output/critic-pair-phi-quality-no-macro-summary.json` (v0.1 pair-only)
- `Output/pairs-predictor-per-pair-records.npz` (1200 per-pair training records)
