# `apps/critic` — Φ(state, action) → predicted-deployment-Sharpe value function

## Closed 2026-05-15

[`confirmed-null`](../leaderboard.md#verdict-labels) on the
value-function-from-aggregate-data hypothesis. Two arms tested per
pre-registration:

- **v0 (cross-app window-level)**: 134 window-level triples, tiny MLP,
  LOO-by-(app, window). Apparent rank quality with oracle arms in the
  vocab (gate r=+0.75, pairs r=+0.65) collapsed to NEGATIVE Spearman
  when oracles dropped (gate r=−0.85, pairs r=−0.51). FAIL per pre-reg.
- **v0.1 (pair-level for pairs)**: 1200 per-pair training records, the
  natural rescue at finer granularity. Φ ties LR on RMSE, both worse
  than predict-window-mean (with macro); without macro LR posts a real
  but small r=+0.20 signal that Φ overfits past. Top-50 capture 7.7%
  (Φ) vs 5.5% (LR) vs 100% (oracle) — confirms the pairs v1 LR
  predictor's 5.4% capture rate is at the feature-space ceiling, not
  the model-class ceiling. FAIL per pre-reg.

**Day-2 (policy training against -Φ) was NOT attempted** — the
pre-reg's exit condition was explicit: day-1 FAIL closes the arc as
`confirmed-null`.

**Open follow-ups** (not pre-registered as next experiments):

1. Per-rebal emission across apps. Requires re-running factor/vol/gate
   walk-forwards with new emission code that captures per-rebal
   (state, action, realized-PnL) triples. ~3-6 hour workstream.
   Would give ~10× more samples per app at the granularity where the
   oracle "knows" which action wins.
2. Richer pair features for pairs. Half-life trajectory across train
   bars, cross-pair rank within window, sector co-membership,
   term-structure of correlation. Tests "feature-space, not
   classifier-class" as the binding constraint.
3. End-to-end decision-structure training (the factor mixture
   pattern). Skip the explicit value function, train the action
   distribution against PnL directly. The only approach in the
   five-app oracle survey that crossed meaningful headroom (~42%
   capture on factor).

See the closing prose in
[`findings/critic-phi-quality-v0`](../findings/critic-phi-quality-v0.md)
("Arc closure" section) for the full diagnostic chain.

Leaderboard rows: 2026-05-15 v0 (`confirmed-null`) + 2026-05-15 v0.1
(`confirmed-null`).

---

## Original pre-registration (preserved for the record)

**Status at time of writing**: pre-registered 2026-05-15, day-1
(Φ-quality) not yet run.

### Hypothesis

A small neural network Φ trained on the (state, action, realized-Sharpe)
triples accumulated across the five-app oracle survey (factor, vol, gate,
pairs, dca) can predict held-out-window Sharpe **better than a per-action
marginal-mean baseline**. If so, Φ becomes a differentiable training signal
for policy networks (training against `-Φ`), unlocking learned action
selection in apps where the cross-app oracle survey already proved
headroom exists but where post-hoc classifiers captured only 0-5% of it
(vol v3.1, pairs v1).

## Why this is the next experiment

The five-app oracle survey closed 2026-05-15 with a consistent pattern:

| App | Oracle headroom | Best real-time selector | Capture |
|---|---:|---|---:|
| factor | +0.11 | learned mixture α=0 | ~42% |
| vol v3 | +2.86 | composite vix-or-disp | 0% |
| gate v0 | +0.32 | (not tested) | — |
| pairs | +1.79 | 7-feature LR | 5.4% |

The one app that crossed meaningful oracle headroom (factor, ~42%) is the
only one where the decision structure was trained end-to-end against PnL,
rather than as a post-hoc classifier on hand-picked features. **Φ is the
generalization of that pattern** — separate the learned-from-history value
model from the per-app policy, so any app can train its policy against the
same Φ-as-loss signal. See the closing diagnostic in
[`pairs-eg-gate-falsified.md`](../findings/pairs-eg-gate-falsified.md)
and [`vol-surface-v3-regime-gated.md`](../findings/vol-surface-v3-regime-gated.md)
for the per-app evidence that small-feature/linear-classifier approaches
fail at this codebase's shape.

## Test design

### Available data (per 2026-05-15 disk audit)

**Window-level** (`(app, window, action) → realized-window-Sharpe`):

| App | Source JSON / NPZ | Triples |
|---|---|---:|
| factor (mixture entropy sweep) | `horizon-mixture-sweep-summary.json` | 6 × 5 = 30 |
| factor (fixed-h baselines) | `horizon-mixture-windows.npz` (`val_fixed_sharpe_h{5,10,20,40,60}`) | 6 × 5 = 30 |
| vol v3.1 composite | `vol-walkforward-v3-1-composite-summary.json` | 5 × 7 = 35 |
| vol v3 regime-gated (lookback sweep) | `vol-walkforward-v3-regime-gated-summary.json` | 5 × 3 = 15 |
| gate v0 | `gate-walkforward-summary.json` (unc / gated / oracle-DD / oracle-day) | 6 × 4 = 24 |
| pairs v0/v1/oracle | `pairs-{walkforward,predictor-walkforward,oracle-walkforward,eg-gate}-summary.json` | 6 × ~6 = 36 |

Window-level total: ~170 unique triples.

**Pair-level for pairs** (`(window, pair_id, pair_features) → realized-pair-Sharpe`)
from `pairs-walkforward-summary.json`: 6 × 50 = 300 triples. Pairs gets the
richer dataset since the v1 LR predictor already proved per-pair features
are discriminating.

### State features (per app)

- `app_id` one-hot (4 dims).
- `vix_level` at val_start (FRED `VIXCLS`).
- `vix_6m_change` (val_start − val_start − 126 trading days).
- `spy_trailing_252d_ret` at val_start.
- `cross_sectional_dispersion` (252d trailing stdev of cross-sectional log-returns) at val_start.
- `val_window_length_days` (varies across apps).

For pairs at the pair level, additionally use the seven v1 features
(`log_train_half_life`, `abs_corr`, `log_eg_pvalue`, `abs_hedge_beta`,
`train_sharpe`, `train_pct_in_trade`, `log_train_n_trades`).

### Action features

Per-app one-hot encoding of the arm/action choice (different action vocab
per app: factor → entropy weight in {0, 0.05, 0.1, 0.2, 0.3} ∪ fixed-h in
{5, 10, 20, 40, 60}; vol → gate type in {vix-126d, disp-126d, ...}; gate
→ {unconditional, gated, oracle-DD, oracle-day}; pairs → {all-pairs,
predictor-thr-0.5, predictor-top-50, oracle-pos, oracle-top-quartile}).

### Model

Tiny MLP, two hidden layers of width 16, ReLU activations, dropout 0.1
on hidden activations. Inputs: concatenated `[state_features, action_features]`.
Output: scalar predicted-Sharpe. Loss: MSE against realized-Sharpe.
L2 regularization on hidden weights at 1e-3. Adam, lr=1e-2,
~200 steps. **tinygrad** runtime (matching codebase convention).

### Cross-validation

**Leave-one-window-out (LOO-by-window)** per app:

1. Hold out walk-forward window `w` from app `a`.
2. Train Φ on **all** remaining triples from **all** apps (cross-app
   pooling — Φ shares hidden weights across apps via the `app_id`
   one-hot input).
3. Predict Sharpe for each action on held-out (`a`, `w`).
4. Aggregate across all held-out (app, window) folds.

This tests both per-app calibration (within-app Spearman across arms on a
held-out window) and cross-app transfer (does training on factor+vol+gate
data improve the held-out-window pairs prediction?).

### Baselines

1. **Global-mean baseline**: predict the global mean of all training-set
   Sharpes (single scalar predictor). Establishes the "no signal at all"
   floor.
2. **Per-app-mean baseline**: predict the per-app mean of training-set
   Sharpes for action's app. Captures app-level effects only.
3. **Per-action-mean baseline**: predict the per-(app, action) training-set
   mean. Captures app + action-level effects but ignores state.
4. **Φ (the model)**: must beat per-action-mean to demonstrate state
   information adds something.

### Metrics (per app, aggregated across LOO folds)

- **Spearman r** between predicted and realized Sharpe (across all
  held-out (window, action) pairs for the app).
- **RMSE** of predictions vs realized Sharpe.
- **RMSE relative to per-action-mean baseline**: `RMSE_Φ / RMSE_baseline`.
  <1.0 = Φ adds signal; ≥1.0 = Φ does not add signal beyond app+action
  marginal means.
- **Within-window argmax-action precision** (auxiliary): does Φ pick the
  actually-best action on the held-out window? Compared against marginal
  argmax.

### Day-1 pre-registered cuts

**Per-app verdict**:
- **App-PASS**: `RMSE_Φ / RMSE_baseline ≤ 0.75` AND `Spearman r ≥ +0.20`.
- **App-MARGINAL**: `RMSE_Φ / RMSE_baseline ≤ 0.90` AND `Spearman r > 0`.
- **App-FAIL**: otherwise.

**Overall day-1 verdict**:
- **STRONG-PASS**: App-PASS in ≥ 3 of 4 apps.
- **PASS**: App-PASS in ≥ 2 of 4 apps OR (App-PASS + App-MARGINAL ≥ 3).
- **MARGINAL**: App-MARGINAL in ≥ 3 of 4 apps (verdict label: `partial-OOS`).
- **FAIL**: otherwise (verdict label: `confirmed-null` on the value-function hypothesis).

### Day-2 (contingent on day-1 PASS or better)

Train a policy network `π(state) → action_distribution` to maximize
`Φ(state, π(state))` via a stochastic-policy gradient. Add a
**CQL-style penalty** keeping `π` near the action-distribution observed
in the training data (concretely: KL(π || empirical_action_marginal) ≤
0.5, enforced via Lagrangian).

Pick the app with the highest Φ-quality lift AND the largest oracle
headroom. Walk-forward eval `π` against the existing best-real-time
selector for that app.

**Day-2 cuts**:
- **PASS**: Δ-Sharpe vs existing best selector ≥ +0.15 AND ≥ 4/6 positive windows.
- **STRONG-PASS**: Δ-Sharpe ≥ +0.30 AND ≥ 5/6 positive windows (closes
  ≥ 25% of the app's oracle headroom).
- **FAIL**: Δ-Sharpe < +0.05 OR ≤ 3/6 positive windows.

## Honest acknowledgements before running

1. **Tiny-data regime**. ~30 triples per app, 6-fold LOO. Even with strict
   regularization, statistical noise on Spearman is large; expect
   confidence intervals around r that span 0.0–0.4 for any single LOO
   fold. The cut at `r ≥ +0.20` is calibrated to be passable but not
   trivially passable.
2. **Off-policy distribution shift**. The actions we have triples for
   were sampled from a small set of hand-built policies (entropy weights,
   binary gate compositions, ...). Φ will be calibrated only near these
   sampled (state, action) points. Day-2's CQL penalty exists exactly to
   keep π from optimizing into action regions Φ has never seen.
3. **Window features are coarse**. The state features above are all
   single-scalar window-level summaries. The granularity at which the
   oracle "knows" which action wins is per-rebal, not per-window — so
   even a perfect window-level Φ can only capture the *window-stratified*
   signal, not the per-rebal signal. If day-1 PASSes only marginally,
   the natural follow-up is to extend the walk-forward scripts to emit
   per-rebal triples (deferred until day-1 lands).

## Exit conditions

- **Day-1 FAIL** → close as `confirmed-null` on the value-function
  hypothesis. Predictor-quality at the window level is not learnable
  from window-aggregate features. Pivots to "extend per-rebal emission
  and retry" become a separate, narrower workstream.
- **Day-1 PASS + Day-2 PASS** → Φ is operationally useful; promote to
  `packages/critic/` (`ss_critic`) and integrate into the most promising
  app as the policy training loss. Verdict label `partial-OOS` until a
  second app shows lift.
- **Day-1 PASS + Day-2 FAIL** → Φ is calibrated for prediction but not
  for policy optimization (distribution-shift gap is the binding
  constraint). Document the asymmetry; the value function survives as a
  diagnostic but does not replace existing selectors. Verdict `partial-OOS`
  on Φ-quality, `confirmed-null` on Φ-as-policy-loss.

## Concept link

If this works, it instantiates the
[strategy-as-dot-product](../notes.md) framing one step further:
*policy* and *value* both become learnable. If it doesn't, the closing
diagnostic should specify whether the failure was feature space, label
noise, or distributional shift — each suggests a different orthogonal
follow-up.
