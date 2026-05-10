---
tags:
  - factor-narrow
  - partial-OOS
  - confirmed-null
---

# Factor — multi-task auxiliary head regularizes the MLP arm but does not clear the indicator baseline

**Operational rule (post-sweep, 2026-05-10):** the
[`mlp_multitask`](https://github.com/sughodke/StockSurvey/commit/239ebf9)
scorer's auxiliary head against cross-sectional winsorized z-score
targets (`forward_robust_z`) **does not extract magnitude
information at any tested gain**. At `aux_weight=0.1` the aux head
doesn't train (val MSE ≈ 1.0) but the gradient-noise side-effect
regularizes the trunk for a +0.012 val IC lift over `mlp`. At
`aux_weight=1.0` the aux head trains (train MSE 0.78) but val MSE
goes **above** 1.0 (1.16) — predictions are *anti-correlated* with
val magnitudes despite fitting train, dragging primary val IC to
−0.0084. At `aux_weight=10.0` the trunk fully collapses on 3/6
windows (output → 0). The cross-sectional winsorized z-score of
forward 20-day log returns is **regime-non-stationary** at this
universe scale; magnitude leaders flip between train and val
windows. The aux=0.1 lift is a regularization artifact, not a
new information channel — and pushing for the information channel
makes things worse. None of the three multitask arms clear the
[linear-on-encoder baseline (+0.0031)](factor-ssl-walkforward.md)
or the
[deterministic-indicator baseline (+0.0120)](factor-indicator-baseline.md).
The [supervision-is-binding](factor-ssl-walkforward.md) reading
from the prior result holds, and now generalizes: the binding
constraint isn't just the supervision target, it's the
*non-stationarity* of cross-sectional return structure at H=20.

## Setup

- Backbone: same `Output/cwtonly-AAPL+294tickers-h631e9d47-rsi+macd+vol+cci-cnn-nogit.npz`
  as the [supervised-`cnn` walkforward](factor-ssl-walkforward.md),
  frozen. K=96, F=105, hidden=64, hidden_flat=5632.
- Universe / dates / windowing identical to the walkforward
  finding: 297 stooq_us_long tickers, `2000-01-03 → 2026-04-01`,
  6 rolling windows of 63 train / 39 val / 39 step blocks at
  `rebal_days=20`.
- Three arms (each with fresh AdamW, frozen backbone, head only):
    - `linear` — 5632 → 1, baseline.
    - `mlp` — 5632 → hidden=64 → 1, single hidden layer, baseline.
    - `mlp_multitask` — shared trunk 5632 → 64, two parallel
      `(64, 1)` projections (`Wp` primary, `Wa` aux). Loss is
      `−pearson_rank_ic(s_p, fwd_log_ret) + 0.1 · masked_mse(s_a, robust_z)`,
      where `robust_z = forward_robust_z(prices, winsor=(0.01, 0.99))` — cross-sectionally
      winsorized + z-scored forward log returns.
- All three arms: 200 AdamW steps at `lr=1e-2`, `weight_decay=1e-3`.
- f64 forward log returns
  ([`9209fa9`](https://github.com/sughodke/StockSurvey/commit/9209fa9))
  — without this fix the linear baseline would have read +0.0005 instead of
  +0.0031, distorting the multitask comparison; see
  [`factor-f32-precision-cancellation`](factor-f32-precision-cancellation.md).
- Modal `cpu=4, gpu=T4, memory=192GB`, wall ~22 min for the 3-arm
  rerun (linear 63s, mlp 75s, multitask 1208s — the trunk + dual
  loss is ~16× more expensive per window than the baselines).

## Walk-forward result (2026-05-10)

Per-window val IC:

| Window | linear   | mlp      | mlp_multitask | mt − mlp Δ | aux MSE |
|--------|----------|----------|---------------|------------|---------|
| 0      | −0.0000  | −0.0221  | **−0.0105**   | +0.0116    | 1.040   |
| 1      | +0.0012  | −0.0248  | **+0.0072**   | +0.0320    | 1.089   |
| 2      | +0.0027  | −0.0352  | **−0.0159**   | +0.0193    | 1.001   |
| 3      | −0.0022  | +0.0101  | **+0.0308**   | +0.0207    | 1.061   |
| 4      | +0.0105  | −0.0120  | −0.0120       |  0.0000    | 1.202   |
| 5      | +0.0064  | +0.0122  | +0.0013       | −0.0109    | 1.033   |

Train IC per window (mlp_multitask vs mlp baseline):

| Window | mlp train_ic | mt train_ic | Δ      |
|--------|--------------|-------------|--------|
| 0      | +0.765       | +0.783      | +0.018 |
| 1      | +0.747       | +0.757      | +0.010 |
| 2      | +0.827       | +0.693      | −0.134 |
| 3      | +0.894       | +0.793      | −0.101 |
| 4      | +0.835       | +0.791      | −0.044 |
| 5      | +0.884       | +0.756      | −0.128 |

Aggregates:

| Head             | mean val IC | median val IC | pos-val frac | mean train IC |
|------------------|-------------|---------------|--------------|---------------|
| linear           | **+0.0031** | +0.0020       | 4/6 (0.67)   | +0.539        |
| mlp              | **−0.0120** | −0.0171       | 2/6 (0.33)   | +0.825        |
| mlp_multitask    | **+0.0001** | −0.0046       | 3/6 (0.50)   | +0.762        |
| [indicator baseline (linear)](factor-indicator-baseline.md) | **+0.0120** | +0.0168 | 5/6 (0.83) | — |

## Three readings of the result

**1. The +0.012 lift over mlp is real and consistent.** 5/6 windows
benefit; only window 5 regresses (mt +0.0013 vs mlp +0.0122).
Train IC drops from mlp's 0.825 average to multitask's 0.762
average — a clean regularization signature. The mlp arm overfits
heavily (train IC +0.83, val IC −0.012); the multi-task aux
gradient biases the trunk toward features that don't memorize
train-cell noise as aggressively, and val IC moves toward zero.

**2. Multitask still loses to linear-on-encoder.** Encoder + linear
head: +0.0031. Encoder + mlp + multitask aux: +0.0001. The
additional trunk capacity, even regularized, isn't worth its cost
in this regime. The simplest head that consumes the encoder
latent is still the best head. This is consistent with the
prior finding's reading: the encoder's information content is
captured by a linear projection of the latent; the MLP capacity
fits training-set artefacts that don't generalize, and aux
regularization only damps that pathology partway.

**3. The encoder still loses to the indicator baseline.** Whether
you read off the linear arm (+0.0031) or the multitask arm
(+0.0001), the encoder + head pipeline lands far below the
+0.0120 deterministic-indicator baseline. The multitask
mechanism — adding a magnitude-aware auxiliary signal that the
shared trunk can route to features the rank-IC objective alone
would discard — does not change this verdict.

## Why the aux head didn't learn

Aux MSE ≈ 1.0 across every window means the aux output is
~uncorrelated with the z-scored target. The aux head's own
gradient is `aux_weight · ∂(MSE)/∂s_a = 0.1 · (...)` — 10× smaller
than the primary's `∂(IC)/∂s_p`. With 200 AdamW steps at lr=1e-2,
the primary head dominates the joint optimization; the aux head
barely moves. The trunk does see *some* aux gradient flowing back
through `Wa`, and that's where the +0.012 regularization effect
came from — but the aux output itself never reached useful
predictive quality on its own targets.

This is **not** the mechanism originally hypothesized
(["the aux objective forces the trunk to retain magnitude
information that pure rank-IC discards"](factor-ssl-walkforward.md#outstanding-questions)).
What we got instead: trunk regularization via a near-randomly-
initialized auxiliary projection. The two mechanisms are
distinguishable empirically — the magnitude-extraction story
predicts a low aux MSE; the regularization story predicts an aux
MSE near random, with a measurable train-IC drop. The data
matches the regularization story.

## Implications

- **Aux supervision is a regularizer at small `aux_weight`, not a
  signal lift.** Use it the way you'd use weight decay or dropout
  — when the head is over-parameterized for its training data.
  Don't expect a new information channel.
- **The supervision-is-binding reading
  ([factor-ssl-walkforward](factor-ssl-walkforward.md)) holds.**
  Two orthogonal mechanisms — a wider input bundle (polar Morlet
  vs Ricker) and a magnitude-aware aux objective — both fail to
  lift the encoder over the linear-on-encoder baseline. The
  binding constraint is on the supervision side, not the encoder
  capacity side.
- **Stronger `aux_weight` is the natural follow-up.** At
  `aux_weight=0.1` the aux head doesn't learn its target. At
  `aux_weight=1.0` or `10.0`, the aux gradient becomes
  competitive with the primary, and we can disambiguate two
  regimes:
    - aux head learns its target *and* primary IC continues to
      lift → the magnitude-extraction mechanism works at higher
      gain;
    - aux head learns its target *but* primary IC degrades →
      the two objectives compete; the aux objective is genuinely
      orthogonal to return prediction;
    - aux head still doesn't learn → the trunk capacity is too
      small for both tasks; bump `mlp_hidden`.

## `aux_weight` sweep (2026-05-10)

Two arms refired against the same backbone / universe / windowing,
varying only `aux_weight ∈ {1.0, 10.0}`. The pre-registered
decision rule (3 branches) anticipated train-or-val MSE behavior
that didn't materialize — a fourth branch was needed. The full
falsification retrospective is in
[`factor-multitask-aux-weight-sweep`](factor-multitask-aux-weight-sweep.md);
this section keeps the per-window data inline.

### Per-window data across the full sweep

`tr_ic` is rank-IC on train; `vl_ic` is rank-IC on val; `tr_aux`
and `vl_aux` are aux MSE against winsorized z-score targets (1.0
is "predict the mean"; >1.0 means *worse than predicting the mean*).

| aux_weight | window | tr_ic   | vl_ic   | tr_aux | vl_aux |
|-----------:|-------:|--------:|--------:|-------:|-------:|
| 1.0        | 0      | +0.5276 | −0.0027 |  0.953 |  1.010 |
| 1.0        | 1      | +0.6309 | −0.0208 |  0.871 |  1.132 |
| 1.0        | 2      | +0.5184 | −0.0092 |  0.786 |  1.092 |
| 1.0        | 3      | +0.7393 | +0.0003 |  0.611 |  1.307 |
| 1.0        | 4      | +0.6425 | −0.0056 |  0.764 |  1.202 |
| 1.0        | 5      | +0.6513 | −0.0123 |  0.684 |  1.211 |
| 10.0       | 0      | +0.0000 | −0.0000 |  1.000 |  1.000 |
| 10.0       | 1      | +0.3938 | +0.0002 |  0.888 |  1.062 |
| 10.0       | 2      | +0.0000 | −0.0000 |  1.000 |  1.000 |
| 10.0       | 3      | +0.1851 | +0.0254 |  0.981 |  1.009 |
| 10.0       | 4      | +0.2592 | +0.0034 |  0.954 |  1.035 |
| 10.0       | 5      | +0.0000 | −0.0000 |  1.000 |  1.000 |

### Aggregates across the full sweep

| aux_weight | mean tr_ic | mean vl_ic | mean tr_aux | mean vl_aux | pos-vl frac |
|-----------:|-----------:|-----------:|------------:|------------:|------------:|
| 0.1        |     +0.762 |    +0.0001 |        ~1.0 |        ~1.0 | 3/6 (0.50)  |
| 1.0        |     +0.618 | **−0.0084**|   **0.778** |   **1.159** | 1/6 (0.17)  |
| 10.0       |     +0.140 |    +0.0048 |       0.971 |       1.018 | 3/6 (0.50)* |

\* aux=10 pos-val-IC frac is misleading: 3 windows hit *exact*
tr_ic = 0.000 (trunk collapsed); only 3 windows actually trained
and 1 of those (window 3) carries the entire +0.0048 mean.

### Reading the sweep — the fourth branch

The pre-registered decision rule had three branches:

1. **Aux MSE drops + primary lifts** → magnitude extraction works.
2. **Aux MSE drops + primary degrades** → tasks compete; aux is orthogonal.
3. **Aux MSE stays at ~1.0** → trunk capacity insufficient.

What we got at `aux_weight=1.0` is none of these:

> **Train aux MSE drops to 0.78. Val aux MSE rises to 1.16.**
> Primary val IC degrades to −0.0084.

The aux head *does* learn — train MSE drops cleanly below 1.0 in
every window (0.61–0.95). But the function it learns reverses sign
on val: predictions are anti-correlated with val cross-sectional
magnitudes, so val MSE *exceeds* the predict-the-mean baseline of
1.0 in every window (1.01–1.31). This is regime non-stationarity:
**the cross-sectional ranking of which tickers had large vs small
forward 20-day moves on train does not persist into the val
window**. The mechanism we hoped to exercise (a magnitude target
the rank-IC objective discards) doesn't survive its own
out-of-sample test, much less help the primary.

At `aux_weight=10.0` the failure is louder. The aux gradient
becomes large enough relative to the primary that the joint
optimization sometimes resolves to the trivial `(s_p, s_a) = (0, 0)`
solution — output identically zero, train and val IC both 0.000,
both aux MSE pinned at exactly 1.0. Three of six windows fall into
this collapse. The other three windows show the same pattern as
aux=1.0 in miniature: train aux MSE drops below 1.0, val aux MSE
rises above 1.0. The headline +0.0048 mean val IC is window 3's
+0.0254 lifting two near-zero windows; it's not a real lift.

### What the sweep falsifies

- **The "trunk capacity insufficient" hypothesis** is wrong. At
  aux=1.0 the trunk has no trouble fitting the aux task on train
  (MSE 0.78) — the failure is on val. Bumping `mlp_hidden` would
  let the trunk overfit aux even harder; it wouldn't fix
  non-stationarity.
- **The "tasks are orthogonal" hypothesis** is too weak. Orthogonal
  would mean val aux MSE stays at ~1.0 (random) while primary
  degrades. We see val aux MSE *above* 1.0 — not orthogonal,
  *anti-correlated* between train and val. That's a stronger
  statement: the train aux signal predicts val anti-magnitude.
- **The aux=0.1 regularization reading is correct** but the
  mechanism is now clearer: aux=0.1 helps because the aux head
  *fails to train enough to overfit*. Any aux gain large enough to
  actually train the aux head also large enough to drag primary IC
  down via the shared trunk's exposure to overfit aux gradients.

### Implication for further factor work

Don't run more aux-head variants on this universe / horizon. The
aux objective is dead — not because the implementation is broken,
but because there's no out-of-sample cross-sectional magnitude
signal at H=20 on stooq_us_long for the aux head to extract. The
natural next experiment is a different prediction problem
([`TODO/different-prediction-problem`](../TODO/different-prediction-problem.md))
or a different horizon
([`TODO/rebal-days-sweep`](../TODO/rebal-days-sweep.md)), not more
multitask arms — see the
[sweep finding](factor-multitask-aux-weight-sweep.md) for the
full retrospective.

## Master walk-forward log

Three [leaderboard rows](../leaderboard.md) for this experiment:
- `aux_weight=0.1` (2026-05-10) — `partial-OOS` for `mt > mlp`,
  `confirmed-null` for clearing the indicator ceiling.
- `aux_weight=1.0` (2026-05-10) — `confirmed-null` for
  magnitude-extraction; supersedes the regularizer-only reading.
- `aux_weight=10.0` (2026-05-10) — `confirmed-null` (trunk
  collapse contaminates surface-level lift).

Sweep retrospective:
[`factor-multitask-aux-weight-sweep`](factor-multitask-aux-weight-sweep.md).
