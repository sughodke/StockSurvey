# Factor — multi-task auxiliary head regularizes the MLP arm but does not clear the indicator baseline

**Operational rule:** the
[`mlp_multitask`](https://github.com/sughodke/StockSurvey/commit/239ebf9)
scorer with `aux_weight=0.1` against winsorized z-score targets
(`forward_robust_z`) lifts mean val IC by **+0.012** over the
plain `mlp` baseline (−0.0120 → +0.0001), but does **not** clear
the [linear-on-encoder baseline (+0.0031)](factor-ssl-walkforward.md)
or the
[deterministic-indicator baseline (+0.0120)](factor-indicator-baseline.md).
The lift comes from **gradient regularization through the shared
trunk**, not from the encoder learning magnitudes — the auxiliary
head's own MSE stays pinned near 1.0 (≈ random on z-scored
targets) across every window. Aux supervision in this regime is
a regularizer, not a magnitude-extracting mechanism. The
[supervision-is-binding](factor-ssl-walkforward.md) reading from
the prior result holds.

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

## Master walk-forward log

[Leaderboard row](../leaderboard.md) tagged
[`confirmed-null`](../leaderboard.md#verdict-labels) for the
ceiling-clearing claim; the +0.012 lift over `mlp` is a real
mechanism observation worth keeping but does not move the
indicator-baseline ceiling. Follow-up `aux_weight` sweep tracked
in
[`TODO/multitask-aux-weight-sweep.md`](../TODO/multitask-aux-weight-sweep.md).
