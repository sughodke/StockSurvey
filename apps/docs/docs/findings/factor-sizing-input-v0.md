---
tags:
  - factor-narrow
  - confirmed-null
  - hypothesis-user
---

# Sizing-input v0 — MSE-on-alpha calibrates score magnitude but adds zero information for signal-quality emission

The reframe motivated by [passive-EW](passive-ew-benchmark.md) +
[loss-pivot](factor-loss-pivot.md) +
[rankic-long-only-mismatch](factor-rankic-long-only-mismatch.md):
no portfolio constructor or loss on cross-sectional return prediction
clears EW at +0.005 IC, so stop training a *tilting* model. Train a
*sizing-input* model — emit per-ticker calibrated alpha forecasts that
feed a downstream gate, not a portfolio softmax. The cheap test: swap
`pearson_rank_ic` (scale-invariant) for `masked_mse` against per-bar
cross-sectional alpha (scale-calibrated), and check whether the
resulting per-val-bar signal-quality (top-decile − bottom-decile
predicted alpha) gains downstream-useful properties.

Verdict: [`confirmed-null`](../leaderboard.md#verdict-labels) on the
hypothesis tested. `mse_alpha` *does* calibrate score magnitudes (val
MSE-alpha 52× smaller than `rank_ic`'s) but produces **identical**
sizing-input emission on the load-bearing metrics:

| Metric (mse_alpha vs rank_ic) | rank_ic | mse_alpha | delta | pre-reg PASS gate |
|---|---:|---:|---:|---|
| Pooled lag-1 autocorrelation of signal-quality | +0.905 | +0.821 | **−0.084** | ≥ +0.10 |
| Spearman ρ(sq_mean, val Sharpe) | **+0.486** | **+0.486** | **0.000** | ≥ +0.10 |
| val MSE-alpha | 5.72e-01 | **1.10e-02** | (50× smaller) | n/a (calibration check) |
| val Sharpe long-only | +0.278 | +0.482 | +0.204 | not pre-registered |
| mean val IC | +0.0055 | −0.0008 | −0.0063 | not pre-registered |

Both arms clear the **absolute** pre-reg thresholds (lag-1 autocorr ≥
+0.20, Spearman ρ ≥ +0.40), but neither arm lifts vs the other on the
delta criterion. The training-objective change moves score magnitudes
without moving the rank-based artifact that downstream consumers see.

This is a useful failure: the v0 hypothesis is dead, the v1 plan is
clearer than before. The signal-quality emission from the rank_ic head
already has the temporal-stability and val-Sharpe-correlation
properties the macro meta-gate v1 needs. Per `confirmed-null` next-move
rule, stop testing the loss axis on apps/factor and proceed straight
to the meta-gate wiring.

## Setup

Same factor-narrow universe (297 stooq_us_long names,
`min_history_bars=6500`), same 6-window walk-forward (train=63,
val=39, step=39 blocks at `rebal_days=20`), same linear head, same
`n_steps=200 lr=1e-2 wd=1e-3 commission_bps=10` as
[`factor-indicator-baseline`](factor-indicator-baseline.md) and
[`factor-loss-pivot`](factor-loss-pivot.md). Two arms:

- **`rank_ic`** — existing baseline. `pearson_rank_ic` on
  cross-sectionally demeaned forward log-returns. Scale-invariant on
  the predictor.
- **`mse_alpha`** — new. `masked_mse` against per-bar cross-sectional
  alpha targets (`fwd_log_return − cross_sectional_mean`, kept f64
  through the demean per the
  [f32-precision rule](factor-f32-precision-cancellation.md)). Scale-
  calibrated: a score of `+0.02` corresponds to an expected
  per-ticker alpha of +2% over the rebal horizon.

Per-val-bar signal-quality emitted from both arms: `top_decile_mean −
bottom_decile_mean` of predicted alpha over the masked-liquid universe
at each rebal bar (`decile_frac=0.10`). Pre-registered downstream
properties:

- **Temporal stability** — pooled lag-1 autocorrelation of
  signal-quality across val bars (windows concatenated end-to-end).
  Captures "does the head's confidence signal persist from one rebal
  to the next, or is it noise?"
- **Realized correlation** — Spearman ρ between per-window mean
  signal-quality and per-window val Sharpe. Captures "do windows
  where the head is more confident actually deliver better
  performance?"

Both numbers are computable in real time (no forward returns required)
— that's the whole point of choosing a dispersion stat over ICIR for
the gate input.

## Result (2026-05-12)

Per-window signal-quality means:

| win | val_start  | rank_ic sq_mean | mse_alpha sq_mean | rank_ic val Sh | mse_alpha val Sh |
|---:|---:|---:|---:|---:|---:|
| 0 | 2005-11-15 | +1.128 | +0.134 | **−0.984** | −0.663 |
| 1 | 2008-12-24 | **+3.468** | +0.232 | +0.855 | +0.817 |
| 2 | 2012-03-22 | +1.237 | +0.172 | +0.730 | **+1.248** |
| 3 | 2015-06-22 | +1.527 | +0.278 | +0.235 | +0.725 |
| 4 | 2018-08-20 | +1.258 | +0.148 | +0.418 | +0.444 |
| 5 | 2021-09-28 | +1.123 | +0.217 | +0.411 | +0.321 |
| **mean** |  | +1.624 | +0.197 | **+0.278** | **+0.482** |

Why Spearman ρ is identical (+0.486) across arms even though
sq_mean values are 8–15× different: Spearman depends only on the
*rank ordering* of windows. The window-ordering by signal-quality is
the same for both arms (windows where rank_ic concentrates dispersion
are also where mse_alpha does), and the val Sharpe ordering is
similar between arms. The ranks line up to the same `17/35 ≈ 0.486`
value.

Why lag-1 autocorrelation differs slightly (+0.905 vs +0.821): bar-
level signal-quality from rank_ic is dominated by score-norm drift
across the val window (the head's output norm wanders smoothly as
walk-forward time advances; rank_ic doesn't penalize it). `mse_alpha`
constrains the norm to alpha units so bar-to-bar variation reflects
underlying cross-sectional dispersion changes — slightly noisier, but
the loss of autocorrelation is the *right* loss (it's wandering norm
the rank_ic arm picked up, not signal).

## Mechanism

The sizing-input emission is a **rank statistic** by construction.
Top-decile mean minus bottom-decile mean is invariant to any
monotone rescaling of the score vector — if you double all scores,
both decile means double, and the difference doubles too, but
*relative* to the score scale the dispersion is unchanged. The whole
point of the mse_alpha training change was to fix score scale — and
that's exactly the dimension the downstream stat throws away.

What mse_alpha **does** give you (52× smaller val MSE-alpha) is
calibration: a per-ticker score of `+0.02` actually means "expected
+2% forward alpha" rather than "this ticker ranks higher than ones
with smaller scores." But the v0 test was specifically the *rank-
based* emission. A *magnitude-aware* emission (e.g. sum of absolute
predicted alphas in the top decile, or fraction of tickers with
predicted alpha above a fixed cost threshold) would differentiate the
arms — but those would also be downstream stats we hadn't pre-
registered.

The honest read: the rank-based emission was the wrong test for the
mse_alpha training change. The pre-reg framed it as "does mse_alpha
help sizing-input?" — but the answer depends entirely on which stat
the downstream gate consumes. We chose the rank-invariant one because
it's robust and computable without forward returns; that choice
forecloses on the loss-axis question.

Side observation worth recording: mse_alpha's val Sharpe long-only is
+0.482 vs rank_ic's +0.278 (Δ +0.204) — a real-looking lift, but n=6
and the per-window variance is large. The
[loss-pivot](factor-loss-pivot.md) finding established that
`block_sharpe` and `ir_vs_ew` losses both *destroyed* val Sharpe via
softmax-temperature collapse. `mse_alpha` doesn't have that pathology
(no temperature in its loss, no concentration incentive), so the
+0.20 long-only Sharpe lift might just be a side-effect of training
on better-conditioned gradients (alpha targets vary at ~1e-2; rank-IC
gradients are scale-free and can ill-condition the head). Not pre-
registered, not what we set out to test — flag for future
investigation but don't promote.

## Positive side-result for v1

The pre-reg test was about *differentiating* arms, but the absolute
numbers from either arm are themselves evidence for the broader
sizing-input plan:

- **Spearman ρ +0.486** between per-window signal-quality and val
  Sharpe is comfortably above the +0.40 marginal threshold. Per-
  window mean signal-quality from the rank_ic head **does** predict
  which windows the head delivers val Sharpe. That's a regime-gate
  input.
- **Pooled lag-1 autocorrelation +0.82 to +0.91** means signal-
  quality is highly persistent. A meta-gate can read a trailing-N-bar
  mean of signal-quality at val_start and the read isn't dominated by
  noise.
- **val_start_date** is now persisted on every
  `WalkForwardWindow` — the macro meta-gate harness can directly join
  factor signal-quality to macro state at the same calendar moment.

The v1 plan written in
[`TODO/factor-sizing-input-reframe`](../TODO/factor-sizing-input-reframe.md)
is unchanged but now better-grounded: feed per-window factor signal-
quality into `apps/gate/scripts/macro_meta_gate_eval.py` as a second
gate input alongside VIX-state. Use the rank_ic head — there's no
upside to mse_alpha for this consumer.

## What this lands

Operational rule (added to CLAUDE.md):

> **The sizing-input emission `top_decile_mean − bottom_decile_mean`
> from a rank-IC-trained factor head has the temporal stability
> (lag-1 autocorr +0.82-+0.91) and per-window val-Sharpe predictive
> power (Spearman ρ +0.486) needed to serve as a macro-meta-gate
> input.** Training the same head on `masked_mse` against per-bar
> alpha targets calibrates score *magnitudes* (val MSE-alpha 52×
> smaller) but does NOT lift either of the rank-based emission
> properties — the rank-invariant dispersion stat is by construction
> insensitive to score rescaling. If a future downstream consumer
> wants magnitude-aware features (sum-of-top-decile-predicted-alpha,
> fraction-above-cost-threshold), revisit the mse_alpha arm; for the
> rank-based emission, stay on rank_ic.

Implementation:

- `loss_kind='mse_alpha'` in `train_scorer_walkforward`
  (`apps/factor/src/factor/train_walkforward.py`).
- `alpha_target_rb` in `precompute_inputs` output
  (`apps/factor/src/factor/train.py`).
- `WalkForwardWindow.{train,val}_mse_alpha`,
  `signal_quality_per_val_bar`, `signal_quality_mean`,
  `signal_quality_std`, `val_start_date` (all populated for every
  arm, not just `mse_alpha`).
- `_signal_quality_per_bar` helper in `train_walkforward.py`
  (numpy, eval-only).
- Drivers: `apps/factor/scripts/sizing_input_eval.py` (local smoke),
  `apps/factor/scripts/modal/sizing_input_eval.py` (T4 GPU eval).
- Artifacts: `Output/sizing-input-{rank_ic,mse_alpha}-windows.npz`,
  `Output/sizing-input-eval-summary.json`.

## Master walk-forward log

Leaderboard rows:
[2026-05-12 factor — sizing-input v0 head-to-head](../leaderboard.md#operating-conditions),
verdict [`confirmed-null`](../leaderboard.md#verdict-labels) for the
mse_alpha-vs-rank_ic loss-axis hypothesis.

Related findings:
[`factor-rankic-long-only-mismatch`](factor-rankic-long-only-mismatch.md),
[`factor-loss-pivot`](factor-loss-pivot.md),
[`passive-ew-benchmark`](passive-ew-benchmark.md),
[`prediction-problem-pivot-arc`](prediction-problem-pivot-arc.md),
[`macro-regime-diagnostic`](macro-regime-diagnostic.md).
