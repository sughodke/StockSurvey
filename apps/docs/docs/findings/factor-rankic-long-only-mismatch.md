---
tags:
  - factor-narrow
  - confirmed-null
---

# Rank-IC trains a signed signal that long-only top-N can only half-execute

A structural diagnostic on the gap between training metric and
deployment constructor. `pearson_rank_ic` (the factor-app training
objective; also the conceptual frame for several relational scorers)
is sign-symmetric: a head that nails the bottom decile contributes
to IC identically to one that nails the top. Every shipped portfolio
constructor in the repo — `block_sharpe`'s temperature-scaled softmax,
`relational.inference.target_weights`, the live broker path's
`apply_position_cap` — is long-only top-N, which can act on the
positive tail only. **The model is trained to predict in two
directions and deployed to act in one.** This is a candidate
explanation for *part* of the
[passive-EW failure](passive-ew-benchmark.md) that none of our
relational model rows could close.

Verdict: [`confirmed-null`](../leaderboard.md#verdict-labels) —
the cheap test fired and the long-short constructor failed by
both pre-registered cuts (mean val Sharpe **−0.067** vs the
+0.10 floor; **2/6** positive windows vs the 4/6 floor). The
"discarded short signal" hypothesis is falsified. See "Result
(2026-05-10)" below; per the `confirmed-null` next-move rule,
the line of work pivots to a different prediction problem
([`TODO/different-prediction-problem`](../TODO/different-prediction-problem.md))
rather than continuing to tune cross-sectional return forecasting.

## What the metric does

`apps/factor/src/factor/objectives.py:29-61` — per-bar Pearson
correlation between scores and forward log-returns over the masked
liquid universe, mean-aggregated across bars. Lines 51-58:

```python
s_dev = (scores - s_mean.reshape(-1, 1)) * mask
r_dev = (fwd_returns - r_mean.reshape(-1, 1)) * mask
cov = (s_dev * r_dev).sum(axis=1)
denom = (s_var * r_var).maximum(1e-18).sqrt()
per_bar_ic = cov / denom
```

A score of `−5.0` paired with forward return `−0.05` produces the
same `s_dev * r_dev` cell contribution as `+5.0` paired with `+0.05`.
The metric is fully sign-symmetric in the score vector. Per-bar
demeaning (line 52, `s_dev = scores − s_mean`) further means that
the coefficient depends only on cross-sectional dispersion of scores
around the mean — a constant-score predictor (e.g. EW) gets exactly
zero IC.

## What the constructor does

The single-shot inference path in `relational/inference.py` —
spread-gate filter, then renormalize the weight vector to sum to 1
over the surviving names. Whatever sign the upstream scorer produces
on a name is irrelevant once the renormalize step fires: weights
must be ≥ 0 by construction. The `block_sharpe` eval in
`apps/factor/src/factor/objectives.py:89` does the same via
`temperature * softmax(score)` — a softmax over real-valued scores
is non-negative, sums to 1, and reduces to a top-N tilt as
temperature shrinks. There is no path in the deployed code where a
score below the cross-sectional mean produces a *negative* portfolio
weight.

The asymmetry is starkest at the eval boundary: a head that produces
score vector `[+3, +1, −1, −2, +2, −3]` and returns `[+5, +1, −2,
−4, +3, −5]` is rewarded by rank-IC for the *full* score-return
covariance, but a long-only top-N constructor with N=2 would buy
names 1 and 5 (top two scores) and pay full beta on the rest. The
−3 and −2 score predictions — half the head's information — are
discarded.

## Connection to the EW gate

The [passive-EW finding](passive-ew-benchmark.md) showed that
none of the four canonical relational model rows
(Phase-2 analog Ricker, stooq_us_long Morlet, ex-Phase-2 Ricker,
factor-wide-ish — n/a) clear their universe's passive Sharpe. A
long-only top-N portfolio's val Sharpe decomposes (loosely) as:

- **Market-beta component** — proportional to the universe's
  passive Sharpe and the long-only constraint's average tilt.
- **Cross-sectional skill component** — proportional to the
  Grinold IR ≈ IC · √BR.

Long-only top-N forces the first term to be paid in full (you
cannot escape market exposure if you must hold positive weights
summing to 1). The second term has to clear the gap to the
universe's passive Sharpe before any "win" is bookable. On
Phase-2 the passive bar is +1.08 and the model delivered +1.15 —
a gap of +0.07 that is within single-split eval noise. On wider
universes the passive bar drops to +0.68-0.85 *and* the model
delivers worse cross-sectional skill (universe-shift effect),
producing alpha −0.13 to −0.33.

A long-short constructor (e.g. weights = z-score of scores
normalized to `sum(w) = 0`, `sum(|w|) = leverage`) does not pay
the market-beta term. It competes against a zero-Sharpe
benchmark, not against +1.08 / +0.85 / +0.68. **The metric is
already trained for that game; only the deployment is wrong.**
This is why the diagnostic is load-bearing — it points to a
constructor change rather than a different prediction problem
or a wider universe.

## Why the relational scorers are partially exempt

The relational `analog_knn` / `farthest` / `diversified` scorers
build weights from cosine-similarity / distance heuristics that
are inherently non-negative by construction. There is no
"negative-tail" prediction inside them to discard. The
sign-symmetry argument applies most cleanly to `apps/factor`'s
rank-IC head and to the `empirical` / `gmm` / `velocity`
scorers if their underlying score function can take both signs.
The TODO is scoped to factor first because the metric mismatch
is sharpest and the head is the cheapest to re-evaluate.

## Result (2026-05-10)

Implementation: `factor.objectives.long_short_weights` +
`block_sharpe_long_short` (per-bar z-score → clip ±3σ →
re-demean → L1-normalize so `sum(w) = 0` and `sum(|w|) = 1`).
Costs use `commission_frac × L1(Δw)` *without* the 0.5 factor —
for a market-neutral book the L1 of the delta is already the
one-sided turnover (the 0.5 factor in `block_sharpe` exists
because long-only L1(Δw) double-counts under `sum(w) = 1`).
Initial entry from cash pays full leverage. Per-window column
added to `WalkForwardWindow.{train,val}_sharpe_long_short`.

Driver: `apps/factor/scripts/long_short_eval.py`. Same
factor-narrow universe, same 6-window walk-forward, same
linear head, same `n_steps=200 lr=1e-2 wd=1e-3 commission_bps=10`.
Both arms evaluated on the same head trained on rank-IC.

| win | train_ic | val_ic | val Sharpe LO | val Sharpe LS | LS − LO |
|---|---:|---:|---:|---:|---:|
| 0 | +0.1447 | +0.0039 | −0.985 | −0.336 | **+0.649** |
| 1 | +0.0951 | −0.0101 | +0.855 | −0.382 | **−1.237** |
| 2 | +0.1255 | +0.0227 | +0.732 | +0.273 | −0.458 |
| 3 | +0.1245 | +0.0007 | +0.235 | −0.212 | −0.447 |
| 4 | +0.1148 | −0.0088 | +0.418 | −0.212 | −0.631 |
| 5 | +0.1063 | +0.0246 | +0.411 | +0.466 | +0.055 |
| **mean** | **+0.119** | **+0.0055** | **+0.278** | **−0.067** | **−0.345** |

Headline: long-short val Sharpe = **−0.067**, positive windows
= **2/6**, alpha vs long-only = **−0.345**.

| Pre-registered cut | Threshold | Observed | Verdict |
|---|---|---|---|
| mean LS val Sharpe ≥ +0.20 | +0.20 | −0.067 | fail |
| pos-LS-window fraction ≥ 4/6 | ≥ 4 | 2 | fail |
| (alt fail trigger) mean LS < +0.10 *or* ≤ 2/6 positive | either | both | fail |

Falsified.

## Why long-short underperformed

Three observations from the per-window table:

1. **No window where LS materially exceeds LO.** The single
   window where LS lifts (w0, +0.65 Sharpe) is rescuing a
   catastrophic LO (−0.99), not adding new alpha. w5 has +0.06
   alpha, within noise. Every other window has LS strictly
   worse.
2. **Window 1 is the cleanest evidence against the
   discarded-short-signal hypothesis.** LO returned +0.86 on
   that window; if the head's rank-IC had real bottom-tail
   skill, LS would at minimum match it. LS came in at −0.38, a
   1.24-Sharpe destruction. The "edge" LO captured on this
   window was almost entirely market beta of the long-only
   tilt, not cross-sectional skill.
3. **Mean train IC +0.119 with mean val IC +0.0055** is the
   classic overfit signature reported in
   [`factor-indicator-baseline`](factor-indicator-baseline.md):
   the head has substantial in-sample fit but ~2-3% of it
   generalizes. That residual val IC is too small to drive a
   long-short portfolio above costs (10 bps × 2× turnover ≈
   400 bps annualized friction). LO's +0.28 Sharpe was
   overwhelmingly the universe's market-beta tailwind, not
   the +0.005 IC.

This makes the `confirmed-null` verdict structural rather than
implementation-specific: any long-short constructor on a head
with this val-IC magnitude will lose to the friction floor.
Two-tail prediction skill can't help if neither tail has skill
above noise.

## Connection to the EW gate

The
[passive-EW finding](passive-ew-benchmark.md) had already
shown that the long-only path can't clear EW. This result
tightens the diagnosis: the alpha gap isn't *just* a
constructor problem — the underlying head genuinely lacks
cross-sectional dispersion at the +0.005 to +0.012 IC range
the indicator stack hits on the 20-day horizon. Friction
costs eat the signal under either constructor.

The val IC ceiling result was already on the leaderboard
(`factor-indicator-baseline` row, +0.012 ceiling at the same
universe / windowing); this run gives the *constructor-side*
control.

## Implication

`confirmed-null` next-move per the leaderboard's verdict-action
table: stop testing variations of the same lever, find an
orthogonal one. Long-short was the cheapest variation of the
cross-sectional return-forecasting setup; it's now resolved.
The vol-forecast arc (
[`factor-multitask-aux-weight-sweep`](factor-multitask-aux-weight-sweep.md))
already established cross-sectional return prediction is null
at this universe / horizon. The next test should change the
prediction problem itself — pair-spread mean reversion,
drawdown forecasting, or IV-vs-realized — not the constructor
or the friction stack. See
[`TODO/different-prediction-problem`](../TODO/different-prediction-problem.md).

## Master walk-forward log

[2026-05-10 long-short constructor row](../leaderboard.md) —
[`confirmed-null`](../leaderboard.md#verdict-labels).
