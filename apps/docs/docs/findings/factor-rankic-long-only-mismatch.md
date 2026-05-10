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

Verdict: [`diagnostic`](../leaderboard.md#verdict-labels). No
run yet — the falsifiable test is below.

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

## Falsifiable next experiment

Replace the long-only softmax-top-N constructor with a
market-neutral long-short one for the existing factor-app rank-IC
heads. Keep training unchanged — the heads stay frozen for v1.
The test design lives in
[`TODO/long-short-constructor.md`](../TODO/long-short-constructor.md).

Pre-registered pass / fail cuts (per the diagnostic verdict's
"turn it into a falsifiable hypothesis" rule):

- **Pass.** Long-short val Sharpe on the factor-narrow walk-
  forward (linear head, the existing 6-window setup) clears
  +0.20 with non-negative alpha *over each window's zero-
  Sharpe market-neutral baseline*. Implication: the head had
  short-side skill and the constructor was discarding it.
  Next move is to retrain head with a Sharpe-aligned loss
  that knows about the long-short constructor.
- **Fail.** Long-short val Sharpe is below +0.10 or has the
  same +/-3-window split as long-only. Implication: the
  head genuinely lacks cross-sectional dispersion at this
  scale, and the deployment-mismatch hypothesis is falsified.
  Next move is to pivot to a different prediction problem
  ([different-prediction-problem.md](../TODO/different-prediction-problem.md))
  rather than continue to tune cross-sectional return
  forecasting.

The cost of running the test is one walk-forward pass on a
checkpoint we already have — no retraining, no new universe
build. ~80 LoC for the constructor + a new column in the
walk-forward summary.

## Master walk-forward log

This finding will land a leaderboard row when the long-short
test fires. Until then, the
[passive-EW benchmark](../leaderboard.md#verdict-labels) rows
remain the operative `confirmed-null` reading on the long-only
deployment side.
