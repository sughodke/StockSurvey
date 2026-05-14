# Endogenous-horizon mixture — entropy-weight sweep (verdict gate)

Follow-up to
[`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
([`partial-OOS`](../leaderboard.md#verdict-labels), 2026-05-14): the
mixture-of-horizons head collapses to π ≈ δ(h=60) in 4/6 walk-forward
windows. The 2/6 windows where the model uses a real mixture (w3
2015, w5 2021) beat best-fixed by +0.02 each; the 4/6 where it
collapses tie or lose, with w0 (2005-11) catastrophic at −0.37.

Hypothesis: an entropy regularization term `−α·H(π_t)` on the
mixture loss prevents the collapse-to-h=60 degenerate minimum. If
the hypothesis is right, an α exists that makes all 6 windows look
like w3/w5 — and that arm clears the `+0.10 vs best-fixed` threshold
that the unregularized run missed.

The `entropy_weight` knob is **already wired** through
`objectives.horizon_mixture_loss` and the
`train_scorer_horizon_walkforward` trainer; the only thing this
experiment requires is the sweep + analysis.

## Falsifiable test design

**Sweep**: re-run the Modal entrypoint with `entropy_weight ∈ {0.0,
0.05, 0.1, 0.2, 0.3}` keeping everything else at the canonical
config (`horizons=(5,10,20,40,60)`, `n_steps=200`,
`learning_rate=1e-3`, `weight_decay=1e-3`, `mlp_hidden=32`,
`commission_bps=10`, `factor-narrow` universe, 6-window
walk-forward).

**Pre-registered cuts** (same as the unregularized run, applied
per α):

| # | Null | Threshold |
|---|---|---|
| N1 | π collapse (argmax-bin global share > 0.90) | worst share ≤ 0.90 |
| N2 | beats fixed h_max | endog Sharpe > fixed-h=60 |
| N3 | beats best fixed by ≥ +0.10 | delta ≥ +0.10 |
| N4 | beats random-π | delta ≥ 0 |

**Diagnostic cuts** (per α, not pre-reg gates):

| Cut | Threshold | Why |
|---|---|---|
| Per-window entropy floor | min(H(π_t)) over windows ≥ 0.50 | confirms the model isn't collapsing even on bars where one horizon is locally optimal |
| Mixture windows survive | val Sharpe on w3 + w5 ≥ unregularized run's values | confirms α doesn't kill the configurations that worked |
| 2005-11 (w0) catastrophe | val Sharpe on w0 ≥ −0.10 | confirms α prevents the GFC-run-up confident-h=60 bet |

**Verdict logic**:

- If any α PASSES all four pre-reg nulls: the architecture works
  with regularization. Promote that α to canonical and run a
  rebal-grid sensitivity check (h_min=10 vs 5, K=3 vs 5) before
  declaring `confirmed-OOS`.
- If multiple α PASS N1/N2/N4 but none clears N3 (+0.10 threshold):
  partial-OOS persists; the architecture has real but small
  state-conditional content and the threshold is binding. Document
  and move on.
- If no α clears even N3 by +0.05: closes
  [`confirmed-null`](../leaderboard.md#verdict-labels) on discrete
  mixture-of-horizons-IC. Orthogonal next lever is a different
  *kind* of horizon-emission (continuous horizon head + REINFORCE,
  or explicit regime-classifier → horizon mapping rather than
  end-to-end soft attention).

## Cost

5 α values × ~13 min Modal wall = ~65 min total ≈ $0.35 at T4
prices.

Reuses the existing Modal entrypoint with `--entropy-weight <α>`
flag (already plumbed through). No new code needed — just five
invocations and a summary table.

## Where the result lands

Each α gets a row in
[Leaderboard](../leaderboard.md) under the
`endogenous-horizon mixture (α=…)` experiment label. If any α
clears the threshold, the canonical-α row supersedes the
2026-05-14 unregularized row and the findings page
[`factor-endogenous-horizon-mixture`](../findings/factor-endogenous-horizon-mixture.md)
gets extended with the regularization section.
