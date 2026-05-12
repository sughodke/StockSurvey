---
tags:
  - factor-narrow
  - meta-gate
  - confirmed-null
  - hypothesis-user
---

# Sizing-input v1 — factor signal-quality as a meta-gate input is dominated by VIX at 6-window resolution

The v1 deployment-layer test following
[`factor-sizing-input-v0`](factor-sizing-input-v0.md). v0 confirmed the
rank_ic head's per-val-bar signal-quality emission has the required
downstream properties in isolation (pooled lag-1 autocorr +0.91,
Spearman ρ vs val Sharpe +0.486). v1 asks: when joined retroactively
to the macro-meta-gate harness as a second deployment-time gate input
alongside VIX state, does it add lift?

Verdict: [`confirmed-null`](../leaderboard.md#verdict-labels) on the
incremental-lift hypothesis. Across all three factor-derived arms
(factor-only, VIX AND factor, VIX OR factor), the best pooled
per-app-z-scored lift is **+0.106**, below the VIX-only baseline's
**+0.215** by 0.109. The factor-only arm goes outright negative
(**−0.143**). VIX dominates factor signal-quality as a meta-gate
input at the temporal resolution this v0 walk-forward produces.

## Setup

Retroactive meta-gate eval across the 17 prediction-problem-pivot
windows (gate × 6 + pairs × 6 + vol × 5), reusing the harness from
[`macro-regime-diagnostic`](macro-regime-diagnostic.md) v1b. For each
pivot val_start, two gate inputs are joined:

- **VIX state** — spot VIX vs trailing 1y rolling median at val_start.
  `high` if spot ≥ median, else `low`. (Existing v1b gate.)
- **Factor state** — most recent factor walk-forward `val_start_date`
  ≤ pivot val_start; use that factor window's `signal_quality_mean`
  from `Output/sizing-input-rank_ic-windows.npz`. Compare against the
  median over the 6 factor windows (`1.248`). `high` if ≥ median,
  else `low`.

The factor-state lookup uses **most-recent-OOS** lookup deliberately
— the factor walk-forward emits each window's signal-quality only
after that window's val period completes, so at pivot deployment time
the available factor read is the latest factor window's val-period
mean. No look-ahead.

Four gated arms vs the no-gate baseline:

1. **VIX-only** — existing v1b baseline.
2. **factor-only** — deploy iff factor state is `high`.
3. **VIX AND factor** — deploy iff both states are `high`.
4. **VIX OR factor** — deploy iff at least one state is `high`.

Pre-registered cuts (from
[`TODO/factor-sizing-input-reframe`](../TODO/factor-sizing-input-reframe.md)):

- **PASS** — any factor arm delivers pooled per-app-z-scored lift ≥
  +0.30 (≈ +0.10 absolute over VIX-only +0.215).
- **FAIL** — best factor arm lift is below VIX-only by ≥ 0.05.
- **INCONCLUSIVE** — between.

## Result (2026-05-12)

| Arm                  | raw mean alpha | gated mean | absolute lift | n deploy |
|----------------------|---------------:|-----------:|--------------:|---------:|
| raw (no gate)        | +0.085         | +0.085     |     +0.000    |    17    |
| VIX-only             | +0.085         | +0.075     |     −0.010    |     5    |
| factor-only          | +0.085         | +0.043     |     −0.042    |     8    |
| VIX AND factor       | +0.085         | −0.005     |     −0.089    |     2    |
| VIX OR factor        | +0.085         | +0.122     |     +0.037    |    11    |

Pooled per-app-z-scored:

| Arm | z-score lift |
|---|---:|
| VIX-only (baseline) | **+0.215** |
| factor-only | **−0.143** |
| VIX AND factor | −0.034 |
| VIX OR factor | +0.106 |

Best factor arm = `VIX OR factor` at +0.106. That's 0.109 below
VIX-only's +0.215 → **FAIL** by the pre-registered cut.

## Mechanism — temporal lag dominates

The 6-window factor walk-forward produces signal-quality readings
*lagged* relative to pivot-arc val_starts. Each factor window's val
period covers ~2.1 years (39 rebal bars × 20 days). The
signal-quality available at pivot val_start is the factor window
that already *finished* its val period — typically 0–3 years stale.

Two clean inversions in the per-window data:

- **2008 GFC pivots** (gate w1, pairs w1, val_start 2008-02):
  factor sq = **+1.128** (low — read from 2005-11 → 2007-09 calm
  pre-GFC). Factor gate says SUSPEND. VIX gate says DEPLOY (VIX
  25 vs median 18.6). The pivot alphas are **+0.321 (gate)** and
  **+0.870 (pairs)** — among the largest in the dataset. VIX correctly
  catches them, factor incorrectly suspends.

- **2011-post-GFC pivots** (gate w2, pairs w2, val_start 2011-03):
  factor sq = **+3.468** (high — read from 2008-12 → 2010-11 GFC
  dispersion). Factor gate says DEPLOY. VIX gate says SUSPEND (VIX 20
  vs median 21). Factor catches **pairs w2 +0.593** correctly, but
  also catches gate w2 (+0.046, marginal). Net win for factor here,
  but the win at 2011 doesn't compensate for the loss at 2008.

The factor reading reflects the *previous* regime's dispersion, not
the current one. VIX, being a real-time market price of forward
options vol, has none of this lag.

This is the same temporal-lag failure mode that killed
[gate-v1a](macro-regime-diagnostic.md#v1-results-2026-05-11) — adding
macro features to a within-app predictor stack failed because macro
distributions are non-stationary across train/val regimes. Here, the
factor head's signal-quality is a function of the *training-window*
regime, not the deployment-time regime; deployments cross regime
boundaries the head trained on, and the factor read is stuck on the
training side.

## Why this isn't fixed by mse_alpha or a different head

Both arms in
[`factor-sizing-input-v0`](factor-sizing-input-v0.md) (`rank_ic` and
`mse_alpha`) produced the same Spearman ρ +0.486 between per-window
sq_mean and val Sharpe. The temporal lag has nothing to do with the
training loss; it's a property of the *walk-forward windowing*. The
6-window setup gives 6 fixed signal-quality readings spaced ~3 years
apart. No re-training-objective change moves the calendar grid.

To remove the lag, you'd need to either:

1. **Finer walk-forward grain** — train on a rolling 1-year window
   with a 1-month step, emitting 200+ signal-quality readings instead
   of 6. Costly: each window is a full Adam pass over a 297-ticker ×
   1y panel; 200 windows ≈ 1.5h on T4.
2. **Expanding-window training with per-bar emission** — train once
   on data up to bar `t`, emit signal-quality at `t`, repeat at every
   rebal. Equivalent to (1) without retraining-from-scratch overhead
   if the head is small enough.
3. **Use the per-val-bar signal-quality time series instead of the
   per-window mean** — we already emit `signal_quality_per_val_bar`
   shape `(6, 39)`. At each pivot val_start we could look up the most
   recent factor *bar* (not window) signal-quality. The lag is
   bounded by `rebal_days=20` instead of `val_window_blocks × rebal_days`
   = 780 days. Worth trying in v2 before doing (1) or (2) — it's a
   2-hour fix, not a new training arc.

## What this lands

Operational rule (added to CLAUDE.md):

> **Factor signal-quality from a 6-window walk-forward at 20-day rebal
> is too lagged to clear a real-time VIX-median meta-gate.** The lag
> is structural (window val periods average ~2 years; signal-quality
> reads inherit that resolution) and cannot be removed by changing
> the training loss. Before deploying factor signal-quality as a
> meta-gate input, retest at finer walk-forward grain — either
> per-bar emission via the existing `signal_quality_per_val_bar`
> arrays (2-hour fix) or a refit with shorter val windows (1h T4
> run). At the v0 6-window grain, VIX-only is the dominant gate.

Implementation:

- Extension in `apps/gate/scripts/macro_meta_gate_eval.py`: loads
  `Output/sizing-input-rank_ic-windows.npz` if present, joins
  factor_sq via most-recent val_start lookup, emits four-arm comparison
  + verdict.

## Master walk-forward log

Leaderboard row:
[2026-05-12 macro — sizing-input v1 retroactive meta-gate](../leaderboard.md#operating-conditions),
verdict [`confirmed-null`](../leaderboard.md#verdict-labels) on the
incremental-lift-over-VIX hypothesis.

Related findings:
[`factor-sizing-input-v0`](factor-sizing-input-v0.md),
[`macro-regime-diagnostic`](macro-regime-diagnostic.md),
[`prediction-problem-pivot-arc`](prediction-problem-pivot-arc.md).
