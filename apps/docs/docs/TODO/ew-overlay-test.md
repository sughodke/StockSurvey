# EW + rank-IC overlay — does the head add anything as a tilt on top of EW?

**Status: parked.** Pursuing
[`different-prediction-problem`](different-prediction-problem.md)
first. Return here if (a) the prediction-problem pivot also
nulls out, or (b) we want a "deploy now while research continues"
hedge that's better than pure EW.

## The question

Three independent tests
([passive-EW](../findings/passive-ew-benchmark.md),
[long-short](../findings/factor-rankic-long-only-mismatch.md),
[loss-pivot](../findings/factor-loss-pivot.md)) all confirmed
that long-only top-N, long-short, and Sharpe-aligned-loss
constructors *all* lose to passive EW at our +0.005 to +0.012
cross-sectional IC scale. The natural follow-on: maybe the
rank-IC head still has *some* skill, just not enough to overcome
the cost of fully replacing EW with concentrated bets. Test
that hypothesis with a convex combination:

```
w_t = (1 − α) · ew_weights_t  +  α · softmax_top_n(scores_t)
```

at various `α ∈ {0.0, 0.05, 0.1, 0.25, 0.5, 1.0}`.
- `α = 0.0` recovers pure EW (sanity check — should match the
  [`passive-ew-benchmark`](../findings/passive-ew-benchmark.md)
  numbers when matched to the same window).
- `α = 1.0` recovers the existing long-only top-N row
  (val Sharpe +0.278 on factor-narrow per the
  [loss-pivot rank_ic arm](../findings/factor-loss-pivot.md)).
- Intermediate α blends: bulk EW exposure + small skill tilt.

If there's a sweet spot at small α where val Sharpe exceeds both
endpoints, the head has overlay-monetizable skill even though
it can't generate alpha as a standalone strategy.

## Pre-registered pass/fail cuts

| Outcome | Best-α val Sharpe vs EW val Sharpe | Verdict | Action |
|---|---|---|---|
| **Pass** | ≥ EW + 0.10, ≥ 4/6 windows positive alpha at that α | `confirmed-OOS` | Ship `EW + α* · rank-IC tilt` as the live overlay strategy. |
| **Marginal** | EW + (0.00, 0.10) for some α | `partial-OOS` | Real but noise-band; deploy only if the overlay also reduces drawdown or improves consistency. |
| **Fail** | No α clears EW val Sharpe by any positive margin | `confirmed-null` | The +0.005 IC cannot be monetized in any constructor at this universe / horizon — strict superset of the existing three nulls. Lock in "deploy EW" as the operational answer, focus all research on prediction-problem pivot. |

The Sharpe ratio is the load-bearing column because the
operational rule
([`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md))
says alpha-over-passive-EW is what determines shippability.
Track turnover and tracking error as secondary columns — at
small α the overlay's friction is bounded near EW's, so the
test is not just "does Sharpe go up" but "does Sharpe go up
*per unit added cost*."

## Test design

- **Universe:** factor-narrow (297 stooq_us_long names, the same
  universe every recent factor row uses).
- **Windowing:** the existing 6-window walk-forward at
  `rebal_days=20`. EW per window is computed within the same val
  block as the model arm — apples-to-apples by construction.
- **Head:** the rank-IC linear head from
  `Output/loss-pivot-rank_ic-windows.npz` (already saved per
  window with `head_params` dict). No retraining — re-apply the
  saved head to the val block under each α.
- **Constructor:** for each window, blend EW (`1/N` over the
  liquid mask) with the softmax-top-N portfolio at the existing
  temperature (1.0). For the pure rank-IC head, the
  `ew_weights_t = mask_t / mask_t.sum()`.
- **Costs:** matched 10 bps × L1 turnover at each rebal — same
  as everything else. EW pays its drift-rebalance turnover, the
  overlay pays the blended turnover. Honest comparison.
- **α grid:** `{0.0, 0.05, 0.1, 0.25, 0.5, 1.0}`. Six points is
  enough to see the contour; if there's a sweet spot we'll know
  where to refine.

## Implementation scope

~80 LoC, no retraining, no new universe build:

- New: `factor.objectives.ew_overlay_weights(scores, mask, log_temp, alpha)`
  returning `(1−α)·ew_weights + α·softmax_top_n_weights`. Already
  composable from existing primitives — likely ~15 LoC.
- New: `factor.objectives.block_sharpe_ew_overlay(scores, log_temp,
  blr, mask, rebal_days, commission_frac, alpha)` mirroring
  `block_sharpe`'s structure. ~25 LoC.
- New driver: `apps/factor/scripts/ew_overlay_eval.py` — loads
  the saved per-window head, sweeps α, reports val Sharpe + EW
  Sharpe + alpha + turnover per (α, window), prints a contour
  table, writes JSON. ~60 LoC.
- (Optional) extend `WalkForwardWindow` with an
  `overlay_alpha_sweep` field for future runs that want it
  baked into the trainer rather than computed post-hoc. Not
  needed for v1.

Local job. Single walk-forward already runs in <2 min on the
cached panel. Multiple α applied to a saved head is cheaper
than one walk-forward arm. Total wall: <30s.

## What this TODO is *not* a test of

- Not a test of training the head with an overlay-aware loss.
  v1 reuses the rank-IC-trained head as-is (free).
- Not a test of dynamic / state-dependent α (e.g. raise α when
  the head is more confident). Static-α first; dynamic-α only
  if static lifts above the noise band.
- Not a test on the wider universe (stooq_us_long, ex-Phase-2,
  factor-wide). Factor-narrow first because it's the universe
  every other factor row uses; if α* > 0 here, repeat on wider.
- Not a test of long-short overlay. Always-positive overlay
  weights to keep the deployment story simple — the long-short
  result already nulled out.

## Implementation order (when picked back up)

1. Implement `ew_overlay_weights` + `block_sharpe_ew_overlay`
   in `factor.objectives`.
2. Driver `ew_overlay_eval.py` — load
   `Output/loss-pivot-rank_ic-windows.npz`, sweep α, write
   summary + per-window npz.
3. Pre-register reading: which α (if any) clears the threshold.
4. Land 1-2 leaderboard rows (best-α + α=0 reference) and the
   verdict-aligned next-move.
5. If `confirmed-OOS`: write a closing finding + ship checklist
   (broker integration uses the same `relational/inference.py`
   pattern but with the blended weight vector).

## Why we deferred this

The
[`factor-loss-pivot`](../findings/factor-loss-pivot.md) result
showed that *training* on Sharpe collapses the head to
near-argmax concentration in the low-IC regime — the optimizer
correctly identifies that high in-sample Sharpe wants
concentration, but OOS that concentration destroys 0.37 of
Sharpe vs the spread-thin rank-IC head. The overlay is
mathematically a different question (it constrains the
deployment, not the training), but it shares the same root
cause: at +0.005 IC, the per-name signal-to-noise is too low
to monetize directly. The most likely outcome of the overlay
test is "α* = 0" — i.e. the optimum blend is pure EW.

If that's the case, we'll have *four* independent tests
converging on "pivot prediction problem," which is conclusive
enough to redirect all research effort. Running this test
without that signal would still be useful (validates the
"deploy EW" answer concretely), but the cost-of-information
ranking favors changing the prediction problem first — that
test class has uncapped upside if any of pair-spread / drawdown
/ IV-vs-realized has materially higher IC. The overlay test
has bounded upside (alpha at most a fraction of the standalone
rank-IC head's Sharpe).

Return here when:
- The prediction-problem pivot also nulls out and we need a
  shippable interim strategy.
- Or if a quick-test mood strikes — it's a 30-minute experiment
  with a clean falsifiable verdict, parking it forever would
  be a small loss but reviving it is cheap.
