---
tags:
  - factor-narrow
  - confirmed-null
---

# Factor — `aux_weight` sweep falsifies cross-sectional magnitude extraction at H=20

**Operational rule.** The
[multi-task aux head](factor-multitask-aux-head.md) cannot extract
out-of-sample cross-sectional magnitude information at the
20-day horizon on `stooq_us_long` at any tested gain. The
`aux_weight=0.1` lift over the `mlp` baseline is a regularization
artifact (the aux head doesn't train); pushing the gain hard
enough to actually train the aux head reverses primary val IC
(`aux_weight=1.0`) or collapses the trunk (`aux_weight=10.0`).
Don't run more aux-head variants — the binding constraint is
that the cross-sectional ranking of forward 20-day magnitudes is
**regime-non-stationary** between train and val windows on this
universe, so no aux-head implementation can extract a signal that
isn't there.

## What was pre-registered

The sweep was set up after the
[`aux_weight=0.1` result](factor-multitask-aux-head.md) showed a
+0.012 val IC lift over the plain `mlp` baseline with the aux
head's own MSE pinned at ~1.0 (random performance on a unit-
variance target). At that point the regularization-vs-magnitude-
extraction reading was ambiguous: aux=0.1 might just be too weak
a gain to exercise the magnitude-extraction mechanism.

The sweep tested the magnitude-extraction hypothesis at higher
gains. The full pre-registration:

### Hypothesis

At `aux_weight ≥ 1.0`, the auxiliary head's own gradient becomes
competitive with the primary head's, the aux MSE drops below 1.0
(aux head learns the cross-sectional winsorized z-score target),
and the trunk reorganizes around the joint objective.

### Decision rule (3 branches)

| Aux MSE behavior | Primary val IC behavior | Reading |
|---|---|---|
| Drops below 0.8 | Lifts above +0.0120 | Magnitude extraction works at higher gain |
| Drops below 0.8 | Drops below +0.0001 or negative | Tasks compete; aux is orthogonal to return prediction |
| Stays at ~1.0 | Stays at ~+0.0001 | Trunk capacity insufficient; bump `mlp_hidden` |

### Test design

Modal-T4, same backbone / universe / windowing as the
[parent finding](factor-multitask-aux-head.md): 297 stooq_us_long
tickers, 6-window walkforward (63 train / 39 val / 39 step blocks
@ rebal_days=20), backbone frozen, `mlp_multitask` head only,
200 AdamW steps at lr=1e-2 wd=1e-3. Two arms, `aux_weight ∈
{1.0, 10.0}`, each ~20 min wall:

```bash
uvx modal run apps/factor/scripts/modal/train_ssl_walkforward.py \
    --scorers mlp_multitask --aux-weight $W \
    --n-steps 200 --weight-decay 1e-3
```

## What we got — the fourth branch

None of the three pre-registered branches matched.

### `aux_weight=1.0` — train aux MSE drops, val aux MSE rises *above* 1.0

The aux head trains cleanly on every window — train MSE 0.61–0.95,
mean 0.78. But the function it fits reverses sign on val: val MSE
1.01–1.31, mean **1.16**, which is *worse than predicting zero*
on a unit-variance target. The aux predictions are not just
uncorrelated with val magnitudes — they are anti-correlated.
Joint loss drags primary val IC to **−0.0084** (1/6 windows
positive), worse than aux=0.1's +0.0001.

This is the fourth branch: train aux fits (so capacity isn't the
limiter), val aux MSE > 1.0 (so the aux task is *anti-stationary*
between train and val), primary val IC degrades. The decision
rule didn't anticipate this because all three pre-registered
branches assumed the aux task was either learnable (branches 1–2)
or unlearnable (branch 3) — not "learnable in-sample but
anti-predictive out-of-sample."

### `aux_weight=10.0` — trunk collapse on half the windows

At 100× the aux=0.1 gain the aux gradient destabilizes joint
optimization: 3/6 windows (0, 2, 5) hit *exact* train IC = 0.000
and val aux MSE = 1.000 — the trunk projections fully zero out
and both heads emit constant zero. The other 3 windows show the
same overfit-and-reverse pattern as aux=1.0 in miniature (train
aux MSE 0.89–0.98, val aux MSE 1.01–1.06). Headline mean val IC
+0.0048 is window 3's +0.0254 carrying two near-zero windows; the
"50% positive frac" is misleading because 3 of those 6 windows
are zero-collapsed, not neutral.

### Aggregates across the full sweep

| aux_weight | mean tr_ic | mean vl_ic | mean tr_aux | mean vl_aux | pos-vl frac |
|---:|---:|---:|---:|---:|---:|
| 0.1  | +0.762 |   +0.0001 | ~1.0 | ~1.0 | 3/6 (0.50) |
| 1.0  | +0.618 | **−0.0084** | **0.778** | **1.159** | 1/6 (0.17) |
| 10.0 | +0.140 | +0.0048 | 0.971 | 1.018 | 3/6 (0.50)\* |

\* aux=10.0 pos-val-IC frac is misleading — 3 windows are
zero-collapsed.

Full per-window data is in the
[parent finding's sweep section](factor-multitask-aux-head.md#aux_weight-sweep-2026-05-10).

## Three things this falsifies

- **The trunk-capacity hypothesis** (branch 3) is wrong. At aux=1.0
  the trunk has no trouble fitting the aux task on train (MSE
  0.78); the failure is on val. Bumping `mlp_hidden` would let
  the trunk overfit aux even harder; it wouldn't fix
  non-stationarity.
- **The orthogonal-tasks hypothesis** (branch 2) is too weak.
  Orthogonal would mean val aux MSE stays at ~1.0 (random) while
  primary degrades. We see val aux MSE *above* 1.0 — the train
  aux signal is anti-predictive on val. That's a stronger
  statement than "uncorrelated."
- **The aux=0.1 regularization reading is correct** but the
  mechanism is now sharper: aux=0.1 helps because the aux head
  *fails to train enough to overfit*. Any aux gain large enough
  to actually train the aux head is also large enough to drag
  primary IC down via the shared trunk's exposure to overfit aux
  gradients.

## What it means for the test-design protocol going forward

The sweep is a useful prior on how to scope auxiliary-objective
follow-ups elsewhere in the workspace:

- **Pre-register an "anti-stationary" branch** when designing
  multi-task experiments on cross-sectional return targets. The
  three-branch rubric (train + val both succeed / train succeeds
  + val random / train fails) misses the most likely failure
  mode for return-prediction problems: train succeeds + val
  *anti-correlates*.
- **Don't conflate aux-head MSE-near-1.0 with "didn't learn."**
  At aux=10.0 most windows hit MSE = 1.0 because the trunk
  collapsed, not because the head didn't try. Read train aux MSE
  alongside train IC before concluding capacity is the limiter.
- **Cross-sectional magnitude is regime-non-stationary at H=20**
  on this universe. Any future probe of magnitude-aware
  supervision (e.g. quantile regression, vol-aware sizing as a
  return-prediction sub-objective) should pre-test the
  stationarity of the cross-sectional ranking before investing
  in the architecture.

## Implication for further factor work

Don't run more aux-head variants on this universe / horizon. The
binding constraint isn't the implementation — it's that the
out-of-sample cross-sectional magnitude signal at H=20 on
`stooq_us_long` doesn't exist for the aux head to extract.
Natural next experiments are orthogonal:

- **Different prediction problem** —
  [TODO/different-prediction-problem](../TODO/different-prediction-problem.md)
  pivots to pair-spread, drawdown, or IV-vs-realized targets,
  where the cross-section may be more stationary.
- **Different horizon** —
  [TODO/rebal-days-sweep](../TODO/rebal-days-sweep.md) tests
  whether longer or shorter horizons stabilize the cross-sectional
  ranking. The 20-day horizon is the canonical default but
  hasn't been ablated.

The `mlp_multitask` scorer itself stays in `factor.scorers` as a
falsified-but-instructive arm; the operational rule on
`aux_weight ≤ 0.1` regularization is preserved in the
[parent finding](factor-multitask-aux-head.md).

## Master walk-forward log

Three [leaderboard rows](../leaderboard.md), all 2026-05-10:

- `aux_weight=0.1` — `partial-OOS` for `mt > mlp`,
  [`confirmed-null`](../leaderboard.md#verdict-labels) for
  clearing the indicator ceiling. (Pre-sweep row.)
- `aux_weight=1.0` —
  [`confirmed-null`](../leaderboard.md#verdict-labels) for
  magnitude-extraction; supersedes the regularizer-only reading.
- `aux_weight=10.0` —
  [`confirmed-null`](../leaderboard.md#verdict-labels); trunk
  collapse contaminates surface-level lift.

This finding closes the multi-task aux-head arc.
