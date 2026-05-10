# Multi-task aux head — `aux_weight` sweep

!!! success "Resolved 2026-05-10 — `confirmed-null` for magnitude-extraction"
    Sweep ran for `aux_weight ∈ {1.0, 10.0}`. Result: aux=1.0
    fits train aux (MSE 0.78) but val aux MSE rises to 1.16
    (worse than predicting zero), val IC −0.0084. aux=10.0
    collapses the trunk on 3/6 windows. None of the three
    pre-registered decision-rule branches matched — a fourth
    branch ("aux head learns train, reverses on val due to
    cross-sectional regime non-stationarity") was added.
    Full data + analysis in
    [`findings/factor-multitask-aux-head#aux_weight-sweep-2026-05-10`](../findings/factor-multitask-aux-head.md#aux_weight-sweep-2026-05-10).
    Next experiment is **not** another aux-head variant — it's a
    different prediction problem ([`different-prediction-problem`](different-prediction-problem.md))
    or a different horizon ([`rebal-days-sweep`](rebal-days-sweep.md)).

The
[`aux_weight=0.1` result](../findings/factor-multitask-aux-head.md)
showed a real +0.012 trunk-regularization lift over the `mlp`
baseline but did not clear the linear-on-encoder baseline (+0.0031)
or the indicator baseline (+0.0120). Aux MSE stayed at ~1.0
throughout — the aux head itself learned nothing. That made the
mechanism observation incomplete: at `aux_weight=0.1` we couldn't
tell whether the magnitude-extraction hypothesis was wrong, or
whether the aux gradient was just too weak to exercise it.

## Falsifiable hypothesis

At `aux_weight ≥ 1.0`, the auxiliary head's own gradient becomes
competitive with the primary head's, the aux MSE drops below 1.0
(aux head learns the magnitude target), and the trunk reorganizes
around the joint objective. Three distinguishable outcomes:

| Aux MSE behavior     | Primary val IC behavior        | Reading                                        |
|----------------------|--------------------------------|------------------------------------------------|
| Drops below 0.8      | Lifts above +0.0120            | magnitude extraction works at higher gain     |
| Drops below 0.8      | Drops below +0.0001 or negative | tasks compete; aux is orthogonal to return prediction |
| Stays at ~1.0        | Stays at ~+0.0001              | trunk capacity insufficient; bump `mlp_hidden` |

## Test design

Modal-T4 sweep, same backbone / universe / windowing as
[factor-multitask-aux-head](../findings/factor-multitask-aux-head.md):

```bash
uvx modal run apps/factor/scripts/modal/train_ssl_walkforward.py \
    --scorers mlp_multitask \
    --aux-weight $W \
    --n-steps 200 \
    --weight-decay 1e-3
```

for `$W ∈ {1.0, 10.0}`. Two arms; each ~20 min wall (the multitask
arm at `aux_weight=0.1` cost 1208s/run on the prior fire). Total
~45 min including cold-start.

Re-fire the existing baselines (`linear`, `mlp`, `mlp_multitask
aux=0.1`) only if the f64 patch lands a new commit between the
prior 2026-05-10 run and this sweep — they're invariant to
`aux_weight` and don't need to be re-run otherwise.

## Decision rule

- **Aux MSE drops + primary lifts** → confirmed magnitude
  extraction; record as a new finding "aux_weight=N clears the
  ceiling" with `confirmed-OOS`. Sweep
  finer (e.g. `aux_weight ∈ {2, 5}`) to find the optimum.
- **Aux MSE drops + primary degrades** → tasks compete, aux is
  orthogonal. Record as `confirmed-null` for the magnitude-
  extraction hypothesis; the regularization-only reading from the
  prior finding becomes the operational rule.
- **Aux MSE stays at ~1.0** → trunk capacity insufficient. Follow
  up with `--mlp-hidden 128` or `256` at `aux_weight=1.0`. Mark
  this sweep itself as `partial-OOS` and the head-architecture
  sweep as the next experiment.

## Out of scope

- Stage 2 (joint head + backbone fine-tune) with multitask. The
  current implementation rejects `aux_weight > 0` with
  `finetune_steps > 0` — that combination is a different beast
  (joint trunk + conv-stack updates under the dual loss) and
  needs its own validation pass after this `aux_weight` sweep
  lands.
- The
  [strict-SSL `masked-ae` vs supervised-`cnn`](../findings/factor-ssl-walkforward.md#outstanding-questions)
  comparison. Independent of the aux-weight question; tracked
  separately.

## Implementation

No code changes — `aux_weight` is already a CLI flag on
`apps/factor/scripts/modal/train_ssl_walkforward.py`. Just fire
the sweep and append rows to the leaderboard for each arm.
