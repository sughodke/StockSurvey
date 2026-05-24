# Factor head trained directly against studentized Sharpe-diff vs EW

**Status: `pending` — pre-registration locked 2026-05-23. Eval not yet
run. This page must be committed BEFORE the head-to-head training run
fires; the falsification bar is not editable post-hoc.**

## Motivation

The methodology rewrite ([`findings/ladder-methodology-rewrite`](../findings/ladder-methodology-rewrite.md))
established that:
1. DSR is a within-arc selection-bias correction, not a cross-arc rank
   key.
2. The literature-canonical cross-arc test is Ledoit-Wolf (2008)
   studentized stationary-bootstrap CI on Sharpe-difference vs DCA.
3. Under that test, **zero arcs beat DCA at 95%** on the current
   ladder.

The differentiable analogue of the Ledoit-Wolf test
(`ss_portfolio.studentized_sharpe_diff` / `factor.objectives.block_
studentized_sharpe_diff_vs_ew`, landed commit `cb8f84e`) is now
available as a training loss. **This arc tests whether directly
training a factor head against `ΔSR/s.e. vs EW` changes the answer**
compared to the existing `ir_vs_ew` and `rank_ic` losses on the same
walk-forward windows.

The honest expectation (recorded for falsifiability): **`confirmed-null`
or weak `partial-OOS`**. The methodology rewrite established the
cross-sectional null is robust on this universe. Direct training
doesn't escape data-snooping — it just concentrates it into the loss
function. If the new loss still produces a `confirmed-null`, the null
conclusion strengthens (robust to optimization method, binding
constraint is the data not the search).

## Locked search / arm definition

| arm | loss_kind | other knobs |
|---|---|---|
| baseline-IR | `ir_vs_ew` | `train_temp=True`, head=linear |
| **candidate** | **`studentized_sharpe_diff_vs_ew`** | `train_temp=True`, head=linear, `with_moments=False` |
| reference | `rank_ic` | `train_temp=False`, head=linear |

All three arms train on **identical walk-forward windows** with
identical hyperparameters (head, optimizer, n_steps, lr, weight_decay,
forward target, commission, rebal_days). The only variable is
`loss_kind`.

**Universe**: factor-narrow (297 names). Locked.
**Rebal cadence**: 5 trading days (the established short-horizon
candidate per `findings/factor-shorthorizon-representation.md`).
**Walk-forward**: 6 windows on factor's existing block schedule
(matches the `sh-indicator-r5-s1` runs already on disk).
**Forward skip**: 1 (microstructure-controlled, matches the existing
`partial-OOS` row).
**Commission**: 10 bps round-trip (same as all factor arcs).
**N_steps**: 200 (same as existing factor sweeps).
**Seed**: 42 (single seed per pre-reg; not a multi-seed sweep — that
would inflate selection cost).

## Pre-registered falsification bar (locked)

For each arm, compute:
- per-window val Sharpe (the existing eval)
- per-window val ΔSR vs EW (the new metric, computed via numpy
  `studentized_sharpe_diff(port_block_ret, ew_block_ret)`)
- **pooled OOS bootstrap CI** of ΔSR vs EW across all 6 windows'
  val streams, via `ss_portfolio.sharpe_difference_ci` with default
  block bootstrap parameters

**Bar**:

| outcome | verdict |
|---|---|
| candidate's pooled OOS bootstrap CI on ΔSR vs EW **excludes 0** on the positive side AND beats baseline-IR by mean val t-stat ≥ +1.0 | **`confirmed-OOS`** — adopt the new loss as the default factor training objective |
| candidate's pooled OOS CI includes 0 BUT mean val t-stat exceeds baseline-IR by ≥ +0.3 | **`partial-OOS`** — useful loss improvement, deploy as an additional arm but don't replace `ir_vs_ew` |
| candidate's pooled OOS CI includes 0 AND mean val t-stat ≤ baseline-IR | **`confirmed-null`** — direct training doesn't escape the binding-data constraint; canonical losses stand |

Side-result either way: a head-to-head table of `rank_ic` /
`ir_vs_ew` / `studentized_sharpe_diff_vs_ew` per-window val numbers,
which will land in the finding regardless of verdict.

## Eval driver (locked)

```bash
uv run python apps/factor/scripts/train_studentized_sharpe_diff.py
```

The script:
1. Loads the 297-name factor-narrow universe (same prep as
   existing factor walk-forwards).
2. For each loss in `{rank_ic, ir_vs_ew, studentized_sharpe_diff_vs_ew}`:
   - Runs `train_scorer_indicators_walkforward(..., loss_kind=loss)`
     at `rebal_days=5, forward_skip=1, train_window=200, val_window=block, step=block`.
   - Records per-window val IC, val Sharpe, val ΔSR-vs-EW.
3. Computes pooled OOS bootstrap CI of ΔSR vs EW for the candidate
   arm.
4. Writes the head-to-head summary to
   `Output/factor-studentized-sharpe-diff-vs-ew.json`.

Artifact written before the verdict is recorded — no post-hoc
parameter changes allowed.

## Why this is the right first user

Per the strategy-selection brief
(`.research-best-candidate-for-studentized-loss.md`):
- **n=936 blocks at 5d cadence** — large enough that the delta-method
  s.e. dominates the null estimation noise; gradient signal beats
  noise.
- **Parameters flow end-to-end**: head weights + log-temperature
  → softmax → port_block_ret → loss. Already trainable via tinygrad.
- **Direct baseline exists**: `ir_vs_ew` runs on identical infra; the
  comparison is apples-to-apples by construction.
- **No new universe / new prep needed**: reuses the established
  factor-narrow universe.

## Expected outcome (recorded for honesty)

`confirmed-null` or weak `partial-OOS`. The cross-sectional null on
factor-narrow is well-established across the factor app's findings:
short-horizon representation (`partial-OOS` at +0.0114 IC after lag),
SSL backbone (confirmed-null), every L/S construction (negative).
Direct training against the studentized t-stat is unlikely to surface
a result the rank-IC + IR-vs-EW losses missed; if it does, the
explanation is likely "different local optimum" rather than "different
information". Recorded so the post-eval analysis can compare against
prior.

## Cross-links

- Differentiable loss commit: `cb8f84e`
- Methodology rewrite: [`findings/ladder-methodology-rewrite`](../findings/ladder-methodology-rewrite.md)
- Strategy-selection brief: `.research-best-candidate-for-studentized-loss.md`
- Factor short-horizon baseline: [`findings/factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md)
