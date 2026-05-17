# `apps/factor` — return-coupled recurrent CWT embedding (rank-IC-trained, not reconstruction)

## Resolved 2026-05-17 — [`confirmed-null`](../leaderboard.md#verdict-labels), arc closed

The pre-registered k-sweep ran on Modal (`factor-narrow`, 6-window,
fixed hyperparams, `k ∈ {2,4,8,13,16,32}`). **Every arm is at or below
the +0.0120 deterministic-indicator baseline** (best k≤4 +0.0063, far
below the +0.0140 positive cut; k=2 −0.0098; max-capacity k=32 +0.0057,
no better than k=4); flat ≈0 band, no low-`k` plateau. The
pre-registered kill criterion fired mechanically — clean
`confirmed-null`, no band-edge call.

A *trained* end-to-end recurrence — the most expressive move available
on standard CWT data — cannot clear even the cheap indicator baseline
at any `k`. Combined with the arc's prior closures this establishes the
terminal claim: **reconstructible ≫ self-predictable ≫
return-predictable ≈ 0**; the binding constraint is the CWT feature
class, not the representation move. This closes the
[`cwt-recursive-compression`](../findings/cwt-recursive-compression.md)
arc's last live sub-question and the CWT-as-predictor question
arc-wide.

**Next experiment (per the standing frame — *not* another CWT
variant):** the verdict's own implication is the orthogonal novel-data
leg, [`vol-borrow-liquid-universe`](vol-borrow-liquid-universe.md), not
an output-side restructure on the same panel. The
`confirmed-null`→"find an orthogonal lever" rule and the
[`factor-reinforce-target-side`](factor-reinforce-target-side.md)
strategic steer point the same way; no CWT follow-up is registered.

Implementation: `factor.cwt_gru_walkforward` (TinyJit per-window joint
GRU+head trainer), `apps/factor/scripts/modal/cwt_return_coupled.py`.
Finding: [`cwt-recursive-compression`](../findings/cwt-recursive-compression.md#return-coupled-embedding-the-arc-closure)
("Return-coupled embedding — the arc closure"). Leaderboard:
2026-05-17 factor return-coupled-CWT-GRU row. Provenance: 6 arms over 2
Modal invocations (timing-probe k∈{2,32} + completion k∈{4,8,13,16}),
same commit `72802c5`, same pinned config, seed=0, arms independent by
construction.

---

The one live path left open by the
[`cwt-recursive-compression`](../findings/cwt-recursive-compression.md)
arc. That arc closed two sub-questions negative — the CWT does **not**
admit a cheap low-dim *reconstruction* state, and the *self-prediction*
latent barely beats lag-1 persistence (so any geometry on it is the
[`lie_test1`](../findings/factor-indicator-baseline.md) kNN-on-CWT null
re-derived). The only target never tested is **forward returns**: train
the recurrent state end-to-end against the `apps/factor` cross-sectional
rank-IC, not against the CWT's own continuation.

## Verdict → next-experiment chain

`diagnostic` (cwt-recursive-compression) → **this experiment**: turn the
"is there a compact *predictive* CWT statistic?" diagnostic into a
falsifiable cross-sectional OOS test.

## Hypothesis (falsifiable)

A GRU recurrent state over the 13-scale causal CWT, trained
**end-to-end against `pearson_rank_ic` at horizon H=20** (the encoder
recurrence weights are in the trained graph, not a frozen reservoir),
extracts a **low-dimensional** predictive statistic: val rank-IC
saturates by `k ≈ 3–4` rather than needing full rank `k ≈ 13` the way
reconstruction does.

This is genuinely orthogonal to the prior CWT nulls
([`lie_test1`](../findings/factor-indicator-baseline.md) kNN-on-CWT
IC≈0, `lie_test4` shape-beats-CWT at H≈21, `relational-dwt-failure`
4/4 distance scorers) **only** because those falsified *frozen
geometry / unsupervised compression* of the CWT; none trained the
recurrence against returns. If this is also null, the conclusion is
strong: the CWT carries no compact return statistic that *any*
representation move on standard data recovers, and the lever must be a
different feature class or novel data — **not** another model on the
CWT.

## Pinned test design

| Knob | Value | Why |
|---|---|---|
| Universe | `factor-narrow` (297 stooq_us_long, `min_history_bars=6500`) | Direct comparability to the indicator baseline + every recent factor row |
| Windowing | rolling 6-window, `train=63` / `val=39` / `step=39` blocks, `rebal_days=20` | Exactly the deterministic-indicator-baseline windowing (the +0.0120 row) |
| Input | 13-scale causal Ricker CWT of **log-returns**, `lookback=90` | Same panel as the diagnostic so `k` vs `p=13` stays apples-to-apples |
| Sequence | last `L=32` CWT vectors per rebal bar (matches the seq-bottleneck arm) | Bounded recurrent context; leak-free (all ≤ bar `i`) |
| Loss | `-pearson_rank_ic` at H=20, joint GRU+linear-head, AdamW | The existing factor training objective; encoder *in* the trained graph |
| Hyperparams | **fixed**: lr=1e-3, wd=1e-3, n_steps=200, seed per window | No val-based tuning → no hyperparameter leakage |
| Sweep | `k ∈ {2, 4, 8, 13, 16, 32}` | `k=13` = the diagnostic's full-rank `p` anchor |
| Compute | Modal T4 + tinygrad (universe-scale, BPTT) | Heavy-work rule |

**Leak-freedom invariants** (asserted in the trainer): CWT is causal
(`causal_cwt`, `output[t]` depends on `input[:t+1]`); the `L`-window
ends at the rebal bar (all past); forward returns are strictly future
and masked; per-window fresh GRU+head init, trained on the train slice
only, frozen for that window's val; the mandatory per-scale
standardisation is fit on the **train rebal slice only** and applied to
val (a global z-score would leak val-period scale magnitudes).

## Pre-registered kill criterion

Baseline: the deterministic-indicator factor-narrow result is **mean
val rank-IC +0.0120, 5/6 windows positive** (`confirmed-OOS`,
2026-04-30). The flat 168-D CWT feature already *loses* to shape
features cross-sectionally at H≈21 (`lie_test4`, cwt t≈−0.98) — so the
prior is low and this is a **fast decisive falsification, not an
open-ended sweep**.

- **Positive** → `confirmed-OOS` candidate: mean val-IC at `k≤4` ≥
  **+0.0140** (≥+0.002 over the indicator baseline) **and** ≥5/6
  windows positive **and** within −0.002 of the `k=13` value (the
  saturation claim). Implies a compact return-coupled CWT state worth
  building out.
- **Null** → `confirmed-null` / `reversed-OOS`: val-IC ≤ the indicator
  baseline, **or** monotone-in-`k` with no plateau (predictable
  structure is full-rank → the CWT carries no compact predictive
  statistic). **Hard stop — one finding, no β-style sweep escalation.**
- **Partial** → `partial-OOS`: clears the IC bar but not the
  saturation claim (needs `k≈13`). Stratify windows per the standard
  rule before any follow-up.

Either way the result extends
[`cwt-recursive-compression`](../findings/cwt-recursive-compression.md)
("Next experiment — task-coupled predictive state") and lands its own
leaderboard row(s). A null here closes the CWT-as-predictor question
arc-wide.

## Strategic note

This is a cleverer-model-on-standard-data experiment, which the
standing frame ([research-strategy: avoid arbitraged
space](../leaderboard.md#verdict-labels); 9/146 confirmed-OOS base
rate) and the user's own recent steers (the
[`factor-reinforce-target-side`](factor-reinforce-target-side.md)
closure, the novel-data
[`vol-borrow-liquid-universe`](vol-borrow-liquid-universe.md)) flag as
lower-EV than the novel-data path. It is run anyway because it is the
*pre-registered* close-out of the CWT diagnostic arc — but with a hard
kill criterion so it cannot decay into a sweep. If null, do **not**
iterate on the CWT here; pivot to the novel-data leg.

## Driver

`apps/factor/scripts/modal/cwt_return_coupled.py` (Modal T4). Smoke:
`uvx modal run apps/factor/scripts/modal/cwt_return_coupled.py
--max-tickers 30 --n-steps 50 --ks 2,8`. Full:
`uvx modal run apps/factor/scripts/modal/cwt_return_coupled.py`.
Trainer: `factor.cwt_gru_walkforward.train_cwt_gru_walkforward`.
