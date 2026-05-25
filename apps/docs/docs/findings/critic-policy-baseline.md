# Critic Φ-imitation policy — first formal walk-forward eval

**Operational rule.** The critic policy (`critic.policy.train_policy`,
vanilla + CQL flavors) is a `partial-OOS` meta-allocator over
pair-level Sharpes by its own driver's pre-reg cuts, but its alpha
ceiling is **bounded above by the upstream `apps/pairs` v0
`confirmed-null`** verdict. Do NOT promote it to a deployed strategy
on the pair-trading basket — even a perfect meta-allocator cannot
manufacture portfolio alpha that the underlying basket doesn't have.
The critic *can* still be useful as a diagnostic for which pair / arm
features carry residual predictive signal across windows; it should
not be wired through any `* live` harness.

## NOT an apples-to-apples row vs HRP / RSI / scalogram / regime-CWT / velocity

This is the load-bearing caveat for the leaderboard row. The critic
operates one *level up* from the cross-sectional weight-construction
layer:

- The other five universe-agnostic rows from 2026-05-25 take a `(T, N)`
  price panel of `stooq_us_long`, produce a weight vector at each
  rebal bar, and report portfolio Sharpe vs passive equal-weight on
  the same universe.
- The critic policy takes a **pre-evaluated pair-Sharpe distribution**
  from `Output/pairs-walkforward-summary.json` (the 6-window
  `apps/pairs` classical-v0 run on factor-narrow ~297 names), learns
  a Φ value function from pair features → realized pair Sharpe, then
  trains a policy to pick top-K=50 pairs out of ~200 per window. Its
  label is **realized pair Sharpe**, not portfolio Sharpe.

These two prediction problems live at different layers of the stack.
This row is comparable to the `cfr` Phase-1 row (2026-05-12) which is
also a meta-allocator over an action menu, not to the five
weight-construction heads on the same date.

## Eval setup

- **Universe**: inherits from upstream `apps/pairs` v0 walk-forward —
  factor-narrow 297 names; 6 windows of 1260-train / 780-val / 780-step
  on stooq_us_long timestamps.
- **Training data**: 1200 pair triples (200 per window × 6 windows),
  each with 4 pair-level features + 5 FRED macro features (vix_level,
  vix_6m_change_pts, vix_1m_change_pts, spy_252d_log_ret,
  spy_63d_log_ret).
- **Φ training**: tinygrad MLP, hidden=16, n_layers=2, n_steps=100,
  lr=5e-3, weight_decay=1e-3. Loss is MSE against realized 20d pair
  Sharpe.
- **Policy training**: same architecture as Φ; π_vanilla trained
  against −Φ(s, σ(π_score)), π_cql adds CQL anchor toward empirical
  inclusion rate 0.25. 100 steps each.
- **Deployment**: rank held-out window's pairs by policy score, take
  top-50, mean realized Sharpe.
- **Pre-reg cuts** (driver's own, predates this row):
  - STRONG-PASS: best policy vs LR ≥ +0.10 AND vs Φ-direct ≥ +0.05
  - PASS / partial-OOS: vs LR ≥ +0.03 AND vs Φ-direct ≥ +0.01
  - MARGINAL / partial-OOS: vs LR ≥ +0.01
  - FAIL / confirmed-null: otherwise

## Mean top-50 deployment Sharpe (LOO-by-window, mean across 6 folds)

| Arm                       | Mean Sharpe | Capture of oracle headroom |
|---------------------------|-------------|-----------------------------|
| All-pairs baseline        | +0.1015     | (0% — baseline)             |
| LR (v1 analog)            | +0.1408     | +5.6%                       |
| Φ direct (v0.1)           | +0.1312     | +4.2%                       |
| π vanilla (−Φ loss)       | +0.1199     | +2.6%                       |
| **π CQL (anchor 0.25)**   | **+0.1558** | **+7.7%**                   |
| Within-window oracle      | +0.8041     | 100%                        |

Oracle headroom: **+0.7026**. Best policy (π CQL) vs LR: **Δ +0.0150**;
vs Φ-direct: **Δ +0.0246**. Verdict: **MARGINAL / `partial-OOS`**.

## Reading the verdict honestly

1. **The headline number lives in the noise band.** The LR baseline
   already captures 5.6% of oracle headroom; π_cql squeezes another
   2.1pp on top. The marginal +0.0150 lift over LR clears the
   driver's lowest pre-reg threshold (Δ ≥ +0.01) by a hair.
2. **CQL anchor genuinely helps.** π_cql beats π_vanilla by +0.036
   Sharpe — anchoring the policy to the empirical 0.25 inclusion
   frequency adds real signal that vanilla over-extrapolation
   destroys.
3. **Φ-direct ranking is worse than LR.** Adding the policy layer
   over LR (Φ-direct alone) loses signal; adding the CQL-anchored
   policy layer on top of Φ recovers it. The architecture of the
   meta-stack matters more than the algorithm at this signal level.
4. **The binding ceiling is upstream pairs v0.** Mean agg val
   portfolio Sharpe of `apps/pairs` v0 is **+0.099** (`confirmed-null`
   per the 2026-05-10 leaderboard row). The critic's +0.156 mean
   *pair Sharpe* number is at a different granularity, but the
   portfolio-level alpha at deployment time is bounded by what the
   underlying classical-v0 mean-reversion machinery captures.

## Mechanism — the brief's gotcha is real

Per the brief: "Φ-imitation against a learned value function is a
behavioral-cloning policy distilled from factor / vol / gate / pairs
outputs. Its alpha ceiling is bounded by whatever the BEST of those
upstream signals is producing — if those are mostly `partial-OOS` or
`confirmed-null`, critic's ceiling is similarly low."

This is exactly what shows up empirically:

- All four meta-allocators (LR, Φ, π_v, π_cql) recover **2.6% to
  7.7%** of the oracle headroom.
- The +0.054 lift π_cql gets vs the all-pairs baseline is 90%
  "you knew the pair would work" residual after the LR-on-features
  baseline already extracts 5.6%. CQL adds the last 2.1pp.
- The features themselves do not carry strong cross-window
  generalization. The +0.7-Sharpe gap between oracle and the policy
  is the "you can't predict next-window pair winners from this
  feature set" signal density limit, not the algorithm choice.

## Upstream artifact availability — the brief's blocked path was NOT triggered

Per the brief: "If the critic's upstream-feature dependencies aren't
already cached in `Output/`, the cost to build them dominates the
eval. In that case: STOP AT EVAL A, write a blocked finding."

All four required upstream artifacts ARE present:

- `Output/horizon-mixture-windows.npz` (factor mixture sweep)
- `Output/horizon-mixture-summary.json` (5 alpha arms × 6 windows)
- `Output/vol-walkforward-v3-regime-gated-summary.json` (vol v3 per-window)
- `Output/gate-walkforward-summary.json` (gate v0)
- `Output/pairs-walkforward-summary.json` (pairs v0 — the pair-level
  triples this row consumes)
- `Output/pairs-predictor-walkforward-summary.json` (pairs v1 LR
  predictor + oracle arms)

The eval RAN end-to-end in 90 seconds on local tinygrad-CPU; no Modal
needed.

## Three surprises

1. **CQL anchor outperforms vanilla policy by +0.036 Sharpe.**
   Anchoring the policy to the empirical 0.25 inclusion frequency
   genuinely helps. The "policy distillation collapses without an
   anchor" pathology is observable in numbers, not just in theory.
2. **Φ-direct (+0.131) is *worse* than LR (+0.141).** A tinygrad
   2-layer MLP trained on 1000 pair triples loses to a 9-feature
   ridge regression on the same data. At this n the LR baseline is
   actually the right model class; Φ adds noise.
3. **All four meta-allocators sit in a 5pp band on oracle-capture
   (2.6% to 7.7%).** The feature stack — 4 pair features + 5 macro
   features — does not contain enough cross-window signal to push
   any meta-allocator past ~8% of the oracle ceiling. The signal
   problem is in the features, not the model.

## CLAUDE.md app-inventory gap (flagged for user)

The `apps/critic/` app is NOT listed in CLAUDE.md's app-inventory
section (which covers `regime`, `relational`, `factor`, `replay`,
`gate`, `pairs`, `vol`, `dca`, `notebook`, `docs`, `v1`, and
elsewhere `cfr`). The brief explicitly instructs to flag this for the
user rather than unilaterally edit. Suggested entry:

> - `apps/critic/` — meta-allocator over already-evaluated arm
>   sweeps. `dataset.py` consolidates per-app walk-forward summaries
>   from `Output/` (factor mixture, vol v3, gate v0, pairs v0/v1)
>   into `(state, action) → realized-Sharpe` triples. `model.py`
>   trains a tinygrad Φ value function on those triples; `policy.py`
>   trains vanilla and CQL policies that imitate Φ for top-K
>   deployment. Verdict to date: `partial-OOS` by the driver's own
>   pre-reg cuts, but capacity-bounded by the upstream
>   `apps/pairs` `confirmed-null`. Not deployed.

## Operational rules extracted

1. **The critic is bounded above by its upstream apps' verdicts.**
   No amount of meta-allocator architecture can manufacture
   portfolio alpha that the underlying arms don't already produce.
2. **The +0.015 lift vs LR is in the noise band.** Treat it as a
   tooling-functions test (the pipeline works end-to-end) rather
   than as a signal claim.
3. **The right next experiment is upstream feature work, not
   meta-allocator architecture.** If the pair-feature stack
   captures only 7.7% of oracle headroom, building a fancier
   policy on top of it cannot help. Either find features that
   predict next-window pair Sharpes more accurately (cross-window
   stability features, cointegration t-stat, half-life), or pivot
   to a different prediction problem.

## Master walk-forward log

- [critic-policy-baseline (2026-05-25) — `partial-OOS`](../leaderboard.md#verdict-labels)
- Upstream bound: [pairs-classical-v0 (2026-05-10) — `confirmed-null`](../leaderboard.md#verdict-labels)
- Adjacent meta-allocator: [cfr-phase1 (2026-05-12) — `partial-OOS`](../leaderboard.md#verdict-labels)

Artifacts:

- `Output/critic-policy-baseline-summary.json` (this row)
- `Output/critic-policy-walkforward-summary.json` (prior runs; same driver)

Driver: `apps/critic/scripts/run_policy_walkforward.py`.
