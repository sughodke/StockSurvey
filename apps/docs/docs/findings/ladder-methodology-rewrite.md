# Ladder methodology rewrite — closing finding

**Operational rule (the load-bearing result of the rewrite):** the
cross-arc DSR ladder, as previously used to rank arcs by deflated-t
magnitude, was methodologically mis-specified. **DSR is a within-arc
selection-bias correction, not a cross-arc rank key** (Bailey-López
de Prado 2014 never claimed otherwise). The literature-canonical
cross-arc test is **Ledoit-Wolf (2008) studentized stationary-
bootstrap CI on the Sharpe-difference of date-aligned streams**. Under
that test, **zero arcs on the current board beat DCA** at the 95%
level. The DCA leg of the DCA + vol ensemble is the only deployable
strategy with statistical backing; the vol-v3 overlay is plausibly
additive but its incremental ΔSR sits inside the bootstrap CI for
vol-v3 alone vs DCA.

## What the rewrite changed

| step | change | effect |
|---|---|---|
| 1 | `standardize_oos` null-noise floor in quadrature | short-stream arcs (vol-v3 n=33, lie n=60, relational-val n=1241) deflate more punitively; DCA-canonical clears t=+2 for the first time (+2.02) |
| 2 | Realistic 200bps options friction on vol-v3-DoltHub | vol-v3 ann Sharpe +2.82 → +2.41; deflated-t +4.08 → +3.43 |
| 3 | Per-arc-class `sharpe_std` empirical re-measurement | DATA-BLOCKED — Optuna JSONs only persist top-10 trials, std≈0 at convergence; kept workspace 0.072 default |
| 4 | Ledoit-Wolf studentized stationary-bootstrap CI vs DCA as new primary cross-arc column | **zero arcs exclude 0 on positive side; four arcs exclude 0 on negative side** |
| 5 | Stratified ladder by (arc class × sample structure × universe) | DSR readings now within-stratum only; cross-stratum DSR-rank explicitly disclaimed |

## The headline Ledoit-Wolf result

ΔSR is the annualized Sharpe difference of each arc minus DCA-canonical
on the date-aligned common window. 95% CI from 2000-iteration
stationary block bootstrap (Politis-Romano 1994), block length
`n^(1/3)`, studentized inversion per Ledoit-Wolf 2008.

### Arcs that include 0 (statistically indistinguishable from DCA)

| arc | n_overlap | ΔSR ann | 95% CI |
|---|---:|---:|---|
| vol-v3-dolthub-c0 (reference) | 29 | +1.50 | [−0.16, +3.45] |
| vol-v3-dolthub-c200 (deployable) | 29 | +1.17 | [−0.38, +3.06] |
| relational-analog-cross-ticker | 1241 | +0.24 | [−0.32, +0.80] |
| gate-v0 (excess vs EW) | 4680 | +0.21 | [−0.38, +0.82] |
| dca-basket-search-winner-4etf | 5232 | +0.17 | [−0.39, +0.73] |
| factor-5d-LO-skip1 | 936 | −0.32 | [−0.96, +0.29] |
| factor-LO-baseline-20d | 234 | −0.49 | [−1.17, +0.30] |
| pairs-v0 | 4680 | −0.44 | [−1.04, +0.21] |
| low-vol-bab-LS | 249 | −0.66 | [−1.29, +0.06] |
| lie-shape-knn-LS-phase2 | 57 | −1.04 | [−2.34, +0.35] |

### Arcs that exclude 0 on the negative side (significantly worse than DCA)

| arc | n_overlap | ΔSR ann | 95% CI |
|---|---:|---:|---|
| momentum-12-1-LS | 249 | **−0.74** | [−1.28, **−0.12**] |
| factor-LS-baseline-20d | 234 | **−0.84** | [−1.45, **−0.11**] |
| factor-5d-LS-skip1 | 936 | **−1.09** | [−1.77, **−0.44**] |
| lie-shape-knn-LS-wide | 94 | **−1.62** | [−2.59, **−0.57**] |

These four arcs are statistically distinguishable from DCA at the 95%
level — and they're worse. This confirms the cross-sectional null
findings documented across the factor app's findings: long-short
cross-sectional books on this universe are net-of-cost negative.

## Why the rewrite was necessary

The three background agents converged independently:

| agent brief | finding |
|---|---|
| [`.research-walkforward-apples-to-apples.md`](../../../../.research-walkforward-apples-to-apples.md) | DSR ≠ cross-arc rank key. Ledoit-Wolf (2008) is the literature-canonical apples-to-apples test. |
| [`.research-findings-assumptions-audit.md`](../../../../.research-findings-assumptions-audit.md) | `sharpe_std_ann=0.25` calibrated on factor arcs only, applied universally; `commission_bps=10` unrealistic for vol options; `n_trials` reconstructions hand-rolled. |
| [`dca-vol-ensemble-optuna`](dca-vol-ensemble-optuna.md) | Even the joint DCA × vol-overlay Optuna search couldn't distinguish basket configurations at n_val=13 — the ranking framework itself was at-or-below the noise floor. |

The synthetic test in [Step 1 of the migration TODO](../TODO/ladder-methodology-rewrite.md)
showed `sharpe_std_pp = sharpe_std_ann/sqrt(ppy)` is mechanically
correct, BUT the workspace 0.25 double-counts the null estimation
noise component. Variance decomposition recovered the structural-only
residual at 0.072 ann.

## Sensitivity table — vol-v3 realistic options friction

The realistic-cost grid documents how deflated-t responds to options
friction. Each row uses the same stream with a different per-fired-
rebal cost in vol points (`commission_bps × 1e-4`):

| commission_bps | ann Sharpe (full panel) | ann Sharpe (fired only) | deflated-t (n_trials=12) |
|---:|---:|---:|---:|
| 0 (reference) | +2.82 | +6.18 | +4.08 |
| 50 | +2.74 | +5.75 | +3.81 |
| 100 | +2.64 | +5.32 | +3.66 |
| **200 (deployable)** | **+2.41** | **+4.46** | **+3.43** |
| 400 (worst case) | +1.74 | +2.75 | +2.66 |

Note: even at the deployable 200bps friction, vol-v3's deflated-t
remains above DCA's +2.02 *within its own stratum*. The Ledoit-Wolf
CI is what kills the "vol-v3 beats DCA" claim — sample length n=29
is too short for the +1.17 lift to be distinguishable from noise at
95%.

## The deployment implication

The DCA + vol-v3 ensemble (commits `9c404fe`, `b187502`, the `ss-vol
ensemble` CLI) remains the deployable system. The methodology
rewrite **does not invalidate the ensemble**, but it does clarify
the claim:

- **DCA alone** has a statistically defensible Sharpe (+0.69 ann, n=5232 daily, deflated-t +2.02). It is the only arc with statistical backing.
- **DCA + vol-v3 ensemble** has a *plausibly higher* Sharpe, but the 95% CI on the lift includes 0. Deployment is a research-hypothesis-test deployment, not a confirmed alpha lift.
- **The regime-tailwind caveat from vol-v3-DoltHub** ([`vol-v3-dolthub-oos`](vol-v3-dolthub-oos.md)) compounds: the 2023-08 → 2026-04 sample is all calm-bull short-vol; no vol crisis in window.

## What I'd recommend for new arcs

1. **Pre-register `sharpe_std_ann`** per-arc-class before any sweep
   runs. Persist all trial Sharpes (not just top-10) so future
   recalibration is possible.
2. **Apples-to-apples comparison goes through Ledoit-Wolf**, not DSR
   magnitude. New arcs report ΔSR vs DCA-canonical (or a stratum-
   specific baseline) with 95% CI.
3. **DSR stays as the within-arc gate** — does the arc's own selection
   produce a Sharpe distinguishable from coin flips at its own search
   cost? Yes/no answer per arc.
4. **Stratify** new arcs into one of the seven existing strata when
   adding to the ladder; cross-stratum claims require justification.

## Master walk-forward log

[Cross-arc ranking — Ledoit-Wolf vs DCA section](../leaderboard.md#cross-arc-ranking--primary-ledoit-wolf-Δsr-vs-dca)
of the leaderboard documents both columns (Ledoit-Wolf primary, DSR
within-stratum secondary). The migration's five-step sequence is
recorded in [`TODO/ladder-methodology-rewrite.md`](../TODO/ladder-methodology-rewrite.md)
(status: done). The audit briefs that drove this work are at
`.research-walkforward-apples-to-apples.md`,
`.research-findings-assumptions-audit.md`, and
`findings/dca-vol-ensemble-optuna.md`.
