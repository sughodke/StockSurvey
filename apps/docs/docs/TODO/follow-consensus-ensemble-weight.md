# Follow-consensus β-hedged ensemble weight — pre-registered

**Status: `pending` — pre-registration locked before the eval runs.**
Sister arc to [`follow-consensus-arm`](follow-consensus-arm.md) (now
[`confirmed-OOS`](../findings/follow-consensus-arm.md), shipped as
ensemble-only). This arc adjudicates a single decision: **what fraction
of the live risk budget — if any — should the SPY-β-hedged follow-
consensus stream take inside the canonical (DCA + 2 × vol-v3) live
stack?**

---

## Why this pre-reg exists

The 2026-05-28 follow-consensus finding shipped the consensus arm
under explicit ensemble-only language:

> "Deploy ... as an **ensemble constituent paired with a SPY beta-
> hedge**, NOT as a standalone single-arm bet."
> — `findings/follow-consensus-arm.md`

But the finding did **not** lock the ensemble weight. The post-2020
ranking work and [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md)
established that **DCA + 2 × vol-v3** is the canonical 2-leg stack
(Ledoit-Wolf ΔSharpe +1.16 → +1.90 vs DCA-only, CI excludes 0). The
open question is what `follow_w` does to that 2-leg baseline when
added as a third leg.

The standalone consensus arm has α defl-t **+1.37** (under the +2.0
brief-bar) and a wide fold-3 CI **[−18, +26] pp/yr**. The ensemble
case is therefore **not** "this is the next standalone strategy" — it
is "this is a low-correlation alpha stream worth a small slice if the
joint Sharpe lifts and max-DD doesn't blow up."

Lock the bar **before** running the sweep so the result is one of
{confirmed-OOS, partial-OOS, confirmed-null} per the locked weight
grid rather than a post-hoc "best-fit" exercise.

---

## Mechanism — the steel-man

Why this could clear the bar (not why it won't):

1. **Source-of-alpha orthogonality.** vol-v3 harvests the
   IV-vs-realized gap on high-OI underlyings; DCA is passive multi-
   asset beta capture. The consensus arm is a *flow-driven* signal
   (member-level disclosure flow as a proxy for inside-information
   diffusion). The three return streams target structurally
   different premia, so a low pairwise correlation is the prior, not
   the surprise.

2. **β-hedge already neutralizes the bull-regime drag.** The
   unhedged consensus arm underperformed SPY on raw return during
   the 2023-24 bull because the basket runs ~0.8 β to SPY. The
   1× SPY-short overlay strips the systematic component, leaving
   the per-name selection edge. The unhedged arm cannot be
   ensemble-added against a passive-beta DCA basket; the hedged
   arm can.

3. **Fold-3 OOS point estimate is strong.** α +5.13 pp/yr / Sh
   +0.86 / 3/3 pos-Q on an unseen post-search fold. The wide CI is
   a sample-length problem (n=198), not a point-estimate problem.
   At even modest weight (5%), the joint Sharpe should pick up some
   of the +5pp/yr α without inheriting the full CI width as
   ensemble noise.

If the joint ΔSharpe ≥ +0.10 and max-DD does not blow out, the
ensemble case clears. If it doesn't, the standalone-CI-width problem
is killing the ensemble case too and we wait for more data.

---

## Search space (locked)

Single lever: `follow_w` ∈ **{0.000, 0.025, 0.050, 0.075, 0.100,
0.150, 0.200}** (7 grid points, including the 0 baseline).

For each `follow_w`, the live return stream is:

```
r_total[t] = (1 - follow_w) × r_dca_vol3[t]
          + follow_w     × r_follow_consensus_β_hedged[t]
```

where:

- `r_dca_vol3[t]` is the canonical 2-leg stream (DCA + 2 × vol-v3),
  matching the deployment recipe at
  [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md).
- `r_follow_consensus_β_hedged[t]` is the consensus arm minus
  SPY × β_t (rolling-60d OLS β estimated from the trailing window;
  re-estimated at each rebal).

`follow_w` is the fraction of **risk budget**, not raw capital — the
3rd leg's daily return is added to the (1-w)-scaled canonical stream
exactly as the vol-v3 stream was added to DCA in the post-2020
ensemble work.

n_trials for deflation = **7** (the grid above; not Optuna).

---

## Datasets + windowing (locked)

| field | value |
|---|---|
| span | 2019-01-01 → 2025-10-16 (3-fold: 2019-21, 2022-24, 2025-YTD) |
| DCA basket | 13-ETF Phase 4d (9 SPDR sectors + TLT/IEF + GLD/DBC), 1/13 EW, 80-day cadence floor + 5% drift |
| vol-v3 sleeve | top-K-gated short-vol with v3 regime gate, scale=2× (matches sizing finding) |
| consensus arm | h=30, k=10, frequency-filter, filed+1, EW long, SPY 1× β-hedge with rolling-60d β |
| friction | DCA 10 bps `\|Δw\|`, vol-v3 measured c_options_bps, consensus 10 bps `\|Δw\|` |
| folds | fold-1 train 2019-21 / fold-2 val 2022-24 / **fold-3 OOS 2025-01 → 2025-10-16** |
| benchmark | DCA-only and (DCA + 2 × vol-v3) — two baselines so the lift attribution is unambiguous |
| metric | block-Sharpe (20-day non-overlapping blocks), Ledoit-Wolf studentized ΔSharpe CI vs (DCA + 2 × vol-v3), max-DD, defl-t |

---

## Pre-locked verdict bar

The grid produces a `follow_w*` that maximises the **fold-1+2 joint
Sharpe** (in-search). The verdict is locked on the **fold-3 OOS**
behaviour at that `follow_w*`:

| condition | verdict |
|---|---|
| OOS ΔSharpe ≥ +0.10 vs (DCA + 2 × vol-v3) AND Ledoit-Wolf 95% CI excludes 0 AND max-DD ≤ 1.2 × baseline max-DD | **confirmed-OOS** — ship the 3-leg ensemble at `follow_w*` |
| OOS ΔSharpe ≥ +0.05 but CI does not exclude 0 OR max-DD between 1.2× and 1.5× baseline | **partial-OOS** — hold for fold-3 data refresh; do not ship |
| OOS ΔSharpe < +0.05 OR max-DD > 1.5 × baseline OR `follow_w*` = 0 in-search | **confirmed-null** — the consensus arm doesn't justify ensemble inclusion even at small weight; archive |

Sample-size honesty: if the fold-3 stationary-bootstrap CI on the
3-leg ΔSharpe is wider than ±0.30, **the verdict is automatically
downgraded one tier** (confirmed → partial, partial → null). A wide
CI on an n=198 fold cannot license a deployment claim even if the
point estimate clears.

---

## Reproducibility constraint — the binding sample-length issue

Fold-3 ends **2025-10-16** (the xlsx end date at finding time). For
the standalone arm, the binding constraint to tighten the +1.37
defl-t to +2.0 is "extend fold-3 from 198 days to ~500+ days." That
constraint binds identically here:

**This pre-reg is `pending` on a Quiver xlsx refresh.**

The eval should be deferred until the disclosure source extends past
~2026-09 (≈500 fold-3 days). Running it now would re-use the same
n=198 sample that's already too short for the standalone case —
adding a fourth lever (weight grid) on the same fragile fold compounds
the sample-size problem. The +5 deflation correction on 7 weight
trials is non-trivial against an already-wide CI.

If the refresh extends fold-3 to ≥ 400 days, run the locked grid and
publish the verdict. If the refresh never arrives, this arc stays
`pending` indefinitely and the consensus arm is deferred from the
live stack.

---

## Why not just ship at `follow_w` = 0.05 with no eval?

Two reasons it would be a mistake:

1. **The 2-leg (DCA + 2×vol-v3) baseline is the load-bearing
   canonical recipe.** Adding a 3rd untested leg risks **lowering**
   the joint Sharpe if the consensus arm's volatility dominates its
   contribution. A 5% slice of a strategy with σ ≈ 25% adds ~1.3%
   to joint σ — non-trivial against a baseline σ of ~10%.

2. **Max-DD is the hard live-trading risk constraint.** The
   consensus arm's fold-3 max-DD was not reported in the finding
   (only the standalone Sharpe and α). A 3-leg max-DD that exceeds
   the baseline by >20% changes the kill-switch math. The eval
   must compute this; eyeballing won't.

---

## Out of scope

- **Non-zero correlation with vol-v3.** This arc assumes the
  consensus arm and vol-v3 are approximately independent. If the
  measured 60-day rolling correlation exceeds 0.3 in any fold, that
  is a separate finding that should be recorded but does not change
  the verdict bar (the joint Sharpe captures the dependence).
- **Optuna weight search.** The 7-point grid is enough resolution
  given the fold-3 sample length. An Optuna sweep here would burn
  deflation budget on noise.
- **Multi-leg vol sleeve scale × follow_w joint search.** The
  vol-v3 scale was locked to 2× by
  [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md). Re-opening that
  lever here would re-open a closed arc.

---

## Acceptance criteria

1. Driver script `apps/follow/scripts/run_ensemble_weight_sweep.py`
   computes the 7-point grid against the two baselines, reports the
   in-search `follow_w*` and the fold-3 OOS ΔSharpe / CI / max-DD.
2. The verdict label lands in `apps/docs/docs/leaderboard.md` per the
   locked table above.
3. Finding (if confirmed-OOS or partial-OOS) writes to
   `apps/docs/docs/findings/follow-consensus-ensemble-weight.md` with
   the deployed `follow_w*` and the joint-stream metrics.
4. If confirmed-OOS, update
   [`vol-v3-sleeve-sizing.md`](vol-v3-sleeve-sizing.md) and the
   `apps/dca` live stack documentation to reference the 3-leg recipe.

---

## Pointers

- Sister arc that opened the door: [`follow-consensus-arm`](follow-consensus-arm.md) → confirmed-OOS / ensemble-only.
- Canonical 2-leg recipe: [`vol-v3-sleeve-sizing`](vol-v3-sleeve-sizing.md).
- Verdict vocab: [`leaderboard.md#verdict-labels`](../leaderboard.md#verdict-labels).
- Data-refresh dependency: Quiver-bundled `Congressional Trades.xlsx` in `.congress-cache/`; refresh weekly per the consensus finding.
