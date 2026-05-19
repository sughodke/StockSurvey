# Factor short-horizon edge — microstructure / cost stress (skip-1)

> **RESOLVED 2026-05-19 — `partial-OOS` (part-real, part bid-ask-bounce).**
> A 1-day implementation lag (`forward_skip=1`) ~halves the 5d edge:
> mean val IC +0.0212→**+0.0114**, 5/6 windows, net Sh +0.529. ≈46% of
> the headline IC was non-tradable same-bar mean reversion (bid-ask
> bounce), but it does **not** collapse — +0.0114 stays above the
> pre-registered reversed-kill (+0.0106) and 5/6 windows hold → the
> pre-reg **partial** band exactly. 10d retains a higher *fraction*
> (64% vs 54%) but lower absolute (+0.0076, 4/6); neither collapses, so
> the signal is not *entirely* microstructure. skip-0 cells reproduced
> 2026-05-18 to ±3×10⁻⁵ (regression anchor — `forward_skip` is
> bit-safe). Decay: skip-1 5d per-window
> `[0.026,0.013,0.013,0.006,−0.005,0.016]` — recent edge ~+0.005–0.016
> with a negative window, marginal. **Net:** the deployable read is "5d
> with explicit cost/turnover control for a modest ~+0.011 IC",
> *gated* on whether that clears the live cost model — not the +0.0212
> headline. Reinforces the standing frame (higher-EV = orthogonal
> prediction problems / novel data, not more factor-return variants).
> **Spawned next (only if the modest edge is worth pursuing):** gross
> decile-spread/turnover decomposition → precise commission break-even;
> then universe-generalization. Evidence: 3 leaderboard rows
> `2026-05-19`; finding
> [`factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md#microstructure--skip-1-follow-up).
> Pre-registration retained below as written.

**Verdict → next-experiment chain (as pre-registered):**

- If the 5d indicator-grid edge **survives a 1-day implementation lag**
  (skip-1 mean val IC ≥ +0.012, ≥5/6 windows) → it is a *tradable*
  predictable cross-sectional move, not bid-ask bounce. Promote the
  cadence pivot; run universe-generalization + decay stratification.
- If skip-1 **collapses** the 5d IC (≥50% drop, < +0.0106, or < 4/6
  windows) → the 5d advantage is largely microstructure
  (`reversed-OOS` on 5d deployability). Fall back to the longest
  horizon that survives skip-1 (likely 10d); if 10d also dies the
  entire short-horizon signal is non-tradable.
- If skip-1 lands **between** (+0.0106 ≤ IC < +0.012, or exactly 4/6) →
  `partial-OOS`: part-real, part-microstructure; deployable only with
  explicit cost/turnover control.

## Why this gates everything

The [`factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md)
finding closed `confirmed-OOS` for the indicator grid at
`rebal_days=5` (mean val IC **+0.0212, 6/6 windows**) — but flagged the
deployability caveat explicitly: short-horizon cross-sectional reversal
is a well-arbitraged anomaly whose measured IC is classically
contaminated by **bid-ask bounce** (a name that closed near its bid
"reverts" up the next day with no tradable move). The IC is
commission-free so it is *not* a commission artifact, but it *can* be a
microstructure artifact. Until that is ruled out, universe
generalization (#2) and decay stratification (#3) are premature — a
non-tradable signal does not become tradable on another universe.

## Hypothesis (falsifiable)

The 5d (and 10d) cross-sectional indicator-grid edge is a predictable,
tradable cross-sectional move that survives a realistic 1-day execution
lag and realistic transaction costs. Null: a 1-day skip removes most of
the 5d IC (the reversal is bid-ask bounce).

## Test design

- **Mechanism — skip-1 implementation lag.** Score on features through
  `close(t)` (unchanged) but realize the return a real trader could
  capture by entering at `close(t+1)` instead of `close(t)`: the
  forward target becomes `log p[t+1+H] − log p[t+1]` and the held block
  return spans `[t+2, t+1+H]`. Implemented as a `forward_skip:int=0`
  offset threaded through `forward_log_returns` → `precompute_inputs`
  (block loop + `align_tickers_at_rebal` edge `rebal_idx+H+skip<D`) →
  `train_scorer_walkforward`. **skip=0 must stay bit-identical** to the
  2026-05-18 run (regression guard in the smoke).
- **Arms:** indicator grid only (the representation question is closed
  `confirmed-null` — this is purely about the *grid* short-horizon
  edge's realness). `rebal_days ∈ {20, 10, 5}` × `forward_skip ∈
  {0, 1}` = 6 cells. Same `factor-narrow` (297), year-comparable block
  scaling, linear head, n_steps=200, the 2026-05-18 operating
  condition. skip-0 cells reproduce the just-recorded rows (anchor).
- **Decision metric: mean val IC** (commission-free; the established
  guardrail). skip-1 vs skip-0 ΔIC per horizon + window-consistency.
- **Cost break-even (analytic, reported not gating):** from the
  realized per-block top-decile−bottom-decile return spread and
  turnover, compute the `commission_bps` at which net 5d (skip-1)
  Sharpe = net 20d Sharpe and = 0. Break-even < ~15–20 bps ⇒ fragile.
- **Decay stratification (free, from existing + new per-window IC):**
  the skip-0 r5 per-window IC was `[0.043,0.029,0.017,0.010,0.013,
  0.016]` (calendar order, 2000s→2020s) — monotone-ish decay. Report
  the skip-1 per-window sequence the same way; the *live* edge is the
  latest-window IC, not the 26-year mean.

## Pre-registered bands (5d, vs skip-0 +0.0212 / 6-of-6)

| skip-1 5d mean val IC | windows | verdict |
|---|---|---|
| ≥ +0.012 | ≥ 5/6 | survives — `confirmed-OOS` upheld, proceed to universe/decay |
| +0.0106 – +0.012 | ≥ 4/6 | `partial-OOS` — part-microstructure |
| < +0.0106 | any | `reversed-OOS` on 5d deployability — bid-ask bounce |
| any | < 4/6 | `reversed-OOS` on 5d deployability |

Cross-check: 10d should be *more* skip-1-robust than 5d (longer
horizon, smaller microstructure fraction). 10d collapsing too ⇒ the
whole short-horizon signal is non-tradable; 10d surviving while 5d dies
⇒ the deployable horizon is 10d.

## Next per outcome

- *Survives* → universe-generalization run (same driver, second
  universe; mega-cap-specificity prior from
  [`relational-universe-shift`](../findings/relational-universe-shift.md))
  + decay stratification; then a cadence-pivot deployment decision.
- *Reversed/partial* → fall back to the skip-1-surviving horizon; if
  none, the factor cross-sectional-return arc is `confirmed-null`
  end-to-end (representation *and* tradable horizon) → pivot to an
  orthogonal prediction problem per the
  [`confirmed-null` playbook](../leaderboard.md#verdict-labels).

## Pointers

- Parent (resolved): [`TODO/factor-shorthorizon-representation`](factor-shorthorizon-representation.md),
  finding [`factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md).
- Commission-confound precedent: leaderboard `2026-05-04` quarterly
  `reversed-OOS` + the leaderboard's closing commission-geometry note.
- Driver: `apps/factor/scripts/modal/train_shorthorizon_repr.py`
  (extended with `--forward-skip-grid`).
