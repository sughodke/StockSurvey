---
tags:
  - cfr
  - macro
  - gate
  - confirmed-null
  - arc-closer
---

# CFR Phase 4d + window-level VIX meta-gate — final close-out (confirmed-null)

!!! note "Update 2026-05-14 — denominator artifact + alternative gates falsified"

    The 2026-05-14 sensitivity follow-up
    ([`cfr-sensitivity-followup`](cfr-sensitivity-followup.md)) shows
    that this row's "positive in 1/5 windows → FAIL" finding was a
    **denominator artifact**: the 4 closed-gate windows contributed
    alpha=0 by construction, so they should never have been counted
    in the positive-window denominator. Under always-deploy CFR
    (no gate), positive-α-windows is **3/5** across the entire
    friction grid {10,15,25,35,50} × {2,5,10} bps. The
    underlying CFR alpha is closer to `partial-OOS` than
    `confirmed-null` (alpha real, signed positive, below +0.10
    threshold at every parameterization). The two audit-proposed
    alternative gates were also tested and both falsified: 60d /
    126d VIX gates never fire; 90d VIX worsens alpha; in-universe
    pairwise-correlation dispersion fires anti-correlated to CFR's
    alpha pattern (−0.115). The deployment decision (use DCA)
    is **unchanged**, but the "bot is fully dead" framing was
    overstated. See the sensitivity follow-up for the bracketed
    description.

**Operational rule (extracted):** *Memory-heavy regime gates (rolling
multi-quarter medians) fight you across regime boundaries.* The 1-year
rolling-median VIX gate stays "closed" for ~3 years after a stress
event because the median is inflated by the stress itself, gating the
bot off during exactly the early-recovery regimes where it would have
been most useful.

This is the final close-out experiment for the CFR arc. After
[`cfr-vs-dca-realistic`](cfr-vs-dca-realistic.md) downgraded
Phase 4d's raw +0.056 alpha to net +0.015, the macro v1b VIX gate
was the one remaining unexplored composition that might have
recovered enough alpha to clear the +0.10 paper-trade threshold. It
didn't. **The bot is fully dead.**

## Experiment

**Hypothesis:** macro v1b's window-level VIX-above-1y-rolling-median
gate composes with Phase 4d. Specifically, w2 (2016-19, the calm-bull
window where CFR loses 0.5 Sharpe to EW) gets correctly suspended,
lifting net alpha above the +0.10 threshold.

**Design** (`apps/cfr/scripts/macro_gated_phase4d_eval.py`):

- For each Phase 4d val window's `val_start` date:
  - Compute VIX value at `val_start - 1` (info-causal, no future leak)
  - Compute 1-year rolling median of VIX as of `val_start - 1`
  - `gate_open = (VIX > 1y median)` → deploy CFR Phase 4d for the window
  - `gate_closed` → defer to multi-asset EW for the window
- Apply realistic friction (same as `cfr-vs-dca-realistic`):
  - CFR-deployed window: 50 bps/yr drag
  - EW-deployed window: 5 bps/yr drag
- Compare per-window net Sharpe to DCA (always EW + 5 bps/yr drag)

**Pre-registered cuts (committed before running):**

| Verdict | Net alpha vs DCA | Positive windows | Action |
|---|---:|---:|---|
| **PASS** | ≥ +0.10 | ≥ 4/5 | Rebuild `apps/dca` as gated CFR-OR-EW; paper-trade for 1 quarter |
| **MARGINAL** | [0, +0.10] | 3/5 | Effect exists but doesn't clear threshold; archive and stop |
| **FAIL (confirmed-null)** | ≤ 0 | ≤ 2/5 | Bot is fully dead; DCA stays canonical |

## Result

| Window | val_start | VIX@start | 1y median | Gate | Deployed | Net Sharpe | DCA | Alpha |
|---:|---|---:|---:|---|---|---:|---:|---:|
| w0 | 2010-03-01 | 19.26 | 25.41 | **CLOSED** | EW | +0.941 | +0.941 | +0.000 |
| w1 | 2013-04-08 | 13.19 | 16.32 | CLOSED | EW | +0.642 | +0.642 | +0.000 |
| w2 | 2016-05-11 | 14.69 | 16.05 | CLOSED | EW | +0.996 | +0.996 | +0.000 |
| w3 | 2019-06-18 | 15.15 | 15.37 | CLOSED | EW | +0.687 | +0.687 | +0.000 |
| w4 | 2022-07-22 | 23.03 | 21.93 | **OPEN** | CFR | +0.998 | +0.736 | **+0.263** |
| **mean** | | | | | | **+0.853** | **+0.800** | **+0.053** |

**Verdict:** [`confirmed-null`](../leaderboard.md#verdict-labels). Mean
alpha +0.053 is in the MARGINAL band by alpha alone, but positive in
only **1/5 windows** trips the FAIL gate on the positive-windows
criterion (`≤ 2/5 → FAIL`).

## What the gate did right vs wrong

- **Right call in w4** (+0.263 net alpha after friction). Post-Fed-pivot
  2022 — VIX 23 vs 1y median 22 → OPEN → CFR captured the cross-asset
  regime alpha that exists when stocks AND bonds are simultaneously
  stressed.
- **Right call in w2** (avoided −0.508 raw alpha). The 2016-19 calm
  bull market was CFR's worst window; deferring to EW saved us.
- **Wrong call in w0** (threw away +0.422 raw alpha). March 2010 VIX
  was 19 vs 1y median 25 → gate said "regime is now calm" because the
  rolling median was inflated by GFC. But w0 was actually CFR's BEST
  window (post-GFC recovery, strong cross-asset rotation). **The gate
  used a memory-heavy criterion that mis-classified the regime
  immediately after the largest stress event in the sample.**
- **Neutral in w1, w3** (CFR and EW within 0.05 Sharpe regardless;
  gate decision didn't move the needle).

## Mechanism — why memory-heavy gates fight us

The 1-year rolling median is, by construction, a smoothed estimate
of the recent regime. After a high-stress event (GFC, 2020 COVID
crash), the median stays elevated for ~12 months even after the acute
shock has passed. During exactly this transition period — early
recovery, stocks rebounding, bonds rallying, cross-asset dispersion
high — CFR Phase 4d posts its strongest results (w0 alpha +0.422,
w4 alpha +0.263). The gate suspends CFR in these regimes because
"current VIX < memory of recent VIX", but operationally the regime
has already shifted to "recovery with rotation".

A shorter-memory criterion (e.g., 60-day median, or VIX percentile
over a 90-day rolling window) would have classified w0 as OPEN. But
shorter-memory criteria are themselves more reactive to noise — and
the macro v1b finding's +0.215 z-score lift specifically came from
the 1-year-median formulation. We can't free-tune the lookback to
fit Phase 4d without overfitting on 5 windows.

## Composition of arcs — what we now know

Three independent arcs converge on the same operational answer:

1. **Phase 4d raw alpha (+0.056) doesn't survive realistic friction**
   ([`cfr-vs-dca-realistic`](cfr-vs-dca-realistic.md)) → net +0.015
2. **Phase 4a bar-level VIX gate destroys Phase 3**
   ([`cfr-phase4`](cfr-phase4.md)) → wrong granularity; suspends 57%
   of bars indiscriminately
3. **This experiment: window-level VIX gate composes badly with
   Phase 4d** → saves 1/5 windows but throws away another 1/5 of
   equal magnitude (mean wash)

After 10 CFR phase variants + 1 macro composition test, **no current
configuration clears the realistic-alpha threshold on this universe**.

## What this changes

- **DCA stays as canonical live strategy.** No hybrid rebuild.
- **CFR arc is fully closed.** The regret-net checkpoint
  (`Output/cfr-phase4d.json`) is preserved; the macro-gated extension
  (`Output/cfr-phase4d-vix-gated.json`) is preserved as a documented
  null. If a future regime change breaks EW dominance (e.g.,
  sustained 5%+ real rates breaking 60/40), the cross-asset CFR
  scaffolding is ready to re-deploy on a different universe.
- **Operational rule preserved**: any future regime gate must be
  tested for memory-window robustness. A 1-year-median gate that
  works on gate/pairs/vol's 6-window walk-forward (where the
  windows happen to span both calm and stressed regimes) does NOT
  necessarily transfer to Phase 4d's 5-window walk-forward.

## Master walk-forward log

Add a leaderboard row at 2026-05-13 for the macro-gated re-eval of
`Output/cfr-phase4d-vix-gated.json`. Verdict label:
[`confirmed-null`](../leaderboard.md#verdict-labels). This is the
final row in the CFR arc.
