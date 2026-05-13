---
tags:
  - cfr
  - dca
  - deployment
  - friction
  - realistic-alpha
---

# CFR Phase 4d (active) vs multi-asset EW DCA (passive) — realistic-friction head-to-head

!!! note "Update 2026-05-14 — friction bracketing (audit-driven sensitivity check)"

    The +0.015 net alpha headline below is the **worst-case corner**
    of the friction grid, not a best estimate. The 2026-05-14
    sensitivity follow-up
    ([`cfr-sensitivity-followup`](cfr-sensitivity-followup.md))
    swept CFR friction over {10, 15, 25, 35, 50} bps/yr × DCA
    friction over {2, 5, 10} bps/yr and observed **mean alpha swings
    5× across the grid: +0.015 → +0.056**. No friction combination
    crosses the +0.10 deployment threshold, so the deployment
    decision (use DCA) is unchanged. But the honest description is
    "alpha is real and positive at every friction level, below
    deployment threshold at every parameterization" rather than the
    single-point "+0.015 ≈ noise" framing below. See the sensitivity
    follow-up for the per-grid-cell breakdown.

**Operational rule (extracted):** *Apply realistic deployment friction
before declaring a backtest-positive strategy deployable.* Phase 4d's
backtest alpha of **+0.056 Sharpe** vs passive EW collapses to a net
**+0.015 Sharpe** once realistic deployment costs (slippage on
multi-asset ETF spreads, short-term tax inefficiency from 11× annual
turnover, operational labor) are subtracted from the bot side and the
EW side is given its own (much smaller) realistic friction. Below the
+0.15 paper-trade threshold from
[`passive-ew-benchmark`](passive-ew-benchmark.md), and below DCA on
worst-window Sharpe.

The deployable strategy is `apps/dca` — a 13-asset equal-weight basket
on the same Phase 4d universe, rebalanced quarterly, ~50 lines of
Python, no AI/Modal/checkpoint persistence.

## TL;DR for the deployment decision

| Metric | CFR Phase 4d (bot) | Multi-asset EW DCA |
|---|---:|---:|
| Mean per-window val Sharpe (raw) | +0.861 | +0.805 |
| Mean per-window val Sharpe (net of friction) | **+0.815** | **+0.800** |
| Worst-window Sharpe (regime stress) | +0.434 | **+0.642** |
| Realistic alpha vs EW | **+0.015** | — |
| Operational tax | broker reconciliation, kill-switch monitoring, tax-lot tracking, model-vs-live drift detection | quarterly rebal script, ~10 min/quarter |
| Full-panel Sharpe (2005-2025, incl GFC) | not measured | +0.669 |
| Bias-corrected Sharpe (post bond tailwind) | ~+0.71 (extrapolated) | ~+0.55 |

## Friction model

Both sides start from the 10 bps commission already in the backtest.
Realistic deployment add-ons:

- **CFR bot — 50 bps/yr.** Phase 4d turnover from `cfr_avg_turnover` ≈
  47% per rebal × 250/20 rebals/yr = ~5.9× annual turnover round-trip
  ≈ **11.8× one-way turnover/yr**. At 5 bps slippage on multi-asset
  ETFs (DBC and TLT have wider spreads than SPY) that's ~30 bps drag.
  Plus short-term tax inefficiency: high-turnover strategies in a
  taxable account convert long-term gains (15-20% rate) to short-term
  (ordinary income, ~30-37%), a structural ~15-20 bps drag depending
  on bracket. Plus operational labor at the user's hourly value.
- **EW DCA — 5 bps/yr.** Quarterly rebal × ~50% one-way turnover/yr
  × 5 bps spread on ETFs ≈ 2.5 bps drag, rounded up to 5 bps. ETF
  in-kind redemption is tax-efficient; mostly long-term gains.

Sharpe drag is computed as `(annual_drag_bps / 10000) / annual_vol`
at each window's vol. At the 13-asset basket's ~13.3% annual vol the
CFR drag amounts to ~0.04 per-window Sharpe, EW drag ~0.004.

## Per-window head-to-head (Phase 4d's 5 val windows)

| Window | val period | CFR raw | CFR net | EW raw | EW net | CFR−EW (net) |
|---|---|---:|---:|---:|---:|---:|
| w0 | 2010-03 → 2013-04 | +1.367 | +1.327 | +0.945 | +0.941 | **+0.386** |
| w1 | 2013-04 → 2016-05 | +0.622 | +0.569 | +0.647 | +0.642 | −0.074 |
| w2 | 2016-05 → 2019-06 | +0.495 | +0.434 | +1.002 | +0.996 | **−0.562** |
| w3 | 2019-06 → 2022-07 | +0.780 | +0.750 | +0.690 | +0.687 | +0.063 |
| w4 | 2022-07 → 2025-08 | +1.041 | +0.998 | +0.740 | +0.736 | +0.263 |
| **mean** | | +0.861 | **+0.815** | +0.805 | **+0.800** | **+0.015** |

**w2 is the binding constraint.** The 2016-2019 "everything works
passive" era — calm bull market, low cross-asset dispersion — is
where any active rotation strategy gets penalized. CFR's worst window
(+0.434) loses to DCA's worst window (+0.642) by **0.21 Sharpe**;
this is exactly when you most want a stable Sharpe floor, and the
bot delivers the opposite.

## EW Sharpe across regimes (full-panel discipline)

The +0.805 EW Sharpe in the head-to-head is val-only (2010-2025) —
GFC was in Phase 4d's train fold, not val. Computing EW over the
**full available panel** including GFC reveals a more honest number:

| Period | Sharpe | CAGR | Max DD | Regime |
|---|---:|---:|---:|---|
| **FULL 2005-2025** | **+0.673** | **+8.4%** | **−40.7%** | (GFC dominates the DD) |
| 2005 → Q3 2007 | +1.10 | +10.0% | −7.4% | pre-GFC bull |
| **2007-Q4 → 2009-Q1** | **−0.79** | **−21.0%** | **−40.7%** | GFC crash |
| 2009 → 2011 | +1.21 | +18.2% | −11.0% | GFC recovery + Euro |
| 2012 → 2014 | +1.42 | +12.1% | −6.1% | taper / calm grind |
| 2015 → 2018 | +0.52 | +4.4% | −13.4% | Trump rally + Q4 2018 |
| 2019 → 2020-Apr | +0.58 | +10.0% | −27.4% | COVID crash + V-shape |
| 2020-May → 2021 | +2.09 | +27.1% | −6.1% | QE rip |
| 2022 only | n/a | — | −15.6% | 60/40 kill year (rates+stocks both down) |
| 2023 → 2025 | +1.19 | +12.2% | −11.4% | rate plateau |

## What is NOT in this backtest

The basket has been tested against:

- ✅ GFC (deflationary credit shock)
- ✅ COVID (acute liquidity shock + V-shape recovery)
- ✅ 2022 (mini-stagflation, stocks+bonds both down)

But has NOT been tested against:

- ❌ **1970s stagflation** (1973-74 stocks −50%, gold +200%, bonds
  −20% real). This is the regime where TLT/IEF actively *hurt* and
  only GLD saves you. Sector ETFs didn't exist; there's no clean
  reconstruction.
- ❌ **1980-82 Volcker recession** (15% Fed funds → bonds savaged →
  gold collapsed after 1980 peak)
- ❌ **1929-1932 Great Depression** (−86% on stocks; deflationary)

The basket is well-tested for the **post-Volcker monetary regime**
(1985-2025) where the dominant macro framework was "Fed cuts rates
in crisis → bonds rally → stocks recover". If we exit that regime
(sustained inflation, financial repression, currency crisis), the
historical Sharpe is not predictive. The +0.673 full-panel Sharpe is
also upward-biased by the 1980-2020 bond bull market that physically
cannot repeat from 4-5% rates today; bias-corrected Sharpe is closer
to **+0.55**.

## Visual: CFR vs EW per-window + concatenated equity

![CFR Phase 4d val-Sharpe per window vs cumulative equity of
deterministic alternatives over the concatenated val span. Top:
CFR (blue) and EW (green) cross repeatedly; CFR wins w0/w3/w4,
EW wins w1/w2 with the +0.5 Sharpe gap in w2 driving the mean
alpha gap. Bottom: passive EW, naive uniform mix, and the best
static menu mode (`lowv252@g1`) all sit on top of each other for
16 years — the menu is so flat that the best single action only
beats EW by +0.039 Sharpe.](images/cfr-phase4d-vs-ew.png)

## Three hard facts

1. **Mean alpha collapses 3.7×** from raw +0.056 → net +0.015. On a
   $100k portfolio that's ~$200/yr extra expected return — basically
   nothing.
2. **Worst-case behavior is WORSE for the bot.** DCA's worst-window
   Sharpe (+0.642) clears CFR's by +0.21. Stress windows are when
   you most want stability; CFR loses on this metric.
3. **Operational tax exceeds the alpha.** ~30 min/month broker
   monitoring + quarterly tax-lot reconciliation + annual model
   retrain at ~$50/hr operator value ≈ **>$200/year**. Net expected
   value of running the bot is negative.

## What this changes

- **`apps/cfr` Phase 4d reclassified** from `PASS` (raw alpha) to
  **`confirmed-null` on realistic-alpha basis**. The architecture
  is validated as far as it can go on this universe; further
  iteration on representation will not move the binding constraint.
- **`apps/dca` is the canonical live strategy** as of 2026-05-13.
  The 13-asset basket from Phase 4d is preserved as the deployable
  artifact; the regret-net is archived but not deployed.
- **CFR research stays as a documented body of work** in case the
  regime shifts. If macro v2 surfaces a multi-decade regime change
  (e.g., sustained 5%+ real rates breaking the 60/40 dominance),
  the cross-asset CFR scaffolding is ready to re-deploy on a
  different universe.

## Master walk-forward log

Add a leaderboard row at 2026-05-13 referencing the realistic-friction
re-evaluation of `Output/cfr-phase4d.json` against
`Output/dca-multiasset.json`. Verdict label:
[`confirmed-null`](../leaderboard.md#verdict-labels) on
realistic-alpha basis (note the prior 2026-05-12 `PASS` row stays
in the log per the append-only protocol; this row references it).
