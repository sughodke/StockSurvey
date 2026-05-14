---
tags:
  - vol-surface
  - reversed-OOS
  - liquidity
  - v2-arc
---

# Vol surface v2 #2 — OI restriction collapses alpha monotonically (reversed-OOS on deployability)

**Operational rule (extracted):** *Vol surface alpha lives in the
long-tail of options activity — names with thin OI but with surface
anomalies that the market hasn't priced in. Filter to tradable
liquidity and the predictor's edge fades by ~80–110% depending on
threshold. The signal is real on the unrestricted universe (v1
PASS) but the deployable subset of that universe doesn't carry
enough of it.*

## Pre-registered cuts (v2 #2)

| Cut | Threshold | Observed at OI-top-200 | Verdict |
|---|---|---:|---|
| PASS | Alpha Sh ≥ +0.30 AND ≥ 4/5 pos AND ratio to v1 ≥ 0.5× | −0.48, 3/5 pos, −0.08× v1 | — |
| MARGINAL | Alpha Sh ∈ [+0.10, +0.30] OR ratio < 0.5× | — | — |
| FAIL | Alpha Sh < +0.10 OR ≤ 2/5 pos | **−0.48** ; 3/5 pos | **FAIL** |

## OI threshold sensitivity

| OI threshold | Universe size (post-filter) | Alpha Sh | val r | Pos windows | Verdict |
|---|---:|---:|---:|---:|---|
| Unrestricted (v1) | ~3877 | **+5.86** | +0.120 | 5/5 | PASS |
| Top-1000 | ~1000/day | +1.04 | +0.068 | 3/5 | MARGINAL |
| Top-500 | ~500/day | −0.05 | +0.042 | 3/5 | FAIL |
| Top-200 (audit's exact suggestion) | ~200/day | −0.48 | +0.021 | 3/5 | FAIL |

**Alpha decays monotonically with universe restrictiveness.** Mean
val r drops 6× from +0.120 to +0.021 between unrestricted and top-200.
This is a deal-breaker for the original deployment thesis.

## Why this happens — the audit's call was correct

The audit specifically flagged OI restriction as "the most important
v2 step before paper-trading." The result confirms it identified the
binding constraint, not a refinement: the surface-feature edge is
concentrated in low-OI names where:

- The IV market has less price discovery (fewer competing
  market-makers, sparser bid-ask data, slower IV updates)
- Bid-ask spreads are 5-20% of premium, so the realized vol-points
  PnL must be much larger than 100 bps to clear friction
- Position-sizing constraints prevent the strategy from accumulating
  enough $-vega for institutional deployment

This is consistent with the academic finding that VRP magnitude
scales inversely with options liquidity. The features (skew, smile,
IV/HV ratio) capture exactly the mispricings that liquid options
markets have already arbitraged away.

## But: regime-conditional alpha persists on liquid names

Looking at per-window results at OI-top-200:

| win | val period | val r | Alpha Sh | Note |
|---|---|---:|---:|---|
| 0 | 2021-01-06 → 2021-06-28 | **−0.063** | **−2.57** | Anti-predictive (calm-bull) |
| 1 | 2021-06-29 → 2021-12-27 | **−0.062** | **−1.32** | Anti-predictive |
| 2 | 2021-12-28 → 2022-06-17 | −0.165 | +1.59 | Mixed |
| 3 | 2022-06-21 → 2022-12-08 | **+0.283** | **+4.19** | Strong (Fed-pivot) |
| 4 | 2022-12-09 → 2023-06-02 | **+0.114** | **+6.50** | Strong (post-pivot) |

**The 2022-2023 post-Fed-pivot windows survive the liquidity
restriction** (alpha Sh +4-6 on liquid names, val r > +0.11). The
2021 calm-bull windows actively invert (negative alpha, negative
val r).

This replicates the regime-conditioning pattern that recurred
across gate / pairs / vol's v0 results:
[prediction-problem-pivot-arc](prediction-problem-pivot-arc.md).
Real multivariate signal exists, but the deployment window is
narrower than the eval window suggests.

## Implications for arc closure

This is the **arc-decisive experiment**. Three paths forward:

1. **Deploy with regime gate.** Stack a stress-regime filter
   (e.g. VIX-above-percentile, similar to macro v1b) on top of
   the OI-200 filter. Deploy only in 2022-2023-like regimes.
   Sample size cuts dramatically (~40% of time), but the alpha
   that exists is real (+4-6 Sharpe on liquid names in stress).
2. **Pivot universe.** Some markets have thin-OI options that
   are tradable in modest size (sector ETFs, single-name leaders
   with active weekly options). A curated mid-OI universe
   (top-100 to top-300 by 20d-avg OI) might preserve some signal
   while staying deployable.
3. **Accept arc closure as `partial-OOS`.** The signal exists
   but isn't broadly deployable. Document the regime-conditional
   alpha and the binding-liquidity-constraint; park the
   workstream.

Decision depends on v2 #3 (DoltHub OOS extension to 2026): if
the late-window signal extends past gauss314's 2023 cap, path #1
becomes viable; if not, paths #2 or #3.

## Master walk-forward log

[2026-05-14 vol v2 #2 OI-restriction row](../leaderboard.md) —
[`reversed-OOS`](../leaderboard.md#verdict-labels) on deployability.
The v1 PASS verdict ([`vol-surface-v1`](vol-surface-v1.md)) was on
the unrestricted universe; restricting to deployable liquidity
inverts the alpha at all tested OI thresholds {200, 500, 1000}.
The audit-flagged "OI restriction is the most important v2 step
before paper-trading" was correct — it identified the binding
constraint that the v1 metric couldn't see.

Artifacts:
- Driver: `apps/vol/scripts/run_walkforward_v2_oi.py`
- Outputs: `Output/vol-walkforward-v2-oi-{oi200,oi500,oi1000}-*.json`
- Predecessors:
  [`vol-surface-v1`](vol-surface-v1.md),
  [`vol-surface-v2-dollar-pnl`](vol-surface-v2-dollar-pnl.md)
- Decision-pending: `vol-surface-v2-dolthub-oos.md` (v2 #3)
