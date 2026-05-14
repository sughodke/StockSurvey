---
tags:
  - vol-surface
  - arc-synthesis
  - partial-OOS
---

# Vol surface arc — v0 → v1 → v2 synthesis (partial-OOS; liquidity is the binding constraint)

**Arc state as of 2026-05-14:** the vol surface signal is **real, OOS-
stable, and dollar-Sharpe-substantial on the unrestricted universe**,
but the deployable subset (top-200-OI tradable options) doesn't carry
it. The arc closes as `partial-OOS`: signal exists but isn't
broadly deployable without further conditioning (regime gate +
narrower universe). The audit's framing of "the strongest signal in
the repo, benched without v1 work" is empirically validated for the
signal-existence claim; the deployability claim requires v3.

## Arc trajectory

| Phase | Finding | Verdict | Key result |
|---|---|---|---|
| v0 | [`vol-surface-v0`](vol-surface-v0.md) | `inconclusive` | Per-cell Sharpe alpha +0.089 (just below +0.10 marginal). 5/5 positive windows. Audit flagged this as load-bearing weak-metric. |
| v1 | [`vol-surface-v1`](vol-surface-v1.md) | `confirmed-OOS` | Per-rebal portfolio Sharpe +5.86 unrestricted, 5/5 positive, real-to-shuffle alpha-PnL ratio 25×. v0 metric was the bottleneck. |
| v2 #1 | [`vol-surface-v2-dollar-pnl`](vol-surface-v2-dollar-pnl.md) | `confirmed-OOS` | Dollar Sharpe +4.60 under standard equal-$-vega sizing; +1.95 under share-count-equal. Both clear +0.30 PASS by 6×. Audit concern about vol-points vs dollar resolved. |
| v2 #2 | [`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md) | `reversed-OOS` | Alpha decays monotonically with OI threshold: −0.48 (top-200), −0.05 (top-500), +1.04 (top-1000), +5.86 (unrestricted). Liquid universe doesn't carry the signal. |
| v2 #3 | [`vol-surface-v2-dolthub-oos`](vol-surface-v2-dolthub-oos.md) | `confirmed-OOS` | OOS extension to 2026-04 on 4-feature DoltHub proxy: r=+0.165 cross-sectional, alpha mean +0.076 vol pts per rebal, 11/11 positive quarters. Signal IS alive in 2024-2026. |

## The arc-decisive finding

**v2 #2 (OI restriction) is the load-bearing experiment.** The
alpha-vs-threshold curve is monotonic and sharp:

| OI threshold (top-N per date) | Alpha Sharpe | Mean val r |
|---:|---:|---:|
| Unrestricted (~3877) | **+5.86** | +0.120 |
| Top-1000 | +1.04 | +0.068 |
| Top-500 | −0.05 | +0.042 |
| Top-200 | **−0.48** | +0.021 |

The audit identified this exactly: "OI restriction is the most
important v2 step before paper-trading." It turned out to be a
deal-breaker, not a refinement.

**Mechanism**: the vol risk premium is larger in less-liquid options
markets (consistent with academic literature on VRP and information
friction). The surface features (skew, smile, IV/HV ratio) capture
mispricings that liquid market-makers have already arbitraged away.

## Why this isn't a `confirmed-null` arc closure

Two reasons the result is `partial-OOS` rather than `confirmed-null`:

1. **The signal exists and is OOS-stable.** v2 #3 confirmed
   continuity from 2019-2023 (gauss314) to 2023-2026 (DoltHub) at
   similar cross-sectional Pearson r magnitudes. The underlying
   inefficiency hasn't been arbitraged away over 2.7 years of OOS.
2. **Regime-conditional liquid-name alpha exists.** v2 #2 per-window
   results: even on top-200 OI names, w3 (2022-06 → 2022-12,
   post-Fed-pivot) and w4 (2022-12 → 2023-06) post alpha Sharpe +4.2
   and +6.5 respectively. The deployment-killer is windows 0-1
   (2021 calm-bull) where the predictor anti-correlates on liquid
   names.

Combined: there IS a deployable strategy, contingent on (a)
restricting to stress regimes (similar to macro v1b VIX gate)
AND (b) accepting deployment ~40% of the time.

## v3 architecture proposal (untested)

**Regime-gated liquid-universe short-vol predictor:**

1. **Liquidity filter**: top-200 OI names per date (v2 #2 tested
   this is necessary).
2. **Regime gate**: only deploy when VIX > trailing-N-day-median
   (similar to macro v1b's window-level gate). Per v2 #2 per-window
   results, w3/w4 (2022-2023 post-Fed-pivot) are the windows where
   liquid-name alpha lives — these were high-VIX windows.
3. **Feature stack**: full v1 10-feature surface (skew + smile + IV/HV
   + OI imbalance + VIX-spread + strike-spread).
4. **Predictor**: linear OLS or small MLP (v2 #4, untested).
5. **Sizing**: equal-$-vega per pick (v2 #1's confirmed convention).
6. **Cost-in-loop**: 100-500 bps friction (deployable range on liquid
   options).

**Pre-reg cuts for v3**: net Sharpe ≥ +0.30 on top-200-OI universe
WITH regime gate applied; positive in ≥ 4/6 deployed windows.

**Estimated work**: ~6h to wire the regime gate to v2 #2's code +
re-run; v3 result determines whether the arc closes as `partial-OOS`
(if v3 PASSES) or `confirmed-null` (if regime gating can't rescue
deployment).

## Live-deployment track (separate from research arc)

Even if v3 PASSES on a regime-gated liquid universe, the live
deployment workstream is substantial:

| Item | Wall | Notes |
|---|---|---|
| `ss-vol live` CLI + checkpoint persist | ~6h | Port pattern from `ss-relational live` |
| Options broker adapter | ~6h | IBKR or Tradier (Alpaca lacks multi-leg options); add to `ss_portfolio.broker` |
| Sizing model with vega-cap | ~3h | Vega budget = 10% of cash notional; per-name vega cap = 1% to limit concentration |
| Risk model (CVaR / max DD) | ~3h | Short-vol portfolios have asymmetric tail risk (uncapped loss on vol spikes) |
| Paper trade (real broker) | 1-2 rebals over 4-8 weeks | Validate end-to-end |

Live infrastructure work is unblocked by v3 results; meanwhile, DCA
remains canonical live ([`apps/dca`](../apps/dca.md)) for simplicity.

## What v0 / v1 / v2 collectively confirm vs the audit

The 2026-05-14 research-directions audit
(`.audit-research-directions.md` at repo root) flagged:

| Audit claim | Empirical outcome |
|---|---|
| "Strongest signal in the repo, benched without v1 work" | **Confirmed by v1** — alpha Sharpe +5.86 vs v0's per-cell Sharpe of +0.089 (different metric, same signal) |
| Per-rebal portfolio aggregator is the v1 fix | **Confirmed by v1** — methodology change unblocked the verdict |
| Costs in the loop are needed | **Confirmed by v1/v2 #1** — alpha is friction-invariant in difference; absolute drag at 100/250/500 bps is small |
| OI restriction is the most important pre-paper-trade step | **Confirmed by v2 #2** — this is the binding constraint that kills broad deployment |
| DoltHub extension to 2026 is "the natural OOS test" | **Confirmed by v2 #3** — signal extends with high cross-sectional consistency |
| MLP head deferred to v2 if linear PASSES | **Deferred; linear already PASSES on signal existence; v2 #2 shows MLP wouldn't help with the binding constraint (liquidity) anyway** |

## Operational rules extracted from the arc

1. **When a v0 finding's headline metric is flagged as "weak", run
   the honest metric before pivoting away** (per
   [`vol-surface-v1`](vol-surface-v1.md)'s lede). v0's per-cell
   Sharpe was the bottleneck, not the signal.

2. **For vol surface prediction, never gate on univariate Pearson r**
   (per [`vol-surface-v0`](vol-surface-v0.md)). Univariate r ≤ +0.003;
   multivariate r = +0.165 OOS. The signal lives in joint structure.

3. **When constructing forward-realization targets for vol prediction,
   use price returns, not lagged HV snapshots.** DoltHub's hv_current
   has autocorrelation 0.85 at lag-4-weeks; using it as the target
   silently leaks features into the target. Verified by the v2 #3
   first-pass bug catch.

4. **The vol risk premium is concentrated in low-OI names** — the
   surface features capture mispricings that liquid market-makers
   arbitrage away. Any deployment strategy must either accept low-OI
   tradability constraints (small size + wide spreads) or compose
   with a regime gate that finds the windows where liquid names also
   carry alpha (`partial-OOS` until v3 tests this composition).

## Master walk-forward log

Three [2026-05-14 leaderboard rows](../leaderboard.md) for v2 #1
([`confirmed-OOS`](../leaderboard.md#verdict-labels)), v2 #2
([`reversed-OOS`](../leaderboard.md#verdict-labels)), v2 #3
([`confirmed-OOS`](../leaderboard.md#verdict-labels)). Synthesis
verdict: arc-level `partial-OOS` (signal real and OOS-stable,
deployability blocked by liquidity, regime-gate composition not
yet tested).

## Recommended next experiment

`TODO/vol-v3-regime-gated-liquid.md` (to be written): pre-register
the regime-gated-liquid v3 test and run as the final research check
before either (a) deploying with the documented regime constraint
or (b) parking the arc as `confirmed-null` on broad-deployment.
