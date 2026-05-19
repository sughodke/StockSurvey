# Dynamic rebalance cadence is a non-binding knob (confirmed-null)

## Operational rule

**Do not pursue rebalance cadence — fixed, per-regime, or model-chosen
("when to be woken up") — as an alpha lever on a diversified basket.**
A hindsight oracle that picks the *best* EW rebalance cadence per
regime-window (from {daily, 5d, 20d, 60d, buy-hold}) beats fixed
daily-EW by **+0.004 mean Sharpe** over 2000–2025 — 37× under the
pre-registered +0.15 kill gate. A deployable selector cannot exceed
its own hindsight oracle, so the lever is closed before any Stage-1
build. The regime-table edge that *does* exist (+0.39 strategy-selection
oracle, see [passive-ew-benchmark](passive-ew-benchmark.md) and
[gate-drawdown-v0](gate-drawdown-v0.md)) is **strategy selection, not
execution cadence**. This is the fifth and final reframing of
state-conditional selection in this arc — see "Lever genealogy" below.

→ **Cost-aware corollary:** in *gross* terms daily (k=1) is marginally
best (wins/ties 3/6 windows); in *net* terms more-frequent rebalancing
is strictly worse, because daily's ≤ +0.009 gross edge is erased and
then exceeded by turnover cost. The deployable reading is therefore the
opposite of "rebalance faster": rebalance as *infrequently* as a drift
bound allows. This is exactly the canonical `apps/dca` recipe
(80-trading-day quarterly cadence + 5% per-name drift trigger).

## Eval setup

| Knob | Value |
|---|---|
| Substrate | gate v0 EW aggregate (so daily-EW reproduces `gate-walkforward-summary.json` `unc_sharpe`) |
| Universe | `stooq_us_long` manifest (312 names), `load_stooq_matrix(min_history=150)` |
| Windowing | 6-window 1260/780/780 daily bars, 2000-01-01 → 2025-12-11 (identical to gate/pairs/cfr) |
| Cadence arms | reset to 1/N every k ∈ {1, 5, 20, 60} bars (drift between), plus buy-hold within val window |
| Metric | `ss_portfolio.annualized_sharpe` (`mean/std(ddof=0)·√252`) per val window |
| Oracle | per window, max Sharpe over the 5 cadence arms (hindsight; strict upper bound on any selector) |
| Pre-reg kill gate | mean(cadence-oracle) − mean(daily-EW) < **+0.15** ⇒ `confirmed-null` |
| Validation gate | max\|daily-EW − artifact `unc_sharpe`\| < 0.06 (else numbers untrustworthy) |

Driver: `apps/gate/scripts/dynamic_rebal_cadence_oracle.py` (local, ~2 min,
no Modal). Artifact: `Output/dynamic-rebal-cadence-oracle.json`.

## Per-window result

| Window | Regime | daily (k=1) | k=5 | k=20 | k=60 | buy-hold | **cadence-oracle** | best |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| w0 2005 | calm bull | +0.699 | +0.614 | +0.605 | +0.605 | +0.659 | +0.699 | k1 |
| w1 2008 | GFC | +0.490 | +0.414 | +0.395 | +0.364 | +0.257 | +0.490 | k1 |
| w2 2011 | euro crisis | +0.956 | +0.946 | +0.947 | +0.924 | +0.900 | +0.956 | k1 |
| w3 2014 | low-vol grind | +0.976 | +0.951 | +0.957 | +0.973 | +0.984 | +0.984 | bh |
| w4 2017 | late cycle | +0.408 | +0.413 | +0.380 | +0.358 | +0.384 | +0.413 | k5 |
| w5 2020 | COVID/recovery | +1.065 | +1.073 | +1.061 | +1.032 | +0.960 | +1.073 | k5 |
| **mean** | | **+0.766** | | | | | **+0.769** | |

Validation: max\|daily-EW − artifact `unc_sharpe`\| = **0.029** (PASS).
**Cadence-oracle headroom = +0.004.** Per-window Δ over daily:
`[0, 0, 0, +0.009, +0.005, +0.008]`.

## "Have we gone the other way — is rebal=1 day the answer?"

A reasonable read of the table is "daily wins, so rebalance daily."
That is true *gross* and misleading *net*:

1. **Daily is the data floor, not a discovered optimum.** StooqData is
   daily bars — k=1 is the finest cadence testable; there is no
   intraday arm to "go the other way" toward (platform constraint).
2. **The gross win is noise.** Daily wins/ties 3/6 windows outright;
   the other 3 favor a slower arm by ≤ +0.009. The *spread across all
   cadences* within any window is ≤ ~0.10 Sharpe and the per-window
   best is scattered (k1·3, k5·2, bh·1) with no regime structure.
3. **Net flips the sign of the recommendation.** The probe is gross
   (no costs), matching the gate `unc_sharpe` baseline. Daily has the
   highest turnover; at `commission_bps=10` its ≤ +0.009 gross edge is
   more than consumed by friction. So the cost-aware optimum is the
   *least* frequent cadence the drift tolerance permits — the opposite
   of "rebalance faster."

So: no, the result is not "rebal=1 is the lever." It is "cadence is
non-binding in *both* directions, and the only cost-relevant gradient
points toward *less* trading" — which is why DCA's slow quarterly
cadence is canonical live and why this lever is closed, not inverted.

## Mechanism

A broad ~312-name daily-EW basket's Sharpe is near-invariant to
rebalance frequency over multi-year windows: idiosyncratic single-name
drift averages out across the cross-section, so the portfolio return
distribution (hence `mean/std`) barely moves whether you reset weights
every 1 or 60 bars. The rebalance-frequency premium is a well-known
small effect (tens of bps/yr), and critically it does **not vary enough
by regime** to make a per-regime selector worthwhile. The +0.39 of real
regime-conditional headroom in the cross-arc table was always
*strategy-selection* (passive vs gate vs pairs) — an orthogonal axis a
cadence knob cannot touch.

## Lever genealogy — why this closes the line

"The model chooses when to be woken up again" is the fifth instantiation
of state-conditional selection tested in this repo. Every prior one is
already null or partial-with-measured-small-ceiling:

| Instantiation | Existing result |
|---|---|
| CFR meta-allocator | `confirmed-null` realistic friction; DCA wins ([cfr-vs-dca-realistic](cfr-vs-dca-realistic.md)) |
| Macro VIX meta-gate | `partial-OOS`/inconclusive, raw lift −0.010 ([macro-regime-diagnostic](macro-regime-diagnostic.md)) |
| Critic Φ(state,action) | `confirmed-null`, oracle-clean Spearman negative ([critic-phi-quality-v0](critic-phi-quality-v0.md)) |
| Factor endogenous-horizon (model picks its own next wake) | `partial-OOS`, 5 failed rescues, +0.11 oracle ([factor-endogenous-horizon-mixture](factor-endogenous-horizon-mixture.md)) |
| **Dynamic rebal cadence per regime (this)** | **`confirmed-null`, +0.004 oracle** |

A deployable wake-time policy is bounded above by its hindsight oracle;
those oracles are now measured per arc (+0.004 here, +0.11 factor, +0.39
gate — the last a predictor-quality problem already `confirmed-null`
when attacked via macro). The lever is closed.

## What's next (the confirmed-null default — orthogonal lever)

Per the `confirmed-null` playbook: stop testing variations of
state-conditional selection / execution timing — find an orthogonal
axis. The only arc in the repo with a large, real, OOS-stable,
un-arbitraged signal is **vol** (+5.86 pooled-α, 11/11 OOS quarters,
[vol-surface-v1](vol-surface-v1.md)). Its blocker is *execution/data*
(no free microcap option quotes,
[vol-borrow-illiquid-vrp-falsified](vol-borrow-illiquid-vrp-falsified.md)),
not modeling or scheduling — orthogonal to everything falsified here.
No new TODO is created: this finding *closes* a line; the orthogonal
next is the already-tracked vol deployability work
([vol-arc-synthesis](vol-arc-synthesis.md)).

## Master walk-forward log pointer

Leaderboard row: `2026-05-18 | gate | Dynamic-rebal-per-regime cadence
oracle` → [`confirmed-null`](../leaderboard.md#verdict-labels).
