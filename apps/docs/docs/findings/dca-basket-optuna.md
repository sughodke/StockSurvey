# DCA basket Optuna search — `partial-OOS` (effectively `confirmed-null`)

**Operational rule.** A pre-registered Optuna search over 3,600
DCA-basket bucket combinations (N_TRIALS = 200, train 2005-2018 / val
2019-2025, deflated-Sharpe objective) **failed to find a basket that
beats the canonical 13-ETF reference by the pre-reg's +1.0 deflated-t
bar**. The "winner" beats canonical by Δ deflated-t = **+0.016** on
val — below noise — which the locked rule labels `partial-OOS` but is
effectively a confirmed-null. **The canonical multi-asset
diversification is defensible**; the search was the right way to test
that. **One operational improvement worth noting**: the winner is a
*simpler* basket (4 names: VTI + TLT + IEF + GLD) that achieves
near-identical Sharpe and a marginally better max-DD — most of the
canonical 13-ETF's complexity is replaceable without loss.

## Pre-registration

Locked at [`TODO/dca-basket-optuna.md`](../TODO/dca-basket-optuna.md)
in commit `d49c672` (2026-05-23) BEFORE the eval ran. The
falsification bar, search space, walk-forward split, trial budget,
and DSR calibration were all fixed before any Optuna trial fired.

## Result

### Headline numbers (winner vs canonical, evaluated under identical method)

| metric | canonical 13-ETF (12 after DBC drop) | Optuna winner |
|---|---:|---:|
| holdings | 12 (XLB-Y + TLT + IEF + GLD) | **4** (VTI + TLT + IEF + GLD) |
| train ann Sharpe | +0.610 | +0.595 (similar) |
| train deflated-t | −0.299 | +0.580 (search target) |
| **val ann Sharpe** | **+0.854** | **+0.858** |
| **val deflated-t** | **+0.419** | **+0.436** |
| val max-DD | −0.239 | −0.222 |

### Verdict per pre-reg

- Δ val deflated-t (winner − canonical) = **+0.016**
- Δ val max-DD                          = **+0.017** (winner slightly better)
- Pre-reg bar: confirmed-OOS requires Δ deflated-t > +1.0 AND Δ
  max-DD > −0.05
- Pre-reg bar: partial-OOS requires Δ deflated-t > 0.0 AND Δ max-DD >
  −0.05
- **Locked verdict: `partial-OOS`** (Δ deflated-t is technically
  positive but well below noise; semantically this is closer to
  `confirmed-null`)

### Why the winner is so close to canonical

- DBC was dropped (no 2005-on Stooq coverage), so the "canonical 13"
  evaluated here was actually 12 — but the live DCA strategy includes
  DBC, which only has continuous data from 2006-02. The post-2006
  DCA backtest (Sharpe +0.67, the leaderboard number) includes DBC;
  this Optuna eval used a slightly earlier-starting 12-ETF version
  for fair walk-forward comparison.
- VEU was dropped (no 2005-on coverage). Optuna's *sampled* winner
  parameters specified `intl_bucket='VEU'`, but the live basket
  dropped VEU when it failed coverage → effective universe is just
  the 3 non-intl buckets the winner chose.
- The 9 SPDR sectors equal-weighted ≈ SPY ≈ VTI (sector-cap weighted
  vs market-cap weighted, but very similar exposure). The winner
  replaces 9 sectors with 1 VTI — same equity exposure, lower
  friction.

## What we learned (and what we didn't)

### The winner IS interesting operationally

A **4-ETF basket (VTI + TLT + IEF + GLD)** with 80-day rebal achieves:
- Same val Sharpe as canonical 13-ETF (+0.858 vs +0.854)
- Marginally better val max-DD (−22.2% vs −23.9%)
- 3× fewer holdings (less friction, simpler operations)

This is the cleanest implementation of the multi-asset thesis:

| asset class | canonical | winner |
|---|---|---|
| US equity | 9 SPDR sectors EW | VTI |
| Long-duration UST | TLT | TLT |
| Mid-duration UST | IEF | IEF |
| Gold | GLD | GLD |
| Broad commodity | DBC | none |

**If you want the simplest defensible DCA basket, deploy the
4-ETF winner.** If you want belt-and-suspenders multi-asset coverage
including commodities exposure, deploy the canonical 13-ETF. Both
have effectively the same deflated-t after the pre-registered
search bar.

### The pre-registered bar was the right discipline

Without the +1.0 bar locked in advance:
- We'd be tempted to call the +0.016 lift "a small improvement worth
  deploying" (i.e., snoop a marginal winner).
- We'd over-fit operational decisions to noise.
- The reported "DCA stays #2 on the DSR ladder" claim would weaken
  with each unprincipled basket tweak.

With the bar locked: the +0.016 lift is correctly priced as noise.
The canonical basket — or the simpler 4-ETF basket — both pass; the
search did not find a defensible alpha lift over either.

### What the search did NOT test (acknowledged in pre-reg)

- **Sub-sector / individual-stock baskets** — out of scope by
  design; the universe was bucketed ETF families, not arbitrary
  tickers.
- **Dynamic weights** — within each bucket combination, allocation
  was equal-weight only. Adding a sizing layer (mean-variance,
  trailing-Sharpe-weighted, risk-parity) would explode the search
  space and the data-snooping risk.
- **Lever changes** — leveraged ETFs (UPRO, TMF) deliberately
  excluded; they're a different strategy class and would need
  separate volatility-targeting machinery.
- **Currency hedging** — international was VEU (no FX hedge);
  hedged-equivalent ETFs not tested.
- **Cost variation** — locked at `commission_bps=10.0`. Real Alpaca
  fills options have wider implicit cost; but DCA leg is equity ETFs,
  not options, so the 10 bps is realistic for the cash leg.

## Top-10 trials by train deflated-t

| rank | equity | intl | bonds | commod | reit | rebal | drift | train t | val t |
|---:|---|---|---|---|---|---:|---:|---:|---:|
| 1 | VTI-only | VEU | TLT+IEF | GLD+DBC | none | 80 | 0.03 | +0.580 | +0.436 |

(Top-10 in `Output/dca-basket-optuna.json`; ranked-by-train-t winner
preserved here for the verdict.)

## Implications

### For the live DCA + vol ensemble

The canonical 13-ETF basket recipe in `Output/dca-multiasset.json` is
**defensible** — the search did not find a basket that beats it
materially on val deflated-t.

**Optional simplification**: the 4-ETF winner (VTI + TLT + IEF + GLD)
delivers near-identical val performance with one-third the holdings.
This is a maintenance / operational decision, not an alpha decision.
If picked, the leaderboard row stays the same (search confirmed
both are equivalent under the bar); the operational note would be
"deployed the simpler basket because it has the same statistical
properties."

### For future basket searches

- The +1.0 deflated-t bar is high but defensible. Future basket
  searches should set similar bars; otherwise the search becomes a
  snooping device.
- Bucket-based search (instead of arbitrary-ticker search) was the
  right discipline — the pre-reg's 3,600-combo space was small
  enough to enumerate effectively while broad enough to find a
  meaningful winner if one existed.
- N_TRIALS = 200 was sufficient — Optuna converged to a stable
  region in <50 trials; the marginal gain from trial 50-200 was
  noise.

## Reproduction

```bash
uv run python apps/dca/scripts/optuna_basket_search.py
```

Inputs: `StooqData/` (the same archive every other arc uses).
Outputs: `Output/dca-basket-optuna.json`. Runs in ~10 seconds on a
local laptop after ticker prefetch (no Modal needed; Optuna's TPE
sampler is the bulk of the work and it's lightweight).

## Master walk-forward log

[Cross-arc DSR ladder](../leaderboard.md#cross-arc-deflated-sharpe-ranking)
— canonical DCA stays at rank #2 (deflated-t +1.93 full-sample, n_trials=4).
This search's val deflated-t (+0.42 on the post-2019 slice with
n_trials=200) is not directly comparable to the rank-#2 number because
the trial counts and sample lengths differ; the comparison that
matters is winner-val (+0.44) vs canonical-val (+0.42) under
identical method. Verdict label
[`partial-OOS`](../leaderboard.md#verdict-labels).
