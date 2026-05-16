---
tags:
  - vol-surface
  - reversed-OOS
  - deployability
  - data-availability
---

# Small-capacity illiquid-VRP re-frame — falsified upstream of the spread question (quote-availability is the deployability gate that bites first)

**Operational rule:** before building a capacity-constrained strategy,
verify *quote-availability* of the exact cohort the signal selects — a
real signal trapped in a cohort with no historical or live quotes is
not an edge. Quote-availability bites *upstream* of transaction cost:
you cannot pay a spread on a contract no data vendor prices. For the
vol arc specifically: the unrestricted-universe VRP is **real**
(+0.14 of straddle premium per cycle, artifact-cleaned) but lives in
microcap names absent from every free option-quote source; the
quote-available subset has **negative** edge (−0.092), independently
re-confirming [`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md)'s
liquid-cohort collapse from a completely different angle (quote-DB
membership, not OI rank). The
[small-capacity re-frame](../TODO/vol-borrow-illiquid-vrp.md) is
[`reversed-OOS`](../leaderboard.md#verdict-labels) for a small
operator on free data; the breakeven-spread test never needed to run.

## Why this was tested

The vol arc closed `partial-OOS`
([`vol-arc-synthesis`](vol-arc-synthesis.md)) with the deployability
killer being [`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md):
alpha Sharpe decays monotonically +5.86 (unrestricted) → −0.48
(top-200 OI). The arc treated "alpha lives where options are
illiquid" as an institutional-scale kill. The re-frame asked whether a
*small operator* — for whom illiquidity is a moat, not a kill — could
harvest it, and whether securities-lending stress (the
mechanism-linked friction) sharpens it. Pre-registered in
[`TODO/vol-borrow-illiquid-vrp`](../TODO/vol-borrow-illiquid-vrp.md).

The arc never reached the borrow-conditioning or breakeven-spread
stages: two cheaper offline gates killed it first.

## Eval setup

- **Edge side (offline):** the v1 linear predictor + `forward_iv_rv_gap`
  on the full gauss314 schema (`.iv-cache/data_IV_USA.csv`,
  3,083,953 `(date,symbol)` rows, 2019-10-14 → 2023-07-28, 918 dates,
  5 walk-forward windows at 300/120/120). Edge expressed as a fraction
  of straddle premium: `edge_frac ≈ (IV − RV)/IV = iv_rv_gap/ATM_IV`
  (decision-grade short-straddle return-on-premium).
- **Cost / tradability side:** DoltHub `option_chain` — the only free
  per-contract source (the `ss_iv` loaders never query it). Coverage
  is broad in *time* (2019-06 → 2025-01+) but **narrow in universe**.
- Drivers: `apps/vol/scripts/run_v0_artifact_decomp.py` (Gate A),
  `apps/vol/scripts/run_v0_tradable_fraction.py` (Gate B-pre).

## Gate A — is the unrestricted edge a delisting/halt artifact?

The first smoke surfaced that the predictor's top picks were
acquired/halted names (BITA private 2020, SHLX/GPX/AVEO acquired
before their rebal date, BIL a T-bill ETF) with `edge_frac ≈ 0.9` —
the signature of `RV_forward → 0` against frozen-high IV (a
cash-settlement/halt artifact, not harvestable VRP). Gate A decomposed
the population:

| Subset | Share of universe | Mean `edge_frac` |
|---|---:|---:|
| Halt (RV<0.05) | **1.8%** | +0.6932 |
| Degenerate OI (<100) | 13.8% | −0.0931 |
| Clean (tradable-candidate) | 84.9% | −0.1942 |
| FULL universe | 100% | −0.1688 |

Predictor top-K (the actual strategy) per walk-forward window:

| Window | val period | val r FULL | val r CLEAN | top-K edge FULL | top-K edge CLEAN |
|---:|---|---:|---:|---:|---:|
| 0 | 2021-01→06 | +0.005 | +0.007 | +0.184 | +0.155 |
| 1 | 2021-06→12 | +0.055 | +0.069 | +0.119 | +0.049 |
| 2 | 2021-12→2022-06 | +0.035 | +0.076 | +0.066 | −0.015 |
| 3 | 2022-06→12 | +0.268 | +0.218 | +0.391 | +0.238 |
| 4 | 2022-12→2023-06 | +0.238 | +0.202 | +0.488 | +0.278 |
| **mean** | | **+0.118** | **+0.114** | **+0.2495** | **+0.1408** |

**Gate A verdict: the edge is real.** The halt artifact is only 1.8%
of rows; val r is essentially unchanged after cleaning; the predictor
top-K retains **+0.1408 of premium per cycle (56% of the +0.2495
full)** on clean non-halt/real-OI names. The signal is genuine VRP,
not survivorship. (Note: the *universe* mean `edge_frac` is negative
−0.169 — the VRP is entirely in the *selection*, never the universe.)

## Gate B-pre — is the clean edge cohort even quote-available?

Direct probing found `option_chain` is a *limited* universe, not the
2,276-ticker breadth of `volatility_history`. On 2022-06-17,
S&P-smallcap **PLCE has 36 contracts** while microcap biotechs
**CTIC / MRNS / MNOV return 0**. Quantified over the clean predicted
picks (793 unique symbols; 250 probed on two liquid reference dates
2021-06-17 / 2023-01-20):

| Metric | Value |
|---|---:|
| Symbols quote-available | 30 / 250 (**12.0%**) |
| Picks quote-available (pick-weighted) | **7.5%** |
| Mean `edge_frac`, quote-available picks | **−0.0919** |
| Mean `edge_frac`, absent picks | **+0.1987** |

**The double kill:** (1) 92.5% of the edge-carrying picks are
un-quotable on free data — you cannot backtest honest cost *or* trade
them live; (2) the 7.5% that *are* quotable have **negative** mean
edge. Even at zero transaction cost, trading only the quotable subset
loses money. The breakeven-spread gate is moot — there is no positive
tradable edge for a spread to erode.

## Mechanism

The VRP the surface features capture is concentrated in low-liquidity
single-name options (consistent with
[`vol-surface-v2-oi-restriction`](vol-surface-v2-oi-restriction.md)'s
VRP-vs-information-friction reading). Quote vendors price the liquid
names and skip the illiquid ones for the same reason market-makers
arbitrage the VRP away in the liquid ones: economics of coverage track
economics of liquidity. So the cohort that carries the edge is
*structurally co-extensive* with the cohort no one quotes. The
re-frame's "illiquidity is the moat" inverts into "illiquidity is an
un-priceable wall": the quote-available subset is precisely the liquid
cohort v2 #2 already falsified — here re-derived from an orthogonal
axis (option-quote-DB membership), with the edge sign-flipping negative
exactly as v2 #2 predicted.

## DoltHub `option_chain` operational notes (for future work)

- Schema: `date, act_symbol, expiration, strike, call_put, bid, ask,
  vol, delta, gamma, theta, vega, rho`. PK is
  `(date, act_symbol, expiration, strike, call_put)`.
- The HTTP API deadline-caps hard. **Only fast query shape:**
  `WHERE date='YYYY-MM-DD' AND act_symbol='X'` (leading-PK equality)
  + the `delta` band for ATM. Any `date >= .. AND date <= ..` *range*
  scan, and *every* aggregate (`COUNT(*)`, `COUNT(DISTINCT)`,
  `DISTINCT`), returns `context deadline exceeded`. This is why the
  first breakeven smoke spuriously reported 0% tradable (range-scan
  bug) — the real blocker is universe coverage, confirmed separately.

## What would change the verdict

Only a paid per-contract source (ORATS / OptionMetrics IvyDB /
Polygon options) that actually covers Russell-microcap options. Even
then the borrow leg (true fee/utilization history) is a second paid
dependency, and the residual question is whether the +0.14 gross
survives a *measured* microcap straddle spread (PLCE-class names
already showed ~15%). Not worth the paid-data spend on a
`reversed-OOS` base.

## Master walk-forward log

One [2026-05-15 leaderboard row](../leaderboard.md) —
[`reversed-OOS`](../leaderboard.md#verdict-labels) for the
small-capacity re-frame (real signal, un-priceable cohort,
negative edge on the quote-available subset). Closes
[`TODO/vol-borrow-illiquid-vrp`](../TODO/vol-borrow-illiquid-vrp.md);
the parent arc verdict in
[`vol-arc-synthesis`](vol-arc-synthesis.md) is unchanged
(`partial-OOS` with the v3 liquid-regime-gated recipe; DCA stays
canonical live).
