# Pairs

Pair-spread mean reversion — Engle-Granger cointegration screening
on per-window train slices, classical z-score-crossing trade rules
on the screened survivors. Numpy + statsmodels, no tinygrad.
Different prediction problem from `apps/factor`'s cross-sectional
ranking and `apps/gate`'s aggregate regime gate: per-pair
time-series mean reversion with long-short-by-construction
deployment.

## Why this app exists (the chain that motivated it)

Three independent
[2026-05-10 confirmed-nulls](../leaderboard.md#verdict-labels) on
the cross-sectional return prediction class:

1. [`passive-ew-benchmark`](../findings/passive-ew-benchmark.md)
   — no relational model row clears its universe's passive EW
   Sharpe.
2. [`factor-rankic-long-only-mismatch`](../findings/factor-rankic-long-only-mismatch.md)
   — long-short constructor on the rank-IC head delivers val
   Sharpe **−0.067**; "discarded short signal" hypothesis
   falsified.
3. [`factor-loss-pivot`](../findings/factor-loss-pivot.md) —
   Sharpe-aligned and IR-vs-EW-aligned losses both *destroy*
   val Sharpe by ~0.37; "wrong loss" hypothesis falsified.

`apps/gate`'s
[`gate-drawdown-v0`](../findings/gate-drawdown-v0.md) (mean val
Pearson r +0.26 — ~25-50× the cross-sectional return IC ceiling)
showed that an orthogonal prediction problem can have a different
data ceiling. Pairs is the second test of the
[different-prediction-problem](../TODO/different-prediction-problem.md)
pivot: pair-spread mean reversion is a third orthogonal problem
class with its own friction stack and deployment shape.

## What's in scope for v1

Classical only — z-score entry / exit thresholds on the EG
spread, no ML head yet. The classical baseline establishes
whether the prediction problem has any signal at all on this
universe at this horizon, before we invest in fancy machinery.
ML head comes in v2 *if* classical clears the pre-registered
pass cuts.

## Pipeline

`apps/pairs/src/pairs/` modules:

- **`cointegration.py`** — `engle_granger_test(log_p_a,
  log_p_b)` returns the EG-corrected p-value, ADF test stat,
  hedge ratio β, and intercept (via `statsmodels.tsa.stattools.coint`
  + `statsmodels.OLS`). Suppresses informational warnings on
  noisy short-history pairs and returns `p_value=1.0` sentinel
  on numerical failures.
- **`spread.py`** — `compute_spread(log_p_a, log_p_b, β, α)`
  returns the residual `log(P_A) − β·log(P_B) − α`. `spread_stats`
  reports mean / std / half-life. `zscore` normalizes against
  train-set stats.
- **`predictor.py`** — `trade_signals(z, entry=2.0, exit_z=0.5,
  stop=∞)` is the classical state machine: flat → long when
  z<−entry, → short when z>+entry, → flat when z crosses back
  past ±exit_z.
- **`pair_universe.py`** — `screen_pairs(log_prices,
  abs_corr_min, eg_p_max, top_k, n_workers)` per-window
  candidate generation. Three-stage filter:
  1. Min-overlap (≥80% of train window) — drop pairs without
     enough joint history.
  2. `|corr(log_p_a, log_p_b)| ≥ 0.7` — pre-EG correlation gate.
     Cuts ~78% of pairs (43k → ~10k for factor-narrow).
  3. EG p < 0.05 — keep, then sort by p-value, take top-K.
  Multiprocessing-parallel over the EG step (the dominant cost).
- **`backtest.py`** — `backtest_pair(...)` per-pair PnL from
  spread returns; `aggregate_pair_pnl(...)` equal-weights across
  N pairs.

`apps/pairs/scripts/`:

- **`smoke_kopep.py`** — single-pair sanity test on the famous
  KO+PEP cointegration. Validates the pipeline runs end-to-end
  with sensible β / intercept / p-value.
- **`run_walkforward.py`** — full 6-window walk-forward on
  factor-narrow (or any subset). Reports per-window agg
  Sharpe, mean-pair Sharpe, top-5 pairs by val Sharpe, and
  applies the pre-registered verdict cuts.

## Trade construction

When `position[t] = +1` (long-spread), hold $0.5 of A and short
$0.5 of B (gross $1, net $0). Per-bar PnL is `position ×
(Δlog P_A − β · Δlog P_B) / (1 + |β|)` — the leverage normalizer
makes Sharpe comparable across pairs with different hedge ratios.

Costs: each leg pays `commission_bps` on each open / close /
flip. `2.0 × commission_bps × |Δposition|` per bar. With
`commission_bps=10`, that's 20 bps per state transition. A
typical pair flips 5-15 times per 780-bar val window → 1-3% of
total turnover lost to costs.

## Pre-registered cuts

Per [`TODO/apps-pairs.md`](../TODO/apps-pairs.md):

| Outcome | Aggregate val Sharpe across windows | Verdict | Action |
|---|---|---|---|
| **Pass** | mean ≥ +0.50, ≥ 4/6 positive | `confirmed-OOS` | Build live deployment (Alpaca pair trading + borrow-check) |
| **Marginal** | +0.20 to +0.50, ≥ 3/6 | `partial-OOS` | Stratify by liquidity / sector before deciding |
| **Fail** | < +0.20 *or* ≤ 2/6 positive | `confirmed-null` | Pivot to [`apps/vol`](../TODO/apps-vol.md) |

Sharpe at the **strategy level** — not Sharpe of the spread
itself (which is well-known to be high in-sample, low OOS for
overfit cointegration screening). The cut is on the deployable
EW-of-pairs portfolio.

## Smoke test result (KO + PEP, 2010-2020 train)

EG p-value **0.086** (above the 0.05 threshold). KO+PEP
cointegration was famously strong in the 1990s/2000s; the link
loosened over 2010-2020 (separate corporate trajectories,
different sector exposures emerging). Pipeline runs cleanly with
sensible β=0.726, intercept=+0.338, n_obs=2769 — the
infrastructure is validated; the famous pair just doesn't
co-integrate strongly in this window.

## Walk-forward result — `confirmed-null` per pre-registration

See [`pairs-classical-v0`](../findings/pairs-classical-v0.md) for
detail. Headline: mean agg val Sharpe **+0.099** across 6 windows
(below the +0.20 fail threshold), but 4/6 windows positive with a
clear regime-conditional structure:

| win | val period | agg Sharpe | EG passing |
|---|---|---:|---:|
| 0 | 2005-01 → 2008-02 | **−1.233** | 3918 |
| 1 | 2008-02 → 2011-03 | **+0.870** | 3522 |
| 2 | 2011-03 → 2014-04 | +0.593 | 3118 |
| 3 | 2014-04 → 2017-05 | +0.392 | 4755 |
| 4 | 2017-06 → 2020-07 | +0.080 | 2249 |
| 5 | 2020-07 → 2023-08 | −0.109 | 2857 |

Window 0's catastrophe (dot-com-trained pairs deployed into
2005-2007 bull market) drags the unconditional mean below the
threshold; ex-w0 mean is +0.365. Pair trading worked in
2008-2017 mean-reverting markets, failed in trending bull
markets (2005-2007, 2020-2023).

## Reuses + dependencies

- `ss_loaders.load_stooq_matrix` — universe loader.
- `ss_portfolio.metrics` — annualized Sharpe / Sortino / CAGR /
  max drawdown.
- `statsmodels` — `tsa.stattools.coint` for EG, `OLS` for the
  hedge-ratio regression and half-life estimation. Only new
  dependency added by this app.

## Caveats

- Cointegration is regime-specific. We re-screen per train slice
  to avoid the "winning pair list" trap that bit the relational
  analog scorer. But the screen-then-deploy gap (1260 bars
  train → 780 bars val) means a pair that cointegrated through
  2018 may not still cointegrate in 2021.
- Sector restriction not applied in v1. Stooq doesn't carry
  sector metadata; cross-sector cointegration is statistically
  noisier (more likely a regression-on-noise artifact than a
  real economic relationship). v2 should add a sector filter
  using a separate metadata source.
- Borrow availability + cost not modeled. Some short legs may
  be unborrowable on live; live-trading shim needs an
  Alpaca-borrow-check wrapper before the first paper trade.
- v1 is classical-only. ML predictor (linear / MLP head over
  z-score history) will be the v2 if classical baseline shows
  signal worth refining.
