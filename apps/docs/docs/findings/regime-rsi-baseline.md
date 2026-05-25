# Regime-app RSI strategy — universe-agnostic walk-forward baseline

**Operational rule.** The `regime` app's third strategy head (`weights_rsi`,
top-N most-oversold names by mean Wilder RSI over the trailing `n_tail`
bars) is **`confirmed-null`** as a standalone universe-agnostic
strategy on `stooq_us_long`. Across 6 walk-forward windows
(1260-train / 780-val / 780-step, rebal_days=20, 10bps), the
canonical config (rsi_n=14, n_tail=5, top_n=20) yields **mean val
Sharpe +0.716** vs **passive EW +0.767** — **mean alpha −0.051**, 4/6
windows with positive alpha but the magnitudes are tiny. An 18-cell
robustness grid (rsi_n ∈ {7,14,21} × top_n ∈ {10,20,50} × n_tail ∈
{5,10}) confirms no cell clears alpha ≥ +0.05: the best is +0.016
(rsi_n=21, top_n=20, n_tail=10) and the median is −0.070. DSR-t is
−0.07 (n_obs=4680). Per the pre-registered bar this is
`confirmed-null`.

Do not pursue the "per-regime universe pre-selection lifts RSI to
`confirmed-OOS`" question. The baseline strategy has no edge over
passive EW to lift — there is no signal here that a universe filter
can amplify. Continue using DCA + vol_v3 sleeve as the canonical
deployment; if the meta-allocator panel needs a counter-trend slot,
the lever is not the RSI strategy.

## Why this finding existed (and why it didn't already)

The `regime` app exposes three strategy heads — `regime` (CWT-power
divergence ranking), `scalogram` (direction−momentum×coherence), and
`rsi` (top-N-most-oversold). The first two have been swept hard via
Optuna and have leaderboard rows. The `rsi` head was wired into the
trainer's dispatch and into the live harness (`regime live --strategy
rsi`) but never formally evaluated. Until 2026-05-25, the regime app's
RSI strategy was an unstamped runtime branch — it could be deployed
via the live CLI but had no OOS receipt anywhere in the docs.

The trigger was a research request: "would per-regime universe
pre-selection lift `weights_rsi` to `confirmed-OOS`?" The honest
prior-step is: do we even know the universe-agnostic baseline? We did
not. This finding fills the receipt.

## Eval setup

- **Strategy under test.** `regime.trainer.weights_rsi(prices,
  lookback=252, n_tail, top_n, rsi_n)` — score each (date, ticker) by
  the trailing-`n_tail` mean of Wilder RSI(`rsi_n`); pick `top_n`
  lowest-scoring names (most persistently oversold), equal-weight
  inside the basket. Strategy is pure-numpy and fully causal (cumsum
  over Wilder RSI).
- **Universe.** `stooq_us_long` (312 names, full history 2000-01-03
  → 2025-12-11) — the canonical wide-universe panel used by `gate`,
  `cfr`, and the per-regime-universe-oracle.
- **Walk-forward.** 6 windows, 1260-bar train / 780-bar val / 780-bar
  step. Train windows are NOT used by the RSI strategy (no
  hyperparameter is fit per-window); they exist purely to align the
  windowing convention with prior arcs (`gate`, `cfr`).
- **Trading.** rebal_days=20, commission_bps=10. One-sided turnover
  cost on each rebal (full L1 entry on the first rebal, 0.5×L1Δ
  thereafter). Returns are lagged by 1 bar from weight construction
  (no same-bar peek).
- **Benchmark.** Passive EW rebalanced on the same cadence over the
  *active* (non-NaN) names. Canonical per
  [`passive-ew-benchmark`](passive-ew-benchmark.md).
- **Hyperparameter grid (robustness, not a search).** 18 cells:
  rsi_n ∈ {7, 14, 21} × top_n ∈ {10, 20, 50} × n_tail ∈ {5, 10}.
  Median-Sharpe cell reported alongside the canonical (rsi_n=14,
  top_n=20, n_tail=5) headline.

### Pre-registered verdict bar (LOCKED before running)

| label | criteria |
|---|---|
| `confirmed-OOS` | mean val alpha vs EW ≥ +0.20 Sharpe **AND** ≥4/6 positive alpha windows **AND** DSR-t > +1.5 |
| `partial-OOS`   | mean val alpha vs EW ≥ +0.05 Sharpe **AND** ≥3/6 positive alpha windows |
| `confirmed-null` | alpha < +0.05 Sharpe **AND** DSR-t < +1.0 |
| `reversed-OOS`  | mean val alpha < −0.10 Sharpe |
| `diagnostic`    | anything else |

The pre-registered bar is persisted as a string field in
`Output/rsi-universe-agnostic-walkforward.npz` (`pre_registered_bar`).

## Per-window results — canonical config (rsi_n=14, n_tail=5, top_n=20)

| win | val_start | val_end | RSI Sharpe | EW Sharpe | alpha | RSI maxDD |
|---:|---|---|---:|---:|---:|---:|
| 0 | 2005-01-06 | 2008-02-12 | +1.012 | +0.708 | **+0.304** | −0.212 |
| 1 | 2008-02-13 | 2011-03-17 | +0.475 | +0.466 | +0.009 | −0.678 |
| 2 | 2011-03-18 | 2014-04-24 | +0.595 | +1.017 | **−0.422** | −0.332 |
| 3 | 2014-04-25 | 2017-05-30 | +0.228 | +0.925 | **−0.697** | −0.328 |
| 4 | 2017-05-31 | 2020-07-06 | +0.909 | +0.440 | **+0.469** | −0.475 |
| 5 | 2020-07-07 | 2023-08-10 | +1.075 | +1.045 | +0.030 | −0.172 |
| **mean** | | | **+0.716** | **+0.767** | **−0.051** | |

4/6 windows are positive-alpha but the two negative windows (w2, w3)
have alpha magnitudes (−0.42, −0.70) that swamp the four positives.
The mean RSI Sharpe of +0.716 looks superficially attractive but
underperforms passive EW by ~5bps Sharpe per year on the same
universe and rebal cadence.

## Robustness grid — 18 cells

| rsi_n | top_n | n_tail | mean RSI Sharpe | mean alpha | pos α windows |
|---:|---:|---:|---:|---:|---:|
| 7  | 10 | 5  | +0.640 | −0.127 | 3 |
| 7  | 10 | 10 | +0.651 | −0.116 | 2 |
| 7  | 20 | 5  | +0.686 | −0.081 | 3 |
| 7  | 20 | 10 | +0.597 | −0.169 | 2 |
| 7  | 50 | 5  | +0.721 | −0.045 | 2 |
| 7  | 50 | 10 | +0.708 | −0.059 | 2 |
| 14 | 10 | 5  | +0.587 | −0.180 | 3 |
| 14 | 10 | 10 | +0.633 | −0.134 | 2 |
| 14 | 20 | 5  | +0.716 | −0.051 | 4 | ← canonical
| 14 | 20 | 10 | +0.682 | −0.085 | 4 |
| 14 | 50 | 5  | +0.742 | −0.024 | 4 |
| 14 | 50 | 10 | **+0.747** | −0.020 | 4 |
| 21 | 10 | 5  | +0.604 | −0.163 | 2 |
| 21 | 10 | 10 | +0.679 | −0.088 | 4 |
| 21 | 20 | 5  | +0.725 | −0.041 | 3 |
| 21 | 20 | 10 | +0.783 | **+0.016** | 3 | ← best
| 21 | 50 | 5  | +0.772 | +0.006 | 4 |
| 21 | 50 | 10 | +0.751 | −0.015 | 3 |

- **Grid alpha spread:** min −0.180, max +0.016, median −0.070.
- **No cell** clears the `partial-OOS` threshold (+0.05 alpha).
- The best cell (rsi_n=21, top_n=20, n_tail=10) clears alpha=0 by
  +0.016 Sharpe — well within single-window noise.
- The pattern across rsi_n is monotone: longer RSI periods → less
  jumpy oversold signal → marginally better alpha but still
  alpha-negative on the median cell.

## DSR-t and ladder placement

Cross-arc DSR-t via `compute_dsr.py` (`n_trials=18` deflation,
`sharpe_std_ann=0.072` workspace default, overlay mode with passive
EW as benchmark):

```
arc                      mode       trials   annSh  DSR-t
rsi-universe-agnostic    overlay        18  +0.264  -0.813
```

The "annSh +0.264" is the annualized Sharpe of the (RSI − EW) edge
stream; the negative DSR-t reflects that the edge stream's mean is
negative once standardized against its own variance and deflated for
the 18-cell grid. The pre-reg's DSR-t < +1.0 criterion is satisfied
with margin.

## Mechanism — why the strategy fails

Mean-reversion on cross-sectional RSI rank picks names that have
*recently* trended down. Over the 2000-2025 window two effects
overpower the implied reversal premium:

1. **Persistence of downtrends.** On `stooq_us_long`'s broad
   mid/large-cap survivor universe, names with the lowest 5-day-mean
   RSI(14) are typically in confirmed downtrends, not transient
   oversold dips. Buying them at the next rebal earns the negative
   drift, not the bounce. This is visible in windows w2 and w3
   (2011-2017 bull market) where the most oversold names were the
   worst performers; the strategy is short the winners and long the
   losers by construction.
2. **Cost drag relative to a low-turnover EW benchmark.** Top_n=20 of
   312 means ~93% of the basket churns every rebal cycle in the
   worst case; at 10bps one-sided turnover that's ~9bps per rebal,
   ~14 rebals/year = ~125bps/yr drag. Passive EW on the same universe
   rotates only on entrance/exit of names from the active set —
   single-digit bps per year.

Combined: a tiny in-sample reversal premium minus turnover drag minus
trend-bleed yields net-negative alpha vs the EW benchmark on this
universe.

## What this means for the per-regime question

The user's original question ("would per-regime universe
pre-selection lift RSI to `confirmed-OOS`?") is **null-conditioned by
this finding**. Per-regime pre-selection works by routing the
strategy to its strongest sub-regime — but the universe-agnostic
baseline has no positive sub-region either. The best window
(w0 2005-2008, alpha +0.304) is *not* the lowest-vol or
highest-vol regime; the second-best (w4 2017-2020, alpha +0.469) is
the COVID-era tail. The two regime-winners do not share an obvious
macro feature that would let a causal regime detector pick them.

Pre-selection can amplify a real positive signal across regimes; it
cannot manufacture a positive signal where none exists in the base
strategy. Per the CLAUDE.md verdict-table for `confirmed-null`:

> Stop testing variations of the same lever — find an orthogonal one.
> Different prediction problem, different feature class, different
> operational use.

Step 1 (per-regime universe Optuna over the RSI strategy) is **not
triggered** by this Step 0 result. The next experiment is not a
refinement of the regime app's RSI head.

## Proposed next-experiment per CLAUDE.md verdict-table

For `confirmed-null`, the canonical next-step is an orthogonal lever.
The `regime` app currently has two formally-evaluated heads (`regime`
CWT-divergence ranking, `scalogram` direction-momentum-coherence) and
one falsified head (`rsi`, this finding). The orthogonal levers
remaining:

1. **Retire the RSI head from the `regime` app.** No deployable path
   was ever justified for it; keeping it in the dispatch table
   invites accidental live-deployment of a null strategy. Lowest-cost
   action.
2. **Pair-spread mean-reversion (`apps/pairs`).** Already
   `confirmed-null` v0 — see `apps/pairs/`. Same general mean-revert
   premise, different feature class (pair spreads vs single-name RSI);
   the prior null suggests this lever is exhausted on free
   public-equity data.
3. **Counter-trend on a different venue.** Per the
   `factor-crypto-venue` finding, the deterministic indicator grid
   does not transfer to crypto cross-section at H=5d. The same is
   plausibly true of RSI mean-reversion. Not worth running.

The recommended action is **(1) — retire the regime-app `rsi`
strategy.** The dispatch + live-CLI surface should be reduced to
`regime` and `scalogram` to avoid surfacing a null strategy through
the live harness. Left as a follow-up gated by maintainer choice;
this finding records the basis.

## Reproduce

```bash
uv run python apps/regime/scripts/rsi_universe_agnostic.py
```

Wall time: ~70s local (Intel Mac, no Modal). Caffeinate optional.

## Artifacts

- Driver: `apps/regime/scripts/rsi_universe_agnostic.py`.
- NPZ: `Output/rsi-universe-agnostic-walkforward.npz` (carries
  `oos_block_returns`, `oos_ew_returns`, `pre_registered_bar`,
  `periods_per_year=252`, `verdict='confirmed-null'`).
- JSON summary: `Output/rsi-universe-agnostic-walkforward.json`
  (per-window + grid).
- DSR ladder entry: `compute_dsr.py` ArcSpec
  `key='rsi-universe-agnostic'`, n_trials=18, mode='overlay',
  benchmark_key='oos_ew_returns'.

## Master walk-forward log pointer

[`confirmed-null`](../leaderboard.md#verdict-labels) — see the
2026-05-25 leaderboard row for the regime-app RSI universe-agnostic
walk-forward.
