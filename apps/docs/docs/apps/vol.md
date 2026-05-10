# Vol

Implied vol surface predictor. Numpy-only OLS over 10 features
(skew, smile curvature, multi-horizon IV/HV ratio, OI imbalance,
single-name vs VIX spread, normalized strike spread) → forward
20-day IV/RV gap. Trade signal: top-quantile gated short-vol.

## Why this app exists

Third app in the
[different-prediction-problem](../TODO/different-prediction-problem.md)
pivot, after [`apps/gate`](gate.md) (`partial-OOS`) and
[`apps/pairs`](pairs.md) (`confirmed-null` per pre-reg).

The
[`NO_OPTIONS.md`](https://github.com/sughodke/StockSurvey/blob/master/apps/relational/NO_OPTIONS.md)
arc tested 9 scorers (5 CWT-dislocation + 4 brainstorm) using
only `ATM_IV` features against the forward IV/RV gap on Phase-2
+ stooq_us_long. All settled in t-stats `[-1.08, +1.49]`, with
the universe-wide short-vol baseline (Sharpe 0.51, MaxDD -83%)
dominating all gated variants. Conclusion at the time: "the IV
market efficiently incorporates the dislocation information."

`apps/vol` tests the user-chosen "untested feature class" angle:
the rich gauss314 schema (full strike grid, multi-horizon HV,
OI per side, VIX) carries surface-shape features that the
prior arc didn't compute — skew, smile, IV/HV ratio,
OI imbalance, VIX-spread.

## Pipeline

`apps/vol/src/vol/`:

- **`data.py`** — `load_gauss314_full()` reads the full-schema
  CSV from `.iv-cache/data_IV_USA.csv` (the `ss_iv.load_atm_iv`
  loader returns only `ATM_IV`; we need DITM/ITM/sITM/ATM/sOTM/
  OTM/DOTM_IV + hv_20...hv_200 + OI + VIX). `build_vol_features`
  computes the 10-feature stack per `(date, symbol)`.
- **`target.py`** — `forward_iv_rv_gap(panel, horizon=20)` gives
  per-`(date, symbol)` `iv_rv_gap = ATM_IV_t − hv_20_{t+20}`.
  Standard short-vol PnL convention matched to
  `ss_iv.short_vol_pnl_panel`.
- **`predictor.py`** — numpy OLS, z-scored features + intercept,
  same convention as `apps/gate/src/gate/predictor.py`.
- **`backtest.py`** — top-quantile gated short-vol PnL via
  `evaluate_gated_short_vol`; per-cell vol-points Sharpe.

`apps/vol/scripts/`:

- **`audit_data.py`** — Stage-0 data-quality audit per the TODO.
  Reports schema, density, IV/HV cleanliness, univariate
  feature-target Pearson r. Run before any walk-forward.
- **`run_walkforward.py`** — 5-window walk-forward (gauss314
  span limits to ~5y), pre-registered cuts.

## v0 result — `inconclusive` per pre-reg

See [`vol-surface-v0`](../findings/vol-surface-v0.md) for
detail. Headline: mean alpha **+0.089 per-cell-Sharpe**, **5/5
positive windows** — strongest directional consistency in the
prediction-problem-pivot arc, but mean just below the **+0.10**
marginal floor.

| win | val period | val Pearson r | unc Sh | gated Sh | alpha |
|---|---|---:|---:|---:|---:|
| 0 | 2021-01 → 2021-06 | +0.005 | +0.005 | +0.028 | +0.023 |
| 1 | 2021-06 → 2021-12 | +0.055 | +0.162 | +0.211 | +0.049 |
| 2 | 2021-12 → 2022-06 | +0.035 | +0.013 | +0.100 | +0.086 |
| 3 | 2022-06 → 2022-12 | **+0.268** | +0.177 | **+0.332** | **+0.155** |
| 4 | 2022-12 → 2023-06 | **+0.238** | +0.187 | **+0.321** | **+0.134** |
| **mean** | | **+0.120** | +0.109 | **+0.198** | **+0.089** |

**Methodological lesson:** univariate Pearson r per feature was
≤ +0.003 (audit looked dead), multivariate val Pearson r =
**+0.12 — 40× larger**. The surface signal lives entirely in
the joint structure, not in any single feature.

**Late windows carry the signal.** 2022-12 → 2023-06 (post-COVID
vol regime, Fed hikes, mega-cap dispersion) shows val r > +0.23;
the 2021 mega-cap melt-up era shows train R² = 0.000 / val r ≤
0.055. Regime-conditional like everything else we've tested.

## v1 follow-ups (parked)

- **Per-rebal portfolio aggregation** — replace per-cell Sharpe
  with per-rebal PnL → annualized portfolio Sharpe.
- **Costs-in-the-loop** — embed 100-1000 bps options friction
  in the gated PnL.
- **DoltHub extension** — proxy surface features from
  DoltHub `volatility_history` to extend the test through 2026.
- **MLP head** — multivariate r +0.12 + train R² > 0 in late
  windows suggests nonlinear structure to capture.
- **Universe restriction by liquidity** — top-100 by OI per
  date.

## Reuses + dependencies

- `ss_iv` (`packages/iv/`) — IV loaders + short-vol PnL
  conventions. Promoted from `apps/relational` 2026-05-10
  when `apps/vol` became the second consumer.
- `ss_loaders`, `ss_features`, `ss_portfolio.metrics`.
- numpy + pandas only. No tinygrad, no statsmodels.

## Caveats

- Per-cell Sharpe is a weak metric — no temporal aggregation
  or annualization. v1 needs the per-rebal portfolio aggregator.
- Costs not modeled in v0. The +0.089 alpha is gross of
  options friction (100-1000 bps round-trip) which would
  consume most of it. The "signal exists" claim is real; the
  "deployable strategy" claim requires the cost model.
- Data span limited to gauss314's 2019-10 → 2023-07 (~4 years,
  938 trading days). 5 windows is fewer than the equity apps'
  6 windows.
- 3,877 symbols enter the regression; many have thin options
  liquidity. v1 should restrict to top-N by OI per date.
