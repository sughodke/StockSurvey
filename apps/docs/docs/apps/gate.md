# Gate

Aggregate drawdown forecaster used as an EW-exposure regime gate.
Numpy-only (no tinygrad needed at this scale): single time series,
small feature stack, OLS predictor with closed-form solution.

## Why this app exists

The first test of the
[different-prediction-problem](../TODO/different-prediction-problem.md)
pivot. The
[2026-05-10 confirmed-null arc](../leaderboard.md#verdict-labels)
on `apps/factor` (passive-EW gate, long-short constructor,
Sharpe/IR loss-pivot) showed cross-sectional return prediction
can't beat passive at our +0.005 to +0.012 IC scale. The
diagnostic conclusion: the binding constraint is *signal magnitude
of cross-sectional return prediction*, not how that signal is
optimized or deployed.

Drawdown forecasting is a structurally different prediction
problem:

- **Time-series, not cross-sectional.** One target value per
  date for the universe-aggregate EW return series.
- **Different target.** Forward 20-day max drawdown of EW
  cumulative log return, not next-period log return.
- **Different deployment shape.** A scalar gate `g(t) ∈ [0,1]`
  that scales overall exposure (1.0 = fully invested in EW;
  0.0 = flat). Consumed by EW (default) or any other strategy
  that wants regime-aware sizing.

## Pipeline

`apps/gate/src/gate/` modules:

- **`aggregate.py`** — `build_ew_aggregate(prices, min_active=10)`
  returns the per-date EW simple + log return series; weights
  are 1/N over tickers with both a current and prior valid
  close. Basket grows as tickers come online; delistings drop
  out at last quoted price.
  `build_aggregate_features(agg)` returns 10 trailing features:
  `vol_{5,20,60}`, `ret_{5,20,60}`, `tdd_{20,60}`, `vol_term`
  (5d − 60d realized vol), `breadth` (% of universe online).
- **`target.py`** — `forward_max_drawdown(log_ret, horizon=20)`
  for each `t` returns the max peak-to-trough drawdown of
  `cumsum(log_ret[t+1:t+horizon+1])`. Trailing entries get NaN.
- **`predictor.py`** — `train_predictor(X, y, names)` fits
  numpy OLS on z-scored features + intercept. `apply_gate(pred,
  threshold, mode)` converts predictions to gate ∈ [0, 1] —
  `binary` (0 or 1) or `sigmoid` (graduated).
- **`backtest.py`** — `gated_returns(ew_ret, gate)` applies
  gate. `evaluate_gated_arm` reports Sharpe / Sortino / CAGR /
  MaxDD / avg exposure / # transitions.

`apps/gate/scripts/`:

- **`run_baseline.py`** — single-split phase-2 (matched to
  passive-EW benchmark window) with threshold sweep across
  train-pred quantiles.
- **`run_walkforward.py`** — 6-window walk-forward, threshold
  chosen on train per window (no peeking), pre-registered cuts
  applied.

## v0 result — `partial-OOS`

See [`gate-drawdown-v0`](../findings/gate-drawdown-v0.md) for
detail. Headline: mean val Pearson r = **+0.264** across 6
walk-forward windows — ~25-50× the cross-sectional return IC
ceiling, validating that the prediction problem is genuinely
orthogonal. But monetization is fragile at v0:

| arm | mean alpha | pos windows | window-1 (GFC) alpha |
|---|---:|---:|---:|
| binary q=0.95 | **+0.067** | **4/6** | +0.32 |
| sigmoid q=0.95 | +0.059 | 2/6 | +0.46 (peak) |
| binary q=0.85 (over-flatting) | −0.114 | 1/6 | −0.28 |

Window 1 (2008-02 → 2011-03) catches +0.32 alpha by flatting to
51% during the GFC; calm windows accumulate small false-positive
losses (−0.02 to −0.07) that nearly cancel it. Mean alpha within
±0.10 noise band.

## v1 follow-ups (parked)

Not blocking the broader pivot — return when `apps/pairs` and
`apps/vol` results are in:

- **Regime-conditional gate.** Only enable the gate when
  trailing vol exceeds a percentile threshold of train vol —
  skip most calm-period false positives.
- **Two-stage classifier + sizing.** First predict
  `P(drawdown_event)`, then size the gate proportional to
  predicted probability × predicted magnitude. Decouples
  regime detection from exposure sizing.
- **Better features.** Cross-sectional dispersion (a VIX-like
  proxy from inside the universe), term-structure of realized
  vol, breadth indicators (% above 200dma).
- **Non-linear predictor.** MLP via tinygrad (port from
  `apps/factor`'s linear → MLP path) for better extrapolation
  on rare events.

## Reuses + dependencies

- `ss_loaders.load_stooq_matrix` — universe loader.
- `ss_portfolio.metrics` — Sharpe / Sortino / CAGR / MaxDD.
- numpy + pandas only — no tinygrad, no scipy, no statsmodels.

## Caveats

- v0 trades-cost not modeled. Binary gate flips 10-40 times per
  780-bar val window at q=0.95; at 10 bps round-trip per flip
  that's roughly 0.13-0.5% per year of friction. Doesn't change
  verdict direction but shaves the modest +0.07 alpha closer to
  the noise band.
- OLS linear can't extrapolate to drawdowns larger than what
  appeared in train. Window 1's GFC alpha came from training
  data containing similar features; other val windows had no
  analogous training signal for their drawdown event.
- Single feature stack of 10 simple aggregates. v1 follow-ups
  list richer features.
