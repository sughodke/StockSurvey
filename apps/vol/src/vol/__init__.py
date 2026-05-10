"""apps/vol — implied vol surface features → forward IV/RV gap predictor.

Third app in the prediction-problem pivot off cross-sectional return
forecasting. NO_OPTIONS.md tested 9 scorers (5 CWT-dislocation + 4
brainstorm) using only single-IV-value features (`ATM_IV` from
DoltHub or gauss314); all settled in t-stats `[-1.08, +1.49]` on the
forward IV/RV gap and the universe-wide short-vol baseline (Sharpe
0.51, MaxDD -83%) dominated all gated variants. The user-chosen
follow-on:

> Test something the 12 phases didn't: term-structure spread, skew,
> single-name vs index VIX dispersion as the predictor.

That requires the **richer gauss314 schema** (full strike grid
`DITM_IV ... DOTM_IV`, multi-horizon HV `hv_20 ... hv_200`, OI per
side, VIX) — DoltHub's `volatility_history` is too thin (only
`iv_current` + `hv_current` per ticker per date).

v0 feature classes:

  - **Skew** — `(DOTM_IV − ATM_IV) / ATM_IV` and `(DITM_IV − ATM_IV)
    / ATM_IV`. Tail-IV vs ATM-IV ratios. Steeper skew historically
    correlates with downside risk pricing.
  - **Smile curvature** — `(DOTM_IV + DITM_IV − 2·ATM_IV) / ATM_IV`.
    The U-shape depth, normalized.
  - **IV/HV ratio at multiple horizons** — `ATM_IV / hv_20`, `/hv_60`,
    `/hv_120`. Direct vol-risk-premium magnitude estimators.
  - **HV term structure** — `hv_20 / hv_200`. Realized-vol mean
    reversion proxy.
  - **OI imbalance** — `puts_open_interest / (puts_open_interest +
    calls_open_interest)`. Positioning indicator.
  - **VIX spread** — `ATM_IV − VIX/100`. Single-name idiosyncratic
    vol vs market vol.
  - **Strike-spread** — `strikes_spread / ATM_IV`. Market-maker
    inventory risk proxy.

These are static cross-sectional features per `(date, ticker)`. The
predictor maps the feature stack to forward IV/RV gap via OLS
linear regression (numpy, no tinygrad — same scope discipline as
`apps/gate`).

Trade construction: short straddle when predicted IV-RV-gap is
positive (predict realized < IV → premium overpriced), long
straddle when negative. PnL accounting reuses
`ss_iv.short_vol_pnl_panel` for matched conventions with the prior
arc.

Pre-registered cuts (per [`TODO/apps-vol.md`](../../docs/TODO/apps-vol.md)):
  pass     : predictor lifts short-vol Sharpe ≥ baseline + 0.30 with
             ≥ 4/6 windows positive alpha
  marginal : baseline + (0.10, 0.30) → stratify by ticker liquidity
  fail     : < baseline + 0.10 → confirmed-null; the IV market
             efficiently incorporates surface-shape information just
             as it does single-IV scorer information per NO_OPTIONS.md

Public API:
  - `load_gauss314_full(cache_dir)` — load the full-schema CSV
  - `build_vol_features(panel)` — derive the feature stack
  - `forward_iv_rv_gap(iv, rv, horizon)` — supervised target
  - `train_predictor(X, y, names)` — OLS fit
  - `evaluate_predictor_walkforward(...)` — n-window rolling eval
"""
from vol.data import (
    Gauss314Panel, FEATURE_NAMES, build_vol_features, load_gauss314_full,
)
from vol.target import forward_iv_rv_gap
from vol.predictor import (
    PredictorResult, evaluate_r2, predict, train_predictor,
)
from vol.backtest import (
    GatedShortVolResult, evaluate_gated_short_vol,
)


__all__ = [
    'FEATURE_NAMES',
    'Gauss314Panel',
    'GatedShortVolResult',
    'PredictorResult',
    'build_vol_features',
    'evaluate_gated_short_vol',
    'evaluate_r2',
    'forward_iv_rv_gap',
    'load_gauss314_full',
    'predict',
    'train_predictor',
]
