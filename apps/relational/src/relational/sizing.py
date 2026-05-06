"""Position-sizing overlays for long-equity baskets.

Two transforms that operate on a `(n_dates, n_tickers)` weight DataFrame
(typically the output of an existing `weights_regime`-style top-N
builder, where each row is sparse and sums to 1.0):

  * `risk_parity_weights` — replace within-basket equal-weighting with
    1 / vol_i normalization, so every selected position contributes
    equal *diagonal-covariance* risk. Tickers with non-finite trailing
    vol are dropped from the basket and remaining mass renormalized.
  * `vol_target_weights` — scale each row by `target_vol / σ_p`, where
    σ_p is the diagonal-cov portfolio-vol estimate
    `sqrt(sum_i (w_i × σ_i)²)`. Capped at `max_leverage` so we don't
    blow up gross exposure during very low-vol regimes.

Both use trailing realized vol from `ss_features.vol.realized_vol` (a
causal rolling std of log returns), annualized via √252. Diagonal-cov
ignores cross-asset correlation; this is intentionally a first-pass
approximation. With the Phase-2 mega-cap basket the cross-correlations
are high enough that diagonal will *under*-estimate portfolio vol,
making the leverage applied here a slight over-shoot. Replace with a
full realized covariance from a trailing window when this matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_features import realized_vol_matrix


def _trailing_realized_vol_annualized(
    prices: pd.DataFrame, window: int,
) -> pd.DataFrame:
    """Per-ticker causal rolling std of log returns × √252."""
    out = realized_vol_matrix(prices, window, annualize=True)
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def risk_parity_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    vol_window: int = 60,
) -> pd.DataFrame:
    """Within-basket 1/vol risk-parity reweighting.

    For each row, redistribute mass across non-zero positions
    proportionally to 1/σ_i. Output rows still sum to ~1 (less if some
    selected positions have NaN trailing vol and get dropped).
    """
    rv = _trailing_realized_vol_annualized(prices, vol_window)
    rv = rv.reindex(index=weights.index, columns=weights.columns)
    w = weights.values
    v = rv.values
    out = np.zeros_like(w, dtype=np.float64)
    for t in range(w.shape[0]):
        active = (w[t] > 0) & np.isfinite(v[t]) & (v[t] > 0)
        if not active.any():
            continue
        inv_vol = np.where(active, 1.0 / v[t], 0.0)
        s = inv_vol.sum()
        if s > 0:
            out[t] = inv_vol / s
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


def vol_target_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    target_vol: float = 0.15,
    vol_window: int = 60,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Row-wise scaling toward `target_vol` (annualized fraction).

    Diagonal-cov estimate: `σ_p ≈ √Σ_i (w_i σ_i)²`. Leverage is clipped
    to `[0, max_leverage]`. When `σ_p == 0` (empty/NaN row) the row is
    passed through unchanged.
    """
    rv = _trailing_realized_vol_annualized(prices, vol_window)
    rv = rv.reindex(index=weights.index, columns=weights.columns)
    w = weights.values
    v = rv.values
    out = np.zeros_like(w, dtype=np.float64)
    for t in range(w.shape[0]):
        wv = w[t] * np.where(np.isfinite(v[t]), v[t], 0.0)
        sigma_p = float(np.sqrt(np.nansum(wv * wv)))
        if sigma_p > 0:
            lev = min(target_vol / sigma_p, max_leverage)
            out[t] = w[t] * lev
        else:
            out[t] = w[t]
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)
