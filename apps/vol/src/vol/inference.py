"""Pure forward pass for the vol-v3 predictor.

Stateless functions: feature-frame in, ranked picks out. No I/O, no
Alpaca, no FRED — those happen in `vol.live`. This module is what an
operator can unit-test in isolation against the v2-dolthub-oos
training output.

The predictor is the v2-dolthub-oos 4-feature OLS:
  iv_rv_gap_pred(sym, t) = b0
    + b1 * z(iv_over_hv)
    + b2 * z(iv_z)
    + b3 * z(iv_change_4w)
    + b4 * z(hv_change_4w)

where z(.) means z-scored using `feat_mean` / `feat_std` from the
checkpoint (frozen train statistics).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol.persist import LIVE_FEATURE_NAMES, VolCheckpoint


def predict_iv_rv_gap(features: pd.DataFrame, cp: VolCheckpoint) -> pd.Series:
    """Predict iv_rv_gap for one cross-section (today's snapshot).

    Parameters
    ----------
    features : pd.DataFrame
        Rows = symbols; columns include at least the four feature
        names in `LIVE_FEATURE_NAMES` (extra columns are ignored).
    cp : VolCheckpoint
        Frozen predictor coefs + train z-score stats.

    Returns
    -------
    pd.Series indexed by symbol, dtype float64, predicted iv_rv_gap.
    Symbols with any NaN feature are dropped.
    """
    missing = [f for f in LIVE_FEATURE_NAMES if f not in features.columns]
    if missing:
        raise ValueError(f'features DataFrame missing columns: {missing}')

    X = features[LIVE_FEATURE_NAMES].astype(np.float64)
    finite_rows = X.notna().all(axis=1)
    X = X[finite_rows]
    if X.empty:
        return pd.Series(dtype=np.float64)

    mu = np.asarray(cp.feat_mean, dtype=np.float64)
    sd = np.asarray(cp.feat_std,  dtype=np.float64)
    Xz = (X.values - mu) / sd                           # (n, k)
    Xa = np.concatenate([Xz, np.ones((len(Xz), 1))], axis=1)
    pred = Xa @ np.asarray(cp.coefs, dtype=np.float64)  # (n,)
    return pd.Series(pred, index=X.index, name='pred_iv_rv_gap')


def select_top_k(
    pred: pd.Series, top_k: int, *, eligible: list[str] | None = None,
) -> pd.Series:
    """Take the top-K predicted iv_rv_gap.

    `eligible` is an optional filter (e.g. only names with current
    optionable contracts that pass strangle's liquidity gates). If
    provided, restrict to that set BEFORE picking top-K.
    """
    if eligible is not None:
        pred = pred.reindex(eligible).dropna()
    return pred.sort_values(ascending=False).head(top_k)


def gate_fires(
    vix_series: pd.Series, lookback_trading_days: int,
    as_of: pd.Timestamp | None = None,
) -> tuple[bool, float, float]:
    """VIX 126d-rolling-median regime gate (v3 deployment recipe).

    Returns (fires, vix_now, rolling_median_now).

    The gate is binary: VIX[t] > median(VIX[t-N:t]) fires. If the
    rolling median can't be computed (insufficient history), returns
    `fires=False` and the operator can choose to abort or proceed.
    """
    if as_of is None:
        as_of = vix_series.index[-1]
    s = vix_series.loc[:as_of].dropna()
    if s.size < lookback_trading_days // 2:
        return False, float('nan'), float('nan')
    vix_now = float(s.iloc[-1])
    window = s.iloc[-lookback_trading_days:]
    med = float(window.median())
    return (vix_now > med), vix_now, med


__all__ = ['predict_iv_rv_gap', 'select_top_k', 'gate_fires']
