"""Smoke test: feature panel builds + sane shapes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from e2e_portfolio.data import (
    F_ASSET, F_MACRO, K_FORWARD, PHASE4D_TICKERS, T_LOOKBACK,
    prepare_panel,
)


def _synthetic_close(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n_days, 13))
    px = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range('2010-01-01', periods=n_days)
    return pd.DataFrame(px, index=idx, columns=PHASE4D_TICKERS)


def _synthetic_macro(idx: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(idx)
    df = pd.DataFrame({
        'vix': 15 + rng.normal(0, 3, n).cumsum() * 0.01,
        'vix_pct_252': rng.uniform(0, 1, n),
        'slope_10y_3m': rng.normal(1, 0.5, n),
        'credit_baa': rng.normal(2, 0.3, n),
    }, index=idx)
    return df


def test_prepare_panel_shapes():
    close = _synthetic_close(500)
    macro = _synthetic_macro(close.index)
    panel = prepare_panel(close, macro_panel=macro)
    # Expected n_eff = T - T_lookback - K_forward = 500 - 60 - 20 = 420 (approx)
    assert panel.X_assets.ndim == 4
    n_eff, N, T_l, F_a = panel.X_assets.shape
    assert N == 13
    assert T_l == T_LOOKBACK
    assert F_a == F_ASSET
    assert panel.X_macro.shape == (n_eff, T_LOOKBACK, F_MACRO)
    assert panel.fwd_ret.shape == (n_eff, 13)
    assert len(panel.dates) == n_eff
    # Features should be finite.
    assert np.isfinite(panel.X_assets).all()
    assert np.isfinite(panel.X_macro).all()
    assert np.isfinite(panel.fwd_ret).all()
