"""Realized volatility — causal rolling std of log returns.

Lifted from ``ss_notebook.replay.features`` so both the SSL pretrain
(apps/notebook) and the cross-sectional scorer (apps/factor) can build
deterministic-vol features without depending on each other.
"""
from __future__ import annotations

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Per-bar log returns padded with NaN at index 0 so the array
    aligns with the price series."""
    log_p = np.log(prices.astype(np.float64))
    return np.concatenate([[np.nan], np.diff(log_p)])


def realized_vol(prices: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std of log returns over the trailing `window` bars.

    Output is NaN until the window is full. Used as a backbone-pretraining
    target — it's what the rolling-z-normed scalogram is best positioned
    to recover (squared-coefficient structure survives the z-norm) and a
    known cross-sectional return predictor in its own right.
    """
    if window < 2:
        raise ValueError(f'realized_vol window must be >= 2, got {window}')
    rets = log_returns(prices)            # (n,) with rets[0] = NaN
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    rets_clean = np.where(np.isnan(rets), 0.0, rets)
    cs = np.cumsum(np.concatenate([[0.0], rets_clean]))
    cs2 = np.cumsum(np.concatenate([[0.0], rets_clean ** 2]))
    for i in range(window, n):
        s = cs[i + 1] - cs[i + 1 - window]
        s2 = cs2[i + 1] - cs2[i + 1 - window]
        m = s / window
        v = max(s2 / window - m * m, 0.0)
        out[i] = np.sqrt(v)
    return out
