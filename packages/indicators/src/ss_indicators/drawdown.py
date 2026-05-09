"""Log-drawdown from a rolling-window high — pure numpy.

Mirrors `apps/lie.ticker_features`'s drawdown shape feature: the log
ratio of today's price to the trailing-window peak. Always <= 0; 0
means today is at the high.

Drawdown is a max-statistic — fundamentally nonlinear in price, so a
linear bandpass CWT cannot reconstruct it from a finite combination of
coefficients. Used by `apps/replay` as a head precisely to surface that
gap: a low CNN R^2 here is direct evidence for the v4 verdict.
"""

from __future__ import annotations

import numpy as np


def drawdown_from_high(prices: np.ndarray, n: int = 63) -> np.ndarray:
    """Log-drawdown from the trailing-(n+1)-bar high, including today.

    For each bar `t >= n`:

        log(prices[t] / max(prices[t-n : t+1]))

    Window covers `n+1` bars (the n-bar lookback plus the current bar);
    matches `apps/lie.ticker_features.build_ticker_features`'s
    `drawdown_horizon` convention exactly. Output <= 0; NaN for `t < n`
    or windows containing NaN.

    O(T * n) via a simple time-axis loop — n is small (typically 21..252)
    and the per-step cost is one vectorized `max` over the trailing
    slice. scipy's `maximum_filter1d` would shave a log factor but the
    boundary-mode dance isn't worth the dependency on this hot path.
    """
    if n < 1:
        raise ValueError(f'drawdown_from_high n must be >= 1, got {n}')
    p = np.asarray(prices, dtype=np.float64)
    T = p.shape[0]
    out = np.full(p.shape, np.nan, dtype=np.float64)
    if T <= n:
        return out
    log_p = np.log(p)
    for t in range(n, T):
        peak = log_p[t - n: t + 1].max(axis=0)
        out[t] = log_p[t] - peak
    return out
