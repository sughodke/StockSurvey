"""Vol-normalized cumulative log-return — pure numpy.

Mirrors `apps/lie.ticker_features`'s vol-normalized momentum shape feature:
the cumulative log-return over a trailing window, divided by the sample
stdev of those returns scaled by sqrt(window). Under a Brownian-motion
null this is approximately N(0, 1) at every bar, which is what makes it
a clean cross-sectional ranking signal.

Used by `apps/replay` as a reconstruction head so we can test whether
the rolling-z-normed CWT carries the information needed to predict
horizon-conditioned momentum (Test: if the CNN reconstructs vol-norm
momentum well, the CWT representation is rich enough; if it can't, that
empirically confirms the v4 head-to-head verdict that bandpass CWT
loses level/cumulative-return content).
"""

from __future__ import annotations

import numpy as np


def vol_norm_momentum(prices: np.ndarray, n: int = 21) -> np.ndarray:
    """Vol-normalized cumulative log-return over a trailing n-bar window.

    For each bar `t >= n`:

        sum_n_returns / (sigma_n * sqrt(n))

    where `sum_n_returns = log(p[t]) - log(p[t-n])` and `sigma_n` is the
    sample stdev of those `n` log returns. Output is same shape as
    `prices`; NaN for `t < n` and any window containing a NaN return.

    Vectorized via cumsum prefix differences — O(T) total. NaN
    propagation through cumsum is sticky (a single mid-series NaN ruins
    everything downstream), which is fine for replay's gap-free Stooq
    inputs and surfaces correctly via the downstream `valid` mask.
    """
    if n < 2:
        raise ValueError(f'vol_norm_momentum n must be >= 2, got {n}')
    p = np.asarray(prices, dtype=np.float64)
    T = p.shape[0]
    out = np.full(p.shape, np.nan, dtype=np.float64)
    if T <= n:
        return out

    log_p = np.log(p)
    rets = np.diff(log_p, axis=0)                              # (T-1, ...)

    zero = np.zeros((1,) + rets.shape[1:], dtype=np.float64)
    cs = np.concatenate([zero, np.cumsum(rets, axis=0)], axis=0)
    cs2 = np.concatenate([zero, np.cumsum(rets ** 2, axis=0)], axis=0)

    t_idx = np.arange(n, T)
    s1 = cs[t_idx] - cs[t_idx - n]
    s2 = cs2[t_idx] - cs2[t_idx - n]
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    sigma = np.sqrt(var)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = sigma * np.sqrt(n)
        vals = np.where(denom > 1e-12, s1 / denom, np.nan)
    out[n:] = vals
    return out
