"""Rolling 3rd / 4th standardized moments of log returns — pure numpy.

`rolling_skew` and `rolling_kurt` mirror `apps/lie.ticker_features`'s
trailing-window return-distribution moments. Both are nonlinear in the
return series (cubed / quartic deviations), so a linear bandpass CWT
cannot reconstruct them — this is the same point as `drawdown`. They
ship as `apps/replay` reconstruction heads partly so the CNN's R^2 here
gives an empirical floor on what the wavelet representation can carry.

Computed via cumsum prefixes of (r, r^2, r^3, r^4), so per-window stats
are O(1) and the whole series is O(T) — same as `realized_vol`.
"""

from __future__ import annotations

import numpy as np


def rolling_skew(prices: np.ndarray, n: int = 63) -> np.ndarray:
    """Standardized 3rd moment (Pearson skew) of trailing-n log returns."""
    return _rolling_standardized_moment(prices, n, order=3)


def rolling_kurt(prices: np.ndarray, n: int = 63) -> np.ndarray:
    """Excess kurtosis (`mean(z^4) - 3`) of trailing-n log returns."""
    return _rolling_standardized_moment(prices, n, order=4)


def _rolling_standardized_moment(
    prices: np.ndarray, n: int, order: int,
) -> np.ndarray:
    """Vectorized rolling 3rd or 4th standardized moment via cumsum prefixes.

    Window convention matches `vol_norm_momentum`: at output index `t`,
    uses the n log returns ending with `log(p[t] / p[t-1])`. Valid for
    `t >= n`; NaN otherwise and where the trailing window is degenerate
    (zero variance).

    Algebra (raw moments `M_k = sum_window r^k`, `mu = M1/n`,
    `mu_k_central = E[(r - mu)^k]`):

        mu2_c = M2/n - mu^2
        mu3_c = M3/n - 3*mu*M2/n + 2*mu^3
        mu4_c = M4/n - 4*mu*M3/n + 6*mu^2*M2/n - 3*mu^4

    Skew = mu3_c / mu2_c^(3/2);  excess kurt = mu4_c / mu2_c^2 - 3.
    """
    if order not in (3, 4):
        raise ValueError(f'order must be 3 or 4, got {order}')
    if n < 3:
        raise ValueError(
            f'_rolling_standardized_moment n must be >= 3, got {n}')
    p = np.asarray(prices, dtype=np.float64)
    T = p.shape[0]
    out = np.full(p.shape, np.nan, dtype=np.float64)
    if T <= n:
        return out

    log_p = np.log(p)
    rets = np.diff(log_p, axis=0)                              # (T-1, ...)

    zero = np.zeros((1,) + rets.shape[1:], dtype=np.float64)
    cs1 = np.concatenate([zero, np.cumsum(rets, axis=0)], axis=0)
    cs2 = np.concatenate([zero, np.cumsum(rets ** 2, axis=0)], axis=0)
    cs3 = np.concatenate([zero, np.cumsum(rets ** 3, axis=0)], axis=0)
    cs4 = (np.concatenate([zero, np.cumsum(rets ** 4, axis=0)], axis=0)
           if order == 4 else None)

    t_idx = np.arange(n, T)
    s1 = cs1[t_idx] - cs1[t_idx - n]
    s2 = cs2[t_idx] - cs2[t_idx - n]
    s3 = cs3[t_idx] - cs3[t_idx - n]
    mu = s1 / n
    m2 = s2 / n
    m3 = s3 / n
    mu2_c = np.maximum(m2 - mu ** 2, 0.0)

    if order == 3:
        mu3_c = m3 - 3 * mu * m2 + 2 * mu ** 3
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(mu2_c > 1e-20, mu2_c ** 1.5, np.nan)
            vals = mu3_c / denom
    else:
        s4 = cs4[t_idx] - cs4[t_idx - n]
        m4 = s4 / n
        mu4_c = m4 - 4 * mu * m3 + 6 * mu ** 2 * m2 - 3 * mu ** 4
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.where(mu2_c > 1e-20, mu2_c ** 2, np.nan)
            vals = mu4_c / denom - 3.0

    out[n:] = vals
    return out
