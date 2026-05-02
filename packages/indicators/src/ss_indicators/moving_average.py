"""Moving-average primitives: SMA, EMA, rolling_std — pure numpy.

All accept a `(T, ...)` array and operate on the leading time axis. SMA
and rolling_std use a cumsum trick (O(T) regardless of window size); EMA
is time-recurrent so uses a Python loop along axis 0.
"""

from __future__ import annotations

import numpy as np


def sma(x: np.ndarray, window: int) -> np.ndarray:
    """Causal simple moving average of length `window` over axis 0.

    Expanding window during warm-up (matches pandas
    `rolling(window, min_periods=1).mean()`).
    """
    x = np.asarray(x)
    T = x.shape[0]
    cs = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=x.dtype),
                         np.cumsum(x, axis=0)], axis=0)
    idx = np.arange(T)
    lo = np.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(x.dtype)
    extra_dims = (None,) * (x.ndim - 1)
    return (cs[idx + 1] - cs[lo]) / counts[(slice(None),) + extra_dims]


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sample-std of length `window` over axis 0.

    Expanding window during warm-up; matches
    `pandas.rolling(window, min_periods=1).std(ddof=0)` (population std).

    Internally promotes to float64 for the `mu2 - mu**2` step (float32
    cumsum suffers catastrophic cancellation here at typical price
    scales — pandas's reference impl does the same), then casts the
    result back to the input dtype.
    """
    x_in = np.asarray(x)
    out_dtype = x_in.dtype
    x = x_in.astype(np.float64, copy=False)
    T = x.shape[0]
    cs = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=np.float64),
                         np.cumsum(x, axis=0)], axis=0)
    cs2 = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=np.float64),
                          np.cumsum(x ** 2, axis=0)], axis=0)
    idx = np.arange(T)
    lo = np.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(np.float64)
    extra_dims = (None,) * (x.ndim - 1)
    cnt = counts[(slice(None),) + extra_dims]
    mu = (cs[idx + 1] - cs[lo]) / cnt
    mu2 = (cs2[idx + 1] - cs2[lo]) / cnt
    return np.sqrt(np.maximum(mu2 - mu ** 2, 0.0)).astype(out_dtype, copy=False)


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with smoothing factor 2/(span+1).

    Time-recurrent so implemented as a Python loop along axis 0. The
    first sample is the seed (matches pandas
    `ewm(span=..., adjust=False)`). For (T, N) inputs each step is a
    vectorized op over N tickers, so per-step cost amortizes over the
    cross-section.
    """
    x = np.asarray(x)
    alpha = np.asarray(2.0 / (span + 1), dtype=x.dtype)
    one_minus = (1.0 - alpha).astype(x.dtype)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, x.shape[0]):
        out[t] = alpha * x[t] + one_minus * out[t - 1]
    return out
