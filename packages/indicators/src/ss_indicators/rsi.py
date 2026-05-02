"""Wilder relative strength index — pure numpy.

Two flavors:
  - `rsi(prices, n)`            — matrix-form RSI(n) over axis 0,
                                  vectorizes over trailing dims
  - `rsi_strided(prices, n, w)` — RSI(n) computed over stride-w price
                                  changes (1-D only); used by the
                                  multi-head CNN trainer for FiLM
                                  conditioning over the (n, w) grid
"""

from __future__ import annotations

import numpy as np


def rsi(prices: np.ndarray, n: int = 7) -> np.ndarray:
    """Wilder RSI of period `n` over axis 0.

    Output range [0, 100]. Positions before index `n` are filled with the
    neutral value 50. Vectorized over all trailing axes (e.g. `(T, N)`
    input gives `(T, N)` output). Time-recurrent so the smoothing tail
    is a Python loop; per-step cost is a single vectorized op over the
    cross-section.
    """
    prices = np.asarray(prices)
    deltas = np.diff(prices, axis=0)
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)

    avg_up = up[:n].mean(axis=0)
    avg_down = down[:n].mean(axis=0)
    rs = avg_up / (avg_down + 1e-9)
    rsi_seed = 100.0 - 100.0 / (1.0 + rs)

    T = prices.shape[0]
    out = np.empty((T,) + prices.shape[1:], dtype=prices.dtype)
    out[:n] = 50.0
    out[n] = rsi_seed.astype(prices.dtype, copy=False)

    for t in range(n + 1, T):
        avg_up = (avg_up * (n - 1) + up[t - 1]) / n
        avg_down = (avg_down * (n - 1) + down[t - 1]) / n
        rs = avg_up / (avg_down + 1e-9)
        out[t] = (100.0 - 100.0 / (1.0 + rs)).astype(prices.dtype, copy=False)
    return out


def rsi_strided(prices: np.ndarray, n: int, w: int = 1) -> np.ndarray:
    """Wilder RSI(n) computed over stride-`w` price changes (1-D only).

    At every bar `t` uses `Δ_i = price[i] - price[i-w]` instead of the
    standard 1-bar `Δ_i = price[i] - price[i-1]`. Wilder-smoothes the
    gains and losses over `n` strided observations. Output is a 1-D
    numpy array aligned with `prices`; positions before the warmup
    (`w + n - 1` bars) are NaN.

    `w=1` reduces to the canonical daily RSI(n) (matches
    `ss_indicators.rsi` for indices ≥ n). `w>1` is the rolling
    weekly/biweekly/monthly view evaluated at every bar — equals the
    discretely-resampled RSI(n) on the resampled-bar boundaries and
    smoothly interpolates off-boundary, giving dense supervision for
    FiLM-conditioned heads.
    """
    if w < 1:
        raise ValueError(f'rsi_strided w must be >= 1, got {w}')
    if n < 2:
        raise ValueError(f'rsi_strided n must be >= 2, got {n}')
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    out = np.full(T, np.nan, dtype=np.float64)
    if T < w + n:
        return out
    deltas = np.empty(T, dtype=np.float64)
    deltas[:w] = 0.0
    deltas[w:] = prices[w:] - prices[:-w]
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)
    avg_up = up[w:w + n].mean()
    avg_down = down[w:w + n].mean()
    rs = avg_up / (avg_down + 1e-9)
    out[w + n - 1] = 100.0 - 100.0 / (1.0 + rs)
    for t in range(w + n, T):
        avg_up = (avg_up * (n - 1) + up[t]) / n
        avg_down = (avg_down * (n - 1) + down[t]) / n
        rs = avg_up / (avg_down + 1e-9)
        out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out
