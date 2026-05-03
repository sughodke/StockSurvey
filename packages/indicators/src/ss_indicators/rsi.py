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


def rsi_strided_grid(
    prices: np.ndarray,
    n_grid: tuple[int, ...] | list[int] | np.ndarray,
    w_grid: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    """Wilder RSI for every `(w, n)` in the Cartesian product of two grids.

    Returns shape `(T, len(w_grid), len(n_grid))`. Values match
    `np.stack([rsi_strided(prices, n, w) for n in n_grid], axis=-1)` per
    `w` (modulo float64 round-off). The inner Wilder recurrence loops
    once per bar over the full `n_grid` vector instead of restarting
    Python's per-cell loop.

    Per-cell warmup bars (positions before `w + n - 1`) stay NaN, matching
    `rsi_strided`. The seed phase (`t in [w + min(n) - 1, w + max(n) - 2]`)
    walks each n separately because they warm up at different offsets;
    after the slowest n is seeded, the steady-state phase advances all
    n's together with one broadcast per t.

    **Performance caveat.** Vectorizing across n only wins when `len(n_grid)`
    is large enough to amortize numpy's per-op dispatch overhead — empirically
    that crossover is around 30+. At smaller `n_grid` (e.g. 6 cells, the
    current default in `factor.IndicatorGridConfig`) Python scalar arithmetic
    in the per-cell `rsi_strided` loop is faster, so callers with small grids
    should keep using per-cell. This function exists for callers with denser
    grids (apps/replay's FiLM head supervision when grids are extended).
    """
    n_arr_int = np.asarray(n_grid, dtype=np.int64)
    w_arr_int = np.asarray(w_grid, dtype=np.int64)
    if (n_arr_int < 2).any():
        raise ValueError(f'rsi_strided_grid requires every n >= 2, got {n_grid}')
    if (w_arr_int < 1).any():
        raise ValueError(f'rsi_strided_grid requires every w >= 1, got {w_grid}')

    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    n_n = len(n_arr_int)
    out = np.full((T, len(w_arr_int), n_n), np.nan, dtype=np.float64)
    if T == 0:
        return out

    n_arr = n_arr_int.astype(np.float64)
    n_minus_1 = n_arr - 1.0
    max_n_int = int(n_arr_int.max())

    for wi, w in enumerate(w_arr_int):
        w = int(w)
        if T < w + max_n_int:
            # Some n's never reach a full window; rsi_strided would also
            # return all-NaN here, so stay consistent and skip the column.
            continue

        deltas = np.empty(T, dtype=np.float64)
        deltas[:w] = 0.0
        deltas[w:] = prices[w:] - prices[:-w]
        up = np.where(deltas > 0, deltas, 0.0)
        down = np.where(deltas < 0, -deltas, 0.0)

        # Seed each n at its own warmup bar t = w + n - 1 by averaging the
        # first n strided gains/losses, exactly as rsi_strided does.
        avg_up = np.empty(n_n, dtype=np.float64)
        avg_down = np.empty(n_n, dtype=np.float64)
        for ni in range(n_n):
            n_i = int(n_arr_int[ni])
            avg_up[ni] = up[w:w + n_i].mean()
            avg_down[ni] = down[w:w + n_i].mean()
            rs_seed = avg_up[ni] / (avg_down[ni] + 1e-9)
            out[w + n_i - 1, wi, ni] = 100.0 - 100.0 / (1.0 + rs_seed)

            # Walk this n alone from its seed up to where the slowest n
            # is also seeded. Once everyone is seeded, the loop below
            # advances them all together.
            for t in range(w + n_i, w + max_n_int):
                avg_up[ni] = (avg_up[ni] * (n_i - 1) + up[t]) / n_i
                avg_down[ni] = (avg_down[ni] * (n_i - 1) + down[t]) / n_i
                rs_t = avg_up[ni] / (avg_down[ni] + 1e-9)
                out[t, wi, ni] = 100.0 - 100.0 / (1.0 + rs_t)

        # Steady-state: all n's warmed; advance the (n_n,) vector in one
        # broadcast per bar. This is the loop the per-cell version pays
        # `n_n` times in Python; here it's once.
        for t in range(w + max_n_int, T):
            avg_up = (avg_up * n_minus_1 + up[t]) / n_arr
            avg_down = (avg_down * n_minus_1 + down[t]) / n_arr
            rs = avg_up / (avg_down + 1e-9)
            out[t, wi] = 100.0 - 100.0 / (1.0 + rs)

    return out
