"""Commodity Channel Index — close-only, pure numpy.

CCI(n) at bar t:
    SMA_n     = mean(close over trailing n bars)
    MAD_n     = mean(|close - SMA_n| over the same n bars)   (mean abs dev)
    CCI       = (close[t] - SMA_n) / (0.015 * MAD_n)

The 0.015 is Lambert's (1980) original constant — fixed, not tunable.
Output is centered at 0; ~70% of values fall within ±100 (it's not
strictly bounded but trades like a bounded oscillator). Close-only
substitutes `close` for the canonical typical-price `(H+L+C)/3` so the
function takes a single price series like the rest of `ss_indicators`.

Two flavors:
  - `cci(prices, n)`            — matrix-form over axis 0,
                                  vectorizes over trailing dims
  - `cci_strided(prices, n, w)` — CCI(n) computed over stride-w price
                                  history (1-D); with w=1 reduces to
                                  the standard daily CCI(n), w>1 gives
                                  weekly/biweekly/monthly cadence
                                  evaluated at every bar (dense
                                  supervision for FiLM-conditioned
                                  heads on the (n, w) grid)
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS: float = 1e-12
LAMBERT: float = 0.015


def cci(prices: np.ndarray, n: int = 20) -> np.ndarray:
    """Lambert CCI(n) over axis 0, close-only.

    Output is `nan` for the first `n - 1` bars (warmup). Vectorized
    over trailing axes via stride_tricks; per-bar cost is O(n) but the
    full series is one fused stride view + numpy reduction.
    """
    prices = np.asarray(prices, dtype=np.float64)
    T = prices.shape[0]
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    if T < n:
        return out
    # Build sliding-window view along axis 0; result shape
    # (T - n + 1, n) for 1-D, (T - n + 1, n, *trailing) for 2-D+.
    win = sliding_window_view(prices, n, axis=0)
    # win has the window axis at position `prices.ndim`; move to -1 for
    # uniform reduction.
    win = np.moveaxis(win, prices.ndim, -1)
    mu = win.mean(axis=-1)
    mad = np.abs(win - mu[..., None]).mean(axis=-1)
    safe_mad = np.where(mad < EPS, 1.0, mad)
    raw = (prices[n - 1:] - mu) / (LAMBERT * safe_mad)
    raw = np.where(mad < EPS, 0.0, raw)
    out[n - 1:] = raw
    return out


def cci_strided(prices: np.ndarray, n: int, w: int = 1) -> np.ndarray:
    """CCI(n) over stride-`w` price history (1-D only).

    At every bar `t` evaluates CCI using the n historical samples
    `[prices[t], prices[t - w], prices[t - 2w], ..., prices[t - (n-1)*w]]`.
    With `w=1` reduces to the canonical daily CCI(n). With `w=5` gives
    a rolling-weekly CCI evaluated at every bar (matches the
    discretely-resampled CCI on the resampling boundaries and smoothly
    interpolates off-boundary), giving dense supervision for FiLM-
    conditioned heads.

    Output is a 1-D numpy array aligned with `prices`; the first
    `(n - 1) * w` positions are NaN (warmup).
    """
    if w < 1:
        raise ValueError(f'cci_strided w must be >= 1, got {w}')
    if n < 2:
        raise ValueError(f'cci_strided n must be >= 2, got {n}')
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    out = np.full(T, np.nan, dtype=np.float64)
    span = (n - 1) * w + 1
    if T < span:
        return out
    # Sliding window of length `span` (covers the full history footprint
    # back from t), then sub-sample with stride w to get the n strided
    # observations [t - (n-1)*w, ..., t].
    full = sliding_window_view(prices, span)         # (T - span + 1, span)
    sub = full[:, ::w]                                # (T - span + 1, n)
    mu = sub.mean(axis=1)
    mad = np.abs(sub - mu[:, None]).mean(axis=1)
    safe_mad = np.where(mad < EPS, 1.0, mad)
    raw = (prices[span - 1:] - mu) / (LAMBERT * safe_mad)
    raw = np.where(mad < EPS, 0.0, raw)
    out[span - 1:] = raw
    return out


def cci_strided_grid(
    prices: np.ndarray,
    n_grid: tuple[int, ...] | list[int] | np.ndarray,
    w_grid: tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    """CCI for every `(w, n)` in the Cartesian product of two grids.

    Returns shape `(T, len(w_grid), len(n_grid))`. Bit-equivalent to
    `np.stack([cci_strided(prices, n, w) for n in n_grid], axis=-1)` for
    each `w`. Each cell preserves its own per-cell warmup: the first
    valid bar for `(n, w)` is at index `(n - 1) * w`, exactly matching
    `cci_strided`.

    Unlike RSI's Wilder recurrence (which broadcasts cleanly across n),
    the CCI mean-abs-deviation is non-linear and resists sharing
    reductions across different n values. The inner `mean` + `abs`
    reductions still happen per-cell.

    **Performance caveat.** Same as `rsi_strided_grid`: at small grids
    (~25 cells, the default in `factor.IndicatorGridConfig`) this is
    not faster than 25 individual `cci_strided` calls because the
    per-cell numpy work dominates and there's nothing to share. Useful
    primarily as a single entry point for callers whose grids will grow.
    """
    n_arr_int = np.asarray(n_grid, dtype=np.int64)
    w_arr_int = np.asarray(w_grid, dtype=np.int64)
    if (n_arr_int < 2).any():
        raise ValueError(f'cci_strided_grid requires every n >= 2, got {n_grid}')
    if (w_arr_int < 1).any():
        raise ValueError(f'cci_strided_grid requires every w >= 1, got {w_grid}')

    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    n_n = len(n_arr_int)
    out = np.full((T, len(w_arr_int), n_n), np.nan, dtype=np.float64)
    if T == 0:
        return out

    for wi, w in enumerate(w_arr_int):
        w = int(w)
        for ni in range(n_n):
            n_i = int(n_arr_int[ni])
            span_i = (n_i - 1) * w + 1
            if T < span_i:
                continue
            # Each cell uses its own sliding view so its valid output
            # starts at `span_i - 1`, not the worst-case `max_span - 1`.
            # `sliding_window_view` itself is a view (O(1)); the cost
            # lives in the mean+abs reductions below.
            full = sliding_window_view(prices, span_i)    # (M, span_i)
            sub = full[:, ::w]                             # (M, n_i)
            mu = sub.mean(axis=1)
            mad = np.abs(sub - mu[:, None]).mean(axis=1)
            safe_mad = np.where(mad < EPS, 1.0, mad)
            raw = (prices[span_i - 1:] - mu) / (LAMBERT * safe_mad)
            raw = np.where(mad < EPS, 0.0, raw)
            out[span_i - 1:, wi, ni] = raw

    return out
