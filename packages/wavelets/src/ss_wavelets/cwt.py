"""Causal continuous wavelet transform (Ricker / Mexican-hat).

`output[t]` depends only on `input[:t+1]` — no look-ahead. Each ticker is
z-normalized over a *causal* rolling window of length `lookback` before
convolution; the window stats are computed via cumulative sums so the
cost is O(n_dates) per ticker rather than O(n_dates * lookback).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


# 13 logarithmically spaced lookbacks from 3 days (intra-week noise) to
# 126 days (~half year). Trained models typically concentrate weight on
# the 26-126 day band.
ALL_SCALES: list[int] = [3, 5, 7, 10, 12, 15, 21, 26, 42, 50, 63, 90, 126]

# Half-extent of the Ricker kernel in scale-normalized time. Numerically
# `(1 - t^2) * exp(-t^2 / 2)` has under 0.3% of its squared energy past
# |t| = 3 — captured fraction 0.997 vs 0.99996 at |t| = 4. Truncating
# at 3 instead of the textbook 4 saves 25% of the per-scale kernel
# size and the same fraction of the per-day data dependency budget,
# at negligible loss of wavelet response. Single source of truth shared
# by `_ricker_causal` and downstream callers that need to compute
# warm-up requirements (e.g. `regime.trainer.DEFAULT_PER_WINDOW_MIN_HISTORY`).
KERNEL_HALF_EXTENT: int = 3


def _ricker_causal(scale: int, n_dates: int) -> np.ndarray:
    """One-sided Ricker (Mexican-hat) wavelet on t in [-points, 0]."""
    points = min(KERNEL_HALF_EXTENT * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale)


def causal_cwt(
    prices: np.ndarray,
    scales: list[int],
    lookback: int,
) -> np.ndarray:
    """Causal CWT over a `(n_dates, n_tickers)` price matrix.

    Returns `(n_scales, n_dates, n_tickers)` of float32 wavelet
    coefficients. `output[t]` depends only on `input[:t+1]` (strictly
    causal): the rolling z-normalization uses only past prices, and the
    convolution with the one-sided Ricker kernel is sliced as `full[:T]`
    so each output index aligns with the kernel's right edge at time t.

    Warm-up note: for `t < kernel_len - 1`, the convolution sees fewer
    than `kernel_len` past samples (zero-padded outside x's range), so
    early outputs have reduced wavelet support. Callers typically drop
    the first `lookback` outputs anyway via `precompute_windows`.
    """
    n_dates, n_tickers = prices.shape

    cs = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices]), axis=0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices ** 2]), axis=0)

    idx = np.arange(n_dates)
    lo = np.maximum(0, idx - lookback + 1)
    counts = idx - lo + 1
    mu = (cs[idx + 1] - cs[lo]) / counts[:, None]
    mu2 = (cs2[idx + 1] - cs2[lo]) / counts[:, None]
    std = np.sqrt(np.maximum(mu2 - mu ** 2, 1e-4))

    x_norm = (prices - mu) / std

    coeffs = np.zeros((len(scales), n_dates, n_tickers), dtype=np.float32)
    for si, s in enumerate(scales):
        kernel = _ricker_causal(s, n_dates)
        full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
        # full[t] = sum_k x_norm[k] * kernel[t-k] over valid k, which uses
        # x_norm[max(0, t-points) .. t] — strictly causal at index t.
        coeffs[si] = full[:n_dates].astype(np.float32)
    return coeffs
