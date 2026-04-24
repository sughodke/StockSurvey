"""Causal continuous wavelet transform and pre-computed regime windows.

Both routines run in numpy and are *not* differentiated through. The CWT
power matrix and the recent / historical window means are computed once
per call to `train()`; only the small block-level tensors that follow
participate in JAX autograd.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


# 13 logarithmically spaced lookbacks from 3 days (intra-week noise) to
# 126 days (~half year). The trained model typically concentrates weight
# on the 26-126 day band.
ALL_SCALES: list[int] = [3, 5, 7, 10, 12, 15, 21, 26, 42, 50, 63, 90, 126]


def _ricker_causal(scale: int, n_dates: int) -> np.ndarray:
    """One-sided Ricker (Mexican-hat) wavelet on t in [-points, 0]."""
    points = min(4 * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale)


def causal_cwt(
    prices: np.ndarray,
    scales: list[int],
    lookback: int,
) -> np.ndarray:
    """Causal CWT: output[i] depends only on input[:i+1].

    Each ticker is z-normalized over a *causal* rolling window of length
    `lookback` before convolution. The window stats are computed via
    cumulative sums (O(n_dates) per ticker rather than O(n_dates * lookback)).
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
        coeffs[si] = full[-n_dates:].astype(np.float32)
    return coeffs


def precompute_windows(
    power: np.ndarray,
    lookback: int,
    n_tail: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (recent, historical) windowed power means for every valid date.

    Each output has shape (n_scales, n_valid, n_tickers) where
    n_valid = n_dates - lookback. For valid index i (date t = lookback + i):

        recent[..., i, ...]     = mean of power over (t - n_tail + 1, t]
        historical[..., i, ...] = mean of power over [i, t - n_tail + 1)

    The historical window has length lookback - n_tail + 1, the recent
    has length n_tail; together they tile the lookback + 1 days ending
    at t. Computed via cumsum so the cost is O(n_scales * n_dates * n_tickers)
    independent of `lookback`.

    Per-ticker normalization
    ------------------------
    Each ticker's power is divided by its mean over (scales, time) before
    cumsum, to keep the float32 cumsum well-conditioned across ~3000 dates
    (raw CWT power on high-priced names exceeds 1e11 and overflows
    float32 cumsum). This *does* use full-history information, but the
    KL divergence in `regime_scores` is invariant to a per-ticker uniform
    rescaling of power, so no future information actually leaks into
    training scores. If the divergence is ever swapped for a non-scale-
    invariant one, this normalizer must be made causal.
    """
    n_scales, n_dates, n_tickers = power.shape
    n_valid = n_dates - lookback
    n_hist = lookback - n_tail

    pm = power.mean(axis=(0, 1), keepdims=True)
    power = power / np.maximum(pm, 1e-12)

    cs = np.cumsum(power.astype(np.float64), axis=1)
    cs = np.concatenate(
        [np.zeros((n_scales, 1, n_tickers), dtype=np.float64), cs],
        axis=1,
    )

    recent = (cs[:, lookback + 1:, :]
              - cs[:, lookback - n_tail + 1: n_dates - n_tail + 1, :]) / n_tail
    historical = (cs[:, n_hist + 1: n_valid + n_hist + 1, :]
                  - cs[:, :n_valid, :]) / (n_hist + 1)
    return recent.astype(np.float32), historical.astype(np.float32)
