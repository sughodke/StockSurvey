"""Causal continuous wavelet transform (Ricker, complex Morlet, Gaussian).

`output[t]` depends only on `input[:t+1]` — no look-ahead. Three
kernels share the same one-sided truncation strategy and FFT
convolution machinery:

  * `causal_cwt`          — real Ricker (Mexican-hat) over rolling-
                            z-normed prices. Bandpass, zero DC.
  * `causal_cwt_morlet`   — complex Morlet over rolling-z-normed
                            prices. Bandpass with phase information.
  * `causal_cwt_gaussian` — real Gaussian (scaling function) over the
                            input series as-is. Lowpass / trend
                            companion that recovers the DC content
                            the bandpass kernels structurally cannot
                            carry; caller passes a stationary series
                            (e.g. cumulative log-returns) so growth
                            stays additive.

Rolling z-norm uses cumulative sums so the cost is O(n_dates) per
ticker rather than O(n_dates * lookback).
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
# at negligible loss of wavelet response. The Morlet and Gaussian
# kernels share the same Gaussian envelope, so the same bound applies.
KERNEL_HALF_EXTENT: int = 3

# Field-standard Morlet center frequency. Above ~5 the admissibility
# correction (the DC-killing constant) is negligible, and 6 is the
# value canonical Morlet implementations (Torrence & Compo 1998) use.
DEFAULT_MORLET_OMEGA0: float = 6.0


def _ricker_causal(scale: int, n_dates: int) -> np.ndarray:
    """One-sided Ricker (Mexican-hat) wavelet on t in [-points, 0]."""
    points = min(KERNEL_HALF_EXTENT * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0) / np.sqrt(scale)


def _morlet_causal(
    scale: int, n_dates: int, omega0: float = DEFAULT_MORLET_OMEGA0,
) -> np.ndarray:
    """One-sided complex Morlet on t in [-points, 0].

    `psi(t) = pi^(-1/4) * exp(i * omega0 * t) * exp(-t^2 / 2)`, scaled
    by `1/sqrt(s)` so different scales are L2-comparable. Returns
    complex128.
    """
    points = min(KERNEL_HALF_EXTENT * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    envelope = np.exp(-t ** 2 / 2.0) / (np.pi ** 0.25 * np.sqrt(scale))
    return envelope * np.exp(1j * omega0 * t)


def _gaussian_causal(scale: int, n_dates: int) -> np.ndarray:
    """One-sided Gaussian scaling function on t in [-points, 0].

    `phi(t) = (1 / sqrt(2 * pi)) * exp(-t^2 / 2)`, scaled by
    `1/sqrt(s)` to match the L2 normalization of the wavelet kernels.
    Real-valued; the lowpass companion to the Ricker / Morlet
    bandpass kernels that share the same Gaussian envelope.
    """
    points = min(KERNEL_HALF_EXTENT * scale, n_dates - 1)
    t = np.arange(-points, 1) / scale
    return (np.exp(-t ** 2 / 2.0)
            / (np.sqrt(2.0 * np.pi) * np.sqrt(scale)))


def _rolling_z_norm(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Causal rolling z-norm of a `(n_dates, n_tickers)` price matrix."""
    n_dates, n_tickers = prices.shape
    cs = np.cumsum(np.vstack([np.zeros((1, n_tickers)), prices]), axis=0)
    cs2 = np.cumsum(
        np.vstack([np.zeros((1, n_tickers)), prices ** 2]), axis=0)
    idx = np.arange(n_dates)
    lo = np.maximum(0, idx - lookback + 1)
    counts = idx - lo + 1
    mu = (cs[idx + 1] - cs[lo]) / counts[:, None]
    mu2 = (cs2[idx + 1] - cs2[lo]) / counts[:, None]
    std = np.sqrt(np.maximum(mu2 - mu ** 2, 1e-4))
    return (prices - mu) / std


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
    n_dates, _ = prices.shape
    x_norm = _rolling_z_norm(prices, lookback)

    coeffs = np.zeros((len(scales), n_dates, prices.shape[1]),
                      dtype=np.float32)
    for si, s in enumerate(scales):
        kernel = _ricker_causal(s, n_dates)
        full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
        # full[t] = sum_k x_norm[k] * kernel[t-k] over valid k, which uses
        # x_norm[max(0, t-points) .. t] — strictly causal at index t.
        coeffs[si] = full[:n_dates].astype(np.float32)
    return coeffs


def causal_cwt_morlet(
    prices: np.ndarray,
    scales: list[int],
    lookback: int,
    omega0: float = DEFAULT_MORLET_OMEGA0,
) -> np.ndarray:
    """Causal complex Morlet CWT over a `(n_dates, n_tickers)` price matrix.

    Same rolling-z-norm-then-convolve pipeline as `causal_cwt` (so the
    Morlet sees the same level-invariant input), but with a complex
    kernel — the output carries amplitude and phase per scale.

    Returns `(n_scales, n_dates, n_tickers)` of complex64. Use
    `np.abs(coeffs)` for envelope (volatility / extrema), and
    `np.angle(coeffs)` (or the unit-circle pair `cos/sin`) for phase.
    """
    n_dates, _ = prices.shape
    x_norm = _rolling_z_norm(prices, lookback).astype(np.complex128)

    coeffs = np.zeros((len(scales), n_dates, prices.shape[1]),
                      dtype=np.complex64)
    for si, s in enumerate(scales):
        kernel = _morlet_causal(s, n_dates, omega0=omega0)
        full = fftconvolve(x_norm, kernel[:, None], mode='full', axes=0)
        coeffs[si] = full[:n_dates].astype(np.complex64)
    return coeffs


def causal_cwt_gaussian(
    series: np.ndarray,
    scales: list[int],
) -> np.ndarray:
    """Causal Gaussian (scaling-function) CWT over `(n_dates, n_tickers)`.

    Real-valued lowpass companion to the bandpass Ricker / Morlet.
    Unlike `causal_cwt` / `causal_cwt_morlet`, this routine **does not**
    rolling-z-norm its input — z-normalization strips the local mean,
    which is exactly the DC content the Gaussian is meant to recover.
    Pass a series that is already approximately stationary by
    construction (typically `np.cumsum(log_returns)`); the Gaussian
    convolution at scale `s` then yields the multi-scale rolling mean
    of that series at horizon ~`s`.

    Returns `(n_scales, n_dates, n_tickers)` of float32. Output[t] is
    a function of input[:t+1] only (strictly causal).
    """
    n_dates, n_tickers = series.shape
    x = series.astype(np.float64)

    coeffs = np.zeros((len(scales), n_dates, n_tickers), dtype=np.float32)
    for si, s in enumerate(scales):
        kernel = _gaussian_causal(s, n_dates)
        full = fftconvolve(x, kernel[:, None], mode='full', axes=0)
        coeffs[si] = full[:n_dates].astype(np.float32)
    return coeffs
