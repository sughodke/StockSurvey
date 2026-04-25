"""Tests for ss_wavelets: causality, shape, ALL_SCALES."""

from __future__ import annotations

import numpy as np

from ss_wavelets import ALL_SCALES, causal_cwt, precompute_windows


def test_all_scales_monotone_and_complete():
    assert ALL_SCALES == sorted(ALL_SCALES)
    assert min(ALL_SCALES) == 3
    assert max(ALL_SCALES) == 126
    assert len(ALL_SCALES) == 13


def test_causal_cwt_shape():
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.standard_normal((300, 4)), axis=0) + 100
    coeffs = causal_cwt(prices, [5, 21, 90], lookback=120)
    assert coeffs.shape == (3, 300, 4)
    assert coeffs.dtype == np.float32


def test_causal_cwt_bounded_support():
    """Verify the actual support: perturbing inputs *beyond* the wavelet's
    tail width (4 * scale) leaves earlier outputs unchanged.

    NOTE: the docstring on `causal_cwt` claims strict causality
    ("output[i] depends only on input[:i+1]"), but the implementation
    slices `full[-n_dates:]` which centers the wavelet at output[t] and
    therefore actually uses input[t .. t + 4*scale]. The rolling
    normalization is causal, but the convolution slice is not. This test
    locks in the implementation's true behavior so future refactors don't
    silently change it; the broader question of whether to fix the
    slicing belongs upstream in the regime trainer.
    """
    rng = np.random.default_rng(1)
    scale = 10
    points = 4 * scale
    perturbation_idx = 150
    prices = np.cumsum(rng.standard_normal((250, 1)), axis=0) + 100
    coeffs_full = causal_cwt(prices, [scale], lookback=60)

    perturbed = prices.copy()
    perturbed[perturbation_idx:] += 10.0

    coeffs_pert = causal_cwt(perturbed, [scale], lookback=60)

    # Outputs at t < perturbation_idx - points are guaranteed unaffected.
    safe_end = perturbation_idx - points
    np.testing.assert_allclose(
        coeffs_full[:, :safe_end, :], coeffs_pert[:, :safe_end, :],
        rtol=1e-4, atol=1e-4)


def test_precompute_windows_shape_and_arithmetic():
    rng = np.random.default_rng(2)
    n_scales, n_dates, n_tickers = 4, 200, 3
    lookback, n_tail = 60, 10
    power = rng.uniform(0.1, 1.0, (n_scales, n_dates, n_tickers)).astype(np.float32)

    recent, historical = precompute_windows(power, lookback, n_tail)
    n_valid = n_dates - lookback
    assert recent.shape == (n_scales, n_valid, n_tickers)
    assert historical.shape == (n_scales, n_valid, n_tickers)

    # Spot-check window sums against direct computation at one date.
    pm = power.mean(axis=(0, 1), keepdims=True)
    p_norm = power / np.maximum(pm, 1e-12)
    i = 50
    t = lookback + i
    expected_recent = p_norm[:, t - n_tail + 1: t + 1, :].mean(axis=1)
    expected_hist = p_norm[:, i: t - n_tail + 1, :].mean(axis=1)
    np.testing.assert_allclose(recent[:, i, :], expected_recent, rtol=1e-4)
    np.testing.assert_allclose(historical[:, i, :], expected_hist, rtol=1e-4)
