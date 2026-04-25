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


def test_causal_cwt_strict_causality():
    """Modifying input[t+1:] must not change output[..t]."""
    rng = np.random.default_rng(1)
    prices = np.cumsum(rng.standard_normal((250, 1)), axis=0) + 100
    coeffs_full = causal_cwt(prices, [5, 21, 90], lookback=60)

    for cut in (50, 100, 150, 200):
        perturbed = prices.copy()
        perturbed[cut:] += 10.0
        coeffs_pert = causal_cwt(perturbed, [5, 21, 90], lookback=60)
        # Outputs at t in [0, cut-1] depend only on input[..cut-1] = unchanged.
        np.testing.assert_allclose(
            coeffs_full[:, :cut, :], coeffs_pert[:, :cut, :],
            rtol=1e-4, atol=1e-4,
            err_msg=f'leakage detected at perturbation cut={cut}')


def test_causal_cwt_impulse_response_is_one_sided():
    """An impulse at t=I should produce response only at t in [I, I+points]
    (post-impulse), never before. Confirms the wavelet's right-edge
    aligns with the output time index.
    """
    scale = 10
    points = 4 * scale
    T = 80
    impulse_at = 30
    x = np.zeros((T, 1))
    x[impulse_at, 0] = 1.0
    coeffs = causal_cwt(x, [scale], lookback=20)[0, :, 0]

    nz = np.where(np.abs(coeffs) > 1e-6)[0]
    # Response must start at or after the impulse — nothing before.
    assert nz.min() >= impulse_at, (
        f'response starts at t={nz.min()}, before impulse at t={impulse_at}')
    # And must die out within `points` samples of the impulse (kernel support).
    # We allow a small slack because the rolling normalization smears the
    # impulse across the lookback window.
    assert nz.max() <= impulse_at + points + 20


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
