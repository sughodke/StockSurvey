"""Tests for the matrix-form polar Morlet helper.

The 1-D `compute_scalogram_polar(prices, scales, lookback)` is already
covered indirectly via the SSL trainer's reconstruction stats. This
test file targets `causal_polar_morlet_matrix`, which is the new
relational-app entry point — it returns a `(C * n_scales, n_dates,
n_tickers)` channel-stacked panel and per-ticker outputs must match
what the 1-D version produces (proves the matrix path doesn't bleed
information across ticker columns).
"""
from __future__ import annotations

import numpy as np
import pytest

from ss_features import (
    RELATIONAL_CHANNELS_PER_SCALE, causal_polar_morlet_matrix,
    compute_scalogram_polar,
)


def _synthetic_prices(seed: int, T: int, N: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(0.0005, 0.012, (T, N))
    return 100.0 * np.exp(np.cumsum(log_steps, axis=0))


def test_panel_shape():
    prices = _synthetic_prices(0, 600, 5)
    scales = [5, 10, 21, 50, 90]
    panel = causal_polar_morlet_matrix(prices, scales, lookback=120)
    assert panel.dtype == np.float32
    expected_rows = RELATIONAL_CHANNELS_PER_SCALE * len(scales)
    assert panel.shape == (expected_rows, 600, 5)
    assert np.isfinite(panel).all()


def test_polar_unit_circle_post_warmup():
    prices = _synthetic_prices(1, 600, 3)
    scales = [5, 10, 21]
    S = len(scales)
    panel = causal_polar_morlet_matrix(prices, scales, lookback=120)
    cos_block = panel[S:2 * S]
    sin_block = panel[2 * S:3 * S]
    # cos^2 + sin^2 should equal 1 (modulo fp32 noise) at every bar past
    # the warmup region — phase pair lives on the unit circle.
    unit = cos_block ** 2 + sin_block ** 2
    np.testing.assert_allclose(unit[:, 200:, :], 1.0, atol=1e-5)


def test_ticker_independence_matches_1d_path():
    prices = _synthetic_prices(2, 400, 4)
    scales = [5, 10, 21, 50]
    S = len(scales)
    lookback = 120
    panel = causal_polar_morlet_matrix(prices, scales, lookback=lookback)
    # Per-ticker independence: column j of the matrix path must equal
    # the 1-D `compute_scalogram_polar` applied to that ticker alone,
    # for each polar channel. This is what makes it safe to share the
    # rolling z-norm machinery across tickers — they're fully decoupled.
    for j in range(prices.shape[1]):
        abs1, cos1, sin1, g1 = compute_scalogram_polar(
            prices[:, j], scales, lookback=lookback)
        abs_panel = panel[0:S, :, j]
        cos_panel = panel[S:2 * S, :, j]
        sin_panel = panel[2 * S:3 * S, :, j]
        g_panel = panel[3 * S:4 * S, :, j]
        np.testing.assert_allclose(abs_panel, abs1, atol=1e-5,
                                   err_msg=f'|c| mismatch on ticker {j}')
        np.testing.assert_allclose(cos_panel, cos1, atol=1e-5,
                                   err_msg=f'cos mismatch on ticker {j}')
        np.testing.assert_allclose(sin_panel, sin1, atol=1e-5,
                                   err_msg=f'sin mismatch on ticker {j}')
        np.testing.assert_allclose(g_panel, g1, atol=1e-5,
                                   err_msg=f'gauss mismatch on ticker {j}')


def test_panel_is_causal_under_future_perturbation():
    """Perturbing future bars must not change the panel at past bars (modulo
    FFT fp32 noise — `fftconvolve` operates on the full padded signal,
    so its numerical accumulator at any output index depends on the
    sum of all coefficient products, not just the kernel-support
    window. The noise floor on `(|c|, cos, sin, g)` channels at this
    universe size is ~1e-3 in float32; that's the bound we assert,
    matching `causal_cwt_morlet`'s actual precision."""
    prices_a = _synthetic_prices(3, 500, 2)
    prices_b = prices_a.copy()
    rng = np.random.default_rng(99)
    prices_b[300:] = (prices_b[299:300]
                      * np.exp(np.cumsum(rng.normal(0.0, 0.02, (200, 2)),
                                         axis=0)))

    scales = [5, 10, 50]
    panel_a = causal_polar_morlet_matrix(prices_a, scales, lookback=120)
    panel_b = causal_polar_morlet_matrix(prices_b, scales, lookback=120)
    np.testing.assert_allclose(
        panel_a[:, :300, :], panel_b[:, :300, :], atol=2e-3,
        err_msg='polar Morlet panel leaks future data into past bars')


def test_input_shape_validation():
    prices_1d = np.linspace(100, 110, 200, dtype=np.float64)
    with pytest.raises(ValueError, match='must be \\(n_dates, n_tickers\\)'):
        causal_polar_morlet_matrix(prices_1d, [5, 10], lookback=60)
