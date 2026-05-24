"""Tests for ss_portfolio.sharpe_diff — Ledoit-Wolf studentized
stationary-bootstrap CI for Sharpe-difference."""
from __future__ import annotations

import numpy as np
import pytest

from ss_portfolio import SharpeDiffCI, sharpe_difference_ci


def test_identical_streams_delta_zero_and_ci_includes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.01, size=500)
    res = sharpe_difference_ci(a, a.copy(), n_bootstraps=500, seed=1)
    assert isinstance(res, SharpeDiffCI)
    assert res.delta_sr == pytest.approx(0.0, abs=1e-12)
    assert res.includes_zero


def test_obvious_winner_excludes_zero():
    """Stream A has much higher Sharpe than B → CI excludes 0 reliably."""
    rng = np.random.default_rng(1)
    n = 1000
    a = rng.normal(0.003, 0.01, size=n)   # SR ≈ 0.30/period
    b = rng.normal(-0.001, 0.01, size=n)  # SR ≈ -0.10/period
    res = sharpe_difference_ci(a, b, n_bootstraps=500, seed=2)
    assert res.delta_sr > 0
    assert not res.includes_zero
    assert res.ci_lo > 0


def test_noise_band_includes_zero():
    """Two zero-skill streams → CI should usually include 0."""
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.01, size=200)
    b = rng.normal(0.0, 0.01, size=200)
    res = sharpe_difference_ci(a, b, n_bootstraps=500, seed=3)
    assert res.includes_zero, f'CI {res.ci_lo, res.ci_hi} did not include 0'


def test_block_length_default_is_n_cube_root():
    rng = np.random.default_rng(3)
    n = 1000
    a = rng.normal(0.001, 0.01, size=n)
    b = rng.normal(0.0005, 0.01, size=n)
    res = sharpe_difference_ci(a, b, n_bootstraps=300, seed=4)
    expected = max(2, int(round(n ** (1.0/3.0))))
    assert res.block_length == expected


def test_unequal_length_raises():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0, 1, 80)
    with pytest.raises(ValueError, match='date-aligned'):
        sharpe_difference_ci(a, b, n_bootstraps=100, seed=5)


def test_very_short_series_raises():
    a = np.array([0.01, -0.005, 0.003, 0.002])  # n=4 < 5
    b = np.array([0.0, 0.001, -0.002, 0.0015])
    with pytest.raises(ValueError, match='n >= 5'):
        sharpe_difference_ci(a, b, n_bootstraps=100, seed=6)


def test_ci_width_shrinks_with_sample_length():
    """Same signal at n=200 vs n=2000 → wider CI at smaller n."""
    rng = np.random.default_rng(5)
    for n_small, n_large in [(200, 2000)]:
        a_s = rng.normal(0.001, 0.01, n_small)
        b_s = rng.normal(0.0, 0.01, n_small)
        a_l = rng.normal(0.001, 0.01, n_large)
        b_l = rng.normal(0.0, 0.01, n_large)
        res_s = sharpe_difference_ci(a_s, b_s, n_bootstraps=400, seed=7)
        res_l = sharpe_difference_ci(a_l, b_l, n_bootstraps=400, seed=7)
        w_s = res_s.ci_hi - res_s.ci_lo
        w_l = res_l.ci_hi - res_l.ci_lo
        assert w_s > w_l, f'expected CI to shrink with n; got {w_s} (n={n_small}) vs {w_l} (n={n_large})'


def test_studentized_handles_correlated_streams():
    """When streams are highly correlated, ΔSR has lower variance."""
    rng = np.random.default_rng(6)
    n = 500
    common = rng.normal(0.001, 0.01, size=n)
    a = common + rng.normal(0.0002, 0.001, size=n)
    b = common + rng.normal(0.0, 0.001, size=n)
    res = sharpe_difference_ci(a, b, n_bootstraps=500, seed=8)
    # With ρ ≈ 1, ΔSR s.e. is much smaller than independent case
    assert res.se_delta_sr < 0.5  # sanity bound; independent at this n would be > 0.5
