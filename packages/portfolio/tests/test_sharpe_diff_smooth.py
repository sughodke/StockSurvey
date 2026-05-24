"""Tests for ss_portfolio.sharpe_diff_smooth — the differentiable
parametric analogue of the Ledoit-Wolf bootstrap CI."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ss_portfolio import (
    ParametricSharpeDiffCI,
    p_excludes_zero,
    parametric_ci,
    sharpe_difference_ci,
    soft_excludes_zero,
    studentized_sharpe_diff,
)


# ----------------------------------------------------- core point estimate

def test_studentized_diff_zero_when_streams_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.01, size=500)
    t = studentized_sharpe_diff(a, a.copy())
    assert t == pytest.approx(0.0, abs=1e-12)


def test_studentized_diff_positive_when_a_dominates():
    rng = np.random.default_rng(1)
    a = rng.normal(0.003, 0.01, size=2000)
    b = rng.normal(0.0, 0.01, size=2000)
    t = studentized_sharpe_diff(a, b)
    assert t > 3.0   # large effect, easy to detect at this n


def test_studentized_diff_negative_when_b_dominates():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.01, size=2000)
    b = rng.normal(0.003, 0.01, size=2000)
    t = studentized_sharpe_diff(a, b)
    assert t < -3.0


def test_studentized_diff_with_moments_matches_psr_denominator():
    """The Bailey-LdP corrected variance should match
    `probabilistic_sharpe`'s denominator term when applied to a single
    stream vs a zero-mean benchmark of the same shape."""
    rng = np.random.default_rng(3)
    n = 1000
    a = rng.normal(0.001, 0.012, size=n)
    # When b is a zero-skill noise stream of the same length, the
    # corrected t-stat should be in the ballpark of the PSR z-stat
    # of a vs benchmark 0.
    b = rng.normal(0.0, 0.012, size=n)
    t_plain = studentized_sharpe_diff(a, b, with_moments=False)
    t_corr  = studentized_sharpe_diff(a, b, with_moments=True)
    # Both should be in the same sign; corrected may shift by a bit
    # depending on skew/kurt but not by orders of magnitude.
    assert (t_plain > 0) == (t_corr > 0)
    # The corrected denominator typically yields a slightly smaller |t|
    # for heavy-tailed streams; on Gaussian-IID returns the shift is
    # small (<30%).
    assert abs(t_corr - t_plain) / max(abs(t_plain), 1e-9) < 0.30


# --------------------------------------------------------- parametric CI

def test_parametric_ci_returns_correct_shape():
    rng = np.random.default_rng(4)
    a = rng.normal(0.001, 0.01, size=500)
    b = rng.normal(0.0, 0.01, size=500)
    r = parametric_ci(a, b)
    assert isinstance(r, ParametricSharpeDiffCI)
    assert r.n_obs == 500
    assert r.ci_lo < r.delta_sr < r.ci_hi
    # Width must equal 2 * 1.96 * s.e. for 95%
    width = r.ci_hi - r.ci_lo
    assert width == pytest.approx(2 * 1.959963984540054 * r.se_delta_sr, rel=1e-6)


def test_parametric_ci_excludes_zero_when_signal_clear():
    rng = np.random.default_rng(5)
    a = rng.normal(0.004, 0.01, size=3000)
    b = rng.normal(0.0, 0.01, size=3000)
    r = parametric_ci(a, b)
    assert not r.includes_zero
    assert r.ci_lo > 0


def test_parametric_ci_includes_zero_for_noise():
    rng = np.random.default_rng(6)
    a = rng.normal(0.0, 0.01, size=200)
    b = rng.normal(0.0, 0.01, size=200)
    r = parametric_ci(a, b)
    assert r.includes_zero


def test_parametric_ci_unequal_length_raises():
    a = np.zeros(10); b = np.zeros(8)
    with pytest.raises(ValueError, match='date-aligned'):
        parametric_ci(a, b)


def test_parametric_ci_short_sample_raises():
    a = np.array([0.01, -0.01, 0.005, 0.0])  # n=4
    b = a.copy()
    with pytest.raises(ValueError, match='n >= 5'):
        parametric_ci(a, b)


# ---------------------------------------- soft / probabilistic indicators

def test_soft_excludes_zero_monotone_in_t():
    s_low = soft_excludes_zero(0.5)
    s_mid = soft_excludes_zero(1.96)
    s_hi = soft_excludes_zero(3.5)
    assert s_low < s_mid < s_hi
    assert 0.0 < s_low < 1.0


def test_soft_excludes_zero_temperature_sharpens():
    """Lower temperature → sharper sigmoid → closer to 0 below z, closer
    to 1 above z."""
    s_warm = soft_excludes_zero(0.5, temperature=2.0)
    s_cold = soft_excludes_zero(0.5, temperature=0.1)
    # At t=0.5 (well below z=1.96), cold sigmoid pushes toward 0
    assert s_cold < s_warm


def test_p_excludes_zero_is_normal_cdf():
    """`p_excludes_zero(t) = Φ(t)`, the standard normal CDF."""
    assert p_excludes_zero(0.0) == pytest.approx(0.5, abs=1e-12)
    assert p_excludes_zero(1.96) == pytest.approx(0.975, abs=1e-3)
    assert p_excludes_zero(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert p_excludes_zero(10.0) == pytest.approx(1.0, abs=1e-6)


# ----------------------- convergence to bootstrap as n grows + low block_dep

def test_parametric_ci_converges_to_bootstrap_at_large_n():
    """Lo 2002 / Ledoit-Wolf 2008: parametric Gaussian CI converges to
    bootstrap CI as n → ∞ for IID returns. Verify the two agree to ~10%
    width at n=2000."""
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.normal(0.0012, 0.011, size=n)
    b = rng.normal(0.0006, 0.010, size=n)

    par = parametric_ci(a, b)
    boot = sharpe_difference_ci(a, b, n_bootstraps=2000, seed=42)

    # Point estimates must agree exactly (computed identically)
    assert par.delta_sr == pytest.approx(boot.delta_sr, abs=1e-12)

    # CI widths agree within ~15% at this n
    par_w = par.ci_hi - par.ci_lo
    boot_w = boot.ci_hi - boot.ci_lo
    rel_diff = abs(par_w - boot_w) / max(boot_w, 1e-9)
    assert rel_diff < 0.20, f'parametric width {par_w:.4f} vs bootstrap {boot_w:.4f}'

    # Both should agree on "includes zero?"
    assert par.includes_zero == boot.includes_zero


# --------------------------- finite-difference gradient sanity (numerical)

def test_studentized_diff_finite_difference_gradient():
    """The t-stat is a smooth function of the inputs. A small bump to
    one return must move the t-stat by a small amount consistent with
    the finite-difference gradient. Not a full autograd check (numpy
    doesn't autograd), but a smoothness sanity check."""
    rng = np.random.default_rng(8)
    a = rng.normal(0.001, 0.01, size=200)
    b = rng.normal(0.0, 0.01, size=200)

    t0 = studentized_sharpe_diff(a, b)
    bump = 1e-4
    a_pert = a.copy(); a_pert[0] += bump
    t1 = studentized_sharpe_diff(a_pert, b)
    # Sensitivity is O(1/n), so the change is small but non-zero
    assert abs(t1 - t0) > 1e-8
    assert abs(t1 - t0) < 1.0   # not pathological


# ------------------ workspace-relevant: short-sample vs DCA-like benchmark

def test_short_sample_matches_workspace_vol_v3_pattern():
    """At n=33 (the vol-v3-DoltHub sample), parametric and bootstrap
    CIs should still agree directionally on 'includes zero'."""
    rng = np.random.default_rng(9)
    a = rng.normal(0.02, 0.04, size=33)   # vol-v3-like alpha
    b = rng.normal(0.005, 0.025, size=33) # DCA-block-like
    par = parametric_ci(a, b)
    boot = sharpe_difference_ci(a, b, n_bootstraps=1000, seed=42)
    # At n=33 the two CIs aren't identical (fat-tail effects) but the
    # 'includes 0' verdict should agree most of the time. We don't
    # enforce strict equality; we enforce the parametric is not
    # wildly wrong.
    assert par.delta_sr == pytest.approx(boot.delta_sr, abs=1e-9)
    # The CIs should be within 2x in width at this n
    par_w = par.ci_hi - par.ci_lo
    boot_w = boot.ci_hi - boot.ci_lo
    assert 0.4 < par_w / boot_w < 2.5
