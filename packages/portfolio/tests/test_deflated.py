"""Tests for the deflated-Sharpe / PSR cross-arc statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ss_portfolio import (
    MetricBlock,
    TRADING_DAYS,
    expected_max_sharpe,
    probabilistic_sharpe,
    standardize_oos,
)
from ss_portfolio.deflated import _norm_cdf, _norm_ppf


def test_norm_cdf_known_values():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert _norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_ppf_inverts_cdf():
    for p in (0.001, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_known_value():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)


def test_expected_max_sharpe_single_trial_is_zero():
    # One trial => no selection bias => no Sharpe to beat.
    assert expected_max_sharpe(1, 0.5) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    # More trials => higher expected best-of-N Sharpe under the null.
    e10 = expected_max_sharpe(10, 0.3)
    e1000 = expected_max_sharpe(1000, 0.3)
    assert 0.0 < e10 < e1000


def test_expected_max_sharpe_scales_with_dispersion():
    assert expected_max_sharpe(100, 0.4) == pytest.approx(
        2.0 * expected_max_sharpe(100, 0.2), rel=1e-9
    )


def test_probabilistic_sharpe_gaussian_matches_tstat():
    # Under normality (skew 0, kurt 3), denom = 1 + 0.5*SR^2, so
    # z = SR*sqrt(N-1)/sqrt(1 + 0.5*SR^2).
    sr_pp, n = 0.1, 250
    psr, z = probabilistic_sharpe(sr_pp, n, skew=0.0, kurtosis=3.0)
    expected = sr_pp * math.sqrt(n - 1) / math.sqrt(1.0 + 0.5 * sr_pp ** 2)
    assert z == pytest.approx(expected, rel=1e-12)


def test_negative_skew_fat_tails_lower_psr():
    # Negative skew + excess kurtosis should reduce PSR vs the gaussian case.
    sr_pp, n = 0.1, 250
    _, z_norm = probabilistic_sharpe(sr_pp, n, skew=0.0, kurtosis=3.0)
    _, z_fat = probabilistic_sharpe(sr_pp, n, skew=-1.0, kurtosis=8.0)
    assert z_fat < z_norm


def test_standardize_oos_basic_shape_and_annualization():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=2000)
    mb = standardize_oos(r, periods_per_year=TRADING_DAYS)
    assert isinstance(mb, MetricBlock)
    assert mb.n_obs == 2000
    # ann_sharpe == per-period * sqrt(ppy)
    assert mb.ann_sharpe == pytest.approx(
        mb.sharpe_per_period * math.sqrt(TRADING_DAYS), rel=1e-9
    )
    # Single implied trial via null s.e. fallback => DSR <= PSR.
    assert mb.dsr <= mb.psr + 1e-12


def test_deflation_penalizes_many_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, size=1500)
    base = standardize_oos(r, periods_per_year=TRADING_DAYS, n_trials=1)
    many = standardize_oos(
        r, periods_per_year=TRADING_DAYS, n_trials=200, sharpe_std=0.1
    )
    # Same returns, more trials => higher bar => lower deflated t-stat & DSR.
    assert many.deflated_tstat < base.deflated_tstat
    assert many.dsr < base.dsr
    assert many.expected_max_sharpe > base.expected_max_sharpe


def test_trial_sharpes_sets_n_trials_and_dispersion():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0008, 0.012, size=1200)
    ts = rng.normal(0.05, 0.08, size=37)
    mb = standardize_oos(r, periods_per_year=TRADING_DAYS, trial_sharpes=ts)
    assert mb.n_trials == 37
    assert mb.sharpe_std == pytest.approx(ts.std(ddof=1), rel=1e-12)


def test_information_ratio_zero_when_returns_equal_benchmark():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, size=800)
    mb = standardize_oos(r, periods_per_year=TRADING_DAYS, benchmark=r.copy())
    assert mb.ir_vs_bench == pytest.approx(0.0, abs=1e-12)


def test_block_returns_use_block_annualization():
    rng = np.random.default_rng(4)
    r = rng.normal(0.01, 0.04, size=120)  # ~quarterly-ish blocks
    rebal_days = 20
    mb = standardize_oos(r, periods_per_year=TRADING_DAYS / rebal_days)
    assert mb.periods_per_year == pytest.approx(TRADING_DAYS / rebal_days)


def test_null_floor_combines_in_quadrature():
    """TODO/ladder-methodology-rewrite.md Step 1 regression: the
    `sharpe_std` argument is now interpreted as the structural-only
    component; the null estimation floor 1/sqrt(n-1) is combined in
    quadrature. So effective sharpe_std is at least the null floor."""
    import math
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, size=33)  # short sample like vol-v3

    # Pass a tiny structural component; null floor should dominate.
    mb_tiny = standardize_oos(r, periods_per_year=12.6, n_trials=12,
                              sharpe_std=0.01)
    null_floor = 1.0 / math.sqrt(33 - 1)
    expected = math.sqrt(0.01**2 + null_floor**2)
    assert mb_tiny.sharpe_std == pytest.approx(expected, rel=1e-10)

    # When struct is much larger than null floor, struct dominates.
    mb_large = standardize_oos(r, periods_per_year=12.6, n_trials=12,
                               sharpe_std=2.0)
    expected_large = math.sqrt(2.0**2 + null_floor**2)
    assert mb_large.sharpe_std == pytest.approx(expected_large, rel=1e-10)


def test_null_floor_under_deflates_legacy_fix():
    """The pre-Step-1 code used `1/sqrt(n)` as a default and treated
    caller-supplied sharpe_std as the total. For a short-sample arc
    with workspace 0.25/sqrt(ppy) calibration, the new code must give
    a STRICTLY LARGER (more punitive) deflated-t penalty."""
    import math
    rng = np.random.default_rng(6)
    r = rng.normal(0.005, 0.015, size=33)  # short like vol-v3-dolthub-oos
    ppy = 12.6

    # Old behavior would have used sharpe_std=0.25/sqrt(12.6) ≈ 0.070
    # AS THE TOTAL. New behavior treats 0.070 as the structural part
    # and adds the null floor in quadrature.
    workspace_struct = 0.072 / math.sqrt(ppy)
    mb_new = standardize_oos(r, periods_per_year=ppy, n_trials=200,
                             sharpe_std=workspace_struct)
    # Manually reconstruct what the old code would have produced:
    null_floor = 1.0 / math.sqrt(32)
    old_s_std = workspace_struct           # old code = struct as-is
    new_s_std = math.sqrt(workspace_struct**2 + null_floor**2)
    # New effective std is larger, so expected_max_sharpe is larger,
    # so deflated-t is SMALLER (more punitive).
    assert new_s_std > old_s_std
    assert mb_new.sharpe_std == pytest.approx(new_s_std, rel=1e-10)
    assert mb_new.expected_max_sharpe > old_s_std  # punitive direction


def test_null_floor_only_when_sharpe_std_omitted():
    """If sharpe_std is None, only the null floor is used (most
    conservative possible deflation)."""
    import math
    rng = np.random.default_rng(7)
    r = rng.normal(0.001, 0.01, size=100)
    mb = standardize_oos(r, periods_per_year=252, n_trials=10)
    null_floor = 1.0 / math.sqrt(99)
    assert mb.sharpe_std == pytest.approx(null_floor, rel=1e-10)
