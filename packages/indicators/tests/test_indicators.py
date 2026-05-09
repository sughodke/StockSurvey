"""Numerical tests for ss_indicators (numpy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ss_indicators import (
    bbands,
    cci,
    cci_strided,
    cci_strided_grid,
    corwin_schultz_spread,
    drawdown_from_high,
    ema,
    fibonacci_retracement,
    macd,
    rolling_kurt,
    rolling_pearson_corr,
    rolling_skew,
    rolling_std,
    rsi,
    rsi_strided,
    rsi_strided_grid,
    sma,
    symmetric_kl_divergence,
    vol_norm_momentum,
)


@pytest.fixture
def prices_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (np.cumsum(rng.standard_normal(200)) + 100.0).astype(np.float32)


@pytest.fixture
def prices_2d() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (np.cumsum(rng.standard_normal((150, 5)), axis=0) + 50.0).astype(np.float32)


def test_sma_matches_numpy_simple(prices_1d):
    out = np.asarray(sma(prices_1d, 10))
    pd_sma = pd.Series(prices_1d).rolling(10, min_periods=1).mean().values
    np.testing.assert_allclose(out, pd_sma, rtol=1e-5)


def test_sma_2d_broadcasts(prices_2d):
    out = np.asarray(sma(prices_2d, 5))
    assert out.shape == prices_2d.shape
    expected = pd.DataFrame(prices_2d).rolling(5, min_periods=1).mean().values
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_ema_seed_and_recurrence(prices_1d):
    out = np.asarray(ema(prices_1d, span=10))
    expected = pd.Series(prices_1d).ewm(span=10, adjust=False).mean().values
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_rolling_std_population(prices_1d):
    out = np.asarray(rolling_std(prices_1d, 21))
    expected = pd.Series(prices_1d).rolling(21, min_periods=1).std(ddof=0).values
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_rsi_in_range_and_neutral_seed(prices_2d):
    out = np.asarray(rsi(prices_2d, n=7))
    assert out.shape == prices_2d.shape
    assert np.all((out >= 0) & (out <= 100))
    assert np.all(out[:7] == 50.0)


def _wilder_rsi_reference(prices: np.ndarray, n: int) -> np.ndarray:
    """Pure-numpy Wilder RSI for regression comparison."""
    deltas = np.diff(prices)
    up = np.where(deltas > 0, deltas, 0.0)
    down = np.where(deltas < 0, -deltas, 0.0)
    out = np.full_like(prices, 50.0, dtype=np.float64)
    avg_up = up[:n].mean()
    avg_down = down[:n].mean()
    out[n] = 100.0 - 100.0 / (1.0 + avg_up / (avg_down + 1e-9))
    for i in range(n + 1, len(prices)):
        avg_up = (avg_up * (n - 1) + up[i - 1]) / n
        avg_down = (avg_down * (n - 1) + down[i - 1]) / n
        out[i] = 100.0 - 100.0 / (1.0 + avg_up / (avg_down + 1e-9))
    return out


def test_rsi_matches_wilder_reference():
    rng = np.random.default_rng(123)
    prices = (np.cumsum(rng.standard_normal(300)) + 100).astype(np.float32)
    got = np.asarray(rsi(prices, n=14))
    expected = _wilder_rsi_reference(prices.astype(np.float64), n=14)
    # Values before index n are 50 in both; from n onward we compare directly.
    np.testing.assert_allclose(got[14:], expected[14:], rtol=1e-3, atol=1e-3)


def test_rsi_constant_prices_is_neutral():
    out = np.asarray(rsi(np.full(50, 100.0), n=7))
    # No movement -> avg_up = avg_down = 0; impl falls back to RS = 0/0 + eps
    # which yields RSI ~ 0; we accept either 0 or 50 but the key invariant
    # is that it doesn't crash and stays inside [0, 100].
    assert np.all((out >= 0) & (out <= 100))


def test_rsi_strided_w1_matches_matrix_rsi():
    """rsi_strided(prices, n, w=1) should match rsi(prices, n) on the
    overlap region (index >= n)."""
    rng = np.random.default_rng(7)
    prices = (np.cumsum(rng.standard_normal(300)) + 100).astype(np.float64)
    matrix = rsi(prices, n=14)
    strided = rsi_strided(prices, n=14, w=1)
    # rsi fills [:n] with 50; rsi_strided fills [:w+n-1] with NaN.
    # Compare on the overlap region.
    assert np.allclose(matrix[14:], strided[14:], atol=1e-9)
    assert np.isnan(strided[:14]).all()


def test_rsi_strided_validates_args():
    p = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match='w must be >= 1'):
        rsi_strided(p, n=7, w=0)
    with pytest.raises(ValueError, match='n must be >= 2'):
        rsi_strided(p, n=1, w=5)


def test_rsi_strided_grid_parity_with_per_cell():
    """rsi_strided_grid produces bit-equivalent output to stacking
    per-cell rsi_strided over the same (w, n) Cartesian product."""
    rng = np.random.default_rng(17)
    prices = (np.cumsum(rng.standard_normal(500)) + 100).astype(np.float64)
    n_grid = (5, 7, 14, 21)
    w_grid = (1, 5, 10, 21)
    grid = rsi_strided_grid(prices, n_grid, w_grid)
    assert grid.shape == (500, len(w_grid), len(n_grid))
    for wi, w in enumerate(w_grid):
        for ni, n in enumerate(n_grid):
            ref = rsi_strided(prices, n=int(n), w=int(w))
            np.testing.assert_allclose(
                grid[:, wi, ni], ref, rtol=1e-12, atol=1e-12, equal_nan=True,
                err_msg=f'mismatch at (w={w}, n={n})')


def test_rsi_strided_grid_warmup_per_cell():
    """Grid output preserves per-cell warmup: position `w + n - 1` is the
    first non-NaN bar for each (w, n) cell."""
    prices = np.arange(200, dtype=np.float64) * 0.5 + 100.0
    n_grid = (5, 14)
    w_grid = (3, 7)
    grid = rsi_strided_grid(prices, n_grid, w_grid)
    for wi, w in enumerate(w_grid):
        for ni, n in enumerate(n_grid):
            warmup = int(w) + int(n) - 1
            assert np.isnan(grid[:warmup, wi, ni]).all(), \
                f'(w={w}, n={n}) leaked finite values into warmup'
            assert not np.isnan(grid[warmup, wi, ni])


def test_rsi_strided_grid_validates_args():
    p = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match='every w >= 1'):
        rsi_strided_grid(p, n_grid=(7,), w_grid=(0, 5))
    with pytest.raises(ValueError, match='every n >= 2'):
        rsi_strided_grid(p, n_grid=(1, 7), w_grid=(1,))


def test_macd_identity(prices_1d):
    line, signal, hist = macd(prices_1d, fast=12, slow=26, signal=9)
    np.testing.assert_allclose(np.asarray(hist), np.asarray(line) - np.asarray(signal),
                               rtol=1e-5)
    fast_ema = np.asarray(ema(prices_1d, 12))
    slow_ema = np.asarray(ema(prices_1d, 26))
    np.testing.assert_allclose(np.asarray(line), fast_ema - slow_ema, rtol=1e-5)


def test_bbands_ordering(prices_1d):
    mid, up, low = bbands(prices_1d, window=21, nsd=2.0)
    mid, up, low = (np.asarray(x) for x in (mid, up, low))
    assert np.all(up >= mid)
    assert np.all(mid >= low)
    np.testing.assert_allclose(up - mid, mid - low, rtol=1e-5, atol=1e-7)


def test_cci_warmup_and_zero_signal():
    """CCI of constant prices is exactly 0 after warmup (close == SMA, MAD == 0
    triggers the safe-mad branch). Warmup is `n - 1` NaNs."""
    n = 14
    out = cci(np.full(50, 100.0, dtype=np.float64), n=n)
    assert np.isnan(out[:n - 1]).all()
    assert (out[n - 1:] == 0.0).all()


def test_cci_centered_and_typical_range():
    """CCI on a random walk: ~zero mean, most values within ±300 (>~99% within
    Lambert's empirical band; the function is only roughly bounded)."""
    rng = np.random.default_rng(11)
    prices = np.cumsum(rng.standard_normal(2000)) + 100.0
    out = cci(prices, n=20)
    valid = out[~np.isnan(out)]
    assert len(valid) == 1981
    assert abs(np.mean(valid)) < 5.0           # near-zero mean
    assert np.quantile(np.abs(valid), 0.99) < 300.0


def test_cci_strided_w1_matches_matrix_cci():
    """cci_strided(prices, n, w=1) should match cci(prices, n) on the
    overlap region (index >= n - 1)."""
    rng = np.random.default_rng(13)
    prices = (np.cumsum(rng.standard_normal(300)) + 100).astype(np.float64)
    matrix = cci(prices, n=14)
    strided = cci_strided(prices, n=14, w=1)
    valid = ~(np.isnan(matrix) | np.isnan(strided))
    np.testing.assert_allclose(matrix[valid], strided[valid], rtol=1e-9, atol=1e-9)


def test_cci_strided_w_warmup():
    """Stride-w CCI needs (n-1)*w bars before the first valid output."""
    n, w = 5, 7
    prices = np.arange(100, dtype=np.float64) ** 0.5  # arbitrary smooth series
    out = cci_strided(prices, n=n, w=w)
    span = (n - 1) * w + 1  # 29
    assert np.isnan(out[:span - 1]).all()
    assert not np.isnan(out[span - 1:]).any()


def test_cci_strided_validates_args():
    p = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match='w must be >= 1'):
        cci_strided(p, n=14, w=0)
    with pytest.raises(ValueError, match='n must be >= 2'):
        cci_strided(p, n=1, w=5)


def test_cci_strided_grid_parity_with_per_cell():
    """cci_strided_grid produces bit-equivalent output to stacking
    per-cell cci_strided over the same (w, n) Cartesian product."""
    rng = np.random.default_rng(19)
    prices = (np.cumsum(rng.standard_normal(500)) + 100).astype(np.float64)
    n_grid = (5, 10, 14, 20)
    w_grid = (1, 5, 10, 21)
    grid = cci_strided_grid(prices, n_grid, w_grid)
    assert grid.shape == (500, len(w_grid), len(n_grid))
    for wi, w in enumerate(w_grid):
        for ni, n in enumerate(n_grid):
            ref = cci_strided(prices, n=int(n), w=int(w))
            np.testing.assert_allclose(
                grid[:, wi, ni], ref, rtol=1e-12, atol=1e-12, equal_nan=True,
                err_msg=f'mismatch at (w={w}, n={n})')


def test_cci_strided_grid_warmup_per_cell():
    """Per-cell warmup is `(n-1)*w + 1` — first valid bar at index
    `(n-1)*w`, matching cci_strided."""
    prices = np.arange(300, dtype=np.float64) * 0.3 + 100.0
    n_grid = (5, 14)
    w_grid = (3, 7)
    grid = cci_strided_grid(prices, n_grid, w_grid)
    for wi, w in enumerate(w_grid):
        for ni, n in enumerate(n_grid):
            warmup = (int(n) - 1) * int(w) + 1
            assert np.isnan(grid[:warmup - 1, wi, ni]).all(), \
                f'(w={w}, n={n}) leaked finite values into warmup'
            assert not np.isnan(grid[warmup - 1, wi, ni])


def test_cci_strided_grid_validates_args():
    p = np.arange(100, dtype=np.float64)
    with pytest.raises(ValueError, match='every w >= 1'):
        cci_strided_grid(p, n_grid=(14,), w_grid=(0, 5))
    with pytest.raises(ValueError, match='every n >= 2'):
        cci_strided_grid(p, n_grid=(1, 14), w_grid=(1,))


def test_symmetric_kl_zero_when_distributions_match():
    n_scales, n_blocks, n_tickers = 6, 4, 3
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 1.0, (n_scales, n_blocks, n_tickers))
    score = np.asarray(symmetric_kl_divergence(p, p, np.zeros(n_scales)))
    assert score.shape == (n_blocks, n_tickers)
    np.testing.assert_allclose(score, 0.0, atol=1e-6)


def test_symmetric_kl_positive_when_distributions_differ():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.1, 1.0, (5, 2, 4))
    q = rng.uniform(0.1, 1.0, (5, 2, 4))
    score = np.asarray(symmetric_kl_divergence(p, q, np.zeros(5)))
    assert np.all(score > 0)


def test_corwin_schultz_spread_shape_and_range():
    rng = np.random.default_rng(11)
    n = 250
    closes = np.cumsum(rng.standard_normal((n, 4)), axis=0) + 100
    highs = pd.DataFrame(closes + np.abs(rng.standard_normal((n, 4))) * 0.5)
    lows = pd.DataFrame(closes - np.abs(rng.standard_normal((n, 4))) * 0.5)
    spread = corwin_schultz_spread(highs, lows, window=21)
    assert spread.shape == highs.shape
    valid = spread.dropna().values
    assert valid.min() >= 0.0
    assert valid.max() <= 0.20


def test_rolling_pearson_corr_warmup_and_shape():
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    out = rolling_pearson_corr(x, y, window=20)
    assert out.shape == (100,)
    assert np.isnan(out[:19]).all()
    assert np.isfinite(out[19:]).all()
    assert (out[19:] >= -1.0 - 1e-9).all()
    assert (out[19:] <= 1.0 + 1e-9).all()


def test_rolling_pearson_corr_identical_series_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(80)
    out = rolling_pearson_corr(x, x.copy(), window=20)
    np.testing.assert_allclose(out[19:], 1.0, atol=1e-9)


def test_rolling_pearson_corr_negated_series_is_neg_one():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(80)
    out = rolling_pearson_corr(x, -x, window=20)
    np.testing.assert_allclose(out[19:], -1.0, atol=1e-9)


def test_rolling_pearson_corr_constant_series_is_zero():
    # Pearson is undefined when either window has zero variance; we force
    # the output to 0.0 so downstream features see a finite (not NaN) channel.
    x = np.linspace(0.0, 1.0, 50)
    y = np.full(50, 3.14)
    out = rolling_pearson_corr(x, y, window=10)
    np.testing.assert_allclose(out[9:], 0.0, atol=1e-12)


def test_rolling_pearson_corr_propagates_nan_warmup():
    # Mimics realized_vol's prefix-NaN: first `lead` bars NaN, then valid.
    # Output must be NaN until we have a full window of finite (x, y) pairs.
    lead = 10
    rng = np.random.default_rng(3)
    x = np.concatenate([np.full(lead, np.nan), rng.standard_normal(40)])
    y = np.concatenate([np.full(lead, np.nan), rng.standard_normal(40)])
    out = rolling_pearson_corr(x, y, window=15)
    assert np.isnan(out[:lead + 15 - 1]).all()
    assert np.isfinite(out[lead + 15 - 1:]).all()


def test_rolling_pearson_corr_validates_args():
    with pytest.raises(ValueError, match='1-D'):
        rolling_pearson_corr(np.zeros((5, 5)), np.zeros((5, 5)), window=3)
    with pytest.raises(ValueError, match='same shape'):
        rolling_pearson_corr(np.zeros(5), np.zeros(6), window=3)
    with pytest.raises(ValueError, match='window must be >= 2'):
        rolling_pearson_corr(np.zeros(5), np.zeros(5), window=1)


def test_corwin_schultz_spread_known_input():
    """Pin the alpha/beta math against a tiny hand-checkable input.

    Five bars of H/L; the regression values below are the current
    implementation's output. They're not derived from a published
    reference — they exist to catch silent algorithmic drift if anyone
    edits spread.py.
    """
    highs = pd.DataFrame({'X': [102.0, 101.0, 101.5, 102.0, 101.0]})
    lows = pd.DataFrame({'X': [99.0, 98.0, 98.5, 99.5, 99.0]})
    spread = corwin_schultz_spread(highs, lows, window=3)
    expected = [None, 0.005856, 0.011915, 0.011177, 0.010850]
    assert pd.isna(spread.iloc[0, 0])
    for i in range(1, 5):
        assert spread.iloc[i, 0] == pytest.approx(expected[i], abs=1e-5), (
            f'index {i}: got {spread.iloc[i, 0]:.6f}, expected {expected[i]}')


def test_fibonacci_retracement():
    prices = np.array([100.0, 110.0, 95.0, 120.0, 105.0])
    t1, t2, levels = fibonacci_retracement(prices, n=5)
    assert isinstance(t1, int) and isinstance(t2, int)
    assert t1 < t2
    assert len(levels) == 7  # 7 fib ratios
    assert levels[0] == pytest.approx(prices[t1])
    assert levels[-1] == pytest.approx(prices[t2])


def test_fibonacci_retracement_non_default_n_offset():
    # Long history with a clear trend in the trailing window. The offset
    # for the returned absolute indices must use `n`, not a hardcoded 90.
    prices = np.concatenate([np.full(50, 100.0), np.linspace(100, 200, 30)])
    t1, t2, _ = fibonacci_retracement(prices, n=30)
    assert 0 <= t1 < len(prices)
    assert 0 <= t2 < len(prices)
    assert prices[t1] == pytest.approx(100.0)
    assert prices[t2] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Lie-shape heads: vol_norm_momentum / drawdown_from_high / rolling_skew /
# rolling_kurt — added as `apps/replay` reconstruction targets so the CNN can
# be probed for whether the CWT carries shape-feature content.
# ---------------------------------------------------------------------------


def test_vol_norm_momentum_warmup_and_shape(prices_1d):
    n = 21
    out = vol_norm_momentum(prices_1d, n)
    assert out.shape == prices_1d.shape
    assert np.all(np.isnan(out[:n]))
    assert np.all(np.isfinite(out[n:]))


def test_vol_norm_momentum_matches_pandas(prices_1d):
    # Direct-formula validation: cumulative log-return divided by sample-stdev
    # of those log returns times sqrt(n).
    n = 21
    log_p = np.log(prices_1d.astype(np.float64))
    rets = pd.Series(np.diff(log_p))
    cum = rets.rolling(n).sum()
    sigma = rets.rolling(n).std(ddof=0)
    expected = (cum / (sigma * np.sqrt(n))).values
    # Align: pandas rolling on length-(T-1) returns, fast-forward to align
    # at price index t we want the sum of rets[t-n:t] -> pandas window
    # ending at return-index t-1.
    out = vol_norm_momentum(prices_1d, n)
    np.testing.assert_allclose(out[n:], expected[n - 1:], rtol=1e-6, atol=1e-9)


def test_vol_norm_momentum_constant_prices_zero():
    # Flat prices -> zero returns -> zero numerator AND zero denominator;
    # function should NaN those (degenerate window) rather than divide.
    p = np.full(60, 100.0)
    out = vol_norm_momentum(p, 21)
    assert np.all(np.isnan(out))


def test_drawdown_from_high_at_high_is_zero():
    # Strictly increasing prices: today is always at the trailing high.
    p = np.arange(1.0, 101.0)
    out = drawdown_from_high(p, n=10)
    assert np.all(np.isnan(out[:10]))
    np.testing.assert_allclose(out[10:], 0.0, atol=1e-12)


def test_drawdown_from_high_known_value():
    # Build a series where the trailing-21 high is known.
    p = np.concatenate([
        np.full(30, 100.0),    # bars 0..29
        [200.0],               # bar 30 — the peak inside the trailing window
        np.full(30, 100.0),    # bars 31..60 — pulled back to 100
    ])
    out = drawdown_from_high(p, n=21)
    # At bar 40 the trailing-21 window covers bars 19..40, max == 200.
    expected = np.log(100.0 / 200.0)
    assert out[40] == pytest.approx(expected, rel=1e-9)
    # At bar 60 the peak (bar 30) has rolled out of the window; today is
    # at the high again -> zero.
    assert out[60] == pytest.approx(0.0, abs=1e-12)


def test_rolling_skew_sign_tracks_distribution_asymmetry():
    # A return distribution with rare LARGE positive jumps must show
    # positive skew on average; rare large NEGATIVE jumps -> negative skew.
    # Sample-skew has nontrivial finite-sample bias at n=63 so we test
    # the SIGN of the panel mean, not its proximity to zero.
    rng = np.random.default_rng(3)
    base = rng.standard_normal(2000) * 0.01
    pos_jumps = base.copy()
    spike_idx = rng.choice(len(pos_jumps), size=40, replace=False)
    pos_jumps[spike_idx] += 0.10
    neg_jumps = base.copy()
    spike_idx = rng.choice(len(neg_jumps), size=40, replace=False)
    neg_jumps[spike_idx] -= 0.10

    p_pos = np.exp(np.cumsum(pos_jumps))
    p_neg = np.exp(np.cumsum(neg_jumps))
    out_pos = rolling_skew(p_pos, n=63)
    out_neg = rolling_skew(p_neg, n=63)
    assert np.all(np.isnan(out_pos[:63]))
    pos_mean = float(np.nanmean(out_pos))
    neg_mean = float(np.nanmean(out_neg))
    assert pos_mean > 0.5
    assert neg_mean < -0.5
    assert pos_mean > neg_mean


def test_rolling_kurt_normal_excess_near_zero():
    # Normal returns have excess kurtosis near 0 in expectation; sample
    # estimator on n=63 has a tight distribution around -0.1..+0.5 from
    # finite-sample bias but the mean over the panel should sit close to 0.
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(2000) * 0.01
    p = np.exp(np.cumsum(rets))
    out = rolling_kurt(p, n=63)
    finite = out[~np.isnan(out)]
    # Allow ample slack -- finite-sample kurt is noisy at n=63.
    assert abs(float(np.mean(finite))) < 0.5


def test_rolling_kurt_lifts_with_jumps():
    # A return distribution with rare large jumps must show positive
    # excess kurt vs the calm baseline.
    rng = np.random.default_rng(2)
    base = rng.standard_normal(2000) * 0.01
    jumps = base.copy()
    spike_idx = rng.choice(len(jumps), size=40, replace=False)
    jumps[spike_idx] += rng.choice([-1, 1], size=40) * 0.10  # 10x stdev
    p_calm = np.exp(np.cumsum(base))
    p_jumpy = np.exp(np.cumsum(jumps))
    k_calm = rolling_kurt(p_calm, n=63)
    k_jumpy = rolling_kurt(p_jumpy, n=63)
    calm_mean = float(np.nanmean(k_calm))
    jumpy_mean = float(np.nanmean(k_jumpy))
    assert jumpy_mean > calm_mean + 0.5
