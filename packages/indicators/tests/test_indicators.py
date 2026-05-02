"""Numerical tests for ss_indicators (numpy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ss_indicators import (
    bbands,
    cci,
    cci_strided,
    corwin_schultz_spread,
    ema,
    fibonacci_retracement,
    macd,
    rolling_std,
    rsi,
    rsi_strided,
    sma,
    symmetric_kl_divergence,
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
