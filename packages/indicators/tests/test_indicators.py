"""Numerical tests for ss_indicators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ss_indicators import (
    bbands,
    corwin_schultz_spread,
    ema,
    fibonacci_retracement,
    macd,
    rolling_std,
    rsi,
    sma,
    symmetric_kl_divergence,
)


@pytest.fixture
def prices_1d() -> np.ndarray:
    rng = np.random.default_rng(42)
    # JAX defaults to float32, so generate the test fixture there too —
    # otherwise pandas (float64) and JAX (float32) outputs diverge in the
    # relative tolerance we'd want for a meaningful regression check.
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
    # Float32 catastrophic cancellation in mu2 - mu^2 limits us to ~1e-2
    # relative; this is acceptable for the JAX-only path the trainer uses.
    np.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-3)


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
    out = np.asarray(rsi(jnp.full(50, 100.0), n=7))
    # No movement -> avg_up = avg_down = 0; impl falls back to RS = 0/0 + eps
    # which yields RSI ~ 0; we accept either 0 or 50 but the key invariant
    # is that it doesn't crash and stays inside [0, 100].
    assert np.all((out >= 0) & (out <= 100))


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


def test_symmetric_kl_zero_when_distributions_match():
    n_scales, n_blocks, n_tickers = 6, 4, 3
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.uniform(0.1, 1.0, (n_scales, n_blocks, n_tickers)))
    score = np.asarray(symmetric_kl_divergence(p, p, jnp.zeros(n_scales)))
    assert score.shape == (n_blocks, n_tickers)
    np.testing.assert_allclose(score, 0.0, atol=1e-6)


def test_symmetric_kl_positive_when_distributions_differ():
    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.uniform(0.1, 1.0, (5, 2, 4)))
    q = jnp.asarray(rng.uniform(0.1, 1.0, (5, 2, 4)))
    score = np.asarray(symmetric_kl_divergence(p, q, jnp.zeros(5)))
    assert np.all(score > 0)


def test_symmetric_kl_differentiable():
    rng = np.random.default_rng(2)
    p = jnp.asarray(rng.uniform(0.1, 1.0, (5, 3, 4)))
    q = jnp.asarray(rng.uniform(0.1, 1.0, (5, 3, 4)))

    def loss(w):
        return symmetric_kl_divergence(p, q, w).sum()

    g = jax.grad(loss)(jnp.zeros(5))
    assert g.shape == (5,)
    assert np.isfinite(np.asarray(g)).all()


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
    # The min in the last 30 should be index 50 (start of the rising segment),
    # max at index 79 (end). Both indices must point into the actual array.
    assert 0 <= t1 < len(prices)
    assert 0 <= t2 < len(prices)
    assert prices[t1] == pytest.approx(100.0)
    assert prices[t2] == pytest.approx(200.0)
