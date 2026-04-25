"""Sanity checks on the legacy v1 1D indicator helpers.

These exist to lock in the legacy behavior — the v1 implementations
predate `ss_indicators` and have different defaults / shapes. If any of
these fail, the parked v1 web service will silently behave differently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v1.util.indicators import (
    bbands,
    fibonacci_retracement,
    interesting_fib,
    moving_average,
    moving_average_convergence,
    relative_strength,
)


def test_relative_strength_in_range():
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.standard_normal(200)) + 100
    rsi = relative_strength(prices, n=14)
    assert rsi.shape == prices.shape
    assert np.all((rsi >= 0) & (rsi <= 100))


def test_moving_average_simple_first_n_filled():
    x = np.arange(20.0)
    ma = moving_average(x, 5, type='simple')
    # Initial entries are filled with the first valid average
    assert np.isfinite(ma).all()
    assert len(ma) == len(x)


def test_moving_average_convergence_returns_three():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(100)) + 100
    slow, fast, macd = moving_average_convergence(x, nslow=26, nfast=12)
    assert slow.shape == fast.shape == macd.shape == x.shape
    np.testing.assert_allclose(macd, fast - slow, rtol=1e-5)


def test_bbands_returns_three():
    s = pd.Series(np.cumsum(np.random.default_rng(2).standard_normal(100)) + 100)
    mid, up, dn = bbands(s, length=20, numsd=2)
    assert mid.shape == up.shape == dn.shape == s.shape


def test_fibonacci_retracement_levels():
    prices = np.array([100.0, 110.0, 95.0, 120.0, 105.0])
    t1, t2, levels = fibonacci_retracement(prices, n=5)
    assert t1 < t2
    assert len(levels) == len(interesting_fib)
    assert levels[0] == pytest.approx(prices[t1])
    assert levels[-1] == pytest.approx(prices[t2])
