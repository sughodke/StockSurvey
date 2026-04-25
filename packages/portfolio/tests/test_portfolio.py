"""Tests for ss_portfolio: metrics, weight cap, differentiable Sharpe."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ss_portfolio import (
    annualized_sharpe,
    apply_position_cap,
    block_sharpe_with_costs,
    cagr,
    calmar,
    max_drawdown,
    sortino,
    softmax_weights,
)


def test_annualized_sharpe_known_value():
    # Daily returns with mean 0.001 and std 0.01
    daily = np.full(252, 0.001) + 0.01 * np.array([1, -1] * 126)
    s = annualized_sharpe(daily)
    # mean / std * sqrt(252) where mean=0.001, std=0.01 -> 0.1 * sqrt(252) ~ 1.587
    assert s == pytest.approx(0.001 / 0.01 * np.sqrt(252), rel=1e-3)


def test_annualized_sharpe_zero_std():
    assert annualized_sharpe(np.zeros(100)) == 0.0


def test_cagr_one_year_double():
    # 252 days that double overall
    daily = np.full(252, np.power(2.0, 1 / 252) - 1)
    assert cagr(daily) == pytest.approx(1.0, rel=1e-4)


def test_max_drawdown_simple():
    # Equity goes 1 -> 2 -> 1 -> 1.5 (peak 2, trough 1, dd = -50%)
    rets = np.array([1.0, -0.5, 0.5])
    assert max_drawdown(rets) == pytest.approx(-0.5, abs=1e-9)


def test_max_drawdown_zero_when_monotone():
    rets = np.array([0.01] * 50)
    assert max_drawdown(rets) == pytest.approx(0.0, abs=1e-9)


def test_sortino_only_downside():
    # All up days -> downside dev = 0 -> impl returns 0
    assert sortino(np.full(100, 0.001)) == 0.0


def test_calmar_positive():
    rets = np.array([0.02, -0.01, 0.03, -0.02, 0.04, -0.01] * 50)
    c = calmar(rets)
    assert np.isfinite(c)


def test_softmax_weights_sums_to_one():
    scores = np.array([1.0, 2.0, 3.0, 0.5])
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    w = softmax_weights(scores, mask, temperature=0.5)
    assert w.sum() == pytest.approx(1.0, rel=1e-6)
    assert w[2] == pytest.approx(0.0, abs=1e-9)  # masked out


def test_apply_position_cap_uniform_when_too_few_names():
    w = pd.Series([0.5, 0.5], index=['A', 'B'])
    capped = apply_position_cap(w, max_position=0.25)
    np.testing.assert_allclose(capped.values, [0.5, 0.5])  # n*cap < 1 -> uniform 1/n


def test_apply_position_cap_water_fill():
    # Heavy concentration on one name; cap at 0.25
    w = pd.Series([0.5, 0.3, 0.1, 0.05, 0.05], index=list('ABCDE'))
    capped = apply_position_cap(w, max_position=0.25)
    assert capped.sum() == pytest.approx(1.0, rel=1e-6)
    assert capped.max() <= 0.25 + 1e-9
    assert capped.loc['A'] == pytest.approx(0.25, abs=1e-9)


def test_apply_position_cap_no_op_when_within():
    w = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=list('ABCDE'))
    capped = apply_position_cap(w, max_position=0.25)
    np.testing.assert_allclose(capped.values, w.values)


def test_apply_position_cap_invalid_max():
    with pytest.raises(ValueError):
        apply_position_cap(pd.Series([1.0]), max_position=0.0)
    with pytest.raises(ValueError):
        apply_position_cap(pd.Series([1.0]), max_position=1.5)


def test_apply_position_cap_zero_sum_input():
    """All-zero input weights short-circuit to whatever the original is —
    no division by zero, no NaN. We don't strictly require sum==1 here
    (caller would have an empty portfolio anyway), only that the result
    is finite and non-negative.
    """
    w = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=list('ABCDE'))
    capped = apply_position_cap(w, max_position=0.30)
    assert np.all(np.isfinite(capped.values))
    assert (capped.values >= 0).all()


def test_block_sharpe_with_costs_shape():
    n_blocks, n_tickers = 12, 5
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal((n_blocks, n_tickers)))
    block_log_ret = jnp.asarray(rng.standard_normal((n_blocks, n_tickers)) * 0.01)
    mask = jnp.ones((n_blocks, n_tickers))
    s = block_sharpe_with_costs(
        scores, jnp.log(jnp.asarray(0.5)),
        block_log_ret, mask, rebal_days=20, commission_frac=0.001)
    assert s.shape == ()  # scalar
    assert np.isfinite(np.asarray(s))


def test_block_sharpe_differentiable():
    n_blocks, n_tickers = 8, 4
    rng = np.random.default_rng(1)
    scores = jnp.asarray(rng.standard_normal((n_blocks, n_tickers)))
    block_log_ret = jnp.asarray(rng.standard_normal((n_blocks, n_tickers)) * 0.01)
    mask = jnp.ones((n_blocks, n_tickers))

    def loss(log_temp):
        return -block_sharpe_with_costs(
            scores, log_temp, block_log_ret, mask, 20, 0.001)

    g = jax.grad(loss)(jnp.log(jnp.asarray(0.5)))
    assert np.isfinite(np.asarray(g))
