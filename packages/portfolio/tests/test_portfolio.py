"""Tests for ss_portfolio: metrics, weight cap, block Sharpe."""

from __future__ import annotations

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
    # n_nonzero=2, cap=0.25, 2*0.25=0.5 < 1 -> uniform 1/n_nonzero across nonzero
    w = pd.Series([0.5, 0.5], index=['A', 'B'])
    capped = apply_position_cap(w, max_position=0.25)
    np.testing.assert_allclose(capped.values, [0.5, 0.5])


def test_apply_position_cap_skips_zero_weight_names():
    """Sparse input (most names zero, 20 nonzero) with a cap so small
    that the degenerate branch trips. The redistribution must NOT
    re-introduce the zero names — they were zeroed out by an upstream
    spread gate or selection step and must stay out.

    Regression for code-review finding #1 (2026-05-06).
    """
    n_total, n_nonzero = 250, 20
    values = np.zeros(n_total)
    values[:n_nonzero] = 1.0 / n_nonzero
    w = pd.Series(values, index=[f'T{i}' for i in range(n_total)])
    # 20 * 0.003 = 0.06 < 1 -> degenerate branch
    capped = apply_position_cap(w, max_position=0.003)
    assert capped.sum() == pytest.approx(1.0, abs=1e-9)
    assert (capped.iloc[n_nonzero:] == 0).all(), 'zero-weight names must stay zero'
    np.testing.assert_allclose(
        capped.iloc[:n_nonzero].values, np.full(n_nonzero, 1.0 / n_nonzero))


def test_apply_position_cap_all_zero_input():
    w = pd.Series([0.0, 0.0, 0.0], index=list('ABC'))
    capped = apply_position_cap(w, max_position=0.25)
    np.testing.assert_allclose(capped.values, [0.0, 0.0, 0.0])


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


def test_select_top_n_matrix_descending():
    from ss_portfolio import select_top_n_matrix
    scores = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    w = select_top_n_matrix(scores, top_n=2, ascending=False)
    # Top-2 by score: [4,5] in row 0 (idx 3,4); [4,5] in row 1 (idx 0,1)
    assert w[0, 3] == 0.5 and w[0, 4] == 0.5 and w[0, :3].sum() == 0
    assert w[1, 0] == 0.5 and w[1, 1] == 0.5 and w[1, 2:].sum() == 0


def test_select_top_n_matrix_ascending_with_nan():
    from ss_portfolio import select_top_n_matrix
    scores = np.array([
        [1.0, np.nan, 3.0, 2.0],
        [np.nan, np.nan, 1.0, 2.0],
    ])
    w = select_top_n_matrix(scores, top_n=2, ascending=True)
    # Row 0: lowest two valid = idx 0, 3 (scores 1, 2)
    assert w[0, 0] == 0.5 and w[0, 3] == 0.5
    # Row 1: only 2 valid, picked
    assert w[1, 2] == 0.5 and w[1, 3] == 0.5


def test_select_top_n_matrix_skips_when_too_few_valid():
    from ss_portfolio import select_top_n_matrix
    scores = np.array([[1.0, 2.0, np.nan, np.nan]])
    w = select_top_n_matrix(scores, top_n=3, ascending=True)
    # Fewer than top_n valid → entire row stays zero
    assert (w == 0).all()


def test_apply_spread_mask():
    from ss_portfolio import apply_spread_mask
    scores = np.zeros((3, 4))  # n_valid=3, n_tickers=4
    spread = np.array([
        [0.01, 0.05, 0.01, 0.01],  # warmup
        [0.01, 0.05, 0.01, 0.01],
        [0.01, 0.05, 0.01, 0.01],
        [0.01, 0.05, 0.01, 0.01],
        [0.01, 0.05, 0.01, 0.01],
    ])  # n_dates=5
    out = apply_spread_mask(scores.copy(), spread, lookback=2, max_spread=0.02)
    # Ticker 1 always has spread > 0.02 → all-NaN column in scores
    assert np.isnan(out[:, 1]).all()
    # Other tickers untouched
    assert (out[:, [0, 2, 3]] == 0).all()


def test_apply_nan_mask():
    from ss_portfolio import apply_nan_mask
    scores = np.zeros((3, 3))  # n_valid=3, n_tickers=3
    prices = np.array([
        [100, np.nan, 100],
        [100, np.nan, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ], dtype=float)
    out = apply_nan_mask(scores.copy(), prices, lookback=2)
    # At valid index 0 (date=2), lookback window is dates 0..2 — ticker 1
    # had NaN in dates 0..1, so its score is NaN-masked.
    assert np.isnan(out[0, 1])
    # By valid index 2 (date=4), window dates 2..4 — ticker 1 has no NaN.
    assert out[2, 1] == 0


def test_vbt_backtest_smoke():
    """End-to-end smoke against a tiny synthetic universe.

    Skipped when vectorbt isn't importable (i.e., outside the nix shell
    on Intel macOS Python 3.13). When present, verifies the returned
    metrics dict has the expected keys with finite values.
    """
    pytest.importorskip('vectorbt')
    from ss_portfolio import vbt_backtest

    rng = np.random.default_rng(0)
    n_days, n_tickers = 252, 5
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    closes = pd.DataFrame(
        np.cumprod(1 + rng.standard_normal((n_days, n_tickers)) * 0.01, axis=0) * 100,
        index=dates, columns=[f'T{i}' for i in range(n_tickers)])
    # Hold all five names equally for the whole window.
    weights = pd.DataFrame(
        np.full((n_days, n_tickers), 1.0 / n_tickers),
        index=dates, columns=closes.columns)

    out = vbt_backtest(closes, weights, rebalance_days=20, commission_bps=5)
    assert set(out) == {'sharpe', 'cagr', 'max_drawdown', 'total_return'}
    assert all(np.isfinite(v) for v in out.values())
    assert out['max_drawdown'] <= 0


def test_vbt_backtest_fill_lag_shifts_orders():
    """`fill_lag=1` moves the rebalance fills one bar forward vs `fill_lag=0`,
    so the two backtests on the same weights produce different total returns
    (because they're filling at different prices). Confirms the shift is
    actually applied, not silently ignored."""
    pytest.importorskip('vectorbt')
    from ss_portfolio import vbt_backtest

    rng = np.random.default_rng(42)
    n_days, n_tickers = 252, 5
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    closes = pd.DataFrame(
        np.cumprod(1 + rng.standard_normal((n_days, n_tickers)) * 0.02, axis=0) * 100,
        index=dates, columns=[f'T{i}' for i in range(n_tickers)])
    weights = pd.DataFrame(
        np.full((n_days, n_tickers), 1.0 / n_tickers),
        index=dates, columns=closes.columns)

    same_bar = vbt_backtest(closes, weights, rebalance_days=20, fill_lag=0)
    next_bar = vbt_backtest(closes, weights, rebalance_days=20, fill_lag=1)
    # Different fill prices → different total return. Sign of the gap is
    # noisy on synthetic data, so we only assert the values aren't equal.
    assert same_bar['total_return'] != next_bar['total_return']


def test_vbt_backtest_spread_increases_cost():
    """Passing a non-zero spread_df must depress total_return vs the same
    backtest with no spread. Confirms the per-(date, ticker) fees matrix
    is wired into vectorbt and not silently dropped."""
    pytest.importorskip('vectorbt')
    from ss_portfolio import vbt_backtest

    rng = np.random.default_rng(0)
    n_days, n_tickers = 252, 5
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    closes = pd.DataFrame(
        np.cumprod(1 + rng.standard_normal((n_days, n_tickers)) * 0.01, axis=0) * 100,
        index=dates, columns=[f'T{i}' for i in range(n_tickers)])
    weights = pd.DataFrame(
        np.full((n_days, n_tickers), 1.0 / n_tickers),
        index=dates, columns=closes.columns)

    cheap = vbt_backtest(closes, weights, rebalance_days=20, commission_bps=5)
    spread = pd.DataFrame(0.05, index=dates, columns=closes.columns)  # 5%
    expensive = vbt_backtest(
        closes, weights, rebalance_days=20, commission_bps=5, spread_df=spread)
    # 5% spread → +2.5% per-side fee on top of 5bps commission. Across
    # ~12 rebalances over the year, this must reduce total_return.
    assert expensive['total_return'] < cheap['total_return']


def test_block_sharpe_with_costs_shape():
    n_blocks, n_tickers = 12, 5
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((n_blocks, n_tickers))
    block_log_ret = rng.standard_normal((n_blocks, n_tickers)) * 0.01
    mask = np.ones((n_blocks, n_tickers))
    s = block_sharpe_with_costs(
        scores, np.log(0.5),
        block_log_ret, mask, rebal_days=20, commission_frac=0.001)
    assert s.shape == ()  # scalar
    assert np.isfinite(s)


def test_block_sharpe_with_costs_costs_reduce_sharpe():
    # Higher commission must reduce the reported Sharpe (or leave it equal
    # only in the degenerate zero-turnover case, which this random init
    # is not).
    n_blocks, n_tickers = 12, 5
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((n_blocks, n_tickers))
    block_log_ret = rng.standard_normal((n_blocks, n_tickers)) * 0.01
    mask = np.ones((n_blocks, n_tickers))
    s_cheap = block_sharpe_with_costs(
        scores, np.log(0.5), block_log_ret, mask, 20, 0.0)
    s_expensive = block_sharpe_with_costs(
        scores, np.log(0.5), block_log_ret, mask, 20, 0.05)
    assert s_expensive < s_cheap
