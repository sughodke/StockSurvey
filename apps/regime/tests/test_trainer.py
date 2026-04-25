"""Tests for the new vectorbt+Optuna walk-forward trainer.

These tests import vectorbt; outside the nix shell on Intel macOS
Python 3.13 they skip cleanly. The smoke tests use a tiny synthetic
universe and `n_trials=2` to keep CI fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic(n_days: int = 1500, n_tickers: int = 6, seed: int = 0):
    """Long enough to fit one walk-forward window of train+val."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_days)
    closes = np.cumprod(
        1 + rng.standard_normal((n_days, n_tickers)) * 0.01, axis=0) * 100
    prices = pd.DataFrame(
        closes, index=dates,
        columns=[f'T{i}' for i in range(n_tickers)])
    spread_df = pd.DataFrame(0.005, index=dates, columns=prices.columns)
    return prices, spread_df


def test_regime_weights_smoke():
    """`regime_weights` produces a DataFrame with the right shape and a
    valid hard-top-N row sum at every non-warmup date."""
    pytest.importorskip('vectorbt')
    from regime.trainer import regime_weights

    prices, spread_df = _synthetic(n_days=300, n_tickers=8)
    weights = regime_weights(
        prices, lookback=60, n_tail=10, top_n=3,
        scales=[5, 21, 90], divergence='kl',
        spread_df=spread_df, max_spread=0.02)

    # Index drops the lookback warmup
    assert len(weights) == len(prices) - 60
    assert list(weights.columns) == list(prices.columns)
    # On every date with full data, weights sum to 1.0 (hard top-N) or 0.
    row_sums = weights.sum(axis=1)
    assert ((np.isclose(row_sums, 1.0)) | (row_sums == 0.0)).all()
    # Each held name gets exactly 1/top_n
    nonzero = weights.values[weights.values > 0]
    if len(nonzero):
        assert np.allclose(nonzero, 1.0 / 3)


def test_train_walk_forward_smoke():
    """End-to-end train() with one walk-forward window + 2 Optuna trials.

    Just verifies the wiring runs and produces a TrainResult with at
    least one WindowResult. Doesn't assert on Sharpe values (trial
    count too low for meaningful numbers).
    """
    pytest.importorskip('vectorbt')
    from regime.trainer import TrainResult, train

    prices, spread_df = _synthetic(n_days=2200, n_tickers=8)  # ~8.5y
    result = train(
        prices, spread_df,
        n_trials=2, rebalance_days=20, metric='sharpe',
        commission_bps=10.0, max_spread=0.02,
        train_years=5, val_years=3, step_years=10,  # one window only
    )
    assert isinstance(result, TrainResult)
    assert len(result.windows) >= 1
    w0 = result.windows[0]
    assert w0.train_start < w0.train_end <= w0.val_end
    assert set(w0.best_params).issuperset(
        {'lookback', 'n_tail', 'top_n', 'divergence'})
    # Best-window accessor doesn't crash
    bw = result.best_window
    assert bw is not None
