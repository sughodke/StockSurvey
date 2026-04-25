"""End-to-end trainer test on a tiny synthetic universe.

Two Adam steps over a 6-ticker / 100-day universe — fast enough for CI
but exercises the full CWT -> windowing -> JAX trace -> Sharpe path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.trainer import TrainResult, train


def _synthetic_universe(n_days: int = 120, n_tickers: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    closes = np.cumsum(rng.standard_normal((n_days, n_tickers)) * 0.5, axis=0) + 100
    prices = pd.DataFrame(closes, index=dates,
                          columns=[f'T{i}' for i in range(n_tickers)])
    spreads = pd.DataFrame(0.005, index=dates, columns=prices.columns)  # all liquid
    return prices, spreads


def test_train_runs_end_to_end():
    prices, spread_df = _synthetic_universe(n_days=120, n_tickers=6)
    result = train(
        prices, spread_df,
        scales=[5, 12, 21],
        lookback=30, n_tail=5, rebal_days=10,
        commission_bps=10, max_spread=0.02,
        n_steps=2, learning_rate=0.05, train_frac=0.7,
    )
    assert isinstance(result, TrainResult)
    assert len(result.train_history) == 2
    assert result.params['scale_log_weights'].shape == (3,)
    assert np.isfinite(result.train_sharpe)
    assert np.isfinite(result.val_sharpe)
    assert result.train_dates[0] < result.train_dates[1] <= result.val_dates[1]
