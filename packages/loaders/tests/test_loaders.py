"""Tests for ss_loaders: CSV matrix loader + symbol-list constants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import (
    MY_FAVES,
    NDX_CONSTITUENTS,
    load_price_matrix,
)


def _write_synthetic_csvs(tmp: Path, n_tickers: int = 5, n_days: int = 600) -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    for i in range(n_tickers):
        prices = np.cumsum(rng.standard_normal(n_days)) + 100
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': prices + rng.uniform(0.1, 0.5, n_days),
            'low': prices - rng.uniform(0.1, 0.5, n_days),
            'close': prices + rng.standard_normal(n_days) * 0.1,
        })
        df.to_csv(tmp / f'TICK{i}.csv', index=False)


def test_load_price_matrix_synthetic(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=5, n_days=600)
    prices, highs, lows = load_price_matrix(str(tmp_path), min_history=504)
    assert prices.shape == (600, 5)
    assert prices.shape == highs.shape == lows.shape
    assert list(prices.columns) == sorted(['TICK0', 'TICK1', 'TICK2', 'TICK3', 'TICK4'])
    # Index alignment
    assert prices.index.equals(highs.index)
    assert prices.index.equals(lows.index)
    # No NaN after ffill+dropna
    assert not prices.isna().any().any()


def test_load_price_matrix_drops_short_tickers(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=4, n_days=600)
    # Add one short ticker that should be dropped
    short = pd.DataFrame({
        'date': pd.bdate_range('2020-01-01', periods=100),
        'open': np.arange(100), 'high': np.arange(100), 'low': np.arange(100),
        'close': np.arange(100),
    })
    short.to_csv(tmp_path / 'SHORT.csv', index=False)
    prices, _, _ = load_price_matrix(str(tmp_path), min_history=504)
    assert 'SHORT' not in prices.columns
    assert len(prices.columns) == 4


def test_load_price_matrix_date_filter(tmp_path):
    _write_synthetic_csvs(tmp_path, n_tickers=3, n_days=600)
    prices, _, _ = load_price_matrix(
        str(tmp_path), min_history=504,
        start_date='2021-01-01', end_date='2021-12-31')
    assert prices.index.min() >= pd.Timestamp('2021-01-01')
    assert prices.index.max() <= pd.Timestamp('2021-12-31')


def test_symbol_constants_present():
    assert isinstance(NDX_CONSTITUENTS, list)
    assert 'AAPL' in NDX_CONSTITUENTS
    assert 'MSFT' in NDX_CONSTITUENTS
    assert isinstance(MY_FAVES, list)
    assert len(MY_FAVES) > 0
