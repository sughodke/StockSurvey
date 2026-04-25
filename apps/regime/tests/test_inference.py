"""Tests for regime.inference.target_weights on synthetic OHLC."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from regime.inference import target_weights
from regime.persist import load_checkpoint, save_checkpoint
from regime.trainer import TrainResult


def _build_checkpoint(tmp_path: Path, *, lookback: int = 30, n_tail: int = 5,
                       universe: list[str]) -> Path:
    n_scales = 3
    result = TrainResult(
        params={
            'scale_log_weights': jnp.zeros(n_scales, dtype=jnp.float32),
            'log_temperature': jnp.asarray(np.log(0.5), dtype=jnp.float32),
        },
        train_history=[0.0],
        val_history=[(0, 0.0)],
        train_sharpe=0.0,
        val_sharpe=0.0,
        scales=[5, 12, 21],
        train_dates=(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31')),
        val_dates=(pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31')),
    )
    return save_checkpoint(
        tmp_path / 'cp.json', result,
        universe=universe, lookback=lookback, n_tail=n_tail,
        rebal_days=20, max_spread=0.02, commission_bps=10)


def _synthetic_ohlc(n_days: int, tickers: list[str], seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    n = len(tickers)
    closes = np.cumsum(rng.standard_normal((n_days, n)) * 0.5, axis=0) + 100
    highs = closes + np.abs(rng.standard_normal((n_days, n))) * 0.5
    lows = closes - np.abs(rng.standard_normal((n_days, n))) * 0.5
    return (
        pd.DataFrame(closes, index=dates, columns=tickers),
        pd.DataFrame(highs, index=dates, columns=tickers),
        pd.DataFrame(lows, index=dates, columns=tickers),
    )


def test_target_weights_sums_to_one(tmp_path: Path):
    tickers = ['A', 'B', 'C', 'D']
    cp_path = _build_checkpoint(tmp_path, lookback=30, n_tail=5, universe=tickers)
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(80, tickers)

    weights = target_weights(prices, highs, lows, cp)
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert (weights >= 0).all()
    assert weights.name == prices.index[-1]


def test_target_weights_validates_columns(tmp_path: Path):
    cp_path = _build_checkpoint(tmp_path, lookback=10, n_tail=3, universe=['A', 'B'])
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(40, ['A', 'B'])
    bad_lows = lows.rename(columns={'A': 'X'})
    with pytest.raises(ValueError, match='share columns'):
        target_weights(prices, highs, bad_lows, cp)


def test_target_weights_requires_enough_history(tmp_path: Path):
    cp_path = _build_checkpoint(tmp_path, lookback=50, n_tail=10, universe=['A', 'B'])
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(20, ['A', 'B'])  # too short
    with pytest.raises(ValueError, match='need at least'):
        target_weights(prices, highs, lows, cp)
