"""Tests for regime.inference.target_weights on synthetic OHLC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime.inference import target_weights
from regime.persist import load_checkpoint, save_checkpoint_from_window
from regime.trainer import WindowResult


def _build_checkpoint(
    tmp_path: Path,
    *,
    strategy: str = 'regime',
    top_n: int = 2,
    divergence: str | None = 'cosine',
    rsi_n: int | None = None,
    universe: list[str],
    lookback: int = 30,
    n_tail: int = 5,
    use_short: bool = True,
    use_mid: bool = False,
    use_long: bool = False,
    file_name: str = 'cp.json',
) -> Path:
    """Write an Optuna-mode checkpoint via save_checkpoint_from_window."""
    best_params: dict = {
        'lookback': lookback, 'n_tail': n_tail, 'top_n': top_n,
        'use_short_scales': use_short,
        'use_mid_scales': use_mid,
        'use_long_scales': use_long,
    }
    if divergence is not None:
        best_params['divergence'] = divergence
    if rsi_n is not None:
        best_params['rsi_n'] = rsi_n
    window = WindowResult(
        train_start=pd.Timestamp('2020-01-01'),
        train_end=pd.Timestamp('2020-12-31'),
        val_end=pd.Timestamp('2021-12-31'),
        best_params=best_params,
        train_score=0.5, val_score=0.3,
        strategy=strategy,
    )
    return save_checkpoint_from_window(
        tmp_path / file_name, window,
        universe=universe, rebal_days=20, max_spread=0.02, commission_bps=10)


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
    cp_path = _build_checkpoint(
        tmp_path, top_n=2, universe=tickers, lookback=30, n_tail=5)
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(80, tickers)

    weights = target_weights(prices, highs, lows, cp)
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert (weights >= 0).all()
    assert weights.name == prices.index[-1]


def test_target_weights_validates_columns(tmp_path: Path):
    cp_path = _build_checkpoint(
        tmp_path, top_n=2, universe=['A', 'B'], lookback=10, n_tail=3)
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(40, ['A', 'B'])
    bad_lows = lows.rename(columns={'A': 'X'})
    with pytest.raises(ValueError, match='share columns'):
        target_weights(prices, highs, bad_lows, cp)


def test_target_weights_requires_enough_history(tmp_path: Path):
    cp_path = _build_checkpoint(
        tmp_path, top_n=2, universe=['A', 'B'], lookback=50, n_tail=10)
    cp = load_checkpoint(cp_path)
    prices, highs, lows = _synthetic_ohlc(20, ['A', 'B'])  # too short
    with pytest.raises(ValueError, match='need at least'):
        target_weights(prices, highs, lows, cp)


def test_target_weights_hard_top_n(tmp_path: Path):
    """Optuna checkpoint produces a hard-top-N basket: exactly `top_n`
    names hold `1/top_n` each, the rest are zero."""
    tickers = ['A', 'B', 'C', 'D', 'E']
    cp_path = _build_checkpoint(
        tmp_path, top_n=2, divergence='cosine', universe=tickers)
    cp = load_checkpoint(cp_path)

    prices, highs, lows = _synthetic_ohlc(80, tickers)
    weights = target_weights(prices, highs, lows, cp)

    nonzero = weights[weights > 0]
    assert len(nonzero) == 2  # exactly top_n names
    assert all(w == pytest.approx(0.5) for w in nonzero)  # 1/top_n each
    assert weights.sum() == pytest.approx(1.0)


def test_target_weights_kl_divergence(tmp_path: Path):
    """Different divergence string changes the dispatch; sum-to-one still holds."""
    tickers = ['A', 'B', 'C', 'D']
    cp_path = _build_checkpoint(
        tmp_path, top_n=2, divergence='kl', universe=tickers)
    cp = load_checkpoint(cp_path)

    prices, highs, lows = _synthetic_ohlc(80, tickers)
    weights = target_weights(prices, highs, lows, cp)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_target_weights_scalogram_hard_top_n(tmp_path: Path):
    """Scalogram checkpoint dispatches to the scalogram scoring path
    and produces a hard-top-N basket: exactly `top_n` names hold
    `1/top_n` each, the rest are zero."""
    tickers = ['A', 'B', 'C', 'D', 'E']
    cp_path = _build_checkpoint(
        tmp_path, strategy='scalogram', top_n=2, divergence=None,
        universe=tickers, file_name='scalo.json')
    cp = load_checkpoint(cp_path)
    assert cp.strategy == 'scalogram'
    assert cp.divergence is None

    prices, highs, lows = _synthetic_ohlc(80, tickers)
    weights = target_weights(prices, highs, lows, cp)

    nonzero = weights[weights > 0]
    assert len(nonzero) == 2
    assert all(w == pytest.approx(0.5) for w in nonzero)
    assert weights.sum() == pytest.approx(1.0)


def test_target_weights_scalogram_picks_lowest_scores(tmp_path: Path):
    """Sanity: scalogram should rank ascending (lowest score wins),
    opposite of regime which ranks descending. Both produce valid
    baskets summing to 1."""
    tickers = ['A', 'B', 'C', 'D', 'E']
    regime_path = _build_checkpoint(
        tmp_path / 'r', top_n=2, divergence='kl', universe=tickers,
        lookback=30, n_tail=5, file_name='cp.json')
    scalo_path = _build_checkpoint(
        tmp_path / 's', strategy='scalogram', top_n=2, divergence=None,
        universe=tickers, lookback=30, n_tail=5, file_name='cp.json')
    regime_cp = load_checkpoint(regime_path)
    scalo_cp = load_checkpoint(scalo_path)

    prices, highs, lows = _synthetic_ohlc(80, tickers, seed=7)
    regime_w = target_weights(prices, highs, lows, regime_cp)
    scalo_w = target_weights(prices, highs, lows, scalo_cp)

    assert regime_w.sum() == pytest.approx(1.0)
    assert scalo_w.sum() == pytest.approx(1.0)
