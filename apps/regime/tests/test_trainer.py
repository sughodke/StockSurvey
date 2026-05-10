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


def test_weights_regime_smoke():
    """`weights_regime` produces a DataFrame with the right shape and a
    valid hard-top-N row sum at every non-warmup date."""
    pytest.importorskip('vectorbt')
    from regime.trainer import weights_regime

    prices, _ = _synthetic(n_days=300, n_tickers=8)
    weights = weights_regime(
        prices, lookback=60, n_tail=10, top_n=3,
        scales=[5, 21, 90], divergence='kl')

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


def test_weights_scalogram_smoke():
    """`weights_scalogram` produces the same shape contract as
    `weights_regime`. Scalogram has no `divergence` knob — its score
    is fixed (direction − momentum × coherence)."""
    pytest.importorskip('vectorbt')
    from regime.trainer import weights_scalogram

    prices, _ = _synthetic(n_days=300, n_tickers=8)
    weights = weights_scalogram(
        prices, lookback=60, n_tail=10, top_n=3, scales=[5, 21, 90])

    assert len(weights) == len(prices) - 60
    assert list(weights.columns) == list(prices.columns)
    row_sums = weights.sum(axis=1)
    assert ((np.isclose(row_sums, 1.0)) | (row_sums == 0.0)).all()
    nonzero = weights.values[weights.values > 0]
    if len(nonzero):
        assert np.allclose(nonzero, 1.0 / 3)


def test_train_scalogram_walk_forward_smoke():
    """End-to-end train() with strategy='scalogram'. Verifies the
    Optuna objective drops the `divergence` knob for scalogram and
    that WindowResult records the strategy name."""
    pytest.importorskip('vectorbt')
    from regime.trainer import TrainResult, train

    prices, spread_df = _synthetic(n_days=2200, n_tickers=8)
    result = train(
        prices, spread_df, strategy='scalogram',
        n_trials=2, rebalance_days=20, metric='sharpe',
        commission_bps=10.0,
        train_years=5, val_years=3, step_years=10,
    )
    assert isinstance(result, TrainResult)
    assert result.strategy == 'scalogram'
    assert result.windows
    w0 = result.windows[0]
    assert w0.strategy == 'scalogram'
    # Scalogram doesn't search over divergence
    assert 'divergence' not in w0.best_params
    assert set(w0.best_params).issuperset(
        {'lookback', 'n_tail', 'top_n', 'use_short_scales'})


def test_train_rejects_unknown_strategy():
    pytest.importorskip('vectorbt')
    from regime.trainer import train

    prices, _ = _synthetic(n_days=2200, n_tickers=8)
    with pytest.raises(ValueError, match='unknown strategy'):
        train(prices, strategy='nonsense', n_trials=2)


def test_filter_window_universe_drops_leading_nan_and_short_history():
    """Per-window survivorship filter must:
      1. Drop tickers with NaN at the window's first bar (else CWT
         cumsum corrupts every subsequent value for that column)
      2. Drop tickers with fewer than `min_bars` valid observations
         in the window (insufficient history to compute scores)
    """
    from regime.trainer import _filter_window_universe

    dates = pd.bdate_range('2020-01-01', periods=600)
    panel = pd.DataFrame({
        'FULL': 100.0,                 # valid throughout
        'LATE': 100.0,                 # IPO'd at index 100 — leading NaN
        'SHORT': 100.0,                # valid for first 50 bars only
    }, index=dates)
    panel.loc[dates[:100], 'LATE'] = np.nan
    panel.loc[dates[50:], 'SHORT'] = np.nan

    keep = _filter_window_universe(panel, min_bars=200)
    # FULL: passes both checks
    # LATE: fails check 1 (NaN at first bar)
    # SHORT: fails check 2 (only 50 valid bars)
    assert list(keep) == ['FULL']


def test_train_per_window_filtering_keeps_pit_universe():
    """End-to-end: a panel with a late-IPO ticker should successfully
    train, with the late-IPO ticker filtered out of windows whose
    start predates its first bar — but kept in later windows. Verifies
    the survivorship-bias fix doesn't break the trainer when the
    panel has leading NaNs."""
    pytest.importorskip('vectorbt')
    from regime.trainer import train

    prices, spread_df = _synthetic(n_days=2200, n_tickers=8)
    # Make ticker 'T7' an IPO at bar 1500 — leading NaN in the panel.
    prices = prices.copy()
    prices.loc[prices.index[:1500], 'T7'] = np.nan
    spread_df = spread_df.copy()
    spread_df.loc[spread_df.index[:1500], 'T7'] = np.nan

    result = train(
        prices, spread_df, strategy='regime',
        n_trials=2, rebalance_days=20, metric='sharpe',
        commission_bps=10.0, per_window_min_history=252,
        train_years=5, val_years=3, step_years=10,  # one window only
    )
    # The point of this test: the leading-NaN ticker would have
    # propagated through the CWT cumsum and corrupted every score
    # before this fix. Now the per-window filter excludes T7 from the
    # train slice (whose first bar is NaN for T7) so the run completes
    # without raising. We don't assert finite Sharpes — Optuna may not
    # find a good config in 2 trials on a 7-ticker synthetic universe;
    # that's not what this test is checking.
    assert result.windows


def test_train_no_boundary_bar_overlap():
    """The bar at `train_end` must not appear in the train slice — pandas
    `.loc` is end-inclusive on both sides, so without an explicit trim
    the boundary bar would land in both train and val. Regression test
    for review-followups #4."""
    pytest.importorskip('vectorbt')
    from regime import trainer

    prices, spread_df = _synthetic(n_days=2200, n_tickers=8)
    captured: list[pd.DataFrame] = []
    real_make_objective = trainer._make_objective

    def spy(prices_train, **kw):
        captured.append(prices_train)
        return real_make_objective(prices_train, **kw)

    trainer._make_objective = spy
    try:
        result = trainer.train(
            prices, spread_df,
            n_trials=2, rebalance_days=20, metric='sharpe',
            commission_bps=10.0,
            train_years=5, val_years=3, step_years=10,
        )
    finally:
        trainer._make_objective = real_make_objective

    assert captured, 'objective never called'
    train_slice = captured[0]
    train_end = result.windows[0].train_end
    assert train_end not in train_slice.index
    assert train_slice.index[-1] < train_end


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
        commission_bps=10.0,
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
