"""Action-menu shape + invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfr.menu import (
    ActionMenu, CashMode, EqualWeightMode, TopKMode,
    default_phase1_menu, default_phase2a_menu,
)


def _make_panel(n_bars: int = 200, n_tickers: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n_bars, n_tickers))
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2020-01-01', periods=n_bars, freq='B')
    cols = [f'T{i:02d}' for i in range(n_tickers)]
    return pd.DataFrame(prices, index=index, columns=cols)


def test_cash_mode_is_zero():
    p = _make_panel()
    w = CashMode().precompute(p)
    assert w.shape == p.shape
    assert np.allclose(w, 0.0)


def test_ew_sums_to_one_when_liquid():
    p = _make_panel(n_bars=200, n_tickers=8)
    w = EqualWeightMode(min_lookback=21).precompute(p)
    # After warmup the row sum is 1
    np.testing.assert_allclose(w[100].sum(), 1.0)
    np.testing.assert_allclose(w[150].sum(), 1.0)


def test_topk_mode_picks_k_names():
    p = _make_panel(n_bars=200, n_tickers=10)
    mode = TopKMode(name='mom', score_kind='momentum',
                    score_window=21, top_k=3, min_lookback=21)
    w = mode.precompute(p)
    # Post-warmup, each row has at most 3 non-zero entries summing to 1
    row = w[100]
    assert int((row > 0).sum()) == 3
    np.testing.assert_allclose(row.sum(), 1.0)


def test_menu_dedups_cash():
    """gross=0 across all modes should collapse to a single cash action."""
    modes = [EqualWeightMode(), TopKMode(name='mom', score_kind='momentum')]
    menu = ActionMenu(modes=modes, gross_levels=(0.0, 1.0, 2.0))
    cash_actions = [a for a in menu.actions if a.gross == 0.0]
    assert len(cash_actions) == 1


def test_menu_precompute_shape():
    menu = default_phase1_menu(top_k=5)
    p = _make_panel(n_bars=300, n_tickers=15)
    w = menu.precompute(p)
    assert w.shape == (300, menu.n_actions, 15)
    # Cash row is all zeros
    cash_idx = next(i for i, a in enumerate(menu.actions) if a.gross == 0)
    assert np.allclose(w[:, cash_idx, :], 0.0)


def test_menu_action_keys_are_unique():
    menu = default_phase1_menu(top_k=5)
    assert len(set(menu.action_keys)) == menu.n_actions


def test_phase2a_menu_size_and_modes():
    """Phase 2a = Phase 1 (5 modes) + 4 documented-alpha modes (mom121,
    lowv252, shtop, trend); 9 modes × 4 gross levels deduped = 28."""
    menu = default_phase2a_menu(top_k=20)
    assert menu.n_actions == 28
    keys = set(menu.action_keys)
    # All Phase 1 modes present
    assert {'cash', 'ew@g1', 'mom@g1', 'rev@g1', 'lowv@g1', 'highv@g1'}.issubset(keys)
    # All Phase 2a modes present
    assert {'mom121@g1', 'lowv252@g1', 'shtop@g1', 'trend@g1'}.issubset(keys)
    # All gross levels for the new modes
    for new_mode in ('mom121', 'lowv252', 'shtop', 'trend'):
        for g in ('0.5', '1', '2'):
            assert f'{new_mode}@g{g}' in keys


def test_mom_12_1_picks_long_window_winners():
    """12-1 momentum should rank the longest-trending name highest."""
    n_bars = 400
    n_tickers = 6
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.005, size=(n_bars, n_tickers))
    # Inject persistent upward drift on ticker 0 over the [t-252, t-21] window
    drift = np.zeros(n_bars)
    drift[100:380] = 0.005   # large drift over the relevant lookback
    rets[:, 0] += drift
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2020-01-01', periods=n_bars, freq='B')
    p = pd.DataFrame(prices, index=index, columns=[f'T{i}' for i in range(n_tickers)])
    mode = TopKMode(name='m121', score_kind='mom_12_1',
                    score_window=252, top_k=2, min_lookback=252)
    w = mode.precompute(p)
    # At t=395 the trailing 12-1 window covers most of ticker 0's drift
    assert w[395, 0] > 0  # ticker 0 should be in top-K


def test_sharpe_top_picks_high_sharpe_name():
    """sharpe_top should pick names with high mean / low std of log returns."""
    n_bars = 500
    n_tickers = 5
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, size=(n_bars, n_tickers))
    # Ticker 0: high mean low std (high Sharpe). Ticker 1: low mean high std.
    rets[:, 0] = rng.normal(0.002, 0.005, size=n_bars)
    rets[:, 1] = rng.normal(0.000, 0.020, size=n_bars)
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2020-01-01', periods=n_bars, freq='B')
    p = pd.DataFrame(prices, index=index, columns=[f'T{i}' for i in range(n_tickers)])
    mode = TopKMode(name='sh', score_kind='sharpe_top',
                    score_window=252, top_k=1, min_lookback=252)
    w = mode.precompute(p)
    # At a late bar, ticker 0 should be picked (highest Sharpe by far)
    assert int(np.argmax(w[400])) == 0


def test_trend_strength_finite_and_normalizes():
    """trend_str should produce finite scores when there's any drawdown,
    and the resulting weights sum to 1 over the picked names."""
    n_bars = 400
    n_tickers = 6
    rng = np.random.default_rng(2)
    rets = rng.normal(0.001, 0.01, size=(n_bars, n_tickers))
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2020-01-01', periods=n_bars, freq='B')
    p = pd.DataFrame(prices, index=index, columns=[f'T{i}' for i in range(n_tickers)])
    mode = TopKMode(name='ts', score_kind='trend_str',
                    score_window=252, top_k=3, min_lookback=252)
    w = mode.precompute(p)
    # Post-warmup row sum is 1
    np.testing.assert_allclose(w[300].sum(), 1.0, rtol=1e-6)
