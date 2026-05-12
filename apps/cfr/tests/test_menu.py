"""Action-menu shape + invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfr.menu import (
    ActionMenu, CashMode, EqualWeightMode, TopKMode, default_phase1_menu,
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
