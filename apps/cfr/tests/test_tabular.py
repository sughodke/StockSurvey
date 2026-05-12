"""Tabular CFR convergence on a stationary two-action toy.

Construct a synthetic dataset where one fixed action is always
better than the other (price drift makes momentum the right call).
Trained tabular CFR should converge to picking the dominant
action almost surely; this validates the regret-matching update +
average-policy aggregation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfr.menu import ActionMenu, EqualWeightMode, CashMode
from cfr.regret import compute_block_regrets
from cfr.tabular import TabularCFR


def _trending_panel(n_bars: int = 600, n_tickers: int = 5, drift: float = 0.001,
                     seed: int = 0) -> pd.DataFrame:
    """Steady upward drift — EW should beat cash."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.005, size=(n_bars, n_tickers))
    prices = np.cumprod(1 + rets, axis=0) * 100
    index = pd.date_range('2010-01-01', periods=n_bars, freq='B')
    cols = [f'T{i}' for i in range(n_tickers)]
    return pd.DataFrame(prices, index=index, columns=cols)


def test_tabular_cfr_prefers_winning_action():
    """EW vs cash on trending data — CFR should converge to picking EW."""
    p = _trending_panel(n_bars=800, drift=0.002)
    menu = ActionMenu(
        modes=[EqualWeightMode(min_lookback=10), CashMode(name='cash_mode')],
        gross_levels=(0.0, 1.0),
    )
    action_weights = menu.precompute(p)
    # Find the EW@g1 action index
    ew_idx = next(i for i, a in enumerate(menu.actions)
                  if a.mode_name == 'ew' and a.gross == 1.0)

    table = TabularCFR(n_infosets=1, n_actions=menu.n_actions)
    rng = np.random.default_rng(0)
    rebal_days = 20
    log_p = np.log(p.values)
    for t in range(50, len(p) - rebal_days, rebal_days):
        pi = table.current_policy(0)
        block = log_p[t + rebal_days] - log_p[t]
        played = int(rng.choice(menu.n_actions, p=pi))
        regrets = compute_block_regrets(block, action_weights[t], played)
        table.update(0, regrets, pi)

    avg_pi = table.average_policy(0)
    # The EW@g1 action should have the highest average-policy weight.
    assert int(np.argmax(avg_pi)) == ew_idx, (
        f'expected EW@g1 to win, got action {np.argmax(avg_pi)} '
        f'with avg policy {avg_pi}')
    # And its mass should be meaningful — not 1/N uniform.
    assert avg_pi[ew_idx] > 0.5


def test_tabular_cfr_table_shape():
    table = TabularCFR(n_infosets=9, n_actions=11)
    assert table.cumulative_regret.shape == (9, 11)
    assert table.cumulative_strategy.shape == (9, 11)
    assert table.n_visits.shape == (9,)
    pi = table.current_policy(0)
    np.testing.assert_allclose(pi, np.full(11, 1/11))


def test_tabular_avg_policy_falls_back_to_current():
    """Before any update, average policy = current policy = uniform."""
    table = TabularCFR(n_infosets=4, n_actions=5)
    for i in range(4):
        avg = table.average_policy(i)
        cur = table.current_policy(i)
        np.testing.assert_allclose(avg, cur)
        np.testing.assert_allclose(avg, np.full(5, 0.2))


def test_tabular_handles_invalid_infoset():
    """Negative or out-of-bound infoset → uniform fallback, no exception."""
    table = TabularCFR(n_infosets=4, n_actions=3)
    for bad in (-1, 4, 99):
        pi = table.current_policy(bad)
        assert pi.shape == (3,)
        np.testing.assert_allclose(pi, np.full(3, 1/3))
