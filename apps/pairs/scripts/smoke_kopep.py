"""Sanity-check the pipeline on a known cointegrated pair: KO + PEP.

Should land EG p ≪ 0.05 and a positive val Sharpe under classical
z-score trading. If this doesn't work, the pipeline has a bug —
not a model problem.

Run from repo root:
    uv run python apps/pairs/scripts/smoke_kopep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix

from pairs import (
    backtest_pair, compute_spread, engle_granger_test,
    spread_stats, trade_signals,
)
from pairs.spread import zscore


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    print('Loading KO + PEP from Stooq...')
    prices, _, _, _ = load_stooq_matrix(
        './StooqData', min_history=150,
        start_date='2010-01-01', end_date='2025-12-11',
        tickers=['KO', 'PEP'])
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')

    p = prices.dropna()
    log_p = np.log(p.values)
    log_a, log_b = log_p[:, 0], log_p[:, 1]
    dates = p.index

    train_end = pd.Timestamp('2020-12-31')
    train_mask = dates <= train_end
    val_mask = ~train_mask

    print('\nEngle-Granger test on train slice (2010-01 → 2020-12):')
    eg = engle_granger_test(log_a[train_mask], log_b[train_mask])
    print(f'  EG p-value: {eg.p_value:+.4e}')
    print(f'  EG test stat: {eg.test_stat:+.3f}')
    print(f'  hedge β: {eg.hedge_beta:+.4f}')
    print(f'  intercept: {eg.intercept:+.4f}')
    print(f'  n_obs: {eg.n_obs}')

    if eg.p_value > 0.05:
        print(f'\n!! KO+PEP did not pass cointegration p<0.05 — '
              f'pipeline may have a bug, or 2010-2020 train is not '
              f'a clean cointegration regime for this pair.')
        return

    spread_train = compute_spread(
        log_a[train_mask], log_b[train_mask],
        eg.hedge_beta, eg.intercept)
    stats = spread_stats(spread_train)
    print(f'\nSpread stats (train):')
    print(f'  mean: {stats.mean:+.4f}')
    print(f'  std:  {stats.std:+.4f}')
    print(f'  half-life: {stats.half_life:.1f} bars')

    print('\nVal-window backtest (2021-01 → 2025-12) under classical '
          'z=±2 entry, ±0.5 exit, 10bps × 2 commission per leg-flip:')
    val_dates = dates[val_mask]
    bt = backtest_pair(
        log_p_a_train=log_a[train_mask], log_p_b_train=log_b[train_mask],
        log_p_a_val=log_a[val_mask],     log_p_b_val=log_b[val_mask],
        val_dates=val_dates,
        a_name='KO', b_name='PEP',
        hedge_beta=eg.hedge_beta, intercept=eg.intercept)
    print(f'  Sharpe:       {bt.sharpe:+.3f}')
    print(f'  Sortino:      {bt.sortino:+.3f}')
    print(f'  CAGR:         {bt.cagr_pct:+.2f}%')
    print(f'  MaxDD:        {bt.max_drawdown_pct:+.2f}%')
    print(f'  n trades:     {bt.n_trades}')
    print(f'  avg holding:  {bt.avg_holding_bars:.1f} bars')
    print(f'  pct in trade: {bt.pct_in_trade:.3f}')

    if bt.sharpe > 0:
        print('\n  pipeline OK — pair has positive val Sharpe.')
    else:
        print('\n  pair val Sharpe ≤ 0 — known cointegrated pair lost. '
              'Could be regime shift on KO/PEP post-2020; pipeline '
              'still validated by EG p<0.05. Run walk-forward for the '
              'real test.')


if __name__ == '__main__':
    main()
