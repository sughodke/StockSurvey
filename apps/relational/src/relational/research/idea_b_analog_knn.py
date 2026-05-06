"""Idea B — k-NN analog forecasting on scalogram fingerprints.

Head-to-head:
  - regime:  weights_regime (per-stock CWT divergence; baseline)
  - analog:  weights_regime_analog (k-NN over historical fingerprints,
             score = mean realized forward return of the K matches)

Same Phase-2 universe / dates / commission convention as the other
relational head-to-heads. The diagnostic question: does non-parametric
historical-analog forecasting outperform the divergence ranking?

A priori risk: of all four relational ideas, this is the one most
prone to lookahead-bug false positives — anything beating baseline
by a surprising margin should trigger a recheck of the causality
guard in `relational.analog_knn._knn_pick` (`s + horizon < t`).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import weights_regime as _weights_regime_baseline
from ss_portfolio.bt_helpers import build_strategy

from relational.analog_knn import weights_regime_analog
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    divergence: str = 'kl',
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'({list(prices.columns)})')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, fp_window={fp_window}, divergence={divergence}')
    print(f'  k_neighbors={k_neighbors}, forward_horizon={forward_horizon}, '
          f'min_sep_days={min_sep_days}, pool_mode={pool_mode}')

    print('\n[1/2] Computing baseline regime weights...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence)

    print('[2/2] Computing analog k-NN weights...')
    w_analog = weights_regime_analog(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window,
        k_neighbors=k_neighbors, forward_horizon=forward_horizon,
        min_sep_days=min_sep_days, pool_mode=pool_mode)

    print('\nRunning bt backtests...')
    bt_baseline = build_strategy(
        'regime', prices, w_baseline,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_analog = build_strategy(
        'analog', prices, w_analog,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(bt_baseline, bt_analog)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(
        f'Idea B — analog k-NN vs baseline regime — Phase-2 '
        f'({start} → {end}, top-{top_n}, rebal={rebal_days}d, '
        f'k={k_neighbors}, h={forward_horizon}, w={fp_window}, '
        f'pool={pool_mode})')
    fig.tight_layout()
    fig_path = out / 'relational-idea-b-analog-knn-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-idea-b-analog-knn-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(str(result.stats))
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--divergence', default='kl')
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-neighbors', type=int, default=50)
    p.add_argument('--forward-horizon', type=int, default=20)
    p.add_argument('--min-sep-days', type=int, default=21)
    p.add_argument('--pool-mode', default='cross_ticker',
                   choices=['cross_ticker', 'per_ticker'])
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
