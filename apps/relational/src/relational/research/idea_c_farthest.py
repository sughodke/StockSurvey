"""Idea C — farthest-from-centroid scoring on scalogram fingerprints.

Head-to-head:
  - regime:    weights_regime (per-stock CWT divergence; baseline)
  - farthest:  weights_regime_farthest (cross-sectional centroid distance
               on per-date fingerprints)

Same Phase-2 universe / dates / commission convention as the other
`relational` head-to-heads. The diagnostic question: does
*cross-sectional* outlier scoring beat *temporal* divergence scoring
on a tightly-correlated mega-cap basket?

A priori the prediction is mild: with 21 names mostly riding the
same factor, the centroid is a decent proxy for "the market" and
distance from it is an idiosyncratic-momentum signal that should
roughly track raw CWT divergence. The interesting cases would be
crisis windows where everyone moves together (low cross-sectional
spread → low farthest scores → cash-light selection).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import weights_regime as _weights_regime_baseline

from relational.farthest import weights_regime_farthest

warnings.filterwarnings('ignore')


PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


def _make_commission_fn(bps: float):
    frac = bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def _build_strategy(name: str, prices: pd.DataFrame,
                    weight_df: pd.DataFrame, *,
                    rebal_days: int, commission_bps: float):
    rebal_weights = weight_df.iloc[::rebal_days]
    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    return bt.Backtest(strategy, prices,
                       commissions=_make_commission_fn(commission_bps),
                       integer_positions=False)


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    divergence: str = 'kl',
    fp_window: int = 21,
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

    print('\n[1/2] Computing baseline regime weights...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence)

    print('[2/2] Computing farthest-from-centroid weights...')
    w_farthest = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window)

    print('\nRunning bt backtests...')
    bt_baseline = _build_strategy(
        'regime', prices, w_baseline,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_farthest = _build_strategy(
        'farthest', prices, w_farthest,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(bt_baseline, bt_farthest)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(
        f'Idea C — farthest-from-centroid vs baseline regime — Phase-2 '
        f'({start} → {end}, top-{top_n}, rebal={rebal_days}d, '
        f'w={fp_window})')
    fig.tight_layout()
    fig_path = out / 'relational-idea-c-farthest-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-idea-c-farthest-stats.txt'
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
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
