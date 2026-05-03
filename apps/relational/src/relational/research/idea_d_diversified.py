"""Idea D — diversified weights via greedy farthest-first thinning.

Head-to-head:
  - regime:        weights_regime (per-stock CWT divergence; baseline)
  - diversified:   weights_regime_diversified (top_pool by divergence,
                   thin to k_keep by fingerprint farthest-first)

Same Phase-2 universe / dates / commission convention as
`backtest_sector_excess.py`. The question: does forcing scalogram
diversification at the selection layer improve risk-adjusted returns
over picking the raw top-N?

CLI parameters (lookback, n_tail, k_keep, top_pool, divergence,
fp_window) mirror baseline + the new thinning knobs. Defaults are
deliberately mild — `k_keep=10`, `top_pool=15` — so the thinning
overrides the bottom 5 picks in a 15-wide net. Setting `top_pool=10`
collapses to baseline as a sanity check.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import weights_regime as _weights_regime_baseline

from relational.diversify import weights_regime_diversified

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
    k_keep: int = 10,
    top_pool: int = 15,
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
          f'k_keep={k_keep}, top_pool={top_pool}, fp_window={fp_window}, '
          f'divergence={divergence}')

    print('\n[1/2] Computing baseline regime weights (top-N=k_keep)...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=k_keep,
        scales=scales, divergence=divergence)

    print('[2/2] Computing diversified weights (top_pool → thin → k_keep)...')
    w_diverse = weights_regime_diversified(
        prices, lookback=lookback, n_tail=n_tail,
        k_keep=k_keep, top_pool=top_pool,
        scales=scales, divergence=divergence, fp_window=fp_window)

    print('\nRunning bt backtests...')
    bt_baseline = _build_strategy(
        'regime', prices, w_baseline,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_diverse = _build_strategy(
        'diversified', prices, w_diverse,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(bt_baseline, bt_diverse)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(
        f'Idea D — diversified vs baseline regime — Phase-2 '
        f'({start} → {end}, k_keep={k_keep}, top_pool={top_pool}, '
        f'rebal={rebal_days}d, div={divergence}, w={fp_window})')
    fig.tight_layout()
    fig_path = out / 'relational-idea-d-diversified-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-idea-d-diversified-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(str(result.stats))
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--k-keep', type=int, default=10)
    p.add_argument('--top-pool', type=int, default=15)
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
