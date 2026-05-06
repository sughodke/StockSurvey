"""Idea A — empirical sectors via k-means on scalogram fingerprints.

Three-way head-to-head:
  - regime:           weights_regime (per-stock CWT divergence; baseline)
  - excess-gics:      weights_excess_regime (GICS sector aggregates)
  - excess-empirical: weights_excess_regime_empirical (k-means clusters
                      of scalogram fingerprints)

Same Phase-2 universe / dates / commission convention as
`backtest_sector_excess.py`. The question: do empirical clusters from
scalogram shape produce a more relevant aggregate than GICS labels,
recovering or beating the baseline?

Default `k_clusters=11` mirrors the GICS sector count; for the 21-name
Phase-2 universe try `--k-clusters 4` or `--k-clusters 8` for a less
sparse partition (k=11 averages 2 tickers per cluster).
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

from relational.empirical_sectors import weights_excess_regime_empirical
from relational.scoring import weights_excess_regime
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


def run(
    *, data_dir: str,
    top_n: int = 10,
    k_clusters: int = 11,
    lookback: int = 120,
    n_tail: int = 20,
    divergence: str = 'kl',
    fp_window: int = 21,
    refit_days: int = 252,
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    """Programmatic entrypoint."""
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'({list(prices.columns)})')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, divergence={divergence}, '
          f'k_clusters={k_clusters}, fp_window={fp_window}, '
          f'refit_days={refit_days}')

    print('\n[1/3] Computing baseline regime weights...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence)

    print('[2/3] Computing GICS-excess regime weights...')
    w_excess_gics = weights_excess_regime(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence, sector_mode='equal')

    print('[3/3] Computing empirical-excess regime weights '
          f'(k_clusters={k_clusters})...')
    w_excess_emp = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence,
        k_clusters=k_clusters, fp_window=fp_window,
        refit_days=refit_days)

    print('\nRunning bt backtests...')
    bt_baseline = build_strategy(
        'regime', prices, w_baseline,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_excess_gics = build_strategy(
        'excess-gics', prices, w_excess_gics,
        rebal_days=rebal_days, commission_bps=commission_bps)
    bt_excess_emp = build_strategy(
        'excess-empirical', prices, w_excess_emp,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(bt_baseline, bt_excess_gics, bt_excess_emp)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(
        f'Idea A — empirical (k={k_clusters}) vs GICS vs baseline — '
        f'Phase-2 ({start} → {end}, top-{top_n}, rebal={rebal_days}d, '
        f'div={divergence}, w={fp_window}, refit={refit_days}d)')
    fig.tight_layout()
    fig_path = out / 'relational-idea-a-empirical-sectors-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-idea-a-empirical-sectors-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(str(result.stats))
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--divergence', default='kl')
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--refit-days', type=int, default=252)
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
