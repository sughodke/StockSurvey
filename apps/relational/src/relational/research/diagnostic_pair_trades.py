"""Pair / spread overlays atop the scoreboard winners — drawdown test.

Same three scorers as `diagnostic_sizing_overlays` (baseline, empirical,
farthest), but instead of within-basket sizing variants we layer a
short hedge to neutralize market beta. Three constructions per scorer:

  * `long-only`       — top-N at 1/N (the existing scoreboard baseline)
  * `mkt-neutral`     — long top-N, short universe equal-weight
                        (gross = 2, net = 0)
  * `rank-spread`     — long top-N, short bot-N (gross = 2, net = 0;
                        the scorer is used on both tails)

Headline metrics emphasized: max DD, Calmar, daily Sharpe. The
hypothesis the user proposed: pair trades should reduce max DD by
hedging out market beta. Trade-off: lower CAGR, possibly higher
or lower Sharpe depending on whether scorer alpha clears 2× the
commission cost on a doubled-gross portfolio.

Bot-N picks for `rank-spread` are constructed by re-running each
scorer's score function and selecting `select_top_n_matrix` with
`ascending=True` (lowest scores → bot-N).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import (
    apply_nan_mask, select_top_n_matrix, weights_regime as _baseline_long,
)
from ss_portfolio.bt_helpers import build_strategy

from relational.analog_knn import (
    analog_knn_scores, weights_regime_analog,
)
from relational.empirical_sectors import (
    empirical_excess_divergence_scores, weights_excess_regime_empirical,
)
from relational.farthest import centroid_distance_scores, weights_regime_farthest
from relational.pairs import (
    cluster_pair_weights, market_neutral_weights, rank_spread_weights,
)
from relational.scoring import baseline_divergence_scores
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


def _bot_weights_from_scores(
    scores: np.ndarray, prices: pd.DataFrame, top_n: int, lookback: int,
) -> pd.DataFrame:
    """Build a bot-N (lowest-score) weight DataFrame matching the shape
    contract of the existing weights builders."""
    masked = apply_nan_mask(scores, prices.values, lookback)
    arr = select_top_n_matrix(masked, top_n, ascending=True)
    return pd.DataFrame(
        arr,
        index=prices.index[lookback:],
        columns=prices.columns,
    )


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, rebal_days={rebal_days}')

    print('\n[scoring] computing top-N and bot-N weights for each scorer...')

    # baseline (per-stock CWT-power KL divergence)
    base_scores = baseline_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales)
    base_top = _baseline_long(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, scales=scales)
    base_bot = _bot_weights_from_scores(
        base_scores, prices, top_n=top_n, lookback=lookback)

    # empirical (idea A) — also pull cluster_ids for the cluster-aware pair.
    emp_scores, emp_cluster_ids = empirical_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        fp_window=fp_window, k_clusters=k_clusters, return_clusters=True)
    emp_top = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, k_clusters=k_clusters, fp_window=fp_window)
    emp_bot = _bot_weights_from_scores(
        emp_scores, prices, top_n=top_n, lookback=lookback)
    emp_cluster_pair = cluster_pair_weights(
        emp_scores, emp_cluster_ids, prices, lookback=lookback)

    # farthest (idea C)
    far_scores = centroid_distance_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window)
    far_top = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window)
    far_bot = _bot_weights_from_scores(
        far_scores, prices, top_n=top_n, lookback=lookback)

    # analog (idea B) — explicit forward-return forecast; informative on
    # both tails by construction.
    print('  computing analog k-NN scores (slow: ~17 min on 13yr Phase-2)...')
    ana_scores = analog_knn_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window)
    ana_top = weights_regime_analog(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window)
    ana_bot = _bot_weights_from_scores(
        ana_scores, prices, top_n=top_n, lookback=lookback)

    base_strategies = {
        'baseline': (base_top, base_bot),
        'empirical': (emp_top, emp_bot),
        'farthest': (far_top, far_bot),
        'analog': (ana_top, ana_bot),
    }

    print('\n[pairs] constructing market-neutral and rank-spread variants...')
    backtests: list[bt.Backtest] = []
    for strat_name, (top, bot) in base_strategies.items():
        long_only = top
        mkt_neutral = market_neutral_weights(top, prices=prices)
        rank_spread = rank_spread_weights(top, bot)
        for variant_name, w in [
            ('long-only', long_only),
            ('mkt-neutral', mkt_neutral),
            ('rank-spread', rank_spread),
        ]:
            label = f'{strat_name}|{variant_name}'
            row_sums = w.iloc[::rebal_days].sum(axis=1)
            print(f'  {label}: '
                  f'mean_net={row_sums.mean():+.3f} '
                  f'gross={w.iloc[::rebal_days].abs().sum(axis=1).mean():.3f}')
            backtests.append(build_strategy(
                label, prices, w,
                rebal_days=rebal_days, commission_bps=commission_bps))

    # Special: cluster-aware pair on empirical (idea A) — long winner per
    # cluster, short cluster aggregate. Hedges intra-cluster, not universe.
    label = 'empirical|cluster-pair'
    row_sums = emp_cluster_pair.iloc[::rebal_days].sum(axis=1)
    print(f'  {label}: '
          f'mean_net={row_sums.mean():+.3f} '
          f'gross={emp_cluster_pair.iloc[::rebal_days].abs().sum(axis=1).mean():.3f}')
    backtests.append(build_strategy(
        label, prices, emp_cluster_pair,
        rebal_days=rebal_days, commission_bps=commission_bps))

    print(f'\nRunning bt backtests ({len(backtests)} strategies)...')
    result = bt.run(*backtests)
    result.display()

    stats = result.stats
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return', 'worst_year']
    leaderboard = stats.loc[headline, order].T

    print('\n' + '=' * 100)
    print('Pair-trade leaderboard — sorted by daily Sharpe')
    print('=' * 100)
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(leaderboard.to_string())

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    result.plot(ax=ax)
    ax.set_title(
        f'Pair-trade overlays — Phase-2 ({start} → {end}, top-{top_n}, '
        f'rebal={rebal_days}d, commission={commission_bps}bps)')
    fig.tight_layout()
    fig_path = out / 'relational-pair-trades-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-pair-trades-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Pair-trade leaderboard — sorted by daily Sharpe\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(str(result.stats) + '\n')
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
