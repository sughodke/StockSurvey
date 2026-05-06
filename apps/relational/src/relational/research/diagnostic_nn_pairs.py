"""Diagnostic — nearest-neighbor pair construction vs the prior pair
overlays.

Runs a head-to-head bt backtest over the curated 312-ticker
`apps/notebook/data/stooq_us_long` panel, comparing three variants for
each of two scorers:

    Scorer       Variants
    -------      ----------------------------------------------
    empirical    long-only, mkt-neutral, nn-pair
    farthest     long-only, mkt-neutral, nn-pair

`long-only` and `mkt-neutral` are the prior baselines. `nn-pair` is the
new per-pick nearest-neighbor construction in
`relational.nn_pairs.nearest_neighbor_pair_weights`.

Sanity checks reported alongside the leaderboard:
  * **Distinctness** — fraction of (date, pick) pairs where the chosen
    NN partner equals the pick itself. Should be 0% by construction.
  * **Concentration** — fraction of dates where all N picks share a
    single NN partner. With 312 names this should be rare; high values
    signal a degenerate hedge.
  * **Distance distribution** — mean and 5/50/95-percentiles of squared
    L2 between long and partner fingerprints across all (date, pick).
  * **Pair stability** — fraction of (long, short) pairs that persist
    between consecutive rebalances.

Outputs:
  * `Output/relational-nn-pairs-stats.txt` — leaderboard + diagnostics.
  * `Output/relational-nn-pairs-equity.png` — equity curves.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio.bt_helpers import build_strategy

from relational.empirical_sectors import weights_excess_regime_empirical
from relational.farthest import weights_regime_farthest
from relational.nn_pairs import (
    fingerprints_for_weights, nearest_neighbor_pair_weights,
    nearest_non_top_partner,
)
from relational.pairs import market_neutral_weights

warnings.filterwarnings('ignore')


_DEFAULT_DATA_DIR = (Path(__file__).resolve().parents[4]
                     / 'notebook' / 'data' / 'stooq_us_long')
_DEFAULT_CACHE_DIR = '/Users/sidghodke/Code/StockSurvey/.scalogram-cache'


def _diagnostics(
    nn: np.ndarray,
    top_weights: pd.DataFrame,
    fps_eval: np.ndarray,
    rebal_days: int,
) -> dict:
    """Sanity-check numbers for the nearest-neighbor pair construction.

    See module docstring for definitions.
    """
    top_arr = top_weights.fillna(0.0).values
    n_eval, n_tickers = top_arr.shape

    # Walk only the rebalance dates — those are when the construction
    # is actually active. Other rows are duplicated forward by
    # `WeighTarget`.
    rebal_idx = np.arange(0, n_eval, rebal_days)

    distinctness_violations = 0
    total_picks = 0
    all_share_count = 0
    rebal_count = 0
    all_dists: list[float] = []
    pair_sets: list[set[tuple[int, int]]] = []

    for t in rebal_idx:
        in_top = top_arr[t] > 0.0
        long_idx = np.where(in_top)[0]
        if long_idx.size == 0:
            continue
        rebal_count += 1
        partners: list[int] = []
        valid_pairs: set[tuple[int, int]] = set()
        for i in long_idx:
            total_picks += 1
            j = int(nn[t, i])
            if j < 0:
                continue
            if j == i:
                distinctness_violations += 1
            partners.append(j)
            valid_pairs.add((int(i), j))
            # Squared-L2 between long and partner fingerprints.
            diff = fps_eval[t, i] - fps_eval[t, j]
            d = float((diff * diff).sum())
            all_dists.append(d)
        if partners and len(set(partners)) == 1:
            all_share_count += 1
        pair_sets.append(valid_pairs)

    # Pair stability across consecutive rebalances.
    persisted = 0
    compared = 0
    for a, b in zip(pair_sets[:-1], pair_sets[1:]):
        if not a or not b:
            continue
        persisted += len(a & b)
        compared += len(a)
    pair_stability = persisted / compared if compared > 0 else float('nan')

    dists_arr = np.asarray(all_dists, dtype=np.float64)
    if dists_arr.size > 0:
        d_pcts = np.percentile(dists_arr, [5, 50, 95])
        d_mean = float(dists_arr.mean())
    else:
        d_pcts = (float('nan'),) * 3
        d_mean = float('nan')

    return {
        'distinctness_violations': distinctness_violations,
        'total_picks_at_rebal': total_picks,
        'distinctness_frac': (
            distinctness_violations / total_picks if total_picks else float('nan')),
        'all_share_count': all_share_count,
        'rebal_count': rebal_count,
        'all_share_frac': (
            all_share_count / rebal_count if rebal_count else float('nan')),
        'distance_mean': d_mean,
        'distance_p05': float(d_pcts[0]),
        'distance_p50': float(d_pcts[1]),
        'distance_p95': float(d_pcts[2]),
        'pair_stability': pair_stability,
        'n_distance_samples': int(dists_arr.size),
    }


def run(
    *, data_dir: str,
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    # Long-only universe — no `tickers=` filter; the manifest defines
    # the 312-name universe by what's on disk under `daily/`.
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=lookback + n_tail + 10)
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers')

    # Drop tickers with any NaN over the full panel. The empirical
    # scorer's cluster-aggregate step (`_build_cluster_aggregate_prices`)
    # propagates NaN across every member of any cluster that contains a
    # NaN-valued ticker — so a single late-listing ticker poisons every
    # other cluster member's CWT divergence. Fully-finite filtering
    # keeps ~289/312 names and the full 2000-2026 date range; the
    # alternative (start_date='2003-07-16' to admit all 312) trims four
    # years off the panel for fewer than 25 names. Net loser.
    finite_cols = (~prices.isna()).all(axis=0)
    n_dropped = int((~finite_cols).sum())
    if n_dropped > 0:
        prices = prices.loc[:, finite_cols]
        print(f'  dropped {n_dropped} tickers with NaN periods; '
              f'kept {prices.shape[1]} (fully-finite over the full '
              f'date range)')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, rebal_days={rebal_days}, '
          f'fp_window={fp_window}, k_clusters={k_clusters}')
    print(f'  cache_dir={cache_dir}')

    print('\n[scoring] computing top-N weights for each scorer...')
    emp_top = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, k_clusters=k_clusters, fp_window=fp_window,
        cache_dir=cache_dir)
    far_top = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window, cache_dir=cache_dir)

    print('[scoring] extracting fingerprints aligned to eval index ...')
    fps_eval = fingerprints_for_weights(
        prices, emp_top, scales=scales, lookback=lookback,
        fp_window=fp_window, cache_dir=cache_dir)
    # Both top_weights have identical index/columns (same prices,
    # same lookback) — share the fps cube.

    print('\n[pairs] building variants and running NN diagnostics ...')
    diagnostics: dict[str, dict] = {}
    backtests: list[bt.Backtest] = []

    for strat_name, top in [('empirical', emp_top), ('farthest', far_top)]:
        long_only = top
        mkt_neutral = market_neutral_weights(top, prices=prices)
        nn_partner = nearest_non_top_partner(fps_eval, top)
        nn_pair = nearest_neighbor_pair_weights(top, fps_eval)

        diag = _diagnostics(nn_partner, top, fps_eval, rebal_days)
        diagnostics[strat_name] = diag

        for variant_name, w in [
            ('long-only', long_only),
            ('mkt-neutral', mkt_neutral),
            ('nn-pair', nn_pair),
        ]:
            label = f'{strat_name}|{variant_name}'
            row_sums = w.iloc[::rebal_days].sum(axis=1)
            gross = w.iloc[::rebal_days].abs().sum(axis=1).mean()
            print(f'  {label}: '
                  f'mean_net={row_sums.mean():+.3f} '
                  f'gross={gross:.3f}')
            backtests.append(build_strategy(
                label, prices, w,
                rebal_days=rebal_days, commission_bps=commission_bps))

    print(f'\nRunning bt backtests ({len(backtests)} strategies) ...')
    result = bt.run(*backtests)

    stats = result.stats
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return', 'worst_year']
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    leaderboard = stats.loc[headline, order].T

    print('\n' + '=' * 100)
    print('NN-pair leaderboard — sorted by daily Sharpe')
    print('=' * 100)
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(leaderboard.to_string())

    print('\n' + '=' * 100)
    print('Sanity checks — nearest-neighbor pair construction')
    print('=' * 100)
    diag_rows = []
    for scorer, d in diagnostics.items():
        diag_rows.append({
            'scorer': scorer,
            'distinctness_violations': d['distinctness_violations'],
            'distinctness_frac': d['distinctness_frac'],
            'rebal_dates': d['rebal_count'],
            'all_share_count': d['all_share_count'],
            'all_share_frac': d['all_share_frac'],
            'd_mean': d['distance_mean'],
            'd_p05': d['distance_p05'],
            'd_p50': d['distance_p50'],
            'd_p95': d['distance_p95'],
            'pair_stability': d['pair_stability'],
            'n_samples': d['n_distance_samples'],
        })
    diag_df = pd.DataFrame(diag_rows).set_index('scorer')
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(diag_df.to_string())

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    result.plot(ax=ax)
    ax.set_title(
        f'NN-pair vs prior overlays — stooq_us_long ({prices.index[0].date()} '
        f'→ {prices.index[-1].date()}, top-{top_n}, rebal={rebal_days}d, '
        f'commission={commission_bps}bps)')
    fig.tight_layout()
    fig_path = out / 'relational-nn-pairs-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-nn-pairs-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('NN-pair leaderboard — sorted by daily Sharpe\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Sanity checks — nearest-neighbor pair construction\n')
        f.write('=' * 100 + '\n')
        f.write(diag_df.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(str(result.stats) + '\n')
    print(f'Saved {stats_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(_DEFAULT_DATA_DIR))
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--cache-dir', default=_DEFAULT_CACHE_DIR)
    args = p.parse_args()
    run(**vars(args))
