"""Short-vol P&L diagnostic.

Two-part question this answers:
  1. Does any scorer pick top-N names whose short-straddle P&L beats
     the universe-wide vrp baseline?
  2. Or does selling vol equally on every active ticker just dominate
     all attempted scorer-driven baskets?

Reuses the score-computation dispatcher from `diagnostic_dislocation_vs_vol`
to test all 9 scorers (5 dislocation + 4 brainstorm). For each scorer
we evaluate **both** sort directions — descending picks the high-score
tail, ascending picks the low-score tail — since different scorers'
"short-vol candidate" side differs (idea C / r1_ot lean negative, n1_entropy
leans positive in the t-stat diagnostic).

P&L unit is **vol points per cycle**: e.g., +0.05 means realized came
in 5 vol points below IV, capturing 5% of vol notional. Annualized
Sharpe assumes 252/rebal_days cycles per year; cycles overlap so this
is an approximation, not a portfolio Sharpe.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ss_features.vol import realized_vol
from ss_loaders import load_stooq_matrix

from relational.iv_data import load_atm_iv, load_dolthub_iv_parquet
from relational.research.diagnostic_dislocation_vs_vol import (
    ALL_SCORERS, BRAINSTORM_SCORERS, _compute_scores,
)
from relational.short_vol import (
    evaluate_short_vol, evaluate_universe_short_vol,
)

warnings.filterwarnings('ignore')


PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


def _per_ticker_realized_vol(prices: pd.DataFrame, window: int) -> np.ndarray:
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    arr = prices.values
    for j in range(arr.shape[1]):
        out[:, j] = realized_vol(arr[:, j], window)
    return out


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    rebal_days: int = 20,
    vol_window: int = 20,
    start: str = '2018-06-01',
    end: str = '2026-04-30',
    iv_source: str = 'dolthub',
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=lookback + 50,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  {prices.shape[0]} dates x {prices.shape[1]} tickers')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]

    print('\nComputing forward realized vol...')
    rv = _per_ticker_realized_vol(prices, vol_window)
    forward_ann = rv * np.sqrt(252)

    print(f'Loading ATM IV (source={iv_source})...')
    if iv_source == 'gauss314':
        iv = load_atm_iv()
        iv = iv.reindex(index=prices.index, columns=prices.columns)
    elif iv_source == 'dolthub':
        iv = load_dolthub_iv_parquet(tickers=list(prices.columns))
        iv = iv.reindex(index=prices.index).ffill(limit=7)
        iv = iv.reindex(columns=prices.columns)
    else:
        raise ValueError(f'unknown iv_source: {iv_source!r}')
    iv_ann = iv.values.astype(np.float64)
    n_iv = int(np.isfinite(iv_ann).sum())
    print(f'  IV coverage: {n_iv}/{iv_ann.size} '
          f'({100 * n_iv / iv_ann.size:.1f}%) cells finite')

    # Universe baseline first.
    print('\n[universe] sell vol equally on every active ticker each rebalance')
    universe_stats = evaluate_universe_short_vol(
        iv_ann, forward_ann, prices,
        lookback=lookback, rebal_days=rebal_days, vol_window=vol_window)
    print(f'  n_cycles={universe_stats["n_cycles"]} '
          f'mean_pnl={universe_stats["mean_cycle_pnl"]:+.4f} '
          f'sharpe={universe_stats["sharpe"]:.2f}')

    rows: list[dict] = []
    rows.append({
        'scorer': 'UNIVERSE', 'direction': '—',
        **universe_stats,
    })

    all_scorers = list(ALL_SCORERS) + list(BRAINSTORM_SCORERS)
    for name in all_scorers:
        print(f'\n[{name}] computing scores...')
        scores = _compute_scores(
            name, prices, lookback=lookback, n_tail=n_tail,
            scales=scales, fp_window=fp_window, k_clusters=k_clusters)
        for direction_label, descending in [('top-desc', True), ('bot-asc', False)]:
            stats = evaluate_short_vol(
                scores, iv_ann, forward_ann, prices,
                lookback=lookback, top_n=top_n,
                rebal_days=rebal_days, vol_window=vol_window,
                descending=descending)
            print(f'  {direction_label}: '
                  f'n_cycles={stats["n_cycles"]} '
                  f'mean_pnl={stats["mean_cycle_pnl"]:+.4f} '
                  f'sharpe={stats["sharpe"]:.2f} '
                  f'win_rate={stats["win_rate_per_cycle"]:.2f}')
            rows.append({
                'scorer': name, 'direction': direction_label,
                **stats,
            })

    summary = pd.DataFrame.from_records(rows)
    print('\n' + '=' * 100)
    print('Short-vol P&L leaderboard '
          f'(top-{top_n}, rebal={rebal_days}d, vol_window={vol_window}d)')
    print('=' * 100)
    cols = ['scorer', 'direction', 'mean_cycle_pnl', 'std_cycle_pnl',
            'sharpe', 'win_rate_per_cycle', 'cum_pnl', 'max_dd',
            'n_cycles']
    sorted_summary = summary.sort_values('sharpe', ascending=False)
    with pd.option_context(
        'display.float_format', lambda x: f'{x:+.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(sorted_summary[cols].to_string(index=False))

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    txt_path = out / 'relational-short-vol-pnl.txt'
    with open(txt_path, 'w') as f:
        f.write('Short-vol P&L leaderboard\n')
        f.write('=' * 100 + '\n')
        f.write(sorted_summary[cols].to_string(index=False) + '\n')
    print(f'\nSaved {txt_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--vol-window', type=int, default=20)
    p.add_argument('--start', default='2018-06-01')
    p.add_argument('--end', default='2026-04-30')
    p.add_argument('--iv-source', default='dolthub',
                   choices=('gauss314', 'dolthub'))
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
