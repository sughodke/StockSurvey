"""Transition-triggered rebal vs scheduled-20d on the wider 312-ticker
universe.

Hypothesis (structural, validated by the human): a stock's
cluster-membership transition in scalogram-fingerprint space is a
*leading* indicator of regime change. Most equity allocators don't
run k-means on CWT fingerprints, so transitions are asymmetric
information that materializes before realized vol catches up. Cluster
events are asynchronous (one stock at a time, on dates that aren't
multiples of 20) — fixed-cadence rebal misses them or is forced to wait.

Three head-to-head bt backtests on `apps/notebook/data/stooq_us_long/`:

  * `scheduled-20d`        — baseline: rebal every 20 trading days.
  * `transition-only`      — rebal *only* on dates where any current
                             top-N ticker transitions. No scheduled
                             cadence.
  * `transition-or-20d`    — union of transition events and the 20-day
                             grid. Catches transitions early *and*
                             keeps a fallback cadence.

All three use idea-A weights (`weights_excess_regime_empirical`) under
identical knobs (`lookback=120, n_tail=20, top_n=20, k_clusters=11`).
Identical commission (10 bps), identical `integer_positions=False`,
same date range.

Outputs
-------
  Output/relational-transition-triggered-equity.png
  Output/relational-transition-triggered-stats.txt
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix

from relational.cluster_tracking import stabilize_cluster_ids
from ss_portfolio import apply_nan_mask, select_top_n_matrix

from relational.empirical_sectors import empirical_excess_divergence_scores
from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt
from relational.transitions import (
    detect_transition_dates,
    trigger_dates_from_transitions,
)

warnings.filterwarnings('ignore')


def _make_commission_fn(bps: float):
    frac = bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def _bt_safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.ffill().bfill()


def _mask_weights_to_active(
    weights: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    valid = np.isfinite(prices.reindex(
        index=weights.index, columns=weights.columns).values)
    w = weights.fillna(0).values * valid
    sums = w.sum(axis=1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    w = w / sums
    return pd.DataFrame(w, index=weights.index, columns=weights.columns)


def _build_strategy(
    name: str, prices: pd.DataFrame, weights: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    *, commission_bps: float,
) -> tuple[bt.Backtest, int]:
    """Build a bt.Backtest with `RunOnDate(*rebal_dates)`.

    Returns (backtest, n_rebals_actually_used).
    """
    # Snap rebal_dates onto the weights index, drop non-actionable rows
    # (gross < 0.1 → empty basket from incomplete scoring).
    aligned = sorted(set(d for d in rebal_dates if d in weights.index))
    if not aligned:
        # Fall back to first valid weights row to avoid bt crashing.
        aligned = [weights.index[0]]
    rebal_weights = weights.loc[aligned]
    nonzero = rebal_weights.abs().sum(axis=1) > 0.1
    if nonzero.any():
        rebal_weights = rebal_weights.loc[nonzero]
    n_rebals = len(rebal_weights)

    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    bt_prices = _bt_safe_prices(prices)
    return (
        bt.Backtest(strategy, bt_prices,
                    commissions=_make_commission_fn(commission_bps),
                    integer_positions=False),
        n_rebals,
    )


def run(
    *, data_dir: str,
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    refit_days: int = 252,
    persistence: int = 5,
    start: str | None = None,
    end: str | None = None,
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> pd.DataFrame:
    print(f'Loading Stooq prices from {data_dir} (no ticker filter — '
          'whole subset) ...')
    min_history = lookback + n_tail + 10
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=min_history,
        start_date=start, end_date=end,
        tickers=None)
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'(min_history={min_history})')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, k_clusters={k_clusters}, fp_window={fp_window}, '
          f'refit_days={refit_days}, persistence={persistence}')

    # --- Score panel + cluster IDs (idea A). ----------------------------
    print('\n[scoring] empirical excess-divergence scores + cluster IDs ...')
    emp_scores, raw_cluster_ids_eval = empirical_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        fp_window=fp_window, k_clusters=k_clusters,
        refit_days=refit_days, return_clusters=True)
    # Reconstruct the full-panel `cluster_ids` matrix by left-padding
    # the eval slice with the same `-1` sentinel `_refit_cluster_assignments`
    # uses for pre-lookback rows. Avoids re-running KMeans across the panel.
    n_dates_full = len(prices)
    raw_cluster_ids_full = np.full(
        (n_dates_full, prices.shape[1]), -1, dtype=np.int64)
    raw_cluster_ids_full[lookback:] = raw_cluster_ids_eval

    # We do still need fingerprints for centroid computation during
    # stabilization (one CWT pass + a stride-trick reshape — both fast).
    coeffs = load_or_compute_cwt(prices, scales, lookback)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)

    # --- Sanity 1: pre vs post stabilization at first refit boundary. ---
    refit_dates_idx = list(range(lookback, fps.shape[0], refit_days))
    if len(refit_dates_idx) >= 2:
        t0, t1 = refit_dates_idx[0], refit_dates_idx[1]
        before_pre  = Counter(raw_cluster_ids_full[t0].tolist())
        before_post = Counter(raw_cluster_ids_full[t1].tolist())
        print(f'\n[sanity-1] Cluster sizes around first refit boundary '
              f'(pre-stabilization):')
        print(f'  segment 0 (t={t0}): '
              f'{dict(sorted((k, v) for k, v in before_pre.items() if k >= 0))}')
        print(f'  segment 1 (t={t1}): '
              f'{dict(sorted((k, v) for k, v in before_post.items() if k >= 0))}')

    print('\n[stabilizing] Hungarian-matching cluster IDs across refits ...')
    stable_full = stabilize_cluster_ids(
        raw_cluster_ids_full, fps,
        refit_days=refit_days, lookback=lookback)

    if len(refit_dates_idx) >= 2:
        t0, t1 = refit_dates_idx[0], refit_dates_idx[1]
        after_pre  = Counter(stable_full[t0].tolist())
        after_post = Counter(stable_full[t1].tolist())
        print(f'[sanity-1] Cluster sizes around first refit boundary '
              f'(post-stabilization):')
        print(f'  segment 0 (t={t0}): '
              f'{dict(sorted((k, v) for k, v in after_pre.items() if k >= 0))}')
        print(f'  segment 1 (t={t1}): '
              f'{dict(sorted((k, v) for k, v in after_post.items() if k >= 0))}')

    # Eval-window slice (matches transition_mask convention).
    cluster_ids_eval = stable_full[lookback:]
    n_eval, n_tickers = cluster_ids_eval.shape

    # --- Detect transitions (with persistence filter). ------------------
    print(f'\n[transitions] persistence={persistence} days ...')
    transitions = detect_transition_dates(
        cluster_ids_eval, persistence=persistence)
    n_events = int(transitions.sum())
    n_years = (prices.index[-1] - prices.index[lookback]).days / 365.25
    per_ticker_per_year = n_events / max(n_years, 1e-9) / max(n_tickers, 1)
    print(f'  total events: {n_events}  '
          f'(~{per_ticker_per_year:.2f}/ticker/yr)')

    # --- Sanity 2: top transitions, e.g. "ticker → cluster X on date D".
    eval_index = prices.index[lookback:]
    event_locations = np.argwhere(transitions)
    transition_records: list[tuple[str, int, pd.Timestamp]] = []
    for t, i in event_locations:
        new_id = int(cluster_ids_eval[t, i])
        transition_records.append((prices.columns[i], new_id, eval_index[t]))
    print('\n[sanity-2] Top 10 most-frequent (ticker, new_cluster_id) pairs:')
    pair_counter = Counter(
        (rec[0], rec[1]) for rec in transition_records)
    for (tk, cid), cnt in pair_counter.most_common(10):
        print(f'  {tk:>5s} → cluster {cid:>2d}   x{cnt}')

    print('\n[sanity-3] Top 10 single-date transition clusters '
          '(date with most simultaneous transitions):')
    date_counter = Counter(rec[2] for rec in transition_records)
    for d, cnt in date_counter.most_common(10):
        # Show which cluster IDs this date moved into.
        these = [rec for rec in transition_records if rec[2] == d]
        clust_dist = Counter(c for _, c, _ in these).most_common(3)
        print(f'  {d.date()}  events={cnt:>3d}   '
              f'top-clusters={clust_dist}')

    # --- Build top-N membership mask for the eval window. ---------------
    print('\n[weights] empirical top-N (idea A) ...')
    masked_scores = apply_nan_mask(emp_scores, prices.values, lookback)
    weights_arr = select_top_n_matrix(masked_scores, top_n, ascending=False)
    emp_top = pd.DataFrame(
        weights_arr, index=prices.index[lookback:], columns=prices.columns)
    emp_top = _mask_weights_to_active(emp_top, prices)

    # Restrict trigger events to top-N picks: at each row, the pick mask
    # is `emp_top.values > 0`. The transition-triggered baskets fire
    # only when a transition lands on a *current* top-N name. Otherwise
    # cross-sectional irrelevance (transition on rank-300 ticker) would
    # over-fire.
    pick_mask = emp_top.values > 0
    # `transitions` is indexed eval-rows = prices.index[lookback:]; same
    # for emp_top. They share the same row count.
    assert transitions.shape == pick_mask.shape, (
        f'shape mismatch: transitions {transitions.shape} vs '
        f'pick_mask {pick_mask.shape}')

    # --- Define rebal-date sets. ----------------------------------------
    weights_index = emp_top.index
    scheduled_dates = list(weights_index[::rebal_days])
    transition_dates = trigger_dates_from_transitions(
        transitions, prices.index, lookback,
        selected_columns=pick_mask)
    union_dates = sorted(set(scheduled_dates) | set(transition_dates))

    print(f'\n[rebal sets]')
    print(f'  scheduled-20d:        {len(scheduled_dates)} dates')
    print(f'  transition-only:      {len(transition_dates)} dates')
    print(f'  transition-or-20d:    {len(union_dates)} dates')

    # --- Build & run the three backtests. -------------------------------
    print('\n[bt] running 3 backtests ...')
    backtests: list[bt.Backtest] = []
    rebal_counts: dict[str, int] = {}
    for label, dates in [
        ('scheduled-20d',     scheduled_dates),
        ('transition-only',   transition_dates),
        ('transition-or-20d', union_dates),
    ]:
        bt_obj, n_rebals = _build_strategy(
            label, prices, emp_top, dates,
            commission_bps=commission_bps)
        backtests.append(bt_obj)
        rebal_counts[label] = n_rebals
        print(f'  {label}: {n_rebals} rebal rows')

    result = bt.run(*backtests)

    stats = result.stats
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return']
    leaderboard = stats.loc[headline, order].T.copy()
    leaderboard['n_rebals'] = leaderboard.index.map(rebal_counts)
    # Reorder columns to scorer/sharpe/cagr/dd/calmar/n_rebals/vol.
    leaderboard = leaderboard[
        ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
         'n_rebals', 'daily_vol']
    ]

    print('\n' + '=' * 100)
    print('Transition-triggered leaderboard — sorted by daily Sharpe')
    print(f'  universe: {n_tickers} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')
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
        f'Transition-triggered rebal — '
        f'{prices.index[0].date()} → {prices.index[-1].date()}, '
        f'{n_tickers} tickers, top-{top_n}, '
        f'k={k_clusters}, persistence={persistence}, '
        f'commission={commission_bps}bps')
    fig.tight_layout()
    fig_path = out / 'relational-transition-triggered-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-transition-triggered-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Transition-triggered leaderboard — sorted by daily Sharpe\n')
        f.write(f'  universe: {n_tickers} tickers\n')
        f.write(f'  date range: {prices.index[0].date()} → '
                f'{prices.index[-1].date()}\n')
        f.write(f'  top_n={top_n}  lookback={lookback}  n_tail={n_tail}  '
                f'fp_window={fp_window}  k_clusters={k_clusters}  '
                f'refit_days={refit_days}  persistence={persistence}\n')
        f.write(f'  rebal_days={rebal_days}  '
                f'commission_bps={commission_bps}\n')
        f.write(f'  total transition events: {n_events} '
                f'(~{per_ticker_per_year:.2f}/ticker/yr)\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Top 10 most-frequent (ticker → cluster) transitions:\n')
        for (tk, cid), cnt in pair_counter.most_common(10):
            f.write(f'  {tk:>5s} → cluster {cid:>2d}   x{cnt}\n')
        f.write('\nTop 10 single-date transition clusters:\n')
        for d, cnt in date_counter.most_common(10):
            these = [rec for rec in transition_records if rec[2] == d]
            clust_dist = Counter(c for _, c, _ in these).most_common(3)
            f.write(f'  {d.date()}  events={cnt}  '
                    f'top-clusters={clust_dist}\n')
        f.write('\nFull bt stats:\n')
        f.write(str(result.stats) + '\n')
    print(f'Saved {stats_path}')

    return leaderboard


def _default_data_dir() -> str:
    import os
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        cand = ancestor / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
        if cand.exists():
            return str(cand)
    return os.environ.get(
        'STOOQ_US_LONG_DIR', './apps/notebook/data/stooq_us_long')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=_default_data_dir())
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--refit-days', type=int, default=252)
    p.add_argument('--persistence', type=int, default=5)
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
