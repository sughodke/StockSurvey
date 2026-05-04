"""Wide-universe pair-trade diagnostic — 312-ticker `stooq_us_long`.

Sibling of `diagnostic_pair_trades.py`. Same scorers + variants but on
the curated 312-ticker subset at `apps/notebook/data/stooq_us_long/`
(daily Stooq layout, ~26y of history). The Phase-2 21-mega-cap pair-
trade test was largely a null result (negative Sharpe across mkt-neutral
and rank-spread variants); the structural prediction is that 312 names
with diversified factor exposure should let intra-cluster (cluster-pair)
and rank-spread variants extract real cross-sectional alpha.

Idea-B (analog k-NN) is **deliberately skipped**: the inner-loop k-NN
search over historical fingerprints is `O(n_eval × n_tickers × n_cand)`
and would take many hours on 312 tickers even for the scoring pass.
The cluster-pair variant exists only on the empirical (idea-A) scorer
because the construction needs cluster ids returned by
`empirical_excess_divergence_scores(..., return_clusters=True)`.

Backtest count: 3 scorers × 3 variants (long-only, mkt-neutral,
rank-spread) + 1 cluster-pair on empirical = **10 backtests**.

Outputs
-------
  Output/relational-pair-trades-wide-equity.png
  Output/relational-pair-trades-wide-stats.txt

Wall time
---------
312 tickers × ~6618 dates × 8 scales is ~40-50× the scoring work of
Phase-2 (21 × 3239). Causal CWT on the wider panel is one-shot and
cached (the scalogram cache is keyed by content hash on prices, so
identical panels short-circuit on subsequent runs). The bt rebalance
solver is the cubic-ish time hog; we parallelize the 10 backtests via
`multiprocessing.Pool` to soak up cores. Expect 30-50 min wall serial,
~10-15 min parallel on an 8-core box.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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

from relational.empirical_sectors import (
    empirical_excess_divergence_scores, weights_excess_regime_empirical,
)
from relational.farthest import centroid_distance_scores, weights_regime_farthest
from relational.pairs import (
    cluster_pair_weights, market_neutral_weights, rank_spread_weights,
)

warnings.filterwarnings('ignore')


def _make_commission_fn(bps: float):
    frac = bps / 10000.0

    def commission(q, p):
        return abs(q) * p * frac

    return commission


def _bt_safe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill + back-fill NaN prices for bt's price-feed.

    bt's rebalance solver raises if a held position's price becomes NaN
    mid-holding (e.g. a ticker delists between rebalance dates). The
    Phase-2 21-mega-cap panel had no such gaps, but the wider 312-name
    `stooq_us_long` subset has plenty. We forward-fill so a delisted /
    gapped name's last known price persists until the next rebalance
    closes the position; back-fill handles tickers whose first NaN-free
    price is after the panel start (early-listed names not yet trading).

    Returns a *separate* DataFrame so the scoring path keeps the
    original NaN-laden prices (NaN is meaningful there — `apply_nan_mask`
    needs it). Only the bt price-feed sees the filled version.
    """
    return prices.ffill().bfill()


def _mask_weights_to_active(
    weights: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Zero-out any weight (long or short) for a ticker that doesn't
    have a finite price on that row. Prevents shorting names that
    haven't started trading yet and longs/shorts on dates after a
    delisting. Renormalizes long-only rows back to sum=1; long-short
    rows are left as-is (they intentionally sum to ~0)."""
    valid = np.isfinite(prices.reindex(
        index=weights.index, columns=weights.columns).values)
    w = weights.fillna(0).values * valid
    is_long_only = (w >= 0).all(axis=1) & (w.sum(axis=1) > 0)
    # Long-only rows: renormalize so the basket sums to 1.
    if is_long_only.any():
        sums = w[is_long_only].sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        w[is_long_only] = w[is_long_only] / sums
    return pd.DataFrame(w, index=weights.index, columns=weights.columns)


def _build_strategy(
    name: str, prices: pd.DataFrame, weights: pd.DataFrame,
    *, rebal_days: int, commission_bps: float,
) -> bt.Backtest:
    rebal_weights = weights.iloc[::rebal_days]
    # Drop the first rebalance row(s) where the score isn't fully
    # populated yet (gross < 0.1 → no actionable basket). Keeps bt from
    # holding a ~0 basket through the first 20 days then snapping into
    # a full position at the first useful rebalance.
    nonzero = rebal_weights.abs().sum(axis=1) > 0.1
    if nonzero.any():
        rebal_weights = rebal_weights.loc[nonzero]
    strategy = bt.Strategy(name, [
        bt.algos.RunOnDate(*rebal_weights.index),
        bt.algos.WeighTarget(rebal_weights),
        bt.algos.Rebalance(),
    ])
    bt_prices = _bt_safe_prices(prices)
    return bt.Backtest(strategy, bt_prices,
                       commissions=_make_commission_fn(commission_bps),
                       integer_positions=False)


def _baseline_scores(prices: pd.DataFrame, *, lookback, n_tail, scales,
                     divergence='kl') -> np.ndarray:
    """Replicate `weights_regime`'s per-stock CWT-power KL divergence,
    but return the raw `(n_eval, n_tickers)` score matrix so we can also
    pull bot-N picks for rank-spread."""
    from ss_indicators import get_divergence
    from ss_wavelets import causal_cwt, precompute_windows
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)
    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    return np.array(
        div_fn(recent, historical, scale_log_weights), copy=True)


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


def _run_one_backtest(args) -> tuple[str, pd.DataFrame]:
    """Worker target for the bt-parallel pool.

    bt.Backtest objects are not always picklable across processes (the
    Strategy graph holds bound algos with closures), so we accept the
    plumbing inputs (label, prices, weights, etc.) and reconstruct the
    Backtest inside the worker. Returns `(label, stats_dataframe)` —
    bt.Result is reconstructable client-side from the stats DataFrame
    plus the equity curve.

    The first element returned is also `(equity_series_name,
    equity_series)` so the main process can stitch a combined equity
    plot without re-running anything.
    """
    label, prices, weights, rebal_days, commission_bps = args
    backtest = _build_strategy(
        label, prices, weights,
        rebal_days=rebal_days, commission_bps=commission_bps)
    result = bt.run(backtest)
    stats = result.stats
    # Equity series for the plot.
    equity = result.prices[label].copy()
    equity.name = label
    return label, stats[label].copy(), equity


def run(
    *, data_dir: str,
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    start: str | None = None,
    end: str | None = None,
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
    parallel: bool = True,
    n_workers: int | None = None,
) -> None:
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
          f'top_n={top_n}, rebal_days={rebal_days}, '
          f'k_clusters={k_clusters}, fp_window={fp_window}')

    print('\n[scoring] computing top-N and bot-N weights for each scorer...')

    # --- baseline (per-stock CWT-power KL divergence) ----------------
    print('  baseline (CWT-power KL)...')
    base_scores = _baseline_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales)
    base_top = _baseline_long(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n, scales=scales)
    base_bot = _bot_weights_from_scores(
        base_scores, prices, top_n=top_n, lookback=lookback)

    # --- empirical (idea A) — also pull cluster_ids for cluster-pair --
    print(f'  empirical (k-means k={k_clusters} on scalogram fingerprints)...')
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

    # --- farthest (idea C) ------------------------------------------
    print('  farthest (cross-sectional centroid distance)...')
    far_scores = centroid_distance_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window)
    far_top = weights_regime_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window)
    far_bot = _bot_weights_from_scores(
        far_scores, prices, top_n=top_n, lookback=lookback)

    # No analog (idea B) — O(n_eval * n_tickers * n_cand) k-NN inner
    # loop is hours of wall time at this universe size. Skipped in the
    # wide diagnostic; can be added later via a Modal-parallelized
    # version if the result here motivates the spend.

    base_strategies = {
        'baseline':  (base_top, base_bot),
        'empirical': (emp_top, emp_bot),
        'farthest':  (far_top, far_bot),
    }

    print('\n[pairs] constructing market-neutral and rank-spread variants...')
    bt_jobs: list[tuple] = []
    for strat_name, (top, bot) in base_strategies.items():
        long_only = _mask_weights_to_active(top, prices)
        mkt_neutral = _mask_weights_to_active(
            market_neutral_weights(top, prices=prices), prices)
        rank_spread = _mask_weights_to_active(
            rank_spread_weights(top, bot), prices)
        for variant_name, w in [
            ('long-only', long_only),
            ('mkt-neutral', mkt_neutral),
            ('rank-spread', rank_spread),
        ]:
            label = f'{strat_name}|{variant_name}'
            row_sums = w.iloc[::rebal_days].sum(axis=1)
            gross = w.iloc[::rebal_days].abs().sum(axis=1).mean()
            print(f'  {label}: '
                  f'mean_net={row_sums.mean():+.3f} gross={gross:.3f}')
            bt_jobs.append((label, prices, w, rebal_days, commission_bps))

    # Special: cluster-aware pair on empirical (idea A) — long winner per
    # cluster, short cluster aggregate. Hedges intra-cluster, not universe.
    label = 'empirical|cluster-pair'
    cp = _mask_weights_to_active(emp_cluster_pair, prices)
    row_sums = cp.iloc[::rebal_days].sum(axis=1)
    gross = cp.iloc[::rebal_days].abs().sum(axis=1).mean()
    print(f'  {label}: mean_net={row_sums.mean():+.3f} gross={gross:.3f}')
    bt_jobs.append((label, prices, cp, rebal_days, commission_bps))

    print(f'\nRunning bt backtests ({len(bt_jobs)} strategies, '
          f'parallel={parallel}; this is the slow step) ...')
    stats_by_label: dict[str, pd.Series] = {}
    equity_by_label: dict[str, pd.Series] = {}
    if parallel:
        nw = n_workers or min(len(bt_jobs), max(1, (mp.cpu_count() or 2) - 1))
        print(f'  pool: {nw} workers')
        with mp.Pool(nw) as pool:
            for i, (label, stats_col, equity) in enumerate(
                    pool.imap_unordered(_run_one_backtest, bt_jobs)):
                stats_by_label[label] = stats_col
                equity_by_label[label] = equity
                print(f'  [{i+1}/{len(bt_jobs)}] done: {label}')
    else:
        for i, args in enumerate(bt_jobs):
            label, stats_col, equity = _run_one_backtest(args)
            stats_by_label[label] = stats_col
            equity_by_label[label] = equity
            print(f'  [{i+1}/{len(bt_jobs)}] done: {label}')

    # Stitch into a single stats DataFrame with the same row labels bt
    # uses, in the order the jobs were submitted (deterministic across
    # parallel/serial runs).
    job_order = [j[0] for j in bt_jobs]
    stats = pd.concat(
        [stats_by_label[label].rename(label) for label in job_order],
        axis=1)

    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return', 'worst_year']
    leaderboard = stats.loc[headline, order].T

    print('\n' + '=' * 100)
    print('Wide-universe pair-trade leaderboard — sorted by daily Sharpe')
    print(f'  universe: {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')
    print('=' * 100)
    with pd.option_context(
        'display.float_format', lambda x: f'{x:.4f}',
        'display.max_columns', None, 'display.width', 200,
    ):
        print(leaderboard.to_string())

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    # Equity-comparison plot (build manually since we ran each bt in its
    # own pool worker — bt.Result for the combined plot is not at hand).
    fig, ax = plt.subplots(figsize=(14, 8))
    for label in order:
        eq = equity_by_label[label]
        ax.plot(eq.index, eq.values, label=label, linewidth=1.0)
    ax.set_title(
        f'Wide-universe pair-trade overlays — '
        f'{prices.index[0].date()} → {prices.index[-1].date()}, '
        f'{prices.shape[1]} tickers, top-{top_n}, rebal={rebal_days}d, '
        f'commission={commission_bps}bps')
    ax.set_ylabel('equity (start = 100)')
    ax.set_xlabel('date')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out / 'relational-pair-trades-wide-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-pair-trades-wide-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Wide-universe pair-trade leaderboard — sorted by daily Sharpe\n')
        f.write(f'  universe: {prices.shape[1]} tickers\n')
        f.write(f'  date range: {prices.index[0].date()} → '
                f'{prices.index[-1].date()}\n')
        f.write(f'  top_n={top_n}  lookback={lookback}  n_tail={n_tail}  '
                f'fp_window={fp_window}  k_clusters={k_clusters}\n')
        f.write(f'  rebal_days={rebal_days}  commission_bps={commission_bps}\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(stats.to_string() + '\n')
    print(f'Saved {stats_path}')


def _default_data_dir() -> str:
    """Repo-relative default: `apps/notebook/data/stooq_us_long/`. Falls
    back to the env var `STOOQ_US_LONG_DIR` if the relative path can't
    be resolved (e.g. running from a different repo)."""
    import os
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        cand = ancestor / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
        if cand.exists():
            return str(cand)
    return os.environ.get('STOOQ_US_LONG_DIR', './apps/notebook/data/stooq_us_long')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=_default_data_dir())
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-clusters', type=int, default=11)
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--no-parallel', dest='parallel', action='store_false')
    p.add_argument('--n-workers', type=int, default=None)
    args = p.parse_args()
    run(**vars(args))
