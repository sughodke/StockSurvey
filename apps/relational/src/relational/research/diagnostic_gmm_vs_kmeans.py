"""Head-to-head diagnostic: hard k-means clusters vs. soft GMM
posteriors on the empirical-sector idea.

Backs out 4 strategies on the wide 312-ticker `stooq_us_long`
universe:

  1. `empirical|kmeans`              — long-only top-N excess
                                       divergence using hard k-means
                                       clusters (existing baseline,
                                       `weights_excess_regime_empirical`).
  2. `empirical|gmm`                 — same shape but GMM soft
                                       cluster aggregates
                                       (`weights_excess_regime_gmm`).
  3. `empirical|kmeans|cluster-pair` — universe pair on hard clusters
                                       (the v2 result that was -0.07
                                       Sharpe on Phase-2; ported here
                                       to the wider universe).
  4. `empirical|gmm|cluster-pair`    — universe pair on GMM
                                       MAP-clusters with soft-aggregate
                                       scoring. The test of whether
                                       boundary-jitter was the problem.

Sanity diagnostics printed before the bt loop:

  * GMM posterior entropy histogram — most ticker-dates should be
    peaked (one component dominates). Uniform = degenerate fit.
  * MAP vs. k-means agreement at the first refit boundary — should
    overlap meaningfully but not be identical.
  * Cluster-size histograms before/after Hungarian stabilization at
    the first refit boundary (analog of the transition-triggered
    diagnostic) — sizes should be close pre/post; only the labels
    move, not the partition.
  * Mean correlation between the hard k-means aggregate and the soft
    GMM aggregate price series across tickers — quantifies how much
    the soft mix actually differs from the hard mean.

Outputs
-------
  Output/relational-gmm-vs-kmeans-equity.png
  Output/relational-gmm-vs-kmeans-stats.txt

Wall time
---------
312 tickers × 6618 dates is the same scoring panel as the wide
pair-trade diagnostic. The CWT cache hits (shared via the absolute
`cache_dir` path), so per-strategy wall time is dominated by the bt
rebalance solver — same ~5-10 min serial, ~2-3 min parallel on 4
strategies given a 4-worker pool.
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
from ss_portfolio.bt_helpers import build_strategy

from relational.empirical_sectors import (
    empirical_excess_divergence_scores, weights_excess_regime_empirical,
    _build_cluster_aggregate_prices, _refit_cluster_assignments,
)
from relational.empirical_sectors_gmm import (
    gmm_excess_divergence_scores, weights_excess_regime_gmm,
    gmm_cluster_pair_weights, _build_soft_cluster_aggregate_prices,
    _refit_gmm_assignments,
)
from relational.fingerprints import extract_fingerprints
from relational.pairs import cluster_pair_weights
from relational.scalogram_cache import load_or_compute_cwt

warnings.filterwarnings('ignore')


# Shared CWT cache with the main repo / sibling diagnostics. The
# wide-universe scoring CWT is ~50MB compressed; recomputing on 312
# tickers is ~5min. Absolute path so the worktree shares the canonical
# cache rather than a per-worktree duplicate.
_SHARED_CACHE_DIR = '/Users/sidghodke/Code/StockSurvey/apps/.scalogram-cache'


def _mask_weights_to_active(
    weights: pd.DataFrame, prices: pd.DataFrame,
) -> pd.DataFrame:
    """Zero-out weights on rows where the ticker has no finite price.
    Renormalize long-only rows back to sum=1; long-short rows pass
    through (intentionally net ~0)."""
    valid = np.isfinite(prices.reindex(
        index=weights.index, columns=weights.columns).values)
    w = weights.fillna(0).values * valid
    is_long_only = (w >= 0).all(axis=1) & (w.sum(axis=1) > 0)
    if is_long_only.any():
        sums = w[is_long_only].sum(axis=1, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        w[is_long_only] = w[is_long_only] / sums
    return pd.DataFrame(w, index=weights.index, columns=weights.columns)


def _run_one_backtest(args) -> tuple[str, pd.Series, pd.Series]:
    label, prices, weights, rebal_days, commission_bps = args
    backtest = build_strategy(
        label, prices, weights,
        rebal_days=rebal_days, commission_bps=commission_bps,
        drop_empty=True, safe_prices=True)
    result = bt.run(backtest)
    stats = result.stats
    equity = result.prices[label].copy()
    equity.name = label
    return label, stats[label].copy(), equity


def _print_sanity_checks(
    *,
    prices: pd.DataFrame,
    posteriors: np.ndarray,
    kmeans_cluster_ids: np.ndarray,
    fps: np.ndarray,
    lookback: int,
    refit_days: int,
    n_components: int,
) -> None:
    """Diagnostic prints — see module docstring for the four checks."""
    n_dates, n_tickers, _ = posteriors.shape
    print('\n[sanity] GMM posterior diagnostics')

    # 1) Posterior entropy histogram.
    valid = np.isfinite(posteriors).all(axis=2)
    if valid.any():
        post_valid = posteriors[valid]               # (n_valid, K)
        # Avoid log(0).
        eps = 1e-12
        ent = -np.sum(post_valid * np.log(post_valid + eps), axis=1)
        # Max entropy = ln(n_components) (uniform).
        max_ent = np.log(n_components)
        norm_ent = ent / max_ent
        bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        hist, edges = np.histogram(norm_ent, bins=bins)
        pct = 100 * hist / max(hist.sum(), 1)
        print('  posterior entropy histogram (normalized to ln(K)):')
        for i, p in enumerate(pct):
            print(f'    [{edges[i]:.2f}, {edges[i+1]:.2f}]: '
                  f'{hist[i]:>7d}  ({p:5.1f}%)')
        peaked = float((norm_ent < 0.5).mean())
        print(f'  peaked (entropy<0.5*ln(K)): {peaked:.1%}; '
              f'mean entropy: {ent.mean():.3f} '
              f'(max for uniform = {max_ent:.3f})')

    # 2) MAP vs k-means agreement at first refit boundary.
    if n_dates > lookback:
        t0 = lookback
        post_t = posteriors[t0]
        finite = np.isfinite(post_t).all(axis=1)
        kmeans_t = kmeans_cluster_ids[t0]
        km_finite = kmeans_t >= 0
        both_finite = finite & km_finite
        if both_finite.any():
            map_ids = np.argmax(post_t[both_finite], axis=1)
            km_ids = kmeans_t[both_finite]
            # Direct agreement is meaningless because the two label
            # spaces are independent; instead compute the mutual-info-
            # like best-label overlap via a Hungarian match.
            from scipy.optimize import linear_sum_assignment
            confusion = np.zeros((n_components, n_components), dtype=np.int64)
            for m, k in zip(map_ids, km_ids):
                if 0 <= int(k) < n_components and 0 <= int(m) < n_components:
                    confusion[int(m), int(k)] += 1
            row_idx, col_idx = linear_sum_assignment(-confusion)
            best = confusion[row_idx, col_idx].sum()
            agree = best / max(both_finite.sum(), 1)
            print(f'  MAP vs k-means agreement at t={t0}: '
                  f'{agree:.1%} (Hungarian-matched best assignment)')

    # 3) Cluster-size histograms pre/post stabilization at first refit
    #    boundary inside the GMM helper. We compare segment 1 (anchor)
    #    and segment 2 sizes — the stabilization should leave segment 2's
    #    *partition* untouched, only re-label.
    refit_dates = list(range(lookback, n_dates, refit_days))
    if len(refit_dates) >= 2:
        t1, t2 = refit_dates[0], refit_dates[1]
        sizes_seg1 = np.bincount(
            np.argmax(posteriors[t1, np.isfinite(
                posteriors[t1]).all(axis=1)], axis=1),
            minlength=n_components)
        sizes_seg2 = np.bincount(
            np.argmax(posteriors[t2, np.isfinite(
                posteriors[t2]).all(axis=1)], axis=1),
            minlength=n_components)
        print(f'  segment 1 (t={t1}) cluster sizes (post-stabilization): '
              f'{sizes_seg1.tolist()}')
        print(f'  segment 2 (t={t2}) cluster sizes (post-stabilization): '
              f'{sizes_seg2.tolist()}')
        # Sorted sizes — partition-only check (label-invariant).
        print(f'    sorted sizes seg1: {sorted(sizes_seg1.tolist(), reverse=True)}')
        print(f'    sorted sizes seg2: {sorted(sizes_seg2.tolist(), reverse=True)}')


def _aggregate_correlation(
    prices: pd.DataFrame,
    kmeans_aggregate: np.ndarray,
    gmm_aggregate: np.ndarray,
    lookback: int,
) -> tuple[float, float]:
    """Per-ticker Pearson correlation between hard-mean and soft-mix
    aggregate price series, computed on log-returns of the aggregate
    series over the post-lookback window.

    Returns (mean_corr, median_corr) across tickers."""
    n_dates, n_tickers = prices.shape
    rng = slice(lookback, n_dates)
    km = kmeans_aggregate[rng]
    gm = gmm_aggregate[rng]
    # Log-returns.
    km_safe = np.where(km > 0, km, np.nan)
    gm_safe = np.where(gm > 0, gm, np.nan)
    km_lr = np.diff(np.log(km_safe), axis=0)
    gm_lr = np.diff(np.log(gm_safe), axis=0)
    corrs: list[float] = []
    for i in range(n_tickers):
        a = km_lr[:, i]
        b = gm_lr[:, i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            continue
        a, b = a[m], b[m]
        if a.std() < 1e-10 or b.std() < 1e-10:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        corrs.append(c)
    if not corrs:
        return float('nan'), float('nan')
    return float(np.mean(corrs)), float(np.median(corrs))


def run(
    *, data_dir: str,
    top_n: int = 20,
    lookback: int = 120,
    n_tail: int = 20,
    fp_window: int = 21,
    k_clusters: int = 11,
    refit_days: int = 252,
    start: str | None = None,
    end: str | None = None,
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
    parallel: bool = True,
    n_workers: int | None = None,
    cache_dir: str = _SHARED_CACHE_DIR,
) -> None:
    print(f'Loading Stooq prices from {data_dir} '
          f'(no ticker filter — whole subset) ...')
    min_history = lookback + n_tail + 10
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=min_history,
        start_date=start, end_date=end,
        tickers=None)
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'(min_history={min_history})')
    print(f'  cache_dir = {cache_dir}')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, rebal_days={rebal_days}, '
          f'k_clusters={k_clusters}, fp_window={fp_window}, '
          f'refit_days={refit_days}')

    # ---- Score both clusterers, return cluster ids / posteriors ----
    print('\n[scoring] empirical k-means baseline ...')
    km_scores, km_cluster_ids = empirical_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        k_clusters=k_clusters, fp_window=fp_window, refit_days=refit_days,
        cache_dir=cache_dir, return_clusters=True)
    km_top = weights_excess_regime_empirical(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, k_clusters=k_clusters, fp_window=fp_window,
        refit_days=refit_days, cache_dir=cache_dir)
    km_cluster_pair = cluster_pair_weights(
        km_scores, km_cluster_ids, prices, lookback=lookback)

    print('[scoring] empirical GMM (soft posteriors) ...')
    gmm_scores, gmm_post_eval = gmm_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        n_components=k_clusters, fp_window=fp_window, refit_days=refit_days,
        cache_dir=cache_dir, return_posteriors=True)
    gmm_top = weights_excess_regime_gmm(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, n_components=k_clusters, fp_window=fp_window,
        refit_days=refit_days, cache_dir=cache_dir)
    gmm_cp = gmm_cluster_pair_weights(
        gmm_scores, gmm_post_eval, prices, lookback=lookback)

    # ---- Sanity checks (need full-range posteriors+cluster_ids) ----
    # `empirical_excess_divergence_scores(..., return_clusters=True)` only
    # returns the post-lookback slice, but the aggregate-correlation
    # diagnostic and the hard/soft cluster-size histograms need the
    # full-range arrays so refit boundaries fall on the same indices for
    # both clusterers. Re-derive both directly from the cached fingerprints.
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    full_posteriors = _refit_gmm_assignments(
        fps, n_components=k_clusters, refit_days=refit_days,
        lookback=lookback)
    full_km_cluster_ids = _refit_cluster_assignments(
        fps, k_clusters=k_clusters, refit_days=refit_days, lookback=lookback)
    _print_sanity_checks(
        prices=prices, posteriors=full_posteriors,
        kmeans_cluster_ids=full_km_cluster_ids, fps=fps, lookback=lookback,
        refit_days=refit_days, n_components=k_clusters)

    # Aggregate correlation hard vs soft. Both helpers accept the full
    # `(n_dates, n_tickers)` price matrix and the matching full cluster
    # tensor, so dates align row-for-row.
    km_agg = _build_cluster_aggregate_prices(
        prices.values, full_km_cluster_ids)
    gmm_agg = _build_soft_cluster_aggregate_prices(
        prices.values, full_posteriors)
    mean_c, med_c = _aggregate_correlation(
        prices, km_agg, gmm_agg, lookback)
    print(f'  hard-vs-soft aggregate per-ticker corr: '
          f'mean={mean_c:.4f}  median={med_c:.4f}')

    # ---- Build the four bt jobs ----
    print('\n[strategies] preparing the 4 bt jobs ...')
    bt_jobs: list[tuple] = []
    candidates: list[tuple[str, pd.DataFrame]] = [
        ('empirical|kmeans', km_top),
        ('empirical|gmm', gmm_top),
        ('empirical|kmeans|cluster-pair', km_cluster_pair),
        ('empirical|gmm|cluster-pair', gmm_cp),
    ]
    for label, w in candidates:
        masked = _mask_weights_to_active(w, prices)
        rs = masked.iloc[::rebal_days]
        gross = rs.abs().sum(axis=1).mean()
        net = rs.sum(axis=1).mean()
        print(f'  {label}: mean_net={net:+.3f} gross={gross:.3f}')
        bt_jobs.append((label, prices, masked, rebal_days, commission_bps))

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
    print('GMM-vs-kmeans diagnostic — sorted by daily Sharpe')
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

    fig, ax = plt.subplots(figsize=(14, 8))
    for label in order:
        eq = equity_by_label[label]
        ax.plot(eq.index, eq.values, label=label, linewidth=1.0)
    ax.set_title(
        f'GMM-vs-kmeans — '
        f'{prices.index[0].date()} → {prices.index[-1].date()}, '
        f'{prices.shape[1]} tickers, top-{top_n}, rebal={rebal_days}d, '
        f'commission={commission_bps}bps')
    ax.set_ylabel('equity (start = 100)')
    ax.set_xlabel('date')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out / 'relational-gmm-vs-kmeans-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-gmm-vs-kmeans-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('GMM-vs-kmeans diagnostic — sorted by daily Sharpe\n')
        f.write(f'  universe: {prices.shape[1]} tickers\n')
        f.write(f'  date range: {prices.index[0].date()} → '
                f'{prices.index[-1].date()}\n')
        f.write(f'  top_n={top_n}  lookback={lookback}  n_tail={n_tail}  '
                f'fp_window={fp_window}  k_clusters={k_clusters}  '
                f'refit_days={refit_days}\n')
        f.write(f'  rebal_days={rebal_days}  commission_bps={commission_bps}\n')
        f.write(f'  hard-vs-soft aggregate per-ticker corr: '
                f'mean={mean_c:.4f}  median={med_c:.4f}\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        f.write('Full bt stats:\n')
        f.write(stats.to_string() + '\n')
    print(f'Saved {stats_path}')


def _default_data_dir() -> str:
    """Resolve `apps/notebook/data/stooq_us_long/` relative to this
    file's repo. Falls back to env `STOOQ_US_LONG_DIR`."""
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
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    p.add_argument('--no-parallel', dest='parallel', action='store_false')
    p.add_argument('--n-workers', type=int, default=None)
    p.add_argument('--cache-dir', default=_SHARED_CACHE_DIR)
    args = p.parse_args()
    run(**vars(args))
