"""Head-to-head: regime-velocity vs farthest vs baseline regime.

Tests the **vector arithmetic / regime-velocity** interpretation of
the CWT fingerprint space — that the *direction and magnitude of
recent fingerprint motion* is a richer signal than either the
snapshot fingerprint (idea C / farthest) or discretized cluster
transitions.

Four strategies on the same 312-ticker `apps/notebook/data/stooq_us_long`
universe and rebal cadence:

  * `baseline`           — `weights_regime` (per-stock CWT-power-divergence
                           top-N; the canonical numpy reference).
  * `farthest`           — snapshot dispersion: top-N by L2 distance from
                           the cross-sectional fingerprint centroid at
                           date `t`. This is the idea-C control.
  * `velocity-magnitude` — top-N by ||v[t, i]||, the (a) variant of the
                           regime-velocity hypothesis. "Motion regardless
                           of direction."
  * `axis-alignment`     — top-N by max-|projection onto top-K SVD axes|
                           fit on the **first 252 post-lookback days
                           only** (look-ahead-free). "Directed motion
                           along a stable axis."

Diagnostics printed alongside the leaderboard:
  - Singular-value spectrum (top-K explained variance fraction).
  - Per-axis ±-extremum spot-check (3 stocks loading +/- max on each
    of the top axes during the training window).
  - Velocity-magnitude time series — should be roughly stationary; if
    it drifts, fingerprints aren't well-z-normed.

Outputs
-------
  Output/relational-velocity-equity.png
  Output/relational-velocity-stats.txt
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
from ss_portfolio.bt_helpers import build_strategy
from ss_portfolio import (
    apply_nan_mask, select_top_n_matrix, weights_regime as _weights_regime_baseline,
)

from relational.regime_velocity import (
    axis_alignment_scores,
    extract_fingerprints,
    load_or_compute_cwt,
    velocity_magnitude_scores,
    weights_axis_alignment,
    weights_velocity_magnitude,
)

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# Inline farthest-from-centroid baseline (the relational.farthest
# module post-dates this worktree's branch). Same semantics as the
# canonical implementation: per-date L2 distance to the
# finite-fingerprint centroid, NaN-masked, top-N.
# ---------------------------------------------------------------------

def _centroid_distance_scores(
    prices: pd.DataFrame, *,
    lookback: int, scales: list[int], fp_window: int = 21,
    cache_dir=None,
) -> np.ndarray:
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    fps_eval = fps[lookback:]
    n_eval, n_tickers, _ = fps_eval.shape
    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)
    for t in range(n_eval):
        row = fps_eval[t]
        finite = np.isfinite(row).all(axis=1)
        if finite.sum() < 2:
            continue
        centroid = row[finite].mean(axis=0)
        d = np.linalg.norm(row - centroid[None, :], axis=-1)
        scores[t, finite] = d[finite].astype(np.float32, copy=False)
    return scores


def _weights_farthest(
    prices: pd.DataFrame, *,
    lookback: int, top_n: int, scales: list[int], fp_window: int = 21,
    cache_dir=None,
) -> pd.DataFrame:
    scores = _centroid_distance_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, cache_dir=cache_dir)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


# ---------------------------------------------------------------------
# Weight-mask helper (kept here — local to the velocity diagnostic).
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Diagnostics: spectrum + axis spot-check + magnitude stationarity.
# ---------------------------------------------------------------------

def _print_axis_spotcheck(
    prices: pd.DataFrame, *,
    lookback: int, scales: list[int], fp_window: int, w_delta: int,
    n_axes: int, train_window_days: int,
    cache_dir,
) -> dict:
    """Run axis-alignment scoring with diagnostics, then print a per-axis
    +/- extremum spot-check over the training window.

    Returns the diagnostics dict (axes / singular values / explained
    variance) so the caller can serialize.
    """
    print(f'\n[axis fit] training window: first {train_window_days} '
          f'post-lookback days, n_axes={n_axes}')
    _scores, diag = axis_alignment_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window,
        w_delta=w_delta, n_axes=n_axes,
        train_window_days=train_window_days, cache_dir=cache_dir,
        return_diagnostics=True)

    sv = diag['singular_values']
    expl = diag['explained_variance_fraction']
    print(f'[axis fit] top-{n_axes} singular values: '
          f'{[f"{float(s):.3f}" for s in sv]}')
    print(f'[axis fit] explained variance fraction (top-{n_axes}): '
          f'{expl:.4f}')

    # Axis spot-check: project per-(date, ticker) velocity in the
    # training window onto each axis, then take per-ticker mean projection
    # and report top 3 / bottom 3 tickers per axis. Mean over training
    # window is a stable summary even for noisy individual days.
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    # velocity: v[t] = fps[t] - fps[t-w_delta], for t >= w_delta
    velocities = np.full_like(fps, np.nan)
    velocities[w_delta:] = fps[w_delta:] - fps[:-w_delta]

    train_start = diag['train_start']
    train_end = diag['train_end']
    v_train = velocities[train_start:train_end]   # (T_train, N, D)
    axes = diag['axes']                            # (K, D)

    # Mean per-ticker projection, masking NaN dates per ticker.
    proj_train = np.einsum('tnd,kd->tnk', v_train, axes)  # (T, N, K)
    finite = np.isfinite(v_train).all(axis=-1)             # (T, N)
    mask3 = finite[..., None].astype(np.float32)
    safe = np.where(np.isnan(proj_train), 0.0, proj_train)
    sums = (safe * mask3).sum(axis=0)                      # (N, K)
    counts = mask3.sum(axis=0)                             # (N, K)
    mean_proj = np.where(counts > 0, sums / np.maximum(counts, 1.0),
                         np.nan)

    tickers = list(prices.columns)
    print(f'\n[axis spot-check] per-axis ±-extremum tickers '
          f'(mean projection over training window):')
    for k in range(n_axes):
        col = mean_proj[:, k]
        valid_idx = np.where(np.isfinite(col))[0]
        if len(valid_idx) < 6:
            print(f'  axis {k+1}: too few finite tickers')
            continue
        order = valid_idx[np.argsort(col[valid_idx])]
        bot = order[:3]
        top = order[-3:][::-1]
        top_strs = [f'{tickers[i]}({col[i]:+.3f})' for i in top]
        bot_strs = [f'{tickers[i]}({col[i]:+.3f})' for i in bot]
        print(f'  axis {k+1}:  +{", ".join(top_strs)}   '
              f' -{", ".join(bot_strs)}')

    return diag


def _print_magnitude_stationarity(
    prices: pd.DataFrame, *,
    lookback: int, scales: list[int], fp_window: int, w_delta: int,
    cache_dir,
) -> None:
    """Cross-sectional median ||v|| by year — should be roughly flat."""
    scores = velocity_magnitude_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, w_delta=w_delta, cache_dir=cache_dir)
    eval_index = prices.index[lookback:]
    df = pd.DataFrame(scores, index=eval_index, columns=prices.columns)
    by_year = df.median(axis=1).groupby(df.index.year).median()
    print(f'\n[magnitude stationarity] median ||v|| per year '
          f'(cross-sectional median, then yearly median):')
    for year, val in by_year.items():
        bar = '#' * max(1, int(val * 50))
        print(f'  {year}: {val:.4f}  {bar}')


# ---------------------------------------------------------------------
# Top-level run.
# ---------------------------------------------------------------------

def run(
    *, data_dir: str,
    top_n: int = 20,
    lookback: int = 120,
    fp_window: int = 21,
    w_delta: int = 20,
    n_axes: int = 5,
    train_window_days: int = 252,
    n_tail: int = 20,
    divergence: str = 'kl',
    start: str | None = None,
    end: str | None = None,
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    cache_dir: str | None = '/Users/sidghodke/Code/StockSurvey/.scalogram-cache',
    output_dir: str = 'Output',
) -> pd.DataFrame:
    """Runs all 4 backtests + diagnostics, prints leaderboard, returns it."""
    print(f'Loading Stooq prices from {data_dir} ...')
    min_history = lookback + max(n_tail, w_delta) + train_window_days + 10
    prices, _, _, _ = load_stooq_matrix(
        data_dir, min_history=min_history,
        start_date=start, end_date=end, tickers=None)
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'(min_history={min_history})')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, n_tail={n_tail}, '
          f'top_n={top_n}, fp_window={fp_window}, w_delta={w_delta}, '
          f'n_axes={n_axes}, train_window_days={train_window_days}')

    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        print(f'  cache_dir={cache_dir_path}')

    # --- diagnostics first (also primes the CWT cache) ----------------
    diag = _print_axis_spotcheck(
        prices, lookback=lookback, scales=scales, fp_window=fp_window,
        w_delta=w_delta, n_axes=n_axes,
        train_window_days=train_window_days, cache_dir=cache_dir)
    _print_magnitude_stationarity(
        prices, lookback=lookback, scales=scales, fp_window=fp_window,
        w_delta=w_delta, cache_dir=cache_dir)

    # --- weight builders ----------------------------------------------
    print('\n[weights 1/4] baseline regime (weights_regime, KL divergence) ...')
    w_baseline = _weights_regime_baseline(
        prices, lookback=lookback, n_tail=n_tail, top_n=top_n,
        scales=scales, divergence=divergence)
    w_baseline = _mask_weights_to_active(w_baseline, prices)

    print('[weights 2/4] farthest-from-centroid (snapshot dispersion) ...')
    w_farthest = _weights_farthest(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window, cache_dir=cache_dir)
    w_farthest = _mask_weights_to_active(w_farthest, prices)

    print('[weights 3/4] velocity-magnitude (||v||) ...')
    w_velmag = weights_velocity_magnitude(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window, w_delta=w_delta,
        cache_dir=cache_dir)
    w_velmag = _mask_weights_to_active(w_velmag, prices)

    print('[weights 4/4] axis-alignment (max-|proj| on K SVD axes) ...')
    w_axis = weights_axis_alignment(
        prices, lookback=lookback, top_n=top_n,
        scales=scales, fp_window=fp_window, w_delta=w_delta,
        n_axes=n_axes, train_window_days=train_window_days,
        cache_dir=cache_dir)
    w_axis = _mask_weights_to_active(w_axis, prices)

    # --- backtests ----------------------------------------------------
    print('\n[bt] running 4 backtests ...')
    backtests = []
    for label, w in [
        ('baseline',           w_baseline),
        ('farthest',           w_farthest),
        ('velocity-magnitude', w_velmag),
        ('axis-alignment',     w_axis),
    ]:
        backtests.append(build_strategy(
            label, prices, w,
            rebal_days=rebal_days, commission_bps=commission_bps,
            drop_empty=True, safe_prices=True))

    result = bt.run(*backtests)

    stats = result.stats
    sharpe_row = stats.loc['daily_sharpe'].astype(float)
    order = sharpe_row.sort_values(ascending=False).index.tolist()
    headline = ['daily_sharpe', 'cagr', 'max_drawdown', 'calmar',
                'daily_vol', 'total_return']
    leaderboard = stats.loc[headline, order].T.copy()

    print('\n' + '=' * 100)
    print('Regime-velocity leaderboard — sorted by daily Sharpe')
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
    result.plot(ax=ax)
    ax.set_title(
        f'Regime-velocity diagnostic — '
        f'{prices.index[0].date()} → {prices.index[-1].date()}, '
        f'{prices.shape[1]} tickers, top-{top_n}, '
        f'rebal={rebal_days}d, w_delta={w_delta}, '
        f'commission={commission_bps}bps')
    fig.tight_layout()
    fig_path = out / 'relational-velocity-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-velocity-stats.txt'
    with open(stats_path, 'w') as f:
        f.write('Regime-velocity leaderboard — sorted by daily Sharpe\n')
        f.write(f'  universe: {prices.shape[1]} tickers\n')
        f.write(f'  date range: {prices.index[0].date()} → '
                f'{prices.index[-1].date()}\n')
        f.write(f'  top_n={top_n}  lookback={lookback}  n_tail={n_tail}  '
                f'fp_window={fp_window}  w_delta={w_delta}  '
                f'n_axes={n_axes}  train_window_days={train_window_days}\n')
        f.write(f'  rebal_days={rebal_days}  '
                f'commission_bps={commission_bps}\n')
        f.write(f'  scales={scales}\n')
        f.write('=' * 100 + '\n')
        f.write(leaderboard.to_string() + '\n\n')
        sv = diag['singular_values']
        f.write(f'top-{n_axes} singular values: '
                f'{[float(s) for s in sv]}\n')
        f.write(f'explained variance fraction (top-{n_axes}): '
                f'{diag["explained_variance_fraction"]:.4f}\n\n')
        f.write('Full bt stats:\n')
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
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--w-delta', type=int, default=20)
    p.add_argument('--n-axes', type=int, default=5)
    p.add_argument('--train-window-days', type=int, default=252)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--divergence', default='kl')
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument(
        '--cache-dir',
        default='/Users/sidghodke/Code/StockSurvey/.scalogram-cache')
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
