"""Walk-forward A/B: Ricker vs polar Morlet for the analog k-NN scorer.

Mirrors `idea_b_analog_knn_dwt_walkforward.py`'s segmentation logic
(canonical Phase-2 train 2013-2020 / val 2021-2025), but the A/B axis
is the wavelet kernel rather than fingerprint compression.

The two arms share every other knob — same scales, same lookback, same
fp_window, same k_neighbors / forward_horizon / min_sep_days /
pool_mode / rebal_days / commission. The only thing that changes is
how the per-(date, ticker) fingerprint is built:

  * `analog-ricker` — legacy single-channel real Ricker coefficients,
    flattened over the trailing `fp_window` bars. fp_dim = S * w.
  * `analog-morlet` — polar Morlet + Gaussian bundle from
    `ss_features.causal_polar_morlet_matrix`: 4 channels per scale
    `(|c|, cos(arg), sin(arg), g)` flattened over the same trailing
    `fp_window` bars. fp_dim = 4 * S * w.

The Phase-2 21-ticker pool gives this enough headroom that the 4×
fingerprint dim doesn't dominate compute (BLAS matmul is the
bottleneck, scales linearly with fp_dim). On the canonical
configuration the val Sharpe of the Ricker baseline is 1.146; the
question this script answers is whether the Morlet bundle clears that
bar.

Uses `analog_knn_scores_fast(n_workers=...)` rather than
`weights_regime_analog` (which still routes through the serial slow
path that the Phase-2 8-arm Modal A/B used) so a local Phase-2 sweep
finishes in minutes — `n_workers=8` matches the 8-core Intel Mac the
repo lives on. Picks differ from the slow path at FP-noise level
(~1e-5) per the docstring, indistinguishable in Sharpe terms.

Output: `Output/relational-idea-b-analog-knn-morlet-walkforward-{stats.txt,
equity.png}`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_portfolio.bt_helpers import build_strategy

from relational.analog_knn import analog_knn_scores_fast
from relational.research.idea_b_analog_knn_dwt_walkforward import (
    TRAIN_END, TRAIN_START, VAL_END, VAL_START, segment_stats,
)
from relational.sectors import PHASE2_TICKERS

warnings.filterwarnings('ignore')


def run(
    *, data_dir: str,
    top_n: int = 10,
    lookback: int = 120,
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    n_workers: int | None = None,
    output_dir: str = 'Output',
) -> None:
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f'Loading Stooq prices from {data_dir} ...', flush=True)
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir, min_history=lookback + 30,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers',
          flush=True)

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales}, lookback={lookback}, top_n={top_n}, '
          f'fp_window={fp_window}, k={k_neighbors}, h={forward_horizon}, '
          f'n_workers={n_workers}', flush=True)
    print(f'  train: {TRAIN_START} → {TRAIN_END}', flush=True)
    print(f'  val:   {VAL_START} → {VAL_END}', flush=True)

    arms = [
        ('analog-ricker', 'ricker'),
        ('analog-morlet', 'morlet'),
    ]

    weights_by_arm: dict[str, pd.DataFrame] = {}
    for name, wavelet in arms:
        print(f'\n[{name}] computing weights (wavelet={wavelet}) ...',
              flush=True)
        scores = analog_knn_scores_fast(
            prices, lookback=lookback, scales=scales,
            fp_window=fp_window,
            k_neighbors=k_neighbors, forward_horizon=forward_horizon,
            min_sep_days=min_sep_days, pool_mode=pool_mode,
            wavelet=wavelet, n_workers=n_workers)
        scores = apply_nan_mask(scores, prices.values, lookback)
        weights = select_top_n_matrix(scores, top_n, ascending=False)
        weights_by_arm[name] = pd.DataFrame(
            weights,
            index=prices.index[lookback:],
            columns=prices.columns)
        print(f'[{name}] done; nonzero rows={int((weights.sum(axis=1) > 0).sum())}',
              flush=True)

    print('\nRunning bt backtests over the full period ...', flush=True)
    strategies = [
        build_strategy(name, prices, w,
                       rebal_days=rebal_days, commission_bps=commission_bps)
        for name, w in weights_by_arm.items()
    ]
    result = bt.run(*strategies)

    eq_panel = result.prices
    rows: list[dict] = []
    for name, _wavelet in arms:
        equity = eq_panel[name]
        for window_name, w_start, w_end in (('full', TRAIN_START, VAL_END),
                                            ('train', TRAIN_START, TRAIN_END),
                                            ('val', VAL_START, VAL_END)):
            stats = segment_stats(equity, w_start, w_end)
            stats.update({'arm': name, 'window': window_name,
                          'start': w_start, 'end': w_end})
            rows.append(stats)
    summary = pd.DataFrame(rows)[
        ['arm', 'window', 'start', 'end', 'n_bars',
         'total_return', 'cagr', 'sharpe', 'sortino', 'max_dd']
    ]

    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print('\n=== Segmented walk-forward stats ===', flush=True)
    print(summary.to_string(index=False), flush=True)

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    stats_path = out / 'relational-idea-b-analog-knn-morlet-walkforward-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(summary.to_string(index=False))
        f.write('\n\n--- delta (morlet minus ricker) ---\n')
        for window in ('full', 'train', 'val'):
            base = summary[(summary.arm == 'analog-ricker') &
                           (summary.window == window)].iloc[0]
            comp = summary[(summary.arm == 'analog-morlet') &
                           (summary.window == window)].iloc[0]
            f.write(
                f'  {window:6s}  '
                f'Δsharpe={comp.sharpe - base.sharpe:+.4f}  '
                f'Δcagr={comp.cagr - base.cagr:+.4f}  '
                f'Δret={comp.total_return - base.total_return:+.4f}  '
                f'Δmaxdd={comp.max_dd - base.max_dd:+.4f}\n')
    print(f'\nSaved {stats_path}', flush=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    eq_panel.plot(ax=ax, lw=1.5)
    split_date = pd.Timestamp(VAL_START)
    ax.axvline(split_date, color='k', ls='--', alpha=0.5,
               label=f'train/val split ({VAL_START})')
    ax.set_title(
        f'Idea B walk-forward — analog k-NN ricker vs polar Morlet — Phase-2 '
        f'(top-{top_n}, rebal={rebal_days}d, k={k_neighbors}, '
        f'h={forward_horizon}, w={fp_window})')
    ax.legend()
    fig.tight_layout()
    fig_path = out / 'relational-idea-b-analog-knn-morlet-walkforward-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Saved {fig_path}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--fp-window', type=int, default=21)
    p.add_argument('--k-neighbors', type=int, default=50)
    p.add_argument('--forward-horizon', type=int, default=20)
    p.add_argument('--min-sep-days', type=int, default=21)
    p.add_argument('--pool-mode', default='cross_ticker',
                   choices=['cross_ticker', 'per_ticker'])
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--n-workers', type=int, default=None)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
