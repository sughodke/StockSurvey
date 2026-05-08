"""Idea B variant — DWT keep-LL compression of analog-kNN fingerprints.

Three-arm head-to-head over the same Phase-2 universe / dates / commission
convention used by `idea_b_analog_knn.py`:

  - analog        — baseline analog k-NN (full-resolution `(S, w)` fingerprint)
  - analog-dwt-L1 — fingerprint compressed via 2D Haar DWT keep-LL, 1 level
  - analog-dwt-L2 — same, 2 levels (4× tighter than baseline along each axis)

Hypothesis: at fp_window=21 and 8 scales the raw fingerprint is 168-dim
L2-normalised. kNN distances in 168-dim unit-norm space concentrate
(curse of dimensionality), and the high-frequency components of the
`(S, w)` tile carry CWT noise unrelated to forward-return predictability.
2D Haar keep-LL acts as a low-pass denoiser: L=1 gives a 44-dim
fingerprint (8/2 × 21/2+1 = 4 × 11), L=2 gives 12-dim (2 × 6). If the
denoising hypothesis holds, the compressed arms should show better
out-of-sample Sharpe at the same backtest config.

Same causality + autocorrelation guards as the baseline (the kNN code
is unchanged; only the fingerprint encoder differs).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bt
import matplotlib.pyplot as plt

from ss_features import Compression
from ss_loaders import load_stooq_matrix
from ss_portfolio.bt_helpers import build_strategy

from relational.analog_knn import weights_regime_analog
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
    start: str = '2013-01-29',
    end: str = '2025-12-11',
    rebal_days: int = 20,
    commission_bps: float = 10.0,
    output_dir: str = 'Output',
) -> None:
    print(f'Loading Stooq prices from {data_dir} ...')
    prices, _highs, _lows, _vol = load_stooq_matrix(
        data_dir, min_history=lookback + 30,
        start_date=start, end_date=end,
        tickers=list(PHASE2_TICKERS))
    print(f'  loaded {prices.shape[0]} dates x {prices.shape[1]} tickers '
          f'({list(prices.columns)})')

    scales = [5, 7, 10, 12, 21, 26, 50, 90]
    print(f'  scales={scales} ({len(scales)} total), '
          f'lookback={lookback}, top_n={top_n}, fp_window={fp_window}')
    print(f'  k_neighbors={k_neighbors}, forward_horizon={forward_horizon}, '
          f'min_sep_days={min_sep_days}, pool_mode={pool_mode}')

    arms = [
        ('analog',        None,
         len(scales) * fp_window),
        ('analog-dwt-L1', Compression(kind='dwt', levels=1, wavelet='haar',
                                      pad_mode='periodization'),
         _ll_dim(len(scales), fp_window, 1)),
        ('analog-dwt-L2', Compression(kind='dwt', levels=2, wavelet='haar',
                                      pad_mode='periodization'),
         _ll_dim(len(scales), fp_window, 2)),
    ]
    for name, comp, dim in arms:
        levels = 0 if comp is None else comp.levels
        print(f'    {name:16s}  compression L={levels}  fp_dim={dim}')

    weights_by_arm: dict[str, object] = {}
    for name, comp, _dim in arms:
        print(f'\n[{name}] Computing analog-kNN weights ...')
        w = weights_regime_analog(
            prices, lookback=lookback, top_n=top_n,
            scales=scales, fp_window=fp_window,
            k_neighbors=k_neighbors, forward_horizon=forward_horizon,
            min_sep_days=min_sep_days, pool_mode=pool_mode,
            compression=comp)
        weights_by_arm[name] = w

    print('\nRunning bt backtests...')
    strategies = [
        build_strategy(name, prices, w,
                       rebal_days=rebal_days, commission_bps=commission_bps)
        for name, w in weights_by_arm.items()
    ]
    result = bt.run(*strategies)
    result.display()

    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    result.plot(ax=ax)
    ax.set_title(
        f'Idea B — analog k-NN vs DWT-LL compressed fingerprints — Phase-2 '
        f'({start} → {end}, top-{top_n}, rebal={rebal_days}d, '
        f'k={k_neighbors}, h={forward_horizon}, w={fp_window})')
    fig.tight_layout()
    fig_path = out / 'relational-idea-b-analog-knn-dwt-equity.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fig_path}')

    stats_path = out / 'relational-idea-b-analog-knn-dwt-stats.txt'
    with open(stats_path, 'w') as f:
        f.write(str(result.stats))
    print(f'Saved {stats_path}')


def _ll_dim(n_scales: int, w: int, levels: int) -> int:
    """Per-tile fp_dim after L levels of 2D Haar DWT keep-LL with
    pywt periodization mode."""
    s_p = n_scales
    w_p = w
    for _ in range(levels):
        s_p = -(-s_p // 2)
        w_p = -(-w_p // 2)
    return s_p * w_p


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
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')
    args = p.parse_args()
    run(**vars(args))
