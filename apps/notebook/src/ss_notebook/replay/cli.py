"""CLI for `ss-replay`: train one+ ticker decoders, eval zero-shot on a held-out
ticker, save reconstruction figures.

Examples
--------
    # Single-ticker fit, no held-out eval (in-sample expressivity probe):
    uv run ss-replay AAPL --window-cols 64 --include-zscore-stats \\
        --decoder mlp

    # Train on AAPL, zero-shot eval on TSLA:
    uv run ss-replay AAPL --val-ticker TSLA --window-cols 64 \\
        --include-zscore-stats --decoder mlp

    # Multi-ticker train pool, held-out eval on TSLA:
    uv run ss-replay AAPL --train-tickers MSFT,GOOGL,AMZN \\
        --val-ticker TSLA --window-cols 64 --include-zscore-stats \\
        --decoder mlp
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ss_notebook.replay.features import TARGET_NAMES, load_ticker
from ss_notebook.replay.plot import plot_reconstruction
from ss_notebook.replay.reconstruct import fit_and_evaluate
from ss_notebook.scalogram import DEFAULT_STOOQ_DIR
from ss_wavelets import ALL_SCALES


def _git_sha_and_dirty() -> tuple[str, bool]:
    """Short git SHA + dirty-tree flag for the current working dir.
    Returns `('nogit', False)` outside a git repo."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'nogit', False
    try:
        dirty = subprocess.run(
            ['git', 'diff', '--quiet', 'HEAD'],
            stderr=subprocess.DEVNULL).returncode != 0
    except Exception:
        dirty = False
    return sha, dirty


def _split_tickers(value: str | None) -> list[str]:
    """Split a comma-separated ticker list, ignoring empty entries."""
    if not value:
        return []
    return [t.strip() for t in value.split(',') if t.strip()]


def _print_target_stats(label: str, stats: dict[str, dict[str, float]]) -> None:
    for name, s in stats.items():
        print(f'  {label} {name:5s}  R²={s["r2"]:.6f}  '
              f'RMSE={s["rmse"]:.3e}  max|Δ|={s["max_abs"]:.3e}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reconstruct RSI / MACD / price from per-bar CWT '
                    'slices via a trained decoder; plot vs ground truth '
                    'with R² + error.')
    parser.add_argument('ticker',
                        help='Primary train ticker. Always saves a figure '
                             'at <output-dir>/<ticker>-replay.png.')
    parser.add_argument('--train-tickers', default=None,
                        help='Comma-separated additional train tickers '
                             '(e.g. MSFT,GOOGL,AMZN). Their features are '
                             'pooled with `ticker` into a single decoder '
                             'fit. No figures are saved for them.')
    parser.add_argument('--val-ticker', default=None,
                        help='Optional held-out ticker. The pooled-train '
                             'decoder is applied zero-shot here. Reports '
                             'val stats and saves '
                             '<output-dir>/<val-ticker>-replay-zeroshot-'
                             'from-<ticker>.png.')
    parser.add_argument('--stooq-dir', default=None,
                        help=f'Stooq archive root. Default: {DEFAULT_STOOQ_DIR}.')
    parser.add_argument('--kaggle-dir', default=None,
                        help='Slice columns from a Nasdaq3347-style CSV '
                             'matrix instead of using Stooq.')
    parser.add_argument('--start', default=None,
                        help='Trim every loaded ticker to start at this date.')
    parser.add_argument('--end', default=None,
                        help='Trim every loaded ticker to end at this date.')
    parser.add_argument('--lookback', type=int, default=252,
                        help='Causal z-norm window for the CWT (default 252, '
                             'matches the regime trainer\'s raw-close path).')
    parser.add_argument('--window-cols', type=int, default=1,
                        help='Trailing-window size K — features at date t are '
                             'the (coeffs, power) columns over [t-K+1 ... t]. '
                             'K=1 (default) is single-column; K large is '
                             'closer to the full CWT matrix and approaches the '
                             'invertibility ceiling. Cost is O(K) extra '
                             'features per bar.')
    parser.add_argument('--include-zscore-stats', action='store_true',
                        help='Append the causal rolling mean and std (the same '
                             'rolling z-norm stats `causal_cwt` strips out '
                             'before convolution) as 2 extra features per '
                             'date. Restores level information that the CWT '
                             'discards — turns price R² from ~0 into '
                             'near-perfect.')
    parser.add_argument('--decoder',
                        choices=['linear', 'mlp', 'cnn'], default='linear',
                        help='`linear` = OLS via `np.linalg.lstsq`. `mlp` = '
                             'tiny JAX MLP (Adam) — captures the RSI sigmoid '
                             'nonlinearity that OLS can\'t. `cnn` = 1-D '
                             'Conv1D over the trailing-K window with '
                             'shared weights across lags — the right '
                             'inductive bias for fixed linear filters like '
                             'Wilder/EMA. Requires --window-cols > 1.')
    parser.add_argument('--mlp-hidden', type=int, default=128)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--mlp-steps', type=int, default=2000)
    parser.add_argument('--cnn-hidden', type=int, default=64,
                        help='CNN channel width per conv layer. Default 64.')
    parser.add_argument('--cnn-kernel', type=int, default=5,
                        help='CNN kernel size in lags. Default 5.')
    parser.add_argument('--cnn-layers', type=int, default=2,
                        help='Number of stacked Conv1D + ReLU blocks. '
                             'Default 2.')
    parser.add_argument('--cnn-steps', type=int, default=2000,
                        help='Adam steps for the CNN fit. Default 2000.')
    parser.add_argument('--rsi-n', type=int, default=7)
    parser.add_argument('--macd-fast', type=int, default=12)
    parser.add_argument('--macd-slow', type=int, default=26)
    parser.add_argument('--macd-signal', type=int, default=9)
    parser.add_argument('--output-dir', default='Output',
                        help='Where reconstruction figures are saved. '
                             'matplotlib never opens an interactive window.')
    parser.add_argument('--targets', default=','.join(TARGET_NAMES),
                        help='Comma-separated subset of '
                             f'{",".join(TARGET_NAMES)}. Each target = one '
                             'extra decoder fit, so dropping unused ones '
                             'is a real compute lever (each fit is a full '
                             'Adam pass).')
    args = parser.parse_args()
    targets = tuple(_split_tickers(args.targets))
    unknown = set(targets) - set(TARGET_NAMES)
    if unknown:
        parser.error(f'unknown targets {sorted(unknown)!r}; '
                     f'valid: {TARGET_NAMES}')
    if not targets:
        parser.error('--targets must list at least one target')

    scales = list(ALL_SCALES)
    load_kwargs = dict(
        stooq_dir=args.stooq_dir, kaggle_dir=args.kaggle_dir,
        start=args.start, end=args.end,
        scales=scales, lookback=args.lookback,
        window_cols=args.window_cols,
        include_zscore_stats=args.include_zscore_stats,
        decoder=args.decoder,
        rsi_n=args.rsi_n, macd_fast=args.macd_fast,
        macd_slow=args.macd_slow, macd_signal=args.macd_signal,
    )

    primary = load_ticker(args.ticker, **load_kwargs)
    extra_train = [load_ticker(t, **load_kwargs)
                   for t in _split_tickers(args.train_tickers)]
    train_data = [primary, *extra_train]

    val_data = []
    if args.val_ticker is not None:
        val_data = [load_ticker(args.val_ticker, **load_kwargs)]

    cnn_channels_per_lag = 2 * len(scales)
    results, params_per_target = fit_and_evaluate(
        train_data, val_data,
        decoder=args.decoder, cnn_channels_per_lag=cnn_channels_per_lag,
        targets=targets,
        mlp_hidden=args.mlp_hidden, mlp_layers=args.mlp_layers,
        mlp_steps=args.mlp_steps,
        cnn_hidden=args.cnn_hidden, cnn_kernel=args.cnn_kernel,
        cnn_layers=args.cnn_layers, cnn_steps=args.cnn_steps,
    )

    n_features = primary.features.shape[1]
    train_names = ', '.join(d.name for d in train_data)
    print(f'train pool: {train_names}  |  '
          f'{sum(d.valid.sum() for d in train_data)} valid rows  |  '
          f'{len(scales)} scales, lookback={args.lookback}, '
          f'window_cols={args.window_cols}, '
          f'zscore_stats={args.include_zscore_stats}, '
          f'decoder={args.decoder}, targets={",".join(targets)}, '
          f'n_features={n_features}')

    for d in train_data:
        per_target = {n: results[d.name][n]['stats'] for n in targets}
        print(f'\n{d.name} (train, {d.valid.sum()} valid bars):')
        _print_target_stats('train', per_target)
    for d in val_data:
        per_target = {n: results[d.name][n]['stats'] for n in targets}
        print(f'\n{d.name} (zero-shot val, {d.valid.sum()} valid bars):')
        _print_target_stats('val  ', per_target)

    os.makedirs(args.output_dir, exist_ok=True)

    train_tag = primary.name
    if extra_train:
        train_tag += '+' + '+'.join(t.name for t in extra_train)

    primary_targets = {n: primary.targets[n] for n in targets}
    primary_recon = {n: results[primary.name][n]['recon'] for n in targets}
    primary_stats = {n: results[primary.name][n]['stats'] for n in targets}
    fig = plot_reconstruction(
        primary.name, primary.dates, primary_targets, primary_recon,
        primary_stats,
        rsi_n=args.rsi_n, macd_fast=args.macd_fast,
        macd_slow=args.macd_slow, macd_signal=args.macd_signal,
        n_features=n_features, decoder=args.decoder,
        window_cols=args.window_cols,
        include_zscore_stats=args.include_zscore_stats)
    fname = Path(args.output_dir) / f'{primary.name}-replay.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'\nSaved {fname}')

    for d in val_data:
        v_targets = {n: d.targets[n] for n in targets}
        v_recon = {n: results[d.name][n]['recon'] for n in targets}
        v_stats = {n: results[d.name][n]['stats'] for n in targets}
        title = f'{d.name} (zero-shot from {train_tag})'
        val_fig = plot_reconstruction(
            title, d.dates, v_targets, v_recon, v_stats,
            rsi_n=args.rsi_n, macd_fast=args.macd_fast,
            macd_slow=args.macd_slow, macd_signal=args.macd_signal,
            n_features=n_features, decoder=args.decoder,
            window_cols=args.window_cols,
            include_zscore_stats=args.include_zscore_stats)
        val_fname = (Path(args.output_dir) /
                     f'{d.name}-replay-zeroshot-from-{train_tag}.png')
        val_fig.savefig(val_fname, dpi=150)
        plt.close(val_fig)
        print(f'Saved {val_fname}')

    sha, dirty = _git_sha_and_dirty()
    sha_tag = sha + ('-dirty' if dirty else '')
    targets_tag = '+'.join(targets)
    weights_arrays: dict[str, np.ndarray] = {}
    for target_name, params in params_per_target.items():
        for key, arr in params.items():
            weights_arrays[f'{target_name}__{key}'] = arr
    metadata = {
        'train_tickers': [d.name for d in train_data],
        'val_tickers': [d.name for d in val_data],
        'targets': list(targets),
        'decoder': args.decoder,
        'window_cols': args.window_cols,
        'include_zscore_stats': args.include_zscore_stats,
        'lookback': args.lookback,
        'scales': scales,
        'n_features': n_features,
        'rsi_n': args.rsi_n,
        'macd_fast': args.macd_fast,
        'macd_slow': args.macd_slow,
        'macd_signal': args.macd_signal,
        'mlp_hidden': args.mlp_hidden,
        'mlp_layers': args.mlp_layers,
        'mlp_steps': args.mlp_steps,
        'cnn_hidden': args.cnn_hidden,
        'cnn_kernel': args.cnn_kernel,
        'cnn_layers': args.cnn_layers,
        'cnn_steps': args.cnn_steps,
        'start': args.start,
        'end': args.end,
        'git_sha': sha,
        'git_dirty': dirty,
        'train_stats': {
            d.name: {
                t: results[d.name][t]['stats'] for t in targets
            } for d in train_data
        },
        'val_stats': {
            d.name: {
                t: results[d.name][t]['stats'] for t in targets
            } for d in val_data
        },
    }
    weights_arrays['_meta'] = np.array(json.dumps(metadata, default=float))
    weights_fname = (Path(args.output_dir) /
                     f'{train_tag}-{targets_tag}-{args.decoder}-'
                     f'{sha_tag}.npz')
    np.savez(weights_fname, **weights_arrays)
    print(f'Saved {weights_fname}')


if __name__ == '__main__':
    main()
