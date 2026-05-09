"""Optuna study over `replay.reconstruct_indicators` for the MLP decoder.

Searches the MLP-decoder hyperparameter space (window size, hidden
width, depth, training steps) to maximize the mean R² across the
three reconstruction targets (price, RSI, MACD).

Usage:
    uv run ss-replay-optuna AAPL --start 2013-01-29 --end 2025-12-11 \\
        --n-trials 40

The CWT-slice mechanics (scales, lookback, causal z-norm) are inherited
from `replay.reconstruct_indicators`; this module only varies the
decoder side.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import optuna

from replay import reconstruct_indicators
from ss_features import DEFAULT_STOOQ_DIR, load_prices
from ss_wavelets import ALL_SCALES


def _objective(trial: optuna.Trial, prices: np.ndarray, *,
               lookback: int, rsi_n: int,
               macd_fast: int, macd_slow: int, macd_signal: int) -> float:
    K = trial.suggest_categorical(
        'window_cols', [1, 4, 8, 16, 32, 64, 96, 128])
    hidden = trial.suggest_categorical('mlp_hidden', [32, 64, 128, 256, 512])
    layers = trial.suggest_int('mlp_layers', 1, 4)
    steps = trial.suggest_categorical('mlp_steps', [500, 1000, 2000, 4000])

    t0 = time.time()
    (_gt, _recon, stats, n_feat,
     _val_gt, _val_recon, _val_stats) = reconstruct_indicators(
        prices,
        scales=list(ALL_SCALES),
        lookback=lookback,
        rsi_n=rsi_n,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        window_cols=K,
        decoder='mlp',
        mlp_hidden=hidden,
        mlp_layers=layers,
        mlp_steps=steps,
    )
    dt = time.time() - t0

    r2s = [stats[k]['r2'] for k in ('price', 'rsi', 'macd')]
    if any(math.isnan(v) for v in r2s):
        return float('-inf')

    trial.set_user_attr('price_r2', stats['price']['r2'])
    trial.set_user_attr('rsi_r2', stats['rsi']['r2'])
    trial.set_user_attr('macd_r2', stats['macd']['r2'])
    trial.set_user_attr('n_features', n_feat)
    trial.set_user_attr('wall_s', dt)
    return float(np.mean(r2s))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Optuna study over replay.reconstruct_indicators '
                    '(MLP decoder). Maximizes mean R² across price, RSI, '
                    'MACD.')
    parser.add_argument('ticker')
    parser.add_argument('--stooq-dir', default=None,
                        help=f'Stooq archive root. Default: {DEFAULT_STOOQ_DIR}.')
    parser.add_argument('--kaggle-dir', default=None,
                        help='Use a Nasdaq3347-style CSV matrix instead.')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--n-trials', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lookback', type=int, default=252)
    parser.add_argument('--rsi-n', type=int, default=7)
    parser.add_argument('--macd-fast', type=int, default=12)
    parser.add_argument('--macd-slow', type=int, default=26)
    parser.add_argument('--macd-signal', type=int, default=9)
    args = parser.parse_args()

    series = load_prices(
        args.ticker,
        stooq_dir=args.stooq_dir,
        kaggle_dir=args.kaggle_dir,
        start=args.start,
        end=args.end,
    )
    prices = series.values.astype(np.float64)
    print(f'{args.ticker}: {len(prices)} bars from '
          f'{series.index[0].date()} to {series.index[-1].date()}\n')

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        ua = trial.user_attrs
        if not ua:
            return
        print(f'  trial {trial.number:>3} '
              f'value={trial.value:6.3f}  '
              f'K={trial.params["window_cols"]:>3} '
              f'h={trial.params["mlp_hidden"]:>3} '
              f'L={trial.params["mlp_layers"]} '
              f'steps={trial.params["mlp_steps"]:>4}  '
              f'price={ua["price_r2"]:6.3f} '
              f'rsi={ua["rsi_r2"]:6.3f} '
              f'macd={ua["macd_r2"]:6.3f}  '
              f'{ua["wall_s"]:5.1f}s')

    study.optimize(
        lambda t: _objective(
            t, prices,
            lookback=args.lookback, rsi_n=args.rsi_n,
            macd_fast=args.macd_fast, macd_slow=args.macd_slow,
            macd_signal=args.macd_signal),
        n_trials=args.n_trials,
        callbacks=[cb],
        show_progress_bar=False,
    )

    bt = study.best_trial
    print('\nbest trial:')
    print(f'  value (mean R²): {bt.value:.4f}')
    print(f'  params:          {bt.params}')
    print(f'  price R²: {bt.user_attrs["price_r2"]:.4f}  '
          f'rsi R²: {bt.user_attrs["rsi_r2"]:.4f}  '
          f'macd R²: {bt.user_attrs["macd_r2"]:.4f}')
    print(f'  n_features: {bt.user_attrs["n_features"]}  '
          f'wall: {bt.user_attrs["wall_s"]:.1f}s\n')

    print('| trial | mean R² | K | hidden | layers | steps | '
          'price R² | rsi R² | macd R² | wall (s) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    rows = sorted(
        (t for t in study.trials if t.value is not None),
        key=lambda t: -(t.value or float('-inf')),
    )
    for t in rows:
        p = t.params
        ua = t.user_attrs
        print(f'| {t.number} | {t.value:.3f} | {p["window_cols"]} | '
              f'{p["mlp_hidden"]} | {p["mlp_layers"]} | {p["mlp_steps"]} | '
              f'{ua["price_r2"]:.3f} | {ua["rsi_r2"]:.3f} | '
              f'{ua["macd_r2"]:.3f} | {ua["wall_s"]:.1f} |')


if __name__ == '__main__':
    main()
