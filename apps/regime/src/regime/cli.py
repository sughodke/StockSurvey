"""Command-line entry point.

Two subcommands:

    regime train --data-dir ./Nasdaq3347
        Run an Optuna walk-forward search using vectorbt backtests.
        Optimizes 7 discrete hyperparameters (lookback, n_tail, top_n,
        divergence, scale subsets) over rolling train/val windows.
        Reports per-window best params + Sharpes; prints the highest-
        validation-Sharpe window at the end.

    regime live --params model.json --dry-run
        Load a checkpoint, fetch recent OHLC from Alpaca, compute target
        weights, and submit (or print) the trades needed to reach them.

For the gradient-descent alternative on continuous params, call
`regime.research.optimize_adam.train()` directly — no CLI shim.
"""

from __future__ import annotations

import argparse
import warnings

from regime.persist import save_checkpoint_from_window
from regime.trainer import print_summary, train
from ss_indicators import corwin_schultz_spread
from ss_loaders import load_price_matrix

DEFAULT_KILLSWITCH: str = '~/.regime-killswitch'

warnings.filterwarnings('ignore')


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--data-dir', required=True,
                   help='Directory of per-ticker OHLC CSVs (e.g. ./Nasdaq3347).')
    p.add_argument('--n-trials', type=int, default=50,
                   help='Optuna trials per walk-forward window.')
    p.add_argument('--metric', default='sharpe',
                   choices=['sharpe', 'cagr', 'max_drawdown', 'total_return'],
                   help='Metric maximized by the search. Default sharpe.')
    p.add_argument('--rebalance-days', type=int, default=20,
                   help='Rebalance every N trading days. Default 20.')
    p.add_argument('--commission-bps', type=float, default=10.0,
                   help='Per-side commission cost in basis points. Default 10.')
    p.add_argument('--max-spread', type=float, default=0.02,
                   help='Live-trading sanity gate (fraction). Recorded into the '
                        'checkpoint so `regime live` rejects names whose spread '
                        'exceeds this on the rebalance bar. Training itself does '
                        'not filter on spread — costs are charged via vbt fees.')
    p.add_argument('--train-years', type=int, default=5,
                   help='Training window length, in years. Default 5.')
    p.add_argument('--val-years', type=int, default=3,
                   help='Validation window length, in years. Default 3.')
    p.add_argument('--step-years', type=int, default=2,
                   help='Step forward between walk-forward windows. Default 2.')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-31')
    p.add_argument('--min-history', type=int, default=504,
                   help='Min trading days of history per ticker. Default 504.')
    p.add_argument('--save-params',
                   help='Write the highest-val-Sharpe window to this JSON path '
                        '(consumable by `regime live`).')


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a checkpoint JSON written by `regime train --save-params`.')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and print trades without submitting (default).')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders. Default is dry-run.')
    p.add_argument('--max-position', type=float, default=0.25,
                   help='Per-name weight cap (0-1). Default 0.25.')
    p.add_argument('--max-data-age-days', type=int, default=3,
                   help='Abort if latest bar is older than this many days. Default 3.')
    p.add_argument('--killswitch', default=DEFAULT_KILLSWITCH,
                   help=f'Abort if this file exists. Default {DEFAULT_KILLSWITCH}.')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='regime',
        description='Regime-divergence strategy: walk-forward search + live trade.')
    sub = p.add_subparsers(dest='cmd', required=True)
    _add_train_args(sub.add_parser(
        'train', help='Walk-forward Optuna search via vectorbt.'))
    _add_live_args(sub.add_parser(
        'live', help='Score the universe and rebalance via Alpaca.'))
    return p


def _run_train(args: argparse.Namespace) -> None:
    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=args.min_history,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)
    median_spread = spread_df.iloc[-1].median()
    print(f'Median latest-bar spread across universe: {median_spread:.2%} '
          f'(passed into vbt as per-side cost = commission + spread/2)')

    result = train(
        prices, spread_df,
        n_trials=args.n_trials,
        rebalance_days=args.rebalance_days,
        metric=args.metric,
        commission_bps=args.commission_bps,
        train_years=args.train_years,
        val_years=args.val_years,
        step_years=args.step_years,
    )
    print_summary(result)

    if args.save_params and result.windows:
        path = save_checkpoint_from_window(
            args.save_params, result.best_window,
            universe=list(prices.columns),
            rebal_days=args.rebalance_days,
            max_spread=args.max_spread,
            commission_bps=args.commission_bps,
        )
        print(f'\nSaved best-window checkpoint to {path}')


def _run_live(args: argparse.Namespace) -> None:
    from regime.live import format_run, run_live

    result = run_live(
        args.params,
        dry_run=args.dry_run,
        max_position=args.max_position,
        max_data_age_days=args.max_data_age_days,
        killswitch_path=args.killswitch,
    )
    print(format_run(result))


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == 'train':
        _run_train(args)
    elif args.cmd == 'live':
        _run_live(args)


if __name__ == '__main__':
    main()
