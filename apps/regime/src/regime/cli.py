"""Command-line entry point.

Two subcommands:

    regime train --data-dir ./Nasdaq3347 --save-params model.json
        Run the JAX/Adam optimizer over historical CSVs and (optionally)
        write a JSON checkpoint that `regime live` can consume.

    regime live --params model.json --dry-run
        Load a checkpoint, fetch recent OHLC from Alpaca, compute target
        weights, and submit (or print) the trades needed to reach them.
"""

from __future__ import annotations

import argparse
import warnings

from regime.persist import save_checkpoint
from regime.reporting import plot_training, print_results
from regime.trainer import train
from ss_indicators import corwin_schultz_spread
from ss_loaders import load_price_matrix
from ss_wavelets import ALL_SCALES

DEFAULT_KILLSWITCH: str = '~/.regime-killswitch'

warnings.filterwarnings('ignore')


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--data-dir', required=True,
                   help='Directory of per-ticker OHLC CSVs (e.g. ./Nasdaq3347).')
    p.add_argument('--lookback', type=int, default=120,
                   help='Causal normalization + historical window length, in days.')
    p.add_argument('--n-tail', type=int, default=20,
                   help='Recent-window length (must be < --lookback).')
    p.add_argument('--rebal-days', type=int, default=20,
                   help='Rebalance every N trading days.')
    p.add_argument('--commission-bps', type=float, default=10,
                   help='Per-side commission cost in basis points.')
    p.add_argument('--max-spread', type=float, default=0.02,
                   help='Drop tickers whose Corwin-Schultz spread exceeds this fraction.')
    p.add_argument('--n-steps', type=int, default=500,
                   help='Number of Adam steps.')
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--train-frac', type=float, default=0.7,
                   help='Fraction of blocks used for training; remainder is held-out val.')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-31')
    p.add_argument('--save', action='store_true',
                   help='Write training plot to Output/regime-training.png.')
    p.add_argument('--save-params',
                   help='Write trained checkpoint JSON to this path '
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
        description='Differentiable regime-divergence strategy: train and trade.')
    sub = p.add_subparsers(dest='cmd', required=True)
    _add_train_args(sub.add_parser('train', help='Optimize scale weights via JAX Adam.'))
    _add_live_args(sub.add_parser('live', help='Score the universe and rebalance via Alpaca.'))
    return p


def _run_train(args: argparse.Namespace) -> None:
    prices, highs, lows = load_price_matrix(
        args.data_dir, min_history=504,
        start_date=args.start, end_date=args.end)

    print('Computing Corwin-Schultz spreads...')
    spread_df = corwin_schultz_spread(highs, lows)
    liquid_pct = (spread_df.iloc[-1] <= args.max_spread).mean()
    print(f'Liquid tickers (spread <= {args.max_spread:.1%}): {liquid_pct:.1%}')

    result = train(
        prices, spread_df,
        scales=ALL_SCALES,
        lookback=args.lookback,
        n_tail=args.n_tail,
        rebal_days=args.rebal_days,
        commission_bps=args.commission_bps,
        max_spread=args.max_spread,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        train_frac=args.train_frac,
    )

    print_results(result)

    if args.save_params:
        path = save_checkpoint(
            args.save_params, result,
            universe=list(prices.columns),
            lookback=args.lookback,
            n_tail=args.n_tail,
            rebal_days=args.rebal_days,
            max_spread=args.max_spread,
            commission_bps=args.commission_bps,
        )
        print(f'Saved checkpoint to {path}')

    plot_training(
        result,
        save_path='Output/regime-training.png' if args.save else None,
    )


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
