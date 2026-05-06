"""CLI for the relational research app.

Two subcommands:

    ss-relational head-to-head --data-dir ./StooqData --top-n 10 \\
        --start 2013-01-29 --end 2025-12-11
        Side-by-side bt backtest of `weights_regime` vs
        `weights_excess_regime` on the same Phase-2 universe.

    ss-relational live --params model.json --dry-run
        Load a `RelationalCheckpoint`, fetch recent OHLC from Alpaca,
        compute target weights via the chosen relational strategy,
        and submit (or print) the trades needed to reach them.

Risk rails on `live` mirror `regime live`: kill-switch file, max bar
age, per-name cap, dry-run by default.
"""

from __future__ import annotations

import argparse


DEFAULT_KILLSWITCH: str = '~/.relational-killswitch'


def _add_h2h_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--data-dir', required=True, help='Stooq archive root.')
    p.add_argument('--top-n', type=int, default=10)
    p.add_argument('--lookback', type=int, default=120)
    p.add_argument('--n-tail', type=int, default=20)
    p.add_argument('--divergence', default='kl',
                   choices=['kl', 'js', 'cosine', 'l2'])
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--output-dir', default='Output')


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a JSON checkpoint written by '
                        '`relational.persist.save_checkpoint`.')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and print trades without submitting (default).')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders. Default is dry-run.')
    p.add_argument('--max-position', type=float, default=0.25,
                   help='Per-name weight cap (0-1). Default 0.25.')
    p.add_argument('--max-data-age-days', type=int, default=3,
                   help='Abort if latest bar is older than this many days. '
                        'Default 3.')
    p.add_argument('--killswitch', default=DEFAULT_KILLSWITCH,
                   help=f'Abort if this file exists. Default {DEFAULT_KILLSWITCH}.')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='ss-relational',
        description='Relational-CWT scoring research + live trading.')
    sub = parser.add_subparsers(dest='subcmd', required=True)

    _add_h2h_args(sub.add_parser(
        'head-to-head',
        help='Side-by-side bt backtest of weights_regime vs '
             'weights_excess_regime on the same universe + dates.'))

    _add_live_args(sub.add_parser(
        'live',
        help='Score the universe and rebalance via Alpaca.'))

    args = parser.parse_args()
    if args.subcmd == 'head-to-head':
        from relational.research.backtest_sector_excess import run
        run(
            data_dir=args.data_dir,
            top_n=args.top_n,
            lookback=args.lookback,
            n_tail=args.n_tail,
            divergence=args.divergence,
            start=args.start, end=args.end,
            rebal_days=args.rebal_days,
            commission_bps=args.commission_bps,
            output_dir=args.output_dir)
    elif args.subcmd == 'live':
        from relational.live import format_run, run_live
        result = run_live(
            args.params,
            dry_run=args.dry_run,
            max_position=args.max_position,
            max_data_age_days=args.max_data_age_days,
            killswitch_path=args.killswitch,
        )
        print(format_run(result))


if __name__ == '__main__':
    main()
