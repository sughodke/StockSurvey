"""CLI for the relational research app.

Usage
-----
    ss-relational head-to-head --data-dir ./StooqData --top-n 10 \\
        --start 2013-01-29 --end 2025-12-11

Runs the canonical week-1 experiment: same Phase-2 universe, same
date range, same hyperparameters; one bt backtest with the existing
`weights_regime` baseline and one with `weights_excess_regime`.
Prints a side-by-side stats table and saves equity-comparison PNG.

The actual side-by-side bt loop lives in `research/backtest_sector_excess.py`
(it imports vectorbt + bt, which are research-only deps). The CLI just
shells out to that module.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='ss-relational',
        description='Relational-CWT scoring research harness.')
    sub = parser.add_subparsers(dest='subcmd', required=True)

    h2h = sub.add_parser(
        'head-to-head',
        help='Side-by-side bt backtest of weights_regime vs '
             'weights_excess_regime on the same universe + dates.')
    h2h.add_argument('--data-dir', required=True,
                     help='Stooq archive root.')
    h2h.add_argument('--top-n', type=int, default=10)
    h2h.add_argument('--lookback', type=int, default=120)
    h2h.add_argument('--n-tail', type=int, default=20)
    h2h.add_argument('--divergence', default='kl',
                     choices=['kl', 'js', 'cosine', 'l2'])
    h2h.add_argument('--start', default='2013-01-29')
    h2h.add_argument('--end', default='2025-12-11')
    h2h.add_argument('--rebal-days', type=int, default=20)
    h2h.add_argument('--commission-bps', type=float, default=10.0)
    h2h.add_argument('--output-dir', default='Output')

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


if __name__ == '__main__':
    main()
