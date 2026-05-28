"""ss-follow CLI — single-config backtest entrypoint.

The pre-registered walk-forward eval lives in
``scripts/run_walkforward.py``; this CLI is a thin wrapper for
single-shot smoke tests / debugging.
"""

from __future__ import annotations

import argparse

import pandas as pd

from follow.data import build_eligible_disclosures
from follow.backtest import run_backtest


def main() -> None:
    p = argparse.ArgumentParser('ss-follow')
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('backtest', help='Single-config backtest.')
    b.add_argument('--stooq-dir', required=True)
    b.add_argument('--hold-days', type=int, default=60)
    b.add_argument('--top-k', type=int, default=25)
    b.add_argument('--filter', choices=['recency', 'frequency'], default='recency')
    b.add_argument('--start', default='2014-01-01')
    b.add_argument('--end', default=None)
    b.add_argument('--all-members', action='store_true',
                   help='Disable leadership filter (baseline arm).')
    b.add_argument('--commission-bps', type=float, default=10.0)

    args = p.parse_args()
    if args.cmd == 'backtest':
        panel = build_eligible_disclosures(
            stooq_dir=args.stooq_dir,
            leadership_only=not args.all_members,
            start=args.start,
            end=args.end,
        )
        print('drop stats:', panel.drop_stats)
        res = run_backtest(
            panel,
            hold_days=args.hold_days,
            top_k=args.top_k,
            filter_mode=args.filter,
            commission_bps=args.commission_bps,
        )
        nr = res.daily_returns
        ann_sharpe = (nr.mean() / max(nr.std(ddof=0), 1e-12)) * (252 ** 0.5)
        print(f'days: {len(nr)}  ann sharpe (net): {ann_sharpe:+.3f}  '
              f'cum: {(1+nr).prod()-1:+.3f}  max DD est: {_max_dd(nr):+.3f}')


def _max_dd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


if __name__ == '__main__':
    main()
