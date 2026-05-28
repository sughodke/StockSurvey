"""CLI for the crypto-and-carry backtest."""
from __future__ import annotations

import argparse

import pandas as pd

from cnc.backtest import block_sharpe, max_drawdown, pos_quarter_fraction, run_carry
from cnc.data import build_panels


def main() -> None:
    p = argparse.ArgumentParser(prog='ss-cnc')
    sub = p.add_subparsers(dest='cmd', required=True)

    bp = sub.add_parser('backtest', help='Run a single-config cash-and-carry backtest.')
    bp.add_argument('--start', default='2024-01-01')
    bp.add_argument('--end', default=None)
    bp.add_argument('--top-universe', type=int, default=20)
    bp.add_argument('--top-k', type=int, default=5)
    bp.add_argument('--rebal-days', type=int, default=1)
    bp.add_argument('--trailing-window', type=int, default=30)
    bp.add_argument('--sign', choices=['positive', 'both'], default='positive')
    bp.add_argument('--friction-bps', type=float, default=15.0,
                    help='Per-leg friction bps; charged on both legs.')

    args = p.parse_args()

    if args.cmd == 'backtest':
        panels = build_panels(
            start_date=args.start, end_date=args.end,
            top_universe=args.top_universe,
        )
        print(f'Panels: {len(panels.funding_daily)} days × '
              f'{len(panels.coins)} coins '
              f'({panels.start_date.date()} → {panels.end_date.date()})')
        res = run_carry(
            panels.funding_daily,
            top_k=args.top_k,
            rebal_days=args.rebal_days,
            trailing_window=args.trailing_window,
            sign=args.sign,
            rebal_friction_bps_per_leg=args.friction_bps,
        )
        sr = block_sharpe(res.daily_return, periods_per_year=365)
        gsr = block_sharpe(res.gross_return, periods_per_year=365)
        mdd = max_drawdown(res.daily_return)
        pq = pos_quarter_fraction(res.daily_return)
        total = res.daily_return.sum()
        gtotal = res.gross_return.sum()
        ftotal = res.friction_cost.sum()
        print(f'gross Sharpe (ann): {gsr:+.3f}  | net Sharpe (ann): {sr:+.3f}')
        print(f'gross return total: {gtotal*100:+.2f}%  | friction: '
              f'{ftotal*100:.2f}%  | net: {total*100:+.2f}%')
        print(f'max DD net: {mdd*100:.2f}%  | pos-quarter frac: {pq:.2f}')


if __name__ == '__main__':
    main()
