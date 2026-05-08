"""CLI for the `lie` research app.

Two subcommands:

    ss-lie weights --params <ckpt.json> --data-dir ./StooqData
        Load a `LieCheckpoint`, fetch the trailing OHLC for its universe
        from a Stooq archive, run `inference.target_weights`, and print
        the resulting weight series for the latest bar.

    ss-lie rank --data-dir ./StooqData --tickers AAPL,MSFT,... \\
                --lookback 60 [--start ...] [--end ...]
        Compute the daily effective-rank time series for the given
        universe + lookback over the requested date range. Prints CSV to
        stdout (date, n_valid, eff_rank). Useful for eyeballing the
        symmetry-breaking signal before wiring it into a checkpoint.

Live trading (Alpaca-broker integration) is intentionally not in v1 -- the
`weights` subcommand is the research path, and `live.py` will land alongside
the first non-HRP strategy that needs it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def _add_weights_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a JSON checkpoint written by '
                        '`lie.persist.save_checkpoint`.')
    p.add_argument('--data-dir', required=True,
                   help='Stooq archive root (the directory containing '
                        '`daily/`).')
    p.add_argument('--end', default=None,
                   help='ISO date for the rebalance bar. Default: latest '
                        'available in the archive.')
    p.add_argument('--bar-buffer-days', type=int, default=10,
                   help='Extra calendar-day buffer on top of `lookback` '
                        'when fetching bars (covers weekends/holidays). '
                        'Default 10.')


def _add_rank_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--data-dir', required=True, help='Stooq archive root.')
    p.add_argument('--tickers', required=True,
                   help='Comma-separated ticker list, e.g. AAPL,MSFT,GOOGL.')
    p.add_argument('--lookback', type=int, default=60,
                   help='Trailing-window size in bars for the correlation '
                        'matrix. Default 60.')
    p.add_argument('--start', default=None,
                   help='ISO start date (inclusive). Default: archive start.')
    p.add_argument('--end', default=None,
                   help='ISO end date (inclusive). Default: archive end.')


def _run_weights(args: argparse.Namespace) -> None:
    from ss_loaders import load_stooq_matrix
    from lie.inference import target_weights
    from lie.persist import load_checkpoint

    cp = load_checkpoint(args.params)

    # Fetch trailing bars + buffer. Pull a generous slice (3x lookback in
    # calendar days) to ensure `lookback + 1` trading bars survive weekends
    # and any per-name leading NaN.
    closes, highs, lows, _ = load_stooq_matrix(
        args.data_dir,
        tickers=cp.universe,
        end_date=args.end,
        min_history=cp.lookback + 1)

    # Reindex to checkpoint universe order; drop names absent from archive.
    missing = [t for t in cp.universe if t not in closes.columns]
    if missing:
        print(f'WARNING: {len(missing)} ticker(s) absent from archive: '
              f'{missing}', file=sys.stderr)
    present = [t for t in cp.universe if t in closes.columns]
    if len(present) < 2:
        print(f'ERROR: only {len(present)} ticker(s) found; need at least 2.',
              file=sys.stderr)
        sys.exit(2)

    # We need `target_weights` to see the checkpoint universe verbatim, so
    # rebuild the full panel with NaN columns for the missing names.
    closes = closes.reindex(columns=cp.universe)
    highs = highs.reindex(columns=cp.universe)
    lows = lows.reindex(columns=cp.universe)

    # Trim to the trailing bars actually needed (lookback + 1 + buffer).
    needed = cp.lookback + 1 + args.bar_buffer_days
    closes = closes.iloc[-needed:]
    highs = highs.iloc[-needed:]
    lows = lows.iloc[-needed:]

    weights = target_weights(closes, highs, lows, cp)

    nz = weights[weights > 0].sort_values(ascending=False)
    print(f'# rebalance bar: {weights.name}')
    print(f'# nonzero names: {len(nz)} / {len(weights)}')
    print(f'# gross exposure: {float(weights.sum()):.4f}')
    for ticker, w in nz.items():
        print(f'{ticker}\t{w:.4f}')


def _run_rank(args: argparse.Namespace) -> None:
    from ss_loaders import load_stooq_matrix
    from lie.symmetry_rank import effective_rank
    from lie.correlation_network import log_returns

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if len(tickers) < 2:
        print('ERROR: need at least 2 tickers for a correlation matrix.',
              file=sys.stderr)
        sys.exit(2)

    closes, _, _, _ = load_stooq_matrix(
        args.data_dir,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        min_history=args.lookback + 1)

    closes = closes.reindex(columns=tickers)
    panel = closes.to_numpy()
    if panel.shape[0] < args.lookback + 1:
        print(f'ERROR: only {panel.shape[0]} bars in range; need at least '
              f'{args.lookback + 1}.', file=sys.stderr)
        sys.exit(2)

    rets = log_returns(panel)
    n = rets.shape[1]
    print('date,n_valid,eff_rank')
    for t in range(args.lookback - 1, rets.shape[0]):
        window = rets[t - args.lookback + 1: t + 1]
        valid = ~np.isnan(window).any(axis=0)
        if int(valid.sum()) < 2:
            print(f'{closes.index[t + 1].date()},{int(valid.sum())},nan')
            continue
        sub = window[:, valid]
        sub = sub - sub.mean(axis=0, keepdims=True)
        std = sub.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        sub = sub / std
        c = (sub.T @ sub) / (sub.shape[0] - 1)
        c = np.clip(c, -1.0, 1.0)
        er = effective_rank(c)
        print(f'{closes.index[t + 1].date()},{int(valid.sum())},{er:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='ss-lie',
        description='Lie-group / hierarchical-network strategies.')
    sub = parser.add_subparsers(dest='subcmd', required=True)

    _add_weights_args(sub.add_parser(
        'weights',
        help='Compute target weights for the latest bar from a checkpoint.'))

    _add_rank_args(sub.add_parser(
        'rank',
        help='Stream the effective-rank time series for a universe.'))

    args = parser.parse_args()
    if args.subcmd == 'weights':
        _run_weights(args)
    elif args.subcmd == 'rank':
        _run_rank(args)


if __name__ == '__main__':
    main()
