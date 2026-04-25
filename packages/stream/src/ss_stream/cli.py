"""`ss-stream` CLI: ingest a Stooq archive, summarize an event store."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        prog='ss-stream',
        description='Stooq event-stream ingestion and point-in-time universe.')
    sub = p.add_subparsers(dest='cmd', required=True)

    ing = sub.add_parser(
        'ingest', help='ingest a Stooq archive into a parquet event store')
    ing.add_argument('--src', type=Path, required=True,
                     help='archive root containing daily/<country>/...')
    ing.add_argument('--dst', type=Path, required=True,
                     help='output directory for instruments.parquet + bars/')
    ing.add_argument('--no-etfs', action='store_true',
                     help='skip <exchange> etfs subtrees (stocks only)')

    info = sub.add_parser('info', help='summarize an existing event store')
    info.add_argument('--path', type=Path, required=True,
                      help='directory written by `ss-stream ingest`')
    info.add_argument('--at', type=str, default=None,
                      help='ISO date — count tickers active on that date')

    args = p.parse_args()

    if args.cmd == 'ingest':
        from ss_stream.ingest import ingest_stooq
        ingest_stooq(args.src, args.dst, include_etfs=not args.no_etfs)
        return

    if args.cmd == 'info':
        from ss_stream.universe import Universe
        u = Universe(args.path)
        ins = u.instruments
        print(f'instruments:       {len(ins):,}')
        print(f'listing range:     {ins.listing_date.min().date()} -> '
              f'{ins.last_seen_date.max().date()}')
        print(f'median bars/inst:  {int(ins.n_bars.median()):,}')
        print()
        counts = ins.groupby(['country', 'exchange', 'asset_class']).size()
        print(counts.rename('count').to_frame().to_string())
        if args.at is not None:
            active = u.active_at(args.at)
            print()
            print(f'active on {args.at}: {len(active):,}')


if __name__ == '__main__':
    main()
