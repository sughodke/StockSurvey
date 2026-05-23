"""`ss-vol` CLI.

Subcommands:
  live   — run one vol-v3 rebalance pass against Alpaca paper.
           Defaults to --dry-run (the four-rail discipline from
           regime/live and relational/live; --live to actually
           submit orders).

Research drivers stay under apps/vol/scripts/ — see the module-level
docstring of each script for invocation.
"""
from __future__ import annotations

import argparse
import sys


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a VolCheckpoint JSON (e.g. Output/vol-v3.json)')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and log only; do not submit orders (default)')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders (off by default — overrides --dry-run)')
    p.add_argument('--killswitch', default=None,
                   help='Override the kill-switch file path '
                        '(default: ~/.vol-killswitch)')
    p.add_argument('--max-total-vega', type=float, default=5000.0,
                   help='Portfolio-level cap on total |net vega| in USD '
                        '(default $5,000)')
    p.add_argument('--max-data-age-days', type=int, default=3,
                   help='Abort if latest underlying bar is older than this')


def _run_live(args: argparse.Namespace) -> int:
    # Lazy import — only the `live` path needs Alpaca configured.
    from vol.live import DEFAULT_KILLSWITCH, format_run, run_live
    try:
        result = run_live(
            args.params,
            dry_run=args.dry_run,
            max_total_vega_usd=args.max_total_vega,
            max_data_age_days=args.max_data_age_days,
            killswitch_path=args.killswitch or DEFAULT_KILLSWITCH,
        )
    except NotImplementedError as e:
        # The MVP scaffold returns these for the chain-query layer.
        # Surface the message cleanly rather than a stack trace.
        print(f'ss-vol live: NOT YET IMPLEMENTED\n  {e}', file=sys.stderr)
        return 2
    print(format_run(result))
    if result.aborted_reason:
        return 3 if 'kill-switch' in result.aborted_reason else 4
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ss-vol')
    sub = parser.add_subparsers(dest='cmd')
    live = sub.add_parser('live', help='Run one vol-v3 rebalance against Alpaca paper')
    _add_live_args(live)
    args = parser.parse_args(argv)
    if args.cmd == 'live':
        return _run_live(args)
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
