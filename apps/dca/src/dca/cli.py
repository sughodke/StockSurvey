"""CLI for the DCA app.

Two subcommands:

    ss-dca live --params Output/dca-multiasset.json [--dry-run|--live]
        Load checkpoint, fetch latest bar, evaluate cadence + drift
        gate, optionally submit orders via Alpaca.

    ss-dca status --params Output/dca-multiasset.json
        Same as `live --dry-run --no-fetch` — print current vs target
        weights without contacting the broker. (Stub: currently routes
        to `live --dry-run` since the broker call is the source of
        live prices.)
"""

from __future__ import annotations

import argparse

from dca.live import DEFAULT_KILLSWITCH
from dca.state import DEFAULT_STATE_PATH


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to a JSON checkpoint written by '
                        '`dca.persist.save_checkpoint`.')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and print trades without submitting (default).')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders. Default is dry-run.')
    p.add_argument('--max-position', type=float, default=0.15,
                   help='Per-name weight cap (0-1). Default 0.15 (above '
                        '1/13=0.077 target so the cap is diagnostic-only).')
    p.add_argument('--max-data-age-days', type=int, default=3,
                   help='Abort if latest bar is older than this many days. '
                        'Default 3.')
    p.add_argument('--killswitch', default=DEFAULT_KILLSWITCH,
                   help=f'Abort if this file exists. Default {DEFAULT_KILLSWITCH}.')
    p.add_argument('--state-path', default=DEFAULT_STATE_PATH,
                   help=f'Local state file tracking last rebal date. '
                        f'Default {DEFAULT_STATE_PATH}.')
    p.add_argument('--force-rebal', action='store_true',
                   help='Bypass the cadence + drift gate. Operator override.')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='ss-dca',
        description='Multi-asset DCA — fixed-target rebalancer with risk rails.')
    sub = parser.add_subparsers(dest='subcmd', required=True)

    _add_live_args(sub.add_parser(
        'live',
        help='Score current vs target weights and (optionally) rebalance.'))

    args = parser.parse_args()
    if args.subcmd == 'live':
        from dca.live import format_run, run_live
        result = run_live(
            args.params,
            dry_run=args.dry_run,
            max_position=args.max_position,
            max_data_age_days=args.max_data_age_days,
            killswitch_path=args.killswitch,
            state_path=args.state_path,
            force_rebal=args.force_rebal,
        )
        print(format_run(result))


if __name__ == '__main__':
    main()
