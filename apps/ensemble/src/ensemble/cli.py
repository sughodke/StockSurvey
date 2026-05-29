"""CLI for the learned 2-leg ensemble.

Three subcommands:

    ss-ensemble train
        Fit weights on a strictly-prior window, write a JSON
        EnsembleCheckpoint.

    ss-ensemble live --params model.json --dry-run
        Dispatch both legs (DCA + vol_v3) with learned scales.
        --live is opt-in; default is dry-run.

    ss-ensemble inspect --params model.json
        Print checkpoint contents (weights, train range, provenance).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


DEFAULT_KILLSWITCH: str = '~/.ensemble-killswitch'


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--dca-close-pkl', required=True,
                   help='Path to the DCA basket close pickle '
                        '(e.g. Output/cfr_phase4d_multiasset_close.pkl).')
    p.add_argument('--vol-v3-npz', required=True,
                   help='Path to the vol_v3 alpha NPZ '
                        '(e.g. Output/vol-v3-dolthub-oos-c200-returns.npz).')
    p.add_argument('--train-start', required=True,
                   help='Training window start, ISO date (YYYY-MM-DD).')
    p.add_argument('--train-end', required=True,
                   help='Training window end, ISO date.')
    p.add_argument('--learner', default='grad_sharpe',
                   choices=['mv_closed_form', 'grad_sharpe'],
                   help='Default grad_sharpe — more interpretable scale.')
    p.add_argument('--dca-checkpoint', default='',
                   help='Path to the DCA basket checkpoint (consumed by live).')
    p.add_argument('--vol-checkpoint', default='',
                   help='Path to the vol_v3 sleeve checkpoint (consumed by live).')
    p.add_argument('--out', required=True,
                   help='Output path for the EnsembleCheckpoint JSON.')
    p.add_argument('--name', default='learned-ensemble-v1')
    p.add_argument('--notes', default='')


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to an EnsembleCheckpoint JSON.')
    p.add_argument('--dry-run', action='store_true', default=True,
                   help='Compute and log without submitting. Default.')
    p.add_argument('--live', dest='dry_run', action='store_false',
                   help='Actually submit orders on both legs.')
    p.add_argument('--skip-dca', action='store_true', default=False)
    p.add_argument('--skip-vol', action='store_true', default=False)
    p.add_argument('--killswitch', default=DEFAULT_KILLSWITCH)


def _add_inspect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--params', required=True,
                   help='Path to an EnsembleCheckpoint JSON.')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='ss-ensemble',
        description='Learned 2-leg DCA + vol_v3 ensemble.')
    sub = parser.add_subparsers(dest='subcmd', required=True)

    _add_train_args(sub.add_parser(
        'train', help='Fit (w_dca, w_vol) on a strictly-prior window.'))
    _add_live_args(sub.add_parser(
        'live', help='Dispatch both legs with learned scales.'))
    _add_inspect_args(sub.add_parser(
        'inspect', help='Print checkpoint contents.'))

    args = parser.parse_args()

    if args.subcmd == 'train':
        from ensemble.persist import save_checkpoint
        from ensemble.train import train_checkpoint

        cp = train_checkpoint(
            dca_close_pkl=args.dca_close_pkl,
            vol_v3_npz=args.vol_v3_npz,
            train_start=args.train_start,
            train_end=args.train_end,
            learner=args.learner,
            dca_checkpoint_path=args.dca_checkpoint,
            vol_checkpoint_path=args.vol_checkpoint,
            name=args.name,
            notes=args.notes,
        )
        out = save_checkpoint(args.out, cp)
        print(f'Wrote {out}')
        print(f'  w_dca = {cp.w_dca:.4f}')
        print(f'  w_vol = {cp.w_vol:.4f}')
        print(f'  train Sharpe ann = {cp.train_sharpe:+.3f}')
        print(f'  in-sample max DD = {cp.in_sample_max_dd*100:+.2f}%')

    elif args.subcmd == 'live':
        from ensemble.live import format_run, run_live
        result = run_live(
            args.params,
            dry_run=args.dry_run,
            killswitch_path=args.killswitch,
            skip_dca=args.skip_dca,
            skip_vol=args.skip_vol,
        )
        print(format_run(result))
        if result.aborted_reason:
            sys.exit(1)

    elif args.subcmd == 'inspect':
        from ensemble.persist import load_checkpoint
        cp = load_checkpoint(args.params)
        print(json.dumps(asdict(cp), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
