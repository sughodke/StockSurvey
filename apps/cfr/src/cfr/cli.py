"""Command-line entry points.

`ss-cfr smoke`     — Phase 0 sanity run on a small universe.
`ss-cfr walkforward` — Phase 1 multi-window eval.

Both are thin wrappers around `apps/cfr/scripts/{smoke,run_phase1}.py`.
The scripts can also be invoked directly via `uv run python …`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ss-cfr')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sp_smoke = sub.add_parser('smoke', help='Phase 0 sanity test')
    sp_smoke.add_argument('--data-dir', default='./StooqData')
    sp_smoke.add_argument('--n-tickers', type=int, default=30)
    sp_smoke.add_argument('--max-bars', type=int, default=2000)

    sp_wf = sub.add_parser('walkforward', help='Phase 1 walk-forward eval')
    sp_wf.add_argument('--data-dir', default='./StooqData')
    sp_wf.add_argument('--manifest', default='apps/notebook/data/stooq_us_long/manifest.json')
    sp_wf.add_argument('--start', default='2000-01-01')
    sp_wf.add_argument('--end',   default='2025-12-11')
    sp_wf.add_argument('--rebal-days', type=int, default=20)
    sp_wf.add_argument('--top-k', type=int, default=20)
    sp_wf.add_argument('--train-window-days', type=int, default=1260)
    sp_wf.add_argument('--val-window-days',   type=int, default=780)
    sp_wf.add_argument('--step-window-days',  type=int, default=780)
    sp_wf.add_argument('--n-training-passes', type=int, default=1)
    sp_wf.add_argument('--output', default='Output/cfr-walkforward-summary.json')
    sp_wf.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == 'smoke':
        from cfr.scripts_smoke import run_smoke
        return run_smoke(
            data_dir=args.data_dir,
            n_tickers=args.n_tickers,
            max_bars=args.max_bars,
        )
    if args.cmd == 'walkforward':
        from cfr.scripts_walkforward import run_walkforward
        return run_walkforward(
            data_dir=args.data_dir,
            manifest=args.manifest,
            start=args.start, end=args.end,
            rebal_days=args.rebal_days, top_k=args.top_k,
            train_window_days=args.train_window_days,
            val_window_days=args.val_window_days,
            step_window_days=args.step_window_days,
            n_training_passes=args.n_training_passes,
            output=Path(args.output),
            seed=args.seed,
        )
    return 1


if __name__ == '__main__':
    sys.exit(main())
