"""Phase 1 walk-forward eval. Thin shim — see `cfr.scripts_walkforward.run_walkforward`.

Run from repo root:
    uv run python apps/cfr/scripts/run_walkforward.py
    uv run ss-cfr walkforward --output Output/cfr-phase1.json
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default='apps/notebook/data/stooq_us_long/manifest.json')
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--n-training-passes', type=int, default=1)
    p.add_argument('--menu', default='phase1', choices=['phase1', 'phase2a'])
    p.add_argument('--output', default='Output/cfr-walkforward-summary.json')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

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
        menu=args.menu,
        output=Path(args.output),
        seed=args.seed,
    )


if __name__ == '__main__':
    raise SystemExit(main())
