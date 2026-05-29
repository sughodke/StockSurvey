"""CLI: ss-e2e {train,eval}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from e2e_portfolio.data import load_close, load_macro_panel, prepare_panel
from e2e_portfolio.eval import (
    FOLDS, OUT_DIR, baseline_streams, pool_and_report, run_fold,
)
from e2e_portfolio.model import Hparams
from e2e_portfolio.train import TrainConfig


def cmd_train(args):
    close = load_close()
    macro = load_macro_panel(close.index)
    full = prepare_panel(close, macro_panel=macro)
    print(f'panel: {len(full.dates)} samples '
          f'({full.dates[0].date()} -> {full.dates[-1].date()})')

    cfg = TrainConfig(n_steps=args.n_steps, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      seed=args.seed)
    hp = Hparams()
    fold = FOLDS[args.fold - 1]
    run_fold(full, close, fold, cfg, hp, save_prefix=args.save_prefix)


def cmd_eval(args):
    close = load_close()
    macro = load_macro_panel(close.index)
    full = prepare_panel(close, macro_panel=macro)

    cfg = TrainConfig(n_steps=args.n_steps, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay,
                      seed=args.seed)
    hp = Hparams()
    per_fold = []
    for fold_cfg in FOLDS:
        res = run_fold(full, close, fold_cfg, cfg, hp, save_prefix=args.save_prefix)
        per_fold.append(res)

    summary = pool_and_report(per_fold, close, save_prefix=args.save_prefix)
    out_json = OUT_DIR / f'{args.save_prefix}-results.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSaved {out_json}')
    print(json.dumps({
        'pooled_n': summary['pooled_n'],
        'pooled_sharpe_ann': summary['pooled_sharpe_ann'],
        'pooled_max_dd': summary['pooled_max_dd'],
        'vs_dca': summary['baseline_comparisons']['dca'],
        'vs_ew': summary['baseline_comparisons']['ew'],
        'vs_det_2leg': summary['baseline_comparisons']['deterministic_2leg'],
        'vs_learned_2leg': summary['baseline_comparisons']['learned_2leg'],
    }, indent=2))


def main():
    p = argparse.ArgumentParser('ss-e2e')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train one fold')
    p_train.add_argument('--fold', type=int, required=True, choices=[1, 2, 3])
    p_train.add_argument('--n-steps', type=int, default=5000)
    p_train.add_argument('--batch-size', type=int, default=128)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--weight-decay', type=float, default=1e-4)
    p_train.add_argument('--seed', type=int, default=0)
    p_train.add_argument('--save-prefix', type=str, default='e2e-portfolio')
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser('eval', help='Walk-forward eval all 3 folds + pool')
    p_eval.add_argument('--pooled', action='store_true', help='alias for full eval')
    p_eval.add_argument('--n-steps', type=int, default=5000)
    p_eval.add_argument('--batch-size', type=int, default=128)
    p_eval.add_argument('--lr', type=float, default=1e-3)
    p_eval.add_argument('--weight-decay', type=float, default=1e-4)
    p_eval.add_argument('--seed', type=int, default=0)
    p_eval.add_argument('--save-prefix', type=str, default='e2e-portfolio')
    p_eval.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
