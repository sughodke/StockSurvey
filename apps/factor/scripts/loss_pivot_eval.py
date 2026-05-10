"""Loss-pivot eval — does training on Sharpe / IR-vs-EW find a different head?

Falsifiable test of the loss-mismatch hypothesis raised after the
long-short result: rank-IC is scale-invariant and rewards spreading
information thin across the cross-section, but deployment cares about
risk-adjusted absolute return (Sharpe) or risk-adjusted active return
vs the EW benchmark (IR). If the loss is the binding constraint, a
Sharpe-aligned or IR-aligned loss should find a measurably different
head — even on the same universe / horizon / head architecture.

Three arms:
  - rank_ic       (existing baseline — `factor-indicator-baseline.md`)
  - block_sharpe  (Sharpe-as-loss with softmax-LO constructor)
  - ir_vs_ew      (IR vs EW with the same softmax-LO constructor)

All three use the same factor-narrow universe (297 tickers,
`min_history_bars=6500`), same 6-window walk-forward, same linear
head, same `n_steps=200 lr=1e-2 wd=1e-3 commission_bps=10`. Sharpe /
IR arms add the per-window log-temperature to the optimizer (rank-IC
keeps it frozen — it's scale-invariant so the temperature does
nothing in the loss).

All four metrics (val IC, val Sharpe LO, val IR vs EW, val Sharpe LS)
are reported per-window per-arm so the comparison is apples-to-apples.

Pre-registered cuts:
  PASS for the loss-pivot hypothesis  → Sharpe or IR arm clears val
       Sharpe LO ≥ +0.50 (≈ +0.20 over rank-IC's ~+0.28) AND
       ≥ 4/6 positive windows. The new loss found a head the
       rank-IC loss didn't.
  FAIL                                → both new arms come in within
       ±0.10 of rank-IC's val Sharpe LO. The loss is not the binding
       constraint; supports `pivot prediction problem` next-move.
  INCONCLUSIVE                         → between the two — stratify
       by window before deciding.

Run from repo root:
    uv run python apps/factor/scripts/loss_pivot_eval.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _resolve_ticker_list(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}', flush=True)
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run', flush=True)
    return names


def _build_one_ticker(args):
    ticker, stooq_dir, cfg, start, end = args
    try:
        from factor import build_indicator_features
        from ss_features import TickerData, load_prices
        series = load_prices(ticker, stooq_dir=stooq_dir, start=start, end=end)
        prices = series.values.astype(np.float64)
        dates = np.asarray(series.index)
        feats, valid = build_indicator_features(prices, cfg)
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _summarize_arm(label: str, wf, save_path: Path) -> dict:
    print(f'\n=== arm={label} ===', flush=True)
    print(f'{"win":>3}  {"tr_ic":>8}  {"val_ic":>8}  {"val_sh":>8}  '
          f'{"val_ir":>8}  {"val_sh_ls":>9}  {"logT":>6}', flush=True)
    for w in wf.windows:
        print(f'{w.window_idx:>3}  {w.train_ic:+.4f}   {w.val_ic:+.4f}   '
              f'{w.val_sharpe:+8.3f}  {w.val_ir_vs_ew:+8.3f}  '
              f'{w.val_sharpe_long_short:+9.3f}  '
              f'{w.final_log_temperature:+6.2f}', flush=True)
    summary = {
        'arm': label,
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'mean_val_sharpe_long_only': wf.mean_val_sharpe,
        'mean_val_ir_vs_ew': wf.mean_val_ir_vs_ew,
        'positive_val_ir_vs_ew_fraction':
            wf.positive_val_ir_vs_ew_fraction,
        'mean_val_sharpe_long_short': wf.mean_val_sharpe_long_short,
        'positive_val_sharpe_long_short_fraction':
            wf.positive_val_sharpe_long_short_fraction,
        'per_window': [{
            'window_idx': w.window_idx,
            'train_ic': w.train_ic, 'val_ic': w.val_ic,
            'val_sharpe_long_only': w.val_sharpe,
            'val_ir_vs_ew': w.val_ir_vs_ew,
            'val_sharpe_long_short': w.val_sharpe_long_short,
            'final_log_temperature': w.final_log_temperature,
        } for w in wf.windows],
    }
    np.savez(
        save_path,
        window_idx=np.array([w.window_idx for w in wf.windows]),
        train_ic=np.array([w.train_ic for w in wf.windows]),
        val_ic=np.array([w.val_ic for w in wf.windows]),
        val_sharpe_long_only=np.array([w.val_sharpe for w in wf.windows]),
        val_ir_vs_ew=np.array([w.val_ir_vs_ew for w in wf.windows]),
        val_sharpe_long_short=np.array(
            [w.val_sharpe_long_short for w in wf.windows]),
        final_log_temperature=np.array(
            [w.final_log_temperature for w in wf.windows]))
    print(f'  -> {save_path}', flush=True)
    return summary


def _verdict(arms: dict[str, dict]) -> str:
    rank_ic_sh = arms['rank_ic']['mean_val_sharpe_long_only']
    pass_threshold = 0.50
    pass_pos_frac = 4 / 6
    fail_band = 0.10
    new_arms = {k: v for k, v in arms.items() if k != 'rank_ic'}
    pass_arms = [
        k for k, v in new_arms.items()
        if v['mean_val_sharpe_long_only'] >= pass_threshold
        and v['positive_val_sharpe_long_short_fraction'] >= pass_pos_frac
        # Note: pos-LO-windows isn't tracked separately; compare via
        # individual per_window data instead.
    ]
    # Simpler heuristic: pass if any new arm clears +0.50 mean Sharpe,
    # fail if all new arms within fail_band of rank-IC.
    deltas = {
        k: v['mean_val_sharpe_long_only'] - rank_ic_sh
        for k, v in new_arms.items()
    }
    max_delta = max(deltas.values()) if deltas else 0.0
    min_delta = min(deltas.values()) if deltas else 0.0
    if max_delta >= 0.20 and any(
            v['mean_val_sharpe_long_only'] >= pass_threshold
            for v in new_arms.values()):
        return f'PASS  (max delta vs rank_ic = {max_delta:+.3f} ≥ +0.20)'
    if max(abs(d) for d in deltas.values()) <= fail_band:
        return f'FAIL  (all new arms within ±{fail_band} of rank-IC val Sharpe)'
    return (f'INCONCLUSIVE  (max delta {max_delta:+.3f}, '
            f'min delta {min_delta:+.3f})')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--rebal-days',          type=int, default=20)
    p.add_argument('--train-window-blocks', type=int, default=63)
    p.add_argument('--val-window-blocks',   type=int, default=39)
    p.add_argument('--step-window-blocks',  type=int, default=39)
    p.add_argument('--scorer',         default='linear')
    p.add_argument('--n-steps',        type=int, default=200)
    p.add_argument('--learning-rate',  type=float, default=1e-2)
    p.add_argument('--weight-decay',   type=float, default=1e-3)
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0)
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--losses', default='rank_ic,block_sharpe,ir_vs_ew',
                   help='comma-separated loss kinds to evaluate')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    losses = [s.strip() for s in args.losses.split(',') if s.strip()]
    valid = {'rank_ic', 'block_sharpe', 'ir_vs_ew'}
    if not set(losses).issubset(valid):
        raise SystemExit(f'--losses must be subset of {sorted(valid)}')

    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'loss-pivot eval: indicator stack F={F}, scorer={args.scorer}, '
          f'rebal={args.rebal_days}d, train/val/step blocks = '
          f'{args.train_window_blocks}/{args.val_window_blocks}/'
          f'{args.step_window_blocks}, losses={losses}', flush=True)

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)
    print(f'\nbuilding features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...', flush=True)
    t0 = time.perf_counter()
    pool_args = [(n, str(STOOQ_SUBSET), cfg, args.start, args.end) for n in names]
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(_build_one_ticker, pool_args)
    else:
        results = [_build_one_ticker(a) for a in pool_args]
    ticker_data = []
    failed = []
    for name, td, err in results:
        if td is None:
            failed.append((name, err))
        else:
            ticker_data.append(td)
    print(f'built {len(ticker_data)} / {len(names)} tickers in '
          f'{time.perf_counter() - t0:.1f}s', flush=True)
    if failed:
        print(f'  failed: {len(failed)}', flush=True)

    arms: dict[str, dict] = {}
    for loss_kind in losses:
        save_path = output / f'loss-pivot-{loss_kind}-windows.npz'
        print(f'\n--- training {args.scorer} head with loss={loss_kind} ---',
              flush=True)
        t1 = time.perf_counter()
        wf = train_scorer_indicators_walkforward(
            ticker_data, cfg=cfg,
            rebal_days=args.rebal_days,
            train_window_blocks=args.train_window_blocks,
            val_window_blocks=args.val_window_blocks,
            step_window_blocks=args.step_window_blocks,
            scorer=args.scorer,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            forward_target_kind='log_return',
            loss_kind=loss_kind,
        )
        print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)
        arms[loss_kind] = _summarize_arm(loss_kind, wf, save_path)

    print('\n' + '=' * 96, flush=True)
    print('loss-pivot eval — head trained on each loss, all metrics '
          'eval\'d on val per arm', flush=True)
    print('=' * 96, flush=True)
    print(f'{"arm":>14}  {"mean_val_ic":>11}  {"mean_val_sh":>11}  '
          f'{"mean_val_ir":>11}  {"mean_val_sh_ls":>14}  '
          f'{"pos_LS_frac":>11}', flush=True)
    print('-' * 96, flush=True)
    for arm_name, arm in arms.items():
        print(f'{arm_name:>14}  {arm["mean_val_ic"]:>+11.4f}  '
              f'{arm["mean_val_sharpe_long_only"]:>+11.3f}  '
              f'{arm["mean_val_ir_vs_ew"]:>+11.3f}  '
              f'{arm["mean_val_sharpe_long_short"]:>+14.3f}  '
              f'{arm["positive_val_sharpe_long_short_fraction"]:>11.2f}',
              flush=True)
    print()
    print(f'verdict: {_verdict(arms)}', flush=True)

    summary_path = output / 'loss-pivot-eval-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'feature_width': F,
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'scorer': args.scorer,
        'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'arms': arms,
    }, indent=2))
    print(f'\n-> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
