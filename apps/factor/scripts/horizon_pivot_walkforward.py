"""Quarterly-horizon pivot — does +0.012 IC ceiling lift at rebal=63d?

Same setup as `forecast_probe_walkforward.py` but `rebal_days=63`
(quarterly, classical alpha horizon) instead of 20d. Walk-forward
window blocks scaled by 20/63 ratio so train/val window *durations*
stay comparable in years:

  rebal=20d, train=63 / val=39 / step=39 blocks  →  ~5y / 3y / 3y
  rebal=63d, train=20 / val=12 / step=12 blocks  →  ~5y / 3y / 3y

Same 297-ticker stooq_us_long universe, same `IndicatorGridConfig`
(74 channels), same linear head, same n_steps=200 / lr=1e-2 / wd=1e-3.
Only the rebal horizon changes.

Read: if `log_return` mean val IC clears +0.020+ at quarterly when 20d
sat at +0.012, the +0.012 ceiling was horizon-bound and the SSL/feature
arc reopens at a longer rebal. If quarterly stays near +0.012, the
ceiling is universe-bound and we move to the universe pivot.

Run from the repo root:
    uv run python apps/factor/scripts/horizon_pivot_walkforward.py
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
    print(f'\n=== {label} (target={wf.forward_target_kind}, '
          f'rebal={wf.rebal_days}d) ===', flush=True)
    print(f'{"win":>3}  {"train_ic":>9}  {"val_ic":>9}  {"train_sh":>9}  '
          f'{"val_sh":>9}  {"n_train":>7}  {"n_val":>5}', flush=True)
    for w in wf.windows:
        print(f'{w.window_idx:>3}  {w.train_ic:+.4f}    {w.val_ic:+.4f}    '
              f'{w.train_sharpe:+.3f}    {w.val_sharpe:+.3f}     '
              f'{w.n_train_bars:>7d}  {w.n_val_bars:>5d}', flush=True)
    print(f'mean val IC    : {wf.mean_val_ic:+.4f}', flush=True)
    print(f'median val IC  : {wf.median_val_ic:+.4f}', flush=True)
    print(f'mean val Sharpe: {wf.mean_val_sharpe:+.3f}', flush=True)
    print(f'pos val IC frac: {wf.positive_val_ic_fraction:.2f}  '
          f'({sum(1 for w in wf.windows if w.val_ic > 0)}/{wf.n_windows})',
          flush=True)

    blob: dict[str, np.ndarray] = {
        'window_idx':   np.array([w.window_idx for w in wf.windows]),
        'train_ic':     np.array([w.train_ic for w in wf.windows]),
        'val_ic':       np.array([w.val_ic for w in wf.windows]),
        'train_sharpe': np.array([w.train_sharpe for w in wf.windows]),
        'val_sharpe':   np.array([w.val_sharpe for w in wf.windows]),
    }
    blob['_meta'] = np.array(json.dumps({
        'forward_target_kind': wf.forward_target_kind,
        'rebal_days': wf.rebal_days,
        'train_window_blocks': wf.train_window_blocks,
        'val_window_blocks': wf.val_window_blocks,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'n_windows': wf.n_windows,
    }))
    np.savez(save_path, **blob)
    print(f'-> {save_path.name}', flush=True)

    return {
        'arm':             label,
        'target':          wf.forward_target_kind,
        'rebal_days':      wf.rebal_days,
        'mean_val_ic':     wf.mean_val_ic,
        'median_val_ic':   wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'n_windows':       wf.n_windows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',  default='2000-01-01')
    p.add_argument('--end',    default='2026-04-01')
    p.add_argument('--rebal-days', type=int, default=63)
    p.add_argument('--train-window-blocks', type=int, default=20)
    p.add_argument('--val-window-blocks',   type=int, default=12)
    p.add_argument('--step-window-blocks',  type=int, default=12)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--scorer',        default='linear')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0)
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--targets', default='log_return,vol_innovation')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    valid_targets = {'log_return', 'sign_demeaned', 'vol_innovation'}
    if not set(targets).issubset(valid_targets):
        raise SystemExit(f'--targets must be subset of {sorted(valid_targets)}')

    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}', flush=True)
    print(f'horizon pivot: rebal_days={args.rebal_days} '
          f'(scaled blocks train/val/step = '
          f'{args.train_window_blocks}/{args.val_window_blocks}/'
          f'{args.step_window_blocks})', flush=True)

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

    label_for: dict[str, str] = {
        'log_return':     'q-return',
        'sign_demeaned':  'q-sign',
        'vol_innovation': 'q-vol',
    }
    summary_arms: list[dict] = []
    for target_kind in targets:
        label = label_for[target_kind]
        save_path = output / f'horizon-pivot-{label}-windows.npz'
        print(f'\n--- arm: {label} (forward_target_kind={target_kind}, '
              f'rebal={args.rebal_days}d) ---', flush=True)
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
            forward_target_kind=target_kind,
        )
        print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)
        summary_arms.append(_summarize_arm(label, wf, save_path))

    print('\n' + '=' * 96, flush=True)
    print('Horizon-pivot leaderboard — quarterly (rebal=63d) vs documented '
          '20d baselines', flush=True)
    print('=' * 96, flush=True)
    print(f'{"arm":<22} {"target":<18} {"rebal":>7} {"mean_ic":>10} '
          f'{"median_ic":>10} {"mean_sh":>10} {"posfrac":>10}', flush=True)
    for a in summary_arms:
        print(f'{a["arm"]:<22} {a["target"]:<18} '
              f'{a["rebal_days"]:>5d}d  '
              f'{a["mean_val_ic"]:>+10.4f} {a["median_val_ic"]:>+10.4f} '
              f'{a["mean_val_sharpe"]:>+10.3f} '
              f'{a["positive_val_ic_fraction"]:>10.2f}', flush=True)
    print()
    print('20d baseline (forecast probe / NOTES 2026-05-04, factor commits '
          'def3ac9 / ff90762):', flush=True)
    print(f'{"control-20d":<22} {"log_return":<18} {"   20d":>7} '
          f'{"+0.0120":>10} {"+0.0168":>10} {"+0.440":>10} {"0.83":>10}',
          flush=True)
    print(f'{"vol-20d":<22} {"vol_innovation":<18} {"   20d":>7} '
          f'{"+0.4743":>10} {"+0.4735":>10} {"+0.515":>10} {"1.00":>10}',
          flush=True)

    summary_path = output / 'horizon-pivot-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'feature_width': F,
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'scorer': args.scorer, 'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'arms': summary_arms,
        'reference_20d_baseline': {
            'log_return':     {'mean_val_ic': 0.0120, 'mean_val_sharpe': 0.440},
            'vol_innovation': {'mean_val_ic': 0.4743, 'mean_val_sharpe': 0.515},
        },
    }, indent=2))
    print(f'\n-> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
