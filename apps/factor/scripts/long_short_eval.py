"""Long-short constructor eval — close the rank-IC / long-only-top-N mismatch.

The cheap test pre-registered in
`apps/docs/docs/TODO/long-short-constructor.md`: take the existing
deterministic-indicator linear head (the +0.012 mean val IC baseline
recorded in `factor-indicator-baseline.md`) and report both the
existing softmax-top-N val Sharpe and the new market-neutral
long-short val Sharpe over the same 6-window walk-forward.

If long-short val Sharpe ≥ +0.20 with ≥4/6 windows positive: the head
had short-side skill we were discarding, and we proceed to retrain
with a Sharpe-aligned loss. If < +0.10 or ≤2/6: the head genuinely
lacks cross-sectional dispersion, and we pivot to a different
prediction problem instead.

Run from the repo root:
    uv run python apps/factor/scripts/long_short_eval.py

Mirrors the universe / windowing / hyperparams of
`horizon_pivot_walkforward.py` for apples-to-apples comparison vs the
documented baseline.
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


def _summarize(wf, save_path: Path) -> dict:
    print('\n=== per-window comparison ===', flush=True)
    print(f'{"win":>3}  {"train_ic":>9}  {"val_ic":>9}  '
          f'{"val_sh_LO":>10}  {"val_sh_LS":>10}  {"alpha_LS":>10}  '
          f'{"n_train":>7}  {"n_val":>5}', flush=True)
    for w in wf.windows:
        alpha = w.val_sharpe_long_short - w.val_sharpe
        print(f'{w.window_idx:>3}  {w.train_ic:+.4f}    {w.val_ic:+.4f}    '
              f'{w.val_sharpe:+10.3f}  {w.val_sharpe_long_short:+10.3f}  '
              f'{alpha:+10.3f}  {w.n_train_bars:>7d}  {w.n_val_bars:>5d}',
              flush=True)
    summary = {
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'mean_val_sharpe_long_only': wf.mean_val_sharpe,
        'mean_val_sharpe_long_short': wf.mean_val_sharpe_long_short,
        'positive_val_sharpe_long_short_fraction':
            wf.positive_val_sharpe_long_short_fraction,
        'per_window': [{
            'window_idx': w.window_idx,
            'train_ic': w.train_ic, 'val_ic': w.val_ic,
            'val_sharpe_long_only': w.val_sharpe,
            'val_sharpe_long_short': w.val_sharpe_long_short,
        } for w in wf.windows],
    }
    np.savez(save_path,
             window_idx=np.array([w.window_idx for w in wf.windows]),
             train_ic=np.array([w.train_ic for w in wf.windows]),
             val_ic=np.array([w.val_ic for w in wf.windows]),
             val_sharpe_long_only=np.array([w.val_sharpe for w in wf.windows]),
             val_sharpe_long_short=np.array(
                 [w.val_sharpe_long_short for w in wf.windows]))
    print(f'  -> {save_path}', flush=True)
    return summary


def _verdict(mean_ls: float, pos_frac: float) -> str:
    """Apply the pre-registered cuts from
    `apps/docs/docs/TODO/long-short-constructor.md`."""
    pos_count = int(round(pos_frac * 6))  # convert 6-window frac → count
    if mean_ls >= 0.20 and pos_count >= 4:
        return 'PASS  (mean_LS ≥ +0.20 and ≥4/6 windows positive)'
    if mean_ls < 0.10 or pos_count <= 2:
        return 'FAIL  (mean_LS < +0.10 or ≤2/6 windows positive)'
    return 'INCONCLUSIVE  (between thresholds — stratify by window before retrain)'


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
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'long-short eval: indicator stack F={F}, scorer={args.scorer}, '
          f'rebal={args.rebal_days}d, train/val/step blocks = '
          f'{args.train_window_blocks}/{args.val_window_blocks}/'
          f'{args.step_window_blocks}', flush=True)

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

    save_path = output / 'long-short-eval-windows.npz'
    print(f'\n--- training {args.scorer} head with rank-IC, eval long-only + '
          f'long-short ---', flush=True)
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
    )
    print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)
    summary = _summarize(wf, save_path)

    print('\n' + '=' * 96, flush=True)
    print('long-short eval — head trained on rank-IC, both constructors '
          'evaluated on val', flush=True)
    print('=' * 96, flush=True)
    print(f'mean val IC                    {summary["mean_val_ic"]:+.4f}  '
          f'(reference: +0.0120 from factor-indicator-baseline)', flush=True)
    print(f'mean val Sharpe — long-only    '
          f'{summary["mean_val_sharpe_long_only"]:+.3f}  '
          f'(reference: +0.44 from factor-indicator-baseline)', flush=True)
    print(f'mean val Sharpe — long-short   '
          f'{summary["mean_val_sharpe_long_short"]:+.3f}', flush=True)
    print(f'positive long-short windows    '
          f'{summary["positive_val_sharpe_long_short_fraction"]:.2f} '
          f'({int(round(summary["positive_val_sharpe_long_short_fraction"] * 6))}/6)',
          flush=True)
    alpha_mean = (summary['mean_val_sharpe_long_short']
                  - summary['mean_val_sharpe_long_only'])
    print(f'long-short alpha (vs LO)       {alpha_mean:+.3f}', flush=True)
    print()
    print(f'verdict: {_verdict(summary["mean_val_sharpe_long_short"], summary["positive_val_sharpe_long_short_fraction"])}',
          flush=True)

    summary_path = output / 'long-short-eval-summary.json'
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
        **summary,
    }, indent=2))
    print(f'\n-> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
