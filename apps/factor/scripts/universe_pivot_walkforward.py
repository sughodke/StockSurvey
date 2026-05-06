"""Wider-universe pivot — does +0.012 IC ceiling lift on a bigger panel?

Same setup as `forecast_probe_walkforward.py` but loads from the **full**
`StooqData/` archive (12K tickers including delisted) via
`load_stooq_matrix`, not the curated `stooq_us_long` subset (which is
pre-filtered to 312 long-history names so dropping its `min_history`
threshold doesn't widen the universe meaningfully).

Loader filters: `min_history` (panel-wide leniency floor — see
`load_stooq_matrix` docstring), then post-load drop columns whose first
non-NaN date is past `start_date + grace` (otherwise they truncate the
common axis). Resulting universe is a few hundred to ~2K tickers
depending on threshold; default targets ~1500 tickers with a ~20-year
common axis (`min_history=5000`, `start_grace_days=180`).

Same `IndicatorGridConfig` (74 channels), same linear head, same
n_steps=200 / lr=1e-2 / wd=1e-3, rebal=20d (matched to the documented
+0.012 / +0.47 baselines).

Read: if `log_return` mean val IC clears +0.020+ on the wider universe
when 297 tickers sat at +0.012, the +0.012 ceiling was supervision-
bound (cross-section size) per NOTES 2026-04-30. If wider stays near
+0.012, the data simply doesn't carry more cross-sectional return
signal at 20d horizon — we're done with the indicator/CWT arc.

Run from the repo root:
    uv run python apps/factor/scripts/universe_pivot_walkforward.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_DATA = REPO_ROOT / 'StooqData'
STOOQ_CACHE = STOOQ_DATA / '.cache.pkl'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_one_ticker_args(args):
    """Top-level for multiprocessing pickling. `args` is
    `(ticker, prices, dates, cfg)` — prices already loaded so no I/O.

    Trims leading NaN before feature build (MACD's EMA seeds on the
    first sample, propagating NaN forever if leading prices are NaN —
    drops 86% of late-listing tickers if not handled). Pads features
    and valid mask back onto the full date axis so `align_tickers`
    sees a single common axis across all tickers.
    """
    ticker, prices, dates, cfg = args
    try:
        from factor import build_indicator_features
        from ss_features import TickerData
        finite = np.isfinite(prices)
        if not finite.any():
            return ticker, None, '(no finite prices)'
        first_valid = int(np.argmax(finite))
        prices_trimmed = prices[first_valid:]
        feats_trimmed, valid_trimmed = build_indicator_features(
            prices_trimmed, cfg)
        F = feats_trimmed.shape[1]
        feats = np.full((len(prices), F), np.nan, dtype=np.float32)
        feats[first_valid:] = feats_trimmed
        valid = np.zeros(len(prices), dtype=bool)
        valid[first_valid:] = valid_trimmed
        if not valid.any():
            return ticker, None, '(no valid bars)'
        return ticker, TickerData(
            name=ticker, prices=prices, dates=dates,
            features=feats, targets={}, valid=valid,
        ), None
    except Exception as e:
        return ticker, None, f'({type(e).__name__}: {e})'


def _filter_universe(
    close: pd.DataFrame, *,
    min_history: int, start_date: str, start_grace_days: int,
) -> pd.DataFrame:
    """Drop columns whose first non-NaN date is past `start_date +
    start_grace_days`, so the common axis after intersection isn't
    truncated by late-listing names.

    `min_history` is enforced before this (`load_stooq_matrix` did it),
    so we just trim the late-starters here.
    """
    target_start = pd.Timestamp(start_date) + pd.Timedelta(days=start_grace_days)
    first_dates = close.apply(lambda s: s.first_valid_index())
    keep = first_dates[first_dates <= target_start].index.tolist()
    dropped = [c for c in close.columns if c not in keep]
    if dropped:
        print(f'  dropped {len(dropped)} late-starters (first valid date > '
              f'{target_start.date()}): {dropped[:6]}'
              f'{"..." if len(dropped) > 6 else ""}', flush=True)
    return close[keep]


def _summarize_arm(label: str, wf, save_path: Path) -> dict:
    print(f'\n=== {label} (target={wf.forward_target_kind}) ===', flush=True)
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
        'feature_width': wf.feature_width,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
    }))
    np.savez(save_path, **blob)
    print(f'-> {save_path.name}', flush=True)

    return {
        'arm':             label,
        'target':          wf.forward_target_kind,
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
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--train-window-blocks', type=int, default=63)
    p.add_argument('--val-window-blocks',   type=int, default=39)
    p.add_argument('--step-window-blocks',  type=int, default=39)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--scorer',        default='linear')
    p.add_argument('--min-history',   type=int, default=5000,
                   help='load_stooq_matrix min_history (~20y at 5000)')
    p.add_argument('--start-grace-days', type=int, default=180,
                   help='drop tickers whose first valid date is past '
                        'start + this many days, to preserve common axis')
    p.add_argument('--max-tickers',  type=int, default=0)
    p.add_argument('--n-workers',    type=int, default=mp.cpu_count())
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
    from ss_loaders import load_stooq_matrix

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}', flush=True)

    print(f'\nloading from full StooqData/ archive (cache: '
          f'{STOOQ_CACHE.exists()}) ...', flush=True)
    t0 = time.perf_counter()
    cache_arg = str(STOOQ_CACHE) if STOOQ_CACHE.exists() else None
    close, _high, _low, _vol = load_stooq_matrix(
        str(STOOQ_DATA),
        min_history=args.min_history,
        start_date=args.start, end_date=args.end,
        cache_path=cache_arg,
    )
    print(f'  raw load: {close.shape[0]} dates × {close.shape[1]} tickers '
          f'in {time.perf_counter() - t0:.1f}s', flush=True)

    close = _filter_universe(
        close, min_history=args.min_history,
        start_date=args.start, start_grace_days=args.start_grace_days)
    if args.max_tickers > 0:
        close = close.iloc[:, :args.max_tickers]
        print(f'  capped to first {args.max_tickers} for smoke run', flush=True)
    print(f'  filtered: {close.shape[0]} dates × {close.shape[1]} tickers',
          flush=True)
    print(f'  date range: {close.index[0].date()} .. {close.index[-1].date()}',
          flush=True)

    # Build TickerData per column. Pass prices+dates rather than ticker
    # name+stooq_dir so workers don't re-do I/O.
    print(f'\nbuilding features over {close.shape[1]} tickers '
          f'(workers={args.n_workers}) ...', flush=True)
    t0 = time.perf_counter()
    dates = np.asarray(close.index)
    pool_args = [
        (col, close[col].values.astype(np.float64), dates, cfg)
        for col in close.columns
    ]
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            results = pool.map(_build_one_ticker_args, pool_args)
    else:
        results = [_build_one_ticker_args(a) for a in pool_args]
    ticker_data = []
    failed = []
    for name, td, err in results:
        if td is None:
            failed.append((name, err))
        else:
            ticker_data.append(td)
    print(f'built {len(ticker_data)} / {len(close.columns)} tickers in '
          f'{time.perf_counter() - t0:.1f}s', flush=True)
    if failed:
        print(f'  failed: {len(failed)}', flush=True)

    label_for: dict[str, str] = {
        'log_return':     'wide-return',
        'sign_demeaned':  'wide-sign',
        'vol_innovation': 'wide-vol',
    }
    summary_arms: list[dict] = []
    for target_kind in targets:
        label = label_for[target_kind]
        save_path = output / f'universe-pivot-{label}-windows.npz'
        print(f'\n--- arm: {label} (forward_target_kind={target_kind}) ---',
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
            forward_target_kind=target_kind,
        )
        print(f'arm wall: {time.perf_counter() - t1:.1f}s', flush=True)
        summary_arms.append(_summarize_arm(label, wf, save_path))

    print('\n' + '=' * 96, flush=True)
    print('Universe-pivot leaderboard — full StooqData/ vs 297-ticker '
          'stooq_us_long baseline', flush=True)
    print('=' * 96, flush=True)
    print(f'{"arm":<22} {"target":<18} {"N":>6} {"mean_ic":>10} '
          f'{"median_ic":>10} {"mean_sh":>10} {"posfrac":>10}', flush=True)
    for a in summary_arms:
        print(f'{a["arm"]:<22} {a["target"]:<18} '
              f'{len(ticker_data):>6d} '
              f'{a["mean_val_ic"]:>+10.4f} {a["median_val_ic"]:>+10.4f} '
              f'{a["mean_val_sharpe"]:>+10.3f} '
              f'{a["positive_val_ic_fraction"]:>10.2f}', flush=True)
    print()
    print('297-ticker baseline (forecast probe / NOTES 2026-05-04, factor '
          'commits def3ac9 / ff90762):', flush=True)
    print(f'{"narrow-return":<22} {"log_return":<18} {"   297":>6} '
          f'{"+0.0120":>10} {"+0.0168":>10} {"+0.440":>10} {"0.83":>10}',
          flush=True)
    print(f'{"narrow-vol":<22} {"vol_innovation":<18} {"   297":>6} '
          f'{"+0.4743":>10} {"+0.4735":>10} {"+0.515":>10} {"1.00":>10}',
          flush=True)

    summary_path = output / 'universe-pivot-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'min_history': args.min_history,
        'start_grace_days': args.start_grace_days,
        'feature_width': F,
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'scorer': args.scorer, 'n_steps': args.n_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'arms': summary_arms,
        'reference_297_ticker_baseline': {
            'log_return':     {'mean_val_ic': 0.0120, 'mean_val_sharpe': 0.440},
            'vol_innovation': {'mean_val_ic': 0.4743, 'mean_val_sharpe': 0.515},
        },
    }, indent=2))
    print(f'\n-> {summary_path}', flush=True)


if __name__ == '__main__':
    main()
