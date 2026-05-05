"""Forecast-target probe vs the +0.012 IC deterministic baseline.

Tests whether replacing the rank-IC training target (raw forward log
return) with a sign-of-cross-sectionally-demeaned target lifts val IC on
the same 297-ticker walk-forward universe used in the documented
indicator baseline (NOTES.md 2026-04-30, factor README).

The Pearson IC loss already subtracts per-bar cross-sectional means
inside the correlation, so the ":+0.012 baseline" already is the
demeaned IC. The change here is at the *target*: discard magnitude,
keep direction-vs-peers (±1). Two arms run back-to-back at the same
walk-forward config so the comparison is direct:

  * `log_return`  — control. Reproduces the documented +0.012 val IC.
  * `sign_demeaned` — probe. Same features / head / loss / eval, but
                       the target is sign(fwd_log_ret − cross_sectional_mean).

Sharpe eval is identical across arms (block-Sharpe always uses actual
realized returns); only the IC training signal changes.

Universe: `apps/notebook/data/stooq_us_long/manifest.json` filtered to
`min_history_bars=6500` → 297 tickers, ~26-year common axis.

Run from the repo root:

    uv run python apps/factor/scripts/forecast_probe_walkforward.py

Flags expose a smoke mode (`--max-tickers 30 --n-steps 50`) for ~30s
sanity checks before the full run.
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
    """Mirror of the modal harness's universe filter so the local probe
    sees the same 297-ticker subset as the documented baseline."""
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    if min_history_bars > 0:
        entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} tickers; {len(entries)} pass '
          f'min_history_bars={min_history_bars}')
    names = [t['ticker'] for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers} for smoke run')
    return names


def _build_one_ticker(args):
    """Top-level for multiprocessing pickling. Returns (name, TickerData|None,
    err_str|None)."""
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
    """Print per-window detail + aggregate, save (windows, head_params) npz.

    Returns a dict captured into the cross-arm summary JSON.
    """
    print(f'\n=== {label} (target={wf.forward_target_kind}) ===')
    print(f'{"win":>3}  {"train_ic":>9}  {"val_ic":>9}  {"train_sh":>9}  '
          f'{"val_sh":>9}  {"n_train":>7}  {"n_val":>5}')
    for w in wf.windows:
        print(f'{w.window_idx:>3}  {w.train_ic:+.4f}    {w.val_ic:+.4f}    '
              f'{w.train_sharpe:+.3f}    {w.val_sharpe:+.3f}     '
              f'{w.n_train_bars:>7d}  {w.n_val_bars:>5d}')
    print(f'mean val IC    : {wf.mean_val_ic:+.4f}')
    print(f'median val IC  : {wf.median_val_ic:+.4f}')
    print(f'mean val Sharpe: {wf.mean_val_sharpe:+.3f}')
    print(f'pos val IC frac: {wf.positive_val_ic_fraction:.2f}  '
          f'({sum(1 for w in wf.windows if w.val_ic > 0)}/{wf.n_windows})')

    blob: dict[str, np.ndarray] = {
        'window_idx':        np.array([w.window_idx for w in wf.windows]),
        'train_block_start': np.array([w.train_block_start for w in wf.windows]),
        'train_block_end':   np.array([w.train_block_end for w in wf.windows]),
        'val_block_start':   np.array([w.val_block_start for w in wf.windows]),
        'val_block_end':     np.array([w.val_block_end for w in wf.windows]),
        'train_ic':          np.array([w.train_ic for w in wf.windows]),
        'val_ic':            np.array([w.val_ic for w in wf.windows]),
        'train_sharpe':      np.array([w.train_sharpe for w in wf.windows]),
        'val_sharpe':        np.array([w.val_sharpe for w in wf.windows]),
        'n_train_bars':      np.array([w.n_train_bars for w in wf.windows]),
        'n_val_bars':        np.array([w.n_val_bars for w in wf.windows]),
    }
    blob['_meta'] = np.array(json.dumps({
        'forward_target_kind': wf.forward_target_kind,
        'scorer': wf.scorer, 'n_steps': wf.n_steps,
        'learning_rate': wf.learning_rate, 'weight_decay': wf.weight_decay,
        'rebal_days': wf.rebal_days,
        'train_window_blocks': wf.train_window_blocks,
        'val_window_blocks': wf.val_window_blocks,
        'step_window_blocks': wf.step_window_blocks,
        'feature_width': wf.feature_width,
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'median_val_ic': wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
    }))
    np.savez(save_path, **blob)
    print(f'-> {save_path.name}')

    return {
        'arm':             label,
        'target':          wf.forward_target_kind,
        'mean_val_ic':     wf.mean_val_ic,
        'median_val_ic':   wf.median_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'positive_val_ic_fraction': wf.positive_val_ic_fraction,
        'n_windows':       wf.n_windows,
        'per_window_val_ic':     [w.val_ic     for w in wf.windows],
        'per_window_val_sharpe': [w.val_sharpe for w in wf.windows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',  default='2000-01-01')
    p.add_argument('--end',    default='2026-04-01')
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--train-window-blocks', type=int, default=63)  # ~5y at rebal=20
    p.add_argument('--val-window-blocks',   type=int, default=39)  # ~3y
    p.add_argument('--step-window-blocks',  type=int, default=39)
    p.add_argument('--n-steps',       type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=1e-2)
    p.add_argument('--weight-decay',  type=float, default=1e-3)
    p.add_argument('--scorer',        default='linear')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0,
                   help='0 = all tickers passing min_history_bars filter')
    p.add_argument('--n-workers',     type=int, default=mp.cpu_count())
    p.add_argument('--targets', default='log_return,sign_demeaned',
                   help='comma-separated subset of {log_return,sign_demeaned}')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    valid_targets = {'log_return', 'sign_demeaned'}
    if not set(targets).issubset(valid_targets):
        raise SystemExit(
            f'--targets must be subset of {sorted(valid_targets)}; '
            f'got {targets}')

    # Heavy imports happen here so `--help` is fast and worker processes
    # re-import inside `_build_one_ticker` rather than inheriting state.
    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'IndicatorGridConfig.feature_width() = {F}')

    names = _resolve_ticker_list(args.min_history_bars, args.max_tickers)

    print(f'\nbuilding features over {len(names)} tickers '
          f'(workers={args.n_workers}) ...')
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
          f'{time.perf_counter() - t0:.1f}s')
    if failed:
        print(f'  failed: {len(failed)}')
        for name, err in failed[:10]:
            print(f'    {name}: {err}')

    summary_arms: list[dict] = []
    for target_kind in targets:
        label = 'control' if target_kind == 'log_return' else 'probe'
        save_path = output / f'forecast-probe-{label}-windows.npz'
        print(f'\n--- arm: {label} (forward_target_kind={target_kind}) ---')
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
        print(f'arm wall: {time.perf_counter() - t1:.1f}s')
        summary_arms.append(_summarize_arm(label, wf, save_path))

    if len(summary_arms) == 2:
        ctrl = next(a for a in summary_arms if a['target'] == 'log_return')
        prob = next(a for a in summary_arms if a['target'] == 'sign_demeaned')
        d_ic = prob['mean_val_ic'] - ctrl['mean_val_ic']
        d_sh = prob['mean_val_sharpe'] - ctrl['mean_val_sharpe']
        print('\n' + '=' * 72)
        print('Forecast-target probe — control (log_return) vs probe (sign_demeaned)')
        print('=' * 72)
        print(f'{"metric":<28} {"control":>12} {"probe":>12} {"delta":>12}')
        print(f'{"mean val IC":<28} {ctrl["mean_val_ic"]:>+12.4f} '
              f'{prob["mean_val_ic"]:>+12.4f} {d_ic:>+12.4f}')
        print(f'{"median val IC":<28} {ctrl["median_val_ic"]:>+12.4f} '
              f'{prob["median_val_ic"]:>+12.4f} '
              f'{prob["median_val_ic"] - ctrl["median_val_ic"]:>+12.4f}')
        print(f'{"mean val Sharpe":<28} {ctrl["mean_val_sharpe"]:>+12.3f} '
              f'{prob["mean_val_sharpe"]:>+12.3f} {d_sh:>+12.3f}')
        print(f'{"positive-val-IC fraction":<28} '
              f'{ctrl["positive_val_ic_fraction"]:>12.2f} '
              f'{prob["positive_val_ic_fraction"]:>12.2f} '
              f'{prob["positive_val_ic_fraction"] - ctrl["positive_val_ic_fraction"]:>+12.2f}')
        print()
        print(f'documented baseline (NOTES.md / factor README): val IC ≈ +0.0120')
        print(f'control reproduces this within walk-forward seed noise')

    summary_path = output / 'forecast-probe-summary.json'
    summary_path.write_text(json.dumps({
        'universe_size': len(ticker_data),
        'feature_width': F,
        'rebal_days': args.rebal_days,
        'train_window_blocks': args.train_window_blocks,
        'val_window_blocks': args.val_window_blocks,
        'step_window_blocks': args.step_window_blocks,
        'scorer': args.scorer, 'n_steps': args.n_steps,
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'start': args.start, 'end': args.end,
        'arms': summary_arms,
    }, indent=2))
    print(f'\n-> {summary_path}')


if __name__ == '__main__':
    main()
