"""Sizing-input v0 eval — does training the indicator head with
masked MSE on per-bar alpha targets produce a calibrated sizing
signal (vs the existing scale-invariant rank-IC baseline)?

Motivating TODO:
    apps/docs/docs/TODO/factor-sizing-input-reframe.md

Hypothesis: at factor-narrow's IC scale (~+0.005-0.012 mean val IC),
training on `masked_mse` against `fwd_log_return − cross_sectional_mean`
(alpha targets) produces scores whose magnitude tracks expected
per-ticker alpha. The per-val-bar top-decile-minus-bottom-decile
predicted-alpha dispersion is then a candidate sizing input for the
macro meta-gate.

Two arms, same head + windows + commission as `loss_pivot_eval.py`:

  - `rank_ic`   — existing baseline. `pearson_rank_ic` loss; score
                  magnitude uncalibrated.
  - `mse_alpha` — new. `masked_mse` against per-bar alpha targets;
                  score magnitude calibrated to alpha units.

Per-bar artifact emitted by both arms (so the comparison can fire):
`signal_quality_per_val_bar` (shape `(n_windows, val_window_blocks)`)
plus `val_start_date` per window.

Pre-registered verdict (TODO/factor-sizing-input-reframe.md):

  PASS         — both true on `mse_alpha` vs `rank_ic`:
                   (a) lag-1 autocorrelation of signal_quality pooled
                       across windows ≥ +0.20 (vs rank_ic baseline);
                   (b) Spearman ρ between per-window mean
                       signal-quality and per-window val Sharpe ≥ +0.40.
                 Promote signal-quality to a macro-gate feature in v1.
  FAIL         — neither criterion clears for mse_alpha (or both fire
                 equally for rank_ic — no marginal benefit). Pivot.
  INCONCLUSIVE — one fires, one doesn't. Stratify by window.

Run from repo root:
    uv run python apps/factor/scripts/sizing_input_eval.py
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


def _lag1_autocorr(x: np.ndarray) -> float:
    """Pooled lag-1 Pearson autocorrelation, treating NaNs as gaps
    that break the autocorrelation pair (skip pairs with either side
    NaN). Returns NaN if fewer than 5 valid consecutive pairs."""
    if x.size < 2:
        return float('nan')
    a = x[:-1]
    b = x[1:]
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 5:
        return float('nan')
    av = a[valid]
    bv = b[valid]
    am = av.mean()
    bm = bv.mean()
    num = np.sum((av - am) * (bv - bm))
    den = np.sqrt(np.sum((av - am) ** 2) * np.sum((bv - bm) ** 2)) + 1e-18
    return float(num / den)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN on fewer than 3 valid
    pairs or zero-variance ranks."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float('nan')
    xv = x[mask]
    yv = y[mask]
    rx = np.argsort(np.argsort(xv)).astype(np.float64)
    ry = np.argsort(np.argsort(yv)).astype(np.float64)
    rxm = rx.mean(); rym = ry.mean()
    num = np.sum((rx - rxm) * (ry - rym))
    den = np.sqrt(np.sum((rx - rxm) ** 2) * np.sum((ry - rym) ** 2)) + 1e-18
    return float(num / den)


def _summarize_arm(label: str, wf, save_path: Path) -> dict:
    print(f'\n=== arm={label} ===', flush=True)
    print(f'{"win":>3}  {"val_start":>10}  {"tr_ic":>8}  {"val_ic":>8}  '
          f'{"val_sh":>8}  {"val_mse_a":>10}  {"sq_mean":>10}  '
          f'{"sq_std":>10}', flush=True)
    for w in wf.windows:
        print(f'{w.window_idx:>3}  {w.val_start_date:>10}  '
              f'{w.train_ic:+.4f}   {w.val_ic:+.4f}   '
              f'{w.val_sharpe:+8.3f}  {w.val_mse_alpha:>10.3e}  '
              f'{w.signal_quality_mean:>+10.3e}  '
              f'{w.signal_quality_std:>10.3e}', flush=True)

    sq_per_bar = np.stack(
        [w.signal_quality_per_val_bar for w in wf.windows], axis=0)
    pooled = sq_per_bar.reshape(-1)
    lag1 = _lag1_autocorr(pooled)
    per_window_sq = np.array([w.signal_quality_mean for w in wf.windows])
    per_window_sh = np.array([w.val_sharpe for w in wf.windows])
    rho = _spearman(per_window_sq, per_window_sh)

    summary = {
        'arm': label,
        'n_windows': wf.n_windows,
        'mean_val_ic': wf.mean_val_ic,
        'mean_val_sharpe': wf.mean_val_sharpe,
        'mean_val_mse_alpha': wf.mean_val_mse_alpha,
        'mean_signal_quality': wf.mean_signal_quality,
        'lag1_autocorr_signal_quality_pooled': lag1,
        'spearman_sq_vs_val_sharpe': rho,
        'per_window': [{
            'window_idx': w.window_idx,
            'val_start_date': w.val_start_date,
            'train_ic': w.train_ic, 'val_ic': w.val_ic,
            'val_sharpe': w.val_sharpe,
            'val_mse_alpha': w.val_mse_alpha,
            'signal_quality_mean': w.signal_quality_mean,
            'signal_quality_std': w.signal_quality_std,
        } for w in wf.windows],
    }
    np.savez(
        save_path,
        window_idx=np.array([w.window_idx for w in wf.windows]),
        val_start_date=np.array(
            [w.val_start_date for w in wf.windows], dtype='S10'),
        train_ic=np.array([w.train_ic for w in wf.windows]),
        val_ic=np.array([w.val_ic for w in wf.windows]),
        val_sharpe=np.array([w.val_sharpe for w in wf.windows]),
        val_mse_alpha=np.array([w.val_mse_alpha for w in wf.windows]),
        signal_quality_per_val_bar=sq_per_bar,
        signal_quality_mean=per_window_sq,
        signal_quality_std=np.array(
            [w.signal_quality_std for w in wf.windows]))
    print(f'  pooled lag-1 sq autocorr   : {lag1:+.4f}', flush=True)
    print(f'  Spearman(sq_mean, val_sh)  : {rho:+.4f}', flush=True)
    print(f'  -> {save_path}', flush=True)
    return summary


def _verdict(arms: dict[str, dict]) -> str:
    """Pre-registered v0 verdict.

    PASS         — mse_alpha clears both criteria *relative to* rank_ic:
                     (a) lag1_autocorr_signal_quality_pooled ≥ +0.20
                         AND ≥ rank_ic + 0.10;
                     (b) spearman_sq_vs_val_sharpe ≥ +0.40 AND ≥
                         rank_ic + 0.10.
    FAIL         — neither criterion clears for mse_alpha.
    INCONCLUSIVE — one criterion clears, one fails. Stratify per-window.
    """
    if 'mse_alpha' not in arms or 'rank_ic' not in arms:
        return 'N/A (missing arm)'
    new = arms['mse_alpha']; base = arms['rank_ic']
    a_pass = (
        new['lag1_autocorr_signal_quality_pooled'] >= 0.20
        and (new['lag1_autocorr_signal_quality_pooled']
             - base['lag1_autocorr_signal_quality_pooled']) >= 0.10
    )
    b_pass = (
        new['spearman_sq_vs_val_sharpe'] >= 0.40
        and (new['spearman_sq_vs_val_sharpe']
             - base['spearman_sq_vs_val_sharpe']) >= 0.10
    )
    if a_pass and b_pass:
        return 'PASS  (both criteria clear)'
    if not a_pass and not b_pass:
        return 'FAIL  (neither criterion clears)'
    which = 'lag1-autocorr' if a_pass else 'Spearman'
    return f'INCONCLUSIVE  (only {which} clears)'


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
    p.add_argument('--losses', default='rank_ic,mse_alpha',
                   help='comma-separated loss kinds to evaluate')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    losses = [s.strip() for s in args.losses.split(',') if s.strip()]
    valid = {'rank_ic', 'mse_alpha', 'block_sharpe', 'ir_vs_ew'}
    if not set(losses).issubset(valid):
        raise SystemExit(f'--losses must be subset of {sorted(valid)}')

    from factor import IndicatorGridConfig, train_scorer_indicators_walkforward

    cfg = IndicatorGridConfig()
    F = cfg.feature_width()
    print(f'sizing-input eval: indicator stack F={F}, scorer={args.scorer}, '
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
        save_path = output / f'sizing-input-{loss_kind}-windows.npz'
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
    print('sizing-input v0 — calibration layer comparison', flush=True)
    print('=' * 96, flush=True)
    print(f'{"arm":>10}  {"val_ic":>8}  {"val_sh":>8}  '
          f'{"val_mse_a":>10}  {"sq_mean":>11}  {"lag1_ac":>8}  '
          f'{"spearman":>9}', flush=True)
    print('-' * 96, flush=True)
    for arm_name, arm in arms.items():
        print(f'{arm_name:>10}  {arm["mean_val_ic"]:>+8.4f}  '
              f'{arm["mean_val_sharpe"]:>+8.3f}  '
              f'{arm["mean_val_mse_alpha"]:>10.3e}  '
              f'{arm["mean_signal_quality"]:>+11.3e}  '
              f'{arm["lag1_autocorr_signal_quality_pooled"]:>+8.3f}  '
              f'{arm["spearman_sq_vs_val_sharpe"]:>+9.3f}', flush=True)
    print()
    print(f'verdict: {_verdict(arms)}', flush=True)

    summary_path = output / 'sizing-input-eval-summary.json'
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
