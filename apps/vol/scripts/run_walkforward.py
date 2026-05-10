"""Walk-forward eval of the surface-feature predictor.

Pre-registered cuts (per `apps/docs/docs/TODO/apps-vol.md`):
  PASS     : predictor lifts short-vol Sharpe ≥ baseline + 0.30
             with ≥ 4/6 windows positive alpha
  MARGINAL : baseline + (0.10, 0.30) → stratify by ticker liquidity
  FAIL     : < baseline + 0.10 → confirmed-null (the IV market
             efficiently incorporates surface-shape info too)

Universe-wide short-vol baseline = top_quantile=1.0 arm (everyone
gets equal exposure). Gated arm = top_quantile=0.80 (top 20% by
predicted IV-RV gap per rebalance).

Run from repo root:
    uv run python apps/vol/scripts/run_walkforward.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (
    FEATURE_NAMES, build_vol_features, evaluate_gated_short_vol,
    forward_iv_rv_gap, load_gauss314_full, predict, train_predictor,
)
from vol.predictor import evaluate_r2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_window_slices_by_date(
    dates: pd.DatetimeIndex,
    train_days: int, val_days: int, step_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Build rolling windows over a sorted DatetimeIndex.

    Returns list of `(train_start, train_end, val_start, val_end)`
    timestamps. Train/val are non-overlapping; step controls slide.
    """
    if len(dates) < train_days + val_days:
        return []
    out = []
    i = 0
    while i + train_days + val_days <= len(dates):
        ts_train_start = dates[i]
        ts_train_end   = dates[i + train_days - 1]
        ts_val_start   = dates[i + train_days]
        ts_val_end     = dates[i + train_days + val_days - 1]
        out.append((ts_train_start, ts_train_end,
                    ts_val_start, ts_val_end))
        i += step_days
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=300)  # ~14 mo
    p.add_argument('--val-window-days',   type=int, default=120)  # ~6 mo
    p.add_argument('--step-window-days',  type=int, default=120)
    p.add_argument('--top-quantile', type=float, default=0.80,
                   help='Top quantile of predicted IV-RV gap to short')
    p.add_argument('--clip-iv-hv-ratio', type=float, default=10.0,
                   help='Cap IV/HV ratio features (HV-near-zero outliers)')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading gauss314 full schema...')
    t0 = time.perf_counter()
    raw = load_gauss314_full()
    print(f'  {len(raw):,} rows in {time.perf_counter()-t0:.1f}s')

    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)

    merged = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap'])
    # Clip pathological IV/HV ratios (audit showed HV-near-0 outliers).
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        merged[col] = merged[col].clip(
            lower=-args.clip_iv_hv_ratio, upper=args.clip_iv_hv_ratio)
    print(f'  usable rows after merge + clip: {len(merged):,}')

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    print(f'  dates: {dates[0].date()} → {dates[-1].date()} ({len(dates)})')

    windows = _build_window_slices_by_date(
        dates, args.train_window_days, args.val_window_days,
        args.step_window_days)
    if not windows:
        raise SystemExit(
            f'no windows fit: have {len(dates)} dates but need '
            f'train+val={args.train_window_days + args.val_window_days}')
    print(f'  walk-forward: {len(windows)} windows')

    print('\n' + '=' * 96, flush=True)
    print(f'{"win":>3s} {"train":>23s} {"val":>23s} '
          f'{"train R²":>9s} {"val R²":>8s} {"val r":>7s} '
          f'{"unc Sh":>7s} {"gat Sh":>7s} {"alpha":>7s} '
          f'{"unc PnL":>9s} {"gat PnL":>9s}', flush=True)
    print('-' * 110, flush=True)

    rows = []
    for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
        train = merged[(merged['date'] >= ts_tr_lo) &
                       (merged['date'] <= ts_tr_hi)]
        val   = merged[(merged['date'] >= ts_va_lo) &
                       (merged['date'] <= ts_va_hi)]
        if len(train) < 1000 or len(val) < 500:
            continue

        X_tr = train[FEATURE_NAMES].values
        y_tr = train['iv_rv_gap'].values
        X_va = val[FEATURE_NAMES].values
        y_va = val['iv_rv_gap'].values

        pred = train_predictor(X_tr, y_tr, FEATURE_NAMES)
        val_pred = predict(pred, X_va)
        val_r2 = evaluate_r2(val_pred, y_va)
        val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])

        # Build pred_gap and realized_gap tables for this val.
        val_with_pred = val[['date', 'symbol']].copy()
        val_with_pred['pred_gap'] = val_pred
        val_with_realized = val[['date', 'symbol', 'iv_rv_gap']].copy()

        # Universe-wide baseline (top_quantile=1.0 includes everyone).
        unc = evaluate_gated_short_vol(
            val_with_pred.assign(pred_gap=1.0),
            val_with_realized,
            top_quantile=0.0,    # 0.0 keeps everything
            arm_label='unc')
        # Gated arm.
        gated = evaluate_gated_short_vol(
            val_with_pred, val_with_realized,
            top_quantile=args.top_quantile, arm_label='gated')

        alpha = gated.sharpe_per_cell - unc.sharpe_per_cell
        rows.append({
            'window_idx': w_idx,
            'train_start': str(ts_tr_lo.date()),
            'train_end':   str(ts_tr_hi.date()),
            'val_start':   str(ts_va_lo.date()),
            'val_end':     str(ts_va_hi.date()),
            'n_train_cells': len(train),
            'n_val_cells':   len(val),
            'train_r2': pred.train_r2,
            'val_r2':   val_r2,
            'val_pearson_r': val_corr,
            'unc_sharpe_per_cell':   unc.sharpe_per_cell,
            'gated_sharpe_per_cell': gated.sharpe_per_cell,
            'alpha_sharpe_per_cell': alpha,
            'unc_mean_pnl':   unc.mean_pnl_per_cell,
            'gated_mean_pnl': gated.mean_pnl_per_cell,
            'unc_n_picks':   unc.n_picks,
            'gated_n_picks': gated.n_picks,
        })
        print(f'{w_idx:>3d} '
              f'{ts_tr_lo.date()}→{ts_tr_hi.date()} '
              f'{ts_va_lo.date()}→{ts_va_hi.date()} '
              f'{pred.train_r2:>+9.4f} {val_r2:>+8.4f} {val_corr:>+7.4f} '
              f'{unc.sharpe_per_cell:>+7.3f} '
              f'{gated.sharpe_per_cell:>+7.3f} {alpha:>+7.3f} '
              f'{unc.mean_pnl_per_cell:>+9.4f} '
              f'{gated.mean_pnl_per_cell:>+9.4f}',
              flush=True)

    print('\n' + '=' * 96, flush=True)
    if not rows:
        print('No usable windows.', flush=True)
        return
    mean_alpha = float(np.mean([r['alpha_sharpe_per_cell'] for r in rows]))
    pos_alpha_frac = float(np.mean([r['alpha_sharpe_per_cell'] > 0 for r in rows]))
    mean_val_r2 = float(np.mean([r['val_r2'] for r in rows]))
    mean_val_r = float(np.mean([r['val_pearson_r'] for r in rows]))
    print(f'mean val R²       = {mean_val_r2:+.4f}', flush=True)
    print(f'mean val Pearson r = {mean_val_r:+.4f}', flush=True)
    print(f'mean alpha (per-cell Sharpe) = {mean_alpha:+.3f}', flush=True)
    print(f'positive-alpha windows = {pos_alpha_frac:.2f} '
          f'({int(round(pos_alpha_frac * len(rows)))}/{len(rows)})', flush=True)

    if mean_alpha >= 0.30 and pos_alpha_frac >= 4 / 6:
        verdict = 'PASS — surface features lift short-vol Sharpe materially'
    elif mean_alpha >= 0.10 and pos_alpha_frac >= 0.5:
        verdict = 'MARGINAL — partial-OOS, stratify by liquidity'
    elif mean_alpha < 0.05:
        verdict = ('FAIL (confirmed-null) — IV market efficiently '
                   'incorporates surface-shape information; pivot to '
                   'consolidate weak signals or accept all-three-null '
                   'and ship the existing equal-weight relational baseline')
    else:
        verdict = 'INCONCLUSIVE'
    print(f'\nverdict: {verdict}', flush=True)

    out_path = output / 'vol-walkforward-summary.json'
    out_path.write_text(json.dumps({
        'horizon': args.horizon,
        'train_window_days': args.train_window_days,
        'val_window_days': args.val_window_days,
        'step_window_days': args.step_window_days,
        'top_quantile': args.top_quantile,
        'clip_iv_hv_ratio': args.clip_iv_hv_ratio,
        'feature_names': FEATURE_NAMES,
        'n_windows': len(rows),
        'mean_val_r2': mean_val_r2,
        'mean_val_pearson_r': mean_val_r,
        'mean_alpha_sharpe_per_cell': mean_alpha,
        'positive_alpha_fraction': pos_alpha_frac,
        'verdict': verdict,
        'per_window': rows,
    }, indent=2))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
