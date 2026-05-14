"""v2 #2 — OI-restricted walk-forward.

Filter the predictor's pool to top-N by `puts_open_interest +
calls_open_interest` per date BEFORE top-K pick selection. Tests
whether the v1 alpha survives restriction to liquid options (deployment
realism) and whether the v2 #1 sub-finding (val r 37% higher on Stooq-
overlap universe) generalizes to an explicit liquidity filter.

Pre-reg cuts (v2 #2, locked before run):
  PASS:     pooled alpha Sharpe under OI-200 ≥ 0.5× v1's unrestricted
            Sharpe AND >= 4/5 windows positive AND alpha Sharpe >= +0.30
  MARGINAL: alpha Sharpe ∈ [+0.10, +0.30] OR positive in 3/5 windows
  FAIL:     alpha Sharpe < +0.10 OR ≤ 2/5 positive

Default OI threshold: top-200 names by total OI per date. The audit
suggested top-100 OI which is tighter; we use 200 to retain more
sample for the regression while still being deployable.

Run:
    uv run python apps/vol/scripts/run_walkforward_v2_oi.py
    uv run python apps/vol/scripts/run_walkforward_v2_oi.py --oi-top-n 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (
    FEATURE_NAMES, build_vol_features, evaluate_portfolio_short_vol,
    forward_iv_rv_gap, load_gauss314_full, predict, train_predictor,
)
from vol.predictor import evaluate_r2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_window_slices_by_date(
    dates: pd.DatetimeIndex, train_days: int, val_days: int, step_days: int,
) -> list[tuple]:
    if len(dates) < train_days + val_days:
        return []
    out = []
    i = 0
    while i + train_days + val_days <= len(dates):
        out.append((
            dates[i], dates[i + train_days - 1],
            dates[i + train_days], dates[i + train_days + val_days - 1],
        ))
        i += step_days
    return out


def apply_oi_filter(merged: pd.DataFrame, raw: pd.DataFrame,
                    oi_top_n: int) -> pd.DataFrame:
    """Filter `merged` (per-(date, symbol) panel) to top-N by total
    open interest per date. `raw` provides the OI columns (gauss314)
    that build_vol_features already consumed via `oi_imbalance`.
    """
    oi_panel = raw[['date', 'symbol',
                    'puts_open_interest',
                    'calls_open_interest']].copy()
    oi_panel['total_oi'] = (
        oi_panel['puts_open_interest'].fillna(0)
        + oi_panel['calls_open_interest'].fillna(0))
    # Rank within each date; keep top-N.
    oi_panel['oi_rank'] = oi_panel.groupby('date')['total_oi'].rank(
        method='first', ascending=False)
    keep = oi_panel[oi_panel['oi_rank'] <= oi_top_n][
        ['date', 'symbol']].copy()
    keep['keep_flag'] = True
    out = merged.merge(keep, on=['date', 'symbol'], how='left')
    n_before = len(out)
    out = out[out['keep_flag'].fillna(False)].drop(columns='keep_flag')
    n_after = len(out)
    print(f'  OI-top-{oi_top_n} filter: {n_before:,} → {n_after:,} '
          f'({100*n_after/n_before:.1f}% retained)', flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=300)
    p.add_argument('--val-window-days',   type=int, default=120)
    p.add_argument('--step-window-days',  type=int, default=120)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--top-k', type=int, default=50,
                   help='Top-K pick size per rebal AFTER OI filter. '
                        '50 default (smaller than v1\'s 100 because the '
                        'post-filter universe is ~200 names — top-50 is '
                        'top-quartile-of-liquid).')
    p.add_argument('--oi-top-n', type=int, default=200,
                   help='Keep top-N names by total OI per date.')
    p.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading gauss314 full schema...', flush=True)
    t0 = time.perf_counter()
    raw = load_gauss314_full()
    print(f'  {len(raw):,} rows in {time.perf_counter()-t0:.1f}s', flush=True)

    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)

    merged = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap'])
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        merged[col] = merged[col].clip(
            lower=-args.clip_iv_hv_ratio, upper=args.clip_iv_hv_ratio)
    print(f'  usable rows after merge + clip: {len(merged):,}', flush=True)

    print(f'\nApplying OI-top-{args.oi_top_n} filter per date...', flush=True)
    merged = apply_oi_filter(merged, raw, args.oi_top_n)

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    windows = _build_window_slices_by_date(
        dates, args.train_window_days, args.val_window_days,
        args.step_window_days)
    print(f'\n  walk-forward: {len(windows)} windows '
          f'(rebal_days={args.rebal_days}, top_k={args.top_k}, '
          f'oi_top_n={args.oi_top_n})', flush=True)

    print('\n' + '=' * 120, flush=True)
    print(f'{"win":>3s} {"val period":>25s} {"val r":>7s} {"n_reb":>5s} '
          f'{"gated PnL":>10s} {"univ PnL":>10s} {"alpha PnL":>10s} '
          f'{"gated Sh":>9s} {"univ Sh":>9s} {"ALPHA Sh":>10s} {"pos":>3s}',
          flush=True)
    print('-' * 120, flush=True)

    per_window: list[dict] = []
    per_window_alpha_pnls: list[list[float]] = []
    per_window_alpha_sharpe: list[float] = []

    for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
        train = merged[(merged['date'] >= ts_tr_lo) &
                       (merged['date'] <= ts_tr_hi)]
        val   = merged[(merged['date'] >= ts_va_lo) &
                       (merged['date'] <= ts_va_hi)]
        if len(train) < 500 or len(val) < 200:
            continue

        X_tr = train[FEATURE_NAMES].values
        y_tr = train['iv_rv_gap'].values
        X_va = val[FEATURE_NAMES].values
        y_va = val['iv_rv_gap'].values

        pred = train_predictor(X_tr, y_tr, FEATURE_NAMES)
        val_pred = predict(pred, X_va)
        val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])
        val_r2 = evaluate_r2(val_pred, y_va)

        val_with_pred = val[['date', 'symbol']].copy()
        val_with_pred['pred_gap'] = val_pred
        val_with_realized = val[['date', 'symbol', 'iv_rv_gap']].copy()

        gated = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=args.top_k, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='gated')
        universe = evaluate_portfolio_short_vol(
            val_with_pred, val_with_realized,
            top_k=0, friction_bps_roundtrip=0.0,
            rebal_days=args.rebal_days, arm_label='universe')

        g_map = dict(zip(gated.per_rebal_dates,
                         gated.per_rebal_pnl_vol_points))
        u_map = dict(zip(universe.per_rebal_dates,
                         universe.per_rebal_pnl_vol_points))
        common_dates = sorted(set(g_map) & set(u_map))
        g_aligned = [g_map[d] for d in common_dates]
        u_aligned = [u_map[d] for d in common_dates]
        alpha_aligned = [g - u for g, u in zip(g_aligned, u_aligned)]

        a = np.asarray(alpha_aligned, dtype=float)
        ann_factor = float(np.sqrt(252.0 / args.rebal_days))
        if a.size > 1:
            sd = float(a.std(ddof=1))
            alpha_sharpe = (a.mean() / sd * ann_factor) if sd > 1e-12 else 0.0
        else:
            alpha_sharpe = 0.0

        per_window_alpha_pnls.append(alpha_aligned)
        per_window_alpha_sharpe.append(alpha_sharpe)

        per_window.append({
            'window_idx': w_idx,
            'val_start': str(ts_va_lo.date()),
            'val_end':   str(ts_va_hi.date()),
            'val_r2': val_r2,
            'val_pearson_r': val_corr,
            'n_rebals': len(common_dates),
            'top_k': args.top_k,
            'oi_top_n': args.oi_top_n,
            'gated_mean_pnl_vol_pts': float(np.mean(g_aligned)) if g_aligned else 0.0,
            'universe_mean_pnl_vol_pts': float(np.mean(u_aligned)) if u_aligned else 0.0,
            'alpha_mean_pnl_vol_pts': float(np.mean(alpha_aligned)) if alpha_aligned else 0.0,
            'gated_sharpe_gross_annualized': gated.annualized_sharpe,
            'universe_sharpe_gross_annualized': universe.annualized_sharpe,
            'alpha_sharpe_annualized': alpha_sharpe,
            'per_rebal_dates': common_dates,
            'per_rebal_alpha_pnls': alpha_aligned,
        })

        pos_mark = '+' if alpha_sharpe > 0 else '-'
        print(f'{w_idx:>3d} {ts_va_lo.date()}→{ts_va_hi.date()} '
              f'{val_corr:>+7.4f} {len(common_dates):>5d} '
              f'{float(np.mean(g_aligned)):>+10.4f} '
              f'{float(np.mean(u_aligned)):>+10.4f} '
              f'{float(np.mean(alpha_aligned)):>+10.4f} '
              f'{gated.annualized_sharpe:>+9.3f} '
              f'{universe.annualized_sharpe:>+9.3f} '
              f'{alpha_sharpe:>+10.3f} '
              f'{pos_mark:>3s}', flush=True)

    print('\n' + '=' * 120, flush=True)
    if not per_window:
        print('No usable windows.', flush=True)
        return

    all_alpha = [a for w in per_window_alpha_pnls for a in w]
    arr_alpha = np.asarray(all_alpha, dtype=float)
    ann_factor = float(np.sqrt(252.0 / args.rebal_days))
    if arr_alpha.size > 1:
        sd = float(arr_alpha.std(ddof=1))
        pooled_alpha_sharpe = (arr_alpha.mean() / sd * ann_factor
                               if sd > 1e-12 else 0.0)
    else:
        pooled_alpha_sharpe = 0.0
    pooled_alpha_mean = float(arr_alpha.mean()) if arr_alpha.size else 0.0
    n_pos = sum(1 for sh in per_window_alpha_sharpe if sh > 0)
    n_total = len(per_window_alpha_sharpe)
    mean_val_r = float(np.mean([w['val_pearson_r'] for w in per_window]))

    print(f'\n=== HEADLINE (OI-top-{args.oi_top_n}, top-K={args.top_k}): '
          f'pooled alpha Sharpe = {pooled_alpha_sharpe:+.3f}, '
          f'{n_pos}/{n_total} positive ===', flush=True)
    print(f'  mean val Pearson r = {mean_val_r:+.4f}', flush=True)
    print(f'  pooled alpha mean PnL per rebal = {pooled_alpha_mean:+.4f} vol pts',
          flush=True)

    # Pre-reg verdict (compare to v1's alpha Sharpe +5.86).
    v1_alpha_sharpe = 5.855
    sh_ratio = (pooled_alpha_sharpe / v1_alpha_sharpe
                if v1_alpha_sharpe > 0 else 0)
    print(f'  ratio to v1 unrestricted alpha Sharpe = {sh_ratio:.2f}x',
          flush=True)

    if (pooled_alpha_sharpe >= 0.30
            and n_pos >= int(np.ceil(0.8 * n_total))
            and sh_ratio >= 0.5):
        verdict = (f'PASS — OI-restricted alpha Sharpe {pooled_alpha_sharpe:+.3f} '
                   f'is {sh_ratio:.2f}x v1 unrestricted, {n_pos}/{n_total} positive')
    elif pooled_alpha_sharpe >= 0.10 and n_pos >= int(np.ceil(0.6 * n_total)):
        verdict = (f'MARGINAL — alpha Sharpe {pooled_alpha_sharpe:+.3f} in '
                   f'[+0.10, +0.30] OR ratio to v1 ({sh_ratio:.2f}x) below 0.5x')
    else:
        verdict = (f'FAIL — alpha Sharpe {pooled_alpha_sharpe:+.3f} < +0.10 '
                   f'OR positive windows {n_pos}/{n_total} insufficient')
    print(f'\npre-reg verdict: {verdict}', flush=True)

    summary = {
        'horizon': args.horizon,
        'train_window_days': args.train_window_days,
        'val_window_days': args.val_window_days,
        'step_window_days': args.step_window_days,
        'rebal_days': args.rebal_days,
        'top_k': args.top_k,
        'oi_top_n': args.oi_top_n,
        'clip_iv_hv_ratio': args.clip_iv_hv_ratio,
        'n_windows': n_total,
        'mean_val_pearson_r': mean_val_r,
        'pooled_alpha_sharpe_annualized': pooled_alpha_sharpe,
        'pooled_alpha_mean_pnl_vol_pts': pooled_alpha_mean,
        'positive_alpha_windows': n_pos,
        'positive_alpha_windows_total': n_total,
        'v1_unrestricted_alpha_sharpe': v1_alpha_sharpe,
        'ratio_to_v1': sh_ratio,
        'verdict': verdict,
        'per_window': per_window,
        'per_window_alpha_sharpe': per_window_alpha_sharpe,
    }

    suffix = f'-oi{args.oi_top_n}-topk{args.top_k}'
    out_path = output / f'vol-walkforward-v2-oi{suffix}-summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
