"""v3 — regime-gated liquid-universe composition.

Tests whether composing v2 #2's OI restriction with a VIX-based regime
gate rescues deployability. The composition hypothesis:

  v2 #2 showed alpha collapses on top-200 OI BUT per-window alpha is
  strongly positive in stress regimes (w3/w4 = 2022-2023 post-Fed-pivot,
  alpha Sh +4.2 / +6.5) and strongly negative in calm regimes (w0/w1 =
  2021 calm-bull, alpha Sh −2.6 / −1.3). A regime gate that fires only
  in stress should preserve the liquid-name alpha while filtering out
  the calm-regime drag.

Per-rebal-bar gating (NOT window-level): at each rebal date t, check
VIX[t] > rolling_median(VIX, window=N). Only deploy on fired rebals.
This matches the vol-trade horizon (20 days = rebal cadence) and avoids
CFR Phase 4a's "57% of bars suspended kills compounding" problem
(short-vol PnL is non-compounding per-rebal in our accounting).

Pre-reg cuts (locked before run):
  PASS:     alpha Sharpe on fired rebals ≥ +0.30 AND fire-rate ∈ [20%, 80%]
            AND ≥ 4/6 fired-window subsets positive
  MARGINAL: alpha Sharpe ∈ [+0.10, +0.30] OR fire-rate outside band
  FAIL:     alpha Sharpe < +0.10 OR ≤ 2 windows with any fires

Sensitivity: gate lookback ∈ {60, 126, 252} trading days.
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
    oi_panel = raw[['date', 'symbol',
                    'puts_open_interest',
                    'calls_open_interest']].copy()
    oi_panel['total_oi'] = (
        oi_panel['puts_open_interest'].fillna(0)
        + oi_panel['calls_open_interest'].fillna(0))
    oi_panel['oi_rank'] = oi_panel.groupby('date')['total_oi'].rank(
        method='first', ascending=False)
    keep = oi_panel[oi_panel['oi_rank'] <= oi_top_n][
        ['date', 'symbol']].copy()
    keep['keep_flag'] = True
    out = merged.merge(keep, on=['date', 'symbol'], how='left')
    return out[out['keep_flag'].fillna(False)].drop(columns='keep_flag')


def build_vix_gate(raw: pd.DataFrame,
                   lookback: int) -> dict[pd.Timestamp, bool]:
    """For each date in raw, compute whether VIX[t] > rolling_median(VIX, lookback).
    Returns a dict (date → fired?). The VIX column in gauss314 is per-row
    (same value across symbols on a given date), so take the first per date.
    """
    vix_per_date = raw[['date', 'VIX']].drop_duplicates('date').sort_values('date')
    vix_per_date = vix_per_date.set_index('date')['VIX']
    rolling_med = vix_per_date.rolling(window=lookback, min_periods=lookback // 2).median()
    fired = vix_per_date > rolling_med
    return {d: bool(f) for d, f in fired.items() if not pd.isna(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=300)
    p.add_argument('--val-window-days',   type=int, default=120)
    p.add_argument('--step-window-days',  type=int, default=120)
    p.add_argument('--rebal-days', type=int, default=20)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--oi-top-n', type=int, default=200)
    p.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    p.add_argument('--gate-lookback-headline', type=int, default=60,
                   help='VIX rolling-median lookback (trading days) for '
                        'pre-reg headline verdict.')
    p.add_argument('--gate-lookback-sweep', type=int, nargs='+',
                   default=[60, 126, 252])
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
    print(f'  retained: {len(merged):,} rows', flush=True)

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    windows = _build_window_slices_by_date(
        dates, args.train_window_days, args.val_window_days,
        args.step_window_days)

    print(f'\n  walk-forward: {len(windows)} windows; gate sweep over '
          f'lookbacks {args.gate_lookback_sweep} (headline {args.gate_lookback_headline}d)',
          flush=True)

    # Pre-compute gate decisions per lookback
    print('Computing VIX rolling-median gates...', flush=True)
    gates_by_lookback = {
        lb: build_vix_gate(raw, lookback=lb)
        for lb in args.gate_lookback_sweep
    }
    for lb in args.gate_lookback_sweep:
        fire_total = sum(1 for f in gates_by_lookback[lb].values() if f)
        total = len(gates_by_lookback[lb])
        print(f'  lookback {lb}d: {fire_total}/{total} dates fire '
              f'({100*fire_total/total:.0f}%)', flush=True)

    # Run walk-forward, gating per rebal
    summary_by_lookback = {}
    for lookback in args.gate_lookback_sweep:
        gate = gates_by_lookback[lookback]
        print(f'\n{"=" * 124}\nGATE: VIX > {lookback}d rolling median\n{"=" * 124}',
              flush=True)
        print(f'{"win":>3s} {"val period":>25s} {"val r":>7s} '
              f'{"n_reb":>5s} {"fired":>5s} '
              f'{"fired gPnL":>11s} {"fired uPnL":>11s} {"fired α":>9s} '
              f'{"full α Sh":>10s} {"fired α Sh":>11s}', flush=True)
        print('-' * 124, flush=True)

        per_window: list[dict] = []
        full_panel_alpha: list[float] = []  # full panel = fired alpha or 0 for unfired
        fired_only_alpha: list[float] = []  # only fired rebals
        per_window_fired_alpha_sh: list[float] = []

        for w_idx, (ts_tr_lo, ts_tr_hi, ts_va_lo, ts_va_hi) in enumerate(windows):
            train = merged[(merged['date'] >= ts_tr_lo) &
                           (merged['date'] <= ts_tr_hi)]
            val   = merged[(merged['date'] >= ts_va_lo) &
                           (merged['date'] <= ts_va_hi)]
            if len(train) < 300 or len(val) < 100:
                continue

            X_tr = train[FEATURE_NAMES].values
            y_tr = train['iv_rv_gap'].values
            X_va = val[FEATURE_NAMES].values
            y_va = val['iv_rv_gap'].values

            pred = train_predictor(X_tr, y_tr, FEATURE_NAMES)
            val_pred = predict(pred, X_va)
            val_corr = float(np.corrcoef(val_pred, y_va)[0, 1])

            val_with_pred = val[['date', 'symbol']].copy()
            val_with_pred['pred_gap'] = val_pred
            val_with_realized = val[['date', 'symbol', 'iv_rv_gap']].copy()

            gated_arm = evaluate_portfolio_short_vol(
                val_with_pred, val_with_realized,
                top_k=args.top_k, friction_bps_roundtrip=0.0,
                rebal_days=args.rebal_days, arm_label='gated')
            universe_arm = evaluate_portfolio_short_vol(
                val_with_pred, val_with_realized,
                top_k=0, friction_bps_roundtrip=0.0,
                rebal_days=args.rebal_days, arm_label='universe')

            g_map = dict(zip(gated_arm.per_rebal_dates,
                             gated_arm.per_rebal_pnl_vol_points))
            u_map = dict(zip(universe_arm.per_rebal_dates,
                             universe_arm.per_rebal_pnl_vol_points))
            common = sorted(set(g_map) & set(u_map))

            # Gate each rebal date
            fired_g, fired_u, fired_dates = [], [], []
            full_g, full_u = [], []  # full panel: gated × fire-flag
            for d in common:
                # Convert string back to Timestamp for gate lookup
                d_ts = pd.Timestamp(d)
                fires = gate.get(d_ts, False)
                if fires:
                    fired_g.append(g_map[d])
                    fired_u.append(u_map[d])
                    fired_dates.append(d)
                    full_g.append(g_map[d])
                    full_u.append(u_map[d])
                else:
                    full_g.append(u_map[d])  # defer to universe baseline
                    full_u.append(u_map[d])

            fired_alpha = [g - u for g, u in zip(fired_g, fired_u)]
            full_alpha = [g - u for g, u in zip(full_g, full_u)]

            ann = float(np.sqrt(252.0 / args.rebal_days))
            if len(fired_alpha) > 1:
                a = np.asarray(fired_alpha, dtype=float)
                sd = float(a.std(ddof=1))
                fired_alpha_sh = (a.mean() / sd * ann) if sd > 1e-12 else 0.0
            else:
                fired_alpha_sh = 0.0
            if len(full_alpha) > 1:
                a = np.asarray(full_alpha, dtype=float)
                sd = float(a.std(ddof=1))
                full_alpha_sh = (a.mean() / sd * ann) if sd > 1e-12 else 0.0
            else:
                full_alpha_sh = 0.0

            per_window.append({
                'window_idx': w_idx,
                'val_start': str(ts_va_lo.date()),
                'val_end':   str(ts_va_hi.date()),
                'val_r': val_corr,
                'n_rebals_total': len(common),
                'n_rebals_fired': len(fired_dates),
                'fire_rate': len(fired_dates) / len(common) if common else 0.0,
                'fired_mean_alpha_pnl': (float(np.mean(fired_alpha))
                                         if fired_alpha else 0.0),
                'full_mean_alpha_pnl':  (float(np.mean(full_alpha))
                                         if full_alpha else 0.0),
                'fired_alpha_sharpe':    fired_alpha_sh,
                'full_alpha_sharpe':     full_alpha_sh,
                'fired_dates':           fired_dates,
            })
            per_window_fired_alpha_sh.append(fired_alpha_sh)
            full_panel_alpha.extend(full_alpha)
            fired_only_alpha.extend(fired_alpha)

            fpnl_g = float(np.mean(fired_g)) if fired_g else 0.0
            fpnl_u = float(np.mean(fired_u)) if fired_u else 0.0
            fpnl_a = float(np.mean(fired_alpha)) if fired_alpha else 0.0
            print(f'{w_idx:>3d} {ts_va_lo.date()}→{ts_va_hi.date()} '
                  f'{val_corr:>+7.4f} {len(common):>5d} {len(fired_dates):>5d} '
                  f'{fpnl_g:>+11.4f} {fpnl_u:>+11.4f} {fpnl_a:>+9.4f} '
                  f'{full_alpha_sh:>+10.3f} {fired_alpha_sh:>+11.3f}',
                  flush=True)

        # Pooled metrics for this lookback
        a_full = np.asarray(full_panel_alpha, dtype=float)
        a_fired = np.asarray(fired_only_alpha, dtype=float)
        ann = float(np.sqrt(252.0 / args.rebal_days))
        full_pooled_sh = (a_full.mean() / a_full.std(ddof=1) * ann
                          if a_full.size > 1 and a_full.std(ddof=1) > 1e-12 else 0.0)
        fired_pooled_sh = (a_fired.mean() / a_fired.std(ddof=1) * ann
                           if a_fired.size > 1 and a_fired.std(ddof=1) > 1e-12 else 0.0)
        n_pos_fired = sum(1 for w in per_window if w['fired_alpha_sharpe'] > 0)
        n_with_fires = sum(1 for w in per_window if w['n_rebals_fired'] >= 2)
        n_total = len(per_window)
        overall_fire_rate = (a_fired.size / a_full.size if a_full.size else 0.0)

        print(f'\n  pooled fired-only alpha Sharpe = {fired_pooled_sh:+.3f}',
              flush=True)
        print(f'  pooled full-panel alpha Sharpe = {full_pooled_sh:+.3f} '
              f'(closed-gate rebals defer to universe)', flush=True)
        print(f'  overall fire rate = {overall_fire_rate*100:.1f}% '
              f'({a_fired.size}/{a_full.size} rebals fired)', flush=True)
        print(f'  fired-positive windows = {n_pos_fired}/{n_total}', flush=True)
        print(f'  windows with >= 2 fires = {n_with_fires}/{n_total}', flush=True)

        summary_by_lookback[lookback] = {
            'pooled_fired_alpha_sharpe': fired_pooled_sh,
            'pooled_full_alpha_sharpe': full_pooled_sh,
            'overall_fire_rate': overall_fire_rate,
            'fired_positive_windows': n_pos_fired,
            'windows_with_fires': n_with_fires,
            'total_windows': n_total,
            'per_window': per_window,
        }

    # Headline verdict
    headline_lb = args.gate_lookback_headline
    if headline_lb not in summary_by_lookback:
        headline_lb = args.gate_lookback_sweep[0]
    headline = summary_by_lookback[headline_lb]

    print(f'\n{"=" * 90}', flush=True)
    print(f'HEADLINE @ VIX-{headline_lb}d-rolling-median gate, top-{args.oi_top_n} OI:',
          flush=True)
    print(f'  fired-only alpha Sharpe = {headline["pooled_fired_alpha_sharpe"]:+.3f}',
          flush=True)
    print(f'  fire rate = {headline["overall_fire_rate"]*100:.1f}%',
          flush=True)
    print(f'  windows with fires = {headline["windows_with_fires"]}/{headline["total_windows"]}',
          flush=True)
    print(f'  fired-positive windows = {headline["fired_positive_windows"]}/'
          f'{headline["total_windows"]}',
          flush=True)

    fired_sh = headline['pooled_fired_alpha_sharpe']
    fire_rate = headline['overall_fire_rate']
    fired_pos = headline['fired_positive_windows']
    with_fires = headline['windows_with_fires']
    n_total = headline['total_windows']

    in_band = 0.20 <= fire_rate <= 0.80
    if (fired_sh >= 0.30 and in_band
            and fired_pos >= int(np.ceil(0.66 * n_total))):
        verdict = (f'PASS — fired-alpha Sharpe {fired_sh:+.3f} ≥ +0.30, '
                   f'fire-rate {fire_rate*100:.0f}% in band [20%, 80%], '
                   f'{fired_pos}/{n_total} fired-positive')
    elif fired_sh >= 0.10 or in_band:
        verdict = (f'MARGINAL — fired-alpha Sharpe {fired_sh:+.3f} in '
                   f'[+0.10, +0.30] OR fire-rate {fire_rate*100:.0f}% outside band')
    else:
        verdict = (f'FAIL — fired-alpha Sharpe {fired_sh:+.3f} < +0.10 '
                   f'OR ≤ 2 windows with fires (got {with_fires})')

    print(f'\npre-reg verdict: {verdict}', flush=True)

    summary = {
        'oi_top_n': args.oi_top_n,
        'top_k': args.top_k,
        'rebal_days': args.rebal_days,
        'gate_lookback_headline': headline_lb,
        'gate_lookback_sweep': args.gate_lookback_sweep,
        'verdict': verdict,
        'summary_by_lookback': summary_by_lookback,
    }
    out_path = output / 'vol-walkforward-v3-regime-gated-summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
