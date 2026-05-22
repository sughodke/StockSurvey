"""Walk-forward eval of the drawdown-gate hypothesis.

Honest test that the single-split [`run_baseline.py`](run_baseline.py)
can't deliver: 6 rolling train/val windows, threshold chosen on
*train* per window (no peeking at val), reported alpha vs the
within-window unconditional EW baseline.

If alpha is positive in ≥4/6 windows with mean alpha ≥ +0.10
Sharpe, the gate hypothesis has signal at this universe / horizon
and we can build a deployable strategy. Otherwise `confirmed-null`
and we move to [`apps/pairs`](../../docs/TODO/apps-pairs.md).

Run from repo root:
    uv run python apps/gate/scripts/run_walkforward.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix

from gate import (
    apply_gate, build_aggregate_features, build_ew_aggregate,
    evaluate_gated_arm, forward_max_drawdown, predict, train_predictor,
)
from gate.predictor import evaluate_r2


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _slice_window(
    feat: np.ndarray, targ: np.ndarray, dates: np.ndarray,
    start_idx: int, end_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return feat[start_idx:end_idx], targ[start_idx:end_idx], dates[start_idx:end_idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--train-window-days', type=int, default=1260,  # ~5y
                   help='Train window length in trading bars (~5 years).')
    p.add_argument('--val-window-days',   type=int, default=780,   # ~3y
                   help='Val window length in trading bars (~3 years).')
    p.add_argument('--step-window-days',  type=int, default=780,
                   help='Step between successive window starts.')
    p.add_argument('--threshold-quantile', type=float, default=0.90,
                   help='Train-pred quantile for binary gate threshold. '
                        '0.90 → flat when predicted DD is in top 10%% of '
                        'train predictions.')
    p.add_argument('--gate-mode', default='binary',
                   choices=['binary', 'sigmoid'])
    p.add_argument('--with-macro', action='store_true',
                   help='Include FRED macro features (fed_funds, '
                        'credit_baa, m2_yoy, real_yield_10y, vix) in '
                        'the predictor stack. Drops the Pearson-r-noise '
                        'slope_10y_3m feature per the macro-regime-'
                        'diagnostic finding. Adds ~3y to the usable '
                        'date floor (TIPS DFII10 starts 2003).')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    p.add_argument('--dump-returns', action='store_true',
                   help='Concatenate the per-window OOS gated + '
                        'unconditional daily net-return streams and write '
                        'them to Output/gate-returns.npz, for the '
                        'cross-arc deflated-Sharpe harness '
                        '(ss_portfolio.standardize_oos).')
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading universe (stooq_us_long)...')
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')

    print('\nBuilding aggregate + features + target...')
    agg = build_ew_aggregate(prices, min_active=10)
    feat_df = build_aggregate_features(agg)
    target = forward_max_drawdown(agg.ew_log_ret, horizon=args.horizon)

    if args.with_macro:
        # Per macro-regime-diagnostic finding: 5 of 6 FRED features
        # carry directional signal (Pearson r [+0.34, +0.49] or
        # -0.38). Drop the noise feature (slope_10y_3m, |r|=0.06).
        from ss_macro import load_macro_panel
        print('  loading macro panel from FRED...')
        macro = load_macro_panel(target_index=feat_df.index)
        macro_keep = ['fed_funds', 'credit_baa', 'm2_yoy',
                      'real_yield_10y', 'vix']
        macro_features = macro[macro_keep]
        # Concat to gate's existing 10-feature aggregate stack.
        feat_df = pd.concat([feat_df, macro_features], axis=1)
        print(f'  added {len(macro_keep)} macro features → '
              f'feature stack now {len(feat_df.columns)} cols '
              f'({list(feat_df.columns)})')

    feature_names = list(feat_df.columns)

    mask = (~feat_df.isna().any(axis=1).values) & (~np.isnan(target))
    dates = feat_df.index[mask]
    feat = feat_df.values[mask]
    targ = target[mask]
    ew_simple = np.array([
        agg.ew_simple_ret[np.searchsorted(agg.dates, d)]
        for d in dates
    ])
    print(f'  usable rows: {len(dates)} '
          f'({dates[0].date()} → {dates[-1].date()})')

    n = len(dates)
    train_w = args.train_window_days
    val_w   = args.val_window_days
    step    = args.step_window_days
    windows = []
    start = 0
    while start + train_w + val_w <= n:
        windows.append((start, start + train_w, start + train_w + val_w))
        start += step
    if not windows:
        raise SystemExit(
            f'no windows fit: have {n} rows but need '
            f'train+val={train_w + val_w}')
    print(f'  walk-forward: {len(windows)} windows '
          f'(train={train_w} bars, val={val_w} bars, step={step} bars)')

    rows = []
    oos_gated_ret: list[np.ndarray] = []
    oos_unc_ret: list[np.ndarray] = []
    oos_dates: list[np.ndarray] = []
    print(f'\n{"win":>3s} {"train_dates":>23s} {"val_dates":>23s} '
          f'{"train_R2":>9s} {"val_R2":>8s} {"val_r":>7s} '
          f'{"avg_exp":>7s} {"unc_sh":>7s} {"gat_sh":>7s} '
          f'{"alpha":>7s} {"o_dd_α":>7s} {"o_day_α":>8s} {"flips":>5s}')
    print('-' * 152)
    for w_idx, (lo, mid, hi) in enumerate(windows):
        train_X, train_y, train_dates = _slice_window(feat, targ, dates, lo, mid)
        val_X,   val_y,   val_dates   = _slice_window(feat, targ, dates, mid, hi)

        pred = train_predictor(train_X, train_y, feature_names)
        train_pred = predict(pred, train_X)
        threshold = float(np.quantile(train_pred, args.threshold_quantile))

        val_pred = predict(pred, val_X)
        val_r2 = evaluate_r2(val_pred, val_y)
        val_corr = float(np.corrcoef(val_pred, val_y)[0, 1])

        # Apply gate: lag by 1 to avoid same-bar peek (decision at t-1
        # close governs t exposure).
        val_gate = apply_gate(val_pred, threshold, mode=args.gate_mode)
        val_gate_lagged = np.concatenate([[1.0], val_gate[:-1]])
        val_ew = ew_simple[mid:hi]

        unc = evaluate_gated_arm(
            val_ew, np.ones_like(val_ew), val_dates,
            arm_label='unc')
        gated = evaluate_gated_arm(
            val_ew, val_gate_lagged, val_dates,
            arm_label='gated')
        alpha = gated.sharpe - unc.sharpe

        # Accumulate the OOS net-return streams for the cross-arc DSR
        # harness. Val windows are contiguous + non-overlapping at the
        # default step==val_w, so concatenation is a clean OOS series.
        oos_gated_ret.append(gated.daily_ret)
        oos_unc_ret.append(unc.daily_ret)
        oos_dates.append(np.asarray([np.datetime64(d) for d in val_dates]))

        # --- Oracle arm 1: perfect-DD-predictor (uses realized 20-day
        # forward DD as the "prediction"; threshold set on train-realized
        # DDs at the same quantile as the regular arm, so the oracle
        # answers "what if our predictor had perfect 20-day DD foresight,
        # with the same threshold methodology?"). No look-ahead in the
        # threshold itself; only the per-bar gate signal uses future data.
        oracle_dd_threshold = float(np.quantile(train_y, args.threshold_quantile))
        oracle_dd_gate = apply_gate(val_y, oracle_dd_threshold, mode=args.gate_mode)
        oracle_dd_gate_lagged = np.concatenate([[1.0], oracle_dd_gate[:-1]])
        oracle_dd_arm = evaluate_gated_arm(
            val_ew, oracle_dd_gate_lagged, val_dates, arm_label='oracle_dd')
        oracle_dd_alpha = oracle_dd_arm.sharpe - unc.sharpe

        # --- Oracle arm 2: perfect-daily-direction (gate fires iff next
        # day's EW return is negative; strict upper bound on ANY binary
        # gate selector regardless of prediction target).
        oracle_day_gate_effective = np.where(val_ew > 0, 1.0, 0.0)
        oracle_day_arm = evaluate_gated_arm(
            val_ew, oracle_day_gate_effective, val_dates,
            arm_label='oracle_day')
        oracle_day_alpha = oracle_day_arm.sharpe - unc.sharpe

        rows.append({
            'window_idx': w_idx,
            'train_start': str(train_dates[0].date()),
            'train_end':   str(train_dates[-1].date()),
            'val_start':   str(val_dates[0].date()),
            'val_end':     str(val_dates[-1].date()),
            'train_r2': pred.train_r2,
            'val_r2':   val_r2,
            'val_pearson_r': val_corr,
            'threshold': threshold,
            'avg_exposure': gated.avg_exposure,
            'transition_count': gated.transition_count,
            'unc_sharpe': unc.sharpe,
            'gated_sharpe': gated.sharpe,
            'alpha_sharpe': alpha,
            'oracle_dd_sharpe': oracle_dd_arm.sharpe,
            'oracle_dd_alpha': oracle_dd_alpha,
            'oracle_dd_avg_exposure': oracle_dd_arm.avg_exposure,
            'oracle_day_sharpe': oracle_day_arm.sharpe,
            'oracle_day_alpha': oracle_day_alpha,
            'oracle_day_avg_exposure': oracle_day_arm.avg_exposure,
            'unc_max_dd_pct': unc.max_drawdown_pct,
            'gated_max_dd_pct': gated.max_drawdown_pct,
        })
        print(f'{w_idx:>3d} '
              f'{train_dates[0].date()}→{train_dates[-1].date()} '
              f'{val_dates[0].date()}→{val_dates[-1].date()} '
              f'{pred.train_r2:>+9.4f} {val_r2:>+8.4f} {val_corr:>+7.3f} '
              f'{gated.avg_exposure:>7.3f} {unc.sharpe:>+7.3f} '
              f'{gated.sharpe:>+7.3f} {alpha:>+7.3f} '
              f'{oracle_dd_alpha:>+7.3f} {oracle_day_alpha:>+8.3f} '
              f'{gated.transition_count:>5d}')

    print('\n' + '=' * 96)
    mean_alpha = float(np.mean([r['alpha_sharpe'] for r in rows]))
    pos_alpha_frac = float(np.mean([r['alpha_sharpe'] > 0 for r in rows]))
    mean_val_r2 = float(np.mean([r['val_r2'] for r in rows]))
    mean_val_r = float(np.mean([r['val_pearson_r'] for r in rows]))
    mean_unc_sh = float(np.mean([r['unc_sharpe'] for r in rows]))
    mean_gat_sh = float(np.mean([r['gated_sharpe'] for r in rows]))
    print(f'mean val R² = {mean_val_r2:+.4f} ; mean val Pearson r = '
          f'{mean_val_r:+.4f}')
    print(f'mean unc EW Sharpe = {mean_unc_sh:+.3f} ; mean gated Sharpe = '
          f'{mean_gat_sh:+.3f}')
    print(f'mean alpha = {mean_alpha:+.3f} Sharpe ; '
          f'positive-alpha windows = {pos_alpha_frac:.2f} '
          f'({int(round(pos_alpha_frac * len(rows)))}/{len(rows)})')

    # Oracle aggregates.
    mean_oracle_dd_alpha = float(np.mean([r['oracle_dd_alpha'] for r in rows]))
    pos_oracle_dd_frac = float(np.mean(
        [r['oracle_dd_alpha'] > 0 for r in rows]))
    mean_oracle_dd_sh = float(np.mean([r['oracle_dd_sharpe'] for r in rows]))
    mean_oracle_dd_exp = float(np.mean(
        [r['oracle_dd_avg_exposure'] for r in rows]))
    mean_oracle_day_alpha = float(np.mean(
        [r['oracle_day_alpha'] for r in rows]))
    pos_oracle_day_frac = float(np.mean(
        [r['oracle_day_alpha'] > 0 for r in rows]))
    mean_oracle_day_sh = float(np.mean([r['oracle_day_sharpe'] for r in rows]))
    mean_oracle_day_exp = float(np.mean(
        [r['oracle_day_avg_exposure'] for r in rows]))
    print(f'\nORACLE perfect-DD-predictor (uses realized forward DD, '
          f'train-realized-DD quantile threshold):')
    print(f'  mean Sharpe = {mean_oracle_dd_sh:+.3f}  '
          f'mean alpha = {mean_oracle_dd_alpha:+.3f}  '
          f'pos-α windows = {pos_oracle_dd_frac:.2f} '
          f'({int(round(pos_oracle_dd_frac * len(rows)))}/{len(rows)})  '
          f'avg exposure = {mean_oracle_dd_exp:.3f}')
    print(f'ORACLE perfect-daily-direction (gate iff next-day EW > 0, '
          f'strict upper bound on any gate):')
    print(f'  mean Sharpe = {mean_oracle_day_sh:+.3f}  '
          f'mean alpha = {mean_oracle_day_alpha:+.3f}  '
          f'pos-α windows = {pos_oracle_day_frac:.2f} '
          f'({int(round(pos_oracle_day_frac * len(rows)))}/{len(rows)})  '
          f'avg exposure = {mean_oracle_day_exp:.3f}')
    print(f'\nDecomposition: heuristic captures '
          f'{100*mean_alpha/max(mean_oracle_dd_alpha, 1e-9):.1f}% of '
          f'perfect-DD-predictor ceiling, '
          f'{100*mean_alpha/max(mean_oracle_day_alpha, 1e-9):.1f}% of '
          f'perfect-daily-direction ceiling.')

    # Pre-registered cuts (matched to TODO/different-prediction-problem
    # convention).
    if mean_alpha >= 0.10 and pos_alpha_frac >= 4 / 6:
        verdict = ('PASS — gate adds alpha; build full deployment '
                   'in apps/gate (live integration, sigmoid mode '
                   'sweep, sector-restricted variants)')
    elif abs(mean_alpha) <= 0.05:
        verdict = ('FAIL (clean null) — gate adds no signal; pivot '
                   'to apps/pairs as the next prediction problem')
    elif mean_alpha < -0.05:
        verdict = ('FAIL (worse than null) — gate destroys Sharpe; '
                   'predictor R² may be negative on val (overfit).')
    else:
        verdict = ('INCONCLUSIVE — between thresholds; stratify by '
                   'window before deciding')
    print(f'\nverdict: {verdict}')

    suffix = '-with-macro' if args.with_macro else ''
    out_path = output / f'gate-walkforward-summary{suffix}.json'
    out_path.write_text(json.dumps({
        'horizon': args.horizon,
        'gate_mode': args.gate_mode,
        'threshold_quantile': args.threshold_quantile,
        'train_window_days': args.train_window_days,
        'val_window_days': args.val_window_days,
        'step_window_days': args.step_window_days,
        'feature_names': feature_names,
        'n_windows': len(rows),
        'mean_val_r2': mean_val_r2,
        'mean_val_pearson_r': mean_val_r,
        'mean_unc_sharpe': mean_unc_sh,
        'mean_gated_sharpe': mean_gat_sh,
        'mean_alpha_sharpe': mean_alpha,
        'positive_alpha_fraction': pos_alpha_frac,
        'verdict': verdict,
        'per_window': rows,
    }, indent=2))
    print(f'\n-> {out_path}')

    if args.dump_returns:
        ret_path = output / f'gate-returns{suffix}.npz'
        np.savez(
            ret_path,
            gated_ret=np.concatenate(oos_gated_ret),
            unc_ret=np.concatenate(oos_unc_ret),
            dates=np.concatenate(oos_dates),
            periods_per_year=np.float64(252.0),
            threshold_quantile=np.float64(args.threshold_quantile),
            gate_mode=np.str_(args.gate_mode),
        )
        print(f'-> {ret_path} (OOS daily net-return streams for DSR)')


if __name__ == '__main__':
    main()
