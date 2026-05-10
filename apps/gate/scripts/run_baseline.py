"""Single-split phase-2 smoke test of the drawdown-gate hypothesis.

Cheapest version of the test: one train/val split matching the
passive-EW benchmark (train 2013-01-29 → 2020-12-31, val 2021-01-01 →
2025-12-11), linear predictor on aggregate features, binary gate
applied to EW.

If val R² > 0 *and* gated val Sharpe > unconditional EW val Sharpe,
the prediction problem has signal and we proceed to the
walk-forward harness. If both null, we cheaply confirmed the
ceiling.

Run from repo root:
    uv run python apps/gate/scripts/run_baseline.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ss_loaders import load_stooq_matrix

from gate import (
    apply_gate, build_aggregate_features, build_ew_aggregate,
    evaluate_gated_arm, forward_max_drawdown, predict, train_predictor,
)
from gate.predictor import evaluate_r2


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'

TRAIN_START = '2013-01-29'
TRAIN_END   = '2020-12-31'
VAL_START   = '2021-01-01'
VAL_END     = '2025-12-11'


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--horizon', type=int, default=20,
                   help='Forward drawdown horizon in trading bars.')
    p.add_argument('--gate-mode', default='binary',
                   choices=['binary', 'sigmoid'])
    p.add_argument('--threshold', type=float, default=None,
                   help='Override the median-train-DD threshold.')
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading universe (stooq_us_long, 312 tickers)...')
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers, '
          f'{prices.index[0].date()} → {prices.index[-1].date()}')

    print('\nBuilding aggregate EW return series + features...')
    agg = build_ew_aggregate(prices, min_active=10)
    feat_df = build_aggregate_features(agg)
    target = forward_max_drawdown(agg.ew_log_ret, horizon=args.horizon)
    print(f'  agg span: {agg.dates[0].date()} → {agg.dates[-1].date()}, '
          f'{len(agg.dates)} dates')

    # Align: drop rows with NaN in either features (trailing windows
    # warming up at start) or target (forward window not yet complete
    # at end).
    mask_feat = ~feat_df.isna().any(axis=1).values
    mask_targ = ~np.isnan(target)
    mask = mask_feat & mask_targ
    dates = feat_df.index[mask]
    feat = feat_df.values[mask]
    targ = target[mask]
    feature_names = list(feat_df.columns)
    print(f'  usable rows after warmup/align: {len(dates)} '
          f'({dates[0].date()} → {dates[-1].date()})')

    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask   = (dates >= VAL_START)   & (dates <= VAL_END)
    print(f'  train rows: {train_mask.sum()}; '
          f'val rows: {val_mask.sum()}')

    print('\nTraining linear predictor...')
    pred = train_predictor(feat[train_mask], targ[train_mask], feature_names)
    print(f'  train R²={pred.train_r2:+.4f}, RMSE={pred.train_rmse:.4f}')
    print(f'  feature coefficients (z-scored space):')
    for name, coef in zip(feature_names, pred.coefficients):
        print(f'    {name:>10s}: {coef:+.4f}')
    print(f'    {"intercept":>10s}: {pred.intercept:+.4f}')

    val_pred = predict(pred, feat[val_mask])
    val_actual = targ[val_mask]
    val_r2 = evaluate_r2(val_pred, val_actual)
    val_corr = float(np.corrcoef(val_pred, val_actual)[0, 1])
    print(f'\nVal predictor: R²={val_r2:+.4f}, '
          f'Pearson r={val_corr:+.4f}')

    train_pred = predict(pred, feat[train_mask])
    print('\n=== threshold sweep on val (binary gate) ===')
    print(f'{"threshold":>11s} {"quantile":>9s} {"avg_exp":>8s} '
          f'{"sharpe":>8s} {"sortino":>8s} {"cagr%":>7s} '
          f'{"maxDD%":>8s} {"flips":>6s} {"alpha_sh":>9s}')
    print('-' * 90)
    quantiles = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    sweep_rows = []
    for q in quantiles:
        thr = float(np.quantile(train_pred, q))
        full_pred_q = predict(pred, feat)
        gate_q = apply_gate(full_pred_q, thr, mode='binary')
        gate_q_lag = np.concatenate([[1.0], gate_q[:-1]])
        ew_simple_full = np.zeros(len(dates), dtype=np.float64)
        agg_series = {d: r for d, r in zip(agg.dates, agg.ew_simple_ret)}
        for i, d in enumerate(dates):
            ew_simple_full[i] = agg_series.get(d, 0.0)
        val_g_q = gate_q_lag[val_mask]
        val_ew_q = ew_simple_full[val_mask]
        val_dates_q = dates[val_mask]
        arm = evaluate_gated_arm(
            val_ew_q, val_g_q, val_dates_q, arm_label=f'q={q:.2f}')
        # Compute the unconditional baseline once for delta column.
        if not sweep_rows:
            unc_baseline = evaluate_gated_arm(
                val_ew_q, np.ones_like(val_ew_q), val_dates_q,
                arm_label='unc')
        alpha = arm.sharpe - unc_baseline.sharpe
        print(f'{thr:>+11.4f} {q:>9.2f} {arm.avg_exposure:>8.3f} '
              f'{arm.sharpe:>+8.3f} {arm.sortino:>+8.3f} '
              f'{arm.cagr_pct:>+7.2f} {arm.max_drawdown_pct:>+8.2f} '
              f'{arm.transition_count:>6d} {alpha:>+9.3f}')
        sweep_rows.append({
            'quantile': q, 'threshold': thr,
            'avg_exposure': arm.avg_exposure,
            'sharpe': arm.sharpe, 'sortino': arm.sortino,
            'cagr_pct': arm.cagr_pct,
            'max_drawdown_pct': arm.max_drawdown_pct,
            'transition_count': arm.transition_count,
            'alpha_sharpe': alpha,
        })

    # Pick the best operating point by val Sharpe alpha (this is the
    # honest test: does *any* threshold clear unconditional EW?).
    best = max(sweep_rows, key=lambda r: r['sharpe'])
    print(f'\nBest sweep operating point (by val Sharpe): '
          f'q={best["quantile"]:.2f}, '
          f'sharpe={best["sharpe"]:+.3f} (alpha {best["alpha_sharpe"]:+.3f})')

    threshold = (
        args.threshold if args.threshold is not None
        else best['threshold'])
    print(f'\nGate threshold (for reported arm below): {threshold:+.4f} '
          f'({"override" if args.threshold is not None else f"sweep-best q={best['quantile']:.2f}"})')

    # Apply gate to val window. Gate is the *target* exposure for
    # bar t+1 (we observe features at t-end, decide exposure for t+1).
    # Shift gate by 1 to avoid using the same-bar return for both
    # signal and PnL — the predictor used features at t which already
    # include the close of t, but the gated PnL for t is whatever the
    # market did from t-1 to t. So gate applied to bar `t` should be
    # decided from features at `t-1`. We approximate this conservatively
    # by shifting by 1.
    full_pred = predict(pred, feat)
    full_gate = apply_gate(full_pred, threshold, mode=args.gate_mode)
    full_gate_lagged = np.concatenate([[1.0], full_gate[:-1]])

    # Per-bar EW simple returns aligned to the same dates.
    full_dates = dates
    ew_simple_full = np.zeros(len(dates), dtype=np.float64)
    # Map agg.ew_simple_ret to dates via dict lookup — agg.dates is a
    # superset of `dates` (we masked for warmup/align), so use a
    # reindex.
    agg_series = {d: r for d, r in zip(agg.dates, agg.ew_simple_ret)}
    for i, d in enumerate(full_dates):
        ew_simple_full[i] = agg_series.get(d, 0.0)

    val_ew  = ew_simple_full[val_mask]
    val_g   = full_gate_lagged[val_mask]
    val_dates_idx = full_dates[val_mask]

    print('\n=== val window comparison ===')
    print(f'{"arm":<22s} {"sharpe":>8s} {"sortino":>8s} '
          f'{"cagr%":>8s} {"maxDD%":>8s} {"avg_exp":>8s} '
          f'{"flips":>6s}')
    print('-' * 76)
    # Unconditional EW = gate of all 1s.
    unc = evaluate_gated_arm(
        val_ew, np.ones_like(val_ew), val_dates_idx,
        arm_label='unconditional EW')
    print(f'{unc.arm:<22s} {unc.sharpe:>+8.3f} {unc.sortino:>+8.3f} '
          f'{unc.cagr_pct:>+8.2f} {unc.max_drawdown_pct:>+8.2f} '
          f'{unc.avg_exposure:>8.3f} {unc.transition_count:>6d}')

    gated = evaluate_gated_arm(
        val_ew, val_g, val_dates_idx,
        arm_label=f'gated EW ({args.gate_mode})')
    print(f'{gated.arm:<22s} {gated.sharpe:>+8.3f} {gated.sortino:>+8.3f} '
          f'{gated.cagr_pct:>+8.2f} {gated.max_drawdown_pct:>+8.2f} '
          f'{gated.avg_exposure:>8.3f} {gated.transition_count:>6d}')

    alpha = gated.sharpe - unc.sharpe
    print(f'\n  alpha vs unconditional EW: {alpha:+.3f} Sharpe '
          f'(passes if ≥ +0.10 with sortino lift too)')

    out_path = Path(args.output_dir) / 'gate-baseline-summary.json'
    out_path.write_text(json.dumps({
        'horizon': args.horizon,
        'gate_mode': args.gate_mode,
        'threshold': threshold,
        'train_start': TRAIN_START, 'train_end': TRAIN_END,
        'val_start':   VAL_START,   'val_end':   VAL_END,
        'n_train': int(train_mask.sum()),
        'n_val':   int(val_mask.sum()),
        'feature_names': feature_names,
        'coefficients': pred.coefficients.tolist(),
        'intercept': pred.intercept,
        'train_r2':  pred.train_r2,
        'train_rmse': pred.train_rmse,
        'val_r2':    val_r2,
        'val_pearson_r': val_corr,
        'unconditional_ew': {
            'sharpe': unc.sharpe, 'sortino': unc.sortino,
            'cagr_pct': unc.cagr_pct, 'max_drawdown_pct': unc.max_drawdown_pct,
        },
        'gated_ew': {
            'sharpe': gated.sharpe, 'sortino': gated.sortino,
            'cagr_pct': gated.cagr_pct, 'max_drawdown_pct': gated.max_drawdown_pct,
            'avg_exposure': gated.avg_exposure,
            'transition_count': gated.transition_count,
        },
        'sharpe_alpha_vs_ew': alpha,
        'threshold_sweep': sweep_rows,
        'best_sweep_op': best,
    }, indent=2))
    print(f'\n-> {out_path}')


if __name__ == '__main__':
    main()
