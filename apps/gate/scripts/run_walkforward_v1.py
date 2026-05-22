"""Gate v1 — nonlinear predictor + cross-sectional dispersion features.

The hindsight oracle (gate-drawdown-v0, 2026-05-14) showed the v0 OLS gate
captures only 17.2% of a perfect-DD-predictor's +0.387 mean-alpha ceiling —
i.e. the gate is **predictor-bound, not architecture-bound**. v1 attacks the
predictor: (a) swap OLS for a nonlinear head (MLP / gradient-boosted), which
can extrapolate the tail-drawdown nonlinearity OLS misses (it captured only
2% of the COVID-window oracle alpha); (b) add price-derived **cross-sectional
dispersion** features (cross-sectional return std, breadth above 200dma) —
scale-invariant, so they avoid the OOS distribution-shift that sank macro
features inside the predictor (macro v1a). Macro stays OUT of the predictor.

Pre-registered cuts (from gate-drawdown-v0): PASS if mean alpha >= +0.20 AND
>= 5/6 positive windows (captures >= 50% of the +0.387 oracle ceiling);
FAIL if mean alpha < +0.10 or <= 2/6. Baseline: v0 OLS, mean alpha +0.067, 4/6.

    uv run python apps/gate/scripts/run_walkforward_v1.py
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
    evaluate_gated_arm, forward_max_drawdown,
)

REPO = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
OUT = REPO / 'Output'


def _dispersion_features(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time cross-sectional dispersion (price-derived, scale-invariant)."""
    logp = np.log(prices.where(prices > 0))
    ret20 = logp - logp.shift(20)                    # trailing 20d return per name
    cross_std = ret20.std(axis=1, ddof=0)            # dispersion across names
    ma200 = prices.rolling(200, min_periods=100).mean()
    breadth200 = (prices > ma200).mean(axis=1)       # fraction above 200dma
    df = pd.DataFrame({'xs_ret_std20': cross_std, 'breadth_200': breadth200})
    return df.reindex(dates)


def _train_predict(kind, Xtr, ytr, Xva):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    if kind == 'ols':
        from sklearn.linear_model import LinearRegression
        m = LinearRegression()
    elif kind == 'mlp':
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=1000,
                         early_stopping=True, random_state=0)
    else:  # hgb
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                          l2_regularization=1.0, random_state=0)
    m.fit(Xtr_s, ytr)
    return m.predict(Xtr_s), m.predict(Xva_s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days', type=int, default=780)
    p.add_argument('--step-window-days', type=int, default=780)
    p.add_argument('--threshold-quantile', type=float, default=0.95)
    args = p.parse_args()

    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    rows = manifest['tickers'] if isinstance(manifest, dict) else manifest
    universe = sorted(t['ticker'].upper() for t in rows)
    prices, _, _, _ = load_stooq_matrix(str(STOOQ_SUBSET), tickers=universe,
                                        min_history=210, start_date='2000-01-01',
                                        end_date='2025-12-11')
    print(f'loaded {prices.shape[1]} tickers, {prices.index[0].date()} → {prices.index[-1].date()}')

    agg = build_ew_aggregate(prices, min_active=10)
    base = build_aggregate_features(agg)
    disp = _dispersion_features(prices, base.index)
    feat_df = pd.concat([base, disp], axis=1)
    target = forward_max_drawdown(agg.ew_log_ret, horizon=args.horizon)
    feature_names = list(feat_df.columns)
    print(f'features ({len(feature_names)}): {feature_names}')

    mask = (~feat_df.isna().any(axis=1).values) & (~np.isnan(target))
    dates = feat_df.index[mask]
    feat = feat_df.values[mask]
    targ = target[mask]
    ew_simple = np.array([agg.ew_simple_ret[np.searchsorted(agg.dates, d)] for d in dates])

    n = len(dates)
    tw, vw, st = args.train_window_days, args.val_window_days, args.step_window_days
    windows = []
    start = 0
    while start + tw + vw <= n:
        windows.append((start, start + tw, start + tw + vw))
        start += st
    print(f'usable rows {n}; {len(windows)} windows')

    results = {}
    for kind in ('ols', 'mlp', 'hgb'):
        alphas, pos = [], 0
        for (lo, mid, hi) in windows:
            tr_pred, va_pred = _train_predict(kind, feat[lo:mid], targ[lo:mid], feat[mid:hi])
            thr = float(np.quantile(tr_pred, args.threshold_quantile))
            gate = apply_gate(va_pred, thr, mode='binary')
            gate_lag = np.concatenate([[1.0], gate[:-1]])
            val_ew = ew_simple[mid:hi]
            vdates = dates[mid:hi]
            unc = evaluate_gated_arm(val_ew, np.ones_like(val_ew), vdates, arm_label='unc')
            gated = evaluate_gated_arm(val_ew, gate_lag, vdates, arm_label='gated')
            a = gated.sharpe - unc.sharpe
            alphas.append(a)
            pos += int(a > 0)
        mean_a = float(np.mean(alphas))
        results[kind] = {'mean_alpha': mean_a, 'pos_windows': pos,
                         'n_windows': len(windows), 'per_window_alpha': alphas}
        verdict = ('PASS' if (mean_a >= 0.20 and pos >= 5) else
                   'FAIL-null' if (mean_a < 0.10 or pos <= 2) else 'MARGINAL')
        print(f'  {kind:4s}: mean alpha {mean_a:+.3f}  pos {pos}/{len(windows)}  '
              f'[{", ".join(f"{x:+.2f}" for x in alphas)}]  → {verdict}')

    print(f'\nbaseline v0 (OLS, 10 feat): mean alpha +0.067, 4/6')
    OUT.mkdir(exist_ok=True)
    (OUT / 'gate-v1-summary.json').write_text(json.dumps({
        'features': feature_names, 'threshold_quantile': args.threshold_quantile,
        'results': results,
    }, indent=2))
    print(f'-> {OUT / "gate-v1-summary.json"}')


if __name__ == '__main__':
    main()
