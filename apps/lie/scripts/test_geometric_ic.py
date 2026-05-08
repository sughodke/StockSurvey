"""Test 1: do geometric neighbors predict forward returns better than
temporal-baseline neighbors?

Pipeline
--------

1. Load a fixed equity universe from the Stooq archive (default: the
   Phase-2 21 mega-cap subset already used by `apps/relational`).
2. Build a per-date market-state feature matrix via `build_market_state`.
3. Compute the equal-weighted forward `H`-day universe return as the
   prediction target (default H = 21).
4. Split temporally -- first `train_frac` of valid rows are train; the
   remainder is test. No shuffling.
5. Fit a `ManifoldMapper` on the train rows; project both train + test.
6. Fit a `TimelessPredictor` on the train embeddings + targets, with a
   60-day temporal-gap exclusion.
7. Predict on the test embeddings.
8. Report:
     * variance explained by the manifold at k = 5, 8, 10, 15
     * Pearson + Spearman IC of geometric kNN on test
     * Pearson IC of two baselines:
        - climatology mean (predict the train-set mean for every test row)
        - AR(1) (predict the most recent realized H-day return)

A positive geometric IC that meaningfully exceeds the AR(1) baseline is
the lift the timeless-structure hypothesis predicts. If geometric IC <=
AR(1) IC, the manifold isn't adding signal beyond simple autocorrelation.

Usage
-----

    uv run python apps/lie/scripts/test_geometric_ic.py \\
        --data-dir ./StooqData \\
        --start 2010-01-01 --end 2025-12-11 \\
        --horizon 21 --lookback 60 --k 50 --temporal-gap 60 \\
        --n-components 8

Outputs JSON summary to `Output/lie-test-geometric-ic.json` and prints a
text summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lie.manifold import ManifoldMapper, variance_explained_at_k
from lie.predictor import TimelessPredictor, information_coefficient
from lie.state_builder import MarketStateConfig, build_market_state


# Phase-2 mega-cap universe used by apps/relational canonical checkpoints.
# Stable composition; sufficient history (most names trade back to the 90s).
PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WMT', 'V', 'MA', 'UNH', 'JNJ', 'PG',
    'HD', 'XOM', 'CVX', 'KO', 'PEP', 'MRK',
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data-dir', required=True,
                   help='Stooq archive root (the dir containing `daily/`).')
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--horizon', type=int, default=21,
                   help='Forward-return horizon in trading days. Default 21.')
    p.add_argument('--lookback', type=int, default=60,
                   help='Trailing-window for the rolling correlation matrix.')
    p.add_argument('--k', type=int, default=50,
                   help='Number of neighbors. Default 50.')
    p.add_argument('--temporal-gap', type=int, default=60,
                   help='Min trading-day separation between query and any '
                        'admissible neighbor. Default 60.')
    p.add_argument('--n-components', type=int, default=8,
                   help='PCA components for the manifold. Default 8.')
    p.add_argument('--train-frac', type=float, default=0.7,
                   help='Fraction of valid rows used for train (rest is test).')
    p.add_argument('--tickers', default=','.join(PHASE2_TICKERS),
                   help='Comma-separated ticker list. Default Phase-2 21 names.')
    p.add_argument('--output', default='Output/lie-test-geometric-ic.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if len(tickers) < 5:
        print('ERROR: need at least 5 tickers.', file=sys.stderr)
        sys.exit(2)

    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        args.data_dir,
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        min_history=args.lookback + args.horizon + 10)

    # Fix column order to the requested universe; drop names absent in archive.
    present = [t for t in tickers if t in closes.columns]
    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        print(f'WARNING: skipping {len(missing)} missing ticker(s): {missing}',
              file=sys.stderr)
    closes = closes.reindex(columns=present)
    panel = closes.to_numpy()
    dates = closes.index

    # Build state vectors.
    cfg = MarketStateConfig(lookback=args.lookback)
    states, valid_t = build_market_state(panel, cfg)
    print(f'state matrix: {states.shape}  valid rows: {int(valid_t.sum())}')

    # Forward H-day equal-weighted universe return.
    # eq_ret[t] = mean over names of log(prices[t+H]/prices[t]).
    H = args.horizon
    T = panel.shape[0]
    fwd = np.full(T, np.nan)
    if T > H:
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.log(panel)
        log_p = np.where(np.isfinite(log_p), log_p, np.nan)
        forward_block = log_p[H:] - log_p[:-H]                 # (T-H, N)
        fwd[:T - H] = np.nanmean(forward_block, axis=1)        # mean over tickers

    # Keep rows where state and target are both finite.
    use = valid_t & np.isfinite(fwd)
    use_idx = np.where(use)[0]
    if len(use_idx) < 200:
        print(f'ERROR: only {len(use_idx)} usable rows -- need >=200.',
              file=sys.stderr)
        sys.exit(2)

    X = states[use_idx]
    y = fwd[use_idx]
    t_idx = use_idx.astype(np.int64)
    use_dates = dates[use_idx]

    # Temporal split.
    n = len(use_idx)
    split = int(n * args.train_frac)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    t_tr, t_te = t_idx[:split], t_idx[split:]
    d_tr_last = use_dates[split - 1] if split > 0 else None
    d_te_first = use_dates[split] if split < n else None
    print(f'train rows: {split}  test rows: {n - split}')
    if d_tr_last is not None and d_te_first is not None:
        print(f'train end:   {d_tr_last.date()}')
        print(f'test  start: {d_te_first.date()}')

    # Variance-explained sweep (informational; uses train only).
    var_at_k = variance_explained_at_k(X_tr, ks=[5, 8, 10, 15])
    print('variance explained on train:')
    for k in sorted(var_at_k):
        print(f'  k={k:2d}: {var_at_k[k] * 100:.1f}%')

    # Fit manifold on train; project both halves.
    mapper = ManifoldMapper(n_components=args.n_components)
    Z_tr = mapper.fit_transform(X_tr)
    Z_te = mapper.transform(X_te)
    cum = mapper.cumulative_variance_explained_
    print(f'kept {args.n_components} components -> '
          f'{cum[-1] * 100:.1f}% cumulative variance')

    # Geometric kNN.
    pred = TimelessPredictor(
        k=args.k,
        temporal_gap=args.temporal_gap,
        weighting='inverse_distance')
    pred.fit(Z_tr, t_tr, y_tr)
    yhat_geo, n_used = pred.predict(Z_te, t_te)
    n_with_pred = int(np.isfinite(yhat_geo).sum())
    print(f'geometric kNN: predictions for {n_with_pred} / {len(y_te)} test rows')
    print(f'  mean neighbors used: {float(np.mean(n_used)):.1f}')

    ic_pearson = information_coefficient(yhat_geo, y_te, method='pearson')
    ic_spearman = information_coefficient(yhat_geo, y_te, method='spearman')
    print(f'  pearson  IC: {ic_pearson:+.4f}')
    print(f'  spearman IC: {ic_spearman:+.4f}')

    # Baselines.
    yhat_clim = np.full_like(y_te, fill_value=float(np.mean(y_tr)))
    ic_clim = information_coefficient(yhat_clim, y_te, method='pearson')
    # Climatology IC is by definition zero (constant predictor) -- include
    # for completeness so the comparison is honest.

    # AR(1): the most recent realized H-day return at query time t is
    # `fwd[t - H]`, which is known at t (it's the return REALIZED over
    # [t-H, t]). Predict that as the next H-day return.
    yhat_ar1 = np.full(len(y_te), np.nan)
    for j, t in enumerate(t_te):
        t_prev = t - H
        if t_prev >= 0 and np.isfinite(fwd[t_prev]):
            yhat_ar1[j] = fwd[t_prev]
    ic_ar1 = information_coefficient(yhat_ar1, y_te, method='pearson')
    print(f'baseline IC (climatology, constant): {ic_clim:+.4f}')
    print(f'baseline IC (AR(1), prev H-day ret): {ic_ar1:+.4f}')

    lift_vs_ar1 = ic_pearson - ic_ar1
    verdict = (
        'GEOMETRIC > AR(1)' if lift_vs_ar1 > 0.005
        else 'INCONCLUSIVE' if abs(lift_vs_ar1) <= 0.005
        else 'GEOMETRIC < AR(1)')
    print(f'lift vs AR(1): {lift_vs_ar1:+.4f}  -> {verdict}')

    # Persist.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'tickers_present': present,
        'tickers_missing': missing,
        'date_range': [str(args.start), str(args.end)],
        'horizon_bars': H,
        'lookback_bars': args.lookback,
        'k_neighbors': args.k,
        'temporal_gap_bars': args.temporal_gap,
        'n_components': args.n_components,
        'train_rows': int(split),
        'test_rows': int(n - split),
        'train_end': str(d_tr_last.date()) if d_tr_last is not None else None,
        'test_start': str(d_te_first.date()) if d_te_first is not None else None,
        'variance_explained_at_k': var_at_k,
        'cumulative_variance_kept': float(cum[-1]),
        'pearson_ic_geometric': float(ic_pearson),
        'spearman_ic_geometric': float(ic_spearman),
        'pearson_ic_climatology': float(ic_clim),
        'pearson_ic_ar1': float(ic_ar1),
        'lift_vs_ar1': float(lift_vs_ar1),
        'verdict': verdict,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
