"""v3: cross-sectional manifold kNN.

Where Test 1 asked "does the manifold predict universe-mean forward
return?" (answer: no, across all tested horizons), this asks the harder
question: "does the manifold position act as global context that, joined
with per-ticker shape features, ranks tickers by forward excess return?"

That's the cleanest cross-sectional version of the timeless hypothesis:
two `(date, ticker)` pairs from completely different epochs should have
similar forward excess return if both their universe-coordinate AND
their per-ticker shape coordinate are similar.

Pipeline
--------

1. Load fixed equity universe (default Phase-2 21 mega-cap names).
2. Build universe state vectors `(T, F_u)` via `build_market_state`.
3. Build ticker features `(T, N, F_t)` via `build_ticker_features`.
4. Compute forward H-day per-ticker log-return `(T, N)`.
5. Demean per-date: `excess[t, i] = ret[t, i] - mean_j ret[t, j]`.
6. Flatten valid `(t, i)` pairs into a long-format dataset.
7. Temporal split: first `train_frac` of dates -> train, rest -> test.
8. Fit `ManifoldMapper(8)` on train universe states (no leakage).
9. Standardize ticker features on train statistics (no leakage).
10. Build joint embedding `[manifold_z_t, standardized_ticker_t_i]`.
11. Fit `TimelessPredictor` (k=50, 60d gap, distance-weighted).
12. Predict on test in chunks (memory-safe).
13. Compare three predictors:
       * **joined**     -- universe state + ticker features
       * **ticker_only** -- ticker features only (baseline: does the
                           manifold add lift over per-ticker shape?)
       * **ar1**        -- per-ticker last realized H-day return
14. Report per-date Spearman IC mean / std / t-stat / frac-positive.

The decisive comparison is `joined` vs `ticker_only`: if the manifold
adds nothing, they'll be ~equal and the global-context hypothesis fails.
If `joined` > `ticker_only` by a meaningful t-stat margin, the universe
manifold IS providing useful regime-conditioning beyond per-ticker
shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lie.cross_sectional import cross_sectional_ic_summary
from lie.manifold import ManifoldMapper, variance_explained_at_k
from lie.predictor import TimelessPredictor
from lie.state_builder import MarketStateConfig, build_market_state
from lie.ticker_features import TickerFeatureConfig, build_ticker_features


PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WMT', 'V', 'MA', 'UNH', 'JNJ', 'PG',
    'HD', 'XOM', 'CVX', 'KO', 'PEP', 'MRK',
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='v3 cross-sectional manifold kNN.')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--horizon', type=int, default=21)
    p.add_argument('--lookback', type=int, default=60)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--temporal-gap', type=int, default=60)
    p.add_argument('--n-components', type=int, default=8)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--tickers', default=','.join(PHASE2_TICKERS))
    p.add_argument('--batch-size', type=int, default=500,
                   help='kNN query batch size for memory.')
    p.add_argument('--output', default='Output/lie-test3-cross-sectional.json')
    return p.parse_args()


def _batched_predict(
    pred: TimelessPredictor,
    X: np.ndarray,
    t_idx: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked wrapper around TimelessPredictor.predict to keep the
    pairwise-distance matrix memory-bounded."""
    N = X.shape[0]
    preds = np.full(N, np.nan)
    n_used = np.zeros(N, dtype=np.int64)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p, n = pred.predict(X[s:e], t_idx[s:e])
        preds[s:e] = p
        n_used[s:e] = n
    return preds, n_used


def _ar1_baseline(
    panel: np.ndarray,
    t_idx: np.ndarray,
    ticker_idx: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Per-ticker AR(1): predicted forward H-day excess return ==
    last realized H-day return of THAT ticker, demeaned across the
    universe at the same date."""
    M = len(t_idx)
    out = np.full(M, np.nan)
    T, N = panel.shape
    if T <= horizon:
        return out
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(panel)
    log_p = np.where(np.isfinite(log_p), log_p, np.nan)
    # realized[t, i] = log(p[t, i] / p[t-H, i]) -- known at t.
    realized = np.full((T, N), np.nan)
    if T > horizon:
        realized[horizon:] = log_p[horizon:] - log_p[:-horizon]
    # demean per-date by universe-mean realized
    with np.errstate(invalid='ignore'):
        eq = np.nanmean(realized, axis=1, keepdims=True)
    excess = realized - eq
    out = excess[t_idx, ticker_idx]
    return out


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if len(tickers) < 5:
        print('ERROR: need at least 5 tickers.', file=sys.stderr)
        sys.exit(2)

    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        args.data_dir, tickers=tickers,
        start_date=args.start, end_date=args.end,
        min_history=args.lookback + args.horizon + 10)

    present = [t for t in tickers if t in closes.columns]
    missing = [t for t in tickers if t not in closes.columns]
    if missing:
        print(f'WARNING: skipping {len(missing)} missing: {missing}',
              file=sys.stderr)
    closes = closes.reindex(columns=present)
    panel = closes.to_numpy()
    dates = closes.index
    T, N = panel.shape
    print(f'panel: {T} dates x {N} tickers')

    # Build features.
    state_cfg = MarketStateConfig(lookback=args.lookback)
    tf_cfg = TickerFeatureConfig()
    states, valid_state = build_market_state(panel, state_cfg)
    ticker_feats, valid_tick = build_ticker_features(panel, tf_cfg)
    F_u = state_cfg.feature_width()
    F_t = tf_cfg.feature_width()
    print(f'state {states.shape} (valid {int(valid_state.sum())} rows); '
          f'ticker_feats {ticker_feats.shape}')

    # Forward H-day per-ticker excess return target.
    H = args.horizon
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(panel)
    log_p = np.where(np.isfinite(log_p), log_p, np.nan)
    fwd = np.full((T, N), np.nan)
    if T > H:
        fwd[:T - H] = log_p[H:] - log_p[:T - H]
    with np.errstate(invalid='ignore'):
        eq = np.nanmean(fwd, axis=1, keepdims=True)
    excess = fwd - eq

    # Build long-format (t, i) sample where state, ticker_feat, target all valid.
    valid_full = (valid_state[:, None] & valid_tick & np.isfinite(excess))
    t_arr, i_arr = np.where(valid_full)
    M = len(t_arr)
    print(f'long-format samples: M = {M}')
    if M < 1000:
        print('ERROR: too few valid (t, i) samples.', file=sys.stderr)
        sys.exit(2)

    # Temporal split by date.
    unique_dates = np.unique(t_arr)
    split_date = int(np.percentile(unique_dates, args.train_frac * 100))
    is_train = t_arr <= split_date
    is_test = ~is_train
    print(f'train samples: {int(is_train.sum())}  '
          f'test samples: {int(is_test.sum())}')
    print(f'train end date: {dates[split_date].date()}')
    print(f'test  start dt: {dates[t_arr[is_test].min()].date()}')

    # Per-train-row matrices.
    state_train_unique_t = np.unique(t_arr[is_train])
    state_train = states[state_train_unique_t]
    ve = variance_explained_at_k(state_train, ks=[5, 8, 10, 15])
    print('variance explained on TRAIN universe states:')
    for k in sorted(ve):
        print(f'  k={k:2d}: {ve[k] * 100:.1f}%')

    # Fit manifold on train universe states only.
    mapper = ManifoldMapper(n_components=args.n_components)
    mapper.fit(state_train)
    Z_train_dates = mapper.transform(state_train)         # one row per train date
    cum_ve = float(mapper.cumulative_variance_explained_[-1])
    print(f'kept {args.n_components} components -> {cum_ve * 100:.1f}% variance')

    # Build per-sample manifold rows: tile by (t, i) pairs.
    Z_full = mapper.transform(states[np.unique(t_arr)])    # (n_unique_dates, K)
    # Map t -> index in the unique-sorted date set.
    unique_t_sorted = np.unique(t_arr)
    pos_of_t = np.searchsorted(unique_t_sorted, t_arr)
    Z_per_sample = Z_full[pos_of_t]                        # (M, K)

    # Standardize ticker features on TRAIN samples only.
    tf_per_sample = ticker_feats[t_arr, i_arr]             # (M, F_t)
    tf_mean = tf_per_sample[is_train].mean(axis=0)
    tf_std = tf_per_sample[is_train].std(axis=0, ddof=1)
    tf_std = np.where(tf_std <= 0, 1.0, tf_std)
    tf_std_full = (tf_per_sample - tf_mean) / tf_std

    # Joint embedding.
    X_joint = np.hstack([Z_per_sample, tf_std_full])
    X_ticker = tf_std_full

    y = excess[t_arr, i_arr]
    t_int = t_arr.astype(np.int64)

    # ---- Predictor 1: joined (universe state + ticker features) ----
    pred_joined = TimelessPredictor(
        k=args.k, temporal_gap=args.temporal_gap, weighting='inverse_distance')
    pred_joined.fit(X_joint[is_train], t_int[is_train], y[is_train])
    yhat_joint, n_used_j = _batched_predict(
        pred_joined, X_joint[is_test], t_int[is_test], args.batch_size)
    sum_joint = cross_sectional_ic_summary(
        yhat_joint, y[is_test], t_int[is_test], method='spearman')

    # ---- Predictor 2: ticker-only (baseline isolating manifold lift) ----
    pred_ticker = TimelessPredictor(
        k=args.k, temporal_gap=args.temporal_gap, weighting='inverse_distance')
    pred_ticker.fit(X_ticker[is_train], t_int[is_train], y[is_train])
    yhat_ticker, n_used_t = _batched_predict(
        pred_ticker, X_ticker[is_test], t_int[is_test], args.batch_size)
    sum_ticker = cross_sectional_ic_summary(
        yhat_ticker, y[is_test], t_int[is_test], method='spearman')

    # ---- Predictor 3: per-ticker AR(1) ----
    yhat_ar1 = _ar1_baseline(panel, t_int[is_test], i_arr[is_test], H)
    sum_ar1 = cross_sectional_ic_summary(
        yhat_ar1, y[is_test], t_int[is_test], method='spearman')

    # Report.
    def _fmt(s: dict, label: str) -> None:
        print(f'\n[{label}]')
        print(f'  mean IC: {s["mean_ic"]:+.4f}  '
              f'(median {s["ic_p50"]:+.4f})')
        print(f'  std IC : {s["std_ic"]:+.4f}  '
              f'over n_dates = {s["n_dates"]}')
        print(f'  t-stat : {s["t_stat"]:+.3f}')
        print(f'  frac-positive dates: {s["frac_positive"] * 100:.1f}%')

    _fmt(sum_joint, 'JOINED  (universe state + ticker features)')
    _fmt(sum_ticker, 'TICKER  (ticker features only -- no manifold)')
    _fmt(sum_ar1, 'AR(1)   (last realized H-day return per ticker)')

    lift_joint_vs_ticker = sum_joint['mean_ic'] - sum_ticker['mean_ic']
    lift_joint_vs_ar1 = sum_joint['mean_ic'] - sum_ar1['mean_ic']

    print(f'\nlift JOINED vs TICKER : {lift_joint_vs_ticker:+.4f} '
          f'(>0 => manifold context helps)')
    print(f'lift JOINED vs AR(1)  : {lift_joint_vs_ar1:+.4f}')

    summary = {
        'tickers_present': present,
        'tickers_missing': missing,
        'date_range': [str(args.start), str(args.end)],
        'horizon_bars': H,
        'lookback_bars': args.lookback,
        'k_neighbors': args.k,
        'temporal_gap_bars': args.temporal_gap,
        'n_components': args.n_components,
        'train_samples': int(is_train.sum()),
        'test_samples': int(is_test.sum()),
        'train_end_date': str(dates[split_date].date()),
        'variance_explained_at_k': ve,
        'cumulative_variance_kept': cum_ve,
        'joined': {k: v for k, v in sum_joint.items() if k != 'ic_series'},
        'ticker_only': {k: v for k, v in sum_ticker.items() if k != 'ic_series'},
        'ar1': {k: v for k, v in sum_ar1.items() if k != 'ic_series'},
        'lift_joined_vs_ticker': float(lift_joint_vs_ticker),
        'lift_joined_vs_ar1': float(lift_joint_vs_ar1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
