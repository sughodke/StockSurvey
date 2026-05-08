"""v4: head-to-head between lie's per-ticker shape features and
relational-style CWT fingerprints, on the same kNN / IC pipeline.

V3 finding: cross-sectional kNN on per-ticker shape features extracts a
real signal (mean Spearman IC = +0.026, t-stat = +3.75 on Phase-2 21
mega-caps, H=21, 1178 test dates). The universe-state manifold did NOT
add lift over shape features.

V4 question: is the per-ticker SHAPE feature space (vol-normalized
momentum + drawdown + trailing skew/kurt -- 7 features) the right
representation, or do CWT fingerprints (relational's choice -- 168-D L2-
normalized scalogram slices) capture more of the cross-sectional
reversal signal?

Three predictors compared:

* `shape`   -- lie's per-ticker shape features (7-D).
* `cwt`     -- L2-normalized CWT-fingerprint vectors at scales
               [5, 7, 10, 12, 21, 26, 50, 90] over a fingerprint window
               of w=21 bars  ==  168-D per (t, i).
* `joined`  -- shape standardized + L2-normalized, concatenated with
               cwt. Each half normalized to unit length so neither
               dominates the L2 distance metric.

Same kNN (k=50, temporal_gap=60d, distance-weighted), same target (H=21
forward per-ticker excess log-return demeaned by daily universe mean),
same temporal split (~70/30), same valid-mask aligned across both
feature spaces (so all three predictors see the same (t, i) sample set).

This isolates the feature-space question. The downstream pipeline is
identical -- only the embedding changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lie.cross_sectional import cross_sectional_ic_summary
from lie.predictor import TimelessPredictor
from lie.ticker_features import TickerFeatureConfig, build_ticker_features
from ss_wavelets import causal_cwt


PHASE2_TICKERS: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WMT', 'V', 'MA', 'UNH', 'JNJ', 'PG',
    'HD', 'XOM', 'CVX', 'KO', 'PEP', 'MRK',
)

# Relational canonical-checkpoint scales for analog kNN (CLAUDE.md ref).
DEFAULT_CWT_SCALES: tuple[int, ...] = (5, 7, 10, 12, 21, 26, 50, 90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='v4 lie/shape vs cwt head-to-head.')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--horizon', type=int, default=21)
    p.add_argument('--lookback', type=int, default=60,
                   help='Rolling z-norm window for both shape (corr) '
                        'and CWT pipelines.')
    p.add_argument('--cwt-scales', default=','.join(str(s) for s in DEFAULT_CWT_SCALES))
    p.add_argument('--fp-window', type=int, default=21,
                   help='CWT-fingerprint window in bars. Default 21.')
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--temporal-gap', type=int, default=60)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--tickers', default=','.join(PHASE2_TICKERS))
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--output', default='Output/lie-test4-headtohead.json')
    return p.parse_args()


def extract_cwt_fingerprints(
    prices: np.ndarray,
    scales: list[int],
    lookback: int,
    fp_window: int,
) -> np.ndarray:
    """Minimal inline mirror of `relational.fingerprints.extract_fingerprints`
    (no Compression option). Returns `(T, N, S*w)` float32, L2-normalized."""
    coeffs = causal_cwt(prices, scales, lookback)              # (S, T, N)
    n_scales, n_dates, n_tickers = coeffs.shape
    pad = np.zeros((n_scales, fp_window - 1, n_tickers), dtype=coeffs.dtype)
    padded = np.concatenate([pad, coeffs], axis=1)
    sw = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=fp_window, axis=1)                # (S, T, N, w)
    tiles = np.transpose(sw, (1, 2, 0, 3))                      # (T, N, S, w)
    fps = tiles.reshape(n_dates, n_tickers, n_scales * fp_window).astype(
        np.float32, copy=False)
    norms = np.linalg.norm(fps, axis=-1, keepdims=True)
    return fps / np.maximum(norms, 1e-8)


def _l2_row_norm(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length. NaN-safe at the row level
    (rows with any NaN remain NaN)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-8)


def _batched_predict(
    pred: TimelessPredictor,
    X: np.ndarray,
    t_idx: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    preds = np.full(N, np.nan)
    n_used = np.zeros(N, dtype=np.int64)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p, n = pred.predict(X[s:e], t_idx[s:e])
        preds[s:e] = p
        n_used[s:e] = n
    return preds, n_used


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    cwt_scales = [int(s.strip()) for s in args.cwt_scales.split(',') if s.strip()]
    fp_w = args.fp_window
    H = args.horizon

    from ss_wavelets import KERNEL_HALF_EXTENT
    cwt_warmup = KERNEL_HALF_EXTENT * max(cwt_scales) + fp_w
    print(f'CWT scales: {cwt_scales}  fp_window={fp_w}  warmup={cwt_warmup} bars')

    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        args.data_dir, tickers=tickers,
        start_date=args.start, end_date=args.end,
        min_history=cwt_warmup + args.horizon + 10)

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

    # ---- Shape features (lie v3) ----
    tf_cfg = TickerFeatureConfig()
    ticker_feats, valid_shape = build_ticker_features(panel, tf_cfg)
    F_shape = tf_cfg.feature_width()
    print(f'shape features: {ticker_feats.shape}  '
          f'valid (t,i) = {int(valid_shape.sum())}')

    # ---- CWT fingerprints ----
    print(f'computing CWT fingerprints (this is the slow step) ...')
    fps = extract_cwt_fingerprints(
        panel, cwt_scales, lookback=args.lookback, fp_window=fp_w)
    F_cwt = fps.shape[2]
    # CWT before the warmup is computed against zero-padded history --
    # mark those rows invalid for ALL tickers.
    valid_cwt_t = np.arange(T) >= cwt_warmup
    valid_cwt = np.broadcast_to(valid_cwt_t[:, None], (T, N)).copy()
    # And `causal_cwt`'s cumsum over prices propagates any leading NaN
    # forward indefinitely -- a ticker with a single missing day before
    # its IPO date will have NaN fingerprints from that day onward.
    # Mask the fingerprint directly to catch that.
    valid_cwt &= np.all(np.isfinite(fps), axis=-1)
    print(f'cwt features:  {fps.shape}  valid (t,i) = {int(valid_cwt.sum())}')

    # ---- Forward H-day per-ticker excess return target ----
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(panel)
    log_p = np.where(np.isfinite(log_p), log_p, np.nan)
    fwd = np.full((T, N), np.nan)
    if T > H:
        fwd[:T - H] = log_p[H:] - log_p[:T - H]
    with np.errstate(invalid='ignore'):
        eq = np.nanmean(fwd, axis=1, keepdims=True)
    excess = fwd - eq

    # ---- Aligned valid mask: same (t, i) sample set for all 3 predictors ----
    valid_full = valid_shape & valid_cwt & np.isfinite(excess)
    t_arr, i_arr = np.where(valid_full)
    M = len(t_arr)
    print(f'aligned long-format samples: M = {M}')
    if M < 1000:
        print('ERROR: too few aligned samples.', file=sys.stderr)
        sys.exit(2)

    # ---- Temporal split by date ----
    unique_dates = np.unique(t_arr)
    split_date = int(np.percentile(unique_dates, args.train_frac * 100))
    is_train = t_arr <= split_date
    is_test = ~is_train
    print(f'train samples: {int(is_train.sum())}  '
          f'test samples: {int(is_test.sum())}')
    print(f'train end date: {dates[split_date].date()}')
    print(f'test  start dt: {dates[t_arr[is_test].min()].date()}')

    y = excess[t_arr, i_arr]
    t_int = t_arr.astype(np.int64)

    # ---- Build the three feature matrices ----
    # shape: per-feature mean/std standardization on TRAIN only.
    tf_per_sample = ticker_feats[t_arr, i_arr]                  # (M, F_shape)
    tf_mean = tf_per_sample[is_train].mean(axis=0)
    tf_std = tf_per_sample[is_train].std(axis=0, ddof=1)
    tf_std = np.where(tf_std <= 0, 1.0, tf_std)
    X_shape = (tf_per_sample - tf_mean) / tf_std

    # cwt: already L2-row-normed by extract_cwt_fingerprints.
    X_cwt = fps[t_arr, i_arr].astype(np.float64, copy=False)

    # joined: L2-normalize the standardized shape, concat with cwt.
    # Both halves now contribute unit length to the joint distance, so
    # the kNN sees them at equal weight.
    X_shape_l2 = _l2_row_norm(X_shape)
    X_joined = np.hstack([X_shape_l2, X_cwt])

    summaries: dict[str, dict] = {}

    for label, X in [('shape', X_shape),
                     ('cwt', X_cwt),
                     ('joined', X_joined)]:
        print(f'\n[{label}] ({X.shape[1]}-D embedding) running kNN ...')
        pred = TimelessPredictor(
            k=args.k, temporal_gap=args.temporal_gap,
            weighting='inverse_distance')
        pred.fit(X[is_train], t_int[is_train], y[is_train])
        yhat, n_used = _batched_predict(
            pred, X[is_test], t_int[is_test], args.batch_size)
        s = cross_sectional_ic_summary(
            yhat, y[is_test], t_int[is_test], method='spearman')
        print(f'  mean IC: {s["mean_ic"]:+.4f}  median {s["ic_p50"]:+.4f}')
        print(f'  std IC : {s["std_ic"]:+.4f}  n_dates = {s["n_dates"]}')
        print(f'  t-stat : {s["t_stat"]:+.3f}')
        print(f'  frac-positive: {s["frac_positive"] * 100:.1f}%')
        summaries[label] = {k: v for k, v in s.items() if k != 'ic_series'}

    print('\n--- comparison ---')
    print(f'shape  -> mean IC {summaries["shape"]["mean_ic"]:+.4f}  '
          f't={summaries["shape"]["t_stat"]:+.2f}')
    print(f'cwt    -> mean IC {summaries["cwt"]["mean_ic"]:+.4f}  '
          f't={summaries["cwt"]["t_stat"]:+.2f}')
    print(f'joined -> mean IC {summaries["joined"]["mean_ic"]:+.4f}  '
          f't={summaries["joined"]["t_stat"]:+.2f}')
    cwt_vs_shape = summaries['cwt']['mean_ic'] - summaries['shape']['mean_ic']
    join_vs_best = summaries['joined']['mean_ic'] - max(
        summaries['shape']['mean_ic'], summaries['cwt']['mean_ic'])
    print(f'lift CWT  vs SHAPE       : {cwt_vs_shape:+.4f}')
    print(f'lift JOIN vs BEST(s/c)   : {join_vs_best:+.4f}')

    out = {
        'tickers_present': present,
        'tickers_missing': missing,
        'date_range': [str(args.start), str(args.end)],
        'horizon_bars': H,
        'lookback_bars': args.lookback,
        'k_neighbors': args.k,
        'temporal_gap_bars': args.temporal_gap,
        'cwt_scales': cwt_scales,
        'fp_window': fp_w,
        'F_shape': F_shape,
        'F_cwt': F_cwt,
        'F_joined': F_shape + F_cwt,
        'M_aligned_samples': M,
        'train_samples': int(is_train.sum()),
        'test_samples': int(is_test.sum()),
        'train_end_date': str(dates[split_date].date()),
        'test_start_date': str(dates[t_arr[is_test].min()].date()),
        'shape': summaries['shape'],
        'cwt': summaries['cwt'],
        'joined': summaries['joined'],
        'lift_cwt_vs_shape': float(cwt_vs_shape),
        'lift_join_vs_best': float(join_vs_best),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
