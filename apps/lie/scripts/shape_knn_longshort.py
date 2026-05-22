"""Shape-kNN 1-month reversal as a market-neutral long/short book.

The lie v3/v4 tests established that per-ticker **shape features** kNN
predicts forward 1-month cross-sectional excess return at t=+3.58..+3.75
(`memory: lie_test3/4`). Those tests measured *IC* — signal quality, not
a deployable return stream. This converts the same signal into a
**dollar-neutral long/short portfolio** and runs it through the cross-arc
Deflated-Sharpe harness, so it can be ranked against the rest of the
leaderboard on the one apples-to-apples key.

It satisfies all three requirements a legitimate spread trade needs:
  1. shared, tradable cross-section — Phase-2 mega-caps (liquid +
     cheap to borrow);
  2. predictive IC on that cross-section — the shape-kNN signal;
  3. a common risk factor to net out — long/short construction is
     dollar-neutral (sum w = 0), so market beta cancels.

Pipeline mirrors `test_cross_sectional_ic.py` (ticker-only predictor —
the canonical signal; the manifold hurt in v3), then:
  4. subsample non-overlapping rebal dates every H bars;
  5. assemble per-rebal score / forward-return / mask matrices;
  6. build the market-neutral book with the validated
     `factor.objectives.block_port_returns_long_short_np` constructor;
  7. charge commission on turnover + a borrow haircut on the short leg;
  8. dump the OOS per-rebal net-return stream for `compute_dsr.py`.

Repro:
    uv run python apps/lie/scripts/shape_knn_longshort.py \
        --data-dir ./StooqData --start 2010-01-01 --end 2025-12-11 \
        --horizon 21 --lookback 60 --k 50 --temporal-gap 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lie.cross_sectional import cross_sectional_ic_summary
from lie.longshort import long_short_net_returns
from lie.predictor import TimelessPredictor
from lie.ticker_features import TickerFeatureConfig, build_ticker_features
from ss_portfolio import standardize_oos

PHASE2_TICKERS = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'WMT', 'V', 'MA', 'UNH', 'JNJ', 'PG',
    'HD', 'XOM', 'CVX', 'KO', 'PEP', 'MRK',
)
REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--start', default='2010-01-01')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--horizon', type=int, default=21)
    p.add_argument('--lookback', type=int, default=60)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--temporal-gap', type=int, default=60)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--tickers', default=','.join(PHASE2_TICKERS))
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--borrow-bps-yr', type=float, default=50.0,
                   help='Annual borrow cost charged on short notional '
                        '(mega-cap borrow is cheap; 50bps is conservative).')
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--output-dir', default=str(REPO / 'Output'))
    p.add_argument('--dump-returns', action='store_true')
    return p.parse_args()


def _batched_predict(pred, X, t_idx, bs):
    out = np.full(X.shape[0], np.nan)
    for s in range(0, X.shape[0], bs):
        e = min(s + bs, X.shape[0])
        out[s:e], _ = pred.predict(X[s:e], t_idx[s:e])
    return out


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    H = args.horizon

    from ss_loaders import load_stooq_matrix
    closes, _, _, _ = load_stooq_matrix(
        args.data_dir, tickers=tickers, start_date=args.start, end_date=args.end,
        min_history=args.lookback + H + 10)
    present = [t for t in tickers if t in closes.columns]
    closes = closes.reindex(columns=present)
    panel = closes.to_numpy()
    dates = closes.index
    T, N = panel.shape
    print(f'panel: {T} dates x {N} tickers ({present[:3]}...)')

    # Per-ticker shape features (the canonical signal).
    ticker_feats, valid_tick = build_ticker_features(panel, TickerFeatureConfig())

    # Forward H-day per-ticker excess (cross-sectionally demeaned) log-return.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(np.isfinite(panel) & (panel > 0), np.log(panel), np.nan)
    fwd = np.full((T, N), np.nan)
    if T > H:
        fwd[:T - H] = log_p[H:] - log_p[:T - H]
    with np.errstate(invalid='ignore'):
        excess = fwd - np.nanmean(fwd, axis=1, keepdims=True)

    valid_full = valid_tick & np.isfinite(excess)
    t_arr, i_arr = np.where(valid_full)
    print(f'long-format samples: M = {len(t_arr)}')

    # Temporal split by date (no leakage), standardize features on train.
    split_date = int(np.percentile(np.unique(t_arr), args.train_frac * 100))
    is_train = t_arr <= split_date
    is_test = ~is_train
    print(f'train end {dates[split_date].date()}  '
          f'test start {dates[t_arr[is_test].min()].date()}')

    tf = ticker_feats[t_arr, i_arr]
    mu = tf[is_train].mean(axis=0)
    sd = tf[is_train].std(axis=0, ddof=1)
    sd = np.where(sd <= 0, 1.0, sd)
    X = (tf - mu) / sd
    y = excess[t_arr, i_arr]
    t_int = t_arr.astype(np.int64)

    pred = TimelessPredictor(k=args.k, temporal_gap=args.temporal_gap,
                             weighting='inverse_distance')
    pred.fit(X[is_train], t_int[is_train], y[is_train])
    yhat = _batched_predict(pred, X[is_test], t_int[is_test], args.batch_size)

    # Sanity: did the IC signal reproduce on this pipeline? (separates an
    # IC->portfolio translation failure from a broken signal.)
    ic = cross_sectional_ic_summary(yhat, y[is_test], t_int[is_test],
                                    method='spearman')
    print(f'signal check: mean daily IC {ic["mean_ic"]:+.4f}  '
          f't={ic["t_stat"]:+.2f}  over {ic["n_dates"]} dates '
          f'({ic["frac_positive"]*100:.0f}% positive)')

    # Scatter test predictions back into a dense (T, N) score grid.
    score_grid = np.full((T, N), np.nan)
    score_grid[t_arr[is_test], i_arr[is_test]] = yhat

    # Non-overlapping rebal dates every H bars over the test span.
    test_t = np.unique(t_arr[is_test])
    rebal_t = test_t[::H]
    # Need t+H within range for the realized forward return.
    rebal_t = rebal_t[rebal_t + H < T]
    R = len(rebal_t)
    print(f'rebal blocks (every {H}d, non-overlapping): {R}')

    scores = np.nan_to_num(score_grid[rebal_t], nan=0.0)        # (R, N)
    blr = np.where(np.isfinite(fwd[rebal_t]), fwd[rebal_t], 0.0)  # (R, N) realized H-day log-ret
    mask = (np.isfinite(score_grid[rebal_t])
            & np.isfinite(fwd[rebal_t])).astype(np.float64)      # (R, N)

    # Market-neutral long/short book via the validated factor constructor.
    cf = args.commission_bps / 1e4
    net = long_short_net_returns(scores, blr, mask, cf)

    # Borrow haircut on the short leg: L1(w)=1 => gross short ~0.5 per block,
    # held H/252 of a year. Conservative flat charge per block.
    borrow_per_block = (args.borrow_bps_yr / 1e4) * (H / 252.0) * 0.5
    net = net - borrow_per_block

    ppy = 252.0 / H
    mb = standardize_oos(net, periods_per_year=ppy, n_trials=9)
    print(f'\n--- shape-kNN long/short (Phase-2, H={H}, {R} blocks) ---')
    print(f'  per-block mean   {net.mean():+.5f}  std {net.std():.5f}')
    print(f'  annualized Sharpe {mb.ann_sharpe:+.3f}  (ppy={ppy:.1f})')
    print(f'  skew {mb.skew:+.2f}  kurt {mb.kurtosis:.2f}  N={mb.n_obs}')
    print(f'  DSR {mb.dsr:.3f}  deflated t {mb.deflated_tstat:+.3f} '
          f'(n_trials=9, E[maxSR]={mb.expected_max_sharpe:.3f})')

    if args.dump_returns:
        out = Path(args.output_dir) / 'lie-shape-knn-returns.npz'
        np.savez(out, ls_block_returns=net.astype(np.float64),
                 periods_per_year=np.float64(ppy),
                 commission_bps=np.float64(args.commission_bps),
                 borrow_bps_yr=np.float64(args.borrow_bps_yr))
        print(f'-> {out}')
        summ = Path(args.output_dir) / 'lie-shape-knn-longshort-summary.json'
        summ.write_text(json.dumps({
            'universe': present, 'horizon': H, 'lookback': args.lookback,
            'k': args.k, 'n_blocks': int(R),
            'ann_sharpe': mb.ann_sharpe, 'dsr': mb.dsr,
            'deflated_tstat': mb.deflated_tstat, 'n_trials': 9,
            'commission_bps': args.commission_bps,
            'borrow_bps_yr': args.borrow_bps_yr,
        }, indent=2))
        print(f'-> {summ}')


if __name__ == '__main__':
    main()
