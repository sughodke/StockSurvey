"""Pairs v1 — ML half-life-and-friends predictor for per-pair selection.

Pre-registered as the #1 highest-EV pairs follow-up after the
[`pairs-eg-gate-falsified`](../../docs/docs/findings/pairs-eg-gate-falsified.md)
oracle finding (+1.79 Sharpe headroom in per-pair selection). The
oracle showed pair-level selection is the binding lever — train-side
regime gates (EG-pass count, vol, etc.) at window granularity can't
reach the architecture's ceiling because the signal lives within
windows, not across them.

This script implements the predictor side: for each screened pair,
extract 7 train-time features and feed them into an expanding-window
L2-regularized logistic regression that predicts P(val Sharpe > 0).
Window 0 has no prior data → falls back to v0 baseline (all pairs).
Subsequent windows train on (features, label) pairs from ALL prior
windows.

Pre-reg cuts (from pairs-eg-gate-falsified, after oracle ceiling):
  PASS:        mean val Sharpe ≥ +0.50 AND ≥ 4/6 positive windows
               (captures ≥ 28% of oracle's +1.80 ceiling)
  STRONG-PASS: mean ≥ +1.0 AND ≥ 5/6 positive (≥ 55% capture)

Features (per pair, computed on train slice only):
  log_train_half_life     log of OU mean-reversion half-life
  abs_corr                |corr(a, b)| on train log-prices
  log_eg_pvalue           log of EG cointegration p-value
  abs_hedge_beta          |β| from EG regression
  train_sharpe            in-sample Sharpe of v0 strategy on train
  train_pct_in_trade      fraction of train where v0 is in position
  train_n_trades          transitions in train (per train-window scale)

Arms reported:
  all-pairs               v0 baseline
  predictor-thr-0.5       LR with P >= 0.5 selection
  predictor-top-50        top 50 pairs by predicted P (matches oracle
                          top-quartile size)
  oracle-pos              hindsight: keep pairs with val Sharpe > 0
  oracle-top-quartile     hindsight: top 25% by val Sharpe

Run from repo root:
    uv run python apps/pairs/scripts/run_pair_predictor_walkforward.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio.metrics import annualized_sharpe, cagr, max_drawdown, sortino
from pairs.backtest import backtest_pair
from pairs.pair_universe import screen_pairs
from pairs.spread import compute_spread, spread_stats, zscore
from pairs.predictor import trade_signals


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _build_window_slices(n: int, train_days: int, val_days: int, step_days: int):
    out = []
    start = 0
    while start + train_days + val_days <= n:
        out.append((start, start + train_days, start + train_days + val_days))
        start += step_days
    return out


def _in_sample_train_features(
    log_p_a: np.ndarray, log_p_b: np.ndarray, hedge_beta: float,
    intercept: float, entry: float, exit_z: float, commission_bps: float,
) -> dict:
    """Compute the v0 strategy's in-sample (train) stats: train_sharpe,
    train_pct_in_trade, train_n_trades.

    Mirrors the relevant subset of `backtest_pair` but applied with
    train log-prices as both inputs (no look-ahead — these are
    train-side features used to predict the OOS val outcome).
    """
    spread = compute_spread(log_p_a, log_p_b, hedge_beta, intercept)
    stats = spread_stats(spread)
    z = zscore(spread, stats)
    pos = trade_signals(z, entry=entry, exit_z=exit_z, stop=float('inf'))
    pos_lag = np.concatenate([[0], pos[:-1]])

    d_log_a = np.diff(log_p_a, prepend=log_p_a[0])
    d_log_b = np.diff(log_p_b, prepend=log_p_b[0])
    spread_ret = d_log_a - hedge_beta * d_log_b
    leverage_normalizer = 1.0 + abs(hedge_beta)
    pnl = pos_lag * spread_ret / leverage_normalizer
    pos_change = np.abs(np.diff(pos_lag, prepend=0))
    cost = (commission_bps / 1e4) * 2.0 * pos_change
    pnl_net = pnl - cost

    daily = pd.Series(pnl_net)
    in_trade = pos_lag != 0
    pct_in_trade = float(np.mean(in_trade))
    transitions = int(np.sum(np.abs(np.diff(pos_lag)) > 0))
    train_sharpe = float(annualized_sharpe(daily))
    return {
        'train_sharpe':       train_sharpe,
        'train_pct_in_trade': pct_in_trade,
        'train_n_trades':     float(transitions),
        'train_half_life':    float(stats.half_life),
    }


FEATURE_NAMES = [
    'log_train_half_life',
    'abs_corr',
    'log_eg_pvalue',
    'abs_hedge_beta',
    'train_sharpe',
    'train_pct_in_trade',
    'log_train_n_trades',
]


def _feature_vector(rec: dict) -> np.ndarray:
    """Map a per-pair feature record to the canonical FEATURE_NAMES vector."""
    hl = max(rec['train_half_life'], 1.0)
    nt = max(rec['train_n_trades'], 1.0)
    eg_p = max(rec['eg_pvalue'], 1e-10)
    return np.array([
        np.log(hl),
        rec['abs_corr'],
        np.log(eg_p),
        rec['abs_hedge_beta'],
        rec['train_sharpe'],
        rec['train_pct_in_trade'],
        np.log(nt),
    ], dtype=np.float64)


def _train_lr_l2(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, n_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2-regularized logistic regression via Newton-Raphson on numpy.
    Returns (weights, mu, sd) — caller applies same z-norm at predict-time.

    Bias is not L2-penalized.
    """
    if len(X) < 5 or len(np.unique(y)) < 2:
        # Degenerate; return zero weights (predicts 0.5).
        k = X.shape[1]
        return np.zeros(k + 1), np.zeros(k), np.ones(k)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd
    n, k = Xz.shape
    Xb = np.hstack([Xz, np.ones((n, 1))])
    w = np.zeros(k + 1)
    for _ in range(n_iter):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        reg = np.concatenate([w[:-1], [0.0]])
        grad = Xb.T @ (p - y) + l2 * reg
        H = Xb.T @ (W[:, None] * Xb) + l2 * np.eye(k + 1)
        H[-1, -1] -= l2
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.abs(step).max() < 1e-7:
            break
    return w, mu, sd


def _predict_proba(X: np.ndarray, w: np.ndarray, mu: np.ndarray,
                   sd: np.ndarray) -> np.ndarray:
    if np.allclose(w, 0):
        return np.full(len(X), 0.5)
    Xz = (X - mu) / sd
    Xb = np.hstack([Xz, np.ones((len(Xz), 1))])
    z = np.clip(Xb @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _aggregate_arm(pair_results: list, val_dates: pd.DatetimeIndex,
                    log_subset: pd.DataFrame) -> dict:
    if not pair_results:
        return {'n_pairs': 0, 'sharpe': 0.0}
    n = len(pair_results)
    agg = np.zeros(len(val_dates), dtype=np.float64)
    for bt in pair_results:
        pair_idx = log_subset[[bt.a, bt.b]].dropna().index
        ser = pd.Series(bt.val_daily_ret, index=pair_idx)
        ser_full = ser.reindex(val_dates).fillna(0.0)
        agg += ser_full.values / n
    s = pd.Series(agg, index=val_dates)
    return {'n_pairs': n, 'sharpe': float(annualized_sharpe(s))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(STOOQ_SUBSET / 'manifest.json'))
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2023-12-31')
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--abs-corr-min', type=float, default=0.7)
    p.add_argument('--eg-p-max',     type=float, default=0.05)
    p.add_argument('--top-k',        type=int,   default=200)
    p.add_argument('--entry-z',      type=float, default=2.0)
    p.add_argument('--exit-z',       type=float, default=0.5)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--lr-l2',        type=float, default=1.0)
    p.add_argument('--top-k-pred',   type=int,   default=50,
                   help='Top-K pairs by predicted P for the '
                        'predictor-top-K arm; matches oracle-top-quartile.')
    p.add_argument('--n-workers', type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading universe (stooq_us_long)...', flush=True)
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers']
                      if t.get('n_bars', 0) >= args.min_history_bars)
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end, tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers in '
          f'{time.perf_counter()-t0:.1f}s', flush=True)

    log_prices = np.log(prices)
    dates = prices.index
    n = len(dates)

    windows = _build_window_slices(
        n, args.train_window_days, args.val_window_days,
        args.step_window_days)
    print(f'\nwalk-forward: {len(windows)} windows', flush=True)

    # all_pair_records — list of (window_idx, feature_vector, val_sharpe).
    all_pair_records: list[tuple[int, np.ndarray, float]] = []
    arms_order = ['all-pairs', 'predictor-thr-0p5',
                  f'predictor-top-{args.top_k_pred}',
                  'oracle-pos', 'oracle-top-quartile']
    per_arm_window_sharpes: dict[str, list[float]] = {a: [] for a in arms_order}
    per_window_detail: list[dict] = []

    for w_idx, (lo, mid, hi) in enumerate(windows):
        print(f'\n=== window {w_idx} '
              f'({dates[lo].date()} → {dates[mid - 1].date()} train, '
              f'{dates[mid].date()} → {dates[hi - 1].date()} val) ===',
              flush=True)
        train_log = log_prices.iloc[lo:mid]
        val_log   = log_prices.iloc[mid:hi]

        # Train classifier on records from prior windows.
        prior = [r for r in all_pair_records if r[0] < w_idx]
        if prior:
            X_prior = np.stack([r[1] for r in prior])
            y_prior = np.array(
                [1.0 if r[2] > 0 else 0.0 for r in prior], dtype=np.float64)
            w_lr, mu, sd = _train_lr_l2(X_prior, y_prior, l2=args.lr_l2)
            print(f'  LR trained on {len(prior)} prior pairs '
                  f'(positive frac = {y_prior.mean():.3f})', flush=True)
        else:
            w_lr, mu, sd = None, None, None
            print('  no prior pairs; fall back to v0 baseline (all-pairs) '
                  'for predictor arms', flush=True)

        # Screen pairs in the current window.
        pairs_screened = screen_pairs(
            train_log,
            min_overlap=int(args.train_window_days * 0.8),
            abs_corr_min=args.abs_corr_min,
            eg_p_max=args.eg_p_max,
            top_k=args.top_k,
            n_workers=args.n_workers,
            verbose=True)
        if not pairs_screened:
            for a in arms_order:
                per_arm_window_sharpes[a].append(0.0)
            per_window_detail.append({'window_idx': w_idx, 'n_pairs': 0})
            continue

        # Backtest each pair + extract features.
        pair_results = []
        pair_features = []
        pair_proba = []
        skipped = 0
        for c in pairs_screened:
            cols_train = train_log[[c.a, c.b]].dropna()
            cols_val   = val_log[[c.a, c.b]].dropna()
            if len(cols_val) < args.val_window_days * 0.5:
                skipped += 1
                continue
            try:
                bt = backtest_pair(
                    log_p_a_train=cols_train[c.a].values,
                    log_p_b_train=cols_train[c.b].values,
                    log_p_a_val=cols_val[c.a].values,
                    log_p_b_val=cols_val[c.b].values,
                    val_dates=cols_val.index,
                    a_name=c.a, b_name=c.b,
                    hedge_beta=c.hedge_beta, intercept=c.intercept,
                    entry=args.entry_z, exit_z=args.exit_z,
                    commission_bps=args.commission_bps)
                train_stats = _in_sample_train_features(
                    cols_train[c.a].values, cols_train[c.b].values,
                    hedge_beta=c.hedge_beta, intercept=c.intercept,
                    entry=args.entry_z, exit_z=args.exit_z,
                    commission_bps=args.commission_bps)
            except Exception as e:
                skipped += 1
                continue
            rec = {
                'abs_corr':       abs(c.train_corr),
                'eg_pvalue':      c.eg_p,
                'abs_hedge_beta': abs(c.hedge_beta),
                **train_stats,
            }
            fv = _feature_vector(rec)
            if not np.all(np.isfinite(fv)):
                skipped += 1
                continue
            pair_results.append(bt)
            pair_features.append(fv)
            if w_lr is not None:
                p_i = float(_predict_proba(fv[None], w_lr, mu, sd)[0])
            else:
                p_i = 0.5
            pair_proba.append(p_i)
            all_pair_records.append((w_idx, fv, bt.sharpe))

        val_dates = val_log.index
        pair_proba = np.asarray(pair_proba)

        # Arms.
        arm_all = _aggregate_arm(pair_results, val_dates, val_log)
        # predictor-thr-0.5
        sel_thr = [pr for pr, p_i in zip(pair_results, pair_proba) if p_i >= 0.5]
        arm_pred_thr = _aggregate_arm(sel_thr, val_dates, val_log)
        # predictor-top-K
        order_desc = np.argsort(-pair_proba)
        K = min(args.top_k_pred, len(pair_results))
        sel_topk = [pair_results[i] for i in order_desc[:K]]
        arm_pred_topk = _aggregate_arm(sel_topk, val_dates, val_log)
        # oracle-pos
        sel_pos = [r for r in pair_results if r.sharpe > 0]
        arm_oracle_pos = _aggregate_arm(sel_pos, val_dates, val_log)
        # oracle-top-quartile
        sorted_by_sh = sorted(pair_results, key=lambda r: -r.sharpe)
        n_q = max(1, len(sorted_by_sh) // 4)
        arm_oracle_topq = _aggregate_arm(
            sorted_by_sh[:n_q], val_dates, val_log)

        arms = {
            'all-pairs':                  arm_all,
            'predictor-thr-0p5':          arm_pred_thr,
            f'predictor-top-{args.top_k_pred}': arm_pred_topk,
            'oracle-pos':                 arm_oracle_pos,
            'oracle-top-quartile':        arm_oracle_topq,
        }
        for a in arms_order:
            per_arm_window_sharpes[a].append(arms[a]['sharpe'])

        print(f'  pairs backtested: {len(pair_results)} (skipped {skipped})',
              flush=True)
        for a in arms_order:
            print(f'    {a:<25} Sharpe={arms[a]["sharpe"]:+7.3f}  '
                  f'n_pairs={arms[a]["n_pairs"]}', flush=True)
        if w_lr is not None:
            pos_pred = (pair_proba >= 0.5).mean()
            print(f'    LR mean P = {pair_proba.mean():.3f}  '
                  f'frac P>=0.5 = {pos_pred:.3f}', flush=True)

        per_window_detail.append({
            'window_idx': w_idx,
            'val_start':  str(dates[mid].date()),
            'val_end':    str(dates[hi - 1].date()),
            'n_pairs':    len(pair_results),
            'arms':       {a: arms[a] for a in arms_order},
            'lr_mean_p':  float(pair_proba.mean()) if w_lr is not None else 0.0,
            'lr_n_pos_pred': int((pair_proba >= 0.5).sum())
                              if w_lr is not None else 0,
        })

    # Aggregate cross-window.
    print('\n' + '=' * 96, flush=True)
    print(f'{"arm":<28} {"mean":>9s} {"pos_w":>7s}  per-window val Sharpes',
          flush=True)
    print('-' * 96, flush=True)
    arm_aggs = {}
    for arm in arms_order:
        vs = per_arm_window_sharpes[arm]
        mean_s = float(np.mean(vs))
        pos_w = sum(1 for s in vs if s > 0)
        n_w = len(vs)
        arm_aggs[arm] = {
            'mean_val_sharpe': mean_s,
            'positive_windows': pos_w,
            'total_windows':    n_w,
            'per_window':       vs,
        }
        formatted = '  '.join(f'{s:+5.2f}' for s in vs)
        print(f'{arm:<28} {mean_s:>+9.3f} {pos_w:>3d}/{n_w}   {formatted}',
              flush=True)

    print('\n=== Pre-reg verdicts ===', flush=True)
    for arm in arms_order:
        a = arm_aggs[arm]
        m, pw = a['mean_val_sharpe'], a['positive_windows']
        if m >= 1.0 and pw >= 5:
            v = 'STRONG-PASS'
        elif m >= 0.50 and pw >= 4:
            v = 'PASS'
        elif 0.20 <= m < 0.50 and pw >= 3:
            v = 'MARGINAL'
        elif m < 0.20 or pw <= 2:
            v = 'FAIL'
        else:
            v = 'INCONCLUSIVE'
        print(f'  {arm:<28} mean {m:+.3f}  pos {pw}/{a["total_windows"]}  → {v}',
              flush=True)

    # Oracle ceiling capture rate.
    baseline = arm_aggs['all-pairs']['mean_val_sharpe']
    oracle = arm_aggs['oracle-pos']['mean_val_sharpe']
    ceiling = oracle - baseline
    print(f'\n  v0 baseline (all-pairs)  mean: {baseline:+.3f}', flush=True)
    print(f'  oracle-pos ceiling       mean: {oracle:+.3f}', flush=True)
    print(f'  oracle headroom         delta: {ceiling:+.3f}', flush=True)
    print(f'\n  Predictor arms — capture of oracle headroom:', flush=True)
    for arm in ['predictor-thr-0p5', f'predictor-top-{args.top_k_pred}']:
        d = arm_aggs[arm]['mean_val_sharpe'] - baseline
        pct = 100 * d / max(ceiling, 1e-9)
        print(f'    {arm:<28} delta {d:+.3f}  ({pct:.1f}% of oracle headroom)',
              flush=True)

    summary = {
        'universe_size':       len(universe),
        'min_history_bars':    args.min_history_bars,
        'train_window_days':   args.train_window_days,
        'val_window_days':     args.val_window_days,
        'step_window_days':    args.step_window_days,
        'abs_corr_min':        args.abs_corr_min,
        'eg_p_max':            args.eg_p_max,
        'top_k':               args.top_k,
        'top_k_pred':          args.top_k_pred,
        'entry_z':             args.entry_z,
        'exit_z':              args.exit_z,
        'commission_bps':      args.commission_bps,
        'lr_l2':               args.lr_l2,
        'feature_names':       FEATURE_NAMES,
        'arms':                arm_aggs,
        'per_window':          per_window_detail,
    }
    out_path = output / 'pairs-predictor-walkforward-summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n-> {out_path}', flush=True)

    # Per-pair feature dump for downstream consumers (apps/critic v0.1
    # value-function eval). all_pair_records is `[(w_idx, fv, bt.sharpe), …]`.
    if all_pair_records:
        w_idxs = np.array([r[0] for r in all_pair_records], dtype=np.int32)
        feats = np.stack([np.asarray(r[1], dtype=np.float64)
                          for r in all_pair_records], axis=0)
        labels = np.array([r[2] for r in all_pair_records], dtype=np.float64)
        npz_path = output / 'pairs-predictor-per-pair-records.npz'
        np.savez(npz_path,
                 window_idx=w_idxs,
                 features=feats,
                 realized_sharpe=labels,
                 feature_names=np.array(FEATURE_NAMES))
        print(f'-> {npz_path} ({len(all_pair_records)} per-pair records)',
              flush=True)


if __name__ == '__main__':
    main()
