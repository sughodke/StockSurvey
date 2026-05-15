"""Pairs v0 — hindsight oracles on pair selection + per-window gate.

Cross-app oracle diagnostic borrowed from factor + vol + gate. The
[`pairs-eg-gate-falsified`](../../docs/docs/findings/pairs-eg-gate-falsified.md)
finding closed `confirmed-null` on the EG-passing-rate regime gate
(best-gate alpha +0.104 vs audit's predicted +0.5). The arc-level
question that closure flagged: is there ANY gate that lifts above
+0.20, or is the architecture at its ceiling?

The hindsight oracle answers this by directly using realized val data
to (a) pick which screened pairs to keep, and (b) decide whether to
deploy each window at all. Strict upper bound on any real-time
selector — heuristic, learned, EG-pass-based, or otherwise.

Five arms on the same 6 walk-forward windows as v0:

  - all-pairs                v0 baseline (1/N equal-weight)
  - oracle-pos-pairs         keep pairs with val Sharpe > 0 only
  - oracle-top-half          top 50% of pairs by val Sharpe
  - oracle-top-quartile      top 25% of pairs
  - window-gate-oracle       skip windows where v0 aggregate Sharpe < 0
                             (gate the WINDOW, not individual pairs)

Pre-reg cuts (inherited from v0):
  PASS:     mean agg val Sharpe ≥ +0.50 AND ≥ 4/6 windows positive
  MARGINAL: mean ∈ [+0.20, +0.50] AND ≥ 3/6 positive
  FAIL:    < +0.20 OR ≤ 2/6 positive

Verdict logic:
  - If oracle clears PASS (+0.50): predictor of pair-quality is real;
    pair selection has substantial upside; reopens predictor research.
  - If oracle clears MARGINAL (+0.20) but not PASS: signal exists but
    architecture is near its ceiling; verdict remains partial-OOS.
  - If oracle FAILS even +0.20: pair-spread mean reversion is null
    even with perfect hindsight; arc closes harder.

Run from repo root:
    uv run python apps/pairs/scripts/run_oracle_walkforward.py
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
from ss_portfolio.metrics import (
    annualized_sharpe, cagr, max_drawdown, sortino,
)
from pairs.backtest import backtest_pair
from pairs.pair_universe import screen_pairs


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


def _aggregate_arm(
    pair_results: list,
    val_dates: pd.DatetimeIndex,
    log_subset: pd.DataFrame,
) -> dict:
    """Equal-weight 1/N aggregation of per-pair daily PnL into a single
    aggregate daily PnL stream; compute annualized Sharpe + sortino +
    CAGR + max drawdown.
    """
    if not pair_results:
        return {
            'n_pairs': 0,
            'sharpe': 0.0, 'sortino': 0.0,
            'cagr_pct': 0.0, 'max_drawdown_pct': 0.0,
            'mean_pair_sharpe': 0.0, 'pos_pair_frac': 0.0,
        }
    n = len(pair_results)
    agg = np.zeros(len(val_dates), dtype=np.float64)
    for bt in pair_results:
        pair_idx = log_subset[[bt.a, bt.b]].dropna().index
        ser = pd.Series(bt.val_daily_ret, index=pair_idx)
        ser_full = ser.reindex(val_dates).fillna(0.0)
        agg += ser_full.values / n
    s = pd.Series(agg, index=val_dates)
    return {
        'n_pairs': n,
        'sharpe': float(annualized_sharpe(s)),
        'sortino': float(sortino(s)),
        'cagr_pct': float(cagr(s) * 100.0),
        'max_drawdown_pct': float(max_drawdown(s) * 100.0),
        'mean_pair_sharpe': float(np.mean([r.sharpe for r in pair_results])),
        'pos_pair_frac': float(np.mean(
            [r.sharpe > 0 for r in pair_results])),
    }


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
    p.add_argument('--n-workers', type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print('Loading universe (stooq_us_long)...', flush=True)
    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers']
                      if t.get('n_bars', 0) >= args.min_history_bars)
    print(f'  {len(universe)} tickers pass min_history_bars filter',
          flush=True)
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} tickers x {prices.shape[0]} bars '
          f'in {time.perf_counter()-t0:.1f}s', flush=True)

    log_prices = np.log(prices)
    dates = prices.index
    n = len(dates)

    windows = _build_window_slices(
        n, args.train_window_days, args.val_window_days,
        args.step_window_days)
    if not windows:
        raise SystemExit(f'no windows fit: have {n} bars')
    print(f'\nwalk-forward: {len(windows)} windows', flush=True)

    # Per-window: screen, backtest each pair, then compute multiple arms.
    per_window: list[dict] = []
    arms_order = ['all-pairs', 'oracle-pos-pairs', 'oracle-top-half',
                  'oracle-top-quartile', 'window-gate-oracle']
    per_arm_window_sharpes: dict[str, list[float]] = {a: [] for a in arms_order}

    for w_idx, (lo, mid, hi) in enumerate(windows):
        print(f'\n=== window {w_idx} '
              f'({dates[lo].date()} → {dates[mid - 1].date()} train, '
              f'{dates[mid].date()} → {dates[hi - 1].date()} val) ===',
              flush=True)
        train_log = log_prices.iloc[lo:mid]
        val_log   = log_prices.iloc[mid:hi]

        pairs_screened = screen_pairs(
            train_log,
            min_overlap=int(args.train_window_days * 0.8),
            abs_corr_min=args.abs_corr_min,
            eg_p_max=args.eg_p_max,
            top_k=args.top_k,
            n_workers=args.n_workers,
            verbose=True)
        if not pairs_screened:
            for arm in arms_order:
                per_arm_window_sharpes[arm].append(0.0)
            per_window.append({
                'window_idx': w_idx, 'n_pairs_screened': 0,
                'arms': {a: {'sharpe': 0.0, 'n_pairs': 0} for a in arms_order},
            })
            continue

        # Backtest each pair on val.
        pair_results = []
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
            except Exception as e:
                print(f'  ! backtest failed for {c.a}/{c.b}: {e}',
                      flush=True)
                skipped += 1
                continue
            pair_results.append(bt)

        val_dates = val_log.index

        # Arm 1: all pairs (v0 baseline).
        arm_all = _aggregate_arm(pair_results, val_dates, val_log)

        # Arm 2: oracle-pos-pairs — keep pairs with val Sharpe > 0.
        pos_pairs = [r for r in pair_results if r.sharpe > 0]
        arm_pos = _aggregate_arm(pos_pairs, val_dates, val_log)

        # Arm 3: oracle-top-half — top 50% by val Sharpe.
        sorted_by_sh = sorted(pair_results, key=lambda r: -r.sharpe)
        n_half = max(1, len(sorted_by_sh) // 2)
        arm_top_half = _aggregate_arm(
            sorted_by_sh[:n_half], val_dates, val_log)

        # Arm 4: oracle-top-quartile — top 25%.
        n_quarter = max(1, len(sorted_by_sh) // 4)
        arm_top_q = _aggregate_arm(
            sorted_by_sh[:n_quarter], val_dates, val_log)

        # Arm 5: window-gate-oracle — use all pairs iff v0 baseline > 0,
        # else skip (alpha = 0).
        if arm_all['sharpe'] > 0:
            arm_window = arm_all.copy()
        else:
            arm_window = {
                'n_pairs': 0, 'sharpe': 0.0, 'sortino': 0.0,
                'cagr_pct': 0.0, 'max_drawdown_pct': 0.0,
                'mean_pair_sharpe': 0.0, 'pos_pair_frac': 0.0,
            }

        arms = {
            'all-pairs':           arm_all,
            'oracle-pos-pairs':    arm_pos,
            'oracle-top-half':     arm_top_half,
            'oracle-top-quartile': arm_top_q,
            'window-gate-oracle':  arm_window,
        }
        for a in arms_order:
            per_arm_window_sharpes[a].append(arms[a]['sharpe'])
        per_window.append({
            'window_idx': w_idx,
            'val_start':  str(dates[mid].date()),
            'val_end':    str(dates[hi - 1].date()),
            'n_pairs_screened': len(pair_results),
            'arms': {a: {k: arms[a][k] for k in (
                'n_pairs', 'sharpe', 'mean_pair_sharpe', 'pos_pair_frac')}
                     for a in arms_order},
        })
        print(f'  pairs backtested: {len(pair_results)} '
              f'(skipped {skipped})', flush=True)
        for a in arms_order:
            print(f'    {a:<22} Sharpe={arms[a]["sharpe"]:+7.3f}  '
                  f'n_pairs={arms[a]["n_pairs"]}', flush=True)

    # Aggregate cross-window.
    print('\n' + '=' * 96, flush=True)
    print(f'{"arm":<22} {"mean":>9s} {"pos_w":>7s} '
          f'{"per-window val Sharpes":>40s}', flush=True)
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
            'per_window_val_sharpe': vs,
        }
        formatted = '  '.join(f'{s:+5.2f}' for s in vs)
        print(f'{arm:<22} {mean_s:>+9.3f} {pos_w:>3d}/{n_w}   {formatted}',
              flush=True)

    # Verdicts per arm.
    print('\n=== Pre-reg verdicts ===', flush=True)
    for arm in arms_order:
        a = arm_aggs[arm]
        m, pw, nw = a['mean_val_sharpe'], a['positive_windows'], a['total_windows']
        if m >= 0.50 and pw >= 4:
            v = 'PASS'
        elif 0.20 <= m < 0.50 and pw >= 3:
            v = 'MARGINAL'
        elif m < 0.20 or pw <= 2:
            v = 'FAIL'
        else:
            v = 'INCONCLUSIVE'
        print(f'  {arm:<22} mean {m:+.3f}  pos {pw}/{nw}  → {v}',
              flush=True)

    # Decomposition vs v0 baseline.
    baseline = arm_aggs['all-pairs']['mean_val_sharpe']
    print(f'\nv0 baseline (all-pairs)         mean: {baseline:+.3f}',
          flush=True)
    for arm in arms_order[1:]:
        d = arm_aggs[arm]['mean_val_sharpe'] - baseline
        print(f'  {arm:<22} delta vs v0 baseline {d:+.3f}', flush=True)

    summary = {
        'universe_size':       len(universe),
        'min_history_bars':    args.min_history_bars,
        'train_window_days':   args.train_window_days,
        'val_window_days':     args.val_window_days,
        'step_window_days':    args.step_window_days,
        'abs_corr_min':        args.abs_corr_min,
        'eg_p_max':            args.eg_p_max,
        'top_k':               args.top_k,
        'entry_z':             args.entry_z,
        'exit_z':              args.exit_z,
        'commission_bps':      args.commission_bps,
        'arms':                arm_aggs,
        'per_window':          per_window,
    }
    out_path = output / 'pairs-oracle-walkforward-summary.json'
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
