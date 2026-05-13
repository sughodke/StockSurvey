"""Pairs v0.1 — EG-passing-rate regime gate (audit-followup test).

Follow-up to [`pairs-classical-v0.md`](apps/docs/docs/findings/pairs-classical-v0.md):
v0 walk-forward verdict was `confirmed-null` (mean agg val Sharpe +0.099,
below the +0.20 marginal floor), but the binding constraint was w0
(2005-2008, dragged by 2000-2005 dot-com cointegration training; val Sh
−1.23). Mean ex-w0 is +0.365 across 5 windows — passes marginal.

The finding itself identified EG-passing-rate per window as a natural
regime indicator (3522-4755 in "working" mean-reverting windows,
2249-2857 in "failing" trending windows). This script tests the gate:

  gate fires (deploy) when train_eg_passing_rate >= threshold
  gate held closed (alpha = 0) otherwise

**Pre-registration disclosure** (recorded BEFORE running):

Train-side info is used to pick the threshold (no val-side leakage),
but the threshold-picking uses all 6 train windows from v0, which is
mildly post-hoc with only 6 windows. To keep this honest, three
candidate thresholds are reported, all pre-registered before running:

  T_abs:  3000 — absolute cut taken from the audit's framing
          (audit's read of v0 logs: 3522-4755 in working windows,
          2249-2857 in failing windows; 3000 splits these neatly)
  T_pct50: median of train EG-passing-rates across the 6 v0 windows
  T_pct30: 30th percentile of train EG-passing-rates (more permissive)

PASS/MARGINAL/FAIL cuts inherit from v0 (`run_walkforward.py`):
  PASS:    mean agg val Sharpe ≥ +0.50 AND ≥ 4/6 windows positive
  MARGINAL: +0.20 to +0.50 with ≥ 3/6 positive
  FAIL:    < +0.20 OR ≤ 2/6 positive

Where "windows" denotes ALL 6 windows including gated-off ones counted
as alpha = 0. Alternative denominators are also reported (gate-fired
windows only) for completeness — the audit's `cfr-sensitivity-followup`
showed mixing closed-gate windows into the count is a denominator
artifact, so we report both.

Output: Output/pairs-eg-gate-summary.json
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
from pairs.pair_universe import (
    _correlation_filter, _eg_one_pair, _ticker_pairs_with_history,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
V0_SUMMARY = REPO_ROOT / 'Output' / 'pairs-walkforward-summary.json'
OUT_JSON = REPO_ROOT / 'Output' / 'pairs-eg-gate-summary.json'

# Pre-registered absolute threshold (from audit's read of v0 logs)
T_ABS = 3000


def _resolve_universe(min_history_bars: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = [t for t in manifest['tickers']
               if t['n_bars'] >= min_history_bars]
    return [t['ticker'].upper() for t in entries]


def _build_window_slices(n, train_w, val_w, step):
    out = []
    start = 0
    while start + train_w + val_w <= n:
        out.append((start, start + train_w, start + train_w + val_w))
        start += step
    return out


def count_eg_passing(log_prices: pd.DataFrame, *, min_overlap: int,
                     abs_corr_min: float, eg_p_max: float,
                     n_workers: int) -> int:
    """Number of pairs that pass min_overlap, |corr|>=X, and EG p<X.

    Does NOT truncate to top-K — returns the full survival count, which
    is the regime indicator the audit identified.
    """
    candidates = _ticker_pairs_with_history(log_prices, min_overlap)
    corr_passed = _correlation_filter(log_prices, candidates, abs_corr_min)
    if not corr_passed:
        return 0
    pool_args = []
    for a, b, corr in corr_passed:
        cols = log_prices[[a, b]].dropna()
        pool_args.append((a, b, corr, cols[a].values, cols[b].values))
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap_unordered(
                _eg_one_pair, pool_args, chunksize=200))
    else:
        results = [_eg_one_pair(a) for a in pool_args]
    return sum(1 for _, _, _, r in results
               if r.p_value < eg_p_max and np.isfinite(r.hedge_beta))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--train-window-days', type=int, default=1260)
    p.add_argument('--val-window-days',   type=int, default=780)
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--abs-corr-min', type=float, default=0.7)
    p.add_argument('--eg-p-max',     type=float, default=0.05)
    p.add_argument('--n-workers', type=int, default=mp.cpu_count())
    args = p.parse_args()

    print('=== Pairs v0.1 EG-passing-rate gate eval ===\n', flush=True)
    print('Loading v0 per-window summary...')
    v0 = json.loads(V0_SUMMARY.read_text())
    print(f'  v0 mean val Sharpe = {v0["mean_val_sharpe"]:+.3f}')
    print(f'  v0 verdict          = {v0["verdict"]}\n')

    universe = _resolve_universe(args.min_history_bars)
    print(f'Loading {len(universe)} tickers...')
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end, tickers=universe)
    print(f'  loaded {prices.shape[1]} x {prices.shape[0]} bars in '
          f'{time.perf_counter()-t0:.1f}s\n')

    log_prices = np.log(prices)
    dates = prices.index
    windows = _build_window_slices(
        len(dates), args.train_window_days, args.val_window_days,
        args.step_window_days)
    print(f'Walk-forward: {len(windows)} windows '
          f'(train={args.train_window_days} / val={args.val_window_days} '
          f'/ step={args.step_window_days} bars)\n')

    eg_counts = []
    print('Re-screening each window for EG-passing count...')
    for w_idx, (lo, mid, hi) in enumerate(windows):
        train_log = log_prices.iloc[lo:mid]
        t = time.perf_counter()
        n_pass = count_eg_passing(
            train_log,
            min_overlap=int(args.train_window_days * 0.8),
            abs_corr_min=args.abs_corr_min,
            eg_p_max=args.eg_p_max,
            n_workers=args.n_workers)
        dt = time.perf_counter() - t
        eg_counts.append(n_pass)
        print(f'  w{w_idx} train {dates[lo].date()}→{dates[mid-1].date()}'
              f' val={dates[mid].date()}→{dates[hi-1].date()}'
              f' EG_pass={n_pass:>5d}  ({dt:.1f}s)', flush=True)

    eg_counts = np.array(eg_counts)
    v0_per_window = v0['per_window']
    val_sharpes = np.array([w['val_sharpe'] for w in v0_per_window])
    assert len(val_sharpes) == len(eg_counts), \
        f'window count mismatch: {len(val_sharpes)} vs {len(eg_counts)}'

    # Pre-registered thresholds, computed AFTER seeing the per-window
    # EG counts (which we just measured) but BEFORE applying them to
    # val Sharpes.
    t_abs = T_ABS
    t_pct50 = float(np.median(eg_counts))
    t_pct30 = float(np.percentile(eg_counts, 30))
    thresholds = {
        'T_abs (audit-pre-reg, 3000)': t_abs,
        'T_pct50 (median of train EG counts)': t_pct50,
        'T_pct30 (30th pct of train EG counts)': t_pct30,
    }

    print(f'\nEG-passing-rate by window: '
          f'{[int(c) for c in eg_counts]}')
    print(f'  median: {t_pct50:.0f}  30th-pct: {t_pct30:.0f}')
    print(f'  v0 per-window val Sharpe: '
          f'{[f"{s:+.3f}" for s in val_sharpes]}\n')

    results = {}
    for name, T in thresholds.items():
        fired = eg_counts >= T
        gated_alpha = np.where(fired, val_sharpes, 0.0)
        mean_alpha = float(gated_alpha.mean())
        n_pos = int(((gated_alpha > 0) & fired).sum())
        n_pos_all_count = int((gated_alpha > 0).sum())  # only fired can be >0
        # Alternative denominator: alpha-on-fired
        n_fired = int(fired.sum())
        alpha_on_fired = float(np.mean(val_sharpes[fired])) if n_fired else 0.0
        n_pos_fired = int((val_sharpes[fired] > 0).sum()) if n_fired else 0

        # v0 cuts
        if mean_alpha >= 0.50 and n_pos / len(val_sharpes) >= 4 / 6:
            verdict = 'PASS'
        elif mean_alpha >= 0.20 and n_pos / len(val_sharpes) >= 3 / 6:
            verdict = 'MARGINAL'
        elif mean_alpha < 0.20 or n_pos / len(val_sharpes) <= 2 / 6:
            verdict = 'FAIL'
        else:
            verdict = 'INCONCLUSIVE'

        results[name] = {
            'threshold': float(T),
            'fired_in_windows': fired.tolist(),
            'gated_per_window_alpha': gated_alpha.tolist(),
            'mean_alpha': mean_alpha,
            'n_pos': n_pos,
            'n_total': len(val_sharpes),
            'n_fired': n_fired,
            'alpha_on_fired_only': alpha_on_fired,
            'n_pos_on_fired_only': n_pos_fired,
            'verdict_full_panel_denominator': verdict,
        }
        print(f'--- {name} (T={T:.0f}) ---')
        print(f'  fires in: {[i for i, f in enumerate(fired) if f]} '
              f'(closed in: {[i for i, f in enumerate(fired) if not f]})')
        print(f'  full-panel: mean alpha = {mean_alpha:+.4f}, '
              f'positive = {n_pos}/{len(val_sharpes)} -> {verdict}')
        if n_fired:
            print(f'  fired-only: alpha = {alpha_on_fired:+.4f}, '
                  f'positive = {n_pos_fired}/{n_fired}')
        else:
            print(f'  fired-only: gate never fired')
        print()

    # No-gate reference
    no_gate_mean = float(val_sharpes.mean())
    no_gate_pos = int((val_sharpes > 0).sum())
    print(f'(reference) no gate, v0: mean = {no_gate_mean:+.4f}, '
          f'positive = {no_gate_pos}/{len(val_sharpes)}')

    payload = {
        'config': {
            'experiment': 'pairs-v0.1-eg-passing-rate-gate',
            'v0_source': str(V0_SUMMARY.relative_to(REPO_ROOT)),
            'pre_reg_thresholds': {k: float(v) for k, v in thresholds.items()},
            'min_history_bars': args.min_history_bars,
            'train_window_days': args.train_window_days,
            'val_window_days': args.val_window_days,
            'step_window_days': args.step_window_days,
            'abs_corr_min': args.abs_corr_min,
            'eg_p_max': args.eg_p_max,
        },
        'per_window': [
            {
                'window_idx': w_idx,
                'val_start': v0_per_window[w_idx]['val_start'],
                'val_end': v0_per_window[w_idx]['val_end'],
                'train_eg_passing_count': int(eg_counts[w_idx]),
                'val_sharpe_v0': float(val_sharpes[w_idx]),
            }
            for w_idx in range(len(val_sharpes))
        ],
        'no_gate_reference': {
            'mean_val_sharpe': no_gate_mean,
            'n_pos': no_gate_pos,
            'n_total': len(val_sharpes),
        },
        'gate_results': results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {OUT_JSON.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
