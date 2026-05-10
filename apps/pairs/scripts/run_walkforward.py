"""Walk-forward eval of pair-spread mean reversion (classical baseline).

Pre-registered cuts (per `apps/docs/docs/TODO/apps-pairs.md`):

  PASS         : aggregate val Sharpe ≥ +0.50 with ≥ 4/6 windows
                 positive → ship classical baseline (or proceed to
                 ML head if dispersion suggests model lift)
  MARGINAL     : +0.20 to +0.50 with ≥ 3/6 → stratify by liquidity
                 / sector before deciding
  FAIL         : < +0.20 *or* ≤ 2/6 → confirmed-null; pivot to
                 apps/vol

Run from repo root:
    uv run python apps/pairs/scripts/run_walkforward.py
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

from pairs import (
    PairCandidate, aggregate_pair_pnl, backtest_pair, screen_pairs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
DEFAULT_OUTPUT = REPO_ROOT / 'Output'


def _resolve_universe(min_history_bars: int, max_tickers: int) -> list[str]:
    manifest = json.loads((STOOQ_SUBSET / 'manifest.json').read_text())
    entries = list(manifest['tickers'])
    before = len(entries)
    entries = [t for t in entries if t['n_bars'] >= min_history_bars]
    print(f'manifest: {before} → {len(entries)} pass '
          f'min_history_bars={min_history_bars}', flush=True)
    names = [t['ticker'].upper() for t in entries]
    if max_tickers > 0:
        names = names[:max_tickers]
        print(f'  capped to first {max_tickers}', flush=True)
    return names


def _build_window_slices(
    n: int, train_w: int, val_w: int, step: int,
) -> list[tuple[int, int, int]]:
    out = []
    start = 0
    while start + train_w + val_w <= n:
        out.append((start, start + train_w, start + train_w + val_w))
        start += step
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--start', default='2000-01-01')
    p.add_argument('--end',   default='2025-12-11')
    p.add_argument('--min-history-bars', type=int, default=6500)
    p.add_argument('--max-tickers',      type=int, default=0)
    p.add_argument('--train-window-days', type=int, default=1260)  # ~5y
    p.add_argument('--val-window-days',   type=int, default=780)   # ~3y
    p.add_argument('--step-window-days',  type=int, default=780)
    p.add_argument('--abs-corr-min', type=float, default=0.7)
    p.add_argument('--eg-p-max',     type=float, default=0.05)
    p.add_argument('--top-k',        type=int,   default=50)
    p.add_argument('--entry-z',      type=float, default=2.0)
    p.add_argument('--exit-z',       type=float, default=0.5)
    p.add_argument('--commission-bps', type=float, default=10.0)
    p.add_argument('--n-workers', type=int, default=mp.cpu_count())
    p.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f'pairs walk-forward: top-K={args.top_k}, '
          f'|corr|≥{args.abs_corr_min}, EG p<{args.eg_p_max}, '
          f'z=±{args.entry_z} entry / ±{args.exit_z} exit, '
          f'{args.commission_bps}bps × 2 commission per leg-flip',
          flush=True)
    universe = _resolve_universe(args.min_history_bars, args.max_tickers)
    print(f'\nLoading {len(universe)} tickers...', flush=True)
    t0 = time.perf_counter()
    prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'  loaded {prices.shape[1]} x {prices.shape[0]} bars in '
          f'{time.perf_counter()-t0:.1f}s', flush=True)

    log_prices = np.log(prices)
    dates = prices.index
    n = len(dates)

    windows = _build_window_slices(
        n, args.train_window_days, args.val_window_days,
        args.step_window_days)
    if not windows:
        raise SystemExit(
            f'no windows fit: have {n} bars but need '
            f'train+val={args.train_window_days + args.val_window_days}')
    print(f'\nwalk-forward: {len(windows)} windows '
          f'(train={args.train_window_days} / val={args.val_window_days} '
          f'/ step={args.step_window_days} bars)', flush=True)

    rows = []
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
            print('  [skip] no pairs passed screening', flush=True)
            rows.append({
                'window_idx': w_idx, 'n_pairs': 0,
                'val_sharpe': 0.0, 'mean_pair_sharpe': 0.0,
                'pos_pair_frac': 0.0,
                'val_start': str(dates[mid].date()),
                'val_end':   str(dates[hi - 1].date()),
            })
            continue

        # Backtest survivors on val.
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

        # Aggregate. Use the val_log dates from the union of pair val
        # spans — pairs may have different per-pair val spans due to
        # missing-data cleanup.
        val_dates = val_log.index
        # Reindex each pair's val_daily_ret onto val_dates with
        # 0-fill for missing.
        agg_arr = np.zeros(len(val_dates), dtype=np.float64)
        for bt in pair_results:
            ser = pd.Series(bt.val_daily_ret,
                            index=val_log[[bt.a, bt.b]].dropna().index)
            ser_full = ser.reindex(val_dates).fillna(0.0)
            agg_arr += ser_full.values / max(len(pair_results), 1)
        agg_series = pd.Series(agg_arr, index=val_dates)
        from ss_portfolio.metrics import (
            annualized_sharpe, cagr, max_drawdown, sortino,
        )
        agg_sharpe = float(annualized_sharpe(agg_series))
        agg_sortino = float(sortino(agg_series))
        agg_cagr = float(cagr(agg_series) * 100.0)
        agg_dd = float(max_drawdown(agg_series) * 100.0)
        pair_sharpes = np.array([r.sharpe for r in pair_results])
        mean_pair_sh = float(np.mean(pair_sharpes)) if len(pair_sharpes) else 0.0
        pos_frac = float(np.mean(pair_sharpes > 0)) if len(pair_sharpes) else 0.0

        print(f'\n  pairs traded: {len(pair_results)} (skipped {skipped})',
              flush=True)
        print(f'  agg Sharpe:        {agg_sharpe:+.3f}  Sortino: '
              f'{agg_sortino:+.3f}  CAGR: {agg_cagr:+.2f}%  '
              f'maxDD: {agg_dd:+.2f}%', flush=True)
        print(f'  mean pair Sharpe:  {mean_pair_sh:+.3f}  '
              f'pos pair frac: {pos_frac:.2f}', flush=True)
        print('  top 5 pairs by val Sharpe:', flush=True)
        top5 = sorted(pair_results, key=lambda r: -r.sharpe)[:5]
        for r in top5:
            print(f'    {r.a:>5s}/{r.b:<5s}  Sh={r.sharpe:+6.2f}  '
                  f'Sort={r.sortino:+6.2f}  trades={r.n_trades:>3d}  '
                  f'in_trade={r.pct_in_trade:.2f}  '
                  f'half_life={r.train_half_life:.1f}', flush=True)

        rows.append({
            'window_idx': w_idx,
            'val_start': str(dates[mid].date()),
            'val_end':   str(dates[hi - 1].date()),
            'n_pairs': len(pair_results),
            'val_sharpe': agg_sharpe,
            'val_sortino': agg_sortino,
            'val_cagr_pct': agg_cagr,
            'val_max_drawdown_pct': agg_dd,
            'mean_pair_sharpe': mean_pair_sh,
            'pos_pair_frac': pos_frac,
            'pairs': [
                {'a': r.a, 'b': r.b, 'sharpe': r.sharpe,
                 'sortino': r.sortino, 'n_trades': r.n_trades,
                 'pct_in_trade': r.pct_in_trade,
                 'train_half_life': r.train_half_life}
                for r in pair_results
            ],
        })

    print('\n' + '=' * 96, flush=True)
    val_sharpes = [r['val_sharpe'] for r in rows]
    pos_window_frac = float(np.mean([s > 0 for s in val_sharpes]))
    mean_val_sharpe = float(np.mean(val_sharpes))
    print(f'mean agg val Sharpe = {mean_val_sharpe:+.3f}  '
          f'(positive windows: {pos_window_frac:.2f}, '
          f'{int(round(pos_window_frac * len(rows)))}/{len(rows)})',
          flush=True)

    # Pre-registered verdict.
    if mean_val_sharpe >= 0.50 and pos_window_frac >= 4 / 6:
        verdict = ('PASS — pair-spread mean reversion is shippable; '
                   'build live deployment in apps/pairs')
    elif mean_val_sharpe >= 0.20 and pos_window_frac >= 3 / 6:
        verdict = ('MARGINAL — partial-OOS; stratify by liquidity / '
                   'sector before deciding')
    elif mean_val_sharpe < 0.20 or pos_window_frac <= 2 / 6:
        verdict = ('FAIL (confirmed-null) — pair-spread mean reversion '
                   'is not an alpha source on this universe at this '
                   'horizon. Pivot to apps/vol.')
    else:
        verdict = 'INCONCLUSIVE'
    print(f'verdict: {verdict}', flush=True)

    out_path = output / 'pairs-walkforward-summary.json'
    out_path.write_text(json.dumps({
        'universe_size': len(universe),
        'min_history_bars': args.min_history_bars,
        'train_window_days': args.train_window_days,
        'val_window_days': args.val_window_days,
        'step_window_days': args.step_window_days,
        'abs_corr_min': args.abs_corr_min,
        'eg_p_max': args.eg_p_max,
        'top_k': args.top_k,
        'entry_z': args.entry_z, 'exit_z': args.exit_z,
        'commission_bps': args.commission_bps,
        'mean_val_sharpe': mean_val_sharpe,
        'positive_window_fraction': pos_window_frac,
        'verdict': verdict,
        'per_window': rows,
    }, indent=2))
    print(f'\n-> {out_path}', flush=True)


if __name__ == '__main__':
    main()
