"""Passive equal-weight benchmark on Phase-2 / stooq_us_long / ex-Phase-2.

Settles the leaderboard's pending question: how much of the relational
analog cross_ticker val Sharpe (1.146 Phase-2, 0.484 ex-Phase-2, 0.717
Morlet stooq_us_long) is alpha vs market beta of the chosen universe.

For each universe, computes two passive arms over the canonical Phase-2
train/val split (train 2013-01-29 → 2020-12-31, val 2021-01-01 →
2025-12-11):

  - buy_and_hold  : 1/N at t=0, weights drift with prices, no commission.
  - ew_rebal20    : reset to 1/N every 20 trading bars, 10bps commission
                    on L1 turnover at each rebal — matched to the
                    canonical relational checkpoint convention so the
                    comparison vs model val Sharpe is apples-to-apples.

Run:
    uv run python apps/relational/scripts/equal_weight_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import load_stooq_matrix
from ss_portfolio.metrics import (
    annualized_sharpe, cagr, max_drawdown, sortino,
)

from relational.sectors import PHASE2_TICKERS


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / 'apps/notebook/data/stooq_us_long/manifest.json'
DEFAULT_FACTOR_WIDE_PKL = REPO_ROOT / 'Output/universe_pivot_close.pkl'
OUTPUT_DIR = REPO_ROOT / 'Output'

TRAIN_START = '2013-01-29'
TRAIN_END = '2020-12-31'
VAL_START = '2021-01-01'
VAL_END = '2025-12-11'


def equal_weight_returns(
    prices: pd.DataFrame, start: str, end: str,
    rebal_days: int = 0, commission_bps: float = 0.0,
) -> tuple[pd.Series, int]:
    """Daily simple-return series of an equal-weight portfolio.

    Slices `prices` to the [start, end] window first, ffills, then drops
    columns whose pre-window IPO leaves leading NaNs in the slice. This
    way the val window isn't penalized by ticker IPOs that pre-date val
    but post-date train.

    `rebal_days=0` → buy-and-hold (no rebalancing, no commission).
    `rebal_days>0` → reset to 1/N every N bars, commission applied to
    L1 turnover at each rebal.
    """
    p = prices.loc[(prices.index >= start) & (prices.index <= end)]
    p = p.ffill().dropna(axis=1)
    n_dates, n_tickers = p.shape
    if n_tickers == 0:
        raise ValueError(f'no tickers with full coverage in {start}..{end}')
    daily_ret = p.pct_change().fillna(0.0).values    # (T, N)
    target_w = 1.0 / n_tickers
    w = np.full(n_tickers, target_w)
    port_ret = np.zeros(n_dates)
    fee = commission_bps / 10_000.0
    for t in range(1, n_dates):
        gross_w = w * (1.0 + daily_ret[t])
        gross_ret = gross_w.sum() - 1.0
        denom = 1.0 + gross_ret
        w = gross_w / denom if denom != 0 else gross_w
        port_ret[t] = gross_ret
        if rebal_days > 0 and t % rebal_days == 0:
            turnover = float(np.abs(w - target_w).sum())
            port_ret[t] -= fee * turnover
            w = np.full(n_tickers, target_w)
    return pd.Series(port_ret[1:], index=p.index[1:]), n_tickers


def report(name: str, daily_ret: pd.Series, n_tickers: int) -> dict:
    if len(daily_ret) == 0:
        return {'name': name, 'n_days': 0, 'n_tickers': n_tickers}
    return {
        'name': name,
        'n_days': int(len(daily_ret)),
        'n_tickers': n_tickers,
        'sharpe': float(annualized_sharpe(daily_ret)),
        'sortino': float(sortino(daily_ret)),
        'cagr': float(cagr(daily_ret)),
        'maxdd': float(max_drawdown(daily_ret)),
    }


def run_universe(label: str, prices: pd.DataFrame) -> dict[str, dict]:
    n_total = prices.shape[1]
    print(
        f'\n=== {label} '
        f'({n_total} tickers loaded, '
        f'{prices.index[0].date()} → {prices.index[-1].date()}) ==='
    )
    arms: dict[str, dict] = {}
    windows = [
        ('full', TRAIN_START, VAL_END),
        ('train', TRAIN_START, TRAIN_END),
        ('val', VAL_START, VAL_END),
    ]
    arm_specs = [
        ('buy_and_hold', 0, 0.0),
        ('ew_rebal20_10bps', 20, 10.0),
    ]
    for arm_name, rebal_days, commission_bps in arm_specs:
        for window_label, start, end in windows:
            try:
                series, n_held = equal_weight_returns(
                    prices, start, end,
                    rebal_days=rebal_days, commission_bps=commission_bps)
                row = report(f'{arm_name}-{window_label}', series, n_held)
            except ValueError as exc:
                row = {'name': f'{arm_name}-{window_label}',
                       'n_days': 0, 'n_tickers': 0, 'error': str(exc)}
            arms[f'{arm_name}-{window_label}'] = row
    header = (f'{"arm":20s} {"window":6s} {"N":>4s} {"Sharpe":>8s} '
              f'{"Sortino":>8s} {"CAGR":>8s} {"MaxDD":>8s}')
    print(header)
    print('-' * len(header))
    for arm_name, _, _ in arm_specs:
        for window_label, _, _ in windows:
            r = arms[f'{arm_name}-{window_label}']
            if r['n_days'] == 0:
                print(f'{arm_name:20s} {window_label:6s} '
                      f'(empty: {r.get("error", "no data")})')
                continue
            print(f'{arm_name:20s} {window_label:6s} '
                  f'{r["n_tickers"]:>4d} {r["sharpe"]:>+8.3f} '
                  f'{r["sortino"]:>+8.3f} {r["cagr"]*100:>+7.2f}% '
                  f'{r["maxdd"]*100:>+7.2f}%')
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./StooqData')
    ap.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    ap.add_argument('--factor-wide-pkl', default=str(DEFAULT_FACTOR_WIDE_PKL),
                    help='Pre-built factor-wide close panel pickle '
                         '(from prep_universe_pivot_data.py). '
                         'Skipped if missing.')
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    print('Loading universes...')
    p2_prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=list(PHASE2_TICKERS))
    results['phase-2'] = run_universe('Phase-2 (mega-caps)', p2_prices)

    manifest = json.loads(Path(args.manifest).read_text())
    long_universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    long_prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=long_universe)
    results['stooq_us_long'] = run_universe('stooq_us_long', long_prices)

    ex_universe = sorted(t for t in long_universe
                         if t not in set(PHASE2_TICKERS))
    ex_prices, _, _, _ = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=TRAIN_START, end_date=VAL_END,
        tickers=ex_universe)
    results['ex-phase-2'] = run_universe('ex-Phase-2', ex_prices)

    factor_wide_pkl = Path(args.factor_wide_pkl)
    if factor_wide_pkl.exists():
        # Reuse the cached close panel from prep_universe_pivot_data.py
        # (filtered to first_valid_index ≤ 2010-01-01, ~2073 names).
        # Avoids re-running the slow ~10 min full-archive load.
        with factor_wide_pkl.open('rb') as f:
            wide_prices = pickle.load(f)
        results['factor-wide'] = run_universe(
            'factor-wide (full archive, 2010-grace)', wide_prices)
    else:
        print(f'\n[skip] factor-wide: {factor_wide_pkl} not found '
              f'— run apps/factor/scripts/modal/prep_universe_pivot_data.py')

    print('\n=== Headline: passive val Sharpe vs model val Sharpe ===')
    headline = [
        ('Phase-2 analog cross_ticker (Ricker)',
         results['phase-2']['ew_rebal20_10bps-val']['sharpe'], 1.146),
        ('Phase-2 buy-and-hold',
         results['phase-2']['buy_and_hold-val']['sharpe'], 1.146),
        ('stooq_us_long analog Morlet',
         results['stooq_us_long']['ew_rebal20_10bps-val']['sharpe'], 0.717),
        ('stooq_us_long buy-and-hold',
         results['stooq_us_long']['buy_and_hold-val']['sharpe'], 0.717),
        ('ex-Phase-2 analog cross_ticker (Ricker)',
         results['ex-phase-2']['ew_rebal20_10bps-val']['sharpe'], 0.484),
        ('ex-Phase-2 buy-and-hold',
         results['ex-phase-2']['buy_and_hold-val']['sharpe'], 0.484),
    ]
    print(f'{"comparison":50s} {"passive":>10s} {"model":>10s} {"alpha":>10s}')
    print('-' * 84)
    for label, passive, model in headline:
        alpha = model - passive
        print(f'{label:50s} {passive:>+10.3f} {model:>+10.3f} {alpha:>+10.3f}')

    out_path = OUTPUT_DIR / 'equal-weight-benchmark.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
