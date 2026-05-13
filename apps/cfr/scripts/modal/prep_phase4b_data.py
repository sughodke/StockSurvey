"""Local prep for Phase 4b — sector ETF universe.

Loads 9 SPDR sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV,
XLY) + SPY from StooqData. Drops XLRE (started 2015) and XLC
(started 2018) since they don't span the canonical 6-window
walkforward starting in 2005.

Pickles to `Output/cfr_phase4b_sector_close.pkl`.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
STOOQ_DATA = REPO_ROOT / 'StooqData'
STOOQ_CACHE = STOOQ_DATA / '.cache.pkl'
OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = OUTPUT_DIR / 'cfr_phase4b_sector_close.pkl'

SECTOR_ETFS = [
    'XLB',   # materials
    'XLE',   # energy
    'XLF',   # financials
    'XLI',   # industrials
    'XLK',   # technology
    'XLP',   # consumer staples
    'XLU',   # utilities
    'XLV',   # health care
    'XLY',   # consumer discretionary
    'SPY',   # S&P 500 — broad-market reference
]


def main(
    *,
    start: str = '2000-01-01',
    end: str = '2025-12-11',
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'loading {len(SECTOR_ETFS)} sector ETFs from StooqData '
          f'(cache {STOOQ_CACHE.exists()=})...')
    t0 = time.perf_counter()
    cache_arg = STOOQ_CACHE.as_posix() if STOOQ_CACHE.exists() else None
    close, _, _, _ = load_stooq_matrix(
        STOOQ_DATA.as_posix(),
        min_history=150,
        start_date=start, end_date=end,
        tickers=SECTOR_ETFS,
        include_etfs=True,
        cache_path=cache_arg,
    )
    print(f'  loaded {close.shape[0]} dates × {close.shape[1]} ETFs '
          f'in {time.perf_counter() - t0:.1f}s')
    print(f'  date range: {close.index[0].date()} → {close.index[-1].date()}')
    for col in close.columns:
        first_valid = close[col].first_valid_index()
        last_valid = close[col].last_valid_index()
        nn = int(close[col].notna().sum())
        print(f'    {col:6s}  {first_valid.date()} → {last_valid.date()}  ({nn} bars)')

    print(f'\npickling to {PICKLE_PATH} ...')
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(close, f)
    print(f'  {PICKLE_PATH.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
