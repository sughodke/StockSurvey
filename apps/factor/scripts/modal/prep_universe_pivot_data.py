"""One-shot local prep: filter the StooqData/ panel and pickle it.

Run this with `uv run python apps/factor/scripts/modal/prep_universe_pivot_data.py`
*before* invoking the Modal vol arm. Writes `Output/universe_pivot_close.pkl`
which the Modal entrypoint reads as raw bytes (no project-venv deps in
the `uvx modal` local entrypoint).
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
STOOQ_DATA = REPO_ROOT / 'StooqData'
STOOQ_CACHE = STOOQ_DATA / '.cache.pkl'
OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = OUTPUT_DIR / 'universe_pivot_close.pkl'


def main(
    *,
    start: str = '2000-01-01',
    end: str = '2026-04-01',
    min_history: int = 3500,
    start_grace_days: int = 3650,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'loading StooqData/ panel (min_history={min_history}, '
          f'cache exists: {STOOQ_CACHE.exists()}) ...')
    t0 = time.perf_counter()
    cache_arg = STOOQ_CACHE.as_posix() if STOOQ_CACHE.exists() else None
    close, _high, _low, _vol = load_stooq_matrix(
        STOOQ_DATA.as_posix(),
        min_history=min_history,
        start_date=start, end_date=end,
        cache_path=cache_arg,
    )
    print(f'  raw: {close.shape[0]} dates × {close.shape[1]} tickers '
          f'in {time.perf_counter() - t0:.1f}s')

    target = pd.Timestamp(start) + pd.Timedelta(days=start_grace_days)
    first_dates = close.apply(lambda s: s.first_valid_index())
    keep = first_dates[first_dates <= target].index.tolist()
    close = close[keep]
    print(f'  filtered (first_valid ≤ {target.date()}): '
          f'{close.shape[0]} dates × {close.shape[1]} tickers')

    print(f'  date range: {close.index[0].date()} .. {close.index[-1].date()}')

    print(f'pickling to {PICKLE_PATH} ...')
    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(close, f)
    print(f'  {PICKLE_PATH.stat().st_size / 1024 / 1024:.1f} MB '
          f'in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
