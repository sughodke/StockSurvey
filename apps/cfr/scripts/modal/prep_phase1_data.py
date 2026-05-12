"""One-shot local prep: load + filter the stooq_us_long close panel
and pickle it for the Modal Phase 1 walk-forward to consume.

Run with `uv run python apps/cfr/scripts/modal/prep_phase1_data.py`
before invoking the Modal entrypoint. Writes
`Output/cfr_phase1_close.pkl` which the entrypoint reads as raw
bytes (no project-venv deps in the `uvx modal` env).

Uses `StooqData/.cache.pkl` if present to skip the 12K-file scan.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
STOOQ_DATA = REPO_ROOT / 'StooqData'
STOOQ_CACHE = STOOQ_DATA / '.cache.pkl'
OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = OUTPUT_DIR / 'cfr_phase1_close.pkl'

# Phase 1 canonical universe: 312 stooq_us_long names. Same manifest
# as gate / pairs / vol / factor sizing-input arc.
DEFAULT_MANIFEST = REPO_ROOT / 'apps/notebook/data/stooq_us_long/manifest.json'


def main(
    *,
    start: str = '2000-01-01',
    end: str = '2025-12-11',
    min_history: int = 150,
    manifest: Path = DEFAULT_MANIFEST,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe_raw = json.loads(manifest.read_text())
    tickers = sorted(t['ticker'].upper() for t in universe_raw['tickers'])
    print(f'loading StooqData/ panel (manifest n={len(tickers)}, '
          f'min_history={min_history}, cache exists: {STOOQ_CACHE.exists()}) ...')
    t0 = time.perf_counter()
    cache_arg = STOOQ_CACHE.as_posix() if STOOQ_CACHE.exists() else None
    close, _high, _low, _vol = load_stooq_matrix(
        STOOQ_DATA.as_posix(),
        min_history=min_history,
        start_date=start, end_date=end,
        tickers=tickers,
        cache_path=cache_arg,
    )
    print(f'  raw: {close.shape[0]} dates × {close.shape[1]} tickers '
          f'in {time.perf_counter() - t0:.1f}s')
    print(f'  date range: {close.index[0].date()} .. {close.index[-1].date()}')

    print(f'pickling to {PICKLE_PATH} ...')
    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(close, f)
    print(f'  {PICKLE_PATH.stat().st_size / 1024 / 1024:.1f} MB '
          f'in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
