"""One-shot local prep for the regime baseline-vs-augmented A/B on Modal.

Loads the StooqData/ panel filtered to the relational scoreboard's
date range (2013-01-29 → 2025-12-11) and pickles `(close, highs, lows,
volumes)` as a single bundle to `Output/regime_baseline_vs_aug.pkl`.
The Modal entrypoint reads this as raw bytes and ships via RPC,
mirroring the pattern from `apps/factor/scripts/modal/{prep_universe_
pivot_data, universe_pivot_vol_arm}.py`.

Run once locally before invoking the Modal job:
    uv run python apps/regime/scripts/modal/prep_regime_data.py
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
PICKLE_PATH = OUTPUT_DIR / 'regime_baseline_vs_aug.pkl'

# Match the relational scoreboard's date range.
START = '2013-01-29'
END = '2025-12-11'
# Filter at panel-load time to keep the pickle reasonable for RPC.
# 1260 bars ≈ 5 years — matches the trainer's default `train_years=5`,
# so every surviving ticker has enough data to participate in at
# least one walk-forward window. The per-walk-forward survivorship
# floor (630 bars) inside `regime.trainer` does the strict per-window
# filtering downstream.
MIN_HISTORY = 1260


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'loading StooqData/ panel '
          f'(min_history={MIN_HISTORY}, '
          f'cache exists: {STOOQ_CACHE.exists()}) ...')
    t0 = time.perf_counter()
    cache_arg = STOOQ_CACHE.as_posix() if STOOQ_CACHE.exists() else None
    close, highs, lows, volumes = load_stooq_matrix(
        STOOQ_DATA.as_posix(),
        min_history=MIN_HISTORY,
        start_date=START, end_date=END,
        cache_path=cache_arg,
    )
    print(f'  loaded: {close.shape[0]} dates × {close.shape[1]} tickers '
          f'in {time.perf_counter() - t0:.1f}s')
    print(f'  date range: {close.index[0].date()} .. {close.index[-1].date()}')
    print(f'  volume coverage: '
          f'{(volumes.notna() & (volumes > 0)).mean().mean() * 100:.1f}% '
          f'of (date, ticker) cells have positive volume')

    bundle = {
        'close': close,
        'highs': highs,
        'lows': lows,
        'volumes': volumes,
    }

    print(f'pickling bundle to {PICKLE_PATH} ...')
    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    size_mb = PICKLE_PATH.stat().st_size / 1024 / 1024
    print(f'  {size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
