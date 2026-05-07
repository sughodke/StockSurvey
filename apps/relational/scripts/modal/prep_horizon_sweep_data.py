"""One-shot local prep for the relational horizon sweep on Modal.

!!! UNVALIDATED — paired with horizon_sweep.py which never produced a
successful end-to-end run !!! See the banner in horizon_sweep.py for
context. This prep step is the simpler half (just a panel load + pickle)
so it's likely fine, but treat as untested until paired with a working
remote invocation.

Loads the 312-ticker `apps/notebook/data/stooq_us_long/` panel via
`load_stooq_matrix`, then pickles the close DataFrame to
`Output/relational_horizon_sweep_close.pkl`. The Modal entrypoint
(`horizon_sweep.py`) reads this as raw bytes and ships via RPC, mirroring
the pattern from `apps/factor/scripts/modal/{prep_universe_pivot_data,
universe_pivot_vol_arm}.py`.

Run:
    uv run python apps/relational/scripts/modal/prep_horizon_sweep_data.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
STOOQ_SUBSET = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long'
OUTPUT_DIR = REPO_ROOT / 'Output'
PICKLE_PATH = OUTPUT_DIR / 'relational_horizon_sweep_close.pkl'


def main(
    *,
    lookback: int = 120,
    n_tail: int = 20,
    train_window_days: int = 252,
    w_delta: int = 21,
    start: str | None = None,
    end: str | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Match the loader's `min_history` heuristic from both relational
    # diagnostics (transition-triggered uses lookback+n_tail+10; velocity
    # uses lookback + max(n_tail, w_delta) + train_window_days + 10).
    # Use the tighter of the two so both constructions can coexist on
    # the same panel without re-loading.
    min_history = lookback + max(n_tail, w_delta) + train_window_days + 10

    print(f'loading Stooq panel from {STOOQ_SUBSET} '
          f'(min_history={min_history}) ...')
    t0 = time.perf_counter()
    close, _high, _low, _vol = load_stooq_matrix(
        STOOQ_SUBSET.as_posix(),
        min_history=min_history,
        start_date=start, end_date=end,
        tickers=None,
    )
    print(f'  raw: {close.shape[0]} dates x {close.shape[1]} tickers '
          f'in {time.perf_counter() - t0:.1f}s')
    print(f'  date range: {close.index[0].date()} .. {close.index[-1].date()}')

    print(f'pickling close DataFrame to {PICKLE_PATH} ...')
    t0 = time.perf_counter()
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(close, f)
    size_mb = PICKLE_PATH.stat().st_size / 1024 / 1024
    print(f'  {size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
