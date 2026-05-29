"""Local prep step: build the e2e-portfolio panel + pickle for Modal.

This runs in the project venv (has ss_indicators, ss_macro, etc.). The
output pickle is then shipped via Modal RPC to the remote entrypoint
which only depends on numpy / tinygrad and reads the pickle as raw
bytes.

Per CLAUDE.md compute-placement: the Modal local_entrypoint runs in
the uvx ephemeral env (no project-venv deps), so all heavy data prep
must happen here.

Run:
  uv run python apps/e2e_portfolio/scripts/prep_data.py
Output:
  Output/e2e-portfolio-prep.pkl
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e2e_portfolio.data import (
    F_ASSET, F_MACRO, K_FORWARD, PHASE4D_TICKERS, T_LOOKBACK,
    load_close, load_macro_panel, prepare_panel,
)

REPO = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO / 'Output'
PREP_PATH = OUTPUT_DIR / 'e2e-portfolio-prep.pkl'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== e2e-portfolio prep ===')
    t0 = time.perf_counter()

    close = load_close()
    print(f'close: {close.shape}  '
          f'{close.index[0].date()} -> {close.index[-1].date()}')

    print('fetching macro (VIX + T10Y3M + BAA10Y) via FRED ...')
    macro = load_macro_panel(close.index)
    print(f'  macro columns: {macro.columns.tolist()}')

    print('building feature panel ...')
    panel = prepare_panel(close, macro_panel=macro)
    print(f'  X_assets: {panel.X_assets.shape}  dtype={panel.X_assets.dtype}')
    print(f'  X_macro:  {panel.X_macro.shape}  dtype={panel.X_macro.dtype}')
    print(f'  fwd_ret:  {panel.fwd_ret.shape}')
    print(f'  dates:    {panel.dates[0].date()} -> {panel.dates[-1].date()}'
          f' (n={len(panel.dates)})')

    payload = {
        'X_assets': panel.X_assets,           # (T_eff, N, T_lookback, F_asset)
        'X_macro':  panel.X_macro,            # (T_eff, T_lookback, F_macro)
        'fwd_ret':  panel.fwd_ret,            # (T_eff, N)
        'dates':    panel.dates.values.astype('datetime64[ns]'),
        'tickers':  PHASE4D_TICKERS,
        # Close + close index needed for daily-rebal mark-to-market in eval.
        'close':       close[PHASE4D_TICKERS].values.astype(np.float64),
        'close_dates': close.index.values.astype('datetime64[ns]'),
        # Constants for sanity-check on remote side.
        'meta': {
            'T_LOOKBACK': T_LOOKBACK, 'F_ASSET': F_ASSET, 'F_MACRO': F_MACRO,
            'K_FORWARD': K_FORWARD, 'N': 13,
        },
    }
    print(f'pickling {PREP_PATH} ...')
    with open(PREP_PATH, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = PREP_PATH.stat().st_size / (1 << 20)
    print(f'  wrote {size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
