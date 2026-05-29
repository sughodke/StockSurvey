"""v3 prep: build the unified-universe panel + ship to Modal.

Runs in the project venv. Produces Output/e2e-portfolio-v3-prep.pkl.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from e2e_portfolio.data_v3 import (
    F_ASSET_V3, F_MACRO, K_FORWARD, T_LOOKBACK, DEFAULT_K,
    prepare_panel_v3,
)

REPO = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO / 'Output'
PREP_PATH = OUTPUT_DIR / 'e2e-portfolio-v3-prep.pkl'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== e2e-portfolio v3 prep ===', flush=True)
    t0 = time.perf_counter()

    k = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_K
    panel = prepare_panel_v3(k=k)
    print(f'  X_assets: {panel.X_assets.shape}', flush=True)
    print(f'  X_macro:  {panel.X_macro.shape}', flush=True)
    print(f'  fwd_ret:  {panel.fwd_ret.shape}', flush=True)
    print(f'  fwd_vol_pnl: {panel.fwd_vol_pnl.shape}', flush=True)
    print(f'  dates: {panel.dates[0].date()} -> {panel.dates[-1].date()}  '
          f'n={len(panel.dates)}', flush=True)
    print(f'  cohort size: {len(panel.tickers)}', flush=True)

    # Build a close DataFrame for daily marking + DCA baseline.
    from e2e_portfolio.data_v3 import load_price_panel
    prices = load_price_panel(panel.tickers,
                               start='2014-01-01', end='2026-12-31')
    prices = prices[panel.tickers]
    print(f'  close panel: {prices.shape}', flush=True)

    # X_assets is z-scored features; float16 cuts wire size in half with
    # negligible signal loss (z-scored values are O(1)).
    payload = {
        'X_assets': panel.X_assets.astype(np.float16),
        'X_macro':  panel.X_macro,
        'valid_mask': panel.valid_mask,
        'fwd_ret':  panel.fwd_ret,
        'fwd_vol_pnl': panel.fwd_vol_pnl,
        'dates': panel.dates.values.astype('datetime64[ns]'),
        'tickers': panel.tickers,
        'close': prices.values.astype(np.float64),
        'close_dates': prices.index.values.astype('datetime64[ns]'),
        'meta': {
            'T_LOOKBACK': T_LOOKBACK, 'F_ASSET': F_ASSET_V3, 'F_MACRO': F_MACRO,
            'K_FORWARD': K_FORWARD, 'K': len(panel.tickers),
        },
    }
    with open(PREP_PATH, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = PREP_PATH.stat().st_size / (1 << 20)
    print(f'wrote {PREP_PATH} ({size_mb:.1f} MB) in {time.perf_counter()-t0:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
