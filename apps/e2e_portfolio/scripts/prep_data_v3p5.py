"""v3.5 prep: extend v3 prep with forward VIXY returns + VIXY daily series.

Reads the existing Output/e2e-portfolio-v3-prep.pkl (1.08 GB on ss-e2e-iv-data
Volume) if present, attaches `fwd_long_vol_ret` array + `vixy_close` Series,
saves a v3.5 pickle. Falls back to building from scratch if v3 prep absent.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from e2e_portfolio.data_v3p5 import (
    DEFAULT_K, prepare_panel_v3p5, load_vixy_close, build_fwd_long_vol_ret,
)
from e2e_portfolio.data_v3 import K_FORWARD

REPO = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO / 'Output'
PREP_V3_PATH = OUTPUT_DIR / 'e2e-portfolio-v3-prep.pkl'
PREP_V3P5_PATH = OUTPUT_DIR / 'e2e-portfolio-v3p5-prep.pkl'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== e2e-portfolio v3.5 prep ===', flush=True)
    t0 = time.perf_counter()

    k = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_K

    if PREP_V3_PATH.exists():
        print(f'Found v3 prep at {PREP_V3_PATH} — attaching VIXY only', flush=True)
        with open(PREP_V3_PATH, 'rb') as f:
            payload = pickle.load(f)
        dates = pd.DatetimeIndex(payload['dates'])
        vixy = load_vixy_close()
        fwd_lv = build_fwd_long_vol_ret(dates, vixy, K_FORWARD)
        # Also persist VIXY series itself so the eval can compute daily ret.
        vixy_close_for_pkl = vixy.copy()
        payload['fwd_long_vol_ret'] = fwd_lv
        payload['vixy_close'] = vixy_close_for_pkl
        print(f'  fwd_long_vol_ret: shape={fwd_lv.shape} mean={fwd_lv.mean():.5f} '
              f'nonzero_frac={(fwd_lv != 0).mean():.3f}', flush=True)
        print(f'  vixy_close coverage: {vixy.index[0].date()} -> '
              f'{vixy.index[-1].date()} n={len(vixy)}', flush=True)
    else:
        print(f'No v3 prep, building from scratch...', flush=True)
        panel = prepare_panel_v3p5(k=k)
        print(f'  X_assets: {panel.X_assets.shape}', flush=True)
        print(f'  fwd_long_vol_ret: shape={panel.fwd_long_vol_ret.shape} '
              f'mean={panel.fwd_long_vol_ret.mean():.5f}', flush=True)
        from e2e_portfolio.data_v3 import load_price_panel, select_cohort
        from e2e_portfolio.data_v3 import build_iv_hv_panels
        close = load_price_panel(panel.tickers)
        close_arr = close.values.astype(np.float32)
        close_dates_int = close.index.astype('datetime64[ns]').astype(np.int64)
        payload = {
            'X_assets': panel.X_assets,
            'X_macro': panel.X_macro,
            'valid_mask': panel.valid_mask,
            'fwd_ret': panel.fwd_ret,
            'fwd_vol_pnl': panel.fwd_vol_pnl,
            'fwd_long_vol_ret': panel.fwd_long_vol_ret,
            'dates': panel.dates.values,
            'tickers': panel.tickers,
            'close': close_arr,
            'close_dates': close_dates_int,
            'vixy_close': load_vixy_close(),
        }

    with open(PREP_V3P5_PATH, 'wb') as f:
        pickle.dump(payload, f)
    size_mb = PREP_V3P5_PATH.stat().st_size // (1 << 20)
    print(f'wrote {PREP_V3P5_PATH} ({size_mb} MB) in {time.perf_counter() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
