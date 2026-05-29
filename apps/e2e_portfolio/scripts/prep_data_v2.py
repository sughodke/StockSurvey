"""v2 prep: build the panel + ship IV parquet + close pickle for Modal.

Runs in the project venv. Produces:
  Output/e2e-portfolio-v2-prep.pkl
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from e2e_portfolio.data import load_close, load_macro_panel
from e2e_portfolio.data_v2 import (
    F_ASSET_V2, K_FORWARD, PHASE4D_TICKERS, T_LOOKBACK,
    prepare_panel_v2, _build_iv_panel,
)

REPO = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO / 'Output'
PREP_PATH = OUTPUT_DIR / 'e2e-portfolio-v2-prep.pkl'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=== e2e-portfolio v2 prep ===')
    t0 = time.perf_counter()

    close = load_close()
    print(f'close: {close.shape}  {close.index[0].date()} -> {close.index[-1].date()}')

    print('fetching macro ...')
    macro = load_macro_panel(close.index)

    print('building v2 panel (with IV features) ...')
    panel = prepare_panel_v2(close, macro_panel=macro)
    print(f'  X_assets: {panel.X_assets.shape}')
    print(f'  X_macro:  {panel.X_macro.shape}')
    print(f'  fwd_ret:  {panel.fwd_ret.shape}')
    print(f'  fwd_vol_pnl: {panel.fwd_vol_pnl.shape}')
    print(f'  dates: {panel.dates[0].date()} -> {panel.dates[-1].date()}  '
          f'n={len(panel.dates)}')
    print(f'  IV-covered tickers: {panel.covered_tickers}')

    # Build daily vol synth stream for eval.
    from e2e_portfolio.eval_v2 import vol_synth_daily_stream
    vol_synth_daily = vol_synth_daily_stream(close)
    print(f'  vol_synth_daily: mean={vol_synth_daily.mean():.5f}  '
          f'std={vol_synth_daily.std():.5f}')

    payload = {
        'X_assets': panel.X_assets,
        'X_macro':  panel.X_macro,
        'fwd_ret':  panel.fwd_ret,
        'fwd_vol_pnl': panel.fwd_vol_pnl,
        'dates': panel.dates.values.astype('datetime64[ns]'),
        'tickers': PHASE4D_TICKERS,
        'covered_tickers': panel.covered_tickers,
        'close': close[PHASE4D_TICKERS].values.astype(np.float64),
        'close_dates': close.index.values.astype('datetime64[ns]'),
        'vol_synth_daily': vol_synth_daily.values.astype(np.float64),
        'vol_synth_dates': vol_synth_daily.index.values.astype('datetime64[ns]'),
        'meta': {
            'T_LOOKBACK': T_LOOKBACK, 'F_ASSET': F_ASSET_V2, 'F_MACRO': 4,
            'K_FORWARD': K_FORWARD, 'N': 13,
        },
    }
    with open(PREP_PATH, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = PREP_PATH.stat().st_size / (1 << 20)
    print(f'wrote {PREP_PATH} ({size_mb:.1f} MB) in {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
