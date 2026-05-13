"""Local prep for Phase 3 — fetch macro panel from FRED + pickle for Modal.

Run after `prep_phase1_data.py` and `prep_phase2b_data.py`. The macro
fetch is no-auth (FRED `fredgraph.csv` endpoint) and cached at
`.macro-cache/`; first run is ~30s for the 4 features used by Phase
3, subsequent runs are instant.

Writes:
- `Output/cfr_phase3_macro.pkl` — DataFrame indexed by date,
  columns: vix, credit_baa, m2_yoy, real_yield_10y. ffill applied
  to the price-panel index in `state_vec.StateVecBuilder`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from ss_macro import load_macro_panel


REPO_ROOT = Path(__file__).resolve().parents[4]
MACRO_CACHE = REPO_ROOT / '.macro-cache'
OUTPUT_DIR = REPO_ROOT / 'Output'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close_path = OUTPUT_DIR / 'cfr_phase1_close.pkl'
    if not close_path.exists():
        raise SystemExit(
            f'price pickle not found at {close_path}. Run prep first:\n'
            f'  uv run python apps/cfr/scripts/modal/prep_phase1_data.py')

    print(f'loading close panel from {close_path} ...')
    with open(close_path, 'rb') as f:
        close = pickle.load(f)
    print(f'  shape {close.shape}, '
          f'{close.index[0].date()} → {close.index[-1].date()}')

    print(f'\nfetching macro panel (FRED, cache at {MACRO_CACHE})...')
    t0 = time.time()
    macro = load_macro_panel(target_index=close.index)
    print(f'  shape {macro.shape}, columns {list(macro.columns)} '
          f'in {time.time() - t0:.1f}s')

    # Phase 3 uses 4 features per the macro-regime-diagnostic findings.
    # Drop fed_funds (collinear with VIX in our window sample) and
    # slope_10y_3m (noise feature). m2_level → m2_yoy via 252d % change.
    keep = ['vix', 'credit_baa', 'real_yield_10y']
    if 'gold_vix' in macro.columns:
        # Optional: use gold_vix if available (CBOE GVZCLS, daily 2008+).
        # For now we don't include it — it's only available 2008+ which
        # would mask the early train years.
        pass
    if 'm2_level' in macro.columns:
        # Compute m2_yoy from m2_level (12-month % change).
        m2_yoy = (macro['m2_level'].pct_change(periods=252) * 100.0).rename('m2_yoy')
        macro_out = pd.concat([macro[keep], m2_yoy], axis=1)
    else:
        macro_out = macro[keep].copy()

    print(f'\nfinal macro_out: {macro_out.shape}, columns {list(macro_out.columns)}')
    print(f'  date range: {macro_out.index[0].date()} → '
          f'{macro_out.index[-1].date()}')
    # Per-feature non-null count
    for col in macro_out.columns:
        nn = int(macro_out[col].notna().sum())
        print(f'  {col}: {nn}/{len(macro_out)} non-null')

    out_path = OUTPUT_DIR / 'cfr_phase3_macro.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(macro_out, f)
    print(f'\nwrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
