"""Forward IV/RV gap target — `iv_t − rv_{t,t+H}`.

Positive gap = realized came in *below* implied → short-vol won
that cycle (the standard short-vol PnL convention used in
`ss_iv.short_vol_pnl_panel`). The predictor learns to anticipate
when gap will be large-positive (good short-vol setup) vs negative
(realized exceeded implied → long-vol setup).

Implementation is per-`(date, symbol)` cell. Caller drops trailing
rows (no full forward window) and aligns with the feature panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_iv_rv_gap(
    raw_panel: pd.DataFrame, *,
    horizon: int = 20,
    rv_col: str = 'hv_20',
    iv_col: str = 'ATM_IV',
) -> pd.DataFrame:
    """Per-(date, symbol) target = `iv_col − rv_col_at_t+horizon`.

    `rv_col_at_t+horizon` is the trailing-realized vol *measured at*
    `t+horizon`, which approximates the realized vol over the
    `(t, t+horizon]` window. Using the panel's own `hv_20` column at
    `t+horizon` is a clean reuse — the gauss314 schema already
    computes trailing 20-day HV per row, so we just shift it forward.

    Returns a long-form DataFrame with columns `[date, symbol,
    iv_rv_gap]`. NaN where the forward window doesn't exist or where
    a required field is missing on either side.
    """
    if rv_col not in raw_panel.columns:
        raise ValueError(f'rv_col={rv_col!r} not in panel')
    if iv_col not in raw_panel.columns:
        raise ValueError(f'iv_col={iv_col!r} not in panel')

    # Per-symbol forward-shift of the trailing-RV column gives the
    # realized that will print at horizon-end.
    df = raw_panel[['date', 'symbol', iv_col, rv_col]].copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    # Per-symbol forward shift: groupby + shift(-horizon).
    df['rv_forward'] = df.groupby('symbol', sort=False)[rv_col].shift(-horizon)
    df['iv_rv_gap'] = df[iv_col] - df['rv_forward']

    return df[['date', 'symbol', 'iv_rv_gap']]


__all__ = ['forward_iv_rv_gap']
