"""v3.5 data — extends v3's PanelV3 with a forward VIXY return per anchor.

Wraps `data_v3.prepare_panel_v3` and adds one additional array:
  fwd_long_vol_ret: (T_eff,) float32, the K_FORWARD-day compounded
  VIXY arithmetic return at each anchor.

VIXY substrate from Stooq archive `daily/us/nyse etfs/2/vixy.us.txt`.
Coverage: 2011-01-07 → present (full coverage of all v1/v2/v3 folds).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from e2e_portfolio.data_v3 import (
    F_ASSET_V3, F_MACRO, K_FORWARD, T_LOOKBACK, DEFAULT_K, DEFAULT_K_ACTIVE,
    PanelV3, prepare_panel_v3,
)


REPO = Path(__file__).resolve().parents[4]
VIXY_PATH = REPO / 'StooqData' / 'daily' / 'us' / 'nyse etfs' / '2' / 'vixy.us.txt'


@dataclass
class PanelV3p5:
    X_assets: np.ndarray
    X_macro: np.ndarray
    valid_mask: np.ndarray
    fwd_ret: np.ndarray
    fwd_vol_pnl: np.ndarray
    fwd_long_vol_ret: np.ndarray  # NEW: (T_eff,)
    dates: pd.DatetimeIndex
    tickers: list[str]

    def slice_by_date(self, start, end) -> 'PanelV3p5':
        s = pd.Timestamp(start) if start is not None else self.dates[0]
        e = pd.Timestamp(end) if end is not None else self.dates[-1]
        mask = (self.dates >= s) & (self.dates <= e)
        idx = np.where(mask)[0]
        return PanelV3p5(
            X_assets=self.X_assets[idx],
            X_macro=self.X_macro[idx],
            valid_mask=self.valid_mask[idx],
            fwd_ret=self.fwd_ret[idx],
            fwd_vol_pnl=self.fwd_vol_pnl[idx],
            fwd_long_vol_ret=self.fwd_long_vol_ret[idx],
            dates=self.dates[idx],
            tickers=self.tickers,
        )


def load_vixy_close(vixy_path: str | Path | None = None) -> pd.Series:
    """Load Stooq VIXY daily close. Returns pd.Series indexed by date."""
    path = Path(vixy_path) if vixy_path is not None else VIXY_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip('<>').lower() for c in df.columns]
    dates = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
    close = df['close'].astype(np.float64)
    return pd.Series(close.values, index=dates, name='vixy_close')


def build_fwd_long_vol_ret(dates: pd.DatetimeIndex,
                            vixy_close: pd.Series,
                            k_forward: int = K_FORWARD) -> np.ndarray:
    """For each anchor in `dates`, compute the k_forward-day arithmetic
    return on VIXY: vixy[t+k] / vixy[t] - 1.

    Returns zero for anchors outside VIXY coverage (pre-2011 fold-1 days).
    """
    # Align VIXY series to a contiguous business-day index union of dates.
    full_idx = vixy_close.index.union(dates).sort_values()
    vixy_full = vixy_close.reindex(full_idx).ffill()
    # Map each date to a position in full_idx.
    out = np.zeros(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        pos = full_idx.searchsorted(d)
        if pos >= len(full_idx):
            continue
        end_pos = pos + k_forward
        if end_pos >= len(full_idx):
            continue
        start_v = vixy_full.iloc[pos]
        end_v = vixy_full.iloc[end_pos]
        if np.isnan(start_v) or np.isnan(end_v) or start_v <= 0:
            continue
        out[i] = (end_v / start_v) - 1.0
    return out.astype(np.float32)


def prepare_panel_v3p5(k: int = DEFAULT_K,
                       vixy_path: str | Path | None = None) -> PanelV3p5:
    """Build v3 panel, then attach forward VIXY returns."""
    p3 = prepare_panel_v3(k=k)
    vixy = load_vixy_close(vixy_path)
    fwd_lv = build_fwd_long_vol_ret(p3.dates, vixy, K_FORWARD)
    return PanelV3p5(
        X_assets=p3.X_assets,
        X_macro=p3.X_macro,
        valid_mask=p3.valid_mask,
        fwd_ret=p3.fwd_ret,
        fwd_vol_pnl=p3.fwd_vol_pnl,
        fwd_long_vol_ret=fwd_lv,
        dates=p3.dates,
        tickers=p3.tickers,
    )


__all__ = [
    'PanelV3p5', 'prepare_panel_v3p5', 'load_vixy_close',
    'build_fwd_long_vol_ret', 'VIXY_PATH',
]
