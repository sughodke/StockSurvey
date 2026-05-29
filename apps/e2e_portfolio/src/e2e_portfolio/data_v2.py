"""v2 feature panel: v1 features + raw IV features per asset + synthetic
short-vol return stream.

Hard constraint (per user): NO pre-computed meta-layer alpha streams as
inputs. v2 uses **raw IV/HV time series only**. The synthetic short-vol
return stream is computed from raw IV/HV, not from `vol-v3-*-returns.npz`.

Per-asset features (12):
    [0..5]   v1 features (log_return_1d, rv20, rv60, RSI14, normalized_price, mom5)
    6        iv_current     (DoltHub weekly, ffilled, 0 if not covered)
    7        hv_current     (DoltHub weekly, ffilled, 0 if not covered)
    8        iv_minus_hv    (raw IVRP — same substrate vol_v3 reads)
    9        iv_pct_252d    (percentile vs trailing 252d, 0 if not covered)
    10       iv_change_60d  (iv[t] - iv[t-60], 0 if not covered)
    11       iv_available   (binary flag, 1 if DoltHub covers the ticker, else 0)

Synthetic short-vol stream: per-day equal-weighted average of
`(iv_current[t-K] - hv_current[t]) / 252` over the K-day forward window,
across all *covered* ETFs. This is the raw IVRP-per-day; vol_position
multiplies this scalar daily contribution.

Coverage on Phase-4d (verified 2026-05-28):
  COVERED  : XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY  (9 sectors)
  MISSING  : DBC, GLD, IEF, TLT                            (4 non-sector)

IV data starts 2019-02-09 — fold-1 (val 2015-2018) gets all-zero IV
features and a binary `iv_available=0` for every asset; the model can
still learn from the v1 substrate during fold-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from e2e_portfolio.data import (
    F_MACRO, K_FORWARD, PHASE4D_TICKERS, T_LOOKBACK,
    _per_asset_features, load_close, load_macro_panel,
)

REPO = Path(__file__).resolve().parents[4]
F_ASSET_V2 = 12  # 6 v1 + 5 IV + 1 flag


def _build_iv_panel(close_index: pd.DatetimeIndex,
                    parquet_path: Path | None = None,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (iv_df, hv_df, covered_tickers).

    Both frames are (T, N=13) aligned to `close_index`, with strict
    last-known-Friday + 1-day shift forward-fill so we never peek.
    Uncovered tickers get all-zero columns.
    """
    from ss_iv.loaders import DEFAULT_CACHE_DIR
    if parquet_path is None:
        parquet_path = REPO / DEFAULT_CACHE_DIR / 'volatility_history.parquet'
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['act_symbol'].isin(PHASE4D_TICKERS)].copy()
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')

    iv = df.pivot_table(index='date', columns='act_symbol',
                        values='iv_current', aggfunc='last').sort_index()
    hv = df.pivot_table(index='date', columns='act_symbol',
                        values='hv_current', aggfunc='last').sort_index()
    covered = sorted(iv.columns.tolist())

    # Strict-last-known + 1-day shift to avoid same-day peek.
    iv = iv.shift(1)  # the value labelled "today" was published end-of-day Sat
    hv = hv.shift(1)

    # Reindex to full daily close_index, ffill with a 7-day cap.
    iv = iv.reindex(close_index, method='ffill', limit=14)
    hv = hv.reindex(close_index, method='ffill', limit=14)

    # Add any missing Phase-4d ticker columns as zeros.
    for t in PHASE4D_TICKERS:
        if t not in iv.columns:
            iv[t] = np.nan
            hv[t] = np.nan
    iv = iv[PHASE4D_TICKERS]
    hv = hv[PHASE4D_TICKERS]

    # Fill NaN with 0 — model gets the iv_available flag to learn this.
    iv = iv.fillna(0.0)
    hv = hv.fillna(0.0)
    return iv, hv, covered


def _iv_features(iv_series: np.ndarray, hv_series: np.ndarray,
                 available: bool) -> np.ndarray:
    """Compute the 6 IV-side features for one asset.

    Returns (T, 6).
    """
    T = iv_series.shape[0]
    out = np.zeros((T, 6), dtype=np.float64)
    if not available:
        return out
    out[:, 0] = iv_series
    out[:, 1] = hv_series
    out[:, 2] = iv_series - hv_series
    # 252d percentile (past-only, shift 1).
    s = pd.Series(iv_series)
    out[:, 3] = (
        s.rolling(252, min_periods=60)
        .apply(lambda w: (w[-1] >= w[:-1]).mean(), raw=True)
        .fillna(0.5).values
    )
    out[:, 4] = (s - s.shift(60)).fillna(0.0).values
    out[:, 5] = 1.0  # iv_available flag
    return out


@dataclass
class PanelV2:
    X_assets: np.ndarray   # (T_eff, N, T_lookback, F_ASSET_V2)
    X_macro: np.ndarray    # (T_eff, T_lookback, F_MACRO)
    fwd_ret: np.ndarray    # (T_eff, N) K-day forward arithmetic return per asset
    fwd_vol_pnl: np.ndarray  # (T_eff,) synthetic short-vol K-day vol-points
    dates: pd.DatetimeIndex
    tickers: list[str]
    covered_tickers: list[str]

    def slice_by_date(self, start, end) -> 'PanelV2':
        s = pd.Timestamp(start) if start is not None else self.dates[0]
        e = pd.Timestamp(end) if end is not None else self.dates[-1]
        mask = (self.dates >= s) & (self.dates <= e)
        idx = np.where(mask)[0]
        return PanelV2(
            X_assets=self.X_assets[idx],
            X_macro=self.X_macro[idx],
            fwd_ret=self.fwd_ret[idx],
            fwd_vol_pnl=self.fwd_vol_pnl[idx],
            dates=self.dates[idx],
            tickers=self.tickers,
            covered_tickers=self.covered_tickers,
        )


def _zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def prepare_panel_v2(
    close: pd.DataFrame | None = None,
    *,
    t_lookback: int = T_LOOKBACK,
    k_forward: int = K_FORWARD,
    macro_panel: pd.DataFrame | None = None,
    parquet_path: Path | None = None,
) -> PanelV2:
    if close is None:
        close = load_close()
    close = close[PHASE4D_TICKERS]
    close_arr = close.values
    T, N = close_arr.shape
    assert N == 13

    iv_df, hv_df, covered = _build_iv_panel(close.index, parquet_path=parquet_path)
    iv_arr = iv_df.values  # (T, N)
    hv_arr = hv_df.values

    # Per-asset features: stack v1 (6) + IV (6).
    feat_blocks = []
    for j in range(N):
        v1_feats = _per_asset_features(close_arr[:, j])  # (T, 6)
        is_covered = PHASE4D_TICKERS[j] in covered
        iv_feats = _iv_features(iv_arr[:, j], hv_arr[:, j], is_covered)  # (T, 6)
        feat_blocks.append(np.concatenate([v1_feats, iv_feats], axis=1))
    feats = np.stack(feat_blocks, axis=1)  # (T, N, 12)

    if macro_panel is None:
        macro_panel = load_macro_panel(close.index)
    macro_arr = macro_panel.values.astype(np.float64)

    start = t_lookback - 1
    end = T - k_forward - 1
    n_eff = max(0, end - start)
    if n_eff <= 0:
        raise ValueError(f'no samples: T={T}')

    asset_mean = feats.reshape(-1, F_ASSET_V2).mean(axis=0)
    asset_std = feats.reshape(-1, F_ASSET_V2).std(axis=0)
    macro_mean = macro_arr.mean(axis=0)
    macro_std = macro_arr.std(axis=0)

    feats_n = _zscore(feats, asset_mean[None, None, :], asset_std[None, None, :])
    macro_n = _zscore(macro_arr, macro_mean[None, :], macro_std[None, :])

    # Pre-compute per-day per-asset realized HV using a forward-looking
    # window of (hv at t+K) — for synthetic short-vol pnl.
    # Short-straddle pnl in vol-points = iv[t] - realized_vol_over_[t, t+K].
    # Use hv_current at t+K (a slow-moving 30d HV) as proxy.
    fwd_vol_pnl_per_day = np.zeros(T, dtype=np.float64)
    avail_mask = np.array([1.0 if t_ in covered else 0.0
                           for t_ in PHASE4D_TICKERS])
    iv_t = iv_arr  # (T, N)
    # forward HV: shift back -k_forward.
    hv_fwd = np.concatenate([hv_arr[k_forward:],
                             np.repeat(hv_arr[-1:], k_forward, axis=0)], axis=0)
    vol_points_per_asset = (iv_t - hv_fwd) * avail_mask[None, :]
    denom = max(int(avail_mask.sum()), 1)
    # per-rebal vol-points captured by EW basket of covered assets.
    fwd_vol_pnl_per_day = vol_points_per_asset.sum(axis=1) / denom
    # 10 bps friction per K-day period.
    fwd_vol_pnl_per_day = fwd_vol_pnl_per_day - 10e-4

    X_assets = np.empty((n_eff, N, t_lookback, F_ASSET_V2), dtype=np.float32)
    X_macro = np.empty((n_eff, t_lookback, F_MACRO), dtype=np.float32)
    fwd_ret = np.empty((n_eff, N), dtype=np.float32)
    fwd_vol_pnl = np.empty((n_eff,), dtype=np.float32)
    dates_out = []
    for k, t in enumerate(range(start, start + n_eff)):
        X_assets[k] = feats_n[t - t_lookback + 1:t + 1].transpose(1, 0, 2)
        X_macro[k] = macro_n[t - t_lookback + 1:t + 1]
        entry = close_arr[t + 1]
        exit_ = close_arr[min(t + 1 + k_forward, T - 1)]
        fwd_ret[k] = (exit_ / np.maximum(entry, 1e-9) - 1.0).astype(np.float32)
        # vol-points for K-day window starting at t+1, converted to per-K-day return.
        # IV is annualized — divide by sqrt(252/K) ... actually we treat
        # vol-points * (K/252) as approximate return contribution.
        fwd_vol_pnl[k] = float(fwd_vol_pnl_per_day[t + 1] * (k_forward / 252.0))
        dates_out.append(close.index[t])

    return PanelV2(
        X_assets=X_assets,
        X_macro=X_macro,
        fwd_ret=fwd_ret,
        fwd_vol_pnl=fwd_vol_pnl,
        dates=pd.DatetimeIndex(dates_out),
        tickers=PHASE4D_TICKERS,
        covered_tickers=covered,
    )
