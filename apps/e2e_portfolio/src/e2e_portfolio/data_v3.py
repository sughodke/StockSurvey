"""v3 unified-universe feature builder.

ONE UNIVERSE: K=200 candidates drawn from the DoltHub IV parquet that
have continuous IV+price coverage across the train+val span. Each name
carries BOTH an equity-weight decision and a vol-weight decision.

Per-name features (F_asset=11):
   0: log_ret_1d
   1: rv_20d (std of 1d log returns)
   2: rv_60d
   3: RSI(14), normalized to ~[-1,1] by (rsi-50)/50
   4: close / SMA252 - 1.0
   5: 5d momentum
   6: iv_current   (DoltHub weekly, ffilled, shifted 1 day)
   7: hv_current   (DoltHub weekly, ffilled, shifted 1 day)
   8: iv_minus_hv  (raw IVRP)
   9: iv_pct_252d  (past-only percentile)
  10: iv_change_60d

The cohort selection happens at fold-build time: per fold the top-K
names by coverage density over the train+val window are picked. Fold-1
(pre-2019) has no IV coverage; the cohort there falls back to top-K by
price coverage and the IV side of features is zero. The model learns
to suppress the vol head (via vol_scale ≈ 0) when IV is unavailable.

Macro side-channel (F_macro=4): vix, vix_pct_252, slope_10y_3m,
credit_baa. Same as v1/v2.

Per-name forward short-vol PnL: vol-points (iv[t] - hv_forward[t+K])
* (K/252), with a 10 bps per-rebal friction. The model's vol_weights
multiply this per-name PnL, then vol_scale multiplies the sum.

This file produces a `PanelV3` keyed on per-fold cohort. Prep selects
one master cohort large enough to cover all three folds (union, sized
K=200 of the most-covered names across 2014-01-01 -> 2026-12-31).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ss_indicators import rsi
from ss_iv.loaders import DEFAULT_CACHE_DIR

REPO = Path(__file__).resolve().parents[4]
T_LOOKBACK = 60
K_FORWARD = 20
F_ASSET_V3 = 11
F_MACRO = 4
DEFAULT_K = 200
DEFAULT_K_ACTIVE = 50


def _per_name_price_features(close: np.ndarray) -> np.ndarray:
    """(T,) close -> (T, 6) price features."""
    T = close.shape[0]
    out = np.zeros((T, 6), dtype=np.float64)
    log_close = np.log(np.maximum(close, 1e-12))
    log_ret = np.diff(log_close, prepend=log_close[0])
    out[:, 0] = log_ret
    s = pd.Series(log_ret)
    out[:, 1] = s.rolling(20, min_periods=5).std().bfill().values
    out[:, 2] = s.rolling(60, min_periods=10).std().bfill().values
    rsi_v = rsi(close.reshape(-1, 1), n=14).reshape(-1)
    rsi_v = np.nan_to_num(rsi_v, nan=50.0)
    out[:, 3] = (rsi_v - 50.0) / 50.0
    sma252 = pd.Series(close).rolling(252, min_periods=60).mean().bfill().values
    out[:, 4] = close / np.maximum(sma252, 1e-9) - 1.0
    out[:, 5] = s.rolling(5, min_periods=1).sum().fillna(0.0).values
    return out


def _per_name_iv_features(iv: np.ndarray, hv: np.ndarray,
                          available: np.ndarray) -> np.ndarray:
    """(T,) iv, hv, (T,) availability flag -> (T, 5) IV features.

    Returns iv, hv, iv-hv, iv_pct_252, iv_change_60. Each is masked to 0
    where availability=0.
    """
    T = iv.shape[0]
    out = np.zeros((T, 5), dtype=np.float64)
    out[:, 0] = iv
    out[:, 1] = hv
    out[:, 2] = iv - hv
    s = pd.Series(iv)
    out[:, 3] = (
        s.rolling(252, min_periods=60)
         .apply(lambda w: (w[-1] >= w[:-1]).mean(), raw=True)
         .fillna(0.5).values
    )
    out[:, 4] = (s - s.shift(60)).fillna(0.0).values
    out = out * available[:, None]
    return out


@dataclass
class PanelV3:
    X_assets: np.ndarray         # (T_eff, K, T_lookback, F_ASSET_V3) float32
    X_macro: np.ndarray          # (T_eff, T_lookback, F_MACRO) float32
    valid_mask: np.ndarray       # (T_eff, K) float32, 1 if IV-covered at anchor
    fwd_ret: np.ndarray          # (T_eff, K) float32, K-day forward arithmetic return per name
    fwd_vol_pnl: np.ndarray      # (T_eff, K) float32, per-name short-vol pnl
    dates: pd.DatetimeIndex      # (T_eff,)
    tickers: list[str]           # K names

    def slice_by_date(self, start, end) -> 'PanelV3':
        s = pd.Timestamp(start) if start is not None else self.dates[0]
        e = pd.Timestamp(end) if end is not None else self.dates[-1]
        mask = (self.dates >= s) & (self.dates <= e)
        idx = np.where(mask)[0]
        return PanelV3(
            X_assets=self.X_assets[idx],
            X_macro=self.X_macro[idx],
            valid_mask=self.valid_mask[idx],
            fwd_ret=self.fwd_ret[idx],
            fwd_vol_pnl=self.fwd_vol_pnl[idx],
            dates=self.dates[idx],
            tickers=self.tickers,
        )


def select_cohort(
    iv_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    k: int = DEFAULT_K,
) -> list[str]:
    """Pick K names with strongest joint IV + price coverage on the
    intersection of the two panels' date ranges.
    """
    common_dates = iv_panel.index.intersection(price_panel.index)
    if len(common_dates) < 100:
        # IV panel may not overlap (pre-2019). Fall back to price coverage.
        common_dates = price_panel.index
        # Use just price coverage scores.
        coverage = price_panel.notna().sum(axis=0)
        return list(coverage.sort_values(ascending=False).head(k).index)
    iv_cov = iv_panel.loc[common_dates].notna().sum(axis=0)
    px_cov = price_panel.loc[common_dates].notna().sum(axis=0)
    joint = set(iv_cov.index) & set(px_cov.index)
    score = pd.Series({t: float(iv_cov.get(t, 0)) + 0.5 * float(px_cov.get(t, 0))
                       for t in joint})
    return list(score.sort_values(ascending=False).head(k).index)


def load_price_panel(
    tickers: list[str],
    start: str = '2010-01-01',
    end: str = '2026-12-31',
    stooq_dir: Path | None = None,
) -> pd.DataFrame:
    """Load Stooq close panel for the given tickers."""
    from ss_loaders import load_stooq_matrix
    if stooq_dir is None:
        stooq_dir = REPO / 'StooqData'
    prices, _, _, _ = load_stooq_matrix(
        str(stooq_dir), min_history=100,
        start_date=start, end_date=end, tickers=tickers)
    return prices


def build_iv_hv_panels(
    parquet_path: Path | None = None,
    tickers: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if parquet_path is None:
        parquet_path = REPO / DEFAULT_CACHE_DIR / 'volatility_history.parquet'
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    if tickers is not None:
        df = df[df['act_symbol'].isin(tickers)]
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')
    iv = (df.pivot_table(index='date', columns='act_symbol',
                         values='iv_current', aggfunc='last').sort_index())
    hv = (df.pivot_table(index='date', columns='act_symbol',
                         values='hv_current', aggfunc='last').sort_index())
    return iv, hv


def _load_macro(close_index: pd.DatetimeIndex) -> pd.DataFrame:
    from ss_macro.loaders import load_fred_series
    vix = load_fred_series('VIXCLS')
    slope = load_fred_series('T10Y3M')
    credit = load_fred_series('BAA10Y')
    out = pd.DataFrame(index=close_index)
    out['vix'] = vix.reindex(close_index, method='ffill')
    out['slope_10y_3m'] = slope.reindex(close_index, method='ffill')
    out['credit_baa'] = credit.reindex(close_index, method='ffill')
    vix_aligned = out['vix']
    out['vix_pct_252'] = (
        vix_aligned.rolling(252, min_periods=60)
        .apply(lambda w: (w[-1] >= w[:-1]).mean(), raw=True)
    )
    out = out.ffill().bfill()
    return out[['vix', 'vix_pct_252', 'slope_10y_3m', 'credit_baa']]


def _zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def prepare_panel_v3(
    cohort: list[str] | None = None,
    *,
    k: int = DEFAULT_K,
    t_lookback: int = T_LOOKBACK,
    k_forward: int = K_FORWARD,
    parquet_path: Path | None = None,
    stooq_dir: Path | None = None,
    start: str = '2010-01-01',
    end: str = '2026-12-31',
) -> PanelV3:
    """Build the v3 unified-universe panel.

    cohort: list of K names. If None, auto-select from IV + price coverage.
    """
    print(f'[v3-prep] loading IV panel...', flush=True)
    iv_full, hv_full = build_iv_hv_panels(parquet_path=parquet_path)
    print(f'  IV panel: {iv_full.shape}  '
          f'{iv_full.index[0].date()}..{iv_full.index[-1].date()}', flush=True)

    if cohort is None:
        print(f'[v3-prep] selecting cohort of K={k}...', flush=True)
        # First pass: load Stooq prices for the full IV universe to score.
        # That's 2,276 tickers -> too heavy. Instead, score on IV coverage
        # alone, then load prices for top 2*K candidates and re-rank.
        iv_cov = iv_full.notna().sum(axis=0)
        candidates = list(iv_cov.sort_values(ascending=False).head(2 * k).index)
        print(f'  loading prices for top {len(candidates)} IV-coverage names...',
              flush=True)
        try:
            prices_cand = load_price_panel(candidates, start=start, end=end,
                                           stooq_dir=stooq_dir)
        except Exception as e:
            print(f'  price load fallback: {e}', flush=True)
            prices_cand = pd.DataFrame()
        # Score on joint IV + price coverage over the full IV span.
        if not prices_cand.empty:
            common = iv_full.index.intersection(prices_cand.index)
            iv_cov_c = iv_full.loc[common, [c for c in candidates if c in iv_full.columns]].notna().sum(axis=0)
            px_cov_c = prices_cand.loc[common].notna().sum(axis=0)
            joint = set(iv_cov_c.index) & set(px_cov_c.index)
            score = pd.Series({t: float(iv_cov_c.get(t, 0)) + 0.5 * float(px_cov_c.get(t, 0))
                               for t in joint})
            cohort = list(score.sort_values(ascending=False).head(k).index)
        else:
            cohort = candidates[:k]
        print(f'  cohort[:10] = {cohort[:10]}', flush=True)

    # Load price panel for the chosen cohort over the full date span.
    print(f'[v3-prep] loading prices for cohort (K={len(cohort)})...', flush=True)
    prices = load_price_panel(cohort, start=start, end=end, stooq_dir=stooq_dir)
    # Some cohort names may not appear in Stooq; restrict cohort.
    cohort = [c for c in cohort if c in prices.columns]
    prices = prices[cohort]
    K = len(cohort)
    print(f'  K={K}  prices: {prices.shape}', flush=True)

    # Align IV/HV to price trading-day index, ffill weekly with 1-day shift.
    iv_p = iv_full[[c for c in cohort if c in iv_full.columns]].copy()
    hv_p = hv_full[[c for c in cohort if c in hv_full.columns]].copy()
    iv_p = iv_p.shift(1)  # 1-day point-in-time shift
    hv_p = hv_p.shift(1)
    iv_p = iv_p.reindex(prices.index, method='ffill', limit=14)
    hv_p = hv_p.reindex(prices.index, method='ffill', limit=14)
    for c in cohort:
        if c not in iv_p.columns:
            iv_p[c] = np.nan
            hv_p[c] = np.nan
    iv_p = iv_p[cohort]
    hv_p = hv_p[cohort]

    iv_arr = iv_p.values.astype(np.float64)
    hv_arr = hv_p.values.astype(np.float64)
    avail_arr = (~np.isnan(iv_arr)).astype(np.float64)  # (T, K)
    iv_arr = np.nan_to_num(iv_arr, nan=0.0)
    hv_arr = np.nan_to_num(hv_arr, nan=0.0)

    close_arr = prices.values.astype(np.float64)  # (T, K)
    # Fill missing prices with forward-fill (rare; some tickers go private).
    close_df = pd.DataFrame(close_arr, index=prices.index, columns=cohort).ffill().bfill()
    close_arr = close_df.values
    T = close_arr.shape[0]
    print(f'  T={T}  date range {prices.index[0].date()}..{prices.index[-1].date()}',
          flush=True)

    # Per-name features.
    print(f'[v3-prep] computing per-name features...', flush=True)
    feat_blocks = np.zeros((T, K, F_ASSET_V3), dtype=np.float64)
    for j in range(K):
        px_f = _per_name_price_features(close_arr[:, j])  # (T, 6)
        iv_f = _per_name_iv_features(iv_arr[:, j], hv_arr[:, j], avail_arr[:, j])
        feat_blocks[:, j, :6] = px_f
        feat_blocks[:, j, 6:] = iv_f

    # Macro panel.
    print(f'[v3-prep] loading macro...', flush=True)
    macro = _load_macro(prices.index)
    macro_arr = macro.values.astype(np.float64)

    # Z-score (full-history; mild leak, documented).
    asset_mean = feat_blocks.reshape(-1, F_ASSET_V3).mean(axis=0)
    asset_std = feat_blocks.reshape(-1, F_ASSET_V3).std(axis=0)
    macro_mean = macro_arr.mean(axis=0)
    macro_std = macro_arr.std(axis=0)
    feats_n = _zscore(feat_blocks, asset_mean[None, None, :], asset_std[None, None, :])
    macro_n = _zscore(macro_arr, macro_mean[None, :], macro_std[None, :])

    # Per-name forward short-vol PnL.
    # Convention: vol-points (iv[t] - hv_forward[t+K]) * (K/252), minus 10bps friction.
    hv_fwd = np.concatenate(
        [hv_arr[k_forward:], np.repeat(hv_arr[-1:], k_forward, axis=0)], axis=0)
    vol_points_per_day = (iv_arr - hv_fwd) * avail_arr  # (T, K)
    # Per-rebal friction: 10bps when we hold a non-zero weight. Apply uniformly per name.
    vol_pnl_per_rebal = vol_points_per_day * (k_forward / 252.0) - 10e-4 * avail_arr

    # Build sliding windows.
    start_idx = t_lookback - 1
    end_idx = T - k_forward - 1
    n_eff = max(0, end_idx - start_idx)
    print(f'[v3-prep] building {n_eff} sliding windows...', flush=True)
    X_assets = np.empty((n_eff, K, t_lookback, F_ASSET_V3), dtype=np.float32)
    X_macro = np.empty((n_eff, t_lookback, F_MACRO), dtype=np.float32)
    valid_mask = np.empty((n_eff, K), dtype=np.float32)
    fwd_ret = np.empty((n_eff, K), dtype=np.float32)
    fwd_vol_pnl = np.empty((n_eff, K), dtype=np.float32)
    dates_out = []
    for k_idx, t in enumerate(range(start_idx, start_idx + n_eff)):
        X_assets[k_idx] = feats_n[t - t_lookback + 1:t + 1].transpose(1, 0, 2)
        X_macro[k_idx] = macro_n[t - t_lookback + 1:t + 1]
        valid_mask[k_idx] = avail_arr[t]
        entry = close_arr[t + 1]
        exit_ = close_arr[min(t + 1 + k_forward, T - 1)]
        fwd_ret[k_idx] = (exit_ / np.maximum(entry, 1e-9) - 1.0).astype(np.float32)
        fwd_vol_pnl[k_idx] = vol_pnl_per_rebal[t + 1].astype(np.float32)
        dates_out.append(prices.index[t])

    return PanelV3(
        X_assets=X_assets,
        X_macro=X_macro,
        valid_mask=valid_mask,
        fwd_ret=fwd_ret,
        fwd_vol_pnl=fwd_vol_pnl,
        dates=pd.DatetimeIndex(dates_out),
        tickers=cohort,
    )
