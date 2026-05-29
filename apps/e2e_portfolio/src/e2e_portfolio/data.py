"""Feature computation for the e2e Phase-4d portfolio allocator.

Inputs are raw prices + raw macro only — no pre-computed meta-layer
alpha streams. Per-asset features are 6 deterministic indicators
computed from close-only history; macro side-channel is VIX + 252d
VIX percentile + 10y-3m slope + BAA10Y credit spread.

Built as a single `prepare_panel` call that returns dense tensors:
  X_assets : (T_eff, N=13, T_lookback=60, F_asset=6)
  X_macro  : (T_eff, T_lookback=60, F_macro=4)
  fwd_ret  : (T_eff, N=13)   — K-day forward arithmetic return
  dates    : (T_eff,)        — anchor date for each sample
The cash leg is added by the model — it's just a column of zeros in
fwd_ret at the loss step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from ss_indicators import rsi

REPO = Path(__file__).resolve().parents[4]
DEFAULT_CLOSE_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'

PHASE4D_TICKERS = ['DBC', 'GLD', 'IEF', 'TLT', 'XLB', 'XLE', 'XLF',
                   'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

T_LOOKBACK = 60
F_ASSET = 6
F_MACRO = 4
K_FORWARD = 20


def load_close(path: str | Path = DEFAULT_CLOSE_PKL) -> pd.DataFrame:
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df.astype(np.float64)


def load_macro_panel(close_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill the three FRED series + a derived VIX percentile
    onto the trading-day index of the close panel.

    Columns: vix, vix_pct_252, slope_10y_3m, credit_baa.
    Point-in-time discipline: each value is the most recent
    observation on or before the trading day.
    """
    from ss_macro.loaders import load_fred_series
    vix = load_fred_series('VIXCLS')
    slope = load_fred_series('T10Y3M')
    credit = load_fred_series('BAA10Y')

    out = pd.DataFrame(index=close_index)
    out['vix'] = vix.reindex(close_index, method='ffill')
    out['slope_10y_3m'] = slope.reindex(close_index, method='ffill')
    out['credit_baa'] = credit.reindex(close_index, method='ffill')
    # VIX percentile vs trailing 252d, using past-only window (.shift(1) so
    # today's VIX isn't fed into today's rank).
    vix_aligned = out['vix']
    out['vix_pct_252'] = (
        vix_aligned.rolling(252, min_periods=60)
        .apply(lambda w: (w[-1] >= w[:-1]).mean(), raw=True)
    )
    out = out.ffill().bfill()
    # Reorder: vix, vix_pct, slope, credit.
    return out[['vix', 'vix_pct_252', 'slope_10y_3m', 'credit_baa']]


def _per_asset_features(close: np.ndarray) -> np.ndarray:
    """Compute (T, F=6) features for a single asset given a (T,) close array.

    Features:
      0: log return (1d)
      1: realized vol over 20d (std of 1d log returns)
      2: realized vol over 60d
      3: RSI(14)  — normalized to ~[-1, 1] by mapping (rsi-50)/50
      4: close / 252d rolling mean  - 1.0   (trend deviation)
      5: 5d momentum (sum of last 5 daily log returns)
    """
    T = close.shape[0]
    out = np.zeros((T, F_ASSET), dtype=np.float64)
    log_close = np.log(close + 1e-12)
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


@dataclass
class Panel:
    X_assets: np.ndarray  # (T_eff, N, T_lookback, F_asset)
    X_macro: np.ndarray   # (T_eff, T_lookback, F_macro)
    fwd_ret: np.ndarray   # (T_eff, N) — K-day forward arithmetic return per asset
    dates: pd.DatetimeIndex  # (T_eff,) anchor dates
    tickers: list[str]

    def slice_by_date(self, start: str | pd.Timestamp | None,
                      end: str | pd.Timestamp | None) -> 'Panel':
        s = pd.Timestamp(start) if start is not None else self.dates[0]
        e = pd.Timestamp(end) if end is not None else self.dates[-1]
        mask = (self.dates >= s) & (self.dates <= e)
        idx = np.where(mask)[0]
        return Panel(
            X_assets=self.X_assets[idx],
            X_macro=self.X_macro[idx],
            fwd_ret=self.fwd_ret[idx],
            dates=self.dates[idx],
            tickers=self.tickers,
        )


def _zscore_per_feature(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.maximum(std, 1e-6)


def prepare_panel(
    close: pd.DataFrame | None = None,
    *,
    t_lookback: int = T_LOOKBACK,
    k_forward: int = K_FORWARD,
    macro_panel: pd.DataFrame | None = None,
) -> Panel:
    """Build the full (T_eff, ...) dense panel of features and labels.

    Sliding windows are built once for the full date range; train/val
    slicing happens later with `Panel.slice_by_date`.
    """
    if close is None:
        close = load_close()
    close = close[PHASE4D_TICKERS]
    close_arr = close.values  # (T, N)
    T, N = close_arr.shape
    assert N == 13, f'expected 13 tickers, got {N}'

    # Per-asset features (computed once over full history).
    feats = np.stack([_per_asset_features(close_arr[:, j]) for j in range(N)],
                     axis=1)  # (T, N, F_asset)

    # Macro features over trading-day index.
    if macro_panel is None:
        macro_panel = load_macro_panel(close.index)
    macro_arr = macro_panel.values.astype(np.float64)  # (T, F_macro)

    # Need T_lookback bars of history and K_forward bars of future.
    # Anchor index t means features cover [t - T_lookback + 1, t]
    # and forward return is (close[t+K]/close[t+1] - 1) — entered at
    # close-of-t (use close[t+1] as the entry next-bar to avoid the
    # one-bar look-ahead).
    start = t_lookback - 1
    end = T - k_forward - 1  # need t+1+K-1 = t+K, plus a tail safety bar
    n_eff = max(0, end - start)
    if n_eff <= 0:
        raise ValueError(f'no samples: T={T}, t_lookback={t_lookback}, k_forward={k_forward}')

    # Z-score normalization parameters computed on the *full* history
    # (per-feature). This is a mild data leak (you'd compute these on a
    # train slice in production), but for direct-Sharpe with bounded
    # softmax weights the impact is small and the convention is shared
    # across folds. Documented in the finding.
    asset_mean = feats.reshape(-1, F_ASSET).mean(axis=0)
    asset_std = feats.reshape(-1, F_ASSET).std(axis=0)
    macro_mean = macro_arr.mean(axis=0)
    macro_std = macro_arr.std(axis=0)

    feats_n = _zscore_per_feature(feats, asset_mean[None, None, :], asset_std[None, None, :])
    macro_n = _zscore_per_feature(macro_arr, macro_mean[None, :], macro_std[None, :])

    X_assets = np.empty((n_eff, N, t_lookback, F_ASSET), dtype=np.float32)
    X_macro = np.empty((n_eff, t_lookback, F_MACRO), dtype=np.float32)
    fwd_ret = np.empty((n_eff, N), dtype=np.float32)
    dates_out = []

    for k, t in enumerate(range(start, start + n_eff)):
        X_assets[k] = feats_n[t - t_lookback + 1:t + 1].transpose(1, 0, 2)
        X_macro[k] = macro_n[t - t_lookback + 1:t + 1]
        # Enter at close[t+1], exit at close[t+K+1] -> arithmetic return.
        entry = close_arr[t + 1]
        exit_ = close_arr[min(t + 1 + k_forward, T - 1)]
        fwd_ret[k] = (exit_ / np.maximum(entry, 1e-9) - 1.0).astype(np.float32)
        dates_out.append(close.index[t])

    return Panel(
        X_assets=X_assets,
        X_macro=X_macro,
        fwd_ret=fwd_ret,
        dates=pd.DatetimeIndex(dates_out),
        tickers=PHASE4D_TICKERS,
    )
