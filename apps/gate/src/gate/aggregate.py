"""Aggregate-universe pre-processing — EW return series + per-date features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregateSeries:
    """Per-date aggregate state.

    `dates` is a `(T,)` `DatetimeIndex` of trading days; `ew_simple_ret`
    is the same-length EW simple return series; `ew_log_ret` is its
    log-return version (used for drawdown and feature construction
    because additive accumulation is cleaner in log space).
    """
    dates: pd.DatetimeIndex
    ew_simple_ret: np.ndarray
    ew_log_ret: np.ndarray
    n_active_per_date: np.ndarray   # (T,) int — # tickers w/ data at t


def build_ew_aggregate(
    prices: pd.DataFrame, *, min_active: int = 10,
) -> AggregateSeries:
    """Build the EW return series from a panel of prices.

    `prices` is a `DatetimeIndex × ticker` close panel. For each date,
    weight = `1/N_active_t` over tickers with both a current and a
    prior valid close. Tickers with leading NaN at the start of the
    span are excluded from the weight on that date — the basket
    grows as more tickers come online but doesn't pay phantom return
    on tickers that haven't IPO'd yet. Tickers that delist after
    their last quoted price drop out of subsequent weights (they
    contribute zero return on the drop-out date — a small bias we
    accept rather than model).

    `min_active` skips dates where fewer than `min_active` tickers
    have valid simple returns — typically only matters at the very
    start of the panel before sufficient tickers come online.
    """
    if prices.empty:
        raise ValueError('prices panel is empty')
    p = prices.sort_index()
    # Per-bar simple return per ticker; first bar per ticker is NaN.
    ret = p.pct_change()
    # Mask: this ticker had a return on this bar (i.e. prior bar valid).
    valid = ret.notna().values
    ret_arr = np.where(valid, ret.values, 0.0).astype(np.float64)
    n_active = valid.sum(axis=1)
    safe_n = np.maximum(n_active, 1)
    ew_simple = ret_arr.sum(axis=1) / safe_n
    # Drop dates with too few active tickers (start-of-panel).
    keep = n_active >= min_active
    if not keep.any():
        raise ValueError(
            f'no dates with >= {min_active} active tickers')
    ew_simple = ew_simple[keep]
    ew_log = np.log1p(ew_simple)
    return AggregateSeries(
        dates=p.index[keep],
        ew_simple_ret=ew_simple,
        ew_log_ret=ew_log,
        n_active_per_date=n_active[keep],
    )


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing-only stdev. Sample size grows from 1 to `window` at
    the start; NaN where sample is < 2."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        sample = x[lo:i + 1]
        if len(sample) >= 2:
            out[i] = float(np.std(sample, ddof=1))
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        lo = max(0, i + 1 - window)
        sample = x[lo:i + 1]
        if len(sample) >= 1:
            out[i] = float(np.mean(sample))
    return out


def _trailing_max_drawdown(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Max drawdown over the trailing `window` bars (positive number).

    Computed in log space: drawdown = max(running_peak - cum_log_ret)
    over the window. Reset cumulative log return to 0 at the window
    start so the answer is local to the window, not cumulative since
    inception.
    """
    out = np.full_like(log_ret, np.nan, dtype=np.float64)
    for i in range(len(log_ret)):
        lo = max(0, i + 1 - window)
        sample = log_ret[lo:i + 1]
        if len(sample) >= 2:
            cum = np.cumsum(sample)
            peak = np.maximum.accumulate(cum)
            out[i] = float(np.max(peak - cum))
    return out


def build_aggregate_features(agg: AggregateSeries) -> pd.DataFrame:
    """Per-date feature stack observed at time `t` (no peeking).

    All features are point-in-time — at row `t` the value uses only
    bars `[..., t]`, never future bars. Features:

      vol_5     : trailing 5-day stdev of EW log return
      vol_20    : trailing 20-day stdev
      vol_60    : trailing 60-day stdev
      ret_5     : trailing 5-day mean log return
      ret_20    : trailing 20-day mean log return
      ret_60    : trailing 60-day mean log return
      tdd_20    : trailing-20-day max drawdown
      tdd_60    : trailing-60-day max drawdown
      vol_term  : vol_5 - vol_60   (term structure of realized vol)
      breadth   : n_active_per_date / max(n_active_per_date)
                  (how much of the universe is online — proxy for
                  the universe's effective breadth at this date)

    Returns a DataFrame indexed by `agg.dates` with NaN rows for the
    early period before all rolling windows have warmed up. Caller
    should `.dropna()` before training.
    """
    log_ret = agg.ew_log_ret
    df = pd.DataFrame(index=agg.dates)
    df['vol_5']  = _rolling_std(log_ret, 5)
    df['vol_20'] = _rolling_std(log_ret, 20)
    df['vol_60'] = _rolling_std(log_ret, 60)
    df['ret_5']  = _rolling_mean(log_ret, 5)
    df['ret_20'] = _rolling_mean(log_ret, 20)
    df['ret_60'] = _rolling_mean(log_ret, 60)
    df['tdd_20'] = _trailing_max_drawdown(log_ret, 20)
    df['tdd_60'] = _trailing_max_drawdown(log_ret, 60)
    df['vol_term'] = df['vol_5'] - df['vol_60']
    n_max = float(np.max(agg.n_active_per_date))
    df['breadth'] = agg.n_active_per_date / n_max if n_max > 0 else 0.0
    return df


__all__ = [
    'AggregateSeries',
    'build_aggregate_features',
    'build_ew_aggregate',
]
