"""Forward-drawdown target construction.

`forward_max_drawdown(log_ret, horizon)` returns, for each date `t`,
the maximum peak-to-trough drawdown of the EW log-return series over
the *next* `horizon` bars `(t, t+horizon]`. This is the supervised
target the predictor learns from.

Computed in log space so accumulation is additive: at each future
bar `s ∈ (t, t+horizon]`, drawdown is
`max_running_peak_in_(t,s] − cum_log_ret_at_s`. The target at row
`t` is the max of those over `s`.

Trailing entries where the next `horizon` bars don't fully exist get
NaN — caller drops them before training (`scripts/run_*.py` does
this in `align_features_target`).
"""
from __future__ import annotations

import numpy as np


def forward_max_drawdown(
    log_ret: np.ndarray, horizon: int = 20,
) -> np.ndarray:
    """For each `t`, max DD of `cumsum(log_ret[t+1:t+horizon+1])`.

    Returns a numpy array shape `(len(log_ret),)`. Last `horizon`
    entries are NaN (no full forward window).
    """
    if log_ret.ndim != 1:
        raise ValueError(f'expected 1-D, got shape {log_ret.shape}')
    if horizon < 1:
        raise ValueError(f'horizon={horizon} must be >= 1')
    n = len(log_ret)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(n - horizon):
        # Forward window is (t, t+horizon]. Cum log return starts at
        # 0 at t (we're measuring drawdown of the *future* trajectory,
        # not cumulative since inception).
        future = log_ret[t + 1: t + horizon + 1]
        cum = np.cumsum(future)
        # Running peak includes the implicit starting point of 0 (we
        # could have closed at t for zero drawdown, so the peak can't
        # go below 0).
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        out[t] = float(np.max(peak - cum))
    return out


__all__ = ['forward_max_drawdown']
