"""Spread + z-score + half-life primitives.

Once a pair is screened cointegrated and has a fitted hedge ratio
`β`, the spread is:

    s_t = log(P_A,t) − β · log(P_B,t) − α

where `α` is the EG intercept (mean spread under cointegration).
The trading signal is the z-score of `s_t` against its train-set
mean and stdev: `z_t = (s_t − μ_train) / σ_train`. Crossings of
predefined `z` thresholds (typically ±2σ for entry, ±0.5σ for
exit) are the trade events.

Half-life is reported as a diagnostic — pairs with very short
half-lives (< few days) are over-fit cointegration; pairs with
very long half-lives (> hundreds of days) won't revert within
a typical val window. v1 reports it but doesn't gate on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm


@dataclass(frozen=True)
class SpreadStats:
    """Train-set statistics needed to z-score val-side spread."""
    mean:      float
    std:       float
    half_life: float    # mean-reversion half-life in bars (NaN if non-stationary)


def compute_spread(
    log_p_a: np.ndarray, log_p_b: np.ndarray,
    beta: float, intercept: float,
) -> np.ndarray:
    """Spread = `log(P_A) − β · log(P_B) − α`."""
    return log_p_a - beta * log_p_b - intercept


def spread_stats(spread: np.ndarray) -> SpreadStats:
    """Mean / stdev / half-life of a spread series.

    Half-life: fit `Δs_t = θ · s_{t-1} + ε`. Half-life = `−ln(2) / θ`
    if θ < 0 (mean-reverting), else NaN.
    """
    if len(spread) < 5:
        return SpreadStats(
            mean=float('nan'), std=float('nan'),
            half_life=float('nan'))
    mean = float(np.mean(spread))
    std = float(np.std(spread, ddof=1))
    if std < 1e-12:
        return SpreadStats(mean=mean, std=std, half_life=float('nan'))
    # Half-life via OLS on diff-and-lag.
    s_lag = spread[:-1]
    ds    = np.diff(spread)
    try:
        x = sm.add_constant(s_lag - mean)
        theta = sm.OLS(ds, x).fit().params[1]
        if theta < 0:
            hl = float(-np.log(2) / theta)
        else:
            hl = float('nan')
    except (ValueError, np.linalg.LinAlgError):
        hl = float('nan')
    return SpreadStats(mean=mean, std=std, half_life=hl)


def zscore(spread: np.ndarray, stats: SpreadStats) -> np.ndarray:
    """Z-score using train-set mean and stdev (no peeking)."""
    if stats.std < 1e-12 or not np.isfinite(stats.std):
        return np.zeros_like(spread)
    return (spread - stats.mean) / stats.std


__all__ = ['SpreadStats', 'compute_spread', 'spread_stats', 'zscore']
