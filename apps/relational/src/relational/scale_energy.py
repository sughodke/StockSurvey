"""N1 — per-(ticker, date) scale-energy summary stats from cached CWT.

Two related signals, each a single scalar per (ticker, date) computed
directly from the cached coeffs (no fingerprint extraction needed):

  * `scale_energy_ratio_scores` — sum of short-scale power divided by
    sum of long-scale power. Trend-exhaustion proxy: when short-scale
    energy spikes while long stays flat, the trend is breaking down on
    the short horizon.
  * `scale_entropy_scores` — *negative* Shannon entropy of the scale-
    energy distribution. The brainstorm's mechanistic claim is that
    entropy collapses *before* regime breaks; we negate so descending-
    sort top-N picks the lowest-entropy (most concentrated) names —
    consistent with other scorers' "high score = pick me" convention.

Both functions return `(n_eval, n_tickers)` where
`n_eval = n_dates - lookback`, matching the shape contract used by
`apply_nan_mask` and `select_top_n_matrix` downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relational.scalogram_cache import load_or_compute_cwt


def _split_scales(scales: list[int], split_point: int | None) -> tuple[list[int], list[int]]:
    """Default split = midpoint of the scale list (sorted ascending)."""
    if split_point is None:
        split_point = len(scales) // 2
    return list(range(split_point)), list(range(split_point, len(scales)))


def scale_energy_ratio_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    split_point: int | None = None,
    cache_dir=None,
) -> np.ndarray:
    """Per-(date, ticker) short/long scale-energy ratio."""
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    power = (coeffs ** 2).astype(np.float32)
    short_idx, long_idx = _split_scales(scales, split_point)
    short_e = power[short_idx].sum(axis=0)
    long_e = power[long_idx].sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(long_e > 0, short_e / long_e, np.nan)
    return ratio[lookback:].astype(np.float32, copy=True)


def scale_entropy_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    cache_dir=None,
) -> np.ndarray:
    """Per-(date, ticker) negative Shannon entropy of scale-energy."""
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    power = (coeffs ** 2).astype(np.float64)
    p_sum = power.sum(axis=0, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(p_sum > 0, power / p_sum, 0.0)
    plogp = np.where(p > 0, p * np.log(p), 0.0)
    entropy = -plogp.sum(axis=0)        # (n_dates, n_tickers)
    return (-entropy[lookback:]).astype(np.float32, copy=True)
