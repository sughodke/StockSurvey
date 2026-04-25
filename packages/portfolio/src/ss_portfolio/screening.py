"""Universe-screening masks for ranking strategies.

Both functions mutate `scores` in place and return it. They turn cells
to NaN so that downstream `select_top_n_matrix` excludes those names
from the basket without disturbing the row's other rankings.

`scores` is `(n_valid, n_tickers)` where `n_valid = n_dates - lookback`
— rows correspond to "the date `lookback + i` had this score for each
ticker." The `lookback` argument lines `scores` up against the wider
`spread_arr` / `price_arr` matrices that include the warm-up window.
"""

from __future__ import annotations

import numpy as np


def apply_spread_mask(
    scores: np.ndarray,
    spread_arr: np.ndarray,
    lookback: int,
    max_spread: float,
) -> np.ndarray:
    """NaN out scores for tickers with estimated spread > `max_spread`."""
    if spread_arr is None:
        return scores
    n_dates_scores = scores.shape[0]
    for i in range(n_dates_scores):
        spread_row = spread_arr[i + lookback]
        scores[i, spread_row > max_spread] = np.nan
    return scores


def apply_nan_mask(
    scores: np.ndarray,
    price_arr: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """NaN out scores for tickers with any NaN price in the lookback window."""
    n_dates = price_arr.shape[0]
    for i in range(lookback, n_dates):
        chunk = price_arr[i - lookback:i + 1]
        has_nan = np.any(np.isnan(chunk), axis=0)
        scores[i - lookback, has_nan] = np.nan
    return scores
