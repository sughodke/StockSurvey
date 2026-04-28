"""Multi-ticker alignment of replay-style `TickerData` for cross-sectional
training.

The replay loader produces one `TickerData` per ticker, each with its
own date range. Training a cross-sectional scorer needs every ticker's
features and prices on a common date axis so the per-rebalance Pearson
IC has well-defined cross-sections.

`align_tickers` takes a list of `TickerData`, finds the common date
range (intersection — start = max of starts, end = min of ends), and
returns aligned arrays. Tickers whose date arrays don't overlap raise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ss_notebook.replay.features import TickerData


@dataclass(frozen=True)
class AlignedTickers:
    """Aligned multi-ticker tensors on a common date axis.

    Shapes (D = aligned dates, N = n_tickers, K = window_cols, F = channels):
      - dates:     `(D,)` numpy datetime64 / object array
      - names:     `(N,)` ticker names
      - features:  `(D, N, K, F)` float32
      - prices:    `(D, N)` float64
      - valid:     `(D, N)` bool — replay's per-ticker valid mask, aligned
    """
    dates: np.ndarray
    names: tuple[str, ...]
    features: np.ndarray
    prices: np.ndarray
    valid: np.ndarray


def align_tickers(
    tickers: list[TickerData], *, K: int, F: int,
) -> AlignedTickers:
    """Intersect ticker date ranges and stack into common-axis tensors.

    `K` and `F` are needed to reshape each ticker's `(n_dates, K*F)`
    feature matrix into `(n_dates, K, F)` before stacking. Pass them
    from a loaded `Backbone` so the reshape matches what the backbone
    was trained on.
    """
    if not tickers:
        raise ValueError('align_tickers needs at least one TickerData')
    for td in tickers:
        if td.features.shape[1] != K * F:
            raise ValueError(
                f'ticker {td.name!r}: features shape {td.features.shape} '
                f'incompatible with K*F = {K * F} (K={K}, F={F})')

    indexes = [pd.DatetimeIndex(td.dates) for td in tickers]
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    if len(common) == 0:
        raise ValueError(
            'tickers have no overlapping dates; check --start / --end')
    common = common.sort_values()

    D = len(common)
    N = len(tickers)
    features = np.empty((D, N, K, F), dtype=np.float32)
    prices = np.empty((D, N), dtype=np.float64)
    valid = np.zeros((D, N), dtype=bool)
    for j, (td, idx) in enumerate(zip(tickers, indexes)):
        loc = idx.get_indexer(common)
        if (loc < 0).any():
            missing = common[loc < 0]
            raise ValueError(
                f'ticker {td.name!r}: {len(missing)} common dates not '
                f'found in its index — duplicate dates?')
        features[:, j] = td.features[loc].reshape(-1, K, F).astype(np.float32)
        prices[:, j] = td.prices[loc]
        valid[:, j] = td.valid[loc]

    return AlignedTickers(
        dates=common.to_numpy(),
        names=tuple(td.name for td in tickers),
        features=features,
        prices=prices,
        valid=valid,
    )


def forward_log_returns(
    prices: np.ndarray, *, rebal_days: int,
) -> np.ndarray:
    """`(D, N)` of log-returns summed over the *next* `rebal_days` bars.

    `out[i, j] = sum(log(p[i+k+1] / p[i+k]) for k in range(rebal_days))`,
    so it's the log return realized by holding ticker j from close-of-i
    to close-of-(i+rebal_days). The trailing `rebal_days` rows are NaN
    (the future window doesn't fit).
    """
    log_p = np.log(np.maximum(prices.astype(np.float64), 1e-12))
    D, N = prices.shape
    fwd = np.full((D, N), np.nan, dtype=np.float64)
    if D > rebal_days:
        fwd[:D - rebal_days] = log_p[rebal_days:] - log_p[:D - rebal_days]
    return fwd
