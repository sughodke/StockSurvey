"""Rolling correlation primitive shared by every `lie` strategy.

A Pearson correlation matrix over trailing log-returns is the empirical object
that downstream code interprets as either (a) the adjacency of a similarity
graph (HRP, hub centrality, clique density) or (b) the Gram matrix whose
spectrum encodes the effective symmetry rank.

The primitives here are intentionally numpy-only and stateless -- they take a
`(T, N)` price panel and return either a `(T-1, N)` log-return matrix or a
single `(N, N)` correlation snapshot. Names with NaN over the trailing window
are masked out and the surviving sub-block is placed back at the original
indices, with NaN at the masked positions, so callers can rely on stable
column ordering.
"""

from __future__ import annotations

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Per-bar log-returns of a `(T, N)` price panel; output is `(T-1, N)`.

    Zero / negative prices propagate as NaN rather than -inf so downstream
    masking treats them uniformly with missing data."""
    if prices.ndim != 2:
        raise ValueError(f'expected 2-D price panel, got shape {prices.shape}')
    if prices.shape[0] < 2:
        raise ValueError('need at least 2 bars to compute log-returns')
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.log(prices[1:] / prices[:-1])
    rets = np.where(np.isfinite(rets), rets, np.nan)
    return rets


def trailing_correlation(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Pearson correlation of log-returns over the last `lookback` bars.

    Returns an `(N, N)` matrix with NaN at the rows/cols of any name that had
    a NaN return inside the window. The valid sub-block is z-scored and the
    Gram matrix is computed in one matmul; eigenvalues of the result feed
    `symmetry_rank.effective_rank` and the linkage in `clustering`."""
    if prices.shape[0] < lookback + 1:
        raise ValueError(
            f'need at least {lookback + 1} bars for lookback={lookback}; '
            f'got {prices.shape[0]}')
    rets = log_returns(prices[-(lookback + 1):])
    n = rets.shape[1]
    valid = ~np.isnan(rets).any(axis=0)
    out = np.full((n, n), np.nan)
    if int(valid.sum()) < 2:
        return out
    sub = rets[:, valid]
    sub = sub - sub.mean(axis=0, keepdims=True)
    std = sub.std(axis=0, ddof=1, keepdims=True)
    # zero-variance columns (a flatlined name over the window) would NaN the
    # whole sub-block under naive division; treat them as already-centered
    # and let the resulting zero row/col fall out of clustering naturally.
    std = np.where(std == 0, 1.0, std)
    sub = sub / std
    csub = (sub.T @ sub) / (sub.shape[0] - 1)
    csub = np.clip(csub, -1.0, 1.0)
    idx = np.where(valid)[0]
    out[np.ix_(idx, idx)] = csub
    return out


__all__ = ['log_returns', 'trailing_correlation']
