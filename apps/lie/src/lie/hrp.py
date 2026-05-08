"""Hierarchical Risk Parity weights -- Lopez de Prado (2016).

Three stages, transparently mapped to the published algorithm:

1. **Tree clustering**  -- correlation matrix -> distance metric -> single-
   linkage hierarchical clustering. `clustering.hierarchical_linkage` does
   this; the leaf order from `quasi_diagonal_order` is HRP's "quasi-
   diagonalization" step.

2. **Quasi-diagonalization** (sortIx) -- permute the covariance matrix so
   similar names sit adjacent. Implicit in our pipeline: we keep the original
   indexing and permute on the fly via `sort_ix`, which gives an identical
   result without copying.

3. **Recursive bisection** -- walk the leaf order top-down, splitting each
   contiguous block in half, and allocate weight between the halves inversely
   proportional to each half's inverse-variance-portfolio variance:

       alpha_left = 1 - V_left / (V_left + V_right)

   where each half's V is the variance of its own IVP sub-portfolio. Within
   each leaf cluster, weight is then assigned by IVP. The recursion respects
   the cluster geometry at every level -- a tightly-correlated trio of names
   is sized like a single position, not three.

Inputs are a `(T, N)` price panel and a `lookback`. We compute log-returns
over the trailing window, build the correlation/covariance pair, drop NaN
columns, run HRP on the survivors, and pad the result back to the full
`N`-length output (zeros for masked names). Total weight sums to 1 over the
survivors; the caller can apply caps / spread gates downstream.

Why not riskfolio-lib: this is ~80 lines of numpy, the pinned-dep blast
radius isn't worth it, and we want the recursive-bisection step inspectable
in case we layer on group / sector constraints later.
"""

from __future__ import annotations

import numpy as np

from lie.clustering import hierarchical_linkage, quasi_diagonal_order
from lie.correlation_network import log_returns


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance-portfolio weights for an `(n, n)` covariance block."""
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov: np.ndarray, idx: np.ndarray) -> float:
    """Variance of the IVP portfolio over the sub-block `cov[idx, idx]`."""
    sub = cov[np.ix_(idx, idx)]
    w = _ivp_weights(sub).reshape(-1, 1)
    return float((w.T @ sub @ w).item())


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Lopez de Prado HRP step 3.

    Walks the leaf order, halving each contiguous block, and accumulates
    multiplicative weight scaling per index. Returns a length-`n` vector
    aligned with the original (un-sorted) indexing of `cov`."""
    n = cov.shape[0]
    w = np.ones(n)
    clusters: list[list[int]] = [list(sort_ix)]
    while clusters:
        next_clusters: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left = cl[:half]
            right = cl[half:]
            v_l = _cluster_var(cov, np.array(left))
            v_r = _cluster_var(cov, np.array(right))
            denom = v_l + v_r
            # both halves degenerate (e.g. two flatlined names in a window) ->
            # split 50/50 rather than 0/0; this is rare on real OHLC.
            alpha = 0.5 if denom <= 0 else 1.0 - v_l / denom
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return w


def weights_hrp(
    prices: np.ndarray,
    lookback: int,
    linkage_method: str = 'single',
) -> np.ndarray:
    """HRP target weights for the latest bar of a `(T, N)` price panel.

    Names with NaN log-returns inside the trailing window are masked out;
    the returned vector has zero weight at their indices and sums to 1
    over the survivors. Returns an all-zero vector if fewer than 2 names
    have a full window of data."""
    if prices.shape[0] < lookback + 1:
        raise ValueError(
            f'need at least {lookback + 1} bars for lookback={lookback}; '
            f'got {prices.shape[0]}')
    n = prices.shape[1]
    rets = log_returns(prices[-(lookback + 1):])
    valid = ~np.isnan(rets).any(axis=0)
    if int(valid.sum()) < 2:
        return np.zeros(n)

    sub_rets = rets[:, valid]
    cov = np.cov(sub_rets, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    diag = np.diag(cov).copy()
    # zero-variance survivors get a tiny floor so IVP doesn't divide by zero;
    # flatlined names contribute nothing meaningful but shouldn't NaN the run.
    diag[diag <= 0] = np.finfo(float).eps
    np.fill_diagonal(cov, diag)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    link = hierarchical_linkage(corr, method=linkage_method)
    sort_ix = quasi_diagonal_order(link)
    w_sub = _recursive_bisection(cov, sort_ix)

    w = np.zeros(n)
    w[valid] = w_sub
    s = float(w.sum())
    return w / s if s > 0 else w


__all__ = ['weights_hrp']
