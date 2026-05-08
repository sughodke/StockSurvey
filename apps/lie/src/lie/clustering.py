"""Hierarchical-clustering primitives over a correlation matrix.

Two pieces:

* `correlation_distance` -- the Lopez de Prado / Mantegna metric
  `d_ij = sqrt(0.5 * (1 - rho_ij))`, the canonical way to feed a correlation
  matrix into a Euclidean clustering algorithm. It IS a metric (not just a
  dissimilarity): triangle inequality holds, `d=0` iff `rho=1`.

* `hierarchical_linkage` + `quasi_diagonal_order` -- scipy single-linkage
  agglomerative clustering and the leaf-order traversal that produces a
  "quasi-diagonal" correlation matrix when its rows / columns are permuted.
  The leaf order is what HRP's recursive bisection walks.

Single linkage is the HRP default because it's what Lopez de Prado used and
what most published HRP comparisons benchmark against. Other linkage methods
(`average`, `ward`) are exposed via the `method` arg for future research --
they trade off chaining (single) vs compactness (ward) and aren't always
better in practice on financial correlation matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to the Lopez-de-Prado metric distance.

    `d_ij = sqrt(0.5 * (1 - rho_ij))`. Inputs are clipped to `[-1, 1]` first
    -- numerical noise can push the diagonal slightly past 1, which would
    otherwise yield a negative under the sqrt."""
    rho = np.clip(corr, -1.0, 1.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - rho), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def hierarchical_linkage(corr: np.ndarray, method: str = 'single') -> np.ndarray:
    """Compute a scipy linkage matrix from an `(N, N)` correlation matrix.

    Returns the standard scipy linkage `(N-1, 4)` array; pass to
    `quasi_diagonal_order` to recover the leaf order, or to
    `scipy.cluster.hierarchy.fcluster` to cut at a level for cluster IDs.

    The caller must pass a fully-populated correlation matrix -- NaN cells
    (masked-out names) will propagate through `squareform` and produce a
    NaN linkage. HRP filters its universe to liquid names before calling."""
    d = correlation_distance(corr)
    condensed = squareform(d, checks=False)
    return linkage(condensed, method=method)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order from a scipy linkage matrix.

    This is the order in which HRP's recursive bisection visits leaves, and
    permuting the correlation matrix to this order yields a "quasi-diagonal"
    block-structured matrix that visualizes the discovered clustering."""
    return [int(i) for i in leaves_list(link)]


__all__ = [
    'correlation_distance',
    'hierarchical_linkage',
    'quasi_diagonal_order',
]
