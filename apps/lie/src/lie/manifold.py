"""PCA-based manifold mapper.

Compresses a high-dimensional market-state feature matrix into a low-
dimensional embedding. Used to build the geometric coordinate system the
`TimelessPredictor` runs kNN over.

Key methodological commitments enforced here:

* **Train-only fit.** `fit(X_train)` learns the standardization statistics
  (per-feature mean, std) and the principal axes. `transform(X)` applies
  them; it never re-estimates from the input. This is what prevents the
  test set from shaping the manifold it's evaluated on.

* **Reports variance explained.** The "low-dimensional manifold" claim
  hinges on most of the variance fitting in few components. The mapper
  exposes `cumulative_variance_explained_` so a calling script can
  empirically check whether 8 components is the right cut, or whether
  the data wants 5 / 12 / 20.

Why hand-rolled SVD instead of `sklearn.decomposition.PCA`: the workspace
keeps `sklearn` to packages that already need it (relational uses k-means
+ GMM). Adding a hard sklearn dep just for `fit_transform` was overkill,
and `np.linalg.svd` on a `(T, F)` matrix at workspace-typical sizes
(T ~ 4000, F ~ 26) is microseconds.
"""

from __future__ import annotations

import numpy as np


class ManifoldMapper:
    """Standardize -> PCA project."""

    def __init__(self, n_components: int = 8) -> None:
        self.n_components = int(n_components)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> 'ManifoldMapper':
        """Fit the standardization + principal axes on `X`. Inputs must be
        finite (mask out NaN rows upstream via `valid_t` from
        `build_market_state`)."""
        if X.ndim != 2:
            raise ValueError(f'expected 2-D feature matrix, got {X.shape}')
        if not np.all(np.isfinite(X)):
            raise ValueError('X contains NaN/Inf -- mask via valid_t first')
        if X.shape[0] < self.n_components:
            raise ValueError(
                f'need at least {self.n_components} rows to fit '
                f'{self.n_components}-component PCA; got {X.shape[0]}')

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        # Constant features get std=1 so they pass through (zero-centered)
        # rather than NaN-ing out the whole matrix.
        self.std_ = np.where(self.std_ <= 0, 1.0, self.std_)
        Z = (X - self.mean_) / self.std_

        # SVD of the centered/standardized matrix.
        # Z = U S V^T;  components are rows of V^T (i.e., columns of V).
        # explained variance per component: S^2 / (n - 1).
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        n = Z.shape[0]
        ev = (S ** 2) / max(n - 1, 1)
        total = ev.sum() or 1.0
        self.explained_variance_ = ev[: self.n_components]
        self.explained_variance_ratio_ = ev[: self.n_components] / total
        self.components_ = Vt[: self.n_components]  # (n_components, F)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project `X` onto the fitted axes. `(T, F) -> (T, n_components)`."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError('ManifoldMapper not fit yet')
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f'expected (T, {self.mean_.shape[0]}) input; got {X.shape}')
        Z = (X - self.mean_) / self.std_
        return Z @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience for the train-set projection (equivalent to
        `fit(X).transform(X)`)."""
        return self.fit(X).transform(X)

    @property
    def cumulative_variance_explained_(self) -> np.ndarray:
        """Cumulative variance ratio across the kept components."""
        if self.explained_variance_ratio_ is None:
            raise RuntimeError('ManifoldMapper not fit yet')
        return np.cumsum(self.explained_variance_ratio_)


def variance_explained_at_k(X: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Standalone helper: cumulative variance at each k in `ks`.

    Useful before committing to a particular `n_components`. Avoids fitting
    a `ManifoldMapper` per k by computing the full SVD once."""
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError('X must be a finite 2-D matrix')
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.where(std <= 0, 1.0, std)
    Z = (X - mean) / std
    _, S, _ = np.linalg.svd(Z, full_matrices=False)
    n = Z.shape[0]
    ev = (S ** 2) / max(n - 1, 1)
    total = float(ev.sum()) or 1.0
    cum = np.cumsum(ev) / total
    out: dict[int, float] = {}
    for k in ks:
        if k <= 0:
            continue
        idx = min(k, len(cum)) - 1
        out[k] = float(cum[idx])
    return out


__all__ = ['ManifoldMapper', 'variance_explained_at_k']
