"""kNN forward-return predictor with hard temporal-gap exclusion.

This is the operationalization of "geometric neighbors predict better than
temporal ones." For a query embedding at date T:

1. Find the k training points closest in embedding-space (L2).
2. Drop any candidate within `temporal_gap` trading days of T.
3. Predict the distance-weighted mean of the candidates' forward returns.

The temporal-gap step is the critical methodological commitment. Without it
the model can use "yesterday's state" as a neighbor of "today's state" and
get a high IC purely from autocorrelation -- which is not the geometric-
structure claim we're testing. The 60-day default puts the closest
admissible neighbor a full quarter away, far past any plausible near-term
mean-reversion or short-horizon momentum window.

Distance weighting (`'inverse_distance'` default) is `1 / (d + eps)`. This
makes the predictor approximately a Nadaraya-Watson regressor in the
embedding space. `'uniform'` weighting recovers the classical kNN regressor.
"""

from __future__ import annotations

import numpy as np


class TimelessPredictor:
    """k-Nearest-Neighbor predictor over manifold embeddings."""

    def __init__(
        self,
        k: int = 50,
        temporal_gap: int = 60,
        weighting: str = 'inverse_distance',
        eps: float = 1e-9,
    ) -> None:
        if k < 1:
            raise ValueError(f'k must be >= 1, got {k}')
        if temporal_gap < 0:
            raise ValueError(f'temporal_gap must be >= 0, got {temporal_gap}')
        if weighting not in ('uniform', 'inverse_distance'):
            raise ValueError(
                f'weighting must be "uniform" or "inverse_distance"; '
                f'got {weighting!r}')
        self.k = int(k)
        self.temporal_gap = int(temporal_gap)
        self.weighting = weighting
        self.eps = float(eps)

        self._embeddings: np.ndarray | None = None
        self._t_idx: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        t_idx: np.ndarray,
        targets: np.ndarray,
    ) -> 'TimelessPredictor':
        """Store the training set.

        Parameters
        ----------
        embeddings :
            `(N_train, D)` manifold positions.
        t_idx :
            `(N_train,)` integer trading-day indices. Must be monotonically
            increasing. The temporal-gap filter compares query indices to
            these; passing a calendar `Timestamp` array would defeat the
            point because real time has weekends/holidays in it.
        targets :
            `(N_train,)` forward returns aligned with `embeddings`.
        """
        if embeddings.ndim != 2:
            raise ValueError(f'embeddings must be 2-D, got {embeddings.shape}')
        if t_idx.shape != (embeddings.shape[0],):
            raise ValueError('t_idx must align with embeddings')
        if targets.shape != (embeddings.shape[0],):
            raise ValueError('targets must align with embeddings')
        if not np.all(np.isfinite(embeddings)):
            raise ValueError('embeddings contain NaN/Inf')
        if not np.all(np.diff(t_idx) >= 0):
            raise ValueError('t_idx must be non-decreasing')

        self._embeddings = embeddings.astype(np.float64, copy=False)
        self._t_idx = t_idx.astype(np.int64, copy=False)
        self._targets = targets.astype(np.float64, copy=False)
        return self

    def predict(
        self,
        embeddings: np.ndarray,
        t_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict forward returns for query embeddings.

        Returns
        -------
        (preds, n_used)
            `preds` is `(N_query,)`; entries are NaN where fewer than `k`
            admissible neighbors existed (after the temporal-gap filter).
            `n_used` is `(N_query,)` int -- the number of training points
            consumed for each prediction. Useful for diagnosing how often
            the gap is biting.
        """
        if self._embeddings is None:
            raise RuntimeError('TimelessPredictor not fit yet')
        if embeddings.ndim != 2 or embeddings.shape[1] != self._embeddings.shape[1]:
            raise ValueError(
                f'expected (N, {self._embeddings.shape[1]}) embeddings; '
                f'got {embeddings.shape}')
        if t_idx.shape != (embeddings.shape[0],):
            raise ValueError('t_idx must align with embeddings')

        Q = embeddings.astype(np.float64, copy=False)
        Qt = t_idx.astype(np.int64, copy=False)

        # Pairwise squared distances: |q - x|^2 = |q|^2 + |x|^2 - 2 q.x
        train = self._embeddings
        train_targets = self._targets
        train_t = self._t_idx

        q_sq = (Q ** 2).sum(axis=1, keepdims=True)               # (Nq, 1)
        x_sq = (train ** 2).sum(axis=1, keepdims=True).T          # (1, Ntr)
        d2 = q_sq + x_sq - 2.0 * (Q @ train.T)
        # numerical noise can produce small negatives -> clip
        d2 = np.maximum(d2, 0.0)

        # Temporal-gap mask: allow only training points whose t-index is at
        # least `temporal_gap` away from the query t-index.
        dt = np.abs(Qt[:, None] - train_t[None, :])
        admissible = dt >= self.temporal_gap

        Nq = Q.shape[0]
        preds = np.full(Nq, np.nan)
        n_used = np.zeros(Nq, dtype=np.int64)

        for i in range(Nq):
            mask = admissible[i]
            n_adm = int(mask.sum())
            if n_adm == 0:
                continue
            adm_idx = np.where(mask)[0]
            adm_d2 = d2[i, adm_idx]
            k_use = min(self.k, n_adm)
            # argpartition picks the k smallest distances cheaply.
            sel = np.argpartition(adm_d2, k_use - 1)[:k_use] if k_use > 1 else np.array([int(np.argmin(adm_d2))])
            sel_global = adm_idx[sel]
            sel_d = np.sqrt(d2[i, sel_global])
            sel_t = train_targets[sel_global]

            if self.weighting == 'uniform':
                preds[i] = float(np.mean(sel_t))
            else:
                w = 1.0 / (sel_d + self.eps)
                preds[i] = float(np.sum(w * sel_t) / np.sum(w))
            n_used[i] = k_use

        return preds, n_used


def information_coefficient(
    preds: np.ndarray,
    targets: np.ndarray,
    method: str = 'pearson',
) -> float:
    """Pearson (default) or Spearman correlation between predictions and
    realized targets, ignoring NaN pairs."""
    mask = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 3:
        return float('nan')
    p = preds[mask]
    t = targets[mask]
    if method == 'spearman':
        from scipy.stats import spearmanr
        return float(spearmanr(p, t).correlation)
    pm = p - p.mean()
    tm = t - t.mean()
    denom = float(np.sqrt(np.sum(pm * pm) * np.sum(tm * tm)))
    if denom <= 0:
        return float('nan')
    return float(np.sum(pm * tm) / denom)


__all__ = ['TimelessPredictor', 'information_coefficient']
