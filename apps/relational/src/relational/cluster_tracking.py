"""Hungarian-matched cluster-ID stabilization across k-means refits.

`relational.empirical_sectors._refit_cluster_assignments` refits k-means
every `refit_days` bars. K-means produces *arbitrary* cluster numbering
on every fit — segment 1's cluster `3` has nothing to do with segment
2's cluster `3`. Without alignment, every refit boundary looks like a
mass cluster transition for every ticker, drowning the true regime
signal in label-permutation noise.

This module's `stabilize_cluster_ids` post-processes the per-segment
cluster_ids matrix from `_refit_cluster_assignments`, computes per-
segment centroids by averaging fingerprints within each cluster on the
refit date, then for each successive segment runs
`scipy.optimize.linear_sum_assignment` on the cost matrix
`||old_centroid_k - new_centroid_j||²` to find the permutation that
maps new IDs back to old IDs (minimizing total centroid drift). The
output has the same shape and -1 sentinel convention but cluster IDs
that are continuous identities across refit boundaries.

Caveat: identity is well-defined only within an unbroken refit
sequence. If a cluster is empty in one segment (no fingerprints
mapped to it) it gets a fresh ID. We don't claim global identity
across the whole panel — only that within the locally-aligned chain,
"ticker X moved from cluster Y to cluster Z" reflects a real
fingerprint transition rather than a label permutation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def stabilize_cluster_ids(
    cluster_ids: np.ndarray,
    fps: np.ndarray,
    *,
    refit_days: int,
    lookback: int,
) -> np.ndarray:
    """Re-label `cluster_ids` so identities persist across refit boundaries.

    Parameters
    ----------
    cluster_ids : np.ndarray, shape `(n_dates, n_tickers)`, int64
        Output of `_refit_cluster_assignments`. Values in `[0, k)` for
        dates with a refit and `-1` for pre-lookback / non-finite cells.
    fps : np.ndarray, shape `(n_dates, n_tickers, fp_dim)`, float
        Same fingerprint tensor passed into `_refit_cluster_assignments`.
        Used to compute per-segment centroids by averaging fingerprints
        within each cluster on the refit date.
    refit_days : int
        Refit cadence (bars). Must match what was passed to the refit
        helper — used to derive segment boundaries.
    lookback : int
        First refit date index. Must match the refit helper.

    Returns
    -------
    stable : np.ndarray, shape `(n_dates, n_tickers)`, int64
        Cluster IDs re-labeled so segment 2's "cluster X" is the
        Hungarian-matched continuation of segment 1's "cluster X". The
        first segment's IDs are kept as-is (the anchor). `-1` cells are
        preserved.
    """
    n_dates, n_tickers = cluster_ids.shape
    if n_dates == 0:
        return cluster_ids.copy()

    refit_dates = list(range(lookback, n_dates, refit_days))
    if len(refit_dates) <= 1:
        return cluster_ids.copy()

    stable = cluster_ids.copy()

    # Cluster id range. Skip if everything is -1 (no refits succeeded).
    valid_mask = cluster_ids >= 0
    if not valid_mask.any():
        return stable
    k_max = int(cluster_ids[valid_mask].max()) + 1

    def _segment_centroids(t_refit: int, ids_row: np.ndarray) -> np.ndarray:
        """Per-cluster mean fingerprint at `t_refit`, shape `(k, fp_dim)`.

        Returns `np.nan` rows for clusters that have no members in the
        segment (so the cost-matrix row for that id is uninformative
        and gets a sentinel large cost — effectively a free assignment).
        """
        fp_dim = fps.shape[2]
        centroids = np.full((k_max, fp_dim), np.nan, dtype=np.float64)
        for c in range(k_max):
            members = np.where(ids_row == c)[0]
            if members.size == 0:
                continue
            row = fps[t_refit, members, :]
            finite = np.isfinite(row).all(axis=1)
            if not finite.any():
                continue
            centroids[c] = row[finite].mean(axis=0)
        return centroids

    prev_centroids = _segment_centroids(refit_dates[0], stable[refit_dates[0]])

    for seg_idx in range(1, len(refit_dates)):
        t_refit = refit_dates[seg_idx]
        seg_end = (refit_dates[seg_idx + 1]
                   if seg_idx + 1 < len(refit_dates) else n_dates)
        ids_row = stable[t_refit]
        new_centroids = _segment_centroids(t_refit, ids_row)

        # Cost matrix: (old_id, new_id) -> ||old_centroid - new_centroid||^2.
        # Empty centroids (NaN) get a large sentinel cost so
        # linear_sum_assignment treats them as low-priority matches.
        big = 1e9
        cost = np.full((k_max, k_max), big, dtype=np.float64)
        for o in range(k_max):
            if not np.isfinite(prev_centroids[o]).all():
                continue
            for n in range(k_max):
                if not np.isfinite(new_centroids[n]).all():
                    continue
                d = prev_centroids[o] - new_centroids[n]
                cost[o, n] = float(np.dot(d, d))
        old_idx, new_idx = linear_sum_assignment(cost)
        # `mapping[new_id] = old_id` — re-label new to align with old.
        mapping = np.arange(k_max, dtype=np.int64)
        for o, n in zip(old_idx, new_idx):
            mapping[n] = o

        # Apply the permutation across the segment's rows for valid ids.
        # TODO(review #12): the np.clip(..., 0, k_max-1) is needed only
        # to make `mapping[seg_block]` index-safe for the -1 cells that
        # the np.where then masks out. Cleaner: precompute on the valid
        # slice only — `result = np.full_like(seg_block, -1); result[valid]
        # = mapping[seg_block[valid]]`. Pure readability, behavior identical.
        seg_block = stable[t_refit:seg_end]
        valid = seg_block >= 0
        seg_block = np.where(valid, mapping[np.clip(seg_block, 0, k_max - 1)],
                             seg_block)
        stable[t_refit:seg_end] = seg_block

        # Recompute the centroid table under the re-labeled IDs so the
        # next iteration matches against the same identity space.
        prev_centroids = _segment_centroids(t_refit, stable[t_refit])

    return stable
