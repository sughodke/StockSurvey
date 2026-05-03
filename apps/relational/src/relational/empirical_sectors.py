"""Idea A — empirical sectors via k-means on scalogram fingerprints.

Drop-in replacement for `relational.scoring.weights_excess_regime` that
uses **k-means clusters of per-(date, ticker) scalogram fingerprints**
as the sector aggregate, instead of static GICS labels from
`relational.sectors.PHASE2_TICKER_TO_SECTOR`.

Hypothesis being tested: the GICS-based excess-divergence scorer
underperformed the raw `weights_regime` baseline on Phase-2
(Sharpe 0.99 vs 1.07; CAGR 19% vs 21%). One explanation is that GICS
labels are stale — NVDA in 2005 (GPU graphics card vendor) and NVDA
in 2024 (AI infrastructure megacap) trade nothing alike, but share a
"Information Technology" tag. Empirical clusters from current scalogram
shape adapt across time and should yield a more relevant aggregate.

Scoring math is identical to `excess_divergence_scores`: per-stock CWT
divergence minus its (now empirical) cluster's CWT divergence. The
empirical-cluster aggregate is built per-(date, ticker) and run through
the same `causal_cwt + precompute_windows + get_divergence` pipeline
that the GICS path uses.

Refit cadence
-------------
K-means is refit every `refit_days` bars on the panel of fingerprints
at the refit date. Membership is then carried forward until the next
refit. This stabilizes cluster assignments within a year while letting
them adapt across decades. The refit at date `t` only sees fingerprints
through date `t` (causality: fingerprints are slices of the **causal**
CWT scalogram). Tickers with non-finite fingerprints at a refit date
are masked out of the fit and inherit the previous segment's
assignment (or the nearest-centroid projection if no prior segment).

Degenerate cluster case
-----------------------
Same as GICS Energy = {XOM} on Phase-2: a cluster with a single ticker
produces an aggregate identical to that ticker, so the excess score
collapses to exactly 0 for that name. Clusters can also end up empty
(rare with k <= n_tickers / 2) — we fall back to the raw stock score
for tickers without a populated cluster aggregate at that date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ss_indicators import get_divergence
from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_wavelets import precompute_windows

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def _refit_cluster_assignments(
    fps: np.ndarray,
    *,
    k_clusters: int,
    refit_days: int,
    lookback: int,
    random_state: int = 0,
) -> np.ndarray:
    """Per-date cluster assignment matrix.

    Parameters
    ----------
    fps : np.ndarray, shape `(n_dates, n_tickers, fp_dim)`
        Output of `extract_fingerprints`. Causal: fingerprint at date
        `t` only sees CWT slices through date `t`.
    k_clusters : int
        Number of empirical sectors.
    refit_days : int
        Refit cadence in bars. K-means is fit on fingerprints AT date
        `t_refit` (not over a window — single cross-section), and the
        assignment is carried forward to the next refit.
    lookback : int
        First refit happens at date index `lookback` (the first date
        with full CWT history). Earlier dates get assignment -1.
    random_state : int
        Passed to `KMeans` for reproducible centroids.

    Returns
    -------
    cluster_ids : np.ndarray, shape `(n_dates, n_tickers)`, int64
        Each entry is the cluster id (0 to k-1) for that (date, ticker)
        pair. Values are -1 for dates before `lookback` (no CWT
        history) and for tickers with non-finite fingerprints at the
        most recent refit (with no prior assignment to inherit).
    """
    n_dates, n_tickers, _ = fps.shape
    cluster_ids = np.full((n_dates, n_tickers), -1, dtype=np.int64)

    # Refit dates: lookback, lookback+refit_days, lookback+2*refit_days, ...
    refit_dates = list(range(lookback, n_dates, refit_days))
    if not refit_dates:
        return cluster_ids

    last_assignment = np.full(n_tickers, -1, dtype=np.int64)
    last_centroids: np.ndarray | None = None

    for seg_idx, t_refit in enumerate(refit_dates):
        fps_t = fps[t_refit]                                 # (n_tickers, fp_dim)
        finite_mask = np.isfinite(fps_t).all(axis=1)
        finite_idx = np.where(finite_mask)[0]

        new_assignment = last_assignment.copy()

        if finite_idx.size >= k_clusters:
            X = fps_t[finite_idx]
            km = KMeans(
                n_clusters=k_clusters,
                n_init=10,
                random_state=random_state,
            )
            labels = km.fit_predict(X)
            new_assignment[finite_idx] = labels.astype(np.int64)
            last_centroids = km.cluster_centers_.astype(np.float32)

            # Tickers with non-finite fingerprints: keep previous
            # assignment if we have one, otherwise leave -1.
            non_finite_idx = np.where(~finite_mask)[0]
            for i in non_finite_idx:
                if last_assignment[i] < 0:
                    new_assignment[i] = -1
        elif last_centroids is not None and finite_idx.size > 0:
            # Not enough finite fingerprints to refit, but we have prior
            # centroids: project the finite tickers onto them.
            X = fps_t[finite_idx]
            d = np.linalg.norm(
                X[:, None, :] - last_centroids[None, :, :], axis=-1)
            new_assignment[finite_idx] = np.argmin(d, axis=1).astype(np.int64)
        # else: no centroids and not enough finite fps → carry forward
        # last_assignment as-is (which is all -1 on the first segment).

        # Fill cluster_ids from this refit until the next one.
        seg_end = (refit_dates[seg_idx + 1]
                   if seg_idx + 1 < len(refit_dates) else n_dates)
        cluster_ids[t_refit:seg_end, :] = new_assignment[None, :]
        last_assignment = new_assignment

    return cluster_ids


def _build_cluster_aggregate_prices(
    prices: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Per-(date, ticker) cluster-aggregate price matrix.

    For each (t, i): mean of `prices[t, j]` over all tickers `j` whose
    cluster id equals `cluster_ids[t, i]`. Tickers with cluster id -1
    fall back to their own price (so the excess score collapses to 0
    for that cell — same as the degenerate single-constituent GICS
    case).

    Parameters
    ----------
    prices : np.ndarray, shape `(n_dates, n_tickers)`
    cluster_ids : np.ndarray, shape `(n_dates, n_tickers)`, int64

    Returns
    -------
    agg : np.ndarray, shape `(n_dates, n_tickers)`, same dtype as prices
    """
    n_dates, n_tickers = prices.shape
    agg = np.empty_like(prices, dtype=np.float64)

    # Vectorize over dates by detecting refit segments — within a
    # segment, cluster_ids[t] is constant in t, so the aggregation is
    # the same matrix multiply repeated across the segment's rows.
    # We detect segment boundaries by comparing successive rows of
    # cluster_ids.
    seg_start = 0
    for t in range(1, n_dates + 1):
        if t == n_dates or not np.array_equal(
                cluster_ids[t], cluster_ids[seg_start]):
            ids = cluster_ids[seg_start]              # (n_tickers,)
            seg_prices = prices[seg_start:t]          # (seg_len, n_tickers)

            # Build per-ticker aggregate via a (n_tickers, n_tickers)
            # membership matrix M where M[i, j] = 1 / count(cluster of i)
            # if cluster_ids[i] == cluster_ids[j] and >= 0, else 0.
            # Fallback: if cluster_ids[i] < 0, M[i, i] = 1 (own price).
            M = np.zeros((n_tickers, n_tickers), dtype=np.float64)
            for c in np.unique(ids):
                if c < 0:
                    continue
                members = np.where(ids == c)[0]
                M[np.ix_(members, members)] = 1.0 / members.size
            for i in np.where(ids < 0)[0]:
                M[i, i] = 1.0

            agg[seg_start:t] = seg_prices @ M.T
            seg_start = t

    return agg.astype(prices.dtype, copy=False)


def empirical_excess_divergence_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    scales: list[int],
    divergence: str = 'kl',
    k_clusters: int = 11,
    fp_window: int = 21,
    refit_days: int = 252,
    cache_dir=None,
    return_clusters: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-(date, ticker) excess-divergence scores using empirical
    clusters instead of GICS sectors.

    Parameters mirror `excess_divergence_scores` in the GICS path,
    plus the empirical knobs.

    Parameters
    ----------
    k_clusters : int
        Number of empirical sectors. Default 11 mirrors the GICS sector
        count; for small universes (e.g. Phase-2's 21 tickers) try
        k=4 or k=8 — k=11 averages 2 tickers per cluster which is
        statistically thin.
    fp_window : int
        Bars per fingerprint (matches `relational.diversify`).
    refit_days : int
        K-means refit cadence in bars. Default 252 ~= 1 trading year.

    Returns
    -------
    scores : np.ndarray, shape `(n_eval_dates, n_tickers)`
    cluster_ids : np.ndarray (only if `return_clusters=True`),
        shape `(n_eval_dates, n_tickers)`. Useful for diagnostics.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)

    cluster_ids = _refit_cluster_assignments(
        fps, k_clusters=k_clusters, refit_days=refit_days,
        lookback=lookback)

    # Per-stock CWT divergence (same recipe as scoring.py).
    stock_power = (coeffs ** 2).astype(np.float32)
    stock_recent, stock_hist = precompute_windows(
        stock_power, lookback, n_tail)

    # Empirical-cluster aggregate prices (per-(date, ticker)) → CWT
    # divergence on identical scales. We carry the per-ticker
    # broadcast through the CWT step instead of building (n_dates, k)
    # cluster series and indexing back, because cluster identities
    # re-shuffle at refit boundaries (id 3 in segment 1 has nothing in
    # common with id 3 in segment 2). Per-(date, ticker) aggregation
    # sidesteps the identity-matching problem entirely.
    agg_prices = _build_cluster_aggregate_prices(
        prices.values, cluster_ids)
    agg_coeffs = load_or_compute_cwt(
        pd.DataFrame(
            agg_prices, index=prices.index, columns=prices.columns),
        scales, lookback, cache_dir=cache_dir)
    agg_power = (agg_coeffs ** 2).astype(np.float32)
    agg_recent, agg_hist = precompute_windows(agg_power, lookback, n_tail)

    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    stock_div = np.asarray(
        div_fn(stock_recent, stock_hist, scale_log_weights))
    agg_div = np.asarray(
        div_fn(agg_recent, agg_hist, scale_log_weights))

    excess = np.array(stock_div - agg_div, copy=True)

    if return_clusters:
        return excess, cluster_ids[lookback:]
    return excess


def weights_excess_regime_empirical(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    k_clusters: int = 11,
    fp_window: int = 21,
    refit_days: int = 252,
    cache_dir=None,
) -> pd.DataFrame:
    """Drop-in replacement for `weights_excess_regime` that uses
    empirical k-means clusters of scalogram fingerprints instead of
    static GICS sectors.

    Output is the same shape and convention as `weights_regime` /
    `weights_excess_regime`: a `(n_eval_dates, n_tickers)` DataFrame
    of one-hot top-N selection rows.
    """
    scores = empirical_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        divergence=divergence, k_clusters=k_clusters,
        fp_window=fp_window, refit_days=refit_days,
        cache_dir=cache_dir)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)
