"""Idea A — empirical sectors via Gaussian Mixture Model on scalogram
fingerprints (soft cluster membership variant).

Drop-in soft-cluster sibling of `relational.empirical_sectors`. Same
scoring math (per-stock CWT divergence minus its sector aggregate's
CWT divergence), same refit cadence, same Hungarian-stabilization
across refits — but the sector membership is a *posterior distribution*
over `n_components` clusters instead of a hard cluster id.

Why a soft assignment
---------------------
The hard k-means version produced jittery cluster aggregates on
boundary tickers — a stock that sits near two centroids flips between
them every refit, which on top of the per-segment refit-jitter creates
a noisy aggregate price series. With a GMM, each ticker on each refit
gets posterior probabilities `p[t, i, k]` over the `k` Gaussian
components, and the aggregate is computed as a *posterior-weighted*
mix of cluster means rather than a hard membership lookup.

Three places this changes things vs. `empirical_sectors.py`:

1. **Cluster aggregate** (`_build_soft_cluster_aggregate_prices`):
   instead of `agg[t, i] = mean(prices[t, j] for j where cluster_id[j] == c(i))`,
   use `agg[t, i] = sum_k p[t, i, k] * cluster_mean_price[t, k]`. The
   cluster mean prices are the (NaN-safe) per-cluster equal-weight
   means of the constituents (using MAP / argmax-posterior assignment
   to define membership for the cluster *means*, but the per-ticker
   aggregate is the soft mix of those means). This way a boundary
   ticker's aggregate moves smoothly between two cluster regimes
   instead of flipping discretely.

2. **Cluster-pair weights** (`gmm_cluster_pair_weights`): with hard
   k-means, each cluster has one winner and shorts equal-weight peers.
   With GMM, we keep the same construction but assign tickers to
   clusters via *MAP posterior* (argmax) so the long/short
   construction has a clean cluster definition; the smoothing benefit
   is in the soft aggregate that drives the score itself.

3. **Score itself**: the divergence is `stock_div − soft_aggregate_div`
   where `soft_aggregate` is the personalized posterior-weighted mix
   of cluster aggregates per ticker.

GMM choice — `covariance_type='diag'` keeps params manageable
(`n_components × fp_dim` means + `n_components × fp_dim` variances
~= 11×168×2 ≈ 3700 params) on the typical 168-dim fingerprint at
n=11 components. Random_state pinned for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from ss_indicators import get_divergence
from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_wavelets import precompute_windows

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def _refit_gmm_assignments(
    fps: np.ndarray,
    *,
    n_components: int,
    refit_days: int,
    lookback: int,
    random_state: int = 42,
) -> np.ndarray:
    """Per-(date, ticker) GMM posterior over `n_components` clusters.

    Mirrors `_refit_cluster_assignments` from the k-means version but
    returns a `(n_dates, n_tickers, n_components)` posterior tensor
    instead of a `(n_dates, n_tickers)` int label matrix.

    Refit cadence: every `refit_days` bars starting at index `lookback`.
    Tickers with non-finite fingerprints inherit the previous
    segment's posterior (or a uniform fallback if no prior segment
    exists). Pre-lookback dates get NaN posteriors (caller masks).

    Hungarian-matched stabilization across refits: between adjacent
    segments, the cost matrix `||old_means_k - new_means_j||²` is
    solved with `linear_sum_assignment`; the new segment's component
    columns (and posterior columns) are permuted so component `k` in
    segment `s+1` is the continuation of component `k` in segment `s`.
    Without this, segment 1's "component 3" has nothing in common with
    segment 2's "component 3" — see `relational.cluster_tracking` for
    the same concern in the k-means path.

    Parameters
    ----------
    fps : np.ndarray, shape `(n_dates, n_tickers, fp_dim)`
        Causal fingerprints from `extract_fingerprints`.
    n_components : int
        Number of GMM components (the analog of `k_clusters`).
    refit_days, lookback : int
        Same semantics as the k-means refit helper.
    random_state : int
        Seed for `GaussianMixture` so refits are reproducible.

    Returns
    -------
    posteriors : np.ndarray, shape `(n_dates, n_tickers, n_components)`,
        float32. Sums to 1 along axis=2 for valid cells; NaN for
        pre-lookback dates and for tickers with no prior segment whose
        fingerprint is non-finite at the first refit.
    """
    n_dates, n_tickers, fp_dim = fps.shape
    posteriors = np.full(
        (n_dates, n_tickers, n_components), np.nan, dtype=np.float32)

    refit_dates = list(range(lookback, n_dates, refit_days))
    if not refit_dates:
        return posteriors

    last_post = np.full(
        (n_tickers, n_components), np.nan, dtype=np.float32)
    last_means: np.ndarray | None = None  # `(n_components, fp_dim)` float64

    for seg_idx, t_refit in enumerate(refit_dates):
        fps_t = fps[t_refit]                              # (n_tickers, fp_dim)
        finite_mask = np.isfinite(fps_t).all(axis=1)
        finite_idx = np.where(finite_mask)[0]

        new_post = last_post.copy()

        if finite_idx.size >= n_components:
            X = fps_t[finite_idx]
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                random_state=random_state,
                reg_covar=1e-4,    # extra reg in case of tight clusters
                max_iter=200,
            )
            gmm.fit(X)
            probs = gmm.predict_proba(X).astype(np.float32)
            new_means = gmm.means_.astype(np.float64)

            # Hungarian-match new component IDs to the previous segment's
            # component identity space. Cost is `||old_means − new_means||²`.
            if last_means is not None:
                cost = np.full(
                    (n_components, n_components), 1e9, dtype=np.float64)
                for o in range(n_components):
                    if not np.isfinite(last_means[o]).all():
                        continue
                    for k in range(n_components):
                        if not np.isfinite(new_means[k]).all():
                            continue
                        d = last_means[o] - new_means[k]
                        cost[o, k] = float(np.dot(d, d))
                old_idx, new_idx = linear_sum_assignment(cost)
                # `mapping[new_id] = old_id` — re-index columns of probs
                # and rows of new_means.
                mapping = np.empty(n_components, dtype=np.int64)
                for o, n in zip(old_idx, new_idx):
                    mapping[n] = o
                # Rebuild probs/means under the old-ID space:
                # column k of new probs becomes column mapping[k] of stable.
                stable_probs = np.zeros_like(probs)
                stable_probs[:, mapping] = probs
                stable_means = np.empty_like(new_means)
                stable_means[mapping] = new_means
                probs = stable_probs
                new_means = stable_means

            new_post[finite_idx] = probs
            last_means = new_means

            # Tickers with non-finite fingerprints: inherit last_post if
            # we have one, otherwise leave NaN.
            non_finite_idx = np.where(~finite_mask)[0]
            for i in non_finite_idx:
                if not np.isfinite(last_post[i]).all():
                    new_post[i] = np.nan
                # else: already copied via new_post = last_post.copy()
        elif last_means is not None and finite_idx.size > 0:
            # Not enough finite fps to refit, but we have prior means:
            # project finite tickers to nearest mean (hard) and emit a
            # one-hot posterior. Soft-projection via the prior GMM would
            # need the full `precompute` machinery; one-hot is fine here.
            X = fps_t[finite_idx]
            d = np.linalg.norm(
                X[:, None, :] - last_means[None, :, :], axis=-1)
            hard = np.argmin(d, axis=1)
            one_hot = np.zeros(
                (finite_idx.size, n_components), dtype=np.float32)
            one_hot[np.arange(finite_idx.size), hard] = 1.0
            new_post[finite_idx] = one_hot
        # else: no last_means and not enough finite fps; carry forward
        # last_post (NaN on first segment).

        seg_end = (refit_dates[seg_idx + 1]
                   if seg_idx + 1 < len(refit_dates) else n_dates)
        posteriors[t_refit:seg_end, :, :] = new_post[None, :, :]
        last_post = new_post

    return posteriors


def _build_soft_cluster_aggregate_prices(
    prices: np.ndarray,
    posteriors: np.ndarray,
) -> np.ndarray:
    """Per-(date, ticker) soft-cluster-aggregate prices.

    For each (t, i):
        agg[t, i] = sum_k posteriors[t, i, k] * cluster_mean_price[t, k]

    where `cluster_mean_price[t, k]` is the equal-weight, NaN-safe mean
    of all *finite* prices on date `t` whose **MAP** assignment is
    cluster `k`. (We use MAP for the cluster-membership-of-mean step
    and posterior weights for the per-ticker mix step. Pure expected-
    cluster-membership weighting would create fixed-point dependencies
    where the mean depends on the weights and vice versa; MAP for
    membership avoids that while keeping the per-ticker aggregate soft.)

    NaN handling mirrors `_build_cluster_aggregate_prices`:
    - rows where the sum of posterior weights × cluster_mean is NaN
      (e.g. all clusters empty or a posterior column is NaN) fall back
      to the ticker's own price so the excess score collapses to 0.
    - tickers with NaN posteriors (pre-lookback or unmatched
      non-finite-fp on first segment) fall back to own price.

    Parameters
    ----------
    prices : np.ndarray, shape `(n_dates, n_tickers)`
    posteriors : np.ndarray, shape `(n_dates, n_tickers, n_components)`

    Returns
    -------
    agg : np.ndarray, shape `(n_dates, n_tickers)`, same dtype as prices
    """
    n_dates, n_tickers = prices.shape
    n_components = posteriors.shape[2]
    agg = np.empty_like(prices, dtype=np.float64)

    # Scan in segments where posteriors are constant (refit boundaries).
    # Within a segment, posteriors[t, :, :] is the same for all t in the
    # segment, so we can compute MAP labels once per segment but the
    # per-row NaN-safe cluster means change daily because of price NaNs.
    seg_start = 0
    for t in range(1, n_dates + 1):
        same = (
            t < n_dates
            and np.array_equal(posteriors[t], posteriors[seg_start],
                               equal_nan=True)
        )
        if same:
            continue
        post_seg = posteriors[seg_start]                    # (n_tickers, K)
        seg_prices = prices[seg_start:t]                    # (seg_len, n_tickers)
        seg_finite = np.isfinite(seg_prices)
        seg_agg = np.empty_like(seg_prices, dtype=np.float64)

        # Per-ticker "valid posterior" mask — all-finite K entries.
        post_valid = np.isfinite(post_seg).all(axis=1)      # (n_tickers,)

        # MAP assignment for cluster-mean membership.
        map_ids = np.full(n_tickers, -1, dtype=np.int64)
        if post_valid.any():
            map_ids[post_valid] = np.argmax(
                post_seg[post_valid], axis=1).astype(np.int64)

        # Per-row, per-cluster mean (NaN-safe).
        # `cluster_mean[t, k]` shape `(seg_len, K)`.
        cluster_mean = np.zeros((seg_prices.shape[0], n_components),
                                dtype=np.float64)
        cluster_count = np.zeros((seg_prices.shape[0], n_components),
                                 dtype=np.float64)
        for k in range(n_components):
            members = np.where(map_ids == k)[0]
            if members.size == 0:
                continue
            vals = seg_prices[:, members]
            mask = seg_finite[:, members]
            num = np.where(mask, vals, 0.0).sum(axis=1)
            den = mask.sum(axis=1).astype(np.float64)
            den_safe = np.where(den > 0, den, 1.0)
            cluster_mean[:, k] = num / den_safe
            cluster_count[:, k] = den

        # Soft-aggregate per ticker — only when the posterior row is
        # valid AND at least one cluster has finite members on that
        # row. `(seg_len, n_tickers)`.
        # post_seg[i, :] is a length-K vector; matrix multiply with
        # cluster_mean[t, :] gives the soft aggregate.
        # Where some cluster columns of cluster_mean are zero (no
        # members), the row's posterior weight on that cluster
        # contributes 0 and we down-weight by re-normalizing across
        # populated clusters.
        populated = (cluster_count > 0).astype(np.float64)  # (seg_len, K)
        # Re-normalized posterior per row, per ticker, per cluster.
        # eff_post[t, i, k] = post_seg[i, k] * populated[t, k] /
        #                     sum_k' post_seg[i, k'] * populated[t, k']
        # — a per-(row, ticker) renormalization. We compute the soft mix
        # as `(post_seg @ cluster_mean.T)` divided by the per-row
        # populated-weight normalizer to drop empty clusters cleanly.
        post_safe = np.where(post_valid[:, None], post_seg, 0.0)
        # Soft sum: `(seg_len, n_tickers)` = `(post_safe (n_tickers, K))
        # @ (cluster_mean (seg_len, K).T)` — but we want per-row
        # cluster_mean, so loop transposed:
        # soft_num[t, i] = sum_k post_safe[i, k] * cluster_mean[t, k] * populated[t, k]
        # soft_den[t, i] = sum_k post_safe[i, k] * populated[t, k]
        soft_num = (post_safe * populated[:, None, :]
                    * cluster_mean[:, None, :]).sum(axis=2)
        soft_den = (post_safe * populated[:, None, :]).sum(axis=2)
        # Where soft_den is 0 (no populated cluster has positive
        # posterior weight) or post_valid is False, fall back to own
        # price so the excess score collapses to 0.
        bad = (soft_den == 0) | (~post_valid[None, :])
        seg_agg = np.where(bad, seg_prices, soft_num / np.where(
            soft_den > 0, soft_den, 1.0))

        agg[seg_start:t] = seg_agg
        seg_start = t

    return agg.astype(prices.dtype, copy=False)


def gmm_excess_divergence_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    scales: list[int],
    divergence: str = 'kl',
    n_components: int = 11,
    fp_window: int = 21,
    refit_days: int = 252,
    cache_dir=None,
    return_posteriors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Soft-cluster GMM analog of `empirical_excess_divergence_scores`.

    Returns the same `(n_eval_dates, n_tickers)` excess-divergence
    score matrix as the k-means version, but the cluster aggregate is
    a posterior-weighted mix of GMM-component cluster means rather
    than a hard k-means membership mean. See module docstring.

    Parameters mirror `empirical_excess_divergence_scores` with
    `k_clusters` renamed to `n_components` to match sklearn's GMM API.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)

    posteriors = _refit_gmm_assignments(
        fps, n_components=n_components, refit_days=refit_days,
        lookback=lookback)

    # Per-stock CWT divergence (same recipe as the k-means version).
    stock_power = (coeffs ** 2).astype(np.float32)
    stock_recent, stock_hist = precompute_windows(
        stock_power, lookback, n_tail)

    agg_prices = _build_soft_cluster_aggregate_prices(
        prices.values, posteriors)
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

    if return_posteriors:
        return excess, posteriors[lookback:]
    return excess


def weights_excess_regime_gmm(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    n_components: int = 11,
    fp_window: int = 21,
    refit_days: int = 252,
    cache_dir=None,
) -> pd.DataFrame:
    """Drop-in replacement for `weights_excess_regime_empirical` using
    soft-cluster GMM aggregates instead of hard k-means."""
    scores = gmm_excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        divergence=divergence, n_components=n_components,
        fp_window=fp_window, refit_days=refit_days,
        cache_dir=cache_dir)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


def gmm_cluster_pair_weights(
    scores: np.ndarray,
    posteriors: np.ndarray,
    prices: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    """GMM analog of `relational.pairs.cluster_pair_weights` using MAP
    posterior to define cluster membership for the long/short
    construction.

    For each MAP-cluster `c` on date `t`:
      - long the highest-`scores[t]` ticker among MAP-members of `c`
      - short equal-weight all MAP-members of `c` (cluster aggregate)

    The smoothing benefit of GMM enters through the *score* (which
    drives the long pick) — the score uses the soft aggregate. The
    long/short construction itself is a hard MAP partition because
    the alternative (posterior-weighted longs across all clusters per
    ticker) blows up gross exposure and obscures the comparison to
    the k-means cluster-pair baseline.
    """
    n_eval, n_tickers = scores.shape
    out = np.zeros((n_eval, n_tickers), dtype=np.float64)
    for t in range(n_eval):
        score_t = scores[t]
        post_t = posteriors[t]                       # (n_tickers, K)
        valid_post = np.isfinite(post_t).all(axis=1)
        if not valid_post.any():
            continue
        map_ids = np.full(n_tickers, -1, dtype=np.int64)
        map_ids[valid_post] = np.argmax(
            post_t[valid_post], axis=1).astype(np.int64)
        active: list[tuple[np.ndarray, np.ndarray]] = []
        for c in np.unique(map_ids):
            if c < 0:
                continue
            members = np.where(map_ids == c)[0]
            valid_score = members[np.isfinite(score_t[members])]
            if len(valid_score) < 2:
                continue
            active.append((members, valid_score))
        if not active:
            continue
        w_each = 1.0 / len(active)
        for members, valid_score in active:
            winner = valid_score[np.argmax(score_t[valid_score])]
            cluster_size = len(members)
            out[t, winner] += w_each
            out[t, members] -= w_each / cluster_size
    return pd.DataFrame(
        out,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
