"""Idea D — diversified selection by greedy farthest-first thinning.

The baseline `weights_regime` ranks by per-stock CWT divergence and
takes the top-N. The failure mode (apps/docs/docs/notes.md "Multi-stock
CWT framings"): "we
just bought 5 tech names because tech moved most." This module
addresses that at the *selection* layer (not the score layer):

  1. Take the top `top_pool` names by raw divergence score (a wider
     net than the final basket).
  2. Greedy farthest-first thin to `k_keep`:
       - Always keep the highest-scoring name.
       - Repeatedly add the still-eligible candidate whose minimum
         scalogram-fingerprint distance to the already-picked set is
         largest.
  3. Equal-weight the kept names (mirrors `select_top_n_matrix`).

`top_pool >= k_keep`. Setting `top_pool == k_keep` degenerates to the
baseline (no thinning to do); setting `top_pool >> k_keep` lets the
farthest-first pass override low-quality-but-distinctive picks.

The fingerprint distance is L2 over unit-norm flattened scalogram
windows (see `relational.fingerprints`). For unit-norm vectors, L2 is
monotone in cosine, so this is direction-of-scalogram-shape distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_features import Compression
from ss_indicators import get_divergence
from ss_portfolio import apply_nan_mask
from ss_wavelets import precompute_windows

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def _greedy_farthest_first(
    scores: np.ndarray,        # (n_tickers,) — higher = better
    distances: np.ndarray,     # (n_tickers, n_tickers) — pairwise fp distance
    *,
    top_pool: int,
    k_keep: int,
) -> np.ndarray:
    """Return indices of `k_keep` selected tickers, in pick order.

    Picks the top-scoring name first, then iteratively the candidate
    in the score-top-`top_pool` whose minimum distance to the
    already-picked set is largest. NaN scores are excluded from the
    candidate pool. Returns fewer than `k_keep` indices if the pool
    has fewer than `k_keep` finite entries.
    """
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        return np.array([], dtype=np.int64)

    # Score-rank the finite candidates and take the top `top_pool`.
    order = finite[np.argsort(-scores[finite])]
    pool = order[:max(top_pool, k_keep)]
    if pool.size <= k_keep:
        return pool

    picks: list[int] = [int(pool[0])]
    remaining = list(pool[1:])
    while len(picks) < k_keep and remaining:
        # Min distance from each remaining candidate to the picked set.
        dmat = distances[np.ix_(remaining, picks)]
        min_d = dmat.min(axis=1)
        # Tie-break by score (already sorted descending in `remaining`).
        next_idx = int(np.argmax(min_d))
        picks.append(int(remaining[next_idx]))
        remaining.pop(next_idx)
    return np.asarray(picks, dtype=np.int64)


def weights_regime_diversified(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    k_keep: int,
    top_pool: int,
    scales: list[int],
    divergence: str = 'kl',
    fp_window: int = 21,
    cache_dir=None,
    compression: Compression | None = None,
) -> pd.DataFrame:
    """`weights_regime` + greedy farthest-first thinning on scalogram
    fingerprints. Returns one-hot equal-weight DataFrame of shape
    `(n_dates - lookback, n_tickers)`, `k_keep` ones per row.

    Parameters
    ----------
    k_keep : int
        Final basket size (mirrors baseline `top_n`).
    top_pool : int
        Number of top-scoring names considered before thinning.
        Must be `>= k_keep`. `top_pool == k_keep` degenerates to baseline.
    fp_window : int
        Bars per fingerprint (default 21 = 1 trading month).
    """
    if top_pool < k_keep:
        raise ValueError(
            f'top_pool ({top_pool}) must be >= k_keep ({k_keep})')

    # --- raw divergence scores (same as weights_regime) ---
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    power = (coeffs ** 2).astype(np.float32)
    recent, hist = precompute_windows(power, lookback, n_tail)

    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    scores = np.array(
        div_fn(recent, hist, scale_log_weights), copy=True)
    scores = apply_nan_mask(scores, prices.values, lookback)

    # --- fingerprints for the same dates ---
    # `coeffs` is over all `n_dates` rows; the score matrix is over
    # rows `lookback:`. Slice fingerprints accordingly.
    fps = extract_fingerprints(
        coeffs, w=fp_window, znorm=True, compression=compression)
    fps = fps[lookback:]   # (n_eval, n_tickers, fp_dim)

    n_eval, n_tickers = scores.shape
    weights = np.zeros((n_eval, n_tickers), dtype=np.float32)
    for i in range(n_eval):
        s_row = scores[i]
        fp_row = fps[i]
        # Pairwise distances on demand — n_tickers ~21 makes this trivial.
        diff = fp_row[:, None, :] - fp_row[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        picks = _greedy_farthest_first(
            s_row, dist, top_pool=top_pool, k_keep=k_keep)
        if picks.size:
            weights[i, picks] = 1.0 / picks.size

    return pd.DataFrame(
        weights,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
