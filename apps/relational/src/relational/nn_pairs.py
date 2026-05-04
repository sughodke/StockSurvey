"""Per-pick nearest-neighbor pair construction (idea: trade the
behavioral spread).

For each rebalance date `t` and each long pick `i` in the top-N basket,
we identify `i`'s **closest behavioral peer** among the tickers *not*
already in the top-N, by squared-L2 distance over (already unit-normed)
fingerprints. We then short that peer at the same per-name notional.

This differs from the prior pair-trade overlays:

  * `mkt-neutral` shorts the universe equal-weight — coarse, removes
    factor exposure indiscriminately.
  * `rank-spread` shorts the bot-N (lowest-score names) — symmetric in
    the scorer, but the bot-N has no behavioral relationship to the
    long pick.
  * `cluster-pair`  shorts each pick's k-means cluster aggregate —
    coarser than nearest-neighbor (group-level rather than per-name).

`nn-pair` is the tightest construction: each long has its own most-
similar non-pick partner, chosen freshly per rebalance from current
fingerprint distance. The hedge is interpretable as "the dislocator
vs its closest non-dislocator peer."

Distance metric: squared L2 on fingerprints. With `znorm=True` upstream
(the default in `extract_fingerprints`), squared L2 is monotone in
cosine distance, so the rank of "closest" is identical to cosine-NN.

Edge cases:
  * If the long itself slips into the candidate pool (it shouldn't —
    we mask it out — but defense in depth), we exclude `j == i` by
    setting its distance to +∞ before argmin.
  * If a candidate's fingerprint is NaN at date `t`, its distance is
    +∞ so it's skipped naturally; we fall through to the second-nearest.
  * If no candidate has a finite fingerprint, the partner is `-1` and
    the row contributes only the long leg. Rare but handled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def nearest_non_top_partner(
    fps_eval: np.ndarray,
    top_weights: np.ndarray | pd.DataFrame,
    *,
    distance: str = 'sq_l2',
) -> np.ndarray:
    """Per-(date, ticker) nearest non-top-N hedge partner.

    Parameters
    ----------
    fps_eval : np.ndarray, shape `(n_eval, n_tickers, fp_dim)`
        Fingerprint cube aligned to the same eval index as
        `top_weights`. Typically `extract_fingerprints(coeffs)[lookback:]`.
        With `znorm=True` upstream (recommended), squared L2 is monotone
        in cosine distance.
    top_weights : np.ndarray | pd.DataFrame, shape `(n_eval, n_tickers)`
        Long-only top-N weight matrix. We treat any strictly-positive
        cell as "in the top-N." Equal-weighted +1/N or any mass shape
        works — we only read the support, not the values.
    distance : {'sq_l2'}
        Distance metric. Only squared L2 is implemented (matches
        `relational.farthest`'s convention on unit-norm fingerprints).

    Returns
    -------
    nn : np.ndarray, shape `(n_eval, n_tickers)`, int32
        `nn[t, i] = j` means at date `t`, the long pick `i`'s nearest
        non-top-N hedge partner is ticker `j`. Off-diagonal: if `i` is
        not in the top-N at date `t`, `nn[t, i] = -1`. Same sentinel
        when no valid partner exists (NaN-only candidates, degenerate
        universes).
    """
    if distance != 'sq_l2':
        raise ValueError(f'unsupported distance metric: {distance}')

    if isinstance(top_weights, pd.DataFrame):
        top_arr = top_weights.fillna(0.0).values
    else:
        top_arr = np.nan_to_num(top_weights, nan=0.0)

    n_eval, n_tickers = top_arr.shape
    if fps_eval.shape[:2] != top_arr.shape:
        raise ValueError(
            f'fps_eval/top_weights shape mismatch: '
            f'{fps_eval.shape[:2]} vs {top_arr.shape}')

    nn = np.full((n_eval, n_tickers), -1, dtype=np.int32)

    for t in range(n_eval):
        in_top = top_arr[t] > 0.0
        if not in_top.any():
            continue
        row = fps_eval[t]                         # (n_tickers, fp_dim)
        finite_mask = np.isfinite(row).all(axis=1)
        # Candidate pool: finite fingerprint AND not in top-N. The top-N
        # mask must take precedence even when a long-pick's fingerprint
        # is finite (we never want a pick to short itself).
        cand_mask = finite_mask & ~in_top
        if not cand_mask.any():
            # No valid partner today — leave row as -1.
            continue

        long_idx = np.where(in_top)[0]
        # Only longs with finite fingerprints can compute distances; if
        # a long has NaN fp, we can't measure similarity. Leave -1.
        long_finite = long_idx[finite_mask[long_idx]]
        if long_finite.size == 0:
            continue

        # Vectorized squared-L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b.
        # Fingerprints are unit-norm in the standard pipeline so
        # ||a||^2 = ||b||^2 ≈ 1, but we don't rely on that — compute
        # the full quadratic form so this stays correct for non-znorm
        # fingerprints too.
        longs = row[long_finite]                  # (L, fp_dim)
        cands = row[cand_mask]                    # (C, fp_dim)
        cand_idx = np.where(cand_mask)[0]
        long_sq = (longs * longs).sum(axis=1, keepdims=True)
        cand_sq = (cands * cands).sum(axis=1)[None, :]
        dot = longs @ cands.T
        dists = long_sq + cand_sq - 2.0 * dot     # (L, C)
        nn_local = dists.argmin(axis=1)
        nn[t, long_finite] = cand_idx[nn_local].astype(np.int32)

    return nn


def nearest_neighbor_pair_weights(
    top_weights: pd.DataFrame,
    fps_eval: np.ndarray,
    *,
    distance: str = 'sq_l2',
) -> pd.DataFrame:
    """Long top-N at +1/N, short each pick's nearest non-top peer at
    -1/N.

    Output is a `(n_eval, n_tickers)` DataFrame in the standard weight
    matrix shape contract (matching `weights_regime` /
    `market_neutral_weights`): per row `sum(weights) ≈ 0` and
    `sum(|weights|) ≈ 2`. If two picks share the same nearest neighbor
    that neighbor's short weight stacks to `-2/N` — natural concentration,
    documented as a feature not a bug.

    Parameters
    ----------
    top_weights : pd.DataFrame, shape `(n_eval, n_tickers)`
        Long-only top-N weight matrix. Index aligns to the eval window
        (i.e. `prices.index[lookback:]` in the standard pipeline).
    fps_eval : np.ndarray, shape `(n_eval, n_tickers, fp_dim)`
        Fingerprint cube aligned to the same eval index. Easiest source:
        `fingerprints_for_weights(prices, top_weights, scales=...,
        lookback=..., fp_window=...)`.
    distance :
        Forwarded to `nearest_non_top_partner`. Only `'sq_l2'` is
        implemented today.

    Returns
    -------
    weights : pd.DataFrame, same index/columns as `top_weights`.
    """
    nn = nearest_non_top_partner(
        fps_eval, top_weights, distance=distance)
    long_arr = top_weights.fillna(0.0).values.astype(np.float64, copy=True)
    out = long_arr.copy()
    n_eval, n_tickers = long_arr.shape
    for t in range(n_eval):
        in_top = long_arr[t] > 0.0
        if not in_top.any():
            continue
        long_idx = np.where(in_top)[0]
        # Use each long's own +1/N notional as its short notional. This
        # preserves balance even if rows are not strictly equal-weight
        # (e.g. if the top-N is later softmax-allocated).
        for i in long_idx:
            j = int(nn[t, i])
            if j < 0:
                continue
            out[t, j] -= long_arr[t, i]
    return pd.DataFrame(
        out, index=top_weights.index, columns=top_weights.columns)


def fingerprints_for_weights(
    prices: pd.DataFrame,
    top_weights: pd.DataFrame,
    *,
    scales: list[int],
    lookback: int,
    fp_window: int = 21,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Helper: extract per-(date, ticker) fingerprints aligned to the
    eval index of `top_weights`.

    The standard pipeline produces `top_weights` with index
    `prices.index[lookback:]`. This helper computes the causal CWT
    over `prices` (cache-aware) and returns the corresponding eval
    slice of `extract_fingerprints(...)` so `(fps_eval, top_weights)`
    pair up directly for `nearest_neighbor_pair_weights`.

    Parameters mirror `centroid_distance_scores` so callers can swap
    in the same tuple.

    Returns
    -------
    fps_eval : np.ndarray, shape `(len(top_weights), n_tickers, S*w)`
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    fps_eval = fps[lookback:]
    if fps_eval.shape[0] != top_weights.shape[0]:
        raise ValueError(
            f'eval-index length mismatch: fps_eval={fps_eval.shape[0]} '
            f'rows vs top_weights={top_weights.shape[0]} rows. Did the '
            f'caller pass a `top_weights` built from a different '
            f'`prices` slice?')
    return fps_eval
