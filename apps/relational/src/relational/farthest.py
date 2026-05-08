"""Idea C — rank tickers by distance from the cross-sectional centroid.

For each rebalance date `t`, compute the fingerprint cloud over all
tickers, take its mean as the "average market state right now," and
score each ticker by L2 distance to that mean. Pick the top-N most
distant — the names doing the most idiosyncratic thing relative to
the universe.

This is orthogonal to `weights_regime` along an interesting axis:

  - `weights_regime`   = TEMPORAL divergence: how much has each ticker's
                         scalogram shifted relative to ITS OWN past?
  - `weights_regime_farthest` = CROSS-SECTIONAL divergence: how unusual
                                is each ticker relative to the universe
                                RIGHT NOW?

The former is "regime shift since this stock's history"; the latter is
"market outlier today." They can disagree — a stock that's moved a lot
but everyone else has too will rank high in baseline and low in idea C.

Distance is L2 over unit-norm flattened scalogram windows (see
`relational.fingerprints`). The fingerprint is z-normalized so
distance is direction-of-shape rather than amplitude — a high-vol
ticker doesn't automatically rank highest just because its raw CWT
coefficients are larger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_features import Compression
from ss_portfolio import apply_nan_mask, select_top_n_matrix

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def centroid_distance_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    cache_dir=None,
    compression: Compression | None = None,
) -> np.ndarray:
    """Per-(date, ticker) cross-sectional centroid distance.

    Returns a `(n_eval, n_tickers)` float32 array where `n_eval =
    n_dates - lookback`. NaN cells correspond to tickers with
    incomplete CWT/fingerprint history at that date — handled
    downstream by `apply_nan_mask` + `select_top_n_matrix`.

    The centroid at date `t` is the mean fingerprint over the
    *finite-fingerprint* tickers only — early in the panel when many
    tickers have NaN, dropped tickers don't pollute the mean.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(
        coeffs, w=fp_window, znorm=True, compression=compression)
    fps_eval = fps[lookback:]    # (n_eval, n_tickers, fp_dim)

    n_eval, n_tickers, _ = fps_eval.shape
    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)

    # Per-date loop: cheap on Phase-2 (n_eval ~3K, n_tickers ~21) and
    # also fine on stooq_us_long (n_eval ~6K, n_tickers ~312). Could
    # vectorize via masked-array mean but the loop body is already
    # ~100 µs per row.
    for t in range(n_eval):
        row = fps_eval[t]
        finite_mask = np.isfinite(row).all(axis=1)
        if finite_mask.sum() < 2:
            continue
        centroid = row[finite_mask].mean(axis=0)
        d = np.linalg.norm(row - centroid[None, :], axis=-1)
        scores[t, finite_mask] = d[finite_mask].astype(np.float32, copy=False)

    return scores


def weights_regime_farthest(
    prices: pd.DataFrame,
    *,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int = 21,
    cache_dir=None,
    compression: Compression | None = None,
) -> pd.DataFrame:
    """Hard-top-N basket ranked by cross-sectional centroid distance.

    Drop-in replacement for `weights_regime` that scores by
    fingerprint-space outlier-ness instead of recent-vs-historical CWT
    divergence. The signature is intentionally narrower — `n_tail` and
    `divergence` don't apply because there's no temporal-window or
    divergence-family choice to make in this scoring family.

    Output is the same shape as `weights_regime`: a
    `(n_dates - lookback, n_tickers)` one-hot DataFrame, equal-weighted
    over the chosen `top_n`.
    """
    scores = centroid_distance_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, cache_dir=cache_dir,
        compression=compression)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
