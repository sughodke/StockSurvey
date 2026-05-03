"""Per-(ticker, date) scalogram fingerprints.

A "fingerprint" at date `t` for ticker `i` is a flattened slice of the
causal CWT scalogram over the most recent `w` bars:

    fp[t, i, :] = coeffs[:, t-w+1:t+1, i].reshape(-1)   # shape (S*w,)

This is the input vector for all four relational ideas:
  - clustering ticker-by-ticker (idea A)
  - k-NN over historical fingerprints (idea B)
  - distance-from-centroid (idea C)
  - pairwise diversification (idea D)

We keep this layer thin and numpy-only — no scoring, no ranker logic.
Everything downstream consumes `(n_dates, n_tickers, fp_dim)` arrays
or per-date `(n_tickers, fp_dim)` slices and decides how to use them.

All four ideas share a single `extract_fingerprints` call upstream so
we can compare them on identical fingerprint vectors. The only knob
is `w` (window length, in bars) and `znorm` (per-fingerprint L2
normalization, makes distance metrics scale-invariant across tickers).
"""

from __future__ import annotations

import numpy as np


def extract_fingerprints(
    coeffs: np.ndarray,
    *,
    w: int,
    znorm: bool = True,
) -> np.ndarray:
    """Compute per-(date, ticker) scalogram fingerprints.

    Parameters
    ----------
    coeffs : np.ndarray, shape `(n_scales, n_dates, n_tickers)`
        Output of `ss_wavelets.causal_cwt` (or the cached equivalent
        from `relational.scalogram_cache.load_or_compute_cwt`).
    w : int
        Window length in bars. Each fingerprint stacks the most recent
        `w` CWT vectors → `S*w`-dim vector. Default in the research
        scripts is 21 (one trading month).
    znorm : bool
        If True, each fingerprint is L2-normalized to unit length.
        Makes distance comparisons scale-invariant across tickers
        (otherwise a high-vol ticker's fingerprint just has bigger
        coefficients and looks "far" from everything). Recommended on.

    Returns
    -------
    fps : np.ndarray, shape `(n_dates, n_tickers, S*w)`, float32
        For dates `t < w-1`, the fingerprint is computed against a
        zero-padded window — caller should drop those rows or apply
        the same `lookback` floor used elsewhere.
    """
    n_scales, n_dates, n_tickers = coeffs.shape
    fp_dim = n_scales * w

    # Pad with `w-1` rows of zeros at the front so a sliding window of
    # length `w` ending at `t` is well-defined for every t.
    pad = np.zeros((n_scales, w - 1, n_tickers), dtype=coeffs.dtype)
    padded = np.concatenate([pad, coeffs], axis=1)
    # `(n_scales, n_dates+w-1, n_tickers)` → use stride trick to build
    # `(w, n_dates, n_scales, n_tickers)`. Cheap; avoids a Python loop.
    sw = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=w, axis=1)
    # `sw` shape: `(n_scales, n_dates, n_tickers, w)`. Reorder so each
    # fingerprint is `(S, w)` flattened in scale-major order.
    fps = np.transpose(sw, (1, 2, 0, 3)).reshape(
        n_dates, n_tickers, fp_dim).astype(np.float32, copy=False)

    if znorm:
        norms = np.linalg.norm(fps, axis=-1, keepdims=True)
        fps = fps / np.maximum(norms, 1e-8)
    return fps


def cross_sectional_centroid(fps_t: np.ndarray) -> np.ndarray:
    """Mean fingerprint across tickers at a single date.

    `fps_t` is `(n_tickers, fp_dim)` for one date; returns `(fp_dim,)`.
    Used by idea C (farthest-from-centroid) and as a baseline for
    idea A's empirical sectoring.
    """
    return fps_t.mean(axis=0)


def pairwise_distances(fps_t: np.ndarray) -> np.ndarray:
    """L2 pairwise distance matrix at a single date.

    `fps_t` is `(n_tickers, fp_dim)`; returns `(n_tickers, n_tickers)`,
    symmetric, zero-diagonal. Used by idea D (diversified selection).

    For unit-norm fingerprints (`znorm=True`), L2 distance is
    monotone in cosine distance, so this is interchangeable with
    cosine for ranking purposes.
    """
    diff = fps_t[:, None, :] - fps_t[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def distance_to_centroid(fps_t: np.ndarray) -> np.ndarray:
    """L2 distance from each ticker's fingerprint to the cross-sectional
    centroid at one date. Returns `(n_tickers,)`. Used by idea C.
    """
    centroid = cross_sectional_centroid(fps_t)
    return np.linalg.norm(fps_t - centroid[None, :], axis=-1)
