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

from ss_features import Compression, compress_tiles


def extract_fingerprints(
    coeffs: np.ndarray,
    *,
    w: int,
    znorm: bool = True,
    compression: Compression | None = None,
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
    compression : Compression | None
        Optional 2D DWT keep-LL compression of each `(S, w)` per-bar
        tile before flattening. With L levels of Haar DWT keep-LL the
        fingerprint dim shrinks `S*w → ceil(S/2^L) * ceil(w/2^L)`.
        Causality is preserved because each tile contains only past
        bars. None = the original full-resolution fingerprint.

    Returns
    -------
    fps : np.ndarray, shape `(n_dates, n_tickers, fp_dim)`, float32
        `fp_dim = S*w` when `compression is None`, else
        `ceil(S/2^L) * ceil(w/2^L)` after the LL keep. For dates
        `t < w-1`, the fingerprint is computed against a zero-padded
        window — caller should drop those rows or apply the same
        `lookback` floor used elsewhere. L2-normalization (when on)
        runs *after* compression, so the unit-norm property is
        preserved in the compressed space.
    """
    n_scales, n_dates, n_tickers = coeffs.shape

    pad = np.zeros((n_scales, w - 1, n_tickers), dtype=coeffs.dtype)
    padded = np.concatenate([pad, coeffs], axis=1)
    sw = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=w, axis=1)
    # `sw` shape: `(n_scales, n_dates, n_tickers, w)`. Move tile axes
    # to the end for either flatten-only or DWT-then-flatten.
    tiles = np.transpose(sw, (1, 2, 0, 3)).astype(np.float32, copy=False)
    # `tiles` shape: `(n_dates, n_tickers, n_scales, w)`.

    if compression is not None:
        # Apply the chosen 2D transform independently per (date, ticker)
        # tile. Reshape to a flat batch so the underlying scipy/pywt
        # call vectorises over all tiles in one pass. DWT returns
        # `(n_batch, S', W')`, DCT returns `(n_batch, k)` — both
        # collapse to a flat fp via `reshape(..., -1)`.
        flat = tiles.reshape(n_dates * n_tickers, n_scales, w)
        compressed = compress_tiles(flat, compression)
        fps = compressed.reshape(n_dates, n_tickers, -1)
    else:
        fps = tiles.reshape(n_dates, n_tickers, n_scales * w)

    if znorm:
        norms = np.linalg.norm(fps, axis=-1, keepdims=True)
        fps = fps / np.maximum(norms, 1e-8)
    return fps.astype(np.float32, copy=False)


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
