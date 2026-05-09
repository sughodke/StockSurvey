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
    channels_per_scale: int = 1,
) -> np.ndarray:
    """Compute per-(date, ticker) scalogram fingerprints.

    Parameters
    ----------
    coeffs : np.ndarray, shape `(L, n_dates, n_tickers)`
        Output of `ss_wavelets.causal_cwt` (Ricker, `L = n_scales`)
        or `ss_features.causal_polar_morlet_matrix` (polar Morlet
        bundle, `L = channels_per_scale * n_scales`). Caller is
        responsible for matching `channels_per_scale` below to the
        actual layout of the input.
    w : int
        Window length in bars. Each fingerprint stacks the most recent
        `w` CWT vectors → `L*w`-dim vector. Default in the research
        scripts is 21 (one trading month).
    znorm : bool
        If True, each fingerprint is L2-normalized to unit length.
        Makes distance comparisons scale-invariant across tickers
        (otherwise a high-vol ticker's fingerprint just has bigger
        coefficients and looks "far" from everything). Recommended on.
    compression : Compression | None
        Optional 2D DWT keep-LL compression of each `(scale-axis, w)`
        per-bar tile before flattening. With L levels of Haar DWT
        keep-LL the per-channel fingerprint dim shrinks
        `S*w → ceil(S/2^L) * ceil(w/2^L)`. Causality is preserved
        because each tile contains only past bars. None = the original
        full-resolution fingerprint.
    channels_per_scale : int
        How the leading axis of `coeffs` decomposes into channel × scale.
        Default 1 (Ricker — the leading axis is just `n_scales`). For
        the polar Morlet bundle pass `RELATIONAL_CHANNELS_PER_SCALE = 4`
        — the leading axis is then read as `channels_per_scale` blocks
        of `n_scales` rows each, in the order the matrix-form Morlet
        helper produces (`|c|, cos, sin, g`). Compression, when on, is
        applied **independently per channel block** so 2D-DWT-keep-LL
        does not mix bandpass amplitude with phase or with the
        Gaussian companion.

    Returns
    -------
    fps : np.ndarray, shape `(n_dates, n_tickers, fp_dim)`, float32
        `fp_dim = L*w` when `compression is None`, else
        `channels_per_scale * ceil(S/2^L) * ceil(w/2^L)` after the LL
        keep. For dates `t < w-1`, the fingerprint is computed against
        a zero-padded window — caller should drop those rows or apply
        the same `lookback` floor used elsewhere. L2-normalization
        (when on) runs *after* compression, so the unit-norm property
        is preserved in the compressed space.
    """
    L_total, n_dates, n_tickers = coeffs.shape
    if channels_per_scale < 1:
        raise ValueError(
            f'channels_per_scale must be >= 1, got {channels_per_scale}')
    if L_total % channels_per_scale != 0:
        raise ValueError(
            f'leading axis ({L_total}) is not divisible by '
            f'channels_per_scale ({channels_per_scale})')
    n_scales = L_total // channels_per_scale

    pad = np.zeros((L_total, w - 1, n_tickers), dtype=coeffs.dtype)
    padded = np.concatenate([pad, coeffs], axis=1)
    sw = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=w, axis=1)
    # `sw` shape: `(L_total, n_dates, n_tickers, w)`. Move tile axes
    # to the end for either flatten-only or DWT-then-flatten.
    tiles = np.transpose(sw, (1, 2, 0, 3)).astype(np.float32, copy=False)
    # `tiles` shape: `(n_dates, n_tickers, L_total, w)`.

    if compression is not None:
        # Per-channel-block compression. Reshape `(n_dates, n_tickers,
        # C, S, w)` and run the 2D transform separately per channel
        # so DWT-keep-LL only averages within a channel — never mixes
        # `|c|` with `cos` / `sin` / `g`.
        per_ch = tiles.reshape(
            n_dates * n_tickers, channels_per_scale, n_scales, w)
        ch_blocks = []
        for c in range(channels_per_scale):
            block = per_ch[:, c]            # (n_batch, n_scales, w)
            ll = compress_tiles(block, compression)
            ch_blocks.append(ll.reshape(n_dates * n_tickers, -1))
        compressed = np.concatenate(ch_blocks, axis=-1)
        fps = compressed.reshape(n_dates, n_tickers, -1)
    else:
        fps = tiles.reshape(n_dates, n_tickers, L_total * w)

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
