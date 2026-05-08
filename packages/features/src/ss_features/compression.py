"""Per-bar 2D compression of CWT tiles before the replay CNN sees them.

Strict causality: each per-bar tile contains only past bars
`[t-K+1, t]` × scales, so applying a 2D transform inside the tile cannot
leak future. Applying the same transform to the full
`(n_scales, n_dates)` matrix would smear future bars into past via the
filter footprint and break the causal contract — don't do that.

Two transforms exposed via `Compression.kind`:

- `'dwt'` — L-level 2D wavelet decomposition, keep LL approximation
  only. Output shape per tile is `(ceil(K/2^L), ceil(S/2^L))` for the
  default `mode='periodization'`. Both CWT-derived channel stacks
  (signed coeffs + power) are compressed independently — the CNN can
  learn cross-channel mixing downstream.
- `'dct'` — 2D type-II DCT (orthonormal), zigzag traversal, keep
  top-k coefficients by zigzag order (low-frequency first). Output is
  a flat `(k,)` vector per tile — loses the 2D `(K, C)` structure the
  replay CNN reshape relies on, so the replay path still rejects DCT
  (a flat-input decoder branch is the prerequisite — see TODO.md).
  The relational fingerprint path consumes a flat fingerprint vector
  by design, so DCT works there as-is.

Compression operates only on the CWT-derived channels. Optional
channels (rolling z-norm stats, raw return, return sign) have no scale
axis and would need a separate handling path; callers must keep them
disabled while compression is active. `build_features_and_targets`
enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pywt
from scipy.fft import dctn


@dataclass(frozen=True)
class Compression:
    """Configuration for per-bar 2D compression of CWT tiles.

    `levels` and `wavelet`/`pad_mode` apply to `kind='dwt'`.
    `keep_top_k` applies to `kind='dct'` (zigzag order).
    """
    kind: str = 'dwt'
    levels: int = 1
    wavelet: str = 'haar'
    pad_mode: str = 'periodization'
    keep_top_k: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ('dwt', 'dct'):
            raise ValueError(
                f"Compression.kind must be 'dwt' or 'dct', got {self.kind!r}")
        if self.kind == 'dwt' and self.levels < 1:
            raise ValueError(
                f'Compression(kind=dwt) requires levels >= 1, '
                f'got {self.levels}')
        if self.kind == 'dct' and self.keep_top_k < 1:
            raise ValueError(
                f'Compression(kind=dct) requires keep_top_k >= 1, '
                f'got {self.keep_top_k}')

    def output_shape(self, K: int, S: int) -> tuple[int, ...]:
        """Per-tile output shape after compression.

        DWT returns `(K', S')`. DCT zigzag-keep-top-k returns `(k,)` —
        flat — clamped to `K * S` if `keep_top_k` exceeds the total
        coefficient count.
        """
        if self.kind == 'dwt':
            probe = np.zeros((K, S), dtype=np.float32)
            coeffs = pywt.wavedec2(
                probe, wavelet=self.wavelet, level=self.levels,
                mode=self.pad_mode)
            return coeffs[0].shape
        return (min(self.keep_top_k, K * S),)


def compress_tiles_2d_dwt(
    tiles: np.ndarray, comp: Compression,
) -> np.ndarray:
    """Apply L-level 2D DWT to each per-bar tile, return LL only.

    `tiles` shape `(n_dates, K, S)`. Output `(n_dates, K', S')` where
    `(K', S') = comp.output_shape(K, S)`.

    Uses `axes=(-2, -1)` so the same call vectorises over the leading
    `n_dates` axis — pywt handles the broadcast internally.
    """
    if comp.kind != 'dwt':
        raise ValueError(
            f'compress_tiles_2d_dwt only supports kind=dwt, got {comp.kind!r}')
    coeffs = pywt.wavedec2(
        tiles, wavelet=comp.wavelet, level=comp.levels,
        mode=comp.pad_mode, axes=(-2, -1))
    return coeffs[0].astype(np.float32)


@lru_cache(maxsize=64)
def _zigzag_indices(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard JPEG-style zigzag traversal of an `rows × cols` grid.

    Returns paired index arrays `(rs, cs)` of length `rows*cols` such
    that `M.flatten()[rs * cols + cs]` walks `M` in zigzag order:
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), ...

    Cached because the order only depends on the tile shape, not the
    data — the same `(K, S)` reuses the same traversal across every
    `(date, ticker)` pair.
    """
    rs = np.empty(rows * cols, dtype=np.int64)
    cs = np.empty(rows * cols, dtype=np.int64)
    out = 0
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            i = min(s, rows - 1)
            j = s - i
            while i >= 0 and j < cols:
                rs[out] = i
                cs[out] = j
                out += 1
                i -= 1
                j += 1
        else:
            j = min(s, cols - 1)
            i = s - j
            while j >= 0 and i < rows:
                rs[out] = i
                cs[out] = j
                out += 1
                i += 1
                j -= 1
    return rs, cs


def compress_tiles_2d_dct_zigzag(
    tiles: np.ndarray, comp: Compression,
) -> np.ndarray:
    """Apply 2D type-II DCT to each tile, return the first
    `comp.keep_top_k` coefficients in zigzag order.

    `tiles` shape `(n_batch, K, S)` (or any leading batch dims, the
    last two axes are the tile). Output `(n_batch, k)` where
    `k = min(comp.keep_top_k, K * S)`.

    Uses `norm='ortho'` so the transform is energy-preserving — the
    coefficient magnitudes are directly comparable to the input
    magnitudes (a useful property when the downstream consumer
    L2-normalizes the result, like
    `relational.fingerprints.extract_fingerprints` does).
    """
    if comp.kind != 'dct':
        raise ValueError(
            f'compress_tiles_2d_dct_zigzag only supports kind=dct, '
            f'got {comp.kind!r}')
    K, S = tiles.shape[-2], tiles.shape[-1]
    k = min(comp.keep_top_k, K * S)
    coeffs = dctn(tiles, type=2, axes=(-2, -1), norm='ortho')
    rs, cs = _zigzag_indices(K, S)
    rs_top = rs[:k]
    cs_top = cs[:k]
    # advanced index along the tile axes; broadcasts over leading dims
    return coeffs[..., rs_top, cs_top].astype(np.float32)


def compress_tiles(
    tiles: np.ndarray, comp: Compression,
) -> np.ndarray:
    """Dispatch on `comp.kind` and apply the matching transform.

    Returns `(n_batch, K', S')` for DWT, `(n_batch, k)` for DCT —
    callers should `.reshape(..., -1)` if they need a uniform flat
    feature vector across compression kinds.
    """
    if comp.kind == 'dwt':
        return compress_tiles_2d_dwt(tiles, comp)
    if comp.kind == 'dct':
        return compress_tiles_2d_dct_zigzag(tiles, comp)
    raise ValueError(f'unknown compression kind: {comp.kind!r}')
