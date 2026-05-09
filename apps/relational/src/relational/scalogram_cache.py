"""Disk cache for causal-CWT panels (Ricker or polar Morlet).

Computing the causal CWT over the Phase-2 universe at all default
scales is ~5-15s; over `stooq_us_long` it's longer. The relational
strategies (empirical sectors / k-NN analog / farthest-from-centroid /
diversified) all hit the same scalogram many times, and the k-NN
search in particular slices it across many historical dates per
rebalance. Caching the raw coefficients to disk turns repeated runs
into a `np.load` and lets us iterate on downstream scoring without
re-paying the CWT cost.

The cache is content-addressed: the key is a hash of
`(wavelet, sorted_tickers, scales, lookback, n_dates, first_date,
last_date, prices_bytes_hash)`. Any drift in the universe, the price
values, or the chosen wavelet silently produces a new cache file, so
stale CWT can never be served under the same key. Existing Ricker
cache files (keyed without the wavelet field) hash to a different
digest from the new wavelet-aware path — they don't collide and don't
get reused; treat them as cold for this code path.

The Morlet path returns the matrix-form polar bundle from
`ss_features.causal_polar_morlet_matrix` — a `(C * n_scales, n_dates,
n_tickers)` panel with `C = RELATIONAL_CHANNELS_PER_SCALE = 4`
channels per scale stacked in the order `(|c|, cos(arg), sin(arg),
g)`. The Ricker path returns the legacy `(n_scales, n_dates,
n_tickers)` real coefficients. Downstream consumers
(`relational.fingerprints.extract_fingerprints` and divergence-based
scorers) treat the leading axis uniformly, so the `(C*S, T, N)` Morlet
panel slots in wherever a `(S, T, N)` Ricker panel was used.

Files live in `apps/relational/.scalogram-cache/{key}.npz` by default.
The directory is gitignored.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ss_features import causal_polar_morlet_matrix
from ss_wavelets import causal_cwt


_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / '.scalogram-cache'

SUPPORTED_WAVELETS: tuple[str, ...] = ('ricker', 'morlet')


def _hash_inputs(
    *,
    wavelet: str,
    tickers: tuple[str, ...],
    scales: tuple[int, ...],
    lookback: int,
    dates: pd.DatetimeIndex,
    prices_arr: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(b'wavelet:' + wavelet.encode())
    h.update(b'|tickers:' + ','.join(tickers).encode())
    h.update(b'|scales:' + ','.join(str(s) for s in scales).encode())
    h.update(b'|lookback:' + str(lookback).encode())
    h.update(b'|n_dates:' + str(len(dates)).encode())
    h.update(b'|first:' + str(dates[0].date()).encode())
    h.update(b'|last:' + str(dates[-1].date()).encode())
    # Price content fingerprint — catches data-dir updates.
    h.update(b'|prices:' + hashlib.sha256(
        np.ascontiguousarray(prices_arr, dtype=np.float64).tobytes()
    ).digest())
    return h.hexdigest()[:16]


def load_or_compute_cwt(
    prices: pd.DataFrame,
    scales: list[int],
    lookback: int,
    *,
    wavelet: str = 'ricker',
    cache_dir: Path | str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Cached wrapper around `causal_cwt` (Ricker) or
    `ss_features.causal_polar_morlet_matrix` (polar Morlet).

    Parameters
    ----------
    wavelet : {'ricker', 'morlet'}
        `'ricker'` returns the legacy `(n_scales, n_dates, n_tickers)`
        real Ricker coefficients (default — backward compatible with
        the existing canonical checkpoints). `'morlet'` returns the
        matrix-form polar Morlet + Gaussian bundle of shape
        `(RELATIONAL_CHANNELS_PER_SCALE * n_scales, n_dates,
        n_tickers)` from `ss_features`. Channel order is
        `(|c|, cos(arg), sin(arg), g)`, with row index
        `c * n_scales + s` for channel `c` at scale `s`.
    cache_dir : Path | str | None
        Override directory; defaults to
        `apps/relational/.scalogram-cache`.
    verbose : bool
        Print cache hit / miss lines.

    Returns
    -------
    np.ndarray, float32, shape `(L, n_dates, n_tickers)` where
    `L = n_scales` for Ricker and `L = 4 * n_scales` for Morlet.

    On cache miss, computes via the appropriate kernel and writes the
    result plus all keying metadata to a `.npz` under `cache_dir`. On
    hit, loads and returns the cached coefficients. Cache invalidation
    is automatic via the input hash — never serves stale data.
    """
    if wavelet not in SUPPORTED_WAVELETS:
        raise ValueError(
            f'unknown wavelet {wavelet!r}; supported: {SUPPORTED_WAVELETS}')

    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers = tuple(prices.columns)
    scales_t = tuple(int(s) for s in scales)
    key = _hash_inputs(
        wavelet=wavelet, tickers=tickers, scales=scales_t,
        lookback=int(lookback),
        dates=prices.index, prices_arr=prices.values)
    cache_path = cache_dir / f'cwt-{wavelet}-{key}.npz'

    if cache_path.exists():
        if verbose:
            print(f'[scalogram_cache] hit  {cache_path.name} '
                  f'({wavelet}, {len(tickers)} tickers, {len(scales_t)} '
                  f'scales, {len(prices)} dates)')
        with np.load(cache_path) as npz:
            return npz['coeffs'].astype(np.float32, copy=False)

    if verbose:
        print(f'[scalogram_cache] miss {cache_path.name} — computing '
              f'{wavelet} CWT ({len(tickers)} tickers, '
              f'{len(scales_t)} scales, {len(prices)} dates)')
    if wavelet == 'ricker':
        coeffs = causal_cwt(prices.values, list(scales_t), int(lookback))
    else:
        # Polar Morlet returns float32 already; matches the Ricker dtype.
        coeffs = causal_polar_morlet_matrix(
            prices.values, list(scales_t), lookback=int(lookback))
    np.savez_compressed(
        cache_path,
        coeffs=coeffs,
        wavelet=np.asarray(wavelet),
        tickers=np.asarray(tickers),
        scales=np.asarray(scales_t),
        lookback=np.int64(lookback),
        first_date=str(prices.index[0].date()),
        last_date=str(prices.index[-1].date()),
    )
    return coeffs
