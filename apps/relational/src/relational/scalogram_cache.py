"""Disk cache for `ss_wavelets.causal_cwt` outputs.

Computing the causal CWT over the Phase-2 universe at all default
scales is ~5-15s; over `stooq_us_long` it's longer. The four relational
ideas (empirical sectors / k-NN analog / farthest-from-centroid /
diversified) all hit the same scalogram many times, and the k-NN
search in particular slices it across many historical dates per
rebalance. Caching the raw coefficients to disk turns repeated runs
into a `np.load` and lets us iterate on downstream scoring without
re-paying the CWT cost.

The cache is content-addressed: the key is a hash of
`(sorted_tickers, scales, lookback, n_dates, first_date, last_date,
prices_bytes_hash)`. Any drift in the universe or the price values
silently produces a new cache file, so stale CWT can never be served
under the same key.

Files live in `apps/relational/.scalogram-cache/{key}.npz` by default.
The directory is gitignored.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ss_wavelets import causal_cwt


_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / '.scalogram-cache'


def _hash_inputs(
    *,
    tickers: tuple[str, ...],
    scales: tuple[int, ...],
    lookback: int,
    dates: pd.DatetimeIndex,
    prices_arr: np.ndarray,
) -> str:
    h = hashlib.sha256()
    h.update(b'tickers:' + ','.join(tickers).encode())
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
    cache_dir: Path | str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Cached wrapper around `ss_wavelets.causal_cwt`.

    Parameters mirror `causal_cwt`. Returns the same `(n_scales,
    n_dates, n_tickers)` float32 array.

    On cache miss, computes via `causal_cwt` and writes the result
    plus all keying metadata to a `.npz` under `cache_dir`. On hit,
    loads and returns the cached coefficients. Cache invalidation is
    automatic via the input hash — never serves stale data.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers = tuple(prices.columns)
    scales_t = tuple(int(s) for s in scales)
    key = _hash_inputs(
        tickers=tickers, scales=scales_t, lookback=int(lookback),
        dates=prices.index, prices_arr=prices.values)
    cache_path = cache_dir / f'cwt-{key}.npz'

    if cache_path.exists():
        if verbose:
            print(f'[scalogram_cache] hit  {cache_path.name} '
                  f'({len(tickers)} tickers, {len(scales_t)} scales, '
                  f'{len(prices)} dates)')
        with np.load(cache_path) as npz:
            return npz['coeffs'].astype(np.float32, copy=False)

    if verbose:
        print(f'[scalogram_cache] miss {cache_path.name} — computing causal_cwt'
              f' ({len(tickers)} tickers, {len(scales_t)} scales, '
              f'{len(prices)} dates)')
    coeffs = causal_cwt(prices.values, list(scales_t), int(lookback))
    np.savez_compressed(
        cache_path,
        coeffs=coeffs,
        tickers=np.asarray(tickers),
        scales=np.asarray(scales_t),
        lookback=np.int64(lookback),
        first_date=str(prices.index[0].date()),
        last_date=str(prices.index[-1].date()),
    )
    return coeffs
