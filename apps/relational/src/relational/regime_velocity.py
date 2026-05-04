"""Regime-velocity scoring on scalogram fingerprints.

A "fingerprint" at date `t` for ticker `i` is a flattened slice of the
causal CWT scalogram over the most recent `w` bars (z-normed to unit
length). The hypothesis here is that *vector arithmetic* in fingerprint
space has a tradeable interpretation analogous to word2vec: the
direction and magnitude of recent fingerprint motion is a richer
signal than either the snapshot fingerprint (idea C / farthest) or
discretized cluster-membership transitions.

Construction
------------
1. Per-(ticker, date) **regime-velocity** vector::

       v[t, i, :] = fp[t, i, :] - fp[t - W, i, :]

   for delta window `W`. Default 20 to match rebal cadence.

2. **Stable behavioral axes** via SVD on training-window velocities.
   Stack all finite `v[t, i]` for t in a held-out training period,
   center, run randomized SVD; the top `K` right singular vectors are
   the "stable behavioral axes."

3. **Scoring**:
     (a) `||v[t, i]||`              — magnitude of regime change.
     (b) max-|projection onto axis| — directed alignment with stable axes.

   The SVD is fit on a **strictly past** training window; projections
   are evaluated on held-out dates only — no look-ahead.

This module is intentionally self-contained — it computes the causal
CWT and fingerprints in-place rather than depending on the (newer)
relational.fingerprints / relational.scalogram_cache helpers, so it
works on older worktree branches that don't have those.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_wavelets import causal_cwt


# ---------------------------------------------------------------------
# Cached CWT — content-addressed by (tickers, scales, lookback, dates,
# price bytes). Identical contract to relational.scalogram_cache so we
# can share files with the wider research suite.
# ---------------------------------------------------------------------

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

    Returns `(n_scales, n_dates, n_tickers)` float32 coefficients.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[3] / '.scalogram-cache'
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers = tuple(prices.columns)
    scales_t = tuple(int(s) for s in scales)
    key = _hash_inputs(
        tickers=tickers, scales=scales_t, lookback=int(lookback),
        dates=prices.index, prices_arr=prices.values)
    cache_path = cache_dir / f'cwt-{key}.npz'

    if cache_path.exists():
        if verbose:
            print(f'[regime_velocity] cache hit  {cache_path.name}')
        with np.load(cache_path) as npz:
            return npz['coeffs'].astype(np.float32, copy=False)

    if verbose:
        print(f'[regime_velocity] cache miss {cache_path.name} — '
              f'computing causal_cwt over {len(tickers)} tickers, '
              f'{len(scales_t)} scales, {len(prices)} dates')
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


def extract_fingerprints(
    coeffs: np.ndarray, *, w: int, znorm: bool = True,
) -> np.ndarray:
    """Per-(date, ticker) flattened scalogram window.

    `coeffs` is `(S, T, N)`; returns `(T, N, S*w)` float32. Z-normed
    to unit L2 by default so fingerprint distances are
    direction-of-shape, not amplitude.

    Dates `t < w-1` are zero-padded; caller drops those via the
    `lookback` floor.
    """
    n_scales, n_dates, n_tickers = coeffs.shape
    fp_dim = n_scales * w
    pad = np.zeros((n_scales, w - 1, n_tickers), dtype=coeffs.dtype)
    padded = np.concatenate([pad, coeffs], axis=1)
    sw = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=w, axis=1)  # (S, T, N, w)
    fps = np.transpose(sw, (1, 2, 0, 3)).reshape(
        n_dates, n_tickers, fp_dim).astype(np.float32, copy=False)
    if znorm:
        norms = np.linalg.norm(fps, axis=-1, keepdims=True)
        fps = fps / np.maximum(norms, 1e-8)
    return fps


# ---------------------------------------------------------------------
# Velocity computation + behavioral-axis SVD.
# ---------------------------------------------------------------------

def compute_velocity(fps: np.ndarray, *, window: int = 20) -> np.ndarray:
    """Per-(date, ticker) fingerprint velocity.

    `fps` is `(T, N, D)`. Returns `(T, N, D)` velocity panel where
    `v[t, i] = fps[t, i] - fps[t - window, i]`. Rows `t < window` are
    NaN — caller masks via the lookback floor or `apply_nan_mask`.
    """
    T, N, D = fps.shape
    v = np.full_like(fps, np.nan)
    if T > window:
        v[window:] = fps[window:] - fps[:-window]
    return v


def fit_behavioral_axes(
    velocities: np.ndarray,
    *,
    n_axes: int = 5,
    train_end_idx: int,
    train_start_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """SVD top-K right singular vectors over a training-only velocity stack.

    Parameters
    ----------
    velocities : (T, N, D) float32
        Per-date per-ticker velocity panel.
    n_axes : int
        Number of axes (top-K right singular vectors) to return.
    train_end_idx : int
        Velocity rows in `[train_start_idx, train_end_idx)` are used to
        fit. **Must be strictly past** the eval window — otherwise
        projections leak future information into the basis.
    train_start_idx : int
        Default 0. Caller can push later to skip warmup rows where
        velocity is undefined / NaN.

    Returns
    -------
    axes : (n_axes, D) float32
        Orthonormal basis (rows are unit-norm right singular vectors).
    singular_values : (n_axes,) float32
        Top-K singular values from the truncated SVD.
    explained_variance_fraction : float
        Sum of top-K squared singular values / total Frobenius norm
        squared. Diagnostic — if low, axes are weak.
    """
    if train_end_idx <= train_start_idx:
        raise ValueError(
            f'train_end_idx ({train_end_idx}) must be > '
            f'train_start_idx ({train_start_idx})')

    train = velocities[train_start_idx:train_end_idx]    # (T_train, N, D)
    flat = train.reshape(-1, train.shape[-1])             # (T_train * N, D)
    finite = np.isfinite(flat).all(axis=1)
    X = flat[finite].astype(np.float32, copy=False)
    if X.shape[0] < n_axes + 1:
        raise ValueError(
            f'too few finite training velocities ({X.shape[0]}) for '
            f'{n_axes}-axis SVD')

    # Center — SVD on raw vs centered matters: PCA/word2vec both center.
    # Velocities are already differences, so the mean is small but
    # non-zero on a 252-day training panel; centering keeps the axes
    # interpretable.
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # Randomized SVD via `np.linalg.svd` on the smaller side. With
    # T_train * N typically ~50-100K rows and D ~ 168, the dense
    # `Vt` from `svd(Xc, full_matrices=False)` is the cheapest path
    # (no scipy/randomized SVD dependency).
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    axes = Vt[:n_axes].astype(np.float32, copy=False)
    sv = S[:n_axes].astype(np.float32, copy=False)

    total_sq = float((S ** 2).sum())
    topk_sq = float((sv ** 2).sum())
    explained = topk_sq / max(total_sq, 1e-12)
    return axes, sv, explained


# ---------------------------------------------------------------------
# Scoring helpers shared by both variants.
# ---------------------------------------------------------------------

def _build_velocity_panel(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int,
    w_delta: int,
    cache_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (fingerprints, velocities) for `prices`.

    Returns `(fps, velocities)` — both `(T, N, D)`. `T` = `n_dates`,
    `N` = `n_tickers`, `D` = `len(scales) * fp_window`.
    """
    coeffs = load_or_compute_cwt(
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
    velocities = compute_velocity(fps, window=w_delta)
    return fps, velocities


def velocity_magnitude_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    w_delta: int = 20,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Variant (a): score = ||v[t, i]||. No SVD, no look-ahead.

    Returns `(n_eval, n_tickers)` float32 with `n_eval = n_dates -
    lookback`. NaN cells flag tickers whose velocity isn't defined yet
    (warmup) — caller threads through `apply_nan_mask`.
    """
    _fps, velocities = _build_velocity_panel(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, w_delta=w_delta, cache_dir=cache_dir)
    v_eval = velocities[lookback:]                        # (n_eval, N, D)
    norms = np.linalg.norm(v_eval, axis=-1)               # (n_eval, N)
    # NaN-propagation: any NaN component → NaN norm. `linalg.norm`
    # silently sums NaN, so we manually re-assert.
    finite_mask = np.isfinite(v_eval).all(axis=-1)
    norms = np.where(finite_mask, norms, np.nan)
    return norms.astype(np.float32, copy=False)


def axis_alignment_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    w_delta: int = 20,
    n_axes: int = 5,
    train_window_days: int = 252,
    cache_dir: Path | str | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Variant (b): score = max_k |v[t, i] @ axis_k|.

    Look-ahead-free construction:
      - Velocities are computed on the full panel (causal: each v[t]
        only depends on fps[t] and fps[t-W], both strictly past).
      - SVD axes are fit on the **first `train_window_days` post-lookback
        velocity rows only**. Eval starts at `lookback + train_window_days`
        — earlier rows are NaN-d in the score matrix so bt won't pick
        on them.

    Parameters
    ----------
    n_axes : int
        Number of behavioral axes (top-K right singular vectors).
    train_window_days : int
        Rows `[lookback, lookback + train_window_days)` are used to fit
        the SVD. Held-out projection starts at row
        `lookback + train_window_days`.
    return_diagnostics : bool
        If True, also return a dict with the fitted axes, singular
        values, and explained-variance fraction. Used by the diagnostic
        backtest for the spectrum + spot-check printout.
    """
    fps, velocities = _build_velocity_panel(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, w_delta=w_delta, cache_dir=cache_dir)
    n_dates, n_tickers, fp_dim = fps.shape

    # SVD train-window in original (n_dates) coordinates.
    train_start = lookback
    train_end = lookback + train_window_days
    if train_end > n_dates:
        raise ValueError(
            f'train_end={train_end} > n_dates={n_dates}; need more data')

    axes, sv, explained = fit_behavioral_axes(
        velocities, n_axes=n_axes,
        train_start_idx=train_start,
        train_end_idx=train_end)
    # axes: (K, D)

    # Project eval window. Score matrix has same shape as the (a) variant
    # so it composes with apply_nan_mask + select_top_n_matrix.
    n_eval = n_dates - lookback
    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)

    # Held-out region only — train rows stay NaN so bt won't trade on
    # them. (apply_nan_mask + first rebal-grid filter would also catch
    # this, but explicit > implicit.)
    eval_offset = train_window_days  # in eval-row coords
    v_held = velocities[lookback + eval_offset:]    # (n_held, N, D)

    # (n_held, N, D) @ (D, K) -> (n_held, N, K)
    proj = v_held @ axes.T
    finite_mask = np.isfinite(v_held).all(axis=-1)   # (n_held, N)
    score_held = np.max(np.abs(proj), axis=-1)       # (n_held, N)
    score_held = np.where(finite_mask, score_held, np.nan)
    scores[eval_offset:] = score_held.astype(np.float32, copy=False)

    if not return_diagnostics:
        return scores

    diagnostics = {
        'axes': axes,
        'singular_values': sv,
        'explained_variance_fraction': float(explained),
        'train_start': int(train_start),
        'train_end': int(train_end),
        'fp_dim': int(fp_dim),
    }
    return scores, diagnostics


# ---------------------------------------------------------------------
# Hard-top-N weight builders matching the existing scorer API.
# ---------------------------------------------------------------------

def weights_velocity_magnitude(
    prices: pd.DataFrame,
    *,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int = 21,
    w_delta: int = 20,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Top-N by ||v[t, i]||. Drop-in for `weights_regime_farthest`."""
    scores = velocity_magnitude_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window,
        w_delta=w_delta, cache_dir=cache_dir)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)


def weights_axis_alignment(
    prices: pd.DataFrame,
    *,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int = 21,
    w_delta: int = 20,
    n_axes: int = 5,
    train_window_days: int = 252,
    cache_dir: Path | str | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Top-N by max-|projection onto top-K SVD axis|.

    SVD fit uses only the first `train_window_days` post-lookback
    velocity rows; held-out region starts at `lookback +
    train_window_days`. Earlier rows are NaN so bt never picks there.
    """
    out = axis_alignment_scores(
        prices, lookback=lookback, scales=scales, fp_window=fp_window,
        w_delta=w_delta, n_axes=n_axes,
        train_window_days=train_window_days, cache_dir=cache_dir,
        return_diagnostics=return_diagnostics)
    if return_diagnostics:
        scores, diagnostics = out
    else:
        scores = out
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    df = pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)
    if return_diagnostics:
        return df, diagnostics
    return df
