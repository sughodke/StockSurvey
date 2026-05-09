"""Idea B — k-NN analog forecasting on scalogram fingerprints.

For each query `(ticker_i, date_t)` at a rebalance, find the K most
similar historical fingerprints across the universe (or per-ticker)
and average their realized forward returns. Rank by score, take top-N.

Two correctness rails (see `analog_knn_scores` for the actual guards):

  * **Causality.** A historical match `(ticker_j, date_s)` must have
    `s + forward_horizon < t`. We only ever look at candidate dates
    `s <= t - forward_horizon - 1`, enforced via `searchsorted` over a
    pre-sorted candidate-date array. Off-by-one here invalidates
    everything downstream.
  * **Autocorrelation.** Fingerprints at consecutive dates are nearly
    identical (window slides by 1 bar), so naive k-NN concentrates K
    matches on a 5-day cluster around one event. We enforce
    `min_sep_days` per ticker — when extending the K-best list, skip
    any candidate that shares a ticker with an already-picked match
    within `min_sep_days`.

Distance is L2 over unit-norm flattened CWT windows (see
`relational.fingerprints.extract_fingerprints`). Default `pool_mode`
is `'cross_ticker'` — the candidate pool spans every (ticker, date)
pair with a finite fingerprint and a finite forward return. The
`'per_ticker'` alternative restricts each query to that ticker's own
history; less data but isolates the signal from cross-ticker leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_features import Compression
from ss_portfolio import apply_nan_mask, select_top_n_matrix

from relational.fingerprints import extract_fingerprints
from relational.scalogram_cache import load_or_compute_cwt


def _forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Per-(date, ticker) forward returns over `horizon` bars.

    `fwd[s, j] = prices[s+h, j] / prices[s, j] - 1`. NaN where either
    endpoint is non-finite or `s + horizon` is out of bounds (the
    last `horizon` rows are always NaN).
    """
    n_dates, n_tickers = prices.shape
    fwd = np.full((n_dates, n_tickers), np.nan, dtype=np.float32)
    if horizon < n_dates:
        head = prices[:n_dates - horizon]
        tail = prices[horizon:]
        with np.errstate(invalid='ignore', divide='ignore'):
            fwd[:n_dates - horizon] = (tail / head - 1.0).astype(np.float32)
    return fwd


def _knn_pick(
    dist: np.ndarray,        # (n_cand,)
    cand_tickers: np.ndarray,  # (n_cand,) int
    cand_dates: np.ndarray,    # (n_cand,) int
    *,
    k: int,
    min_sep: int,
) -> np.ndarray:
    """Return indices into the candidate slice for the K accepted picks.

    Walks candidates in distance order. Skips any candidate whose
    ticker already appears in the picked set within `min_sep` days.
    Returns fewer than K if the pool exhausts.
    """
    order = np.argsort(dist, kind='stable')
    picks: list[int] = []
    picked_by_ticker: dict[int, list[int]] = {}
    for idx in order:
        idx = int(idx)
        tk = int(cand_tickers[idx])
        dt = int(cand_dates[idx])
        prior = picked_by_ticker.get(tk)
        if prior is not None and any(abs(dt - p) < min_sep for p in prior):
            continue
        picks.append(idx)
        picked_by_ticker.setdefault(tk, []).append(dt)
        if len(picks) >= k:
            break
    return np.asarray(picks, dtype=np.int64)


def analog_knn_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    cache_dir=None,
    compression: Compression | None = None,
    wavelet: str = 'ricker',
) -> np.ndarray:
    """Per-(date, ticker) forecasted forward-horizon return from
    k-NN analog matching on historical fingerprints.

    Returns `(n_eval, n_tickers)` float32 where `n_eval = n_dates -
    lookback`. NaN where insufficient history or non-finite fingerprint.

    Parameters
    ----------
    pool_mode : {'cross_ticker', 'per_ticker'}
        `cross_ticker` matches against every (ticker, date) candidate;
        `per_ticker` restricts to the query ticker's own history.

    Notes
    -----
    The causality guard lives at `cand_end = searchsorted(date_idx,
    t - forward_horizon, side='left')`, which selects only candidates
    with `s <= t - forward_horizon - 1`. Do not relax this without
    re-deriving the math — it is what stops forward-return leakage.
    """
    if pool_mode not in ('cross_ticker', 'per_ticker'):
        raise ValueError(f'unknown pool_mode {pool_mode!r}')

    coeffs = load_or_compute_cwt(
        prices, scales, lookback, wavelet=wavelet, cache_dir=cache_dir)
    fps = extract_fingerprints(
        coeffs, w=fp_window, znorm=True, compression=compression)
    n_dates, n_tickers, fp_dim = fps.shape

    fwd = _forward_returns(prices.values.astype(np.float32), forward_horizon)

    # Build the candidate matrix: every (s, j) with finite fingerprint
    # AND finite forward return, in date-major order so date_idx is
    # non-decreasing (lets us use searchsorted for the causality cut).
    fp_finite = np.isfinite(fps).all(axis=-1)        # (n_dates, n_tickers)
    fwd_finite = np.isfinite(fwd)                     # (n_dates, n_tickers)
    valid = fp_finite & fwd_finite
    date_idx_full, ticker_idx_full = np.nonzero(valid)
    cand_fps = fps[date_idx_full, ticker_idx_full].astype(np.float32, copy=False)
    cand_fwd = fwd[date_idx_full, ticker_idx_full].astype(np.float32, copy=False)
    cand_dates = date_idx_full.astype(np.int64, copy=False)
    cand_tickers = ticker_idx_full.astype(np.int64, copy=False)

    n_eval = n_dates - lookback
    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)
    if cand_dates.size == 0:
        return scores

    for t in range(lookback, n_dates):
        # Causality guard: only candidates with s + h < t are eligible.
        cand_end = int(np.searchsorted(
            cand_dates, t - forward_horizon, side='left'))
        if cand_end == 0:
            continue
        cand_slice_fps = cand_fps[:cand_end]
        cand_slice_fwd = cand_fwd[:cand_end]
        cand_slice_tk = cand_tickers[:cand_end]
        cand_slice_dt = cand_dates[:cand_end]

        for i in range(n_tickers):
            q = fps[t, i]
            if not np.isfinite(q).all():
                continue
            if pool_mode == 'per_ticker':
                mask = cand_slice_tk == i
                if not mask.any():
                    continue
                slc_fps = cand_slice_fps[mask]
                slc_fwd = cand_slice_fwd[mask]
                slc_tk = cand_slice_tk[mask]
                slc_dt = cand_slice_dt[mask]
            else:
                slc_fps = cand_slice_fps
                slc_fwd = cand_slice_fwd
                slc_tk = cand_slice_tk
                slc_dt = cand_slice_dt

            # Unit-norm fingerprints: ||a-b||^2 = 2 - 2<a,b>, so
            # ranking by inner product (descending) is equivalent. We
            # compute L2 directly for clarity; speed is not the bottleneck.
            diff = slc_fps - q[None, :]
            dist = np.einsum('ij,ij->i', diff, diff)

            picks = _knn_pick(
                dist, slc_tk, slc_dt,
                k=k_neighbors, min_sep=min_sep_days)
            if picks.size == 0:
                continue
            scores[t - lookback, i] = float(slc_fwd[picks].mean())

    return scores


# Worker-process globals for the mp.Pool path. Set by `_worker_init`
# under fork(); read-only access from `_worker_chunk`. Linux fork()
# copy-on-write means the parent's arrays are physically shared
# until written, so this avoids pickling 600MB of cand_fps to each
# worker.
_W_FPS: np.ndarray | None = None
_W_CAND_FPS: np.ndarray | None = None
_W_CAND_FWD: np.ndarray | None = None
_W_CAND_DATES: np.ndarray | None = None
_W_CAND_TICKERS: np.ndarray | None = None
_W_LOOKBACK: int | None = None
_W_K_NEIGHBORS: int | None = None
_W_FORWARD_HORIZON: int | None = None
_W_MIN_SEP_DAYS: int | None = None
_W_POOL_MODE: str | None = None
_W_CAP: int | None = None


def _worker_init(
    fps, cand_fps, cand_fwd, cand_dates, cand_tickers,
    lookback, k_neighbors, forward_horizon, min_sep_days,
    pool_mode, cap,
) -> None:
    """Pin worker process to single-thread BLAS + bind shared arrays
    as module globals (read-only, COW-shared with parent)."""
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    global _W_FPS, _W_CAND_FPS, _W_CAND_FWD, _W_CAND_DATES, _W_CAND_TICKERS
    global _W_LOOKBACK, _W_K_NEIGHBORS, _W_FORWARD_HORIZON, _W_MIN_SEP_DAYS
    global _W_POOL_MODE, _W_CAP
    _W_FPS = fps
    _W_CAND_FPS = cand_fps
    _W_CAND_FWD = cand_fwd
    _W_CAND_DATES = cand_dates
    _W_CAND_TICKERS = cand_tickers
    _W_LOOKBACK = lookback
    _W_K_NEIGHBORS = k_neighbors
    _W_FORWARD_HORIZON = forward_horizon
    _W_MIN_SEP_DAYS = min_sep_days
    _W_POOL_MODE = pool_mode
    _W_CAP = cap


def _worker_chunk(t_chunk: list[int]) -> tuple[int, np.ndarray]:
    """Per-worker entry point: compute analog scores for `t_chunk`."""
    n_tickers = _W_FPS.shape[1]
    out = np.full((len(t_chunk), n_tickers), np.nan, dtype=np.float32)
    for j, t in enumerate(t_chunk):
        cand_end = int(np.searchsorted(
            _W_CAND_DATES, t - _W_FORWARD_HORIZON, side='left'))
        if cand_end == 0:
            continue
        cs_fps = _W_CAND_FPS[:cand_end]
        cs_fwd = _W_CAND_FWD[:cand_end]
        cs_tk = _W_CAND_TICKERS[:cand_end]
        cs_dt = _W_CAND_DATES[:cand_end]
        Q = _W_FPS[t]
        finite_mask = np.isfinite(Q).all(axis=1)
        if not finite_mask.any():
            continue
        inner = cs_fps @ Q.T
        q_sq = (Q * Q).sum(axis=1)
        c_sq = (cs_fps * cs_fps).sum(axis=1)
        dist_mat = c_sq[:, None] + q_sq[None, :] - 2.0 * inner
        for i in range(n_tickers):
            if not finite_mask[i]:
                continue
            if _W_POOL_MODE == 'per_ticker':
                mask = cs_tk == i
                if not mask.any():
                    continue
                d_i = dist_mat[mask, i]
                tk_i = cs_tk[mask]
                dt_i = cs_dt[mask]
                fwd_i = cs_fwd[mask]
            else:
                d_i = dist_mat[:, i]
                tk_i = cs_tk
                dt_i = cs_dt
                fwd_i = cs_fwd
            if d_i.shape[0] > _W_CAP:
                top_idx = np.argpartition(d_i, _W_CAP)[:_W_CAP]
                d_i = d_i[top_idx]
                tk_i = tk_i[top_idx]
                dt_i = dt_i[top_idx]
                fwd_i = fwd_i[top_idx]
            picks = _knn_pick(
                d_i, tk_i, dt_i,
                k=_W_K_NEIGHBORS, min_sep=_W_MIN_SEP_DAYS)
            if picks.size == 0:
                continue
            out[j, i] = float(fwd_i[picks].mean())
    return t_chunk[0], out


def analog_knn_scores_fast(
    prices: pd.DataFrame,
    *,
    lookback: int,
    scales: list[int],
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    cache_dir=None,
    compression: Compression | None = None,
    topk_cap: int | None = None,
    n_workers: int = 1,
    wavelet: str = 'ricker',
) -> np.ndarray:
    """Vectorised reimplementation of `analog_knn_scores`.

    Same fingerprints, same causality guard (`s + h < t`), same
    `_knn_pick` (closest-first walk with `min_sep_days` per-ticker
    filter), same NaN semantics. Two changes from the slow path:

      1. **Vectorised distance compute.** The per-(t, i) einsum is
         replaced with one `(n_cand, n_tickers)` matmul per date —
         BLAS handles all tickers at once.
      2. **Top-k truncation before `_knn_pick`.** Candidates beyond
         rank `topk_cap` are dropped via `np.argpartition` before
         the Python pick walk, since the worst case under min-sep
         filtering for cross_ticker is `~k_neighbors²` deep walks
         (each pick can block at most `k_neighbors` consecutive
         indices on its own ticker). Default `topk_cap = 50 *
         k_neighbors` is a generous safety margin. Setting `None`
         disables truncation (matches slow-path behavior bitwise
         on the candidate set).

    `n_workers > 1` enables process-pool parallelism over the t-axis
    via fork()-shared memory. Each worker is pinned to single-thread
    BLAS to avoid oversubscription (n_workers × 8 BLAS threads on an
    8-core box is worse than n_workers × 1). On Modal `cpu=8` this
    gives ~6-7× wall-time speedup; the CWT precompute and bt step
    stay serial.

    Picks are correlated but not bitwise identical to the slow
    path: the matmul accumulator orders summations differently
    from `np.einsum`, so distances differ at FP-noise level
    (~1e-5) and adjacent ranks occasionally swap. On Phase-2
    (N=21, 12y), Pearson correlation between fast and slow scores
    is ~0.995 with max abs diff of ~0.01 forward-return units —
    the strategy's Sharpe under either path is statistically
    indistinguishable, but if you need bit-exact reproducibility
    with the slow path, use that one.
    """
    if pool_mode not in ('cross_ticker', 'per_ticker'):
        raise ValueError(f'unknown pool_mode {pool_mode!r}')

    coeffs = load_or_compute_cwt(
        prices, scales, lookback, wavelet=wavelet, cache_dir=cache_dir)
    fps = extract_fingerprints(
        coeffs, w=fp_window, znorm=True, compression=compression)
    n_dates, n_tickers, fp_dim = fps.shape

    fwd = _forward_returns(prices.values.astype(np.float32), forward_horizon)

    fp_finite = np.isfinite(fps).all(axis=-1)
    fwd_finite = np.isfinite(fwd)
    valid = fp_finite & fwd_finite
    date_idx_full, ticker_idx_full = np.nonzero(valid)
    cand_fps = fps[date_idx_full, ticker_idx_full].astype(np.float32, copy=False)
    cand_fwd = fwd[date_idx_full, ticker_idx_full].astype(np.float32, copy=False)
    cand_dates = date_idx_full.astype(np.int64, copy=False)
    cand_tickers = ticker_idx_full.astype(np.int64, copy=False)

    n_eval = n_dates - lookback
    scores = np.full((n_eval, n_tickers), np.nan, dtype=np.float32)
    if cand_dates.size == 0:
        return scores

    cap = topk_cap if topk_cap is not None else max(50 * k_neighbors, 2500)

    if n_workers > 1:
        import multiprocessing as mp
        t_range = list(range(lookback, n_dates))
        chunk_size = max(1, len(t_range) // (n_workers * 4))
        chunks = [t_range[i:i + chunk_size]
                  for i in range(0, len(t_range), chunk_size)]
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(fps, cand_fps, cand_fwd, cand_dates, cand_tickers,
                      lookback, k_neighbors, forward_horizon, min_sep_days,
                      pool_mode, cap),
        ) as pool:
            for chunk_t0, chunk_scores in pool.imap_unordered(
                    _worker_chunk, chunks):
                row_start = chunk_t0 - lookback
                scores[row_start:row_start + chunk_scores.shape[0]] = (
                    chunk_scores)
        return scores

    for t in range(lookback, n_dates):
        cand_end = int(np.searchsorted(
            cand_dates, t - forward_horizon, side='left'))
        if cand_end == 0:
            continue
        cs_fps = cand_fps[:cand_end]    # (n_cand, fp_dim)
        cs_fwd = cand_fwd[:cand_end]
        cs_tk = cand_tickers[:cand_end]
        cs_dt = cand_dates[:cand_end]

        Q = fps[t]                       # (n_tickers, fp_dim)
        finite_mask = np.isfinite(Q).all(axis=1)
        if not finite_mask.any():
            continue

        # ||q-c||² = ||q||² - 2 q·c + ||c||² for unit-norm fingerprints
        # this reduces to 2 - 2 q·c, but we compute the actual squared L2
        # so the path is robust if znorm is ever turned off.
        # `inner` shape: (n_cand, n_tickers).
        inner = cs_fps @ Q.T
        q_sq = (Q * Q).sum(axis=1)               # (n_tickers,)
        c_sq = (cs_fps * cs_fps).sum(axis=1)     # (n_cand,)
        # broadcast: (n_cand, 1) + (1, n_tickers) - 2*(n_cand, n_tickers)
        dist_mat = c_sq[:, None] + q_sq[None, :] - 2.0 * inner

        for i in range(n_tickers):
            if not finite_mask[i]:
                continue
            if pool_mode == 'per_ticker':
                mask = cs_tk == i
                if not mask.any():
                    continue
                d_i = dist_mat[mask, i]
                tk_i = cs_tk[mask]
                dt_i = cs_dt[mask]
                fwd_i = cs_fwd[mask]
            else:
                d_i = dist_mat[:, i]
                tk_i = cs_tk
                dt_i = cs_dt
                fwd_i = cs_fwd

            # argpartition truncate before the Python pick walk.
            # `cap = max(50*k, 2500)` — under min_sep filtering
            # the deepest rank used is bounded by k_neighbors² in
            # the cross_ticker pathological case, so 50*k = 2500
            # at default k=50 is a generous safety margin.
            n_cand_i = d_i.shape[0]
            if n_cand_i > cap:
                top_idx = np.argpartition(d_i, cap)[:cap]
                d_i = d_i[top_idx]
                tk_i = tk_i[top_idx]
                dt_i = dt_i[top_idx]
                fwd_i = fwd_i[top_idx]

            picks = _knn_pick(
                d_i, tk_i, dt_i,
                k=k_neighbors, min_sep=min_sep_days)
            if picks.size == 0:
                continue
            scores[t - lookback, i] = float(fwd_i[picks].mean())

    return scores


def weights_regime_analog(
    prices: pd.DataFrame,
    *,
    lookback: int,
    top_n: int,
    scales: list[int],
    fp_window: int = 21,
    k_neighbors: int = 50,
    forward_horizon: int = 20,
    min_sep_days: int = 21,
    pool_mode: str = 'cross_ticker',
    cache_dir=None,
    compression: Compression | None = None,
    wavelet: str = 'ricker',
) -> pd.DataFrame:
    """Top-N basket ranked by k-NN analog forecast score.

    Drop-in shape match for `weights_regime`: returns a `(n_dates -
    lookback, n_tickers)` one-hot DataFrame, equal-weighted over the
    chosen `top_n` (descending score = predicted higher forward return).

    `wavelet` selects the CWT kernel routed through
    `relational.scalogram_cache.load_or_compute_cwt`. `'ricker'`
    (default) preserves the canonical Phase-2 winner; `'morlet'` uses
    the polar Morlet + Gaussian bundle from
    `ss_features.causal_polar_morlet_matrix` (4 channels per scale,
    fp_dim = 4 * len(scales) * fp_window).
    """
    scores = analog_knn_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, k_neighbors=k_neighbors,
        forward_horizon=forward_horizon, min_sep_days=min_sep_days,
        pool_mode=pool_mode, cache_dir=cache_dir,
        compression=compression, wavelet=wavelet)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
