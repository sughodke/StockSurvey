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
        prices, scales, lookback, cache_dir=cache_dir)
    fps = extract_fingerprints(coeffs, w=fp_window, znorm=True)
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
) -> pd.DataFrame:
    """Top-N basket ranked by k-NN analog forecast score.

    Drop-in shape match for `weights_regime`: returns a `(n_dates -
    lookback, n_tickers)` one-hot DataFrame, equal-weighted over the
    chosen `top_n` (descending score = predicted higher forward return).
    """
    scores = analog_knn_scores(
        prices, lookback=lookback, scales=scales,
        fp_window=fp_window, k_neighbors=k_neighbors,
        forward_horizon=forward_horizon, min_sep_days=min_sep_days,
        pool_mode=pool_mode, cache_dir=cache_dir)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights,
        index=prices.index[lookback:],
        columns=prices.columns,
    )
