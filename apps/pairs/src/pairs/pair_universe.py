"""Per-window pair screening.

Cross-section of all `(i, j)` ticker pairs is `O(N^2)` (~44k for
N=297). To make the EG test tractable we pre-filter by *correlation
of log-prices on the train slice* — only test pairs whose absolute
correlation is above a threshold. This is the standard quant-lit
preprocessing step (see Gatev-Goetzmann-Rouwenhorst, Vidyamurthy).
Cuts the work to ~1-5k pairs in practice.

Per-window screening is *load-bearing*: cointegration is regime-
specific. The relational analog scorer learned this the hard way.
We re-screen per train slice, never reuse a pair list across windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs.cointegration import EngleGrangerResult, engle_granger_test


@dataclass(frozen=True)
class PairCandidate:
    """One screened pair on one train window."""
    a:           str           # ticker A
    b:           str           # ticker B (hedge: A − β·B is stationary)
    eg_p:        float
    eg_stat:     float
    hedge_beta:  float
    intercept:   float
    train_corr:  float          # corr(log_p_a, log_p_b) on train


def _ticker_pairs_with_history(
    log_prices: pd.DataFrame, min_overlap: int,
) -> list[tuple[str, str]]:
    """Pre-filter: only pairs with `min_overlap` bars of joint
    non-NaN history."""
    notna = log_prices.notna().values
    n_dates, n_tickers = notna.shape
    counts = notna.astype(np.int32).T @ notna.astype(np.int32)
    out = []
    names = log_prices.columns.tolist()
    for i, j in combinations(range(n_tickers), 2):
        if counts[i, j] >= min_overlap:
            out.append((names[i], names[j]))
    return out


def _correlation_filter(
    log_prices: pd.DataFrame, candidates: list[tuple[str, str]],
    abs_corr_min: float,
) -> list[tuple[str, str, float]]:
    """Keep only candidates whose joint-non-NaN corr exceeds threshold."""
    out: list[tuple[str, str, float]] = []
    for a, b in candidates:
        cols = log_prices[[a, b]].dropna()
        if len(cols) < 50:
            continue
        c = float(cols.corr().iloc[0, 1])
        if abs(c) >= abs_corr_min:
            out.append((a, b, c))
    return out


def _eg_one_pair(args):
    a, b, corr, log_p_a_arr, log_p_b_arr = args
    res = engle_granger_test(log_p_a_arr, log_p_b_arr)
    return a, b, corr, res


def screen_pairs(
    log_prices: pd.DataFrame, *,
    min_overlap: int = 252,
    abs_corr_min: float = 0.7,
    eg_p_max: float = 0.05,
    top_k: int = 50,
    n_workers: int = 1,
    verbose: bool = True,
) -> list[PairCandidate]:
    """Screen `(A, B)` pairs on the train window.

    Pipeline:
      1. Drop pairs with < `min_overlap` bars of joint coverage.
      2. Drop pairs with `|corr(log_p_a, log_p_b)| < abs_corr_min`.
      3. Engle-Granger test on survivors.
      4. Keep pairs with EG p-value < `eg_p_max`.
      5. Take top `top_k` by ascending p-value.

    `log_prices` must already be the train slice — caller is
    responsible for slicing to avoid look-ahead.
    """
    if verbose:
        print(f'  screening: {log_prices.shape[1]} tickers x '
              f'{log_prices.shape[0]} train bars', flush=True)
    candidates = _ticker_pairs_with_history(log_prices, min_overlap)
    if verbose:
        print(f'  → {len(candidates)} pairs pass min_overlap={min_overlap}',
              flush=True)
    corr_passed = _correlation_filter(
        log_prices, candidates, abs_corr_min)
    if verbose:
        print(f'  → {len(corr_passed)} pairs pass '
              f'|corr|>={abs_corr_min}', flush=True)
    if not corr_passed:
        return []

    # Pre-stash arrays so workers don't pickle the entire DataFrame
    # per pair.
    pool_args = []
    for a, b, corr in corr_passed:
        cols = log_prices[[a, b]].dropna()
        pool_args.append((a, b, corr, cols[a].values, cols[b].values))

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            iterator = pool.imap_unordered(
                _eg_one_pair, pool_args, chunksize=200)
            results = list(tqdm(
                iterator, total=len(pool_args),
                desc='  EG', disable=not verbose, unit='pair'))
    else:
        results = list(tqdm(
            (_eg_one_pair(a) for a in pool_args), total=len(pool_args),
            desc='  EG', disable=not verbose, unit='pair'))

    survivors = [
        PairCandidate(
            a=a, b=b, eg_p=r.p_value, eg_stat=r.test_stat,
            hedge_beta=r.hedge_beta, intercept=r.intercept,
            train_corr=corr)
        for a, b, corr, r in results
        if r.p_value < eg_p_max and np.isfinite(r.hedge_beta)
    ]
    survivors.sort(key=lambda c: c.eg_p)
    if verbose:
        print(f'  → {len(survivors)} pairs pass EG p<{eg_p_max}',
              flush=True)
    return survivors[:top_k]


__all__ = ['PairCandidate', 'screen_pairs']
