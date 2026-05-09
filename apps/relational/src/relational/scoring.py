"""Excess-divergence scoring (apps/docs/docs/notes.md "Multi-stock CWT
framings" idea #1).

For each (date, ticker) cell:
    score = divergence(stock_recent, stock_hist)
          − divergence(its_sector_recent, its_sector_hist)

Subtracts the sector-wide regime shift so what remains is the
idiosyncratic shift — the stock doing something its sector isn't.

The function signature mirrors `regime.trainer.weights_regime` so
this is a drop-in replacement for the existing weights builder in
any vectorbt or bt loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ss_indicators import get_divergence
from ss_portfolio import apply_nan_mask, select_top_n_matrix
from ss_wavelets import causal_cwt, precompute_windows

from relational.aggregates import sector_series
from relational.sectors import ticker_to_sector_idx


def baseline_divergence_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    scales: list[int],
    divergence: str = 'kl',
) -> np.ndarray:
    """Per-(date, ticker) baseline CWT-power divergence scores.

    The non-relational comparison point: same divergence math
    `weights_regime` uses, but returns the raw `(n_eval, n_tickers)`
    score matrix rather than top-N weights, so callers can also pull
    bot-N picks for rank-spread / pair-trade variants.
    """
    coeffs = causal_cwt(prices.values, scales, lookback)
    power = (coeffs ** 2).astype(np.float32)
    recent, historical = precompute_windows(power, lookback, n_tail)
    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    return np.array(
        div_fn(recent, historical, scale_log_weights), copy=True)


def excess_divergence_scores(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    scales: list[int],
    divergence: str = 'kl',
    sector_mode: str = 'equal',
    sector_mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Per-(date, ticker) excess divergence scores.

    Computes:
      stock_div  = div(stock_recent, stock_hist)        shape (n_eval_dates, n_tickers)
      sector_div = div(sector_recent, sector_hist)      shape (n_eval_dates, n_sectors)
      score      = stock_div − sector_div[:, sector_idx_per_ticker]

    `n_eval_dates` = `n_dates − lookback` (matches `precompute_windows`'s
    output: dates < lookback don't have a complete CWT history yet).

    Returns
    -------
    scores : np.ndarray, shape (n_eval_dates, n_tickers)
        Larger values = stronger idiosyncratic regime shift relative to
        sector. Pass straight into `select_top_n_matrix` (or any other
        ranker) to build a portfolio.
    """
    tickers = list(prices.columns)

    # Per-stock CWT divergence (same primitives as weights_regime).
    stock_coeffs = causal_cwt(prices.values, scales, lookback)
    stock_power = (stock_coeffs ** 2).astype(np.float32)
    stock_recent, stock_hist = precompute_windows(stock_power, lookback, n_tail)

    # Per-sector aggregate prices → CWT divergence on the same scales.
    sector_prices, sector_order = sector_series(
        prices, mode=sector_mode, sector_mapping=sector_mapping)
    sector_coeffs = causal_cwt(sector_prices.values, scales, lookback)
    sector_power = (sector_coeffs ** 2).astype(np.float32)
    sector_recent, sector_hist = precompute_windows(
        sector_power, lookback, n_tail)

    # Both divergences use uniform per-scale weighting (matching
    # weights_regime's `scale_log_weights = zeros` convention).
    div_fn = get_divergence(divergence)
    scale_log_weights = np.zeros(len(scales), dtype=np.float32)
    stock_div = np.asarray(div_fn(stock_recent, stock_hist, scale_log_weights))
    sector_div = np.asarray(div_fn(sector_recent, sector_hist,
                                   scale_log_weights))

    # Per-ticker sector index, then broadcast subtract.
    sector_idx = ticker_to_sector_idx(
        tickers, sector_order, mapping=sector_mapping)
    excess = stock_div - sector_div[:, sector_idx]
    return np.array(excess, copy=True)   # writable host buffer for nan-mask


def weights_excess_regime(
    prices: pd.DataFrame,
    *,
    lookback: int,
    n_tail: int,
    top_n: int,
    scales: list[int],
    divergence: str = 'kl',
    sector_mode: str = 'equal',
    sector_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Drop-in replacement for `regime.trainer.weights_regime` that
    ranks by sector-excess divergence instead of raw per-stock
    divergence.

    Output is the same shape and convention: a `(n_eval_dates, n_tickers)`
    DataFrame of one-hot top-N selection (1.0 for picked, 0.0 otherwise).
    """
    scores = excess_divergence_scores(
        prices, lookback=lookback, n_tail=n_tail, scales=scales,
        divergence=divergence, sector_mode=sector_mode,
        sector_mapping=sector_mapping)
    scores = apply_nan_mask(scores, prices.values, lookback)
    weights = select_top_n_matrix(scores, top_n, ascending=False)
    return pd.DataFrame(
        weights, index=prices.index[lookback:], columns=prices.columns)
