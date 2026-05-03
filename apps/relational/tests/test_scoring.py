"""Sanity tests for excess_divergence_scores.

Two regressions worth pinning:
  1. Single-constituent sector → excess_div is exactly 0 for that ticker.
  2. Stock with stronger regime shift than its sector → excess_div > 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relational.scoring import excess_divergence_scores


def _make_prices(rng: np.random.Generator, n_dates: int,
                 n_tickers: int) -> np.ndarray:
    """Random walk panel, dollar-scale around $100."""
    return np.cumsum(rng.standard_normal((n_dates, n_tickers)),
                     axis=0) + 100.0


def test_single_constituent_sector_yields_zero_excess():
    """For a sector with one constituent, sector_aggregate == constituent
    so excess_divergence is identically 0 along that ticker's column."""
    rng = np.random.default_rng(0)
    n_dates = 400
    # XOM is the sole Energy constituent in the Phase-2 mapping → its
    # sector aggregate is itself.
    tickers = ['XOM', 'AAPL', 'MSFT']
    prices = pd.DataFrame(_make_prices(rng, n_dates, len(tickers)),
                          columns=tickers,
                          index=pd.date_range('2020-01-01', periods=n_dates))
    scales = [5, 10, 21]
    scores = excess_divergence_scores(
        prices, lookback=200, n_tail=20, scales=scales, divergence='kl')
    xom_col = tickers.index('XOM')
    # All finite cells must be zero (within fp64 noise).
    finite = scores[np.isfinite(scores[:, xom_col]), xom_col]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 0.0, atol=1e-9)


def test_excess_score_positive_when_stock_diverges_more_than_sector():
    """Inject a regime shift in one tech ticker; its excess divergence
    should be strictly positive in the post-shift window (it diverges
    more than its co-sector tech peers)."""
    rng = np.random.default_rng(1)
    n_dates = 600
    tickers = ['AAPL', 'MSFT', 'NVDA', 'CRM', 'CSCO']  # all Tech
    panel = _make_prices(rng, n_dates, len(tickers))
    # Inject a vol shock into AAPL after date 400 (10x volatility for
    # the rest of the series).
    aapl_col = tickers.index('AAPL')
    panel[400:, aapl_col] += np.cumsum(
        rng.standard_normal(n_dates - 400) * 10.0)
    prices = pd.DataFrame(panel, columns=tickers,
                          index=pd.date_range('2020-01-01', periods=n_dates))
    scales = [5, 10, 21]
    lookback = 252
    scores = excess_divergence_scores(
        prices, lookback=lookback, n_tail=20, scales=scales, divergence='kl')
    # Look at the *fresh* post-shift window — first 30 cells where the
    # shock is in the recent window but not yet in the historical
    # window. Beyond that, the divergence naturally decays as the
    # historical window absorbs the shock period (correct behavior of
    # recent-vs-historical divergence; unrelated to excess-subtraction).
    fresh_start = max(0, 400 - lookback)
    fresh = scores[fresh_start:fresh_start + 30, aapl_col]
    finite = fresh[np.isfinite(fresh)]
    assert finite.size > 0
    assert finite.mean() > 0, (
        f'expected mean excess > 0 in fresh post-shift window, '
        f'got {finite.mean():.4f}')
