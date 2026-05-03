"""Coverage tests for the Phase-2 ticker → sector mapping."""

from __future__ import annotations

import numpy as np
import pytest

from relational.sectors import (
    GICS_SECTORS,
    PHASE2_TICKER_TO_SECTOR,
    SECTOR_ETFS,
    sectors_for_universe,
    ticker_to_sector_idx,
)


PHASE2_UNIVERSE = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'CSCO',
    'JPM', 'BAC', 'GE', 'BA', 'XOM', 'KO', 'WMT', 'JNJ', 'UNH', 'T', 'DIS',
    'TSLA',
)


def test_phase2_universe_fully_mapped():
    """Every ticker in the Stooq Phase-2 subset has a sector."""
    missing = [t for t in PHASE2_UNIVERSE if t not in PHASE2_TICKER_TO_SECTOR]
    assert not missing, f'unmapped Phase-2 tickers: {missing}'


def test_all_mapped_sectors_are_valid_gics():
    """Every mapped sector must be one of the 11 GICS sectors."""
    for ticker, sector in PHASE2_TICKER_TO_SECTOR.items():
        assert sector in GICS_SECTORS, (
            f'{ticker} → {sector!r} not in GICS_SECTORS')


def test_sector_etf_keys_match_gics():
    """SECTOR_ETFS keys must be a subset of GICS_SECTORS."""
    extra = set(SECTOR_ETFS) - set(GICS_SECTORS)
    assert not extra, f'SECTOR_ETFS has non-GICS keys: {extra}'


def test_sectors_for_universe_preserves_order():
    """sectors_for_universe should return sectors in the same order
    as the input ticker list."""
    sectors = sectors_for_universe(['AAPL', 'JPM', 'XOM'])
    assert sectors == ['Information Technology', 'Financials', 'Energy']


def test_sectors_for_universe_raises_on_unknown_by_default():
    with pytest.raises(KeyError, match='ZZZZ'):
        sectors_for_universe(['AAPL', 'ZZZZ'])


def test_sectors_for_universe_skip_returns_unknown_label():
    sectors = sectors_for_universe(['AAPL', 'ZZZZ'], on_unknown='skip')
    assert sectors == ['Information Technology', 'Unknown']


def test_ticker_to_sector_idx_correct_columns():
    """Index returned should match the sector_order column ordering."""
    sector_order = ['Energy', 'Financials', 'Information Technology']
    idx = ticker_to_sector_idx(['AAPL', 'JPM', 'XOM'], sector_order)
    assert isinstance(idx, np.ndarray)
    assert idx.dtype == np.int64
    assert idx.tolist() == [2, 1, 0]


def test_ticker_to_sector_idx_raises_on_missing_sector():
    """If a ticker's sector isn't in sector_order, raise."""
    with pytest.raises(KeyError, match='Energy'):
        ticker_to_sector_idx(['AAPL', 'XOM'],
                             ['Information Technology', 'Financials'])
