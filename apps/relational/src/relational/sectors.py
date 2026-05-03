"""Ticker → GICS sector mapping for the Phase-2 universe.

Hardcoded for the 21 tickers baked into `apps/notebook/data/stooq_phase2/`
plus the canonical 11 US sector ETFs (used as a future alternative to
equal-weighted constituent aggregates).

Mapping is current as of GICS 2024 (Communication Services split off
from IT/Discretionary in 2018; Real Estate split from Financials in
2016; both reflected here).

Extending the universe:
- Add new (ticker, sector) entries to `PHASE2_TICKER_TO_SECTOR`.
- For tickers whose sector isn't in `SECTOR_ETFS.keys()`, add the new
  sector to the ETF map (and pick a canonical ETF for ETF-mode
  aggregates) or rely on the equal-weighted aggregate path.
"""

from __future__ import annotations

import numpy as np


# Current GICS 11-sector classification.
GICS_SECTORS: tuple[str, ...] = (
    'Information Technology',
    'Communication Services',
    'Consumer Discretionary',
    'Consumer Staples',
    'Health Care',
    'Financials',
    'Industrials',
    'Energy',
    'Materials',
    'Utilities',
    'Real Estate',
)


# Canonical Select-Sector SPDR ETFs per GICS sector. Used by ETF-mode
# `aggregates.sector_series` when constituent-weighting isn't desired.
SECTOR_ETFS: dict[str, str] = {
    'Information Technology': 'XLK',
    'Communication Services': 'XLC',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples':       'XLP',
    'Health Care':            'XLV',
    'Financials':             'XLF',
    'Industrials':            'XLI',
    'Energy':                 'XLE',
    'Materials':              'XLB',
    'Utilities':              'XLU',
    'Real Estate':            'XLRE',
}


# 21-ticker Phase-2 universe (mirrors apps/notebook/data/stooq_phase2/).
# Only sectors actually represented here populate the sector aggregate;
# 8 of the 11 GICS sectors are represented (no Materials, Utilities,
# Real Estate constituents in the Phase-2 set).
PHASE2_TICKER_TO_SECTOR: dict[str, str] = {
    # Information Technology
    'AAPL':  'Information Technology',
    'MSFT':  'Information Technology',
    'NVDA':  'Information Technology',
    'CRM':   'Information Technology',
    'CSCO':  'Information Technology',
    # Communication Services
    'GOOGL': 'Communication Services',
    'META':  'Communication Services',
    'NFLX':  'Communication Services',
    'T':     'Communication Services',
    'DIS':   'Communication Services',
    # Consumer Discretionary
    'AMZN':  'Consumer Discretionary',
    'TSLA':  'Consumer Discretionary',
    # Consumer Staples
    'KO':    'Consumer Staples',
    'WMT':   'Consumer Staples',
    # Health Care
    'JNJ':   'Health Care',
    'UNH':   'Health Care',
    # Financials
    'JPM':   'Financials',
    'BAC':   'Financials',
    # Industrials
    'GE':    'Industrials',
    'BA':    'Industrials',
    # Energy
    'XOM':   'Energy',
}


def sectors_for_universe(
    tickers: list[str],
    *,
    mapping: dict[str, str] | None = None,
    on_unknown: str = 'raise',
) -> list[str]:
    """Return the sector name for each ticker in `tickers`, in order.

    `mapping` defaults to `PHASE2_TICKER_TO_SECTOR`. `on_unknown`
    controls the behavior when a ticker is missing from the map:
      - 'raise' (default) — raise KeyError listing the missing tickers
      - 'skip'            — return 'Unknown' for missing entries
    """
    m = mapping if mapping is not None else PHASE2_TICKER_TO_SECTOR
    missing = [t for t in tickers if t not in m]
    if missing and on_unknown == 'raise':
        raise KeyError(
            f'no sector mapping for {missing!r}; extend '
            f'PHASE2_TICKER_TO_SECTOR or pass on_unknown="skip"')
    return [m.get(t, 'Unknown') for t in tickers]


def ticker_to_sector_idx(
    tickers: list[str],
    sector_order: list[str],
    *,
    mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Return a `(n_tickers,)` int array mapping each ticker to its
    column index in `sector_order`. Useful for vectorized lookup like
    `sector_div[:, ticker_to_sector_idx(...)]`.
    """
    sectors = sectors_for_universe(tickers, mapping=mapping)
    sector_idx = {s: i for i, s in enumerate(sector_order)}
    missing = [s for s in sectors if s not in sector_idx]
    if missing:
        raise KeyError(
            f'sectors {sorted(set(missing))!r} not in sector_order; '
            f'available: {sector_order}')
    return np.array([sector_idx[s] for s in sectors], dtype=np.int64)
