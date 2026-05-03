"""Sector-aggregate price series builders.

Two modes:
  - 'equal'   — equal-weighted average of constituent prices per
                sector (default; needs no extra data, validates the
                idea before investing in ETF data plumbing)
  - 'cap'     — TODO: market-cap-weighted (needs shares-outstanding)
  - 'etf'     — TODO: pull a sector ETF series from the same loader
                (XLK / XLF / etc. — see sectors.SECTOR_ETFS)

For week-1 only `equal` is wired. The output is a pandas DataFrame
with the same DatetimeIndex as the input and columns = unique sectors
present in `tickers`.

Single-constituent sectors (e.g. Energy = {XOM} in the Phase-2 set)
produce a sector aggregate that is identical to the constituent. The
excess-divergence math then yields exactly 0 for that ticker — a
documented degenerate case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relational.sectors import sectors_for_universe


def sector_series(
    prices: pd.DataFrame,
    *,
    mode: str = 'equal',
    sector_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build per-sector aggregate price series from constituent prices.

    Parameters
    ----------
    prices : DataFrame, shape (n_dates, n_tickers)
        Per-ticker price history. Index must be a DatetimeIndex; columns
        are ticker symbols.
    mode : {'equal'}
        Aggregation mode. Only 'equal' (equal-weighted price average) is
        implemented; 'cap' and 'etf' are TODO.
    sector_mapping : dict[str, str] | None
        Ticker → sector name. Defaults to PHASE2_TICKER_TO_SECTOR.

    Returns
    -------
    sector_prices : DataFrame, shape (n_dates, n_sectors_present)
        Equal-weighted sector aggregate, columns sorted alphabetically
        by sector name for stable column order.
    sector_order : list[str]
        Sector names in column order — pass to
        `sectors.ticker_to_sector_idx(tickers, sector_order)` to map
        each ticker to its sector column.
    """
    if mode != 'equal':
        raise NotImplementedError(
            f'sector_series mode={mode!r} not yet implemented; '
            'only mode="equal" is wired in week-1')
    tickers = list(prices.columns)
    sectors = sectors_for_universe(tickers, mapping=sector_mapping)
    # Bucket tickers by sector, preserving DataFrame column order.
    buckets: dict[str, list[str]] = {}
    for ticker, sector in zip(tickers, sectors):
        buckets.setdefault(sector, []).append(ticker)
    sector_order = sorted(buckets.keys())
    cols = {
        s: prices[buckets[s]].mean(axis=1) for s in sector_order
    }
    return pd.DataFrame(cols, index=prices.index), sector_order
