"""Disclosure → price-panel join + leadership filter for the follower."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ss_loaders import (
    LeadershipFilter,
    build_leadership_filter,
    load_congressional_trades_xlsx,
    load_legislator_metadata,
    load_stooq_matrix,
)


@dataclass
class DisclosurePanel:
    """Aligned closes + filtered disclosure rows.

    ``closes`` is a wide (date × ticker) DataFrame of Stooq adjusted
    closes restricted to the disclosure-universe ∩ Stooq archive.
    ``disclosures`` is a tidy frame with one row per individual
    disclosed PURCHASE that survives the leadership filter and the
    ticker-in-Stooq filter — columns:
        ``ticker, filed, traded, bioguide, name, chamber, lag_days,
        leadership``.
    Already sorted by ``filed`` ascending. ``filed`` is the basis for
    entry timing: enter at ``filed + 1 trading day``.
    """

    closes: pd.DataFrame
    disclosures: pd.DataFrame
    drop_stats: dict


def build_eligible_disclosures(
    *,
    stooq_dir: str | Path,
    leadership_only: bool,
    min_years_for_tenure_proxy: float = 10.0,
    start: str = '2014-01-01',
    end: str | None = None,
    cache_dir: str | Path | None = None,
    purchases_only: bool = True,
) -> DisclosurePanel:
    """Build the joined (price-panel, disclosure-stream) artifact.

    Parameters
    ----------
    leadership_only :
        If True, drop disclosures from members who don't pass the
        leadership/tenure filter at their disclosure date. If False,
        keep all members (the apples-to-apples baseline arm).
    """
    end = end or pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    # 1. Disclosures (xlsx).
    df = load_congressional_trades_xlsx(cache_dir=cache_dir)
    n_raw = len(df)
    df = df[(df['filed'] >= start) & (df['filed'] <= end)].copy()
    n_in_span = len(df)
    if purchases_only:
        df = df[df['transaction'].str.upper().str.startswith('PURCHASE')].copy()
    n_purchase = len(df)

    # 2. Leadership filter (point-in-time).
    meta = load_legislator_metadata(cache_dir=cache_dir)
    lf: LeadershipFilter = build_leadership_filter(
        meta, min_years=min_years_for_tenure_proxy)
    df['leadership'] = df.apply(
        lambda r: lf.is_leadership(r['bioguide'], r['filed']), axis=1)
    n_leader = int(df['leadership'].sum())
    if leadership_only:
        df = df[df['leadership']].copy()

    # 3. Stooq price panel for the disclosed tickers.
    tickers = sorted(df['ticker'].unique().tolist())
    closes, _highs, _lows, _vol = load_stooq_matrix(
        stooq_dir,
        min_history=60,
        start_date=start,
        end_date=end,
        tickers=tickers,
        include_etfs=True,  # disclosures hit ETFs too (rare; allow)
    )
    in_stooq = set(closes.columns)
    df_before_stooq = len(df)
    df = df[df['ticker'].isin(in_stooq)].copy()
    df['lag_days'] = (df['filed'] - df['traded']).dt.days
    df = df.sort_values('filed').reset_index(drop=True)

    stats = {
        'n_raw': n_raw,
        'n_in_span': n_in_span,
        'n_purchase': n_purchase,
        'n_leadership_passing': n_leader,
        'n_after_stooq_join': len(df),
        'stooq_drop_rate': 1 - (len(df) / max(df_before_stooq, 1)),
        'unique_tickers': df['ticker'].nunique(),
        'unique_bioguides': df['bioguide'].nunique(),
        'median_lag_days': float(df['lag_days'].median()) if len(df) else float('nan'),
        'mean_lag_days': float(df['lag_days'].mean()) if len(df) else float('nan'),
    }
    return DisclosurePanel(closes=closes, disclosures=df, drop_stats=stats)


def trading_dates(closes: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted trading-day index of the closes panel."""
    return pd.DatetimeIndex(sorted(closes.index))


def next_trading_day(idx: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """Smallest trading day strictly greater than `date`. None if past end."""
    pos = int(np.searchsorted(idx.values, np.datetime64(date), side='right'))
    if pos >= len(idx):
        return None
    return idx[pos]
