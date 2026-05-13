"""Per-quarter holdings panel builder.

Aggregates parsed 13F holdings across funds + quarters into a wide
`(quarter_end_date, ticker)` panel of total dollars held — restricted
to a target ticker universe. Names not in the manual `name_to_ticker`
map are silently dropped (typically <10% of dollar volume; see
`funds.py` for the rationale).

Two consumer-facing functions:

- `build_holdings_panel` — full pipeline: fetches submissions for each
  CIK, fetches infotable XML for each filing, parses, maps names to
  tickers, sums per-quarter per-ticker. Returns a DataFrame indexed by
  `period_of_report` with ticker columns.

- `build_consensus_top_k` — derives "top-K most-held tickers per
  quarter" from a holdings panel. The CFR `mode_long_13f_consensus`
  consumer uses this to produce a per-bar EW portfolio over the
  quarter's consensus picks.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ss_edgar.client import EdgarClient, EdgarFiling
from ss_edgar.funds import FundInfo, name_to_ticker, normalize_issuer_name
from ss_edgar.parsing import HoldingRow, parse_13f_xml


def fetch_fund_holdings(
    client: EdgarClient,
    fund: FundInfo,
    *,
    min_period: str = '2010-01-01',
    universe: Iterable[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch + parse all 13F-HR filings for `fund` since `min_period`.

    Returns a long-form DataFrame with columns
    `[period, fund_short, ticker, value_dollars]`. Holdings whose
    issuer name doesn't map to a ticker are dropped silently. If
    `universe` is provided, holdings outside the universe are also
    dropped.

    Caches submissions list + each filing XML to disk via the client.
    Subsequent calls hit the cache.
    """
    universe_set = set(t.upper() for t in universe) if universe is not None else None
    if verbose:
        print(f'  [{fund.short}] fetching submissions list for CIK {fund.cik}...',
              flush=True)
    filings = client.list_13f_hr_filings(fund.cik)
    filings = [f for f in filings if f.period_of_report >= min_period]
    if verbose:
        print(f'  [{fund.short}] {len(filings)} 13F-HR filings since {min_period}',
              flush=True)

    rows: list[dict] = []
    for i, f in enumerate(filings):
        try:
            xml = client.fetch_info_table(f)
        except Exception as e:
            if verbose:
                print(f'  [{fund.short}] {f.accession_no} fetch failed: {e}',
                      flush=True)
            continue
        if not xml:
            continue
        holdings = parse_13f_xml(xml, period_of_report=f.period_of_report)
        for h in holdings:
            ticker = name_to_ticker(h.name_of_issuer)
            if ticker is None:
                continue
            if universe_set is not None and ticker not in universe_set:
                continue
            rows.append({
                'period':        f.period_of_report,
                'fund_short':    fund.short,
                'fund_cik':      fund.cik,
                'ticker':        ticker,
                'value_dollars': h.value_dollars,
                'shares':        h.shares,
            })
        if verbose and (i + 1) % 20 == 0:
            print(f'  [{fund.short}] processed {i + 1}/{len(filings)}',
                  flush=True)
    df = pd.DataFrame(rows, columns=[
        'period', 'fund_short', 'fund_cik', 'ticker', 'value_dollars', 'shares',
    ])
    if df.empty:
        return df
    df['period'] = pd.to_datetime(df['period'])
    return df


def build_holdings_panel(
    client: EdgarClient,
    funds: list[FundInfo],
    *,
    min_period: str = '2010-01-01',
    universe: Iterable[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate holdings across funds × quarters into two wide panels.

    Returns `(value_panel, count_panel)`:

    - `value_panel[period, ticker]` = total dollar value of holdings
      summed across funds (NOT robust to per-filer value-column
      scaling differences; see note below).
    - `count_panel[period, ticker]` = number of distinct funds
      reporting any holding in the ticker that quarter.

    **Use `count_panel` for consensus ranking.** Pre-2023 SEC 13F-HR
    filings reported value in *thousands of dollars*; FY2023+ filings
    report raw dollars. Some filers (e.g., Berkshire 2022-Q4) adopted
    raw-dollar reporting before the official mandate. Without per-
    filer scaling detection, summing value across filers conflates
    units. The count panel is unit-agnostic and a more natural
    definition of "consensus" anyway: 5 funds reporting a ticker is
    a stronger conviction signal than 1 fund reporting a large
    position.

    Tickers not in `universe` are dropped. Quarter rows where no fund
    reported any in-universe holdings are dropped.
    """
    long_df = pd.concat([
        fetch_fund_holdings(client, f, min_period=min_period,
                            universe=universe, verbose=verbose)
        for f in funds
    ], ignore_index=True)
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    value_panel = (long_df
                   .groupby(['period', 'ticker'], as_index=False)['value_dollars']
                   .sum()
                   .pivot(index='period', columns='ticker', values='value_dollars')
                   .fillna(0.0)
                   .sort_index())
    count_panel = (long_df
                   .groupby(['period', 'ticker'])['fund_short']
                   .nunique()
                   .unstack(fill_value=0)
                   .sort_index())
    value_panel.index.name = 'period'
    count_panel.index.name = 'period'
    # Align columns
    cols = sorted(set(value_panel.columns) | set(count_panel.columns))
    value_panel = value_panel.reindex(columns=cols, fill_value=0.0)
    count_panel = count_panel.reindex(columns=cols, fill_value=0)
    return value_panel, count_panel


def build_consensus_top_k(
    count_panel: pd.DataFrame, *, top_k: int = 20,
) -> pd.DataFrame:
    """Per-quarter binary "is this ticker in the top-K consensus?" flag.

    `count_panel` is the per-quarter fund-count panel from
    `build_holdings_panel` — `count_panel[period, ticker]` = number
    of funds holding that ticker.

    Returns a DataFrame indexed by `period_of_report`, columns are
    tickers (same set as `count_panel`), values are 1.0 if the
    ticker is in the top-K most-broadly-held names that quarter,
    else 0.0.

    Ties broken by ticker alphabetic order (deterministic so the
    consensus picks are reproducible). When fewer than `top_k`
    tickers have any holdings, all held tickers are selected.

    The CFR `Top13FConsensusMode` reads this panel by `period <=
    current_bar_date` to get the most recent quarter's consensus,
    then EW-portfolios over the flagged names.
    """
    if count_panel.empty:
        return pd.DataFrame()
    out = pd.DataFrame(0.0, index=count_panel.index,
                       columns=count_panel.columns)
    for period, row in count_panel.iterrows():
        # Top-K by count, ascending-ticker tie-break
        # sort_values is stable; sort by ticker first, then by count desc
        ranked = row.sort_index().sort_values(ascending=False, kind='stable')
        # Drop tickers with zero count
        ranked = ranked[ranked > 0]
        top = ranked.head(top_k).index
        out.loc[period, top] = 1.0
    return out


__all__ = [
    'fetch_fund_holdings',
    'build_holdings_panel',
    'build_consensus_top_k',
]
