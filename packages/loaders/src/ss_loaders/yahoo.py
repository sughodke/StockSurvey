"""Yahoo Finance single-ticker loader via yfinance."""

from __future__ import annotations

import datetime

import pandas as pd


def load_yahoo(
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    ticker: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV + adjusted close for one ticker.

    Returns a DataFrame indexed by date with lowercase columns:
    open, high, low, close, adj_close, volume.
    """
    import yfinance as yf

    df = yf.download(ticker, start=startdate, end=enddate,
                     auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.rename(str.lower, axis='columns', inplace=True)
    df.rename(columns={'adj close': 'adj_close'}, inplace=True)
    df.set_index(pd.to_datetime(df.index), inplace=True)
    return df
