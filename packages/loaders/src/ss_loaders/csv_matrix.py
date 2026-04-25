"""Wide-DataFrame loader for the Kaggle NASDAQ daily-CSV dataset.

The Kaggle dump (svaningelgem/nasdaq-daily-stock-prices) ships one CSV
per ticker with `ticker,date,open,high,low,close` — no volume, no
adj_close. This loader returns three aligned DataFrames (close, high,
low) indexed by date with one column per ticker.
"""

from __future__ import annotations

import os

import pandas as pd
from tqdm import tqdm


def load_price_matrix(
    data_dir: str,
    min_history: int = 504,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-ticker CSVs into aligned (close, high, low) DataFrames.

    Tickers with fewer than `min_history` observations are dropped, as are
    tickers with >20% missing in the [start_date, end_date] window.
    """
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv'))
    close_frames: dict[str, pd.Series] = {}
    high_frames: dict[str, pd.Series] = {}
    low_frames: dict[str, pd.Series] = {}

    for fname in tqdm(csv_files, desc='Loading CSVs', unit='file'):
        ticker = fname.removesuffix('.csv')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        except Exception:
            continue
        if len(df['close'].dropna()) < min_history:
            continue
        close_frames[ticker] = df['close']
        high_frames[ticker] = df['high']
        low_frames[ticker] = df['low']

    prices = pd.DataFrame(close_frames).sort_index()
    highs = pd.DataFrame(high_frames).sort_index()
    lows = pd.DataFrame(low_frames).sort_index()

    if start_date:
        prices, highs, lows = (df.loc[start_date:] for df in (prices, highs, lows))
    if end_date:
        prices, highs, lows = (df.loc[:end_date] for df in (prices, highs, lows))

    min_valid = int(len(prices) * 0.8)
    prices = prices.dropna(axis=1, thresh=min_valid)
    common = prices.columns
    highs = highs[common]
    lows = lows[common]

    prices = prices.ffill().dropna()
    highs = highs.ffill().dropna()
    lows = lows.ffill().dropna()

    common_idx = prices.index.intersection(highs.index).intersection(lows.index)
    prices = prices.loc[common_idx]
    highs = highs.loc[common_idx]
    lows = lows.loc[common_idx]

    print(f'Price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers')
    print(f'Date range:   {prices.index[0].date()} -> {prices.index[-1].date()}')
    return prices, highs, lows
