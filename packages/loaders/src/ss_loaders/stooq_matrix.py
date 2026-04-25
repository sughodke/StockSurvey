"""Wide-DataFrame loader for the Stooq daily-archive bulk dump.

Stooq (`https://stooq.com/db/h/`) ships a single zip per market that
unpacks to a 3-level directory tree:

    daily/us/<exchange> <type>/<bucket>/<ticker>.us.txt

with one text file per ticker. Files are angle-bracketed CSV:

    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    AAPL.US,D,19840907,000000,0.0991725,...,99242379,0

Prices are already split- and dividend-adjusted (AAPL in 1984 reads
as ~$0.10, the post-2014-split adjusted level), and the archive
includes delisted tickers — the two concrete things the Kaggle
Nasdaq3347 dump lacked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Stooq's column header is angle-bracketed; we strip them on read.
_STOOQ_COLS = {
    '<TICKER>': 'ticker',
    '<PER>': 'per',
    '<DATE>': 'date',
    '<TIME>': 'time',
    '<OPEN>': 'open',
    '<HIGH>': 'high',
    '<LOW>': 'low',
    '<CLOSE>': 'close',
    '<VOL>': 'volume',
    '<OPENINT>': 'openint',
}


def _read_stooq_file(path: Path) -> pd.DataFrame | None:
    """Parse one Stooq ticker file into a DataFrame indexed by date.

    Returns None for files that fail to parse or contain no rows —
    Stooq's archive has a few corrupted files that we silently skip.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.rename(columns=_STOOQ_COLS)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    return df[['open', 'high', 'low', 'close', 'volume']]


def _iter_ticker_files(data_dir: Path, include_etfs: bool) -> list[Path]:
    """Walk the Stooq layout and return all ticker file paths.

    `include_etfs=False` keeps only `<exchange> stocks` subdirectories;
    set True to also pull in `<exchange> etfs`. Stooq's tree is exactly
    `daily/<country>/<exchange> <type>/<bucket>/`, so we filter on the
    second-level directory name.
    """
    paths: list[Path] = []
    for country_dir in sorted(data_dir.glob('daily/*')):
        if not country_dir.is_dir():
            continue
        for section_dir in sorted(country_dir.iterdir()):
            if not section_dir.is_dir():
                continue
            name_lower = section_dir.name.lower()
            is_etf = name_lower.endswith('etfs')
            if is_etf and not include_etfs:
                continue
            for bucket in sorted(section_dir.iterdir()):
                if bucket.is_dir():
                    paths.extend(sorted(bucket.glob('*.txt')))
    return paths


def _ticker_from_path(path: Path) -> str:
    """`aapl.us.txt` -> `AAPL`. Strips the `.us.txt` suffix and uppercases.

    Some Stooq tickers contain dashes (e.g. `brk-a.us.txt` -> `BRK-A`).
    We preserve those as-is rather than aliasing to `BRK.A` or similar.
    """
    name = path.name
    for suffix in ('.us.txt', '.txt'):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.upper()


def load_stooq_matrix(
    data_dir: str | os.PathLike,
    *,
    min_history: int = 504,
    start_date: str | None = None,
    end_date: str | None = None,
    include_etfs: bool = False,
    cache_path: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Stooq daily archive into aligned wide DataFrames.

    Parameters
    ----------
    data_dir :
        Root of the unpacked archive — the directory that *contains*
        the `daily/` subtree (e.g. `./StooqData`).
    min_history :
        Drop tickers with fewer than this many close observations.
        Same semantic as `load_price_matrix`; default 504 = 2y.
    start_date, end_date :
        Optional ISO date strings to slice the loaded panel to.
    include_etfs :
        Include `<exchange> etfs` subtrees in addition to stocks.
        Default False (the regime strategy targets equities).
    cache_path :
        If set, store/restore the merged panel as a pickled DataFrame
        at this path. Skips the 12K-file scan on subsequent calls.
        Cache is invalidated manually (delete the file) — we don't
        track source file mtimes. Pickle keeps the loader free of a
        pyarrow/fastparquet dependency, and the file is local-only.

    Returns
    -------
    (close, high, low, volume) : four wide DataFrames
        DatetimeIndex, columns are tickers, all aligned. Volume is
        available because Stooq exports it; the Kaggle Nasdaq dump
        did not.
    """
    data_dir = Path(data_dir)
    cache = Path(cache_path) if cache_path else None

    if cache and cache.exists():
        merged = pd.read_pickle(cache)
    else:
        merged = _scan_archive(data_dir, include_etfs=include_etfs)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            merged.to_pickle(cache)

    # Pivot the long-form merged frame into four wide panels. Without
    # `index='date'` pandas would default to the row's integer index
    # and produce a meaningless panel.
    closes = merged.pivot(index='date', columns='ticker', values='close').sort_index()
    highs = merged.pivot(index='date', columns='ticker', values='high').sort_index()
    lows = merged.pivot(index='date', columns='ticker', values='low').sort_index()
    volumes = merged.pivot(index='date', columns='ticker', values='volume').sort_index()

    # Apply the min_history filter on the close panel; the others
    # follow whatever survives.
    coverage = closes.notna().sum(axis=0)
    keep = coverage[coverage >= min_history].index
    closes = closes[keep]
    highs = highs[keep]
    lows = lows[keep]
    volumes = volumes[keep]

    if start_date:
        closes, highs, lows, volumes = (
            df.loc[start_date:] for df in (closes, highs, lows, volumes))
    if end_date:
        closes, highs, lows, volumes = (
            df.loc[:end_date] for df in (closes, highs, lows, volumes))

    # Same 80%-coverage rule the Kaggle loader applies to handle tickers
    # that exist over the full window but only trade for part of it.
    min_valid = int(len(closes) * 0.8)
    closes = closes.dropna(axis=1, thresh=min_valid)
    common = closes.columns
    highs = highs[common]
    lows = lows[common]
    volumes = volumes[common]

    closes = closes.ffill().dropna()
    highs = highs.ffill().dropna()
    lows = lows.ffill().dropna()
    volumes = volumes.ffill().fillna(0).loc[closes.index]

    common_idx = closes.index.intersection(highs.index).intersection(lows.index)
    closes = closes.loc[common_idx]
    highs = highs.loc[common_idx]
    lows = lows.loc[common_idx]
    volumes = volumes.loc[common_idx]

    print(f'Stooq panel: {closes.shape[0]} dates x {closes.shape[1]} tickers')
    print(f'Date range:  {closes.index[0].date()} -> {closes.index[-1].date()}')
    return closes, highs, lows, volumes


def _scan_archive(data_dir: Path, *, include_etfs: bool) -> pd.DataFrame:
    """Walk every ticker file and concat into a single long-form frame.

    Long-form (ticker, date, OHLCV) is cheaper to round-trip through
    parquet than four separate wide pivots, since wide pivots blow up
    to 12K columns and most cells are NaN. The caller pivots once on
    load; the cache stays compact.
    """
    paths = _iter_ticker_files(data_dir, include_etfs=include_etfs)
    if not paths:
        raise RuntimeError(
            f'no ticker files found under {data_dir} — expected a '
            'Stooq layout like daily/<country>/<exchange>/<bucket>/*.txt')

    frames: list[pd.DataFrame] = []
    for path in tqdm(paths, desc='Loading Stooq', unit='file'):
        df = _read_stooq_file(path)
        if df is None or df.empty:
            continue
        df = df.assign(ticker=_ticker_from_path(path))
        frames.append(df.reset_index())

    if not frames:
        raise RuntimeError(f'all {len(paths)} ticker files failed to parse')
    return pd.concat(frames, ignore_index=True)
