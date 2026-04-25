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


def read_stooq_file(path: Path) -> pd.DataFrame | None:
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


def iter_stooq_ticker_files(data_dir: Path, include_etfs: bool) -> list[Path]:
    """Walk the Stooq layout and return all ticker file paths.

    `include_etfs=False` keeps only `<exchange> stocks` subdirectories;
    set True to also pull in `<exchange> etfs`. Stooq's tree is
    `daily/<country>/<exchange> <type>/[<bucket>/]*.txt`. Sections with
    >2K tickers are sharded into numeric `1/`, `2/`, ... bucket dirs;
    smaller sections (e.g. `nasdaq etfs`, `nysemkt stocks`) keep all
    .txt files directly under the section dir, with no bucket level.
    `rglob` covers both layouts.
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
            paths.extend(sorted(section_dir.rglob('*.txt')))
    return paths


def stooq_ticker_from_path(path: Path) -> str:
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
    min_history: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
    include_etfs: bool = False,
    cache_path: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Stooq daily archive into aligned wide DataFrames.

    Survivorship-bias note: this loader returns a **point-in-time**
    panel — tickers that started trading partway through the date
    range have leading NaN; tickers that delisted during the range
    have trailing NaN. The downstream trainer applies per-walk-forward
    filtering (see `regime.trainer.train`'s `per_window_min_history`)
    to define each window's tradeable universe, instead of imposing
    a panel-wide "must exist for the entire range" rule that would
    silently drop everything that delisted.

    Parameters
    ----------
    data_dir :
        Root of the unpacked archive — the directory that *contains*
        the `daily/` subtree (e.g. `./StooqData`).
    min_history :
        Lenient panel-wide ghost filter — drops tickers with fewer
        than this many valid close observations *anywhere in the
        sliced date range*. Default 252 (≈1 trading year). The real
        per-window survivorship rule is applied downstream in the
        trainer; this is just a sanity floor to drop tickers that
        appear in the Stooq archive but never traded meaningfully
        in the requested range.
    start_date, end_date :
        Optional ISO date strings to slice the loaded panel to.
        `min_history` is enforced *after* this slice, so a ticker
        that traded heavily before `start_date` but barely during
        the requested range still gets dropped.
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
        did not. NaN appears where the ticker wasn't trading on a
        given date — the downstream causal CWT requires per-ticker
        cumsum so the trainer must drop columns with leading NaN
        per-window before computing scores; see
        `regime.trainer._filter_window_universe`.
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

    # Slice to the requested date range FIRST. Doing this before the
    # min_history filter means "min_history valid bars *in this range*"
    # rather than "min_history valid bars across all of history" — a
    # name that traded heavily before 2000 but barely after still gets
    # dropped from a 2000-onward run.
    if start_date:
        closes, highs, lows, volumes = (
            df.loc[start_date:] for df in (closes, highs, lows, volumes))
    if end_date:
        closes, highs, lows, volumes = (
            df.loc[:end_date] for df in (closes, highs, lows, volumes))

    # Lenient ghost filter — drop tickers with truly minimal coverage in
    # the requested range. The trainer's per-walk-forward filter does
    # the strict survivorship enforcement; this just trims the panel
    # so we don't carry around 10K columns of mostly-NaN.
    coverage = closes.notna().sum(axis=0)
    keep = coverage[coverage >= min_history].index
    closes = closes[keep]
    highs = highs[keep]
    lows = lows[keep]
    volumes = volumes[keep]

    # ffill *short* gaps only (1-trading-week halts, holidays missed in
    # the source). Longer gaps stay NaN — leading NaN for tickers that
    # IPO'd partway through the range, trailing NaN for tickers that
    # delisted partway through. The downstream causal CWT cumsum can't
    # tolerate leading NaN, so the trainer drops columns with leading
    # NaN per-window before computing scores — that's where the actual
    # point-in-time tradeable universe is constructed.
    closes = closes.ffill(limit=5)
    highs = highs.ffill(limit=5)
    lows = lows.ffill(limit=5)
    volumes = volumes.ffill(limit=5).fillna(0)

    common_idx = closes.index.intersection(highs.index).intersection(lows.index)
    closes = closes.loc[common_idx]
    highs = highs.loc[common_idx]
    lows = lows.loc[common_idx]
    volumes = volumes.loc[common_idx]

    print(f'Stooq panel: {closes.shape[0]} dates x {closes.shape[1]} tickers '
          f'(point-in-time; per-window filter applied downstream)')
    print(f'Date range:  {closes.index[0].date()} -> {closes.index[-1].date()}')
    return closes, highs, lows, volumes


def _scan_archive(data_dir: Path, *, include_etfs: bool) -> pd.DataFrame:
    """Walk every ticker file and concat into a single long-form frame.

    Long-form (ticker, date, OHLCV) is cheaper to round-trip through
    parquet than four separate wide pivots, since wide pivots blow up
    to 12K columns and most cells are NaN. The caller pivots once on
    load; the cache stays compact.
    """
    paths = iter_stooq_ticker_files(data_dir, include_etfs=include_etfs)
    if not paths:
        raise RuntimeError(
            f'no ticker files found under {data_dir} — expected a '
            'Stooq layout like daily/<country>/<exchange>/<bucket>/*.txt')

    frames: list[pd.DataFrame] = []
    for path in tqdm(paths, desc='Loading Stooq', unit='file'):
        df = read_stooq_file(path)
        if df is None or df.empty:
            continue
        df = df.assign(ticker=stooq_ticker_from_path(path))
        frames.append(df.reset_index())

    if not frames:
        raise RuntimeError(f'all {len(paths)} ticker files failed to parse')
    return pd.concat(frames, ignore_index=True)
