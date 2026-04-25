"""Ingest a Stooq daily archive into a parquet event store.

Walks the same `daily/<country>/<section>/<bucket>/*.txt` layout that
`ss_loaders.load_stooq_matrix` reads, but instead of pivoting to a
wide panel it writes two parquet files:

  * `<dst>/instruments.parquet`  — per-ticker metadata (exchange,
    asset class, listing date, last-seen date, bar count). The
    listing/last-seen pair is the structural piece that lets
    downstream code build a point-in-time universe.
  * `<dst>/bars/data.parquet`    — long-form OHLCV, one row per
    (date, ticker), sorted by date then ticker. Compact enough to
    keep as a single file at ~12K-ticker scale.

We deliberately do not pivot to wide here: the wide form is a view,
constructible cheaply on demand, and the long form survives
universe churn (NaN-padding a 12K-column matrix forever wastes both
disk and memory).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ss_loaders import (
    iter_stooq_ticker_files,
    read_stooq_file,
    stooq_ticker_from_path,
)


def _classify_path(path: Path, src: Path) -> tuple[str, str, str]:
    """Extract `(country, exchange, asset_class)` from a Stooq file path.

    Path shape: `<src>/daily/<country>/<exchange> <asset_class>/<bucket?>/<file>`.
    The `<exchange> <asset_class>` directory is split on the first space —
    Stooq uses names like `nasdaq stocks`, `nyse etfs`, `nysemkt stocks`.
    Any unexpected layout falls back to `'unknown'` so the ingest doesn't
    abort on a single odd file.
    """
    rel = path.relative_to(src).parts
    # rel[0] is 'daily', rel[1] is country, rel[2] is the section dir.
    country = rel[1] if len(rel) >= 3 else 'unknown'
    section = rel[2] if len(rel) >= 3 else 'unknown'
    if ' ' in section:
        exchange, asset_class = section.split(' ', 1)
    else:
        exchange, asset_class = section, 'unknown'
    return country, exchange, asset_class


def ingest_stooq(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    include_etfs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a parquet event store from a Stooq archive at `src`.

    Parameters
    ----------
    src :
        Directory containing the `daily/` subtree (e.g. `./StooqData`).
    dst :
        Output directory. Created if missing. Existing parquet files
        are overwritten.
    include_etfs :
        Include `<exchange> etfs` subtrees in addition to stocks.
        Default True — ETFs are useful both as benchmarks and as
        tradeable instruments themselves; the matrix loader excludes
        them by default but the stream layer is happy to carry them.

    Returns
    -------
    (instruments, bars) DataFrames, also persisted under `dst`.
    """
    src = Path(src)
    dst = Path(dst)

    paths = iter_stooq_ticker_files(src, include_etfs=include_etfs)
    if not paths:
        raise RuntimeError(
            f'no ticker files found under {src} — expected a Stooq layout '
            'like daily/<country>/<exchange> <asset_class>/<bucket>/*.txt')

    instrument_rows: list[dict] = []
    bar_frames: list[pd.DataFrame] = []

    for path in tqdm(paths, desc='Ingesting Stooq', unit='file'):
        df = read_stooq_file(path)
        if df is None or df.empty:
            continue
        ticker = stooq_ticker_from_path(path)
        country, exchange, asset_class = _classify_path(path, src)

        df = df.reset_index()
        df['ticker'] = ticker
        bar_frames.append(df)

        instrument_rows.append({
            'ticker': ticker,
            'country': country,
            'exchange': exchange,
            'asset_class': asset_class,
            'listing_date': df['date'].min(),
            'last_seen_date': df['date'].max(),
            'n_bars': len(df),
        })

    if not bar_frames:
        raise RuntimeError(f'all {len(paths)} ticker files failed to parse')

    bars = pd.concat(bar_frames, ignore_index=True)
    bars = bars[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    bars = bars.sort_values(['date', 'ticker']).reset_index(drop=True)

    # Compact dtypes. Float32 prices are accurate to ~7 significant
    # figures, comfortably above the 4-decimal precision Stooq emits;
    # halves the parquet size vs float64.
    for col in ('open', 'high', 'low', 'close'):
        bars[col] = bars[col].astype('float32')
    bars['volume'] = bars['volume'].fillna(0).astype('int64')

    instruments = pd.DataFrame(instrument_rows)
    instruments = instruments.sort_values(
        ['country', 'exchange', 'asset_class', 'ticker']
    ).reset_index(drop=True)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'bars').mkdir(exist_ok=True)
    bars.to_parquet(dst / 'bars' / 'data.parquet', index=False, compression='zstd')
    instruments.to_parquet(dst / 'instruments.parquet', index=False, compression='zstd')

    print(
        f'Wrote {len(bars):,} bars across {len(instruments):,} instruments '
        f'({instruments.listing_date.min().date()} -> '
        f'{instruments.last_seen_date.max().date()}) to {dst}')
    return instruments, bars
