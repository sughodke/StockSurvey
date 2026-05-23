"""Local IV/HV history cache for the 4-week-change features.

The vol-v3 predictor needs `iv_change_4w` and `hv_change_4w` —
today's value minus the value 4 weeks ago. In live mode we maintain a
local parquet file that gets one row appended per run.

Bootstrap path: the *first* live run has no 4-week-ago history (the
cache is empty). Two options to bootstrap:

  1. Backfill from DoltHub's existing `volatility_history.parquet`
     (subset for our universe; truncate to the last ~6 weeks).
  2. Start fresh, defer the predictor for 4 weeks, fire only the
     gate-then-passive part of the strategy until the cache fills.

Default behavior is `bootstrap_from_dolthub=True` if the parquet
exists; the script `apps/vol/scripts/build_vol_checkpoint.py` calls
the same backfill so a fresh deployment is ready on day 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CACHE_PATH: str = '.iv-cache/vol-live-history.parquet'
DOLTHUB_PATH: str = '.iv-cache/volatility_history.parquet'


def append_snapshot(
    iv_current: pd.Series, hv_current: pd.Series, *,
    as_of: pd.Timestamp | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> None:
    """Append today's (iv, hv) snapshot to the local history.

    `iv_current` and `hv_current` are 1D pandas Series indexed by
    symbol. Existing rows for the same `as_of` are overwritten.
    """
    if as_of is None:
        as_of = pd.Timestamp.utcnow().normalize()
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    long_iv = iv_current.rename('iv_current').to_frame().assign(
        date=as_of, kind='iv').reset_index().rename(columns={'index': 'symbol'})
    long_hv = hv_current.rename('hv_current').to_frame().assign(
        date=as_of, kind='hv').reset_index().rename(columns={'index': 'symbol'})
    df = pd.concat([
        long_iv.rename(columns={'iv_current': 'value'})[['date', 'symbol', 'kind', 'value']],
        long_hv.rename(columns={'hv_current': 'value'})[['date', 'symbol', 'kind', 'value']],
    ], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])

    if path.exists():
        existing = pd.read_parquet(path)
        # Drop any prior snapshot at this same date.
        existing = existing[existing['date'] != pd.to_datetime(as_of)]
        df = pd.concat([existing, df], ignore_index=True)
    df = df.sort_values(['date', 'symbol', 'kind']).reset_index(drop=True)
    df.to_parquet(path, index=False)


def load_history(
    cache_path: str | Path = DEFAULT_CACHE_PATH, *,
    end: pd.Timestamp | None = None, n_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the recent IV + HV histories.

    Returns `(iv_history, hv_history)` as wide DataFrames
    (index=date, columns=symbol). `n_weeks` controls how much back-
    history to include — default 8 weeks is enough for the 4-week
    diff with a margin.

    Returns empty DataFrames if the cache doesn't exist yet.
    """
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    if end is not None:
        df = df[df['date'] <= pd.to_datetime(end)]
    if not df.empty:
        end_actual = df['date'].max()
        start = end_actual - pd.Timedelta(weeks=n_weeks)
        df = df[df['date'] >= start]
    iv = df[df['kind'] == 'iv'].pivot(index='date', columns='symbol', values='value')
    hv = df[df['kind'] == 'hv'].pivot(index='date', columns='symbol', values='value')
    return iv.sort_index(), hv.sort_index()


def bootstrap_from_dolthub(
    universe: list[str], *, dolthub_path: str | Path = DOLTHUB_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    weeks_back: int = 8,
) -> int:
    """Copy the last `weeks_back` weeks of DoltHub history into the
    local cache for `universe`. Returns the number of rows written.

    Called once at deployment to seed the 4-week-change features so
    the first live run already has the lookback. Subsequent runs only
    need to call `append_snapshot` daily.
    """
    src = Path(dolthub_path)
    if not src.exists():
        return 0
    df = pd.read_parquet(src)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'act_symbol': 'symbol'})
    df = df[df['symbol'].isin(universe)]
    if df.empty:
        return 0

    cutoff = df['date'].max() - pd.Timedelta(weeks=weeks_back)
    df = df[df['date'] >= cutoff]
    df = df[['date', 'symbol', 'iv_current', 'hv_current']].dropna()
    df = df[(df['iv_current'] > 0) & (df['hv_current'] > 0)]
    long_iv = df[['date', 'symbol', 'iv_current']].rename(
        columns={'iv_current': 'value'}).assign(kind='iv')
    long_hv = df[['date', 'symbol', 'hv_current']].rename(
        columns={'hv_current': 'value'}).assign(kind='hv')
    out = pd.concat([long_iv, long_hv], ignore_index=True)
    out = out.sort_values(['date', 'symbol', 'kind']).reset_index(drop=True)

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return int(out.shape[0])


__all__ = [
    'DEFAULT_CACHE_PATH', 'DOLTHUB_PATH',
    'append_snapshot', 'load_history', 'bootstrap_from_dolthub',
]
