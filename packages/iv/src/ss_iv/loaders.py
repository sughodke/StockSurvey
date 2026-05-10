"""Loaders for free implied-vol datasets.

Two sources supported, each with `load_*` returning a `(dates × tickers)`
DataFrame of ATM implied vol in fractional form (0.30 = 30% annualized):

  * `load_atm_iv()` — gauss314/options-IV-SP500 on Hugging Face. Daily,
    S&P 500 universe, 2019-10-14 → 2023-07-28. Single ~500MB CSV; values
    arrive in *percent* form here and are converted to fraction.
  * `load_dolthub_iv(tickers, ...)` — post-no-preference/options on
    DoltHub. Weekly Saturday snapshots, 2,276 US tickers,
    2019-02-09 → 2026-04-30. Fetched per-ticker via the DoltHub HTTP
    API (multi-ticker filters time out server-side). Each per-ticker
    response is cached as JSON under `cache_dir`.

Both helpers cache on disk; pass `refresh=True` to re-pull.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HF_URL = (
    'https://huggingface.co/datasets/gauss314/options-IV-SP500/'
    'resolve/main/data_IV_USA.csv'
)
DOLTHUB_API = (
    'https://www.dolthub.com/api/v1alpha1/post-no-preference/options'
)
DEFAULT_CACHE_DIR = Path('.iv-cache')


def load_atm_iv(
    cache_dir: str | Path | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a `(dates × tickers)` DataFrame of ATM implied vol (%).

    First call downloads the ~hundreds-of-MB CSV from Hugging Face;
    subsequent calls read from `cache_dir/data_IV_USA.csv` directly.
    Values are already annualized percent (28.09 = 28.09% annualized).
    Multiply by 0.01 if you need the fractional form.
    """
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    csv_path = cache / 'data_IV_USA.csv'

    if refresh or not csv_path.exists():
        print(f'[iv_data] downloading {HF_URL} → {csv_path}')
        urllib.request.urlretrieve(HF_URL, csv_path)
        size_mb = csv_path.stat().st_size / 1e6
        print(f'[iv_data] downloaded {size_mb:.1f} MB')

    df = pd.read_csv(
        csv_path,
        usecols=['symbol', 'date', 'ATM_IV'],
        parse_dates=['date'],
    )
    pivot = df.pivot_table(
        index='date', columns='symbol', values='ATM_IV', aggfunc='last')
    pivot = pivot.sort_index() / 100.0     # percent → fraction
    return pivot


DOLTHUB_DEFAULT_YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)


def _fetch_dolthub_ticker(
    symbol: str, *, years: tuple[int, ...] = DOLTHUB_DEFAULT_YEARS,
) -> pd.DataFrame:
    """Pull `volatility_history` rows for one ticker via DoltHub HTTP API.

    Multi-symbol IN-filters and full-history single-symbol queries both
    hit the server's query-deadline cap (~50 rows returned then aborted).
    We chunk by calendar year; each year is ~52 rows and finishes well
    inside the cutoff. Returns the concatenated, deduped, sorted frame.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        sql = (
            f"SELECT date, iv_current, hv_current FROM volatility_history "
            f"WHERE act_symbol = '{symbol}' "
            f"AND date >= '{year}-01-01' AND date < '{year + 1}-01-01' "
            f"ORDER BY date"
        )
        url = f"{DOLTHUB_API}?{urllib.parse.urlencode({'q': sql})}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = json.loads(r.read())
        except Exception as e:
            print(f'  ! {symbol} {year}: {e}')
            continue
        if payload.get('query_execution_status') != 'Success':
            # Many queries return partial rows + error status; keep what
            # came back rather than discarding.
            print(f'  ~ {symbol} {year}: '
                  f"{payload.get('query_execution_message', '')[:80]}")
        rows = payload.get('rows', [])
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=['date', 'iv_current', 'hv_current'])
    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    df['hv_current'] = pd.to_numeric(df['hv_current'], errors='coerce')
    df = (df.drop_duplicates('date', keep='last')
            .sort_values('date').reset_index(drop=True))
    return df


def load_dolthub_iv_parquet(
    parquet_path: str | Path | None = None,
    *,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Read DoltHub `volatility_history` from a local parquet file.

    The parquet is produced by:
        cd .iv-cache/options && dolt sql -r parquet \\
            -q 'SELECT date, act_symbol, iv_current, hv_current
                FROM volatility_history' > ../volatility_history.parquet

    Returns a `(dates × tickers)` DataFrame of weekly ATM IV (fraction).
    Pass `tickers=` to filter columns; otherwise returns the full panel
    of 2,276 names. The data is **weekly Saturday snapshots** — at
    daily-rebalance time, forward-fill via
    `df.reindex(daily_index).ffill(limit=7)`.
    """
    if parquet_path is None:
        parquet_path = DEFAULT_CACHE_DIR / 'volatility_history.parquet'
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f'IV parquet not found at {parquet_path}. Build it from the '
            f'DoltHub clone:\n  cd .iv-cache/options && dolt sql -r parquet '
            f'-q \'SELECT date, act_symbol, iv_current, hv_current FROM '
            f'volatility_history\' > ../volatility_history.parquet')
    df = pd.read_parquet(parquet_path,
                         columns=['date', 'act_symbol', 'iv_current'])
    if tickers is not None:
        df = df[df['act_symbol'].isin(tickers)]
    df['date'] = pd.to_datetime(df['date'])
    df['iv_current'] = pd.to_numeric(df['iv_current'], errors='coerce')
    pivot = df.pivot_table(
        index='date', columns='act_symbol', values='iv_current', aggfunc='last')
    return pivot.sort_index()


def load_dolthub_iv(
    tickers: list[str],
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    sleep_between: float = 0.05,
) -> pd.DataFrame:
    """Return a `(dates × tickers)` DataFrame of weekly ATM IV (fraction).

    HTTP-API path (slow, truncates responses on full-history queries —
    see ``load_dolthub_iv_parquet`` for the local-parquet fast path).
    Each ticker's response is cached as `cache_dir/dolthub/<symbol>.json`;
    subsequent calls hit disk only. Values are fractional.
    """
    cache = (Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR) / 'dolthub'
    cache.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for i, sym in enumerate(tickers):
        cache_path = cache / f'{sym}.json'
        if refresh or not cache_path.exists():
            print(f'[iv_data] fetching {sym} from DoltHub ({i + 1}/{len(tickers)})')
            try:
                df = _fetch_dolthub_ticker(sym)
            except Exception as e:
                print(f'  ! {sym}: {e}')
                df = pd.DataFrame(columns=['date', 'iv_current', 'hv_current'])
            df.to_json(cache_path, orient='records', date_format='iso')
            time.sleep(sleep_between)
        else:
            df = pd.read_json(cache_path, convert_dates=['date'])
        if len(df) == 0:
            continue
        df = df[['date', 'iv_current']].copy()
        df['symbol'] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(
        index='date', columns='symbol', values='iv_current', aggfunc='last')
    return pivot.sort_index()
