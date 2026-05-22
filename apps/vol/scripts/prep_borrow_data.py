"""B1 Phase A prep — build the per-(date,symbol) borrow-stress composite.

Pulls FINRA daily short-volume + SEC fails-to-deliver over the gauss314 span
(2019-10 → 2023-07), restricts to the gauss314 symbol set, and builds a
cross-sectionally-z-scored borrow-stress composite:

    borrow_stress[date, sym] = z(short_ratio) + z(log1p(ftd_fails))   (per date)

Cached to `.iv-cache/borrow_composite.parquet` (resumable per source). Local,
free, no Modal. The Stage-0 probe already confirmed coverage 99% + real
dispersion; this is the full pull for Phase A.

    uv run python apps/vol/scripts/prep_borrow_data.py
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
GAUSS = REPO / '.iv-cache' / 'data_IV_USA.csv'
CACHE = REPO / '.iv-cache'
FINRA_CACHE = CACHE / 'borrow_finra.parquet'
FTD_CACHE = CACHE / 'borrow_ftd.parquet'
OUT = CACHE / 'borrow_composite.parquet'
UA = {'User-Agent': 'StockSurvey research (sid.ghodke@gmail.com)'}


def _finra_day(date: pd.Timestamp, symbols: set[str]) -> pd.DataFrame | None:
    url = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date:%Y%m%d}.txt'
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pd.read_csv(io.StringIO(r.text), sep='|')
        df = df[df['Symbol'].notna()].copy()
        df['Symbol'] = df['Symbol'].str.upper()
        df = df[df['Symbol'].isin(symbols) & (df['TotalVolume'] > 0)]
        df['short_ratio'] = df['ShortVolume'] / df['TotalVolume']
        df['date'] = date
        return df[['date', 'Symbol', 'short_ratio']]
    except Exception:
        return None


def _ftd_halfmonth(year: int, month: int, half: str, symbols: set[str]) -> pd.DataFrame | None:
    url = f'https://www.sec.gov/files/data/fails-deliver-data/cnsfails{year}{month:02d}{half}.zip'
    try:
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            return None
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(zf.open(zf.namelist()[0]), sep='|', encoding='latin-1',
                         on_bad_lines='skip')
        df.columns = [c.strip().upper() for c in df.columns]
        sc = next((c for c in df.columns if 'SYMBOL' in c), None)
        qc = next((c for c in df.columns if 'QUANTITY' in c or 'FAILS' in c), None)
        dc = next((c for c in df.columns if 'SETTLEMENT' in c or 'DATE' in c), None)
        if not (sc and qc and dc):
            return None
        df = df.rename(columns={sc: 'Symbol', qc: 'fails', dc: 'date'})
        df['Symbol'] = df['Symbol'].astype(str).str.upper()
        df = df[df['Symbol'].isin(symbols)].copy()
        df['fails'] = pd.to_numeric(df['fails'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        return df[['date', 'Symbol', 'fails']].dropna()
    except Exception:
        return None


def main() -> None:
    g = pd.read_csv(GAUSS, usecols=['symbol', 'date'])
    g['date'] = pd.to_datetime(g['date'])
    symbols = set(g['symbol'].str.upper().unique())
    dates = pd.DatetimeIndex(sorted(g['date'].unique()))
    print(f'gauss314: {len(symbols)} symbols, {len(dates)} dates '
          f'{dates[0].date()}→{dates[-1].date()}')

    # --- FINRA daily short-volume (resumable) ---
    if FINRA_CACHE.exists():
        finra = pd.read_parquet(FINRA_CACHE)
        print(f'FINRA cache: {len(finra)} rows')
    else:
        chunks, t0 = [], time.time()
        for i, d in enumerate(dates):
            r = _finra_day(d, symbols)
            if r is not None:
                chunks.append(r)
            if (i + 1) % 100 == 0:
                print(f'  finra {i+1}/{len(dates)} ({time.time()-t0:.0f}s)', flush=True)
            time.sleep(0.05)
        finra = pd.concat(chunks, ignore_index=True)
        finra.to_parquet(FINRA_CACHE)
        print(f'FINRA pulled: {len(finra)} rows, saved')

    # --- SEC FTD semi-monthly (resumable) ---
    if FTD_CACHE.exists():
        ftd = pd.read_parquet(FTD_CACHE)
        print(f'FTD cache: {len(ftd)} rows')
    else:
        months = pd.period_range(dates[0], dates[-1], freq='M')
        chunks = []
        for p in months:
            for half in ('a', 'b'):
                r = _ftd_halfmonth(p.year, p.month, half, symbols)
                if r is not None:
                    chunks.append(r)
                time.sleep(0.1)
        ftd = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=['date', 'Symbol', 'fails'])
        ftd.to_parquet(FTD_CACHE)
        print(f'FTD pulled: {len(ftd)} rows, saved')

    # --- merge onto the gauss314 trading grid ---
    grid = g.rename(columns={'symbol': 'Symbol'})
    grid['Symbol'] = grid['Symbol'].str.upper()
    m = grid.merge(finra, on=['date', 'Symbol'], how='left')
    # FTD is settlement-date; forward-fill the latest fails per symbol onto the grid.
    ftd_s = ftd.sort_values('date')
    m = pd.merge_asof(m.sort_values('date'), ftd_s, on='date', by='Symbol',
                      direction='backward', tolerance=pd.Timedelta('20D'))
    m['fails'] = m['fails'].fillna(0.0)

    # cross-sectional z per date (within the day's available names)
    def _z(s):
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / sd if sd > 1e-12 else s * 0.0
    m['z_sr'] = m.groupby('date')['short_ratio'].transform(_z)
    m['z_ftd'] = m.groupby('date')['fails'].transform(lambda s: _z(np.log1p(s)))
    m['borrow_stress'] = m['z_sr'].fillna(0.0) + m['z_ftd'].fillna(0.0)
    m['has_borrow'] = m['short_ratio'].notna()

    out = m[['date', 'Symbol', 'short_ratio', 'fails', 'borrow_stress', 'has_borrow']]
    out.to_parquet(OUT)
    cov = m['has_borrow'].mean()
    print(f'\ncomposite: {len(out)} (date,sym) rows; FINRA coverage {cov:.1%}')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
