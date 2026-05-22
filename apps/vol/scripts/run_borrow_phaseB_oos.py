"""B1 Phase B — OOS confirmation of the borrow-stress premium-amplifier (H1).

Phase A discovered, IN-SAMPLE (11 fired rebals, 2019-2023), that high-borrow-
stress names carry richer realized VRP (hi-tercile alpha +0.042 vs lo +0.002,
hi-tercile Sharpe +3.80) — the OPPOSITE of the scoping page's squeeze-tail
assumption. That direction was read off the data, so per Sullivan-Timmermann-
White it is a HYPOTHESIS, not a result. This is the pre-registered OOS test on
never-seen DoltHub 2024-26 data:

  PRE-REGISTERED (flipped) hypothesis: on the liquid cohort, the
  high-borrow-stress tercile earns higher realized VRP (iv_rv_gap) than the
  low-stress tercile, OOS.

  PASS (confirms borrow leg, overweight-hi justified): hi-tercile mean VRP >
    lo-tercile OOS AND hi > lo in >= 60% of OOS quarters.
  FAIL (in-sample mirage, borrow leg closes): hi <= lo OOS, or < 60% quarters.

Substrate: DoltHub volatility_history (weekly, 2023-08 → 2026-04). No OI →
liquid cohort = the gauss314 top-200-OI symbol union (a fixed liquid-optionable
proxy). iv_rv_gap = iv_current − forward-20d realized vol (from stooq prices,
NOT DoltHub hv — avoids the autocorrelation tautology). Borrow-stress: FINRA
short-vol + SEC FTD on the OOS weekly trading days (pulled + cached here).

    uv run python apps/vol/scripts/run_borrow_phaseB_oos.py
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ss_loaders import load_stooq_matrix

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / '.iv-cache'
GAUSS = CACHE / 'data_IV_USA.csv'
DOLT = CACHE / 'volatility_history.parquet'
BORROW_OOS = CACHE / 'borrow_composite_oos.parquet'
OUT = REPO / 'Output'
UA = {'User-Agent': 'StockSurvey research (sid.ghodke@gmail.com)'}
VAL_START, VAL_END = pd.Timestamp('2023-08-01'), pd.Timestamp('2026-04-30')
FWD_DAYS = 20


def _finra(date, symbols):
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


def _ftd(year, month, half, symbols):
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
        if not (sc and qc):
            return None
        df = df.rename(columns={sc: 'Symbol', qc: 'fails'})
        df['Symbol'] = df['Symbol'].astype(str).str.upper()
        df = df[df['Symbol'].isin(symbols)].copy()
        df['fails'] = pd.to_numeric(df['fails'], errors='coerce')
        return df[['Symbol', 'fails']].dropna().groupby('Symbol', as_index=False)['fails'].mean()
    except Exception:
        return None


def _liquid_universe() -> set[str]:
    """Union of gauss314 top-200-OI names — a fixed liquid-optionable proxy."""
    g = pd.read_csv(GAUSS, usecols=['symbol', 'date', 'calls_open_interest',
                                    'puts_open_interest'])
    g['oi'] = g['calls_open_interest'].fillna(0) + g['puts_open_interest'].fillna(0)
    g['rank'] = g.groupby('date')['oi'].rank(method='first', ascending=False)
    return set(g[g['rank'] <= 200]['symbol'].str.upper().unique())


def main() -> None:
    print('loading DoltHub OOS panel ...', flush=True)
    dolt = pd.read_parquet(DOLT)
    dolt['date'] = pd.to_datetime(dolt['date'])
    dolt = dolt.rename(columns={'act_symbol': 'symbol'})
    dolt['symbol'] = dolt['symbol'].str.upper()
    for c in ('iv_current', 'hv_current'):
        dolt[c] = pd.to_numeric(dolt[c], errors='coerce')
    dolt = dolt[(dolt['date'] >= VAL_START) & (dolt['date'] <= VAL_END)
                & (dolt['iv_current'] > 0)].dropna(subset=['iv_current'])

    liquid = _liquid_universe()
    dolt = dolt[dolt['symbol'].isin(liquid)]
    syms = sorted(dolt['symbol'].unique())
    weekly = pd.DatetimeIndex(sorted(dolt['date'].unique()))
    print(f'  OOS liquid cohort: {len(syms)} symbols, {len(weekly)} weekly snapshots '
          f'{weekly[0].date()}→{weekly[-1].date()}')

    # --- forward realized vol from stooq prices (not DoltHub hv) ---
    print('  loading stooq prices for forward realized vol ...', flush=True)
    prices, _, _, _ = load_stooq_matrix('./StooqData', tickers=syms,
                                        start_date='2023-01-01', end_date='2026-05-01',
                                        min_history=30)
    logr = np.log(prices.replace(0.0, np.nan)).diff()
    didx = prices.index.sort_values()
    frv = {}
    for d in weekly:
        pos = didx.searchsorted(d, side='left')
        if pos + FWD_DAYS >= len(didx):
            continue
        w = logr.iloc[pos + 1: pos + 1 + FWD_DAYS]
        if w.shape[0] >= FWD_DAYS * 0.7:
            frv[d] = w.std() * np.sqrt(252)  # decimal annualized vol, matches DoltHub iv_current
    frv = pd.DataFrame(frv).T
    print(f'  forward realized vol computed for {len(frv)} weeks', flush=True)

    # iv_rv_gap per (week, symbol): iv_current - forward realized vol
    dolt = dolt[dolt['date'].isin(frv.index)]
    def _gap(row):
        try:
            return row['iv_current'] - frv.loc[row['date'], row['symbol']]
        except Exception:
            return np.nan
    dolt['fwd_rv'] = dolt.apply(lambda r: frv.loc[r['date']].get(r['symbol'], np.nan)
                                if r['date'] in frv.index else np.nan, axis=1)
    dolt['iv_rv_gap'] = dolt['iv_current'] - dolt['fwd_rv']
    dolt = dolt.dropna(subset=['iv_rv_gap'])
    print(f'  {len(dolt)} (week,symbol) VRP observations', flush=True)

    # --- borrow data for OOS weekly trading days (cached) ---
    fin_cache, ftd_cache = CACHE / 'borrow_oos_finra.parquet', CACHE / 'borrow_oos_ftd.parquet'
    if BORROW_OOS.exists():
        bor = pd.read_parquet(BORROW_OOS)
        print(f'  borrow OOS cache: {len(bor)} rows')
    elif fin_cache.exists() and ftd_cache.exists():
        print('  using cached raw FINRA/FTD pulls')
        fin = pd.read_parquet(fin_cache)
        ftd = pd.read_parquet(ftd_cache)
        fin['month'] = pd.to_datetime(fin['date'].values.astype('datetime64[M]'))
        bor = fin.merge(ftd, on=['Symbol', 'month'], how='left')
        bor['fails'] = bor['fails'].fillna(0.0)
        def _z(s):
            sd = s.std(ddof=0)
            return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0
        bor['z_sr'] = bor.groupby('date')['short_ratio'].transform(_z)
        bor['z_ftd'] = bor.groupby('date')['fails'].transform(lambda s: _z(np.log1p(s)))
        bor['borrow_stress'] = bor['z_sr'].fillna(0) + bor['z_ftd'].fillna(0)
        bor = bor[['date', 'Symbol', 'borrow_stress']]
        bor.to_parquet(BORROW_OOS)
    else:
        print('  pulling FINRA + FTD for OOS weekly dates ...', flush=True)
        finra_chunks = []
        for d in weekly:
            # snap to nearest prior trading day for the FINRA file
            for back in range(0, 5):
                fd = d - pd.Timedelta(days=back)
                r = _finra(fd, set(syms))
                if r is not None and len(r):
                    # key by the weekly SNAPSHOT date (join key to DoltHub),
                    # drop the FINRA file's own trading-day 'date' to avoid collision
                    r = r[['Symbol', 'short_ratio']].copy()
                    r['date'] = d
                    finra_chunks.append(r)
                    break
            time.sleep(0.05)
        fin = pd.concat(finra_chunks, ignore_index=True) if finra_chunks else pd.DataFrame()
        # FTD per covering month
        ftd_rows = []
        for p in pd.period_range(VAL_START, VAL_END, freq='M'):
            for half in ('a', 'b'):
                r = _ftd(p.year, p.month, half, set(syms))
                if r is not None:
                    r = r.copy(); r['month'] = pd.Timestamp(p.start_time)
                    ftd_rows.append(r)
                time.sleep(0.1)
        ftd = pd.concat(ftd_rows, ignore_index=True) if ftd_rows else pd.DataFrame(
            columns=['Symbol', 'fails', 'month'])
        fin.to_parquet(CACHE / 'borrow_oos_finra.parquet')
        ftd.to_parquet(CACHE / 'borrow_oos_ftd.parquet')
        # composite: z(short_ratio) + z(log1p fails) per week, cross-sectional
        fin['month'] = pd.to_datetime(fin['date'].values.astype('datetime64[M]'))
        bor = fin.merge(ftd, on=['Symbol', 'month'], how='left')
        bor['fails'] = bor['fails'].fillna(0.0)
        def _z(s):
            sd = s.std(ddof=0)
            return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0
        bor['z_sr'] = bor.groupby('date')['short_ratio'].transform(_z)
        bor['z_ftd'] = bor.groupby('date')['fails'].transform(lambda s: _z(np.log1p(s)))
        bor['borrow_stress'] = bor['z_sr'].fillna(0) + bor['z_ftd'].fillna(0)
        bor = bor[['date', 'Symbol', 'borrow_stress']]
        bor.to_parquet(BORROW_OOS)
        print(f'  borrow OOS pulled: {len(bor)} rows, saved')

    # --- join + tercile VRP per week (OOS) ---
    m = dolt.merge(bor.rename(columns={'Symbol': 'symbol'}), on=['date', 'symbol'], how='inner')
    print(f'  joined OOS VRP+borrow: {len(m)} rows ({m["date"].nunique()} weeks)')

    lo_v, hi_v, per_week = [], [], []
    for d, day in m.groupby('date'):
        if len(day) < 9:
            continue
        q = day['borrow_stress'].quantile([1/3, 2/3]).values
        lo = day[day.borrow_stress <= q[0]]['iv_rv_gap'].mean()
        hi = day[day.borrow_stress >= q[1]]['iv_rv_gap'].mean()
        lo_v.append(lo); hi_v.append(hi)
        per_week.append({'date': str(d.date()), 'lo': float(lo), 'hi': float(hi)})

    lo_m, hi_m = float(np.nanmean(lo_v)), float(np.nanmean(hi_v))
    n_weeks = len(per_week)
    hi_gt_lo_frac = float(np.mean([w['hi'] > w['lo'] for w in per_week]))
    # quarter-level
    qd = pd.DataFrame(per_week); qd['q'] = pd.PeriodIndex(pd.to_datetime(qd['date']), freq='Q')
    qg = qd.groupby('q').agg(lo=('lo', 'mean'), hi=('hi', 'mean'))
    q_hi_gt_lo = float((qg['hi'] > qg['lo']).mean())

    print(f'\n--- OOS H1 test (DoltHub 2024-26, {n_weeks} weeks, {len(qg)} quarters) ---')
    print(f'  mean realized VRP (vol pts):  lo-stress {lo_m:+.3f}   hi-stress {hi_m:+.3f}')
    print(f'  hi > lo in {hi_gt_lo_frac:.0%} of weeks ; {q_hi_gt_lo:.0%} of quarters')
    confirms = (hi_m > lo_m) and (q_hi_gt_lo >= 0.60)
    verdict = ('PASS — borrow premium-amplifier CONFIRMED OOS (overweight-hi justified)'
               if confirms else
               'FAIL — H1 does not hold OOS; the in-sample +3.80 was an 11-rebal mirage; borrow leg closes')
    print(f'\nPHASE B VERDICT: {verdict}')

    OUT.mkdir(exist_ok=True)
    (OUT / 'vol-borrow-phaseB.json').write_text(json.dumps({
        'n_weeks': n_weeks, 'n_quarters': len(qg),
        'lo_mean_vrp': lo_m, 'hi_mean_vrp': hi_m,
        'hi_gt_lo_week_frac': hi_gt_lo_frac, 'hi_gt_lo_quarter_frac': q_hi_gt_lo,
        'verdict': verdict, 'per_week': per_week,
    }, indent=2))
    print(f'-> {OUT / "vol-borrow-phaseB.json"}')


if __name__ == '__main__':
    main()
