"""B1 Stage 0 — borrow-data feasibility gate (local, free, no Modal).

The HARD PRE-CHECK from TODO/vol-borrow-liquid-universe.md, learned from the
illiquid arc's quote-availability death: before sinking a novel-data arc,
verify the data is retrievable, free, AND coverage-complete over the exact
liquid universe the signal needs. Here the analogue is borrow-data
availability + DISPERSION (the real kill-risk: liquid names are mostly
easy-to-borrow, so borrow-stress may not vary enough to form low/mid/high
terciles — the mirror image of the illiquid coverage problem).

For ~6 sample rebal dates spanning the gauss314 v1/v2 span (2019-10 →
2023-07), reconstruct that date's top-200-OI liquid universe from the local
gauss314 cache, pull FINRA daily short-volume + SEC fails-to-deliver for the
date, and report:
  - COVERAGE: fraction of top-200-OI names present in each borrow source.
  - DISPERSION: spread of short-volume ratio across the cohort (tercile gap),
    and fraction of names with nonzero FTDs — i.e. is there a signal to split?

Gate (per the TODO): PASS needs >=90% coverage AND enough dispersion to form
meaningful terciles. FAIL closes the arc cheaply (a clean verdict).

    uv run python apps/vol/scripts/borrow_stage0_probe.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
GAUSS = REPO / '.iv-cache' / 'data_IV_USA.csv'
OUT = REPO / 'Output'
OI_TOP_N = 200
UA = {'User-Agent': 'StockSurvey research stage0 (sid.ghodke@gmail.com)'}


def finra_shortvol(date: pd.Timestamp) -> pd.DataFrame | None:
    """FINRA consolidated daily short-sale volume for one trading day."""
    url = f'https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date:%Y%m%d}.txt'
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pd.read_csv(io.StringIO(r.text), sep='|')
        df = df[df['Symbol'].notna() & (df.get('TotalVolume', 0) > 0)].copy()
        df['short_ratio'] = df['ShortVolume'] / df['TotalVolume']
        return df[['Symbol', 'short_ratio', 'TotalVolume']]
    except Exception as e:
        print(f'    [finra error {date:%Y-%m-%d}] {type(e).__name__}: {e}')
        return None


def sec_ftd(date: pd.Timestamp) -> pd.DataFrame | None:
    """SEC fails-to-deliver for the semi-month covering `date` (a=1-15, b=16+)."""
    half = 'a' if date.day <= 15 else 'b'
    url = f'https://www.sec.gov/files/data/fails-deliver-data/cnsfails{date:%Y%m}{half}.zip'
    try:
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            return None
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name), sep='|', encoding='latin-1',
                         on_bad_lines='skip')
        df.columns = [c.strip().upper() for c in df.columns]
        sym_col = next((c for c in df.columns if 'SYMBOL' in c), None)
        qty_col = next((c for c in df.columns if 'QUANTITY' in c or 'FAILS' in c), None)
        if sym_col is None or qty_col is None:
            print(f'    [sec ftd] unexpected columns: {list(df.columns)[:8]}')
            return None
        df = df.rename(columns={sym_col: 'Symbol', qty_col: 'fails'})
        df['fails'] = pd.to_numeric(df['fails'], errors='coerce')
        return df[['Symbol', 'fails']].dropna()
    except Exception as e:
        print(f'    [sec ftd error {date:%Y-%m}] {type(e).__name__}: {e}')
        return None


def main() -> None:
    print('loading gauss314 OI columns ...', flush=True)
    g = pd.read_csv(GAUSS, usecols=['symbol', 'date', 'calls_open_interest',
                                    'puts_open_interest'])
    g['date'] = pd.to_datetime(g['date'])
    g['total_oi'] = g['calls_open_interest'].fillna(0) + g['puts_open_interest'].fillna(0)
    all_dates = pd.DatetimeIndex(np.sort(g['date'].unique()))
    print(f'  gauss314 span: {all_dates[0].date()} → '
          f'{all_dates[-1].date()} ({len(all_dates)} dates)')

    # ~6 sample dates spread across the v1/v2 span (COVID, recovery, 2022 bear).
    targets = ['2019-11-01', '2020-04-01', '2020-11-02', '2021-09-01',
               '2022-06-01', '2023-03-01']
    sample_dates = []
    for t in targets:
        idx = min(all_dates.searchsorted(pd.Timestamp(t)), len(all_dates) - 1)
        sample_dates.append(all_dates[idx])

    rows = []
    for d in sample_dates:
        day = g[g['date'] == d]
        top = day.nlargest(OI_TOP_N, 'total_oi')['symbol'].str.upper().tolist()
        top_set = set(top)
        print(f'\n=== {d.date()}  top-{OI_TOP_N}-OI: {len(top)} names ===', flush=True)

        fv = finra_shortvol(d)
        ftd = sec_ftd(d)

        rec = {'date': str(d.date()), 'n_top_oi': len(top)}
        if fv is not None:
            fv['Symbol'] = fv['Symbol'].str.upper()
            cov = fv[fv['Symbol'].isin(top_set)]
            rec['finra_coverage'] = round(len(cov) / max(len(top), 1), 3)
            sr = cov['short_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sr) >= 9:
                q = sr.quantile([1/3, 2/3]).values
                lo, hi = sr[sr <= q[0]].mean(), sr[sr >= q[1]].mean()
                rec['short_ratio_med'] = round(float(sr.median()), 3)
                rec['short_ratio_lo_tercile'] = round(float(lo), 3)
                rec['short_ratio_hi_tercile'] = round(float(hi), 3)
                rec['short_ratio_tercile_gap'] = round(float(hi - lo), 3)
            print(f'  FINRA short-vol: coverage {rec["finra_coverage"]:.0%}  '
                  f'short-ratio med {rec.get("short_ratio_med")}  '
                  f'tercile gap {rec.get("short_ratio_tercile_gap")}')
        else:
            rec['finra_coverage'] = None
            print('  FINRA short-vol: UNAVAILABLE')

        if ftd is not None:
            ftd['Symbol'] = ftd['Symbol'].str.upper()
            ftd_cov = ftd[ftd['Symbol'].isin(top_set)]
            nonzero = ftd_cov[ftd_cov['fails'] > 0]
            rec['ftd_names_with_fails'] = int(nonzero['Symbol'].nunique())
            rec['ftd_frac_cohort_with_fails'] = round(
                nonzero['Symbol'].nunique() / max(len(top), 1), 3)
            print(f'  SEC FTD: {rec["ftd_names_with_fails"]}/{len(top)} cohort names '
                  f'had fails ({rec["ftd_frac_cohort_with_fails"]:.0%})')
        else:
            rec['ftd_names_with_fails'] = None
            print('  SEC FTD: UNAVAILABLE')
        rows.append(rec)

    # ---- Gate verdict ----
    cov_vals = [r['finra_coverage'] for r in rows if r.get('finra_coverage') is not None]
    gap_vals = [r.get('short_ratio_tercile_gap') for r in rows
                if r.get('short_ratio_tercile_gap') is not None]
    ftd_vals = [r.get('ftd_frac_cohort_with_fails') for r in rows
                if r.get('ftd_frac_cohort_with_fails') is not None]
    mean_cov = float(np.mean(cov_vals)) if cov_vals else 0.0
    mean_gap = float(np.mean(gap_vals)) if gap_vals else 0.0
    mean_ftd = float(np.mean(ftd_vals)) if ftd_vals else 0.0

    print('\n' + '=' * 60)
    print(f'mean FINRA coverage of top-{OI_TOP_N}-OI: {mean_cov:.1%}')
    print(f'mean short-ratio hi−lo tercile gap:       {mean_gap:.3f}')
    print(f'mean fraction of cohort with FTDs:        {mean_ftd:.1%}')
    coverage_ok = mean_cov >= 0.90
    dispersion_ok = mean_gap >= 0.05  # hi-tercile short-ratio >= lo + 5pp
    verdict = ('PASS' if (coverage_ok and dispersion_ok)
               else 'FAIL — coverage' if not coverage_ok
               else 'FAIL — borrow-stress too flat to form terciles')
    print(f'\nSTAGE 0 GATE: {verdict}')
    print(f'  coverage>=90%: {coverage_ok} ; dispersion(gap>=0.05): {dispersion_ok}')

    OUT.mkdir(exist_ok=True)
    (OUT / 'vol-borrow-stage0.json').write_text(json.dumps({
        'sample_dates': [r['date'] for r in rows],
        'per_date': rows, 'mean_finra_coverage': mean_cov,
        'mean_short_ratio_tercile_gap': mean_gap,
        'mean_ftd_frac_cohort': mean_ftd, 'verdict': verdict,
    }, indent=2))
    print(f'-> {OUT / "vol-borrow-stage0.json"}')


if __name__ == '__main__':
    main()
