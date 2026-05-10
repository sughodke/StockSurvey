"""Audit gauss314 IV CSV: schema, density, IV / HV cleanliness.

Per the apps-vol TODO's "Stage 0 — data" gate: if the data isn't
sufficient or is too noisy, this stops here. Run before any
walk-forward.

Run from repo root:
    uv run python apps/vol/scripts/audit_data.py
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from vol import build_vol_features, load_gauss314_full
from vol.target import forward_iv_rv_gap


def main() -> None:
    print('Loading gauss314 full schema...')
    t0 = time.perf_counter()
    raw = load_gauss314_full()
    print(f'  loaded {len(raw):,} rows in {time.perf_counter()-t0:.1f}s')

    print(f'\nSchema:')
    for c in raw.columns:
        n_null = int(raw[c].isna().sum())
        pct_null = 100 * n_null / len(raw)
        print(f'  {c:>22s}  {raw[c].dtype}  '
              f'null={n_null:>6d} ({pct_null:>5.2f}%)')

    print(f'\nDate range: {raw["date"].min()} → {raw["date"].max()} '
          f'({raw["date"].nunique()} unique dates)')
    print(f'Symbols: {raw["symbol"].nunique()} unique')

    print('\nUnivariate sanity (fractional, 0.30 = 30% annualized):')
    for c in ('ATM_IV', 'DOTM_IV', 'DITM_IV', 'hv_20', 'hv_60', 'VIX'):
        s = raw[c].dropna()
        print(f'  {c:>10s}  '
              f'mean={s.mean():.3f}  median={s.median():.3f}  '
              f'p10={s.quantile(0.10):.3f}  p90={s.quantile(0.90):.3f}')

    print('\nBuilding features...')
    panel = build_vol_features(raw)
    feat = panel.features
    print(f'  feature stack: {len(feat):,} rows × {len(feat.columns)-2} feature cols')

    print('\nFeature univariate sanity (post-z-score will be unit):')
    from vol import FEATURE_NAMES
    for c in FEATURE_NAMES:
        s = feat[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            print(f'  {c:>22s}  ALL NULL')
            continue
        print(f'  {c:>22s}  mean={s.mean():+.4f}  std={s.std():.4f}  '
              f'p10={s.quantile(0.10):+.4f}  p90={s.quantile(0.90):+.4f}  '
              f'null_pct={100*(1-len(s)/len(feat)):.1f}%')

    print('\nBuilding forward 20-day IV/RV gap target...')
    target = forward_iv_rv_gap(raw, horizon=20)
    valid = target.dropna()
    print(f'  target: {len(valid):,} valid (date, symbol) cells')
    print(f'  iv_rv_gap stats:')
    g = valid['iv_rv_gap']
    print(f'    mean   = {g.mean():+.4f}  (positive = IV > realized → '
          f'short-vol-edge)')
    print(f'    median = {g.median():+.4f}')
    print(f'    p10    = {g.quantile(0.10):+.4f}')
    print(f'    p90    = {g.quantile(0.90):+.4f}')
    print(f'    pct positive = {100*(g > 0).mean():.1f}%')

    print(f'\nMatched (feature, target) cells:')
    merged = feat.merge(
        target, on=['date', 'symbol'], how='inner').dropna()
    print(f'  {len(merged):,} usable rows after dropna')
    print(f'  date range: {merged["date"].min()} → {merged["date"].max()}')
    print(f'  symbols:    {merged["symbol"].nunique()}')

    # Univariate Pearson r of each feature vs target — gives a quick
    # read on which features carry signal at the linear, single-feature
    # level before any joint regression.
    print('\nFeature → target single-variable Pearson r '
          '(higher |r| = more signal):')
    for c in FEATURE_NAMES:
        x = merged[c].replace([np.inf, -np.inf], np.nan)
        y = merged['iv_rv_gap']
        m = ~(x.isna() | y.isna())
        if m.sum() < 100:
            print(f'  {c:>22s}  insufficient data')
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        print(f'  {c:>22s}  r = {r:+.4f}  (n={int(m.sum()):,})')


if __name__ == '__main__':
    main()
