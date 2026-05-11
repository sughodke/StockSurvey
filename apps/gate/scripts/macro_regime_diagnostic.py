"""Macro regime diagnostic — does macro state at window-start predict
which gate / pairs / vol windows win vs lose?

Cheap test before committing to ss-macro integration. Loads the 17
walk-forward windows we already ran (6 gate + 6 pairs + 5 vol),
joins macro features from FRED at each window's val_start, computes
per-app-z-scored alpha, and reports:

  1. Per-feature Pearson r between alpha-z and macro state — how
     much each individual macro indicator predicts window outcome.
  2. A 2x2 contingency table: win/lose × macro-stress/macro-calm,
     where macro-stress is defined by VIX at window-start.

If any |r| > 0.4 or the contingency table shows a clean 4:1+ split,
macro features are worth integrating into apps/gate. If everything
clusters in [-0.2, +0.2] and the table is 50/50, macro alone isn't
the regime signal and we look elsewhere (equity-internal: VIX
dispersion, breadth, sector rotation).

Run from repo root:
    uv run python apps/gate/scripts/macro_regime_diagnostic.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ss_macro import load_macro_panel


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'


def _load_window_outcomes() -> pd.DataFrame:
    """Pull per-window summaries from the three pivot apps. Returns
    long-form `(app, window_idx, val_start, alpha)` table. `alpha`
    is each app's own primary outcome metric:
      - gate    : alpha_sharpe (gated minus unconditional EW)
      - pairs   : val_sharpe   (agg portfolio Sharpe, benchmark = 0)
      - vol     : alpha_sharpe_per_cell (gated minus unconditional)
    """
    rows = []

    with (OUTPUT / 'gate-walkforward-summary.json').open() as f:
        gate = json.load(f)
    for r in gate['per_window']:
        rows.append({
            'app': 'gate',
            'window_idx': r['window_idx'],
            'val_start': r['val_start'],
            'val_end':   r['val_end'],
            'alpha':     r['alpha_sharpe'],
        })

    with (OUTPUT / 'pairs-walkforward-summary.json').open() as f:
        pairs = json.load(f)
    for r in pairs['per_window']:
        rows.append({
            'app': 'pairs',
            'window_idx': r['window_idx'],
            'val_start': r['val_start'],
            'val_end':   r['val_end'],
            'alpha':     r['val_sharpe'],
        })

    with (OUTPUT / 'vol-walkforward-summary.json').open() as f:
        vol = json.load(f)
    for r in vol['per_window']:
        rows.append({
            'app': 'vol',
            'window_idx': r['window_idx'],
            'val_start': r['val_start'],
            'val_end':   r['val_end'],
            'alpha':     r['alpha_sharpe_per_cell'],
        })

    df = pd.DataFrame(rows)
    df['val_start'] = pd.to_datetime(df['val_start'])
    df['val_end']   = pd.to_datetime(df['val_end'])

    # Per-app z-score so the three different alpha units are
    # comparable. Each app's mean is removed; std is per-app
    # (small samples but it's all we have).
    df['alpha_z'] = df.groupby('app')['alpha'].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))

    return df


def main() -> None:
    print('Loading per-window outcomes from apps/{gate,pairs,vol}...')
    out = _load_window_outcomes()
    print(f'  {len(out)} total windows ({out["app"].value_counts().to_dict()})')
    print()

    print('Loading macro panel from FRED...')
    macro = load_macro_panel()
    print(f'  {macro.columns.tolist()}, span {macro.index.min().date()} → '
          f'{macro.index.max().date()}')
    print()

    # For each window, get macro state at val_start (or last
    # available reading before val_start, ffilled).
    feature_cols = ['fed_funds', 'slope_10y_3m', 'credit_baa',
                    'm2_yoy', 'real_yield_10y', 'vix']
    macro_clean = macro[feature_cols].sort_index().ffill()

    def _macro_at(date: pd.Timestamp) -> pd.Series:
        # Latest reading on or before `date`.
        eligible = macro_clean.loc[macro_clean.index <= date]
        if eligible.empty:
            return pd.Series([np.nan] * len(feature_cols),
                             index=feature_cols)
        return eligible.iloc[-1]

    macro_at_window = out['val_start'].apply(_macro_at)
    out_full = pd.concat([out, macro_at_window], axis=1)

    print('Per-window outcomes + macro state at val_start:')
    print(f'{"app":>5s} {"win":>3s} {"val_start":>12s} {"alpha":>8s} '
          f'{"alpha_z":>8s} {"fed":>6s} {"slope":>6s} {"baa":>6s} '
          f'{"m2yoy":>6s} {"real":>6s} {"vix":>6s}')
    print('-' * 84)
    for _, r in out_full.iterrows():
        print(f'{r["app"]:>5s} {r["window_idx"]:>3d} '
              f'{str(r["val_start"].date()):>12s} '
              f'{r["alpha"]:>+8.3f} {r["alpha_z"]:>+8.2f} '
              f'{r["fed_funds"]:>6.2f} {r["slope_10y_3m"]:>+6.2f} '
              f'{r["credit_baa"]:>+6.2f} '
              f'{r["m2_yoy"] if not np.isnan(r["m2_yoy"]) else 0:>+6.1f} '
              f'{r["real_yield_10y"] if not np.isnan(r["real_yield_10y"]) else 0:>+6.2f} '
              f'{r["vix"]:>6.1f}')

    print()
    print('Per-feature Pearson r vs alpha_z (across all 17 windows):')
    print('  positive r → high feature value → high relative alpha')
    print(f'{"feature":>16s} {"Pearson r":>12s} {"|r|":>8s} {"verdict":>20s}')
    print('-' * 60)
    z = out_full['alpha_z'].values
    for c in feature_cols:
        x = out_full[c].values
        m = ~(np.isnan(x) | np.isnan(z))
        if m.sum() < 5:
            continue
        r = float(np.corrcoef(x[m], z[m])[0, 1])
        verdict = (
            'STRONG' if abs(r) > 0.5
            else 'suggestive' if abs(r) > 0.3
            else 'noise')
        print(f'{c:>16s} {r:>+12.3f} {abs(r):>8.3f}  {verdict:>20s}')

    print()
    # Contingency: split each app's alpha at within-app median
    # ('win' = above median for that app), and split macro state at
    # within-window median for each feature. See if win-rate differs.
    print('Contingency tables — (alpha_z above 0) × (feature above median):')
    print(f'{"feature":>16s} {"low/lose":>10s} {"low/win":>10s} '
          f'{"high/lose":>10s} {"high/win":>10s} {"high_winrate":>14s}')
    print('-' * 76)
    for c in feature_cols:
        x = out_full[c].values
        m = ~(np.isnan(x) | np.isnan(z))
        if m.sum() < 5:
            continue
        x_high = x > np.median(x[m])
        z_high = z > 0
        ll = int(np.sum(~x_high[m] & ~z_high[m]))
        lw = int(np.sum(~x_high[m] & z_high[m]))
        hl = int(np.sum(x_high[m] & ~z_high[m]))
        hw = int(np.sum(x_high[m] & z_high[m]))
        if (hl + hw) > 0:
            high_wr = hw / (hl + hw)
        else:
            high_wr = float('nan')
        print(f'{c:>16s} {ll:>10d} {lw:>10d} {hl:>10d} {hw:>10d} '
              f'{high_wr:>14.2f}')

    # Save the joined panel for later inspection.
    out_path = OUTPUT / 'macro-regime-diagnostic.json'
    out_full_serialized = out_full.copy()
    out_full_serialized['val_start'] = out_full_serialized['val_start'].dt.strftime('%Y-%m-%d')
    out_full_serialized['val_end']   = out_full_serialized['val_end'].dt.strftime('%Y-%m-%d')
    out_full_serialized.to_json(out_path, orient='records', indent=2)
    print(f'\n-> {out_path}')


if __name__ == '__main__':
    main()
