"""Meta-gate retroactive eval — does a VIX-above-1y-rolling-median
filter retroactively lift mean alpha across the prediction-problem-
pivot windows?

The macro-regime-diagnostic finding showed macro state at val_start
predicts which v0 windows produce alpha (5/6 features directionally
consistent, VIX-above-median = 6× win-rate lift). The within-app v1
test (`run_walkforward.py --with-macro`) showed adding macro as
direct features makes things WORSE (mean alpha drops +0.067 →
−0.086) — distribution shift between train and val regimes.

This script tests the alternative: don't add macro as features;
use macro state to decide WHEN to deploy each app's existing v0
predictor. Specifically: at each window's val_start, if VIX <
recent rolling median, suspend the app (alpha = 0); if VIX >=
median, take the app's actual val alpha. Compare meta-gated mean
alpha vs ungated.

If the meta-gate lifts mean alpha materially (e.g. above the
+0.10 marginal threshold), it's the right v1 architecture and
the diagnostic's signal monetizes. If not, macro state predicts
window outcomes only at the noisy n=17 level and we need a
different regime classifier.

Run from repo root:
    uv run python apps/gate/scripts/macro_meta_gate_eval.py
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
    """Same loader as macro_regime_diagnostic.py — pull per-window
    outcomes from gate / pairs / vol summaries."""
    rows = []
    for app, path, alpha_key in (
        ('gate',  'gate-walkforward-summary.json',  'alpha_sharpe'),
        ('pairs', 'pairs-walkforward-summary.json', 'val_sharpe'),
        ('vol',   'vol-walkforward-summary.json',   'alpha_sharpe_per_cell'),
    ):
        with (OUTPUT / path).open() as f:
            data = json.load(f)
        for r in data['per_window']:
            rows.append({
                'app': app,
                'window_idx': r['window_idx'],
                'val_start':  r['val_start'],
                'alpha':      r[alpha_key],
            })
    df = pd.DataFrame(rows)
    df['val_start'] = pd.to_datetime(df['val_start'])
    return df


def main() -> None:
    print('Loading per-window outcomes...')
    out = _load_window_outcomes()
    print(f'  {len(out)} windows from {out["app"].value_counts().to_dict()}')

    print('\nLoading VIX from FRED...')
    macro = load_macro_panel()
    vix = macro['vix'].dropna()

    # For each window: VIX at val_start vs trailing 1y rolling median
    # ending the day before val_start (no look-ahead).
    print('\nComputing 1y trailing VIX median at each val_start...')
    vix_df = pd.DataFrame({'vix': vix}).sort_index()
    rolling_median = vix.rolling(window=252, min_periods=60).median()

    def _vix_state(date: pd.Timestamp) -> tuple[float, float, str]:
        """(spot VIX at date, 1y rolling median ending at date,
        'high' or 'low')"""
        eligible = vix_df.loc[vix_df.index <= date]
        if eligible.empty:
            return float('nan'), float('nan'), 'unknown'
        spot = float(eligible['vix'].iloc[-1])
        med_eligible = rolling_median.loc[rolling_median.index <= date]
        med = float(med_eligible.iloc[-1]) if not med_eligible.empty else float('nan')
        state = 'high' if spot >= med else 'low'
        return spot, med, state

    out['vix_spot']     = out['val_start'].apply(lambda d: _vix_state(d)[0])
    out['vix_median1y'] = out['val_start'].apply(lambda d: _vix_state(d)[1])
    out['vix_state']    = out['val_start'].apply(lambda d: _vix_state(d)[2])

    # Apply meta-gate: if VIX low at val_start, alpha → 0 (suspend
    # deployment). Otherwise keep the app's actual alpha.
    out['alpha_meta_gated'] = np.where(
        out['vix_state'] == 'high', out['alpha'], 0.0)

    print('\nPer-window meta-gate decisions:')
    print(f'{"app":>5s} {"win":>3s} {"val_start":>12s} {"vix":>5s} '
          f'{"med1y":>6s} {"state":>5s} {"alpha":>8s} '
          f'{"alpha_gated":>12s}')
    print('-' * 70)
    for _, r in out.iterrows():
        print(f'{r["app"]:>5s} {r["window_idx"]:>3d} '
              f'{str(r["val_start"].date()):>12s} '
              f'{r["vix_spot"]:>5.1f} {r["vix_median1y"]:>6.1f} '
              f'{r["vix_state"]:>5s} {r["alpha"]:>+8.3f} '
              f'{r["alpha_meta_gated"]:>+12.3f}')

    print('\n' + '=' * 60)
    print('Per-app comparison: mean alpha (raw) vs (meta-gated):')
    print(f'{"app":>5s} {"n":>3s} {"raw_mean":>10s} '
          f'{"gated_mean":>11s} {"lift":>8s} {"high":>5s} {"low":>5s}')
    print('-' * 60)
    for app in ('gate', 'pairs', 'vol', 'ALL'):
        sub = out if app == 'ALL' else out[out['app'] == app]
        n = len(sub)
        raw = float(sub['alpha'].mean())
        gated = float(sub['alpha_meta_gated'].mean())
        n_high = int((sub['vix_state'] == 'high').sum())
        n_low = n - n_high
        lift = gated - raw
        print(f'{app:>5s} {n:>3d} {raw:>+10.3f} '
              f'{gated:>+11.3f} {lift:>+8.3f} {n_high:>5d} {n_low:>5d}')

    print('\n' + '=' * 60)
    # Per-app-z-scored pooled lift (apples-to-apples across apps).
    out['alpha_z'] = out.groupby('app')['alpha'].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
    out['alpha_meta_gated_z'] = np.where(
        out['vix_state'] == 'high', out['alpha_z'], 0.0)
    raw_z = float(out['alpha_z'].mean())   # ~0 by construction
    gated_z = float(out['alpha_meta_gated_z'].mean())
    print(f'Pooled per-app-z-scored: raw mean alpha-z = {raw_z:+.3f} '
          f'(should be ~0); meta-gated mean alpha-z = {gated_z:+.3f}')
    print(f'  positive lift confirms meta-gate adds value at the '
          f'pooled level')

    out_path = OUTPUT / 'macro-meta-gate-eval.json'
    out_serialized = out.copy()
    out_serialized['val_start'] = out_serialized['val_start'].dt.strftime('%Y-%m-%d')
    out_serialized.to_json(out_path, orient='records', indent=2)
    print(f'\n-> {out_path}')


if __name__ == '__main__':
    main()
