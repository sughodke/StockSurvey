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

    # ---------------------------------------------------------------
    # v1 extension (2026-05-12): factor signal-quality as a second
    # meta-gate input, plus VIX-AND-factor / VIX-OR-factor composites.
    #
    # Factor signal-quality (per-val-bar top-decile − bottom-decile
    # predicted alpha) was validated as a candidate gate input in
    # `findings/factor-sizing-input-v0.md`: pooled lag-1 autocorr +0.91
    # and Spearman ρ vs val Sharpe +0.486 from the rank_ic-trained
    # indicator head. v1 question: does it add lift on top of the
    # binary VIX-median gate at the meta-level?
    #
    # Temporal alignment: factor walk-forward has 6 val_start dates
    # (~3y spacing); pivot-arc windows have 17 val_starts (~1-2y
    # spacing). For each pivot row we look up the most recent factor
    # val_start ≤ pivot val_start and use that factor window's
    # `signal_quality_mean`. This is the most-recent OOS factor read
    # *available at the pivot deployment moment* (no look-ahead).
    # Resolution is coarse (6 distinct factor values across 17 pivot
    # rows) — v1 is a diagnostic, v2 would refine the factor windowing.
    # ---------------------------------------------------------------
    factor_npz_path = OUTPUT / 'sizing-input-rank_ic-windows.npz'
    if not factor_npz_path.exists():
        print(f'\n[v1 extension skipped] factor signal-quality artifact '
              f'not found at {factor_npz_path} — run '
              f'`uvx modal run apps/factor/scripts/modal/sizing_input_eval.py` '
              f'first.')
    else:
        print('\n' + '=' * 70)
        print(f'v1: loading factor signal-quality from {factor_npz_path.name}')
        f_npz = np.load(factor_npz_path)
        factor_df = pd.DataFrame({
            'factor_val_start': pd.to_datetime(
                [s.decode() for s in f_npz['val_start_date']]),
            'factor_sq': f_npz['signal_quality_mean'].astype(float),
        }).sort_values('factor_val_start').reset_index(drop=True)
        print(factor_df.to_string(index=False))

        def _most_recent_factor_sq(pivot_val_start: pd.Timestamp) -> float:
            eligible = factor_df.loc[
                factor_df['factor_val_start'] <= pivot_val_start]
            if eligible.empty:
                return float('nan')
            return float(eligible['factor_sq'].iloc[-1])

        out['factor_sq'] = out['val_start'].apply(_most_recent_factor_sq)
        factor_median = float(factor_df['factor_sq'].median())
        out['factor_state'] = np.where(
            out['factor_sq'].isna(), 'unknown',
            np.where(out['factor_sq'] >= factor_median, 'high', 'low'))
        print(f'\nfactor median (over {len(factor_df)} windows): '
              f'{factor_median:.3f}  — gate threshold')

        out['alpha_factor_gated'] = np.where(
            out['factor_state'] == 'high', out['alpha'], 0.0)
        out['alpha_and_gated'] = np.where(
            (out['factor_state'] == 'high') & (out['vix_state'] == 'high'),
            out['alpha'], 0.0)
        out['alpha_or_gated'] = np.where(
            (out['factor_state'] == 'high') | (out['vix_state'] == 'high'),
            out['alpha'], 0.0)
        out['alpha_factor_gated_z'] = np.where(
            out['factor_state'] == 'high', out['alpha_z'], 0.0)
        out['alpha_and_gated_z'] = np.where(
            (out['factor_state'] == 'high') & (out['vix_state'] == 'high'),
            out['alpha_z'], 0.0)
        out['alpha_or_gated_z'] = np.where(
            (out['factor_state'] == 'high') | (out['vix_state'] == 'high'),
            out['alpha_z'], 0.0)

        print('\nPer-window factor + composite gate decisions:')
        print(f'{"app":>5s} {"win":>3s} {"val_start":>12s} '
              f'{"vix":>5s} {"f_sq":>6s} {"f_st":>4s} '
              f'{"alpha":>8s} '
              f'{"vix_g":>7s} {"f_g":>7s} {"and_g":>7s} {"or_g":>7s}')
        print('-' * 90)
        for _, r in out.iterrows():
            f_sq_str = (
                f'{r["factor_sq"]:>6.3f}' if not pd.isna(r['factor_sq'])
                else f'{"n/a":>6s}')
            print(f'{r["app"]:>5s} {r["window_idx"]:>3d} '
                  f'{str(r["val_start"].date()):>12s} '
                  f'{r["vix_state"]:>5s} {f_sq_str} '
                  f'{r["factor_state"]:>4s} '
                  f'{r["alpha"]:>+8.3f} '
                  f'{r["alpha_meta_gated"]:>+7.3f} '
                  f'{r["alpha_factor_gated"]:>+7.3f} '
                  f'{r["alpha_and_gated"]:>+7.3f} '
                  f'{r["alpha_or_gated"]:>+7.3f}')

        print('\n' + '=' * 70)
        print('Pooled comparison across all 17 windows:')
        print(f'{"arm":>22s}  {"raw_mean":>9s}  {"gated_mean":>10s}  '
              f'{"lift":>8s}  {"n_deploy":>8s}')
        print('-' * 70)
        n_total = len(out)
        rows_to_print = [
            ('raw (no gate)', out['alpha'].mean(), out['alpha'].mean(),
             n_total),
            ('VIX-only',
             out['alpha'].mean(),
             out['alpha_meta_gated'].mean(),
             int((out['vix_state'] == 'high').sum())),
            ('factor-only',
             out['alpha'].mean(),
             out['alpha_factor_gated'].mean(),
             int((out['factor_state'] == 'high').sum())),
            ('VIX AND factor',
             out['alpha'].mean(),
             out['alpha_and_gated'].mean(),
             int(((out['vix_state'] == 'high')
                  & (out['factor_state'] == 'high')).sum())),
            ('VIX OR factor',
             out['alpha'].mean(),
             out['alpha_or_gated'].mean(),
             int(((out['vix_state'] == 'high')
                  | (out['factor_state'] == 'high')).sum())),
        ]
        for label, raw, gated, n_dep in rows_to_print:
            lift = gated - raw
            print(f'{label:>22s}  {raw:>+9.3f}  {gated:>+10.3f}  '
                  f'{lift:>+8.3f}  {n_dep:>8d}')

        print()
        print(f'{"arm":>22s}  {"raw_z":>9s}  {"gated_z":>10s}  {"lift_z":>8s}')
        print('-' * 60)
        raw_z = float(out['alpha_z'].mean())
        for label, gated_col in (
            ('VIX-only', 'alpha_meta_gated_z'),
            ('factor-only', 'alpha_factor_gated_z'),
            ('VIX AND factor', 'alpha_and_gated_z'),
            ('VIX OR factor', 'alpha_or_gated_z'),
        ):
            gz = float(out[gated_col].mean())
            print(f'{label:>22s}  {raw_z:>+9.3f}  {gz:>+10.3f}  '
                  f'{(gz - raw_z):>+8.3f}')

        print()
        print('Pre-registered v1 cuts '
              '(TODO/factor-sizing-input-reframe.md):')
        print('  PASS — any factor arm pooled z-score lift ≥ +0.30')
        print(f'         (≈ +0.10 absolute over VIX-only baseline +0.215)')
        print(f'  FAIL — within ±0.05 of VIX-only z-score lift')
        print()
        vix_lift_z = float(out['alpha_meta_gated_z'].mean()) - raw_z
        best_arm = None; best_lift = float('-inf')
        for label, gated_col in (
            ('factor-only', 'alpha_factor_gated_z'),
            ('VIX AND factor', 'alpha_and_gated_z'),
            ('VIX OR factor', 'alpha_or_gated_z'),
        ):
            gz = float(out[gated_col].mean()) - raw_z
            if gz > best_lift:
                best_arm = label; best_lift = gz
        # PASS: any factor arm exceeds +0.30 z lift (pre-reg).
        # FAIL: every factor arm is at or below VIX-only minus 0.05
        #       (factor adds no information over VIX alone).
        # INCONCLUSIVE: between the two — best factor arm above
        #               VIX-only-minus-0.05 but below the PASS bar.
        if best_lift >= 0.30:
            verdict = f'PASS — best factor arm "{best_arm}" lift +{best_lift:.3f} ≥ +0.30'
        elif best_lift <= vix_lift_z - 0.05:
            verdict = (f'FAIL (confirmed-null on incremental lift) — best factor arm '
                       f'"{best_arm}" lift +{best_lift:.3f} is below '
                       f'VIX-only +{vix_lift_z:.3f} by ≥0.05; factor signal-quality '
                       f'at this temporal resolution adds no value over VIX alone')
        else:
            verdict = (f'INCONCLUSIVE — best factor arm "{best_arm}" lift '
                       f'+{best_lift:.3f}, VIX-only +{vix_lift_z:.3f}')
        print(f'verdict: {verdict}')

    out_path = OUTPUT / 'macro-meta-gate-eval.json'
    out_serialized = out.copy()
    out_serialized['val_start'] = out_serialized['val_start'].dt.strftime('%Y-%m-%d')
    if 'factor_val_start' in out_serialized.columns:
        out_serialized = out_serialized.drop(columns=['factor_val_start'])
    out_serialized.to_json(out_path, orient='records', indent=2)
    print(f'\n-> {out_path}')


if __name__ == '__main__':
    main()
