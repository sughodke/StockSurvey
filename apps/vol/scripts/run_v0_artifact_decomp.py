"""Gate A (offline, no API) — how much of the unrestricted-universe
short-vol "edge" is a delisting / trading-halt artifact?

The smoke run of `run_v0_breakeven_spread.py` surfaced something more
fundamental than the spread question: the v1 predictor's top-K picks
on the unrestricted gauss314 universe are dominated by names that were
acquired / halted / delisted at the rebal date (BITA, SHLX, GPX, AVEO,
BIL...), with `edge_frac = (IV − RV_forward)/IV ≈ 0.85–0.98`. That
near-1 edge is the signature of `RV_forward → 0`: the stock stopped
moving (acquisition close / halt / stale gauss314 row) while IV stayed
frozen high. That is NOT a vol-risk-premium capture — a short straddle
through an acquisition close is cash-settled / adjusted / assigned,
not held at the modeled PnL — and it is exactly why those names have
no live `option_chain` quote (0% tradable in the smoke).

This decomposes the headline edge into:
  * plausible VRP   — RV_forward in a live-equity range, real OI
  * halt artifact   — RV_forward ≈ 0 (name stopped trading)
  * degenerate OI   — gauss314 OI floor (dead ETFs / stale rows)

If the edge collapses once halt-artifact + degenerate-OI rows are
removed, the vol arc's +5.86 is substantially a survivorship artifact
(an arc-level correction). If a material edge survives on the clean,
potentially-tradable subset, *that* subset goes to the spread gate.

No network. Reuses the v1 predictor + target. Run from repo root:
    uv run python apps/vol/scripts/run_v0_artifact_decomp.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (
    FEATURE_NAMES, build_vol_features, forward_iv_rv_gap,
    load_gauss314_full, predict, train_predictor,
)

OUT = Path(__file__).resolve().parents[3] / 'Output'


def _windows(dates, tr, va, st):
    out, i = [], 0
    while i + tr + va <= len(dates):
        out.append((dates[i], dates[i + tr - 1],
                    dates[i + tr], dates[i + tr + va - 1]))
        i += st
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--train-window-days', type=int, default=300)
    ap.add_argument('--val-window-days', type=int, default=120)
    ap.add_argument('--step-window-days', type=int, default=120)
    ap.add_argument('--rebal-days', type=int, default=20)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    # Artifact cuts (sensitivity-reported).
    ap.add_argument('--rv-floor', type=float, default=0.05,
                    help='RV_forward (annualized frac) below this = '
                         'halt/stale (live equities ~never < 5% 20d RV).')
    ap.add_argument('--oi-floor', type=float, default=100.0,
                    help='oi_total below this = degenerate (no real '
                         'options market; gauss314 dead-row floor).')
    ap.add_argument('--output', default=str(OUT /
                    'vol-v0-artifact-decomp.json'))
    args = ap.parse_args()

    print('Loading gauss314...', flush=True)
    raw = load_gauss314_full()
    raw['oi_total'] = (raw['calls_open_interest'].fillna(0)
                       + raw['puts_open_interest'].fillna(0))
    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)
    m = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).merge(raw[['date', 'symbol', 'ATM_IV', 'oi_total']],
            on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap', 'ATM_IV'])
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        m[col] = m[col].clip(lower=-args.clip_iv_hv_ratio,
                             upper=args.clip_iv_hv_ratio)
    m = m[m['ATM_IV'] > 1e-6].copy()
    # iv_rv_gap = ATM_IV − rv_forward  ⇒  rv_forward = ATM_IV − gap.
    m['rv_forward'] = m['ATM_IV'] - m['iv_rv_gap']
    m['edge_frac'] = m['iv_rv_gap'] / m['ATM_IV']
    m['is_halt'] = m['rv_forward'] < args.rv_floor
    m['is_degen_oi'] = m['oi_total'] < args.oi_floor
    m['is_clean'] = ~m['is_halt'] & ~m['is_degen_oi']

    n = len(m)
    print(f'  {n:,} (date,symbol) rows', flush=True)
    print('\n=== Population decomposition (unrestricted universe) ===',
          flush=True)
    for label, mask in [('halt RV<%.2f' % args.rv_floor, m['is_halt']),
                        ('degen OI<%.0f' % args.oi_floor, m['is_degen_oi']),
                        ('clean (tradable-candidate)', m['is_clean'])]:
        sub = m[mask]
        print(f'  {label:<28s} {len(sub):>9,d} '
              f'({len(sub)/n:6.1%})  mean edge_frac '
              f'{sub["edge_frac"].mean():+.4f}', flush=True)
    print(f'  {"FULL universe":<28s} {n:>9,d} (100.0%)  '
          f'mean edge_frac {m["edge_frac"].mean():+.4f}', flush=True)

    # How concentrated is the universe-mean edge in halt rows?
    full_mean = float(m['edge_frac'].mean())
    clean_mean = float(m.loc[m['is_clean'], 'edge_frac'].mean())
    halt_share_of_mean = (
        float((m['is_halt'].sum() * m.loc[m['is_halt'], 'edge_frac'].mean())
              / (n * full_mean)) if full_mean != 0 else float('nan'))

    dates = pd.DatetimeIndex(sorted(m['date'].unique()))
    wins = _windows(dates, args.train_window_days,
                    args.val_window_days, args.step_window_days)

    # Walk-forward: predictor top-K edge on FULL vs CLEAN universe, and
    # the val Pearson r on each, per window.
    rows = []
    for wi, (tr_lo, tr_hi, va_lo, va_hi) in enumerate(wins):
        tr = m[(m.date >= tr_lo) & (m.date <= tr_hi)]
        va = m[(m.date >= va_lo) & (m.date <= va_hi)]
        if len(tr) < 1000 or len(va) < 500:
            continue
        pr = train_predictor(tr[FEATURE_NAMES].values,
                             tr['iv_rv_gap'].values, FEATURE_NAMES)
        va = va.copy()
        va['pred'] = predict(pr, va[FEATURE_NAMES].values)
        vac = va[va['is_clean']]
        r_full = float(np.corrcoef(va['pred'], va['iv_rv_gap'])[0, 1])
        r_clean = (float(np.corrcoef(vac['pred'], vac['iv_rv_gap'])[0, 1])
                   if len(vac) > 50 else float('nan'))

        def _topk_edge(frame):
            es = []
            for _, day in frame.groupby('date'):
                if len(day) < max(args.top_k, 5):
                    continue
                es.append(day.nlargest(args.top_k, 'pred')['edge_frac']
                          .mean())
            return float(np.mean(es)) if es else float('nan')

        rows.append({
            'window': wi,
            'val_period': f'{va_lo.date()}→{va_hi.date()}',
            'val_r_full': r_full, 'val_r_clean': r_clean,
            'topk_edge_full': _topk_edge(va),
            'topk_edge_clean': _topk_edge(vac),
            'univ_edge_full': float(va['edge_frac'].mean()),
            'univ_edge_clean': float(vac['edge_frac'].mean())
            if len(vac) else float('nan'),
            'clean_frac': float(len(vac) / len(va)),
        })

    wf = pd.DataFrame(rows)
    print('\n=== Walk-forward: FULL vs CLEAN (halt+degen removed) ===',
          flush=True)
    print(wf.to_string(index=False, float_format=lambda x: f'{x:+.4f}'),
          flush=True)

    tef = wf['topk_edge_full'].mean()
    tec = wf['topk_edge_clean'].mean()
    uef = wf['univ_edge_full'].mean()
    uec = wf['univ_edge_clean'].mean()
    collapse = 1.0 - (tec / tef) if tef and not np.isnan(tef) else float('nan')

    print('\n=== HEADLINE ===', flush=True)
    print(f'  halt rows are {m["is_halt"].mean():.1%} of the universe '
          f'and carry {halt_share_of_mean:.0%} of its mean edge', flush=True)
    print(f'  predictor top-K edge  FULL  = {tef:+.4f}', flush=True)
    print(f'  predictor top-K edge  CLEAN = {tec:+.4f}  '
          f'({collapse:+.0%} change)', flush=True)
    print(f'  universe   edge       FULL  = {uef:+.4f}', flush=True)
    print(f'  universe   edge       CLEAN = {uec:+.4f}', flush=True)
    print(f'  mean clean (tradable-candidate) fraction of universe = '
          f'{wf["clean_frac"].mean():.1%}', flush=True)

    if np.isnan(tec) or tec <= 0.02:
        verdict = (
            f'ARC-LEVEL ARTIFACT — clean predictor edge collapses to '
            f'{tec:+.4f} (from {tef:+.4f}). The unrestricted vol "edge" '
            f'is substantially a delisting/halt survivorship artifact, '
            f'not harvestable VRP. Arc verdict revises toward '
            f'reversed-OOS / diagnostic.')
    elif tec < 0.5 * tef:
        verdict = (
            f'HEAVILY CONTAMINATED — clean edge {tec:+.4f} is <50% of '
            f'full {tef:+.4f}; majority of headline edge is artifact. '
            f'Residual real edge exists; spread gate runs on the clean '
            f'cohort only.')
    else:
        verdict = (
            f'EDGE SURVIVES CLEANING — clean edge {tec:+.4f} retains '
            f'>50% of full {tef:+.4f}; artifact is a contaminant not the '
            f'whole story. Proceed to spread gate on the clean cohort.')
    print(f'\npre-reg verdict: {verdict}', flush=True)

    Path(args.output).write_text(json.dumps({
        'rv_floor': args.rv_floor, 'oi_floor': args.oi_floor,
        'n_rows': int(n),
        'halt_frac': float(m['is_halt'].mean()),
        'degen_oi_frac': float(m['is_degen_oi'].mean()),
        'clean_frac': float(m['is_clean'].mean()),
        'full_universe_mean_edge': full_mean,
        'clean_universe_mean_edge': clean_mean,
        'halt_share_of_universe_mean_edge': halt_share_of_mean,
        'walkforward': wf.to_dict(orient='records'),
        'topk_edge_full_mean': float(tef) if not np.isnan(tef) else None,
        'topk_edge_clean_mean': float(tec) if not np.isnan(tec) else None,
        'verdict': verdict,
    }, indent=2))
    print(f'\n-> {args.output}', flush=True)


if __name__ == '__main__':
    main()
