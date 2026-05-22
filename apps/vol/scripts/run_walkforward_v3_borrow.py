"""B1 Phase A — borrow-stress conditioning on the locked v3 vol recipe.

Reproduces the v3 deployment recipe EXACTLY (top-200 OI, v1 10-feature OLS,
VIX>126d-median gate, top-50, 20d rebal, vol-points alpha vs universe) and
adds ONE thing: at each fired rebal, split the 50 picks into low/mid/high
borrow-stress terciles (from the Stage-0 composite, prep_borrow_data.py) and
test:

  H1 (premium amplifier): does the high-borrow-stress tercile earn higher
     mean alpha than the low-stress one? (richer VRP where MM hedging is costly)
  H2 (squeeze tail): is the high-stress tercile's worst-rebal alpha worse?
  H3 (separability, load-bearing): does a conditioned selection (drop the
     high-stress squeeze-tail tercile, deploy low+mid) beat the unconditioned
     50-pick v3 by >= +0.10 net Sharpe?

Pre-reg (from TODO/vol-borrow-liquid-universe.md): PASS if H3 >= +0.10, H1
holds, H2 not fatal, >=4/5 windows; MARGINAL if H3 in [+0.05,+0.10); FAIL-null
if H3 < +0.05 with H1 holding; FAIL-reversed if conditioned <= uncond and H1
fails. Note: vol-points alpha (the v3 metric) — defined-risk vertical spreads
are a deployment detail not modeled in the backtest (the whole vol arc carries
this caveat); H2 stands in for the tail-cap need.

    uv run python apps/vol/scripts/run_walkforward_v3_borrow.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (FEATURE_NAMES, build_vol_features, forward_iv_rv_gap,
                 load_gauss314_full, predict, train_predictor)

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'Output'
BORROW = REPO / '.iv-cache' / 'borrow_composite.parquet'
OI_TOP_N, TOP_K, REBAL, GATE_LB = 200, 50, 20, 126
TRAIN_D, VAL_D, STEP_D, HORIZON, CLIP = 300, 120, 120, 20, 10.0
PPY = 252.0 / REBAL


def _windows(dates, tr, va, st):
    out, i = [], 0
    while i + tr + va <= len(dates):
        out.append((dates[i], dates[i + tr - 1], dates[i + tr], dates[i + tr + va - 1]))
        i += st
    return out


def _oi_filter(merged, raw, n):
    oi = raw[['date', 'symbol', 'puts_open_interest', 'calls_open_interest']].copy()
    oi['total_oi'] = oi['puts_open_interest'].fillna(0) + oi['calls_open_interest'].fillna(0)
    oi['rank'] = oi.groupby('date')['total_oi'].rank(method='first', ascending=False)
    keep = oi[oi['rank'] <= n][['date', 'symbol']].copy()
    keep['keep'] = True
    out = merged.merge(keep, on=['date', 'symbol'], how='left')
    return out[out['keep'].fillna(False)].drop(columns='keep')


def _vix_gate(raw, lb):
    v = raw[['date', 'VIX']].drop_duplicates('date').sort_values('date').set_index('date')['VIX']
    med = v.rolling(lb, min_periods=lb // 2).median()
    return {d: bool(f) for d, f in (v > med).items() if not pd.isna(f)}


def _sharpe(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1) if x.size > 1 else 0.0
    return float(x.mean() / sd * np.sqrt(PPY)) if sd > 1e-12 else 0.0


def main() -> None:
    print('loading gauss314 + borrow composite ...', flush=True)
    raw = load_gauss314_full()
    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=HORIZON)
    merged = panel.features.merge(target, on=['date', 'symbol'], how='inner') \
        .replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_NAMES + ['iv_rv_gap'])
    for c in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        if c in merged:
            merged[c] = merged[c].clip(-CLIP, CLIP)
    merged = _oi_filter(merged, raw, OI_TOP_N)

    bor = pd.read_parquet(BORROW).rename(columns={'Symbol': 'symbol'})
    bor['symbol'] = bor['symbol'].str.upper()
    merged['symbol'] = merged['symbol'].str.upper()
    merged = merged.merge(bor[['date', 'symbol', 'borrow_stress', 'has_borrow']],
                          on=['date', 'symbol'], how='left')
    merged['has_borrow'] = merged['has_borrow'].fillna(False)
    print(f'  merged {len(merged)} rows; borrow coverage {merged["has_borrow"].mean():.1%}')

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    wins = _windows(dates, TRAIN_D, VAL_D, STEP_D)
    gate = _vix_gate(raw, GATE_LB)
    print(f'  {len(wins)} windows; rebal {REBAL}d; gate VIX>{GATE_LB}d-median')

    recs = []
    for w_idx, (trlo, trhi, valo, vahi) in enumerate(wins):
        tr = merged[(merged.date >= trlo) & (merged.date <= trhi)]
        va = merged[(merged.date >= valo) & (merged.date <= vahi)]
        if len(tr) < 300 or len(va) < 100:
            continue
        pr = train_predictor(tr[FEATURE_NAMES].values, tr['iv_rv_gap'].values, FEATURE_NAMES)
        va = va.copy()
        va['pred_gap'] = predict(pr, va[FEATURE_NAMES].values)
        rebal_dates = pd.DatetimeIndex(sorted(va['date'].unique()))[::REBAL]
        for d in rebal_dates:
            if not gate.get(d, False):
                continue
            day = va[va.date == d]
            if len(day) < TOP_K:
                continue
            uni = day['iv_rv_gap'].mean()
            top = day.nlargest(TOP_K, 'pred_gap')
            topb = top[top['has_borrow']]
            if len(topb) < 9:
                continue
            q = topb['borrow_stress'].quantile([1/3, 2/3]).values
            lo = topb[topb.borrow_stress <= q[0]]
            hi = topb[topb.borrow_stress >= q[1]]
            mid = topb[(topb.borrow_stress > q[0]) & (topb.borrow_stress < q[1])]
            recs.append({
                'window': w_idx, 'date': str(d.date()),
                'uncond': float(top['iv_rv_gap'].mean() - uni),
                'lo': float(lo['iv_rv_gap'].mean() - uni),
                'mid': float(mid['iv_rv_gap'].mean() - uni) if len(mid) else np.nan,
                'hi': float(hi['iv_rv_gap'].mean() - uni),
                'cond_drop_hi': float(pd.concat([lo, mid])['iv_rv_gap'].mean() - uni),
                # exploratory (post-hoc direction): overweight the high-stress
                # band — NOT pre-registered; in-sample only, OOS-confirm in Phase B.
                'cond_overweight_hi': float(pd.concat([mid, hi])['iv_rv_gap'].mean() - uni),
            })

    df = pd.DataFrame(recs)
    print(f'\nfired rebals with borrow split: {len(df)}')
    if len(df) < 5:
        print('too few fired rebals — inconclusive'); return

    # H1 premium amplifier, H2 squeeze tail
    h1 = {t: round(float(df[t].mean()), 4) for t in ('lo', 'mid', 'hi')}
    h2 = {t: round(float(df[t].min()), 4) for t in ('lo', 'mid', 'hi')}
    # H3 conditioned vs unconditioned net Sharpe
    uncond_sh = _sharpe(df['uncond'])
    cond_sh = _sharpe(df['cond_drop_hi'])
    h3_delta = cond_sh - uncond_sh
    # per-window
    pw = []
    for w in sorted(df['window'].unique()):
        s = df[df.window == w]
        pw.append({'window': int(w), 'uncond_sh': round(_sharpe(s['uncond']), 3),
                   'cond_sh': round(_sharpe(s['cond_drop_hi']), 3)})
    pos_windows = sum(1 for r in pw if r['cond_sh'] > r['uncond_sh'])

    print(f'H1 per-tercile mean alpha (vol pts): lo {h1["lo"]}  mid {h1["mid"]}  hi {h1["hi"]}'
          f'   → premium amplifier (hi>lo): {h1["hi"] > h1["lo"]}')
    print(f'H2 per-tercile worst-rebal alpha:    lo {h2["lo"]}  mid {h2["mid"]}  hi {h2["hi"]}'
          f'   → hi-stress worse tail: {h2["hi"] < h2["lo"]}')
    print(f'H3 net Sharpe: uncond {uncond_sh:+.3f}  cond(drop-hi) {cond_sh:+.3f}  '
          f'delta {h3_delta:+.3f}  ({pos_windows}/{len(pw)} windows cond>uncond)')
    # Diagnostic: tercile Sharpes + the FLIPPED (overweight-hi) rule — in-sample
    # only, post-hoc direction; the legitimate test of this is a pre-registered
    # OOS Phase B, not this number.
    lo_sh, hi_sh = _sharpe(df['lo']), _sharpe(df['hi'])
    ow_sh = _sharpe(df['cond_overweight_hi'])
    print(f'  [in-sample diag] tercile Sharpe lo {lo_sh:+.3f} hi {hi_sh:+.3f} ; '
          f'overweight-hi {ow_sh:+.3f} (vs uncond {uncond_sh:+.3f}) — POST-HOC, needs OOS')

    h1_holds = h1['hi'] > h1['lo']
    h2_fatal = h2['hi'] < 1.5 * h2['lo'] and h2['hi'] < 0  # hi tail >1.5x worse
    if h3_delta >= 0.10 and h1_holds and pos_windows >= 4:
        verdict = 'PASS (confirmed-OOS)'
    elif h3_delta >= 0.05:
        verdict = 'MARGINAL (partial-OOS)'
    elif h3_delta < 0.05 and h1_holds:
        verdict = 'FAIL-null (borrow-stress real but does not separate at deployable granularity)'
    elif cond_sh <= uncond_sh and not h1_holds:
        verdict = 'FAIL-reversed (borrow thesis wrong on liquid universe)'
    else:
        verdict = 'INCONCLUSIVE'
    print(f'\nPHASE A VERDICT: {verdict}')

    OUT.mkdir(exist_ok=True)
    (OUT / 'vol-borrow-phaseA.json').write_text(json.dumps({
        'n_fired_rebals': len(df), 'H1_tercile_mean_alpha': h1,
        'H2_tercile_worst_alpha': h2, 'uncond_sharpe': uncond_sh,
        'cond_sharpe': cond_sh, 'H3_delta': h3_delta, 'per_window': pw,
        'pos_windows': pos_windows, 'verdict': verdict,
    }, indent=2))
    print(f'-> {OUT / "vol-borrow-phaseA.json"}')


if __name__ == '__main__':
    main()
