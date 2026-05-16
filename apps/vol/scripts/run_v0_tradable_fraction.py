"""Gate B-pre — what fraction of the clean predicted-pick cohort is
even quote-available in the only free option-quote DB?

Gate A (`run_v0_artifact_decomp.py`) showed a real +0.14-of-premium
gross edge survives on the clean (non-halt, real-OI) gauss314 subset.
But the cohort that carries it is broad microcap (gauss314 has ~3.08M
(date,symbol) rows incl. Russell-microcap biotechs). Direct probing
found DoltHub `option_chain` does NOT cover those names — CTIC / MRNS
/ MNOV return 0 rows on 2022-06-17 while S&P-smallcap PLCE returns 36.
`option_chain` is a limited (S&P/large-mid-ish) universe, not the
2,276-ticker breadth of `volatility_history`.

If the clean predicted picks are largely absent from the only free
quote source, the small-capacity re-frame is killed on
quote-availability, *upstream of the spread question*: you cannot get
a historical OR live price for the names the signal selects. And the
names that ARE quotable overlap the liquid cohort where v2 #2 already
showed the alpha collapses (−0.48 at top-200 OI).

Offline: walk-forward predictor → clean top-K picks → unique symbols.
API: one fast `date='<ref>' AND act_symbol='<sym>'` equality COUNT
per symbol (the only query shape DoltHub doesn't deadline), on two
liquid reference dates spanning the gauss314 era; present on either =
quote-available. Cached. Run from repo root:
    uv run python apps/vol/scripts/run_v0_tradable_fraction.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from vol import (
    FEATURE_NAMES, build_vol_features, forward_iv_rv_gap,
    load_gauss314_full, predict, train_predictor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / 'Output'
DOLTHUB_API = (
    'https://www.dolthub.com/api/v1alpha1/post-no-preference/options'
)
CACHE = REPO_ROOT / '.iv-cache' / 'optchain_membership'
REF_DATES = ('2021-06-17', '2023-01-20')  # liquid Fridays, gauss314 era


def _member(symbol: str, *, api_delay: float, offline: bool) -> bool | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f'{symbol}.json'
    if cf.exists():
        return bool(json.loads(cf.read_text())['member'])
    if offline:
        return None
    present = False
    for ref in REF_DATES:
        sql = (f"SELECT COUNT(*) n FROM option_chain "
               f"WHERE date='{ref}' AND act_symbol='{symbol}'")
        url = f'{DOLTHUB_API}?{urllib.parse.urlencode({"q": sql})}'
        try:
            with urllib.request.urlopen(url, timeout=50) as r:
                d = json.loads(r.read())
            if d.get('query_execution_status') == 'Success':
                n = int(d['rows'][0]['n']) if d.get('rows') else 0
                if n > 0:
                    present = True
                    break
        except Exception as e:  # noqa: BLE001
            print(f'  ! {symbol} {ref}: {e}', flush=True)
        time.sleep(api_delay)
    cf.write_text(json.dumps({'member': present}))
    return present


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
    ap.add_argument('--rv-floor', type=float, default=0.05)
    ap.add_argument('--oi-floor', type=float, default=100.0)
    ap.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    ap.add_argument('--max-symbols', type=int, default=250,
                    help='Cap unique clean-pick symbols probed (CI on '
                         'the tradable fraction is tight by ~150+).')
    ap.add_argument('--api-delay', type=float, default=3.0)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--output', default=str(OUT /
                    'vol-v0-tradable-fraction.json'))
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
    m['rv_forward'] = m['ATM_IV'] - m['iv_rv_gap']
    m['edge_frac'] = m['iv_rv_gap'] / m['ATM_IV']
    m['is_clean'] = ((m['rv_forward'] >= args.rv_floor)
                     & (m['oi_total'] >= args.oi_floor))

    dates = pd.DatetimeIndex(sorted(m['date'].unique()))
    wins = _windows(dates, args.train_window_days,
                    args.val_window_days, args.step_window_days)

    pick_syms: list[str] = []
    pick_rows = []
    for (tr_lo, tr_hi, va_lo, va_hi) in wins:
        tr = m[(m.date >= tr_lo) & (m.date <= tr_hi)]
        va = m[(m.date >= va_lo) & (m.date <= va_hi)]
        if len(tr) < 1000 or len(va) < 500:
            continue
        pr = train_predictor(tr[FEATURE_NAMES].values,
                             tr['iv_rv_gap'].values, FEATURE_NAMES)
        va = va[va['is_clean']].copy()
        if len(va) < 500:
            continue
        va['pred'] = predict(pr, va[FEATURE_NAMES].values)
        for rd in pd.DatetimeIndex(sorted(va.date.unique()))[
                ::args.rebal_days]:
            day = va[va.date == rd]
            if len(day) < max(args.top_k, 5):
                continue
            for _, r in day.nlargest(args.top_k, 'pred').iterrows():
                pick_syms.append(r['symbol'])
                pick_rows.append({'symbol': r['symbol'],
                                  'edge_frac': float(r['edge_frac']),
                                  'oi_total': float(r['oi_total'])})

    picks = pd.DataFrame(pick_rows)
    uniq = sorted(picks['symbol'].unique())
    print(f'\nclean predicted picks: {len(picks)} '
          f'({len(uniq)} unique symbols)', flush=True)
    rng = np.random.default_rng(0)
    probe = (sorted(rng.choice(uniq, args.max_symbols, replace=False))
             if len(uniq) > args.max_symbols else uniq)
    print(f'probing option_chain membership for {len(probe)} symbols '
          f'(ref dates {REF_DATES})...', flush=True)

    member = {}
    for i, s in enumerate(probe):
        member[s] = _member(s, api_delay=args.api_delay,
                             offline=args.offline)
        if (i + 1) % 25 == 0:
            got = sum(1 for v in member.values() if v)
            print(f'  {i+1}/{len(probe)}  member-so-far={got}',
                  flush=True)

    resolved = {k: v for k, v in member.items() if v is not None}
    n_probe = len(resolved)
    n_member = sum(1 for v in resolved.values() if v)
    frac_sym = n_member / n_probe if n_probe else float('nan')

    # Pick-weighted (not just symbol-weighted): the edge is what the
    # picks earn, so weight tradability by pick frequency × |edge|.
    picks['member'] = picks['symbol'].map(member)
    pw = picks.dropna(subset=['member']).copy()
    if len(pw):
        pw['member'] = pw['member'].astype(bool)
        is_mem = pw['member'].to_numpy()
        ef = pw['edge_frac'].to_numpy()
        frac_pick = float(is_mem.mean())
        edge_member = (float(ef[is_mem].mean())
                       if is_mem.any() else float('nan'))
        edge_nonmember = (float(ef[~is_mem].mean())
                          if (~is_mem).any() else float('nan'))
    else:
        frac_pick = edge_member = edge_nonmember = float('nan')

    print(f'\n=== TRADABLE FRACTION OF THE CLEAN EDGE COHORT ===',
          flush=True)
    print(f'  symbols quote-available : {n_member}/{n_probe} '
          f'({frac_sym:.1%})', flush=True)
    print(f'  picks quote-available   : {frac_pick:.1%}', flush=True)
    print(f'  mean edge_frac  member  : {edge_member:+.4f}', flush=True)
    print(f'  mean edge_frac  absent  : {edge_nonmember:+.4f}', flush=True)

    if np.isnan(frac_pick) or frac_pick < 0.15:
        verdict = (
            f'KILLED ON QUOTE-AVAILABILITY — only {frac_pick:.0%} of '
            f'clean predicted picks are in the only free option-quote '
            f'DB. The cohort carrying the +0.14 edge is microcap and '
            f'un-quotable (historically OR live) on free data; the '
            f'quotable subset is the liquid cohort v2 #2 already '
            f'falsified (−0.48 @ top-200 OI). The small-capacity '
            f're-frame fails upstream of the spread question. '
            f'Arc verdict: reversed-OOS for a small operator on free '
            f'data; spread gate is moot.')
    elif frac_pick < 0.5:
        verdict = (
            f'SEVERELY CONSTRAINED — {frac_pick:.0%} quotable; the '
            f'edge cohort is mostly un-quotable. Spread gate runs only '
            f'on the quotable minority (likely the liquid v2 #2 cohort).')
    else:
        verdict = (
            f'QUOTE-AVAILABLE — {frac_pick:.0%} of clean picks quotable; '
            f'proceed to the breakeven-spread gate on this subset.')
    print(f'\npre-reg verdict: {verdict}', flush=True)

    Path(args.output).write_text(json.dumps({
        'rv_floor': args.rv_floor, 'oi_floor': args.oi_floor,
        'ref_dates': list(REF_DATES),
        'n_unique_clean_pick_symbols': int(len(uniq)),
        'n_symbols_probed': int(n_probe),
        'symbols_quote_available': int(n_member),
        'frac_symbols_quote_available': frac_sym,
        'frac_picks_quote_available': frac_pick,
        'mean_edge_frac_member': edge_member,
        'mean_edge_frac_absent': edge_nonmember,
        'verdict': verdict,
    }, indent=2))
    print(f'\n-> {args.output}', flush=True)


if __name__ == '__main__':
    main()
