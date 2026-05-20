"""Forward structural coverage probe — does Tradier's OPRA feed carry the
v1 microcap pick cohort *today*?

Pre-reg: `apps/docs/docs/TODO/vol-tradier-forward-coverage.md`. Resolves
the open question the falsified [`vol-borrow-illiquid-vrp`](../../docs/docs/findings/vol-borrow-illiquid-vrp-falsified.md)
arc left dangling: is the missing 88% of the v1 clean-pick cohort
absent from **DoltHub specifically** (publisher cut) or from **OPRA
itself** (not options-listed)? If absent-from-OPRA, no paid vendor at
any price helps — definitive close. If OPRA-listed, paid historical
sources (ORATS $99/mo, Polygon Advanced ~$199/mo) become viable and the
spend question is live.

Tradier sandbox returns delayed-but-real chains from the consolidated
OPRA feed with greeks + IV via ORATS — **the cheapest source of truth
on OPRA listings**, $0, no funding. Caveat: returns *today's* chains,
not 2019-2023 historical, so Tradier-today coverage is a lower bound on
historical OPRA coverage (a symbol delisted since 2021 won't show
today). That's fine for the decision: if Tradier-today already covers
≥50%, paid OPRA vendors definitely carry the cohort; if it's ≈ DoltHub's
~12%, the cohort wasn't broadly OPRA-listed even then.

Pre-registered bands (symbol-weighted, finite ATM bid/ask):
  ≥ 50% → COVERED — paid spend potentially viable; next gate = historical replay
  15-50% → SEVERELY CONSTRAINED — same as falsified arc; don't pay
  < 15% → KILLED — OPRA itself doesn't carry these names; close the arc

Auth: set `TRADIER_TOKEN` in your shell (never paste in chat). Run:
    uv run python apps/vol/scripts/run_v0_tradier_forward_coverage.py
Flags: `--production` (use api.tradier.com, real-time tier; default is
sandbox/delayed which is what the pre-reg uses), `--max-symbols N`
(smoke cap), `--offline` (re-summarize from cache without API calls).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / 'Output'
PICKS_PKL = OUT / 'vol-v0-breakeven-picks.pkl'
CACHE = REPO_ROOT / '.iv-cache' / 'tradier_chains'
SANDBOX = 'https://sandbox.tradier.com'
PRODUCTION = 'https://api.tradier.com'


def _get(url: str, token: str) -> dict | None:
    """One authed GET; returns parsed JSON or None on transport error."""
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')[:200]
        print(f'  ! HTTP {e.code} on {url}\n    {body}', flush=True)
        return None
    except Exception as e:  # noqa: BLE001
        print(f'  ! {type(e).__name__} on {url}: {e}', flush=True)
        return None


def _expirations(symbol: str, base: str, token: str, *,
                 api_delay: float) -> list[str] | None:
    """Return list of YYYY-MM-DD expirations for `symbol`, or None if
    the symbol has no OPRA-listed options."""
    cf = CACHE / f'{symbol}__expirations.json'
    if cf.exists():
        d = json.loads(cf.read_text())
        return d.get('expirations')
    url = (f'{base}/v1/markets/options/expirations?'
           f'symbol={urllib.parse.quote(symbol)}'
           f'&includeAllRoots=true&strikes=false')
    js = _get(url, token)
    time.sleep(api_delay)
    if js is None:
        return None  # transport error: don't cache
    exps = js.get('expirations')
    if not exps or exps == 'null':
        out: list[str] = []
    else:
        d = exps.get('date') if isinstance(exps, dict) else None
        out = list(d) if isinstance(d, list) else ([d] if d else [])
    cf.write_text(json.dumps({'expirations': out}))
    return out


def _chain(symbol: str, expiration: str, base: str, token: str, *,
           api_delay: float) -> list[dict] | None:
    """Full contract list for (symbol, expiration) with greeks. None on
    transport error; [] for empty chain."""
    cf = CACHE / f'{symbol}__{expiration}.json'
    if cf.exists():
        return json.loads(cf.read_text()).get('options', [])
    url = (f'{base}/v1/markets/options/chains?'
           f'symbol={urllib.parse.quote(symbol)}'
           f'&expiration={expiration}&greeks=true')
    js = _get(url, token)
    time.sleep(api_delay)
    if js is None:
        return None
    opts = js.get('options')
    rows = (opts.get('option') if isinstance(opts, dict) else None) or []
    if isinstance(rows, dict):
        rows = [rows]
    cf.write_text(json.dumps({'options': rows}))
    return rows


def _pick_expiration(exps: list[str], min_dte: int = 14) -> str | None:
    """Nearest expiration ≥ `min_dte` calendar days from today."""
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    fut = [(e, (pd.Timestamp(e) - today).days) for e in exps]
    eligible = [(e, d) for e, d in fut if d >= min_dte]
    if not eligible:
        return None
    eligible.sort(key=lambda x: x[1])
    return eligible[0][0]


def _atm_straddle_rel_spread(rows: list[dict]) -> dict | None:
    """Find call (delta near +0.5) + put (delta near −0.5), report
    relative spread + greeks-populated booleans. None if either side
    can't be resolved with finite bid/ask."""
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    # Greeks nested as dict per-row when greeks=true.
    df['delta'] = df['greeks'].apply(
        lambda g: g.get('delta') if isinstance(g, dict) else None)
    df['mid_iv'] = df['greeks'].apply(
        lambda g: g.get('mid_iv') if isinstance(g, dict) else None)
    for c in ('strike', 'bid', 'ask', 'delta', 'mid_iv'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['option_type'] = df['option_type'].str.lower()
    df = df.dropna(subset=['bid', 'ask', 'delta'])
    df = df[(df['bid'] > 0) & (df['ask'] >= df['bid'])]
    calls = df[df['option_type'] == 'call']
    puts = df[df['option_type'] == 'put']
    if calls.empty or puts.empty:
        return None
    c = calls.loc[(calls['delta'] - 0.5).abs().idxmin()]
    p = puts.loc[(puts['delta'] + 0.5).abs().idxmin()]
    c_mid, p_mid = 0.5 * (c.bid + c.ask), 0.5 * (p.bid + p.ask)
    if c_mid <= 0 or p_mid <= 0:
        return None
    strad_mid = c_mid + p_mid
    strad_spread = (c.ask - c.bid) + (p.ask - p.bid)
    return {
        'atm_rel_spread': float(strad_spread / strad_mid),
        'call_strike': float(c.strike), 'put_strike': float(p.strike),
        'call_delta': float(c.delta), 'put_delta': float(p.delta),
        'greeks_ok': bool(np.isfinite(c.delta) and np.isfinite(p.delta)),
        'iv_ok': bool(np.isfinite(c.mid_iv) and np.isfinite(p.mid_iv)
                      and c.mid_iv > 0 and p.mid_iv > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--picks-pkl', default=str(PICKS_PKL))
    ap.add_argument('--production', action='store_true',
                    help='Use api.tradier.com (real-time, funded tier) '
                         'instead of sandbox.')
    ap.add_argument('--api-delay', type=float, default=0.7)
    ap.add_argument('--max-symbols', type=int, default=0,
                    help='Cap unique symbols probed; 0 = all.')
    ap.add_argument('--min-dte', type=int, default=14)
    ap.add_argument('--offline', action='store_true',
                    help='Re-summarize from cache only (no API calls).')
    ap.add_argument('--output', default=str(OUT /
                    'vol-v0-tradier-forward-coverage.json'))
    args = ap.parse_args()

    token = os.environ.get('TRADIER_TOKEN', '').strip()
    if not args.offline and not token:
        print('ERROR: set TRADIER_TOKEN in your shell '
              '(export TRADIER_TOKEN=<your-paper-key>) then re-run.',
              file=sys.stderr)
        return 2
    base = PRODUCTION if args.production else SANDBOX
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    pkl = Path(args.picks_pkl)
    if not pkl.exists():
        print(f'ERROR: cohort artifact {pkl} not found. Run '
              f'`uv run python apps/vol/scripts/run_v0_breakeven_spread.py` '
              'first to generate it.', file=sys.stderr)
        return 2
    picks = pd.read_pickle(pkl)
    if 'symbol' not in picks.columns or 'edge_frac' not in picks.columns:
        print(f'ERROR: picks pkl missing expected columns '
              f'{list(picks.columns)}', file=sys.stderr)
        return 2

    # Pick-weighted denominator = total picks; symbol-weighted = unique
    # symbols. The falsified arc's 7.5%/12% numbers used both.
    by_sym = picks.groupby('symbol').agg(
        n_picks=('edge_frac', 'size'),
        edge_frac_mean=('edge_frac', 'mean')).reset_index()
    uniq = sorted(by_sym['symbol'].astype(str).unique())
    if args.max_symbols > 0:
        uniq = uniq[:args.max_symbols]
        by_sym = by_sym[by_sym['symbol'].isin(uniq)]
    print(f'cohort: {len(uniq)} unique pick-symbols from {pkl.name} '
          f'(total picks={int(by_sym.n_picks.sum())})')
    print(f'endpoint: {base} '
          f'({"PRODUCTION (real-time)" if args.production else "sandbox (delayed)"})')

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, sym in enumerate(uniq):
        rec: dict = {'symbol': sym}
        exps = _expirations(sym, base, token, api_delay=args.api_delay)
        if exps is None:
            rec.update({'status': 'transport_error', 'has_chain': False})
            rows.append(rec)
            continue
        if not exps:
            rec.update({'status': 'no_options_listed', 'has_chain': False})
            rows.append(rec)
            if (i + 1) % 25 == 0:
                print(f'  {i+1}/{len(uniq)} '
                      f'({time.perf_counter()-t0:.0f}s)', flush=True)
            continue
        chosen = _pick_expiration(exps, min_dte=args.min_dte)
        if chosen is None:
            rec.update({'status': 'no_expiry_in_range', 'has_chain': True,
                        'expirations_count': len(exps)})
            rows.append(rec)
            continue
        chain = _chain(sym, chosen, base, token, api_delay=args.api_delay)
        if chain is None:
            rec.update({'status': 'transport_error', 'has_chain': True,
                        'expiration': chosen})
            rows.append(rec)
            continue
        atm = _atm_straddle_rel_spread(chain)
        rec.update({
            'status': 'covered' if atm else 'no_atm_quote',
            'has_chain': True,
            'expiration': chosen,
            'n_contracts': len(chain),
            **(atm or {}),
        })
        rows.append(rec)
        if (i + 1) % 25 == 0:
            covered = sum(1 for r in rows if r.get('status') == 'covered')
            print(f'  {i+1}/{len(uniq)}  covered={covered}  '
                  f'({time.perf_counter()-t0:.0f}s)', flush=True)

    res = pd.DataFrame(rows).merge(by_sym, on='symbol', how='left')

    # ---- Summary against the pre-reg bands ----
    n = len(res)
    n_listed = int((res['has_chain'] == True).sum())  # noqa: E712
    n_covered = int((res['status'] == 'covered').sum())
    frac_sym_listed = n_listed / n if n else float('nan')
    frac_sym_covered = n_covered / n if n else float('nan')
    pw_total = int(res['n_picks'].fillna(0).sum())
    pw_covered = int(res.loc[res['status'] == 'covered', 'n_picks']
                     .fillna(0).sum())
    frac_pick_covered = pw_covered / pw_total if pw_total else float('nan')

    # Spread distribution on the covered subset.
    cov = res[res['status'] == 'covered']
    if len(cov):
        sp = cov['atm_rel_spread'].astype(float).to_numpy()
        spread_pct = {f'p{int(q*100)}': float(np.quantile(sp, q))
                      for q in (.1, .25, .5, .75, .9)}
        edge_cov_mean = float(cov['edge_frac_mean'].mean())
        edge_abs_mean = float(res[res['status'] != 'covered']
                              ['edge_frac_mean'].mean()
                              if (res['status'] != 'covered').any()
                              else float('nan'))
        greeks_ok_rate = float(cov['greeks_ok'].astype(bool).mean())
        iv_ok_rate = float(cov['iv_ok'].astype(bool).mean())
    else:
        spread_pct = {}
        edge_cov_mean = edge_abs_mean = float('nan')
        greeks_ok_rate = iv_ok_rate = 0.0

    if np.isnan(frac_sym_covered) or frac_sym_covered < 0.15:
        verdict = (
            f'KILLED — only {frac_sym_covered:.1%} of v1 pick-symbols '
            f'have a finite ATM bid/ask in OPRA today. The cohort is '
            f'structurally absent from OPRA itself (≈ DoltHub coverage), '
            f'not just from DoltHub. No paid vendor at any price helps. '
            f'Definitive close of the illiquid arc; lock v3 regime-gated '
            f'liquid as the deployable form.')
    elif frac_sym_covered < 0.50:
        verdict = (
            f'SEVERELY CONSTRAINED — {frac_sym_covered:.1%} OPRA-listed '
            f'with ATM quotes; same conclusion as the falsified arc; do '
            f'not pay. Spread distribution on the covered subset (p50): '
            f'{spread_pct.get("p50", float("nan")):.3f}.')
    else:
        verdict = (
            f'COVERED — {frac_sym_covered:.1%} OPRA-listed with ATM '
            f'quotes. The falsified arc\'s DoltHub-coverage gap was NOT '
            f'an OPRA-coverage gap. Paid historical NBBO (ORATS $99/mo '
            f'or Polygon Advanced ~$199/mo) becomes viable; next gate = '
            f'historical replay on a paid source. Spread p50 today: '
            f'{spread_pct.get("p50", float("nan")):.3f}.')

    summary = {
        'endpoint': base,
        'tier': 'production' if args.production else 'sandbox',
        'cohort_source': str(pkl),
        'n_unique_symbols_probed': n,
        'n_with_options_chain': n_listed,
        'n_with_finite_atm_quote': n_covered,
        'frac_sym_options_listed': frac_sym_listed,
        'frac_sym_with_atm_quote': frac_sym_covered,
        'frac_pick_with_atm_quote': frac_pick_covered,
        'atm_rel_spread_pct_covered': spread_pct,
        'mean_edge_frac_covered': edge_cov_mean,
        'mean_edge_frac_absent': edge_abs_mean,
        'greeks_populated_rate_covered': greeks_ok_rate,
        'iv_populated_rate_covered': iv_ok_rate,
        'status_breakdown': res['status'].value_counts().to_dict(),
        'verdict': verdict,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2,
                                            default=float))
    res.to_pickle(OUT / 'vol-v0-tradier-forward-coverage.pkl')
    print('\n=== TRADIER FORWARD COVERAGE ===', flush=True)
    print(f'  unique pick-symbols       : {n}', flush=True)
    print(f'  options-listed (symbol)   : {n_listed}/{n} '
          f'({frac_sym_listed:.1%})', flush=True)
    print(f'  finite ATM quote (symbol) : {n_covered}/{n} '
          f'({frac_sym_covered:.1%})', flush=True)
    print(f'  finite ATM quote (pick-wt): {frac_pick_covered:.1%}',
          flush=True)
    if spread_pct:
        print('  ATM rel-spread (covered)  : '
              f'p25={spread_pct["p25"]:.3f}  p50={spread_pct["p50"]:.3f}'
              f'  p75={spread_pct["p75"]:.3f}  p90={spread_pct["p90"]:.3f}',
              flush=True)
        print(f'  greeks populated rate     : {greeks_ok_rate:.1%}',
              flush=True)
        print(f'  IV populated rate         : {iv_ok_rate:.1%}',
              flush=True)
    print(f'\npre-reg verdict: {verdict}', flush=True)
    print(f'\n-> {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
