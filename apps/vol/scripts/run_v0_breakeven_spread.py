"""Decisive gate — does the unrestricted-universe VRP the v1 predictor
actually picks survive the *measured* thin-name option bid/ask spread,
for a small operator?

`apps/docs/docs/TODO/vol-borrow-illiquid-vrp.md` Stage-1 breakeven
test. The vol arc's +5.86 alpha Sharpe was computed with friction
cancelled by symmetry (alpha-vs-EW). The small-capacity re-frame asks
the question that construction abstracted away: what does it actually
cost to cross the spread on the names the predictor picks?

Cohort definition (fixed after the OI-tercile attempt picked gauss314
ETF/OI=1 noise): the cohort is **whatever the unrestricted v1
predictor top-K picks are, intersected with names that have a real
quoted ATM chain in DoltHub `option_chain`**. The fraction of picks
that are even quotable is itself a deployability result.

Decision-grade conversion (full path-dependent straddle sim deferred
to Stage-1-full iff this survives): short ATM straddle return on
premium ≈ `(IV − RV)/IV = iv_rv_gap/ATM_IV`; round-trip transaction
cost as a fraction of premium = measured straddle `(ask−bid)/mid`.
Held-to-expiry the dominant cost is the entry half-spread (~0.5×
round-trip); both bounds reported.

Edge side: offline gauss314 (reuses v1 predictor + target). Cost side:
near-ATM bid/ask pulled per pick from `option_chain` (delta-band
query, cached to `.iv-cache/option_chain/`, so reruns are free; use
`--offline` to recompute the verdict from cache only).

Run from repo root:
    uv run python apps/vol/scripts/run_v0_breakeven_spread.py \
        --sample-rebal-dates 8 --max-picks-per-rebal 60
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
CACHE_DIR = REPO_ROOT / '.iv-cache' / 'option_chain'


def _fetch_atm(symbol: str, date_str: str, *, horizon_days: int,
               api_delay: float, offline: bool) -> dict | None:
    """Near-ATM call+put bid/ask for `symbol` at the option_chain
    snapshot closest to `date_str`. Cached per (symbol, date)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_f = CACHE_DIR / f'{symbol}_{date_str}.json'
    if cache_f.exists():
        rows = json.loads(cache_f.read_text())
    elif offline:
        return None
    else:
        d1 = (pd.Timestamp(date_str) + pd.Timedelta(days=7)
              ).strftime('%Y-%m-%d')
        sql = (
            "SELECT date,expiration,strike,call_put,bid,ask,delta "
            "FROM option_chain "
            f"WHERE act_symbol='{symbol}' "
            f"AND date >= '{date_str}' AND date <= '{d1}' "
            "AND ((call_put='Call' AND delta BETWEEN 0.40 AND 0.60) "
            "OR (call_put='Put' AND delta BETWEEN -0.60 AND -0.40)) "
            "ORDER BY date,expiration,strike LIMIT 60"
        )
        url = f'{DOLTHUB_API}?{urllib.parse.urlencode({"q": sql})}'
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                rows = json.loads(r.read()).get('rows', [])
        except Exception as e:  # noqa: BLE001
            print(f'  ! {symbol} {date_str}: {e}', flush=True)
            return None
        cache_f.write_text(json.dumps(rows))
        time.sleep(api_delay)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in ('strike', 'bid', 'ask', 'delta'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['bid', 'ask'])
    df = df[(df['bid'] > 0) & (df['ask'] >= df['bid'])]
    if df.empty:
        return None
    df['snap'] = pd.to_datetime(df['date'])
    target = pd.Timestamp(date_str)
    snap = df.loc[(df['snap'] - target).abs().idxmin(), 'snap']
    df = df[df['snap'] == snap].copy()
    df['exp'] = pd.to_datetime(df['expiration'])
    df['dte'] = (df['exp'] - snap).dt.days
    df = df[(df['dte'] >= 7) & (df['dte'] <= 60)]
    if df.empty:
        return None
    exp_pick = df.loc[(df['dte'] - horizon_days).abs().idxmin(), 'exp']
    df = df[df['exp'] == exp_pick]
    calls, puts = df[df['call_put'] == 'Call'], df[df['call_put'] == 'Put']
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
        'straddle_rel_spread': float(strad_spread / strad_mid),
        'dte': int((exp_pick - snap).days),
    }


def _windows(dates: pd.DatetimeIndex, tr: int, va: int, st: int):
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
    ap.add_argument('--sample-rebal-dates', type=int, default=8,
                    help='Evenly-spaced rebal dates sampled for the '
                         'spread pull (bounds API cost; the breakeven '
                         'is a cost characterization, not a Sharpe).')
    ap.add_argument('--max-picks-per-rebal', type=int, default=60)
    ap.add_argument('--clip-iv-hv-ratio', type=float, default=10.0)
    ap.add_argument('--api-delay', type=float, default=0.3)
    ap.add_argument('--offline', action='store_true',
                    help='Recompute verdict from cached fetches only.')
    ap.add_argument('--output', default=str(OUT /
                    'vol-v0-breakeven-spread.json'))
    args = ap.parse_args()

    print('Loading gauss314...', flush=True)
    raw = load_gauss314_full()
    raw['oi_total'] = (raw['calls_open_interest'].fillna(0)
                       + raw['puts_open_interest'].fillna(0))
    panel = build_vol_features(raw)
    target = forward_iv_rv_gap(raw, horizon=args.horizon)
    merged = panel.features.merge(
        target, on=['date', 'symbol'], how='inner'
    ).merge(raw[['date', 'symbol', 'ATM_IV', 'oi_total']],
            on=['date', 'symbol'], how='inner'
    ).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_NAMES + ['iv_rv_gap', 'ATM_IV'])
    for col in ('iv_over_hv20', 'iv_over_hv60', 'iv_over_hv120'):
        merged[col] = merged[col].clip(
            lower=-args.clip_iv_hv_ratio, upper=args.clip_iv_hv_ratio)
    merged = merged[merged['ATM_IV'] > 1e-6].copy()
    merged['edge_frac'] = merged['iv_rv_gap'] / merged['ATM_IV']

    dates = pd.DatetimeIndex(sorted(merged['date'].unique()))
    wins = _windows(dates, args.train_window_days,
                    args.val_window_days, args.step_window_days)
    print(f'  {len(merged):,} rows; {len(dates)} dates; '
          f'{len(wins)} windows', flush=True)

    # Walk-forward → predicted top-K picks on the UNRESTRICTED universe
    # (this is the +5.86 construction). Collect per-rebal pick frames.
    pick_rows = []          # one row per (rebal_date, symbol) predicted pick
    univ_edge_by_date = {}  # rebal_date -> mean edge_frac of all names
    val_rs = []
    for (tr_lo, tr_hi, va_lo, va_hi) in wins:
        tr = merged[(merged.date >= tr_lo) & (merged.date <= tr_hi)]
        va = merged[(merged.date >= va_lo) & (merged.date <= va_hi)]
        if len(tr) < 1000 or len(va) < 500:
            continue
        pr = train_predictor(tr[FEATURE_NAMES].values,
                             tr['iv_rv_gap'].values, FEATURE_NAMES)
        va = va.copy()
        va['pred_gap'] = predict(pr, va[FEATURE_NAMES].values)
        val_rs.append(float(np.corrcoef(
            va['pred_gap'], va['iv_rv_gap'])[0, 1]))
        vd = pd.DatetimeIndex(sorted(va.date.unique()))
        for rd in vd[::args.rebal_days]:
            day = va[va.date == rd]
            if len(day) < max(args.top_k, 5):
                continue
            univ_edge_by_date[rd] = float(day['edge_frac'].mean())
            picks = day.nlargest(args.top_k, 'pred_gap')
            for _, r in picks.iterrows():
                pick_rows.append({
                    'rebal_date': rd, 'symbol': r['symbol'],
                    'edge_frac': float(r['edge_frac']),
                    'oi_total': float(r['oi_total'])})

    picks_df = pd.DataFrame(pick_rows)
    print(f'\nmean val Pearson r = {np.mean(val_rs):+.4f}', flush=True)
    print(f'predicted picks: {len(picks_df)} over '
          f'{picks_df.rebal_date.nunique()} rebal dates', flush=True)

    # Sample evenly-spaced rebal dates to bound the spread pull.
    rebal_dates = sorted(picks_df.rebal_date.unique())
    sidx = np.unique(np.linspace(
        0, len(rebal_dates) - 1, args.sample_rebal_dates).astype(int))
    sampled = [rebal_dates[i] for i in sidx]
    print(f'sampling {len(sampled)} rebal dates for the spread pull '
          f'(<= {args.max_picks_per_rebal} picks each)', flush=True)

    rows = []
    for rd in sampled:
        sub = picks_df[picks_df.rebal_date == rd].head(
            args.max_picks_per_rebal)
        ds = pd.Timestamp(rd).strftime('%Y-%m-%d')
        n_ok = 0
        for _, r in sub.iterrows():
            sp = _fetch_atm(r['symbol'], ds, horizon_days=args.horizon,
                            api_delay=args.api_delay, offline=args.offline)
            if sp is None:
                rows.append({**r, 'tradable': False,
                             'straddle_rel_spread': np.nan})
            else:
                n_ok += 1
                rows.append({**r, 'tradable': True,
                             'straddle_rel_spread':
                             sp['straddle_rel_spread']})
        print(f'  {ds}: {n_ok}/{len(sub)} tradable', flush=True)

    res = pd.DataFrame(rows)
    res.to_pickle(OUT / 'vol-v0-breakeven-picks.pkl')
    trad = res[res.tradable]
    n_attempt, n_trad = len(res), len(trad)
    trad_frac = n_trad / n_attempt if n_attempt else 0.0

    summary: dict = {
        'mean_val_pearson_r': float(np.mean(val_rs)) if val_rs else None,
        'n_picks_attempted': int(n_attempt),
        'n_picks_tradable': int(n_trad),
        'tradable_fraction': float(trad_frac),
        'note': 'edge_frac ≈ (IV−RV)/IV decision-grade short-straddle '
                'return-on-premium; path-dependent sim deferred to '
                'Stage-1-full iff this gate survives',
    }
    print(f'\n=== TRADABILITY: {n_trad}/{n_attempt} '
          f'({trad_frac:.1%}) predicted picks have a quoted ATM chain ===',
          flush=True)

    if n_trad >= 5:
        rs = trad['straddle_rel_spread'].to_numpy()
        E = float(trad['edge_frac'].mean())
        pct = {f'p{int(q*100)}': float(np.quantile(rs, q))
               for q in (.1, .25, .5, .75, .9)}
        C_rt, C_en = pct['p50'], 0.5 * pct['p50']
        net_rt, net_en = E - C_rt, E - C_en
        be_rt = E
        frac_below = float((rs <= be_rt).mean())
        # Honest net alpha Sharpe (symmetry removed): per-rebal mean
        # tradable-pick edge minus per-rebal median measured spread.
        per_reb = (trad.groupby('rebal_date')
                   .agg(g=('edge_frac', 'mean'),
                        c=('straddle_rel_spread', 'median')))
        net_series_rt = (per_reb['g'] - per_reb['c']).to_numpy()
        net_series_en = (per_reb['g'] - 0.5 * per_reb['c']).to_numpy()
        ann = float(np.sqrt(252.0 / args.rebal_days))

        def _sh(a):
            a = np.asarray(a, float)
            return (float(a.mean() / a.std(ddof=1) * ann)
                    if a.size > 1 and a.std(ddof=1) > 1e-12 else 0.0)

        sh_rt, sh_en = _sh(net_series_rt), _sh(net_series_en)
        summary.update({
            'gross_edge_frac_of_premium_mean': E,
            'measured_straddle_rel_spread_pct': pct,
            'net_frac_roundtrip': net_rt,
            'net_frac_entry_only': net_en,
            'breakeven_roundtrip_spread': be_rt,
            'breakeven_entry_only_spread': 2.0 * E,
            'fraction_tradable_below_breakeven': frac_below,
            'net_alpha_sharpe_roundtrip': sh_rt,
            'net_alpha_sharpe_entry_only': sh_en,
            'n_sampled_rebals': int(per_reb.shape[0]),
            'median_pick_oi_total': float(trad['oi_total'].median()),
        })
        print(f'gross edge / premium (mean)        = {E:+.4f}', flush=True)
        print('measured straddle round-trip spread:', flush=True)
        for k in ('p10', 'p25', 'p50', 'p75', 'p90'):
            print(f'  {k}: {pct[k]:.4f}', flush=True)
        print(f'net / premium  round-trip          = {net_rt:+.4f}',
              flush=True)
        print(f'net / premium  entry-only          = {net_en:+.4f}',
              flush=True)
        print(f'breakeven round-trip spread        = {be_rt:.4f} '
              f'(entry-only {2*E:.4f})', flush=True)
        print(f'tradable picks ≤ breakeven         = {frac_below:.1%}',
              flush=True)
        print(f'net alpha Sharpe  round-trip       = {sh_rt:+.3f} '
              f'(n={per_reb.shape[0]} rebals; small-n, directional)',
              flush=True)
        print(f'net alpha Sharpe  entry-only       = {sh_en:+.3f}',
              flush=True)

        if net_en <= 0:
            v = (f'FAIL/reversed-OOS — even entry-only cost '
                 f'({C_en:.3f}) exceeds gross edge ({E:+.4f}); the '
                 f'+5.86 reverses to net-negative for a small '
                 f'operator. Arc original kill correct here too.')
        elif net_rt <= 0:
            v = (f'MARGINAL/partial-OOS — survives entry-only '
                 f'(net {net_en:+.4f}) not round-trip ({net_rt:+.4f}); '
                 f'deployable only held-to-expiry, near-zero early exit.')
        else:
            v = (f'SURVIVES — net positive round-trip '
                 f'({net_rt:+.4f}); escalate to borrow-conditioned '
                 f'full walk-forward (Stage-1-full).')
        summary['verdict'] = v
        print(f'\npre-reg verdict: {v}', flush=True)
    else:
        summary['verdict'] = (
            f'INSUFFICIENT — only {n_trad} tradable picks; '
            f'tradable_fraction={trad_frac:.1%} is itself the '
            f'deployability signal (predicted picks largely un-quotable).')
        print(f'\n{summary["verdict"]}', flush=True)

    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f'\n-> {args.output}', flush=True)


if __name__ == '__main__':
    main()
