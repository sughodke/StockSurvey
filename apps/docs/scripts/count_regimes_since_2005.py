"""Count strategy-winning regimes since 2005.

A *regime* is a maximal contiguous interval (in trading days) over which
the same arc has the highest rolling Sharpe ratio among arcs available
during that window. A regime transition is when the rolling-winner
identity changes.

This is read-only research probe — it does not modify any arc artifact.
See `.research-regimes-since-2005.md` for the deliverable + interpretation.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

LOOKBACKS = [63, 126, 252, 504]


# -----------------------------------------------------------------------------
# Arc loaders → daily-frequency series with DatetimeIndex
# -----------------------------------------------------------------------------

def load_dca_daily() -> pd.Series:
    with open(REPO_ROOT / 'Output/cfr_phase4d_multiasset_close.pkl', 'rb') as f:
        close = pickle.load(f)
    from cfr.baselines import PassiveEW
    daily = np.asarray(
        PassiveEW(rebal_days=80, commission_bps=10.0).daily_returns(close),
        dtype=np.float64)
    return pd.Series(daily, index=close.index, dtype=np.float64).dropna()


def load_gate_daily() -> pd.Series:
    d = np.load(OUTPUT / 'gate-returns.npz', allow_pickle=True)
    dates = pd.to_datetime(np.asarray(d['dates'], dtype=str))
    return pd.Series(np.asarray(d['gated_ret'], dtype=np.float64), index=dates).dropna()


def load_pairs_daily() -> pd.Series:
    d = np.load(OUTPUT / 'pairs-returns.npz', allow_pickle=True)
    dates = pd.to_datetime(np.asarray(d['dates'], dtype=str))
    return pd.Series(np.asarray(d['agg_ret'], dtype=np.float64), index=dates).dropna()


def load_relational_daily(ref_idx: pd.DatetimeIndex) -> pd.Series:
    """1241 daily obs, no dates → tail-align to DCA's last 1241 bars.
    Documented caveat: this is approximate."""
    d = np.load(OUTPUT / 'relational-returns.npz', allow_pickle=True)
    arr = np.asarray(d['val_daily_ret'], dtype=np.float64)
    idx = ref_idx[-arr.size:]
    return pd.Series(arr, index=idx)


def load_dca_winner_4etf_daily(ref_idx: pd.DatetimeIndex) -> pd.Series:
    d = np.load(OUTPUT / 'dca-winner-4etf-returns.npz', allow_pickle=True)
    arr = np.asarray(d['daily_ret'], dtype=np.float64)
    # Tail-align (5245 bars vs DCA's 5232 — drop the head excess)
    n = min(arr.size, len(ref_idx))
    idx = ref_idx[-n:]
    return pd.Series(arr[-n:], index=idx)


def load_regime_scaled_daily(
        ref_idx: pd.DatetimeIndex, arm: str) -> pd.Series:
    """Load `vol_target` or `dd_gate` arm from the regime-app
    scaled-DCA artifact. Length is 5231 (DCA panel less one); tail-align
    to ref_idx. Train-period entries (val_mask=False) are masked NaN so
    rolling-Sharpe is computed only on the arm's deployable val
    horizon — same treatment we'd give any other walk-forward arc."""
    d = np.load(OUTPUT / 'regime-scaled-dca-returns.npz', allow_pickle=True)
    arr = np.asarray(d[arm], dtype=np.float64)
    val_mask = np.asarray(d['val_mask'], dtype=bool)
    arr = np.where(val_mask, arr, np.nan)
    n = min(arr.size, len(ref_idx))
    idx = ref_idx[-n:]
    return pd.Series(arr[-n:], index=idx)


def load_vol_v3_daily(ref_idx: pd.DatetimeIndex) -> pd.Series:
    """Block-stream with rebal_dates → expanded to daily via forward-fill.
    Each rebal block holds for ~20 trading days; we spread each block's
    return uniformly across the trading days from that rebal to the next."""
    d = np.load(OUTPUT / 'vol-v3-dolthub-oos-c200-returns.npz', allow_pickle=True)
    dates = pd.to_datetime(np.asarray(d['rebal_dates'], dtype=str))
    alpha = np.asarray(d['full_panel_alpha'], dtype=np.float64)
    s = pd.Series(np.nan, index=ref_idx)
    for i, (dt, a) in enumerate(zip(dates, alpha)):
        # find DCA-index positions in [dt, next_dt)
        if i + 1 < len(dates):
            mask = (ref_idx >= dt) & (ref_idx < dates[i + 1])
        else:
            # last block: spread over next 20 trading days
            pos = ref_idx.searchsorted(dt, side='left')
            mask = np.zeros(len(ref_idx), dtype=bool)
            mask[pos: pos + 20] = True
        n_days = int(mask.sum())
        if n_days > 0:
            # compound return per day s.t. (1+r_d)^n = (1+a)
            r_d = (1.0 + a) ** (1.0 / n_days) - 1.0 if (1.0 + a) > 0 else -1.0
            s.iloc[mask] = r_d
    return s.dropna()


# -----------------------------------------------------------------------------
# Build daily-frequency master DataFrame
# -----------------------------------------------------------------------------

def build_master(dca_daily: pd.Series) -> pd.DataFrame:
    idx = dca_daily.index
    cols: dict[str, pd.Series] = {}
    cols['dca'] = dca_daily.reindex(idx)
    cols['gate'] = load_gate_daily().reindex(idx)
    cols['pairs'] = load_pairs_daily().reindex(idx)
    cols['relational'] = load_relational_daily(idx).reindex(idx)
    cols['dca_winner_4etf'] = load_dca_winner_4etf_daily(idx).reindex(idx)
    cols['regime_vol_target'] = load_regime_scaled_daily(idx, 'vol_target').reindex(idx)
    cols['regime_dd_gate'] = load_regime_scaled_daily(idx, 'dd_gate').reindex(idx)
    cols['vol_v3'] = load_vol_v3_daily(idx).reindex(idx)
    df = pd.DataFrame(cols, index=idx)
    return df


# -----------------------------------------------------------------------------
# Rolling Sharpe + regime identification
# -----------------------------------------------------------------------------

def rolling_sharpe(returns: pd.DataFrame, L: int, ppy: float = 252.0) -> pd.DataFrame:
    """Annualized rolling Sharpe per arc. NaN if fewer than L*0.7 obs in window
    or if std < 1e-12."""
    mn = returns.rolling(L, min_periods=int(L * 0.7)).mean()
    sd = returns.rolling(L, min_periods=int(L * 0.7)).std(ddof=1)
    sh = (mn / sd) * math.sqrt(ppy)
    sh = sh.where(sd > 1e-12)
    return sh


def winner_series(sharpes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """For each day, return (winner_name, lead_margin_over_runnerup).
    NaN where fewer than 2 arcs are valid."""
    arcs = sharpes.columns.tolist()
    arr = sharpes.to_numpy()
    n = arr.shape[0]
    winners = np.full(n, '', dtype=object)
    margins = np.full(n, np.nan)
    for t in range(n):
        row = arr[t]
        valid = ~np.isnan(row)
        n_valid = int(valid.sum())
        if n_valid < 1:
            winners[t] = None; continue
        idxs = np.flatnonzero(valid)
        vals = row[idxs]
        order = np.argsort(-vals)
        winners[t] = arcs[idxs[order[0]]]
        if n_valid >= 2:
            margins[t] = vals[order[0]] - vals[order[1]]
        else:
            margins[t] = np.nan
    return (
        pd.Series(winners, index=sharpes.index, dtype=object),
        pd.Series(margins, index=sharpes.index, dtype=float),
    )


def find_regimes(winners: pd.Series, margins: pd.Series,
                 min_persistence: int = 1) -> list[dict]:
    """Group consecutive identical-winner days into regimes.
    Skip leading NaN winners (i.e., periods where no rolling-Sharpe is computed)."""
    regs: list[dict] = []
    cur_winner = None
    cur_start = None
    cur_margins: list[float] = []
    for date, w in winners.items():
        if w is None or (isinstance(w, float) and np.isnan(w)):
            # gap: close any current regime
            if cur_winner is not None:
                regs.append({
                    'winner': cur_winner, 'start': cur_start, 'end': prev_date,
                    'length_days': (prev_date - cur_start).days,
                    'length_trading_days': cur_len,
                    'mean_margin': float(np.nanmean(cur_margins)) if cur_margins else float('nan'),
                })
                cur_winner = None; cur_start = None; cur_margins = []; cur_len = 0
            prev_date = date
            continue
        if w != cur_winner:
            if cur_winner is not None:
                regs.append({
                    'winner': cur_winner, 'start': cur_start, 'end': prev_date,
                    'length_days': (prev_date - cur_start).days,
                    'length_trading_days': cur_len,
                    'mean_margin': float(np.nanmean(cur_margins)) if cur_margins else float('nan'),
                })
            cur_winner = w; cur_start = date; cur_margins = []; cur_len = 0
        cur_margins.append(margins.loc[date])
        cur_len += 1
        prev_date = date
    if cur_winner is not None:
        regs.append({
            'winner': cur_winner, 'start': cur_start, 'end': prev_date,
            'length_days': (prev_date - cur_start).days,
            'length_trading_days': cur_len,
            'mean_margin': float(np.nanmean(cur_margins)) if cur_margins else float('nan'),
        })
    return [r for r in regs if r['length_trading_days'] >= min_persistence]


# -----------------------------------------------------------------------------
# Macro cross-tab
# -----------------------------------------------------------------------------

NBER_RECESSIONS = [
    (pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-30')),
    (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-30')),
]
FED_TIGHTENING = [
    (pd.Timestamp('2015-12-01'), pd.Timestamp('2018-12-31')),
    (pd.Timestamp('2022-03-01'), pd.Timestamp('2023-07-31')),
]


def regime_overlaps(reg: dict, intervals: list[tuple]) -> bool:
    for lo, hi in intervals:
        if not (reg['end'] < lo or reg['start'] > hi):
            return True
    return False


def vix_classification(reg: dict, vix: pd.Series, med: float, p90: float) -> dict:
    mask = (vix.index >= reg['start']) & (vix.index <= reg['end'])
    v = vix.loc[mask].dropna()
    if v.empty:
        return {'mean_vix': float('nan'), 'frac_above_med': 0.0, 'frac_above_p90': 0.0}
    return {
        'mean_vix': float(v.mean()),
        'frac_above_med': float((v > med).mean()),
        'frac_above_p90': float((v > p90).mean()),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print('Loading DCA reference + master arc panel...')
    dca = load_dca_daily()
    print(f'  DCA: {dca.index[0].date()} → {dca.index[-1].date()} n={len(dca)}')
    df = build_master(dca)
    print('\n--- arc coverage (non-NaN days, daily-aligned to DCA) ---')
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            print(f'  {c:18s}  EMPTY')
        else:
            print(f'  {c:18s}  n={len(s):>5d}  {s.index[0].date()} → {s.index[-1].date()}')

    # Load VIX for macro cross-tab
    from ss_macro.loaders import load_fred_series
    vix = load_fred_series('VIXCLS')
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix = vix.dropna().astype(float)
    # Restrict VIX to the sample window
    vix = vix.loc[(vix.index >= df.index[0]) & (vix.index <= df.index[-1])]
    vix_med = float(vix.median())
    vix_p90 = float(vix.quantile(0.9))
    print(f'\nVIX sample {vix.index[0].date()}→{vix.index[-1].date()} median={vix_med:.2f} p90={vix_p90:.2f}')

    # Per-lookback regime analysis
    all_results: dict[str, dict] = {}
    for L in LOOKBACKS:
        print(f'\n========== L = {L} trading days ({L/252:.2f}y) ==========')
        sh = rolling_sharpe(df, L)
        winners, margins = winner_series(sh)
        # only count from the first day a winner exists
        first_valid = winners.first_valid_index()
        if first_valid is None:
            continue
        regimes_all = find_regimes(winners.loc[first_valid:], margins.loc[first_valid:])
        regimes_21 = [r for r in regimes_all if r['length_trading_days'] >= 21]
        lengths = np.array([r['length_trading_days'] for r in regimes_all])
        print(f'  Total regimes: {len(regimes_all)}')
        print(f'  Regimes ≥21 trading days (filter): {len(regimes_21)}')
        if len(lengths):
            print(f'  Length stats (TD): min={lengths.min()} median={np.median(lengths):.0f} '
                  f'mean={lengths.mean():.1f} max={lengths.max()}')
        # transitions
        transitions = []
        for prev, nxt in zip(regimes_all[:-1], regimes_all[1:]):
            transitions.append({
                'date': str(nxt['start'].date()),
                'prev': prev['winner'], 'next': nxt['winner'],
                'prev_length_td': prev['length_trading_days'],
            })
        print(f'  Transitions: {len(transitions)}')
        # top-10 most-persistent
        top = sorted(regimes_all, key=lambda r: -r['length_trading_days'])[:10]
        print(f'  Top {len(top)} most-persistent:')
        for r in top:
            print(f'    {r["winner"]:18s}  {r["start"].date()} → {r["end"].date()}  '
                  f'TD={r["length_trading_days"]:>4d}  margin={r["mean_margin"]:+.3f}')
        # macro cross-tab on regimes ≥21d
        macro_rows = []
        for r in regimes_21:
            vclass = vix_classification(r, vix, vix_med, vix_p90)
            macro_rows.append({
                'winner': r['winner'],
                'start': str(r['start'].date()), 'end': str(r['end'].date()),
                'length_td': r['length_trading_days'],
                'mean_vix': vclass['mean_vix'],
                'frac_above_med_vix': vclass['frac_above_med'],
                'frac_above_p90_vix': vclass['frac_above_p90'],
                'overlaps_recession': regime_overlaps(r, NBER_RECESSIONS),
                'overlaps_fed_tightening': regime_overlaps(r, FED_TIGHTENING),
                'mean_margin': r['mean_margin'],
            })
        # Counts per regime context
        n_recess = sum(1 for m in macro_rows if m['overlaps_recession'])
        n_tight = sum(1 for m in macro_rows if m['overlaps_fed_tightening'])
        n_high_vix = sum(1 for m in macro_rows if m['frac_above_med_vix'] >= 0.5)
        n_extreme_vix = sum(1 for m in macro_rows if m['frac_above_p90_vix'] >= 0.2)
        print(f'  Macro overlap (regimes ≥21d): recession={n_recess}/{len(macro_rows)} '
              f'fed-tight={n_tight}/{len(macro_rows)} '
              f'highVIX={n_high_vix}/{len(macro_rows)} extremeVIX={n_extreme_vix}/{len(macro_rows)}')

        # winner-share by total trading-day count
        winner_counts: dict[str, int] = {}
        for r in regimes_all:
            winner_counts[r['winner']] = winner_counts.get(r['winner'], 0) + r['length_trading_days']
        total_td = sum(winner_counts.values())
        print('  Time-weighted winner share:')
        for w, td in sorted(winner_counts.items(), key=lambda x: -x[1]):
            print(f'    {w:18s}  {td:>5d} TD ({100*td/max(total_td,1):.1f}%)')

        all_results[str(L)] = {
            'lookback_td': L,
            'n_regimes_all': len(regimes_all),
            'n_regimes_ge21': len(regimes_21),
            'length_stats_td': {
                'min': int(lengths.min()) if len(lengths) else None,
                'median': float(np.median(lengths)) if len(lengths) else None,
                'mean': float(lengths.mean()) if len(lengths) else None,
                'max': int(lengths.max()) if len(lengths) else None,
            },
            'regimes': [
                {'winner': r['winner'], 'start': str(r['start'].date()),
                 'end': str(r['end'].date()), 'length_td': r['length_trading_days'],
                 'mean_margin': r['mean_margin']}
                for r in regimes_all
            ],
            'transitions': transitions,
            'macro_crosstab_ge21': macro_rows,
            'winner_time_share_td': winner_counts,
        }

    out = OUTPUT / 'regimes-since-2005.json'
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
