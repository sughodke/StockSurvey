"""Post-2020 cross-arc ranking + DCA+vol_v3 ensemble drilldown.

Reuses cached daily-return streams; runs NO new training. See
findings/post-2020-arc-ranking.md for interpretation.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'
sys.path.insert(0, str(Path(__file__).resolve().parent))

from count_regimes_since_2005 import load_dca_daily, build_master  # noqa: E402
from ss_portfolio.sharpe_diff import sharpe_difference_ci  # noqa: E402


PPY = 252.0
CUTOFF = pd.Timestamp('2020-01-01')


def ann_sharpe(r: np.ndarray, ppy: float = PPY) -> float:
    r = r[~np.isnan(r)]
    if r.size < 5:
        return float('nan')
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0


def ann_cagr(r: np.ndarray, ppy: float = PPY) -> float:
    r = r[~np.isnan(r)]
    if r.size < 5:
        return float('nan')
    eq = np.cumprod(1.0 + r)
    n_yrs = r.size / ppy
    if eq[-1] <= 0 or n_yrs <= 0:
        return float('nan')
    return float(eq[-1] ** (1.0 / n_yrs) - 1.0)


def max_dd(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float('nan')
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def pos_quarter_fraction(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return float('nan')
    q = (1.0 + s).resample('QE').prod() - 1.0
    return float((q > 0).mean()) if len(q) else float('nan')


def load_audit_arc(name: str) -> pd.Series:
    p = OUTPUT / f'{name}-universe-agnostic-walkforward.npz'
    d = np.load(p, allow_pickle=True)
    dates = pd.to_datetime(d['oos_dates'])
    ret = np.asarray(d['oos_block_returns'], dtype=np.float64)
    return pd.Series(ret, index=dates).dropna()


def build_extended_panel() -> pd.DataFrame:
    dca = load_dca_daily()
    df = build_master(dca)
    # Add audit-cohort arcs
    for tag in ['rsi', 'scalogram', 'regime-cwt', 'regime-velocity', 'lie-hrp']:
        col = tag.replace('-', '_')
        s = load_audit_arc(tag)
        df[col] = s.reindex(df.index)
    return df


def stats_for(s: pd.Series) -> dict:
    r = s.dropna().to_numpy()
    return dict(
        n_obs=int(r.size),
        first=str(s.dropna().index[0].date()) if r.size else None,
        last=str(s.dropna().index[-1].date()) if r.size else None,
        ann_sharpe=ann_sharpe(r),
        ann_cagr=ann_cagr(r),
        max_dd=max_dd(r),
        pos_quarter_frac=pos_quarter_fraction(s),
    )


def main() -> None:
    print('Building extended post-2020 panel...')
    df = build_extended_panel()
    df_2020 = df.loc[CUTOFF:]
    print(f'Panel shape post-2020: {df_2020.shape}')

    # --- Standalone arc stats ---
    rows = []
    for col in df_2020.columns:
        st = stats_for(df_2020[col])
        st['arc'] = col
        rows.append(st)

    # --- DCA+vol_v3 ensemble (vega=2.0, c_options_bps=200 already baked in vol_v3 c200) ---
    vega = 2.0
    dca_s = df_2020['dca']
    vol_s = df_2020['vol_v3']
    # Where vol_v3 is NaN, sleeve contributes 0
    ens_full = dca_s.add(vega * vol_s.fillna(0.0), fill_value=0.0)
    ens_full = ens_full.where(~dca_s.isna())
    st = stats_for(ens_full)
    st['arc'] = 'dca_plus_vol_v3_vega2'
    rows.append(st)

    # Sub-period stratification
    subperiods = {
        'pre_vol': (pd.Timestamp('2020-01-01'), pd.Timestamp('2023-07-31')),
        'vol_active': (pd.Timestamp('2023-08-01'), df_2020.index[-1]),
        'full_2020+': (df_2020.index[0], df_2020.index[-1]),
    }

    sub_stats = {}
    for name, (lo, hi) in subperiods.items():
        sub_stats[name] = {}
        slc = df_2020.loc[lo:hi]
        # ensemble for this slice
        dca_sl = slc['dca']
        vol_sl = slc['vol_v3']
        ens_sl = dca_sl.add(vega * vol_sl.fillna(0.0), fill_value=0.0)
        ens_sl = ens_sl.where(~dca_sl.isna())
        for col in slc.columns:
            sub_stats[name][col] = stats_for(slc[col])
        sub_stats[name]['dca_plus_vol_v3_vega2'] = stats_for(ens_sl)

    # --- LW Studentized ΔSharpe vs DCA on full post-2020 window ---
    print('\nComputing Ledoit-Wolf ΔSharpe CIs vs DCA (full post-2020)...')
    dca_ref = df_2020['dca'].to_numpy()
    delta_rows = []
    for col in [c for c in df_2020.columns if c != 'dca'] + ['dca_plus_vol_v3_vega2']:
        if col == 'dca_plus_vol_v3_vega2':
            comp = ens_full
        else:
            comp = df_2020[col]
        # align on inner-join (both non-NaN)
        joint = pd.concat([df_2020['dca'], comp], axis=1).dropna()
        if len(joint) < 30:
            delta_rows.append(dict(arc=col, n_obs=len(joint), note='insufficient overlap'))
            continue
        a = joint.iloc[:, 1].to_numpy()  # candidate
        b = joint.iloc[:, 0].to_numpy()  # DCA
        ci = sharpe_difference_ci(a, b, n_bootstraps=2000, seed=42)
        delta_rows.append(dict(
            arc=col,
            n_obs=ci.n_obs,
            sr_arc_periodic=ci.sr_a,
            sr_dca_periodic=ci.sr_b,
            sr_arc_ann=ci.sr_a * math.sqrt(PPY),
            sr_dca_ann=ci.sr_b * math.sqrt(PPY),
            delta_sr_periodic=ci.delta_sr,
            delta_sr_ann=ci.delta_sr * math.sqrt(PPY),
            ci_lo_ann=ci.ci_lo * math.sqrt(PPY),
            ci_hi_ann=ci.ci_hi * math.sqrt(PPY),
            includes_zero=ci.includes_zero,
            block_length=ci.block_length,
        ))

    # --- Ensemble vs DCA on the vol-active subperiod ---
    lo, hi = subperiods['vol_active']
    slc = df_2020.loc[lo:hi]
    dca_sl = slc['dca'].dropna()
    vol_sl = slc['vol_v3']
    ens_sl = dca_sl.add(vega * vol_sl.reindex(dca_sl.index).fillna(0.0), fill_value=0.0)
    a = ens_sl.to_numpy()
    b = dca_sl.to_numpy()
    ci_vol_active = sharpe_difference_ci(a, b, n_bootstraps=2000, seed=42)
    ensemble_vs_dca_vol_active = dict(
        n_obs=ci_vol_active.n_obs,
        delta_sr_ann=ci_vol_active.delta_sr * math.sqrt(PPY),
        ci_lo_ann=ci_vol_active.ci_lo * math.sqrt(PPY),
        ci_hi_ann=ci_vol_active.ci_hi * math.sqrt(PPY),
        includes_zero=ci_vol_active.includes_zero,
    )

    # --- Print ranking ---
    rows_sorted = sorted(rows, key=lambda r: -(r['ann_sharpe'] if not math.isnan(r['ann_sharpe']) else -1e9))
    print('\n=== POST-2020 ARC RANKING (by ann Sharpe) ===')
    hdr = f"{'arc':28s} {'n':>5s} {'first':>12s} {'last':>12s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'pos_q':>6s}"
    print(hdr)
    for r in rows_sorted:
        print(f"{r['arc']:28s} {r['n_obs']:>5d} {str(r['first']):>12s} {str(r['last']):>12s} "
              f"{r['ann_sharpe']:>+8.3f} {r['ann_cagr']:>+8.3f} {r['max_dd']:>+8.3f} {r['pos_quarter_frac']:>6.2f}")

    print('\n=== LW ΔSharpe vs DCA (ann; full post-2020) ===')
    print(f"{'arc':28s} {'n':>5s} {'ΔSR_ann':>9s} {'CI_lo':>9s} {'CI_hi':>9s} {'sig':>4s}")
    for r in delta_rows:
        if 'delta_sr_ann' not in r:
            note = r.get('note', '')
            print(f"{r['arc']:28s} {r['n_obs']:>5d} {'—':>9s} {'—':>9s} {'—':>9s} {'-':>4s} ({note})")
            continue
        sig = '***' if not r['includes_zero'] and r['delta_sr_ann'] > 0 else (
              '---' if not r['includes_zero'] and r['delta_sr_ann'] < 0 else 'ns')
        print(f"{r['arc']:28s} {r['n_obs']:>5d} {r['delta_sr_ann']:>+9.3f} "
              f"{r['ci_lo_ann']:>+9.3f} {r['ci_hi_ann']:>+9.3f} {sig:>4s}")

    print('\n=== Ensemble (DCA + 2*vol_v3) vs DCA on vol_active subperiod ===')
    print(f"  n_obs={ensemble_vs_dca_vol_active['n_obs']}  ΔSR_ann={ensemble_vs_dca_vol_active['delta_sr_ann']:+.3f}  "
          f"CI=[{ensemble_vs_dca_vol_active['ci_lo_ann']:+.3f}, {ensemble_vs_dca_vol_active['ci_hi_ann']:+.3f}]  "
          f"{'EXCLUDES 0' if not ensemble_vs_dca_vol_active['includes_zero'] else 'includes 0'}")

    out = dict(
        cutoff=str(CUTOFF.date()),
        panel_shape=list(df_2020.shape),
        ranking=rows_sorted,
        delta_vs_dca_full_post_2020=delta_rows,
        ensemble_vs_dca_vol_active=ensemble_vs_dca_vol_active,
        subperiod_stats=sub_stats,
        config=dict(vega_scale=vega, c_options_bps=200, ppy=PPY),
    )
    out_path = OUTPUT / 'post-2020-arc-ranking.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\n→ wrote {out_path}')


if __name__ == '__main__':
    main()
