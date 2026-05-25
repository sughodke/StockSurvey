"""Meta-allocator falsification — drop vol_v3, re-run the same walk-forward.

Tests whether the +0.367 ΔSR_ann advantage of B3 inverse-arc-vol over B2 1/N
documented in `findings/meta-allocator-regime-forecasting.md` is structural
to inverse-vol weighting on a heterogeneous-vol panel, or is being carried
by the vol_v3 specialist arc (recent: won 419 TD since 2024-04, within-window
Sharpe +12.77).

Re-uses every piece of machinery from `meta_allocator_run.py`:
- same FOLDS, REBAL_TD, COMMISSION_BPS, L=252
- same allocators (B1/B2/B3/C1/C2/C3/C4/C5)
- same DSR n_trials=8
- same Ledoit-Wolf sharpe_difference_ci bootstrap (2000 resamples, seed=42)

Only difference: ARC_COLS = the 6-arc panel MINUS vol_v3 = 5 arcs.

Output: Output/meta-allocator-no-vol-results.json +
        Output/meta-allocator-no-vol-daily-streams.npz
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / 'apps' / 'docs' / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))
OUTPUT = REPO_ROOT / 'Output'

# Pull in the canonical driver and override ARC_COLS before any function
# uses the module-level constant. The functions in meta_allocator_run all
# take arc_df as a parameter, so the ARC_COLS constant only matters for
# load_arc_panel's column-ordering and for the JSON output metadata.
import meta_allocator_run as M

# Pre-registered falsification: drop vol_v3 from the panel.
ARC_COLS_NO_VOL = ['dca', 'gate', 'pairs', 'relational', 'dca_winner_4etf']
M.ARC_COLS = ARC_COLS_NO_VOL


def load_arc_panel_no_vol() -> pd.DataFrame:
    """Load the 6-arc master panel via the canonical path, then drop vol_v3
    columns. We construct the full panel first (so any cross-arc alignment
    in build_master happens identically) and then subset, which is the
    behaviorally cleanest way to keep the rest of the walk-forward
    bitwise-equivalent to the parent finding modulo the dropped column."""
    from count_regimes_since_2005 import build_master, load_dca_daily
    dca = load_dca_daily()
    df = build_master(dca)
    # Subset to the 5-arc no-vol_v3 panel
    return df[ARC_COLS_NO_VOL]


def main() -> None:
    print('=== Meta-allocator NO-vol_v3 falsification walk-forward ===')
    print(f'  ARC_COLS = {ARC_COLS_NO_VOL}')
    arc_df = load_arc_panel_no_vol()
    print(f'  arc panel: {arc_df.shape[0]} rows × {arc_df.shape[1]} arcs')
    print(f'  span: {arc_df.index[0].date()} → {arc_df.index[-1].date()}')
    for c in arc_df.columns:
        s = arc_df[c].dropna()
        print(f'    {c:20s} n={s.size:>5d} '
              f'{s.index[0].date() if s.size else "—"} → '
              f'{s.index[-1].date() if s.size else "—"}')

    macro_df = M.load_macro_panel(target_index=arc_df.index)
    print(f'  macro: {macro_df.shape[1]} features, '
          f'{macro_df.dropna().shape[0]} fully-populated rows')

    print('\n--- Walk-forward with canonical L=252 ---')
    res = M.run_walkforward(arc_df, macro_df, L=252)
    n_obs = len(res['dates'])
    print(f'  pooled OOS obs: {n_obs} days '
          f'({res["dates"][0].date()} → {res["dates"][-1].date()})')

    N_TRIALS = 8
    benchmarks = {
        'B1_persist_L252': res['daily']['B1_persist_L252'],
        'B2_equal_weight': res['daily']['B2_equal_weight'],
        'B3_inv_vol': res['daily']['B3_inv_vol'],
    }

    print(f'\n--- per-candidate eval (n_trials={N_TRIALS} for DSR) ---')
    rows = []
    for name, daily in res['daily'].items():
        rec = M.eval_candidate(daily, res['dates'], name, benchmarks,
                               n_trials=N_TRIALS)
        rec['mean_daily_ret'] = float(np.mean(daily)) if daily.size else float('nan')
        rec['std_daily_ret'] = float(np.std(daily, ddof=1)) if daily.size > 1 else float('nan')
        for bn in ['B1_persist_L252', 'B2_equal_weight', 'B3_inv_vol']:
            if bn == name:
                continue
            rec[f'verdict_vs_{bn}'] = M.classify_verdict(rec, bn)
        rows.append(rec)

        print(f'\n  {name}:')
        print(f'    n={rec["n_obs"]}  Sharpe_ann={rec.get("sharpe_ann", float("nan")):+.3f}  '
              f'DSR={rec.get("dsr", float("nan")):.3f}  DSR-t={rec.get("dsr_tstat", float("nan")):+.2f}')
        for bn in ['B1_persist_L252', 'B2_equal_weight', 'B3_inv_vol']:
            if bn == name:
                continue
            v = rec.get(f'vs_{bn}', {})
            if 'error' in v:
                print(f'    vs {bn}: ERROR {v["error"]}')
                continue
            print(f'    vs {bn}: ΔSR_ann {v["delta_sr_ann"]:+.3f} '
                  f'[{v["ci_lo_ann"]:+.3f}, {v["ci_hi_ann"]:+.3f}] '
                  f'incl_0={v["includes_zero"]}  '
                  f'verdict={rec.get(f"verdict_vs_{bn}")}')

    out = OUTPUT / 'meta-allocator-no-vol-results.json'
    json_safe = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                rr[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                rr[k] = v
        json_safe.append(rr)
    out.write_text(json.dumps({
        'lookback_L': 252,
        'rebal_td': M.REBAL_TD,
        'commission_bps': M.COMMISSION_BPS,
        'n_trials_dsr': N_TRIALS,
        'pooled_oos_start': str(res['dates'][0].date()) if n_obs else None,
        'pooled_oos_end': str(res['dates'][-1].date()) if n_obs else None,
        'pooled_oos_n': n_obs,
        'arc_cols': list(arc_df.columns),
        'arc_cols_excluded': ['vol_v3'],
        'rows': json_safe,
    }, indent=2, default=str))
    print(f'\n→ {out}')

    np.savez(OUTPUT / 'meta-allocator-no-vol-daily-streams.npz',
             dates=np.array([str(d.date()) for d in res['dates']]),
             **res['daily'])
    print(f'→ {OUTPUT / "meta-allocator-no-vol-daily-streams.npz"}')


if __name__ == '__main__':
    main()
