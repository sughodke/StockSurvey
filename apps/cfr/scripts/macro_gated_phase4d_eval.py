"""Last-swing experiment — CFR Phase 4d + window-level VIX meta-gate.

Hypothesis: macro v1b's window-level VIX-above-1y-rolling-median gate
composes with Phase 4d. Specifically: w2 (2016-19, calm bull) is the
window where CFR loses 0.5 Sharpe to EW; if the gate correctly suspends
the bot in calm regimes, this single substitution lifts mean alpha
above the +0.10 paper-trade threshold.

Design:
  - For each Phase 4d val window's val_start date, compute VIX at
    val_start vs 1y rolling median *as of val_start* (info-causal).
  - gate_open = (VIX_t > median(VIX_{t-252:t})) → use CFR Sharpe
  - gate_closed = otherwise → defer to multi-asset EW Sharpe
  - Apply realistic friction (same model as cfr-vs-dca-realistic):
      * CFR-deployed window: 50 bps/yr drag
      * EW-deployed window: 5 bps/yr drag
  - DCA baseline: EW always, 5 bps/yr drag every window

Pre-registered cuts:
  PASS:   net alpha vs DCA ≥ +0.10 AND positive in ≥ 4/5 windows
  MARGINAL: alpha in [0, +0.10] OR positive in 3/5 windows
  FAIL:   alpha ≤ 0 OR positive in ≤ 2/5 windows  (bot is dead)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
PHASE4D_JSON = REPO / 'Output' / 'cfr-phase4d.json'
MACRO_PKL = REPO / 'Output' / 'cfr_phase3_macro.pkl'
PHASE4D_CLOSE_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
OUT_JSON = REPO / 'Output' / 'cfr-phase4d-vix-gated.json'

CFR_DRAG_BPS_YR = 50.0
EW_DRAG_BPS_YR = 5.0
VIX_LOOKBACK_DAYS = 252


def sharpe_drag(bps_per_yr: float, vol_annual: float) -> float:
    return (bps_per_yr / 10_000.0) / vol_annual


def main() -> None:
    with open(PHASE4D_JSON) as f:
        res = json.load(f)
    with open(MACRO_PKL, 'rb') as f:
        macro: pd.DataFrame = pickle.load(f)
    with open(PHASE4D_CLOSE_PKL, 'rb') as f:
        close: pd.DataFrame = pickle.load(f)

    vix = macro['vix'].dropna()

    # Multi-asset EW daily for window-vol estimation
    ret = close.pct_change(fill_method=None)
    n_active = ret.notna().sum(axis=1)
    ew_daily_full = (ret.fillna(0).sum(axis=1)
                     / n_active.where(n_active > 0, 1)).fillna(0)
    ew_daily_full = ew_daily_full[n_active >= 5]

    print('=== Window-level VIX gate evaluation ===')
    print(f'{"win":>3} {"val_start":>11} {"VIX@start":>9} {"1y med":>7} '
          f'{"gate":>6} {"deployed":>8} {"raw Sh":>7} {"net Sh":>7} '
          f'{"DCA Sh":>7} {"alpha":>7}')
    print('-' * 95)

    rows = []
    for w in res['per_window']:
        val_start = pd.Timestamp(w['val_start'])
        val_end = pd.Timestamp(w['val_end'])

        # Info-causal: VIX up to (and including) val_start - 1 trading day
        vix_window = vix.loc[:val_start - pd.Timedelta(days=1)].tail(VIX_LOOKBACK_DAYS)
        if len(vix_window) < VIX_LOOKBACK_DAYS // 2:
            print(f'  w{w["window_idx"]}: insufficient VIX history before '
                  f'{val_start.date()}; skipping')
            continue
        # VIX at val_start: use the most recent VIX bar at-or-before val_start
        vix_at_start = float(vix.loc[:val_start].iloc[-1])
        vix_median = float(vix_window.median())
        gate_open = vix_at_start > vix_median

        # Window vol for friction conversion
        ew_win = ew_daily_full.loc[val_start:val_end]
        win_vol = float(ew_win.std(ddof=0) * np.sqrt(252))

        if gate_open:
            deployed = 'CFR'
            raw_sh = w['cfr_sharpe']
            net_sh = raw_sh - sharpe_drag(CFR_DRAG_BPS_YR, win_vol)
        else:
            deployed = 'EW'
            raw_sh = w['passive_ew_sharpe']
            net_sh = raw_sh - sharpe_drag(EW_DRAG_BPS_YR, win_vol)

        # DCA baseline = EW with EW-friction always
        dca_sh = w['passive_ew_sharpe'] - sharpe_drag(EW_DRAG_BPS_YR, win_vol)
        alpha = net_sh - dca_sh

        rows.append({
            'window_idx': w['window_idx'],
            'val_start': str(val_start.date()),
            'val_end': str(val_end.date()),
            'vix_at_start': vix_at_start,
            'vix_1y_median': vix_median,
            'gate_open': bool(gate_open),
            'deployed': deployed,
            'window_vol': win_vol,
            'raw_sharpe': float(raw_sh),
            'net_sharpe': float(net_sh),
            'dca_net_sharpe': float(dca_sh),
            'alpha_vs_dca': float(alpha),
            'cfr_raw_sharpe_for_reference': float(w['cfr_sharpe']),
            'ew_raw_sharpe_for_reference': float(w['passive_ew_sharpe']),
        })

        print(f'{w["window_idx"]:>3d} {str(val_start.date()):>11} '
              f'{vix_at_start:>9.2f} {vix_median:>7.2f} '
              f'{("OPEN" if gate_open else "CLOSED"):>6} {deployed:>8} '
              f'{raw_sh:>+7.3f} {net_sh:>+7.3f} '
              f'{dca_sh:>+7.3f} {alpha:>+7.3f}')

    print('-' * 95)
    arr_alpha = np.array([r['alpha_vs_dca'] for r in rows])
    arr_net = np.array([r['net_sharpe'] for r in rows])
    arr_dca = np.array([r['dca_net_sharpe'] for r in rows])
    n_pos = int((arr_alpha > 0).sum())
    n = len(arr_alpha)

    mean_alpha = float(arr_alpha.mean())
    print(f'\nmean net Sharpe (gated CFR) : {arr_net.mean():+.3f}')
    print(f'mean net Sharpe (DCA only)  : {arr_dca.mean():+.3f}')
    print(f'mean alpha vs DCA           : {mean_alpha:+.3f}')
    print(f'positive alpha windows      : {n_pos}/{n}')
    print(f'gate fired open in windows  : '
          f'{[r["window_idx"] for r in rows if r["gate_open"]]}')
    print(f'gate held closed in windows : '
          f'{[r["window_idx"] for r in rows if not r["gate_open"]]}')

    # Verdict
    print()
    if mean_alpha >= 0.10 and n_pos >= 4:
        verdict = (f'PASS — alpha {mean_alpha:+.3f} ≥ +0.10 AND '
                   f'positive in {n_pos}/{n} ≥ 4. Hybrid gate composes '
                   f'with Phase 4d; rebuild apps/dca as gated CFR-OR-EW '
                   f'and paper trade for 1 quarter.')
    elif mean_alpha <= 0 or n_pos <= 2:
        verdict = (f'FAIL (confirmed-null) — alpha {mean_alpha:+.3f} ≤ 0 '
                   f'OR positive in only {n_pos}/{n} ≤ 2. **Bot is fully dead.** '
                   f'DCA stays as canonical live; no further pivots.')
    else:
        verdict = (f'MARGINAL — alpha {mean_alpha:+.3f} in [0, +0.10) OR '
                   f'positive in {n_pos}/{n} = 3. Effect exists but does '
                   f'not clear paper-trade threshold; archive and stop.')
    print(f'VERDICT: {verdict}')

    payload = {
        'config': {
            'experiment': 'cfr-phase4d-vix-window-gated',
            'cfr_drag_bps_yr': CFR_DRAG_BPS_YR,
            'ew_drag_bps_yr': EW_DRAG_BPS_YR,
            'vix_lookback_days': VIX_LOOKBACK_DAYS,
        },
        'summary': {
            'n_windows': n,
            'mean_net_sharpe_gated': float(arr_net.mean()),
            'mean_net_sharpe_dca': float(arr_dca.mean()),
            'mean_alpha_vs_dca': mean_alpha,
            'positive_alpha_windows': n_pos,
            'verdict': verdict,
        },
        'per_window': rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f'\nwrote {OUT_JSON.relative_to(REPO)}')


if __name__ == '__main__':
    main()
