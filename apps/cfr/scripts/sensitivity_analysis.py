"""CFR Phase 4d sensitivity analysis — friction levels and alternative gates.

Follow-up to the audit of cfr-vs-dca-realistic.md and cfr-macro-gate-final.md.
The original closure rests on three load-bearing choices that this script
stress-tests by recomputing analytically from the on-disk per-window Sharpes:

1. **Friction sensitivity** — CFR drag and DCA drag swept independently.
   Original eval pinned CFR=50 bps/yr and DCA=5 bps/yr without sensitivity.
2. **Alternative VIX gate lookbacks** — 60d / 90d / 126d / 252d (baseline).
   Original gate was 252d only; the "memory-heavy gate" diagnosis was
   never confronted with shorter lookbacks.
3. **Cross-asset dispersion gate** — fires when avg pairwise 60d correlation
   among the 13 CFR Phase 4d assets is BELOW its trailing 252d median
   (high dispersion → CFR rotation opportunity). Uses in-universe data
   only; no external VIX.

All three use the raw per-window Sharpes from cfr-phase4d.json so no
walk-forward retraining is required.

Outputs Output/cfr-sensitivity.json with all matrices, and prints summary tables.
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
CLOSE_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
OUT_JSON = REPO / 'Output' / 'cfr-sensitivity.json'


def sharpe_drag(bps_per_yr: float, vol_annual: float) -> float:
    return (bps_per_yr / 10_000.0) / vol_annual


def load_window_vols(close: pd.DataFrame, windows: list[dict]) -> list[float]:
    """Window-vol of multi-asset EW basket for each val window, for drag conversion."""
    ret = close.pct_change(fill_method=None)
    n_active = ret.notna().sum(axis=1)
    ew_daily = (ret.fillna(0).sum(axis=1)
                / n_active.where(n_active > 0, 1)).fillna(0)
    ew_daily = ew_daily[n_active >= 5]
    vols = []
    for w in windows:
        s = pd.Timestamp(w['val_start'])
        e = pd.Timestamp(w['val_end'])
        seg = ew_daily.loc[s:e]
        vols.append(float(seg.std(ddof=0) * np.sqrt(252)))
    return vols


def rolling_avg_pairwise_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise Pearson correlation over a rolling window.

    For each date t, computes the corr matrix over [t-window+1, t] and returns
    the mean of upper-triangle entries (excludes diagonal). Uses numpy for speed.
    """
    X = returns.fillna(0).to_numpy(dtype=np.float64)
    T, N = X.shape
    out = np.full(T, np.nan)
    if N < 2:
        return pd.Series(out, index=returns.index)
    tri_i, tri_j = np.triu_indices(N, k=1)
    for t in range(window - 1, T):
        seg = X[t - window + 1: t + 1]
        # Need at least window-1 non-zero rows to give a meaningful corr
        if (seg != 0).any(axis=0).sum() < 2:
            continue
        with np.errstate(invalid='ignore'):
            c = np.corrcoef(seg, rowvar=False)
        if not np.isfinite(c).all():
            continue
        out[t] = float(c[tri_i, tri_j].mean())
    return pd.Series(out, index=returns.index)


def evaluate_gate(
    res: dict,
    window_vols: list[float],
    gate_decisions: list[bool],
    cfr_drag: float,
    ew_drag: float,
) -> dict:
    """Given a per-window gate decision, compute alpha vs always-DCA."""
    rows = []
    for w, vol, gate_open in zip(res['per_window'], window_vols, gate_decisions):
        cfr_raw = w['cfr_sharpe']
        ew_raw = w['passive_ew_sharpe']
        cfr_net = cfr_raw - sharpe_drag(cfr_drag, vol)
        ew_net = ew_raw - sharpe_drag(ew_drag, vol)
        deployed_net = cfr_net if gate_open else ew_net
        dca_net = ew_net
        alpha = deployed_net - dca_net
        rows.append({
            'window_idx': w['window_idx'],
            'val_start': w['val_start'],
            'gate_open': bool(gate_open),
            'deployed_net_sharpe': float(deployed_net),
            'dca_net_sharpe': float(dca_net),
            'alpha_vs_dca': float(alpha),
            'cfr_raw': float(cfr_raw),
            'ew_raw': float(ew_raw),
            'cfr_minus_ew_raw_alpha': float(cfr_raw - ew_raw),
        })
    alphas = np.array([r['alpha_vs_dca'] for r in rows])
    fired = [r for r in rows if r['gate_open']]
    n_fired = len(fired)
    n_pos = int((alphas > 0).sum())
    mean_alpha = float(alphas.mean())
    # Alternative denominator: only count windows where gate fired
    alpha_on_fired = float(np.mean([r['alpha_vs_dca'] for r in fired])) if fired else 0.0
    n_pos_on_fired = int(sum(1 for r in fired if r['alpha_vs_dca'] > 0))
    return {
        'rows': rows,
        'mean_alpha': mean_alpha,
        'n_pos': n_pos,
        'n': len(rows),
        'n_fired': n_fired,
        'alpha_on_fired': alpha_on_fired,
        'n_pos_on_fired': n_pos_on_fired,
    }


def section_1_friction_no_gate(res: dict, vols: list[float]) -> dict:
    """Always-deploy CFR vs always-deploy DCA at friction levels."""
    cfr_levels = [10, 15, 25, 35, 50]
    dca_levels = [2, 5, 10]
    rows = []
    print('\n=== Section 1: Always-deploy CFR vs always-deploy DCA ===')
    print(f'{"CFR drag":>10} {"DCA drag":>10} {"mean alpha":>11} '
          f'{"pos/n":>6} {"verdict":>10}')
    print('-' * 60)
    for cfr_drag in cfr_levels:
        for dca_drag in dca_levels:
            alphas = []
            for w, vol in zip(res['per_window'], vols):
                cfr_net = w['cfr_sharpe'] - sharpe_drag(cfr_drag, vol)
                dca_net = w['passive_ew_sharpe'] - sharpe_drag(dca_drag, vol)
                alphas.append(cfr_net - dca_net)
            alphas = np.array(alphas)
            mean_alpha = float(alphas.mean())
            n_pos = int((alphas > 0).sum())
            verdict = (
                'PASS' if mean_alpha >= 0.10 and n_pos >= 4
                else 'MARGINAL' if mean_alpha > 0 and n_pos >= 3
                else 'FAIL'
            )
            row = {
                'cfr_drag_bps': cfr_drag,
                'dca_drag_bps': dca_drag,
                'mean_alpha': mean_alpha,
                'n_pos': n_pos,
                'n': len(alphas),
                'verdict': verdict,
                'per_window_alphas': alphas.tolist(),
            }
            rows.append(row)
            print(f'{cfr_drag:>10d} {dca_drag:>10d} {mean_alpha:>+11.4f} '
                  f'{n_pos:>3d}/{len(alphas):<2d} {verdict:>10}')
    return {'rows': rows, 'cfr_levels': cfr_levels, 'dca_levels': dca_levels}


def section_2_alt_gates(res: dict, vols: list[float], vix: pd.Series,
                        returns_close: pd.DataFrame) -> dict:
    """Test alternative gate definitions. Friction pinned at 50/5 (worst case)."""
    cfr_drag, ew_drag = 50.0, 5.0
    val_starts = [pd.Timestamp(w['val_start']) for w in res['per_window']]

    # VIX gate variants
    vix_gates = {}
    for lookback in [60, 90, 126, 252]:
        decisions = []
        details = []
        for vs in val_starts:
            window = vix.loc[:vs - pd.Timedelta(days=1)].tail(lookback)
            vix_at = float(vix.loc[:vs].iloc[-1])
            med = float(window.median())
            decisions.append(vix_at > med)
            details.append({'val_start': str(vs.date()), 'vix_at': vix_at,
                            'median': med, 'gate_open': vix_at > med})
        result = evaluate_gate(res, vols, decisions, cfr_drag, ew_drag)
        result['details'] = details
        result['lookback_days'] = lookback
        vix_gates[f'vix_{lookback}d'] = result

    # Cross-asset dispersion gate
    # rolling 60d avg pairwise corr; gate fires when corr < 252d trailing median
    asset_ret = returns_close.pct_change(fill_method=None)
    avg_corr_60 = rolling_avg_pairwise_corr(asset_ret, window=60)
    avg_corr_30 = rolling_avg_pairwise_corr(asset_ret, window=30)

    dispersion_gates = {}
    for corr_series, label in [(avg_corr_60, 'disp_60d_vs_252d_med'),
                               (avg_corr_30, 'disp_30d_vs_252d_med')]:
        decisions = []
        details = []
        for vs in val_starts:
            hist = corr_series.loc[:vs - pd.Timedelta(days=1)].dropna()
            if len(hist) < 252:
                # Not enough history — default to closed (conservative)
                decisions.append(False)
                details.append({'val_start': str(vs.date()), 'corr_at': None,
                                'corr_median': None, 'gate_open': False,
                                'note': 'insufficient history'})
                continue
            corr_at = float(hist.iloc[-1])
            med = float(hist.tail(252).median())
            # gate fires when correlation is LOW (dispersion HIGH)
            gate_open = corr_at < med
            decisions.append(gate_open)
            details.append({'val_start': str(vs.date()), 'corr_at': corr_at,
                            'corr_median': med, 'gate_open': gate_open})
        result = evaluate_gate(res, vols, decisions, cfr_drag, ew_drag)
        result['details'] = details
        dispersion_gates[label] = result

    # Always-CFR (no gate) at the same friction, for reference
    always_cfr = evaluate_gate(res, vols, [True] * len(val_starts), cfr_drag, ew_drag)
    always_cfr['details'] = [{'val_start': str(vs.date()), 'gate_open': True}
                             for vs in val_starts]

    all_gates = {**vix_gates, **dispersion_gates, 'always_cfr': always_cfr}

    print('\n=== Section 2: Alternative gate variants (CFR 50 bps / DCA 5 bps) ===')
    print(f'{"gate":>28} {"fired":>5} {"mean alpha":>10} '
          f'{"pos/n":>5} {"alpha|fired":>11} {"pos|fired":>9}')
    print('-' * 80)
    for name, r in all_gates.items():
        print(f'{name:>28} {r["n_fired"]:>5d} {r["mean_alpha"]:>+10.4f} '
              f'{r["n_pos"]:>2d}/{r["n"]:<2d} {r["alpha_on_fired"]:>+11.4f} '
              f'{r["n_pos_on_fired"]:>3d}/{r["n_fired"]:<3d}')

    # Also: which gate fires in which window (across-the-row view)
    print('\nGate firing matrix (window → which gates fired):')
    print(f'{"win":>3} {"val_start":>11} {"CFR-EW raw α":>13}  ' +
          ' '.join(f'{n:>6}' for n in all_gates.keys()))
    for i, w in enumerate(res['per_window']):
        cfr_ew_alpha = w['cfr_sharpe'] - w['passive_ew_sharpe']
        flags = []
        for name, r in all_gates.items():
            flags.append('  ✓  ' if r['rows'][i]['gate_open'] else '  .  ')
        print(f'{i:>3d} {w["val_start"]:>11} {cfr_ew_alpha:>+13.3f}  ' +
              ' '.join(f'{f:>6}' for f in flags))

    return all_gates


def section_3_combined(res: dict, vols: list[float], vix: pd.Series,
                       returns_close: pd.DataFrame, best_gate_decisions: list[bool],
                       best_gate_name: str) -> dict:
    """Pick the best gate from Section 2; sweep friction levels."""
    cfr_levels = [10, 15, 25, 35, 50]
    dca_levels = [2, 5, 10]
    rows = []
    print(f'\n=== Section 3: Best gate ({best_gate_name}) × friction sweep ===')
    print(f'{"CFR drag":>10} {"DCA drag":>10} {"mean alpha":>11} '
          f'{"pos/n":>6} {"verdict":>10}')
    print('-' * 60)
    for cfr_drag in cfr_levels:
        for dca_drag in dca_levels:
            result = evaluate_gate(res, vols, best_gate_decisions, cfr_drag, dca_drag)
            mean_alpha = result['mean_alpha']
            n_pos = result['n_pos']
            verdict = (
                'PASS' if mean_alpha >= 0.10 and n_pos >= 4
                else 'MARGINAL' if mean_alpha > 0 and n_pos >= 3
                else 'FAIL'
            )
            rows.append({
                'cfr_drag_bps': cfr_drag,
                'dca_drag_bps': dca_drag,
                'mean_alpha': mean_alpha,
                'n_pos': n_pos,
                'n_fired': result['n_fired'],
                'alpha_on_fired': result['alpha_on_fired'],
                'verdict': verdict,
            })
            print(f'{cfr_drag:>10d} {dca_drag:>10d} {mean_alpha:>+11.4f} '
                  f'{n_pos:>3d}/{result["n"]:<2d} {verdict:>10}')
    return {'rows': rows, 'best_gate_name': best_gate_name}


def main() -> None:
    print(f'Reading {PHASE4D_JSON.relative_to(REPO)}')
    res = json.load(open(PHASE4D_JSON))
    print(f'Reading {MACRO_PKL.relative_to(REPO)}')
    macro: pd.DataFrame = pickle.load(open(MACRO_PKL, 'rb'))
    print(f'Reading {CLOSE_PKL.relative_to(REPO)}')
    close: pd.DataFrame = pickle.load(open(CLOSE_PKL, 'rb'))

    vix = macro['vix'].dropna()
    vols = load_window_vols(close, res['per_window'])

    print(f'\nN val windows: {len(res["per_window"])}')
    print(f'Window vols (annualized): {[f"{v:.3f}" for v in vols]}')

    sec1 = section_1_friction_no_gate(res, vols)
    sec2 = section_2_alt_gates(res, vols, vix, close)

    # Pick the gate with highest mean alpha as the candidate for Section 3
    # (exclude always_cfr — that's the no-gate baseline)
    gate_choices = {k: v for k, v in sec2.items() if k != 'always_cfr'}
    best_name = max(gate_choices, key=lambda k: gate_choices[k]['mean_alpha'])
    best_decisions = [r['gate_open'] for r in gate_choices[best_name]['rows']]
    print(f'\nBest gate by mean alpha: {best_name} '
          f'(alpha = {gate_choices[best_name]["mean_alpha"]:+.4f})')

    sec3 = section_3_combined(res, vols, vix, close, best_decisions, best_name)

    payload = {
        'config': {
            'experiment': 'cfr-phase4d-sensitivity-analysis',
            'rationale': ('Audit follow-up: stress-test the CFR closure '
                          'across friction and gate-design choices.'),
            'phase4d_source': str(PHASE4D_JSON.relative_to(REPO)),
            'n_windows': len(res['per_window']),
            'window_vols_annualized': vols,
        },
        'section_1_friction_no_gate': sec1,
        'section_2_alt_gates': {k: {kk: vv for kk, vv in v.items() if kk != 'rows'}
                                | {'rows': v['rows']}
                                for k, v in sec2.items()},
        'section_3_best_gate_friction_sweep': sec3,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f'\nwrote {OUT_JSON.relative_to(REPO)}')


if __name__ == '__main__':
    main()
