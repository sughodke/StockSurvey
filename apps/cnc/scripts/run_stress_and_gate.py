"""Stress + gate follow-up driver for the CNC arc.

Runs two orthogonal sensitivity sweeps over the existing 36-cell grid:

  (A) Basis-tracking-error stress
      basis_error_bps_per_day in {0, 5, 10, 20}
      → 36 cells × 4 drag levels = 144 runs
      → Output/cnc-stress-walkforward.npz

  (B) Funding-regime gate
      funding_threshold_bps_per_day in {None, 0.5, 1.0, 2.0, 5.0}
      → pre-reg cell only (K=5, rebal=1d, sign='positive', trail=30d)
      → per-year Sharpe reported for each threshold
      → Output/cnc-gate-walkforward.npz

Pre-locked bars (re-stated for the NPZ):

  Drag verdict (on pre-reg cell):
    deployment-robust  : net Sharpe >= +1.0 at 10 bps/d basis drag
    friction-fragile   : net Sharpe <  +1.0 at  5 bps/d basis drag
    boundary           : net Sharpe >= +1.0 at  5 bps/d, < +1.0 at 10 bps/d
                         → document break-even bps/day

  Gate verdict:
    confirmed if 2026YTD Sharpe lifts >= 0 AND
                 2024-2025 mean Sharpe loses <= 1.0
    else falsified.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cnc.backtest import block_sharpe, max_drawdown, pos_quarter_fraction, run_carry
from cnc.data import build_panels


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'

GRID = list(itertools.product(
    [3, 5, 10],          # top_k
    [1, 3, 7],           # rebal_days
    ['positive', 'both'],  # sign
    [7, 30],             # trailing_window
))

PRE_REG = dict(top_k=5, rebal_days=1, sign='positive', trailing_window=30)

DRAG_LEVELS = [0.0, 5.0, 10.0, 20.0]                  # bps/day
GATE_THRESHOLDS = [None, 0.5, 1.0, 2.0, 5.0]          # bps/day


def fold_slice(returns: pd.Series, year: int) -> pd.Series:
    return returns[(returns.index.year == year)]


def stress_arm(funding_daily: pd.DataFrame) -> dict:
    """Sweep 36 cells × 4 drag levels = 144 runs."""
    rows = []
    cell_streams: dict[str, np.ndarray] = {}
    pre_reg_per_drag = {}

    for drag in DRAG_LEVELS:
        for i, (k, rd, sg, tw) in enumerate(GRID):
            r = run_carry(
                funding_daily,
                top_k=k, rebal_days=rd, sign=sg, trailing_window=tw,
                rebal_friction_bps_per_leg=15.0,
                basis_error_bps_per_day=drag,
            )
            sr = block_sharpe(r.daily_return, periods_per_year=365)
            mdd = max_drawdown(r.daily_return)
            pq = pos_quarter_fraction(r.daily_return)
            rows.append(dict(
                cell=i, drag_bps=drag,
                top_k=k, rebal_days=rd, sign=sg, trailing_window=tw,
                sharpe_ann=sr, max_dd=mdd, pos_quarter=pq,
                total_return=float(r.daily_return.sum()),
                gross_sharpe_ann=block_sharpe(r.gross_return, periods_per_year=365),
                friction_total=float(r.friction_cost.sum()),
            ))
            tag = f'drag{int(drag):02d}_cell{i:02d}'
            cell_streams[f'{tag}_returns'] = r.daily_return.values

            is_pre_reg = (k == PRE_REG['top_k'] and rd == PRE_REG['rebal_days']
                          and sg == PRE_REG['sign'] and tw == PRE_REG['trailing_window'])
            if is_pre_reg:
                # Per-fold for pre-reg.
                fold_results = {}
                for y in sorted(set(r.daily_return.index.year)):
                    s = fold_slice(r.daily_return, y)
                    if len(s) < 30:
                        continue
                    fold_results[y] = dict(
                        n_days=len(s),
                        sharpe_ann=block_sharpe(s, periods_per_year=365),
                        total_return=float(s.sum()),
                        max_dd=max_drawdown(s),
                    )
                pre_reg_per_drag[drag] = dict(
                    drag_bps=drag,
                    sharpe_ann=sr, max_dd=mdd, pos_quarter=pq,
                    total_return=float(r.daily_return.sum()),
                    fold_results=fold_results,
                )
            print(f'  drag={drag:>4.1f} cell {i:>2d}: k={k} rd={rd} sg={sg:<8} tw={tw:>2} '
                  f'net_Sh={sr:+.3f} pos_q={pq:.2f} mdd={mdd*100:+.2f}%')

    # Drag-curve verdict on pre-reg cell.
    drag_curve = [(d, pre_reg_per_drag[d]['sharpe_ann']) for d in DRAG_LEVELS]
    # Break-even drag (linear interp between bracketing drag levels) where Sh=+1.0.
    break_even_bps = None
    for j in range(len(drag_curve) - 1):
        d0, s0 = drag_curve[j]
        d1, s1 = drag_curve[j + 1]
        if (s0 >= 1.0) and (s1 < 1.0):
            # linear interp
            if s0 != s1:
                break_even_bps = d0 + (d1 - d0) * (s0 - 1.0) / (s0 - s1)
            else:
                break_even_bps = d0
            break
    if break_even_bps is None:
        # If still above 1.0 at all drag levels:
        if drag_curve[-1][1] >= 1.0:
            break_even_bps = float('inf')
        else:
            # Already < 1.0 at drag=0 — bizarre
            break_even_bps = 0.0

    # Verdict per pre-locked design.
    sh_5 = pre_reg_per_drag[5.0]['sharpe_ann']
    sh_10 = pre_reg_per_drag[10.0]['sharpe_ann']
    if sh_10 >= 1.0:
        drag_verdict = 'deployment-robust'
    elif sh_5 < 1.0:
        drag_verdict = 'friction-fragile'
    else:
        drag_verdict = 'boundary'

    return dict(
        rows=rows,
        pre_reg_per_drag=pre_reg_per_drag,
        drag_curve=drag_curve,
        break_even_bps=break_even_bps,
        drag_verdict=drag_verdict,
        cell_streams=cell_streams,
    )


def gate_arm(funding_daily: pd.DataFrame) -> dict:
    """Pre-reg cell only, swept over funding-threshold gate levels."""
    rows = []
    cell_streams: dict[str, np.ndarray] = {}
    for thr in GATE_THRESHOLDS:
        r = run_carry(
            funding_daily,
            **PRE_REG,
            rebal_friction_bps_per_leg=15.0,
            basis_error_bps_per_day=0.0,
            funding_threshold_bps_per_day=thr,
        )
        sr = block_sharpe(r.daily_return, periods_per_year=365)
        mdd = max_drawdown(r.daily_return)
        pq = pos_quarter_fraction(r.daily_return)
        fold_results = {}
        for y in sorted(set(r.daily_return.index.year)):
            s = fold_slice(r.daily_return, y)
            if len(s) < 30:
                continue
            fold_results[y] = dict(
                n_days=len(s),
                sharpe_ann=block_sharpe(s, periods_per_year=365),
                total_return=float(s.sum()),
                max_dd=max_drawdown(s),
            )
        # Active-fraction: how often the gate let us trade.
        active_frac = float((r.weights.abs().sum(axis=1) > 0).mean())
        rows.append(dict(
            threshold_bps=thr,
            sharpe_ann=sr, max_dd=mdd, pos_quarter=pq,
            total_return=float(r.daily_return.sum()),
            fold_results=fold_results,
            active_fraction=active_frac,
        ))
        tag = 'thrNONE' if thr is None else f'thr{int(thr*10):03d}'  # *10 to keep integer
        cell_streams[f'{tag}_returns'] = r.daily_return.values
        print(f'  thr={str(thr):>6}: net_Sh={sr:+.3f} pos_q={pq:.2f} mdd={mdd*100:+.2f}% '
              f'active={active_frac:.2f}')
        for y, fr in fold_results.items():
            print(f'      {y}: Sh={fr["sharpe_ann"]:+.3f}  tot={fr["total_return"]*100:+.2f}%  n={fr["n_days"]}')

    # Pre-locked verdict: confirmed if 2026 Sh >= 0 AND 2024-2025 mean Sh loses <= 1.0.
    # Build a per-threshold judgement.
    baseline = next(r for r in rows if r['threshold_bps'] is None)
    base_2024 = baseline['fold_results'].get(2024, {}).get('sharpe_ann', None)
    base_2025 = baseline['fold_results'].get(2025, {}).get('sharpe_ann', None)
    base_24_25_mean = (
        np.nanmean([v for v in (base_2024, base_2025) if v is not None])
        if base_2024 is not None or base_2025 is not None else None
    )
    judgements = []
    for r in rows:
        if r['threshold_bps'] is None:
            continue
        f24 = r['fold_results'].get(2024, {}).get('sharpe_ann', None)
        f25 = r['fold_results'].get(2025, {}).get('sharpe_ann', None)
        f26 = r['fold_results'].get(2026, {}).get('sharpe_ann', None)
        gated_24_25_mean = (
            np.nanmean([v for v in (f24, f25) if v is not None])
            if f24 is not None or f25 is not None else None
        )
        loss = (
            (base_24_25_mean - gated_24_25_mean)
            if (base_24_25_mean is not None and gated_24_25_mean is not None)
            else None
        )
        confirmed = (
            f26 is not None and f26 >= 0.0 and
            loss is not None and loss <= 1.0
        )
        judgements.append(dict(
            threshold_bps=r['threshold_bps'],
            sh_2024=f24, sh_2025=f25, sh_2026=f26,
            gated_24_25_mean=gated_24_25_mean,
            sh_24_25_loss_vs_baseline=loss,
            verdict='confirmed' if confirmed else 'falsified',
        ))

    # Optimal threshold = highest 2026 Sharpe among confirmed (if any), else None.
    confirmed_js = [j for j in judgements if j['verdict'] == 'confirmed']
    if confirmed_js:
        optimal = max(confirmed_js, key=lambda j: (j['sh_2026'] if j['sh_2026'] is not None else -1e9))
    else:
        optimal = None

    return dict(
        rows=rows,
        baseline_24_25_mean=base_24_25_mean,
        judgements=judgements,
        optimal=optimal,
        cell_streams=cell_streams,
    )


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    t0 = time.time()
    print('[cnc-stress+gate] building panels')
    panels = build_panels(
        start_date='2024-01-01',
        end_date=None,
        top_universe=20,
        min_history_days=180,
    )
    print(f'[cnc-stress+gate] panels: {len(panels.funding_daily)} days × '
          f'{len(panels.coins)} coins '
          f'({panels.start_date.date()} → {panels.end_date.date()})')

    # --- Stress arm ---
    print()
    print('=' * 70)
    print('(A) Stress arm: 36 cells × 4 drag levels')
    print('=' * 70)
    stress = stress_arm(panels.funding_daily)

    npz_stress = OUTPUT / 'cnc-stress-walkforward.npz'
    np.savez(
        npz_stress,
        drag_levels=np.array(DRAG_LEVELS),
        break_even_bps=np.array([stress['break_even_bps']]),
        drag_verdict=np.array([stress['drag_verdict']]),
        **stress['cell_streams'],
    )
    print(f'[cnc-stress+gate] wrote NPZ → {npz_stress}')

    json_stress = OUTPUT / 'cnc-stress-walkforward.json'
    json_stress.write_text(json.dumps(dict(
        drag_levels=DRAG_LEVELS,
        rows=stress['rows'],
        pre_reg_per_drag=stress['pre_reg_per_drag'],
        drag_curve=stress['drag_curve'],
        break_even_bps=stress['break_even_bps'],
        drag_verdict=stress['drag_verdict'],
    ), indent=2, default=str))
    print(f'[cnc-stress+gate] wrote JSON → {json_stress}')

    # --- Gate arm ---
    print()
    print('=' * 70)
    print('(B) Gate arm: pre-reg cell × {None, 0.5, 1.0, 2.0, 5.0} bps/d threshold')
    print('=' * 70)
    gate = gate_arm(panels.funding_daily)

    npz_gate = OUTPUT / 'cnc-gate-walkforward.npz'
    np.savez(
        npz_gate,
        thresholds=np.array([-1.0 if t is None else t for t in GATE_THRESHOLDS]),
        **gate['cell_streams'],
    )
    print(f'[cnc-stress+gate] wrote NPZ → {npz_gate}')

    json_gate = OUTPUT / 'cnc-gate-walkforward.json'
    json_gate.write_text(json.dumps(dict(
        thresholds=GATE_THRESHOLDS,
        rows=gate['rows'],
        baseline_24_25_mean=gate['baseline_24_25_mean'],
        judgements=gate['judgements'],
        optimal=gate['optimal'],
    ), indent=2, default=str))
    print(f'[cnc-stress+gate] wrote JSON → {json_gate}')

    # --- Final summary ---
    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print('Drag-curve (pre-reg cell):')
    for d, s in stress['drag_curve']:
        print(f'  drag={d:>4.1f} bps/d  →  Sh={s:+.3f}')
    print(f'Break-even drag (Sh=+1.0): {stress["break_even_bps"]} bps/d')
    print(f'Drag verdict: {stress["drag_verdict"]}')
    print()
    print('Gate (pre-reg cell):')
    print(f'  baseline 2024-2025 mean Sh: {gate["baseline_24_25_mean"]:+.3f}')
    for j in gate['judgements']:
        f26 = j["sh_2026"]
        loss = j["sh_24_25_loss_vs_baseline"]
        f26s = f'{f26:+.3f}' if f26 is not None else 'NA'
        losss = f'{loss:+.3f}' if loss is not None else 'NA'
        print(f'  thr={j["threshold_bps"]:>4} bps/d  '
              f'Sh2026={f26s}  '
              f'Δ24-25={losss}  '
              f'verdict={j["verdict"]}')
    print(f'Optimal threshold (confirmed): {gate["optimal"]}')
    print()
    print(f'Wall: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
