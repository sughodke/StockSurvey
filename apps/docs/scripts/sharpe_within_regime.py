"""For each L=252 winner-regime, compute the winning arc's annualized
Sharpe *within its own winning window* — and compare to the same arc's
full-sample Sharpe.

Purpose: the full-period Sharpe of a specialist arc (e.g. vol_v3) is
dragged down by all the windows it is NOT winning. Within its own
winning window the Sharpe should be much higher. This script makes that
explicit.

Run from repo root:
    uv run python apps/docs/scripts/sharpe_within_regime.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'apps' / 'docs' / 'scripts'))
from count_regimes_since_2005 import build_master, load_dca_daily  # noqa: E402

PPY = 252.0


def annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5 or r.std(ddof=1) < 1e-12:
        return float('nan')
    return float(r.mean() / r.std(ddof=1) * math.sqrt(PPY))


def main() -> None:
    print('Building master arc panel...')
    df = build_master(load_dca_daily())

    full = {c: annualized_sharpe(df[c]) for c in df.columns}
    print('\n--- full-sample annualized Sharpe (all available days) ---')
    for c, s in sorted(full.items(), key=lambda kv: -kv[1] if not math.isnan(kv[1]) else 0):
        print(f'  {c:18s}  {s:+.3f}')

    regs = json.loads((REPO / 'Output' / 'regimes-since-2005.json').read_text())['252']['regimes']
    regs = [r for r in regs if r['length_td'] >= 21]
    print(f'\n--- per-regime Sharpe within the winning window (L=252, ≥21TD) ---')
    print(f'{"winner":18s}  {"start":>10s}  {"end":>10s}  {"TD":>4s}  '
          f'{"sh_in_window":>13s}  {"sh_full":>9s}  {"lift":>9s}')
    rows = []
    for r in regs:
        w = r['winner']
        s = pd.Timestamp(r['start']); e = pd.Timestamp(r['end'])
        ret = df[w].loc[s:e]
        sh = annualized_sharpe(ret)
        rows.append({**r, 'sh_in_window': sh, 'sh_full': full[w], 'lift': sh - full[w]})
        print(f'  {w:18s}  {s.date()!s:>10s}  {e.date()!s:>10s}  '
              f'{r["length_td"]:>4d}  {sh:>+13.3f}  {full[w]:>+9.3f}  '
              f'{sh-full[w]:>+9.3f}')

    print('\n--- aggregate by arc: how much does the within-window Sharpe lift vs full-sample? ---')
    print(f'{"arc":18s}  {"n_regimes":>9s}  {"td_total":>8s}  '
          f'{"mean_sh_in_win":>14s}  {"sh_full":>9s}  {"lift":>9s}')
    by_arc: dict[str, list[dict]] = {}
    for row in rows:
        by_arc.setdefault(row['winner'], []).append(row)
    for arc in sorted(by_arc, key=lambda a: -np.mean([r['sh_in_window'] for r in by_arc[a]])):
        wins = by_arc[arc]
        td = sum(r['length_td'] for r in wins)
        mean_in = float(np.mean([r['sh_in_window'] for r in wins]))
        print(f'  {arc:18s}  {len(wins):>9d}  {td:>8d}  '
              f'{mean_in:>+14.3f}  {full[arc]:>+9.3f}  {mean_in-full[arc]:>+9.3f}')


if __name__ == '__main__':
    main()
