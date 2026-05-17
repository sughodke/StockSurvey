"""Is rank-IC a good metric? Quantify the IC ↔ walk-forward-Sharpe
decoupling across the existing factor walk-forward corpus.

Leak-free: reads only already-computed `Output/*-windows.npz`
artifacts (per-window `val_ic`, `val_sharpe*`, `val_ir_vs_ew`). No
training, no Modal.

Question: across every factor walk-forward arm we have run, does a
window's val IC predict that window's deployable metric? Reports
Spearman rank correlation at two granularities:

  * pooled per-(arm, window) — the raw decoupling,
  * per-arm-mean — the granularity the leaderboard reasons at.

Legs: IC vs long-only Sharpe (the leaderboard "val Sharpe"), vs
long-short Sharpe, vs IR-vs-EW (alpha over passive equal-weight), and
Sharpe vs IR-vs-EW (does Sharpe even track alpha-over-passive?).

Decision frame: IC↔Sharpe ≈ 0 is expected (cost/breadth geometry). The
load-bearing cell is IC↔IR-vs-EW — if that is also ≈ 0, IC is not a
good metric for this deployment.
"""

from __future__ import annotations

import glob

import numpy as np


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), n
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _leg(d: dict, *names: str) -> np.ndarray | None:
    for nm in names:
        if nm in d:
            return np.asarray(d[nm], dtype=float)
    return None


def main() -> None:
    paths = sorted(set(glob.glob('Output/*-windows.npz')))
    pooled: dict[str, list] = {k: [] for k in
                               ('ic', 'lo', 'ls', 'ir')}
    arm_means: dict[str, list] = {k: [] for k in
                                  ('ic', 'lo', 'ls', 'ir')}
    used, skipped = [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        ic = _leg(d, 'val_ic')
        if ic is None:
            skipped.append(p.split('/')[-1] + ' (no val_ic)')
            continue
        lo = _leg(d, 'val_sharpe', 'val_sharpe_long_only')
        ls = _leg(d, 'val_sharpe_long_short')
        ir = _leg(d, 'val_ir_vs_ew')
        used.append(p.split('/')[-1])
        for key, arr in (('ic', ic), ('lo', lo), ('ls', ls), ('ir', ir)):
            if arr is None:
                continue
            pooled[key].append((p, arr))
        arm_means['ic'].append(np.nanmean(ic))
        arm_means['lo'].append(np.nanmean(lo) if lo is not None else np.nan)
        arm_means['ls'].append(np.nanmean(ls) if ls is not None else np.nan)
        arm_means['ir'].append(np.nanmean(ir) if ir is not None else np.nan)

    # Align pooled legs by source path so each (arm,window) row is
    # consistent across the two legs being correlated.
    def _pair(a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
        da = {p: v for p, v in pooled[a]}
        db = {p: v for p, v in pooled[b]}
        xs, ys = [], []
        for p in da:
            if p in db and da[p].shape == db[p].shape:
                xs.append(da[p])
                ys.append(db[p])
        if not xs:
            return np.array([]), np.array([])
        return np.concatenate(xs), np.concatenate(ys)

    print(f'\nArms with val_ic: {len(used)}   '
          f'skipped (no IC, e.g. horizon cadence policies): '
          f'{len(skipped)}')
    print(f'Pooled (arm,window) rows: IC present in '
          f'{sum(v.size for _, v in pooled["ic"])} windows\n')

    print('POOLED per-(arm,window) Spearman:')
    for label, a, b in (
        ('IC ↔ long-only Sharpe (leaderboard "val Sharpe")', 'ic', 'lo'),
        ('IC ↔ long-short Sharpe', 'ic', 'ls'),
        ('IC ↔ IR-vs-EW (alpha over passive)', 'ic', 'ir'),
        ('long-only Sharpe ↔ IR-vs-EW', 'lo', 'ir'),
    ):
        x, y = _pair(a, b)
        rho, n = _spearman(x, y)
        print(f'  {label:<48s} ρ={rho:+.3f}  (n={n})')

    print('\nPER-ARM-MEAN Spearman (n = arms):')
    am = {k: np.array(v, dtype=float) for k, v in arm_means.items()}
    for label, a, b in (
        ('mean IC ↔ mean long-only Sharpe', 'ic', 'lo'),
        ('mean IC ↔ mean IR-vs-EW', 'ic', 'ir'),
        ('mean long-only Sharpe ↔ mean IR-vs-EW', 'lo', 'ir'),
    ):
        rho, n = _spearman(am[a], am[b])
        print(f'  {label:<40s} ρ={rho:+.3f}  (n={n})')

    # Sign-agreement: of arms with mean val IC > 0, what fraction post
    # positive IR-vs-EW? (the practical "does signal → alpha" question)
    ic_m, ir_m = am['ic'], am['ir']
    msk = np.isfinite(ic_m) & np.isfinite(ir_m)
    pos_ic = msk & (ic_m > 0)
    if pos_ic.any():
        frac = float(np.mean(ir_m[pos_ic] > 0))
        print(f'\nArms with mean IC>0 that also have mean IR-vs-EW>0: '
              f'{frac:.0%}  ({int(pos_ic.sum())} arms)')
    print(f'\nIncluded arms ({len(used)}): '
          + ', '.join(s.replace('-windows.npz', '') for s in used))


if __name__ == '__main__':
    main()
