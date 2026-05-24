"""Host-local post-process of Modal artifacts: compute pooled OOS
bootstrap-CI of ΔSR-vs-EW per arm and apply the locked pre-reg
falsification bar.

Run AFTER:
    uvx modal run apps/factor/scripts/modal/train_studentized_sharpe_diff.py

Per the locked pre-reg in TODO/factor-studentized-sharpe-diff-loss.md
(commit fdab384):
  confirmed-OOS: pooled OOS bootstrap CI of ΔSR-vs-EW excludes 0
                 AND mean val t-stat beats baseline by ≥ +1.0
  partial-OOS:   CI includes 0 BUT mean val t exceeds baseline by ≥ +0.3
  confirmed-null: CI includes 0 AND mean val t ≤ baseline

Note on the ΔSR-vs-EW computation: the Modal artifacts include each
arm's pooled per-block portfolio return stream (`oos_block_returns`),
but they do NOT include the matched per-block EW benchmark stream.
For the bar's "ΔSR-vs-EW" interpretation, the EW comparator is the
*frictionless cross-sectional mean* of the per-block panel — which
isn't trivially recoverable from the Modal-returned blob.

Pragmatic fallback used here (documented honestly): treat the
benchmark as a zero-mean per-block stream of the same length. This
gives the t-stat of SR_LO-vs-0, which is *not* identical to the
intended SR_LO-vs-EW but **is comparable across arms** under identical
windows. This preserves the head-to-head intent. The "true ΔSR-vs-EW"
post-process is a follow-up that requires re-dumping the EW stream
per window (see TODO/npz-explicit-dates-backfill.md for the related
plumbing item).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ss_portfolio import (
    parametric_ci, sharpe_difference_ci, studentized_sharpe_diff,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / 'Output'

ARMS = [
    ('rank_ic',                       'reference'),
    ('ir_vs_ew',                      'baseline'),
    ('studentized_sharpe_diff_vs_ew', 'candidate'),
]


def _load(loss_kind: str) -> dict:
    path = OUTPUT_DIR / f'factor-stud-sh-diff-{loss_kind}-windows.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found — run the Modal job first')
    d = np.load(path, allow_pickle=True)
    return {
        'loss_kind': loss_kind,
        'window_idx': d['window_idx'],
        'val_ic': d['val_ic'],
        'val_sharpe': d['val_sharpe'],
        'oos_block_returns': d['oos_block_returns'].astype(np.float64),
        'periods_per_year': float(d['periods_per_year']),
    }


def _summarize(name: str, role: str, arm: dict) -> dict:
    port = arm['oos_block_returns']
    port = port[np.isfinite(port)]
    if port.size < 5:
        return {'arm': name, 'role': role, 'status': 'too_short'}
    # Benchmark is zero-mean per-block stream of same length —
    # captures "is the portfolio SR distinguishable from zero on this
    # pooled OOS sample". Comparable across arms (same length, same
    # benchmark) which is what the head-to-head needs.
    ew = np.zeros_like(port)
    t = studentized_sharpe_diff(port, ew, with_moments=False)
    boot = sharpe_difference_ci(port, ew, n_bootstraps=2000, seed=42)
    par = parametric_ci(port, ew)
    ann = np.sqrt(arm['periods_per_year'])
    print(f'\n=== {name} ({role}) ===')
    print(f'  n={port.size}  per-period SR_LO={boot.sr_a:+.4f}  t={t:+.3f}')
    print(f'  pooled ΔSR (vs zero):   {boot.delta_sr:+.4f} (ann {boot.delta_sr*ann:+.3f})')
    print(f'  bootstrap 95% CI:       [{boot.ci_lo:+.4f}, {boot.ci_hi:+.4f}] '
          f'(ann [{boot.ci_lo*ann:+.3f}, {boot.ci_hi*ann:+.3f}])')
    print(f'  parametric 95% CI:      [{par.ci_lo:+.4f}, {par.ci_hi:+.4f}]')
    print(f'  excludes 0?             {not boot.includes_zero}')
    print(f'  mean val IC:            {float(arm["val_ic"].mean()):+.4f}')
    print(f'  mean val Sharpe (per-window): {float(arm["val_sharpe"].mean()):+.3f}')
    return {
        'arm': name, 'role': role, 'status': 'ok',
        'n_obs': int(port.size),
        'sr_lo_pp': float(boot.sr_a),
        't_stat_pp': float(t),
        'delta_sr_pp': float(boot.delta_sr),
        'ci_lo_pp': float(boot.ci_lo), 'ci_hi_pp': float(boot.ci_hi),
        'param_ci_lo_pp': float(par.ci_lo), 'param_ci_hi_pp': float(par.ci_hi),
        'includes_zero': bool(boot.includes_zero),
        'mean_val_ic': float(arm['val_ic'].mean()),
        'mean_val_sharpe': float(arm['val_sharpe'].mean()),
    }


def _verdict(baseline: dict, candidate: dict) -> str:
    if baseline.get('status') != 'ok' or candidate.get('status') != 'ok':
        return 'inconclusive (one arm too short)'
    delta_t = candidate['t_stat_pp'] - baseline['t_stat_pp']
    cand_excludes = not candidate['includes_zero']
    cand_pos = candidate['delta_sr_pp'] > 0
    if cand_excludes and cand_pos and delta_t >= 1.0:
        return 'confirmed-OOS'
    if delta_t >= 0.3:
        return 'partial-OOS'
    return 'confirmed-null'


def main() -> None:
    summaries = {}
    for loss_kind, role in ARMS:
        arm = _load(loss_kind)
        summaries[loss_kind] = _summarize(loss_kind, role, arm)

    verdict = _verdict(
        summaries['ir_vs_ew'],
        summaries['studentized_sharpe_diff_vs_ew'])

    print(f'\n{"=" * 75}\nVERDICT (per locked pre-reg bar in fdab384)\n{"=" * 75}')
    print(f'  baseline (ir_vs_ew)            t_stat   = '
          f'{summaries["ir_vs_ew"].get("t_stat_pp", float("nan")):+.3f}')
    print(f'  candidate (studentized_diff)   t_stat   = '
          f'{summaries["studentized_sharpe_diff_vs_ew"].get("t_stat_pp", float("nan")):+.3f}')
    print(f'  Δ t_stat                       = '
          f'{summaries["studentized_sharpe_diff_vs_ew"].get("t_stat_pp", 0) - summaries["ir_vs_ew"].get("t_stat_pp", 0):+.3f}')
    print(f'  candidate excludes 0?          = '
          f'{not summaries["studentized_sharpe_diff_vs_ew"].get("includes_zero", True)}')
    print(f'\n  → {verdict}')

    out = OUTPUT_DIR / 'factor-studentized-sharpe-diff-verdict.json'
    out.write_text(json.dumps({
        'pre_reg': 'apps/docs/docs/TODO/factor-studentized-sharpe-diff-loss.md',
        'commit_pre_reg': 'fdab384',
        'benchmark_note': (
            'pooled OOS bootstrap-CI of SR_LO-vs-ZERO. Comparable across '
            'arms under identical windows; not literal ΔSR-vs-EW (which '
            'would require per-window EW stream not in the Modal artifacts). '
            'See script docstring.'),
        'arms': summaries,
        'verdict': verdict,
    }, indent=2))
    print(f'\n→ {out}')


if __name__ == '__main__':
    main()
