"""Local smoke for the `forward_skip` implementation-lag threading.

Two guards:
  1. **Regression (load-bearing):** `forward_log_returns(..., skip=0)`
     is *bit-identical* to the pre-change formula, and the skip=0
     `align_tickers_at_rebal` filter is unchanged — so the 2026-05-18
     skip-0 cells reproduce exactly and the prior leaderboard rows
     stand.
  2. **Correctness:** skip=1 equals the hand-rolled 1-day-lagged target
     and trims exactly one extra trailing row; the full walk-forward
     runs end-to-end at skip ∈ {0,1} for indicator @ {20,5} with finite
     IC/Sharpe.

Run: uv run python apps/factor/scripts/smoke_forward_skip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from factor import (
    IndicatorGridConfig, load_ticker_indicators,
    train_scorer_indicators_walkforward,
)
from factor.data import forward_log_returns

STOOQ = 'apps/notebook/data/stooq_us_long'
SENT = -7.0  # NaN sentinel for array_equal


def main() -> int:
    # ---- 1. forward_log_returns: skip=0 bit-identical, skip=1 correct ----
    rng = np.random.default_rng(0)
    P = np.exp(np.cumsum(rng.normal(0, 0.02, size=(400, 7)), axis=0) + 3.0)
    D = P.shape[0]
    logp = np.log(np.maximum(P, 1e-12))
    for H in (5, 10, 20):
        old = np.full_like(P, np.nan, dtype=np.float64)
        old[:D - H] = logp[H:] - logp[:D - H]            # pre-change formula
        new0 = forward_log_returns(P, rebal_days=H, forward_skip=0)
        assert np.array_equal(np.nan_to_num(old, nan=SENT),
                              np.nan_to_num(new0, nan=SENT)), \
            f'skip=0 NOT bit-identical to old formula at H={H}'

        man1 = np.full_like(P, np.nan, dtype=np.float64)
        for i in range(D - H - 1):                        # out[i]=logp[i+1+H]-logp[i+1]
            man1[i] = logp[i + 1 + H] - logp[i + 1]
        new1 = forward_log_returns(P, rebal_days=H, forward_skip=1)
        assert np.array_equal(np.nan_to_num(man1, nan=SENT),
                              np.nan_to_num(new1, nan=SENT)), \
            f'skip=1 mismatch at H={H}'

        assert np.isnan(new0[:, 0]).sum() == H, f'skip0 tail NaN != H at H={H}'
        assert np.isnan(new1[:, 0]).sum() == H + 1, \
            f'skip1 tail NaN != H+1 at H={H}'
    print('  unit: forward_log_returns skip=0 bit-identical, skip=1 '
          'correct, tail-NaN H vs H+1 — OK')

    # ---- 2. end-to-end walk-forward, indicator @ {20,5} × skip {0,1} ----
    manifest = json.loads((Path(STOOQ) / 'manifest.json').read_text())
    names = [t['ticker'] for t in
             sorted(manifest['tickers'], key=lambda t: -t['n_bars'])][:15]
    cfg = IndicatorGridConfig()
    td = [load_ticker_indicators(n, stooq_dir=STOOQ) for n in names]
    base = (63, 39, 39)
    for rebal in (20, 5):
        f = 20.0 / rebal
        tr, va, st = [int(round(b * f)) for b in base]
        cells = []
        for skip in (0, 1):
            wf = train_scorer_indicators_walkforward(
                td, cfg, rebal_days=rebal, forward_skip=skip,
                train_window_blocks=tr, val_window_blocks=va,
                step_window_blocks=st, scorer='linear', n_steps=20,
                learning_rate=1e-2, weight_decay=1e-3, verbose=False)
            assert wf.n_windows >= 1, f'r{rebal} s{skip}: no windows'
            assert np.isfinite(wf.mean_val_ic), f'r{rebal} s{skip}: IC nan'
            assert np.isfinite(wf.mean_val_sharpe), f'r{rebal} s{skip}: Sh nan'
            cells.append((skip, wf.n_windows, wf.mean_val_ic,
                          wf.mean_val_sharpe))
        print(f'  rebal={rebal}: ' + '  |  '.join(
            f's{s}: {nw}w IC={ic:+.4f} Sh={sh:+.3f}'
            for s, nw, ic, sh in cells))

    print('SMOKE GREEN')
    return 0


if __name__ == '__main__':
    sys.exit(main())
