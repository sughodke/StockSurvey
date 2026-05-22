"""Cross-arc Deflated-Sharpe aggregator for the leaderboard.

Reads the per-arc OOS net-return streams dumped by each deployable
arc's eval driver (``Output/<arc>-returns.npz``, produced via that
driver's ``--dump-returns`` flag), runs every stream through
``ss_portfolio.standardize_oos``, and emits one ranked table keyed on
the **deflated-Sharpe t-stat** — the single cross-arc-comparable number
the leaderboard sorts on.

Two framings, picked per row by ``mode``:

* ``standalone`` — the row claims an absolute Sharpe (factor long-short,
  relational basket, vol short-vol book). Deflate the strategy's own
  net-return stream.
* ``overlay`` — the row claims *alpha over a benchmark* (gate, any
  exposure/timing overlay). Deflate the **excess** stream
  (strategy − benchmark), because that excess is the claimed edge.
  Reported ``ann_sharpe`` / DSR are then on the excess, and the
  standalone Sharpe is carried separately for context.

``n_trials`` per arc is the multiple-testing deflation term — the count
of configurations the arc tried before reporting its winner. These are
reconstructed from the leaderboard log (see ``TRIAL_COUNTS`` and the
companion finding); a single pre-registered config is ``n_trials=1``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ss_portfolio import standardize_oos

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / 'Output'


@dataclass(frozen=True)
class ArcSpec:
    key: str                 # arc id (matches leaderboard app/experiment)
    npz: str                 # filename under Output/
    stream_key: str          # npz key holding the strategy net-return stream
    n_trials: int            # configs tried in the arc (deflation term)
    mode: str = 'standalone'  # 'standalone' | 'overlay'
    benchmark_key: str | None = None  # npz key for benchmark stream
    note: str = ''


# Provisional trial counts reconstructed from the leaderboard arc history.
# Refined in the trial-count reconstruction step; conservative where the
# arc's sweep breadth is ambiguous (under-counting trials is the
# anti-conservative error, so we round up when unsure).
SPECS: list[ArcSpec] = [
    # --- completed (local) arcs ---
    ArcSpec(
        key='gate-v0-drawdown',
        npz='gate-returns.npz',
        stream_key='gated_ret',
        benchmark_key='unc_ret',
        mode='overlay',
        n_trials=6,  # v0 threshold sweep {q=.85,.90,.95} x {binary,sigmoid}
        note='gate v0 @ q=0.95; claimed edge is alpha over unconditional EW',
    ),
    ArcSpec(
        key='pairs-v0',
        npz='pairs-returns.npz',
        stream_key='agg_ret',
        mode='standalone',
        n_trials=4,  # pre-reg config + a few screening-param variants
        note='pair-spread mean reversion, aggregate portfolio stream',
    ),
    # --- staged arcs (activate once the driver dumps the npz) ---
    ArcSpec(
        key='factor-indicator-baseline-LO',
        npz='walkforward-linear-s200-wd0.001-windows.npz',
        stream_key='oos_block_returns',
        mode='standalone',
        n_trials=50,  # largest search: horizons x representations x losses x heads x universes
        note='deterministic-indicator long-only top-N; the confirmed-OOS factor anchor',
    ),
    ArcSpec(
        key='factor-long-short',
        npz='walkforward-linear-s200-wd0.001-windows.npz',
        stream_key='oos_block_returns_long_short',
        mode='standalone',
        n_trials=50,
        note='market-neutral long-short book on the same head',
    ),
    ArcSpec(
        key='relational-analog-cross-ticker',
        npz='relational-returns.npz',
        stream_key='val_daily_ret',
        mode='standalone',
        n_trials=16,  # 8-arm scorer x +/-DWT x {cross_ticker, per_ticker}
        note='phase-2 8-arm winner; the confirmed-OOS relational candidate',
    ),
    ArcSpec(
        key='vol-v3-regime-gated',
        npz='vol-returns.npz',
        stream_key='full_panel_alpha',
        mode='standalone',
        n_trials=12,  # v0->v3.1 x sizing conventions x OI filters x regime gates
        note='short-vol v3 deployment recipe; the one large real signal',
    ),
    ArcSpec(
        key='dca-canonical',
        npz='dca-returns.npz',
        stream_key='daily_ret',
        mode='standalone',
        n_trials=4,  # Phase 4a-d basket composition
        note='canonical live multi-asset DCA basket',
    ),
    ArcSpec(
        key='lie-shape-knn-LS',
        npz='lie-shape-knn-returns.npz',
        stream_key='ls_block_returns',
        mode='standalone',
        n_trials=9,  # lie v1/v3/v4 + horizon/lookback/k sweep
        note='shape-kNN 1mo-reversal long/short, Phase-2 (21 names); '
             'IC t=+3.75 but does not survive translation at 21-name breadth',
    ),
]


def _periods_per_year(d, default: float = 252.0) -> float:
    return float(d['periods_per_year']) if 'periods_per_year' in d else default


def compute() -> list[dict]:
    results = []
    for spec in SPECS:
        path = OUTPUT / spec.npz
        if not path.exists():
            print(f'[skip] {spec.key}: {spec.npz} not found (arc not yet re-run)')
            continue
        d = np.load(path, allow_pickle=True)
        if spec.stream_key not in d.files:
            print(f'[skip] {spec.key}: {spec.npz} lacks key '
                  f"'{spec.stream_key}' (stale npz; re-run the arc)")
            continue
        ppy = _periods_per_year(d)
        strat = np.asarray(d[spec.stream_key], dtype=np.float64)
        bench = (np.asarray(d[spec.benchmark_key], dtype=np.float64)
                 if spec.benchmark_key else None)

        if spec.mode == 'overlay':
            if bench is None:
                raise ValueError(f'{spec.key}: overlay mode needs benchmark_key')
            edge = strat - bench
            mb = standardize_oos(edge, periods_per_year=ppy, n_trials=spec.n_trials)
            standalone = standardize_oos(
                strat, periods_per_year=ppy, n_trials=spec.n_trials, benchmark=bench)
            ctx = {'standalone_ann_sharpe': standalone.ann_sharpe,
                   'ir_vs_bench': standalone.ir_vs_bench}
        else:
            mb = standardize_oos(strat, periods_per_year=ppy, n_trials=spec.n_trials,
                                 benchmark=bench)
            ctx = {'ir_vs_bench': mb.ir_vs_bench}

        row = {'key': spec.key, 'mode': spec.mode, 'n_trials': spec.n_trials,
               'note': spec.note, **mb.as_dict(), **ctx}
        results.append(row)

    results.sort(key=lambda r: r['deflated_tstat'], reverse=True)
    return results


def main() -> None:
    results = compute()
    if not results:
        print('No arc return streams found. Re-run arcs with --dump-returns.')
        return
    print(f'\n{"arc":24s} {"mode":10s} {"trials":>6s} {"annSh":>7s} '
          f'{"skew":>6s} {"kurt":>7s} {"E[maxSR]":>8s} {"DSR":>6s} '
          f'{"defl_t":>7s}')
    print('-' * 92)
    for r in results:
        print(f'{r["key"]:24s} {r["mode"]:10s} {r["n_trials"]:>6d} '
              f'{r["ann_sharpe"]:>+7.3f} {r["skew"]:>+6.2f} {r["kurtosis"]:>7.2f} '
              f'{r["expected_max_sharpe"]:>8.4f} {r["dsr"]:>6.3f} '
              f'{r["deflated_tstat"]:>+7.3f}')
    out = OUTPUT / 'dsr-leaderboard.json'
    out.write_text(json.dumps(results, indent=2))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
