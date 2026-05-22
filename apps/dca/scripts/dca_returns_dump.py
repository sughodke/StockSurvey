"""Dump the canonical DCA backtest's daily net-return stream for the
cross-arc Deflated-Sharpe harness (`ss_portfolio.standardize_oos`).

DCA is the live strategy: fixed-target 1/13 equal weight on the 13-ETF
Phase-4d universe, quarterly rebal (80-trading-day floor), 10 bps on L1
turnover. That is exactly `cfr.baselines.PassiveEW(rebal_days=80,
commission_bps=10)` over the Phase-4d close panel — reused here so the
DCA stream is identical to the baseline the CFR-vs-DCA comparison
scored against.

Run from repo root:
    uv run python apps/dca/scripts/dca_returns_dump.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from cfr.baselines import PassiveEW

REPO = Path(__file__).resolve().parents[3]
CLOSE_PKL = REPO / 'Output' / 'cfr_phase4d_multiasset_close.pkl'
OUT = REPO / 'Output' / 'dca-returns.npz'


def main() -> None:
    with open(CLOSE_PKL, 'rb') as f:
        close = pickle.load(f)
    print(f'loaded close panel {close.shape}: '
          f'{close.index[0].date()} -> {close.index[-1].date()}, '
          f'{list(close.columns)}')

    passive = PassiveEW(rebal_days=80, commission_bps=10.0)
    daily = np.asarray(passive.daily_returns(close), dtype=np.float64)
    daily = daily[np.isfinite(daily)]

    sd = daily.std(ddof=0)
    ann_sh = float(daily.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    print(f'DCA daily stream: {daily.size} bars, annualized Sharpe {ann_sh:+.3f}')

    np.savez(OUT, daily_ret=daily, periods_per_year=np.float64(252.0))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
