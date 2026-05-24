"""Dump the basket-search-winner 4-ETF basket's daily-return stream
for the cross-arc Deflated-Sharpe ladder.

This is the deployable book equivalent to `dca_returns_dump.py` (which
dumps the canonical 13-ETF stream), but for the VTI+TLT+IEF+GLD winner
that the Optuna pre-reg arc surfaced. Same PassiveEW engine, same
commission, same rebal cadence — only the universe differs.

The ladder ArcSpec for this stream sets n_trials=200 (the legitimate
search cost of the basket-search arc), so the deflated-t lands lower
than the original DCA row's n_trials=4 — that's the honest accounting
for an arc that searched 200 configurations to find its universe.

Run from repo root:
    uv run python apps/dca/scripts/dump_winner_basket_returns.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cfr.baselines import PassiveEW
from ss_loaders import load_stooq_matrix

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'Output/dca-winner-4etf-returns.npz'

WINNER = ['VTI', 'TLT', 'IEF', 'GLD']
START, END = '2005-02-25', '2025-12-31'


def main() -> None:
    prices, _, _, _ = load_stooq_matrix(
        str(REPO / 'StooqData'), min_history=10, include_etfs=True,
        start_date=START, end_date=END, tickers=WINNER,
    )
    present = [t for t in WINNER if t in prices.columns]
    print(f'loaded panel {prices.shape}: '
          f'{prices.index[0].date()} → {prices.index[-1].date()}, '
          f'present={present}')

    if set(present) != set(WINNER):
        missing = set(WINNER) - set(present)
        raise RuntimeError(f'winner basket incomplete: missing {missing}')

    passive = PassiveEW(rebal_days=80, commission_bps=10.0)
    daily = np.asarray(passive.daily_returns(prices[present]), dtype=np.float64)
    daily = daily[np.isfinite(daily)]

    sd = daily.std(ddof=0)
    ann_sh = float(daily.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    print(f'DCA winner 4-ETF daily stream: {daily.size} bars, '
          f'annualized Sharpe {ann_sh:+.3f}')

    np.savez(
        OUT, daily_ret=daily, periods_per_year=np.float64(252.0),
        universe=np.asarray(WINNER, dtype=str),
        rebal_days=np.int32(80), commission_bps=np.float64(10.0),
    )
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
