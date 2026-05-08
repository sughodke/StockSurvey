"""Local prep step for `relational_dwt_phase2.py` (Modal).

Loads the Phase-2 mega-cap prices DataFrame from the local StooqData/
archive and pickles it to /tmp. The Modal entrypoint reads the pickle
back as raw bytes and ships it to the remote container — avoids
needing the Stooq archive on Modal (5 of 21 Phase-2 tickers — CRM /
GOOGL / META / NFLX / TSLA — aren't in the baked-in stooq_us_long
subset because they post-date the 2000-01-01 cutoff used when that
subset was built).

The local_entrypoint runs in the uvx-isolated env, so pandas /
ss_loaders aren't available there. This script runs in the project
venv where they are.

Run:
    uv run python apps/relational/scripts/modal/prep_phase2_prices.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from ss_loaders import load_stooq_matrix

from relational.sectors import PHASE2_TICKERS


DEFAULT_OUT = Path('/tmp/phase2-prices.pkl')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args()

    prices, _highs, _lows, _vol = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=list(PHASE2_TICKERS))
    print(f'loaded prices: {prices.shape} '
          f'({prices.index[0].date()} → {prices.index[-1].date()})')
    print(f'tickers: {list(prices.columns)}')

    out = Path(args.out)
    with out.open('wb') as f:
        pickle.dump({'prices': prices,
                     'start': args.start, 'end': args.end},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {out}  ({out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
