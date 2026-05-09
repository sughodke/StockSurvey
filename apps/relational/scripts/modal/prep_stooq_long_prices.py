"""Local prep for `relational_morlet_stooq_long.py` (Modal).

Loads the full `apps/notebook/data/stooq_us_long/` 312-ticker panel
via `load_stooq_matrix` and pickles it for Modal RPC, in the bundle
format the morlet-stooq-long entrypoint expects
(`{'prices': DataFrame, 'start': str, 'end': str, 'universe': list}`).

Distinct from `prep_exmegacap_prices.py` — that one excludes the 21
Phase-2 mega-caps to test "does the strategy survive moving off
mega-caps". This one keeps the full 312-name universe so the kNN
candidate pool is the largest available curated long-history set —
the test is "does the polar Morlet bundle's Phase-2 overfit go away
when the candidate pool is ~15× larger" (gating experiment 2 from
`apps/docs/docs/findings/relational-morlet-failure.md`).

Run:
    uv run python apps/relational/scripts/modal/prep_stooq_long_prices.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from ss_loaders import load_stooq_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = (
    REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long' / 'manifest.json')
DEFAULT_OUT = Path('/tmp/stooq-long-prices.pkl')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    universe = sorted(t['ticker'].upper() for t in manifest['tickers'])
    print(f'stooq_us_long manifest: {len(universe)} names')

    prices, _highs, _lows, _vol = load_stooq_matrix(
        args.data_dir, min_history=150,
        start_date=args.start, end_date=args.end,
        tickers=universe)
    print(f'loaded prices: {prices.shape} '
          f'({prices.index[0].date()} → {prices.index[-1].date()})')

    out = Path(args.out)
    with out.open('wb') as f:
        pickle.dump({'prices': prices,
                     'start': args.start, 'end': args.end,
                     'universe': universe},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {out}  ({out.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
