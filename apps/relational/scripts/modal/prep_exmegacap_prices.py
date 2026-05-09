"""Local prep for `relational_exmegacap_modal.py`.

Builds the `non-Phase-2` long-history US universe — `stooq_us_long`
manifest minus the 21 PHASE2_TICKERS mega-caps — and pickles the
prices DataFrame for Modal RPC.

This isn't strictly "all small caps" (we lack market-cap data in the
workspace); it's "all long-history US stocks excluding the curated
mega-cap list". `stooq_us_long` was filtered for >=22y of history,
which biases mid/large-cap-survivor; subtracting Phase-2 leaves the
mid/large-cap names with similar history depth. Roughly:

  | layer            | size | character                          |
  |------------------|------|------------------------------------|
  | full Stooq US    | ~12k | mixed; many ETFs / sparse / delisted |
  | stooq_us_long    | ~291 | curated 22+y, large + mid cap      |
  | minus Phase-2    | ~270 | curated 22+y, ex-mega-cap          |
  | true small caps  |   ?  | not directly accessible            |

Honest framing in any output: this universe answers "does the
strategy survive moving off the 21 hand-picked mega-caps" but does
not answer "does the strategy work on Russell 2000-style small caps".
The latter would need market-cap data we don't have.

Run:
    uv run python apps/relational/scripts/modal/prep_exmegacap_prices.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from ss_loaders import load_stooq_matrix

from relational.sectors import PHASE2_TICKERS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO_ROOT / 'apps' / 'notebook' / 'data' / 'stooq_us_long' / 'manifest.json'
DEFAULT_OUT = Path('/tmp/exmegacap-prices.pkl')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./StooqData')
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    p.add_argument('--start', default='2013-01-29')
    p.add_argument('--end', default='2025-12-11')
    p.add_argument('--out', default=str(DEFAULT_OUT))
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    all_long = [t['ticker'].upper() for t in manifest['tickers']]
    excluded = set(PHASE2_TICKERS)
    universe = sorted(t for t in all_long if t not in excluded)
    print(f'stooq_us_long: {len(all_long)} names; '
          f'excluding {len(excluded)} Phase-2 names; '
          f'universe: {len(universe)} names')

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
