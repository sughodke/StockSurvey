"""Local prep for Phase 2b — fetch 13F holdings + build the consensus
panel + pickle for Modal consumption.

Run after `prep_phase1_data.py`. Total wall ~4 minutes on a clean
.edgar-cache (limited by SEC's 10 req/s rate limit, 15 funds × ~50
quarterly filings × ~2-3 requests per filing). Subsequent runs are
near-instant (cache).

Writes:
- `Output/cfr_phase2b_consensus_panel.pkl` — DataFrame indexed by
  quarter-end, columns are tickers, values 1.0 if in top-K consensus
  for that quarter else 0.0. Restricted to the stooq_us_long universe.
- `Output/cfr_phase2b_count_panel.pkl` — raw fund-count panel
  (diagnostic; not consumed by Modal).

Re-run with `--force-refresh` to bypass the EDGAR cache (rare, only
if SEC fixes a parsing-relevant bug in their archive).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from ss_edgar import (
    CURATED_FUNDS, EdgarClient,
)
from ss_edgar.holdings import build_holdings_panel, build_consensus_top_k


REPO_ROOT = Path(__file__).resolve().parents[4]
EDGAR_CACHE = REPO_ROOT / '.edgar-cache'
OUTPUT_DIR = REPO_ROOT / 'Output'
DEFAULT_MANIFEST = REPO_ROOT / 'apps/notebook/data/stooq_us_long/manifest.json'


def main(
    *,
    min_period: str = '2013-01-01',
    top_k: int = 20,
    user_agent: str = 'StockSurvey research bot research@example.com',
    manifest: Path = DEFAULT_MANIFEST,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe_raw = json.loads(manifest.read_text())
    universe = sorted(t['ticker'].upper() for t in universe_raw['tickers'])
    print(f'universe: {len(universe)} tickers from {manifest.name}')

    client = EdgarClient(cache_dir=EDGAR_CACHE, user_agent=user_agent)

    print(f'\nfetching 13F holdings for {len(CURATED_FUNDS)} funds '
          f'(min_period={min_period}):')
    for f in CURATED_FUNDS:
        print(f'  - {f.short:5s} CIK {f.cik:>10d} {f.name}')

    t0 = time.time()
    value_panel, count_panel = build_holdings_panel(
        client, CURATED_FUNDS,
        min_period=min_period,
        universe=universe,
        verbose=True,
    )
    print(f'\nbuilt panels in {time.time() - t0:.1f}s')
    print(f'  count_panel: {count_panel.shape} '
          f'({count_panel.index[0].date()} → {count_panel.index[-1].date()})')

    # Top-10 broadly held names across the whole window (sanity)
    totals = count_panel.sum(axis=0).sort_values(ascending=False)
    print(f'\ntop-15 most broadly-held names (all quarters):')
    for tk, count in totals.head(15).items():
        print(f'  {tk:6s} {count}')

    consensus = build_consensus_top_k(count_panel, top_k=top_k)
    print(f'\nconsensus_panel ({top_k} per quarter): {consensus.shape}')
    print(f'  first quarter consensus: '
          f'{sorted(consensus.iloc[0][consensus.iloc[0] > 0].index.tolist())}')
    print(f'  last quarter consensus: '
          f'{sorted(consensus.iloc[-1][consensus.iloc[-1] > 0].index.tolist())}')

    consensus_path = OUTPUT_DIR / 'cfr_phase2b_consensus_panel.pkl'
    with open(consensus_path, 'wb') as f:
        pickle.dump(consensus, f)
    print(f'\nwrote {consensus_path} '
          f'({consensus_path.stat().st_size / 1024:.0f} KB)')

    count_path = OUTPUT_DIR / 'cfr_phase2b_count_panel.pkl'
    with open(count_path, 'wb') as f:
        pickle.dump(count_panel, f)
    print(f'wrote {count_path} '
          f'({count_path.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--min-period', default='2013-01-01')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--user-agent',
                   default='StockSurvey research bot research@example.com')
    args = p.parse_args()
    main(min_period=args.min_period, top_k=args.top_k,
         user_agent=args.user_agent)
