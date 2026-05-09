"""relational: research scaffolding for relational-CWT alpha ideas.

The four ideas tracked in apps/docs/docs/notes.md ("Multi-stock CWT
framings — Where the real alpha may live"), in priority order:

  1. Stock minus sector (excess CWT divergence)         — week-1 focus
  2. CWT of cross-sectional dispersion                  — TODO
  3. CWT of cross-sectional correlation                 — TODO
  4. Cross-sector coherence (sector-pair CWT)           — TODO

Public API:
  * `relational.sectors` — Phase-2 ticker → GICS sector mapping +
    canonical sector ETF universe.
  * `relational.aggregates` — sector-aggregate price series builders
    (equal-weighted constituents by default; ETF series later).
  * `relational.scoring` — `excess_divergence_scores` + the matching
    `weights_excess_regime` weights builder (drop-in for any vectorbt
    or bt loop that already accepts a per-(date, ticker) weights df).

The CLI (`ss-relational`) wraps a head-to-head backtest of `weights_regime`
(per-stock CWT, the existing baseline) vs `weights_excess_regime` (with
sector overlay) on the same universe + dates.
"""

from relational.scoring import (
    excess_divergence_scores,
    weights_excess_regime,
)
from relational.sectors import (
    PHASE2_TICKER_TO_SECTOR,
    SECTOR_ETFS,
    sectors_for_universe,
    ticker_to_sector_idx,
)

__all__ = [
    'PHASE2_TICKER_TO_SECTOR',
    'SECTOR_ETFS',
    'excess_divergence_scores',
    'sectors_for_universe',
    'ticker_to_sector_idx',
    'weights_excess_regime',
]
