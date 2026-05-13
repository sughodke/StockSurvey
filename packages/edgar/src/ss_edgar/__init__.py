"""SEC EDGAR 13F-HR filings loader.

Public API:

- `EdgarClient` — rate-limited HTTP client with `.edgar-cache/` disk
  cache. Fetches submission lists per CIK + 13F-HR XML by accession.
- `parse_13f_xml` — bytes → `list[HoldingRow]` per filing.
- `build_holdings_panel` — list of fund × date × XML → wide
  `(quarter_end_date, ticker)` market-value panel restricted to a
  given universe.
- `CURATED_FUNDS` — CIK list of well-known institutional managers.
- `NAME_TO_TICKER` — manual issuer-name → ticker map for the most
  commonly-held mega-cap names (covers >90% of typical hedge fund
  13F dollar volume despite being only ~80 names).

The loader is intentionally lightweight: it doesn't try to be a
complete CUSIP database, and it accepts that ~5-10% of holdings
(small-caps, international ADRs, recent IPOs) will be dropped on
the name-matching step. Phase 2b's 13F-consensus scorer concentrates
in mega-caps anyway — this is a deliberate trade-off between
implementation complexity and signal coverage.

See [`apps/cfr` Phase 2 finding](../../apps/docs/docs/findings/cfr-phase2.md)
for the consumer-side rationale.
"""

from ss_edgar.client import EdgarClient, EdgarFiling
from ss_edgar.parsing import HoldingRow, parse_13f_xml
from ss_edgar.holdings import build_holdings_panel
from ss_edgar.funds import (
    CURATED_FUNDS, FundInfo, NAME_TO_TICKER, normalize_issuer_name,
    name_to_ticker,
)

__all__ = [
    'EdgarClient', 'EdgarFiling',
    'HoldingRow', 'parse_13f_xml',
    'build_holdings_panel',
    'CURATED_FUNDS', 'FundInfo',
    'NAME_TO_TICKER', 'normalize_issuer_name', 'name_to_ticker',
]
