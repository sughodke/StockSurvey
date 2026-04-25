"""ss_stream: a point-in-time view over the Stooq daily archive.

The Stooq archive ships as ~12K per-ticker text files organized by
current exchange listing. That layout is convenient for matrix-style
loading (`ss_loaders.load_stooq_matrix`) but bakes in survivorship
bias and discards the natural staggered listing dates.

`ss_stream` reshapes the same source into a two-table parquet event
store:

  * `instruments.parquet` — one row per ticker with its first/last
    bar date. Listing and (heuristic) delisting dates fall out of
    these directly.
  * `bars/data.parquet`   — long-form OHLCV across all tickers,
    sorted by date then ticker.

The `Universe` class then answers `active_at(date)` and
`bars_between(start, end, tickers)` queries — the API a backtest
engine actually wants when iterating rebalance dates over a
universe whose membership changes daily.

Limitations
-----------
Stooq's free per-exchange archive purges delisted tickers entirely,
so the inferred listing dates are real (first bar of each file) but
delisting events fire only for a handful of recently-stale names.
The data itself is still survivorship-biased on the exit side; the
layer is forward-compatible with a true delisted feed (e.g. Norgate)
dropped into the same directory tree.
"""

from ss_stream.ingest import ingest_stooq
from ss_stream.universe import Universe

__all__ = [
    'Universe',
    'ingest_stooq',
]
