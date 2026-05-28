"""ss_loaders: price-data loaders.

Four sources of historical OHLC are supported:

  * `load_price_matrix` — Kaggle `svaningelgem/nasdaq-daily-stock-prices`
    per-ticker CSVs (no volume, no adj_close). Wide DataFrame output.
  * `load_stooq_matrix` — Stooq daily-archive bulk dump (has volume,
    split/dividend-adjusted, includes delisted tickers). Wide.
  * `load_yahoo`        — single-ticker fetch via yfinance (has volume).
  * `load_cryptocompare`— single-ticker crypto fetch via the public
    cryptocompare.com REST API.

Plus symbol lists in `symbols`: NASDAQ-100, NASDAQ all, top-100 crypto,
S&P 500 sample, and a user-curated favourites list.
"""

from ss_loaders.congress import (
    LEADERSHIP_2014_2025,
    LeadershipFilter,
    build_leadership_filter,
    load_congressional_trades_xlsx,
    load_legislator_metadata,
    load_senate_stock_watcher,
)
from ss_loaders.csv_matrix import load_price_matrix
from ss_loaders.cryptocompare import load_cryptocompare
from ss_loaders.hyperliquid import (
    load_hl_close_panel,
    load_hl_funding_history,
    load_hl_funding_panel,
    load_hl_perp_candles,
    load_hl_perp_universe,
)
from ss_loaders.stooq_matrix import (
    iter_stooq_ticker_files,
    load_stooq_matrix,
    read_stooq_file,
    stooq_ticker_from_path,
)
from ss_loaders.symbols import (
    MY_FAVES,
    NDX_CONSTITUENTS,
    coin100,
    coin100_dataframe,
    nasdaq,
    snp_500,
)
from ss_loaders.yahoo import load_yahoo

__all__ = [
    'LEADERSHIP_2014_2025',
    'LeadershipFilter',
    'MY_FAVES',
    'NDX_CONSTITUENTS',
    'build_leadership_filter',
    'coin100',
    'coin100_dataframe',
    'iter_stooq_ticker_files',
    'load_congressional_trades_xlsx',
    'load_cryptocompare',
    'load_hl_close_panel',
    'load_hl_funding_history',
    'load_hl_funding_panel',
    'load_hl_perp_candles',
    'load_hl_perp_universe',
    'load_legislator_metadata',
    'load_price_matrix',
    'load_senate_stock_watcher',
    'load_stooq_matrix',
    'load_yahoo',
    'nasdaq',
    'read_stooq_file',
    'snp_500',
    'stooq_ticker_from_path',
]
