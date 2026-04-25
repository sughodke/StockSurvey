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

from ss_loaders.csv_matrix import load_price_matrix
from ss_loaders.cryptocompare import load_cryptocompare
from ss_loaders.stooq_matrix import load_stooq_matrix
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
    'MY_FAVES',
    'NDX_CONSTITUENTS',
    'coin100',
    'coin100_dataframe',
    'load_cryptocompare',
    'load_price_matrix',
    'load_stooq_matrix',
    'load_yahoo',
    'nasdaq',
    'snp_500',
]
