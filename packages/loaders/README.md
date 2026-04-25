# ss_loaders

Price-data loaders. Four sources, one shape contract: wide DataFrames
indexed by date, columns are tickers.

```python
from ss_loaders import (
    load_price_matrix,    # Kaggle Nasdaq3347 per-ticker CSVs
    load_stooq_matrix,    # Stooq daily archive (recommended)
    load_yahoo,           # yfinance single-ticker
    load_cryptocompare,   # crypto OHLC via cryptocompare.com REST
)
```

## Which loader to use

| Source | Adj prices | Volume | Delistings | Universe | When to use |
|---|---|---|---|---|---|
| `load_stooq_matrix` | yes | yes | yes | ~12K US tickers, back to 1962 | **Default for new work.** Local archive at `./StooqData`; pickle-cached after first scan. |
| `load_price_matrix` | no | no | survivors only | Kaggle Nasdaq dump (~3.3K tickers) | Legacy. Kept for reproducing pre-Stooq results. |
| `load_yahoo` | yes | yes | partial | single ticker via yfinance | One-off lookups, debugging, plotting. |
| `load_cryptocompare` | n/a | yes | n/a | single crypto pair | Crypto research only. |

## Survivorship bias — read this if using Stooq

`load_stooq_matrix` returns a **point-in-time** panel:

- Tickers that IPO'd partway through the date range have **leading NaN**.
- Tickers that delisted partway through have **trailing NaN** (after a 5-bar `ffill` for short halts).
- The panel-wide `min_history` filter (default 252) is intentionally lenient — only drops "ghost" tickers with truly minimal coverage in the requested range.

The strict survivorship filter is **per-walk-forward window**, applied
downstream in `regime.trainer._filter_window_universe`. Each window
defines its own eligible universe (must have valid first bar +
`per_window_min_history` valid bars within the window). This is the
cure for the "I only ever held survivors" backtest illusion.

The cost: the causal CWT's cumsum-based rolling z-norm can't tolerate
leading NaN — once cumsum hits a NaN, every subsequent cumulative
value is NaN. So **anything consuming the loader's output must either
slice each ticker from its first valid date or apply the per-window
filter before computing CWT**. Tools that operate per-column on
already-non-NaN data are fine; tools that cumsum across the panel
are not.

## Stooq archive layout

The archive ships as a zip per market that unpacks to:

```
daily/<country>/<exchange> <type>/<bucket>/<ticker>.us.txt
```

Files are angle-bracketed CSV:

```
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,19840907,000000,0.0991725,...,99242379,0
```

Free to download from <https://stooq.com/db/h/>. Update by re-downloading;
the loader cache (`<data-dir>/.cache.pkl`) is invalidated by deletion only.

## Symbol lists

`ss_loaders.symbols` exports curated lists for callers that need a
universe definition without the full archive scan:

```python
from ss_loaders import NDX_CONSTITUENTS, snp_500, nasdaq, MY_FAVES, coin100
```

`NDX_CONSTITUENTS` is the Nasdaq-100; `snp_500` is a sample S&P 500
list (not point-in-time); `nasdaq` is a broader Nasdaq listing;
`MY_FAVES` is a hand-picked watchlist; `coin100` and `coin100_dataframe`
return top-100 cryptocurrencies via the cryptocompare API.

## Quick examples

```python
# Stooq with a 2000-onward window, panel cached for fast re-runs
close, high, low, vol = load_stooq_matrix(
    './StooqData',
    start_date='2000-01-01',
    end_date='2025-12-31',
    cache_path='./StooqData/.cache.pkl',
)

# Single-ticker Yahoo for quick spot-check
df = load_yahoo('AAPL', start='2020-01-01', end='2024-12-31')

# Crypto OHLC
btc = load_cryptocompare('BTC', limit=2000)
```
