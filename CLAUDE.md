# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockSurvey is a Python-based stock market analysis and visualization system that uses technical indicators (RSI, MACD, Bollinger Bands) to identify buy/sell signals and evaluate trading strategies. It combines:
- Historical stock/cryptocurrency data fetching and caching
- Technical indicator calculations and trend analysis
- Trading strategy decision-making (buy/sell signal generation)
- Backtesting and performance evaluation
- REST API web service for remote evaluation
- Front-end visualization with interactive plots

The system is designed for analyzing large sets of securities (NASDAQ-100, S&P 500, crypto assets) and ranking them by trading relevance.

## Architecture Overview

### Core Workflow

The application follows a **Strategy Pattern** workflow:

1. **Data Acquisition**: `Security` class loads historical OHLCV data from Yahoo Finance or CryptoCompare API
2. **Data Caching**: Uses joblib to cache Security objects in `DataStore/` directory with version migration support
3. **Time-Series Views**: `Span` context managers create filtered dataset views (daily/weekly/monthly)
4. **Indicator Calculation**: Mixins compute technical indicators (RSI, MACD, Bollinger Bands) on the dataset
5. **Signal Generation**: `Decider` classes identify buy/sell crossover points based on indicator values
6. **Backtesting**: `TheEvaluator` calculates strategy performance (P&L in dollars and percentage)
7. **Visualization**: `PlotMixin` generates SVG/PNG charts with price, indicators, and trade markers
8. **HTTP API**: `webservice.py` serves evaluation requests and relevance scoring

### Key Data Classes

- **`models/security.py`**: `Security` class represents a ticker with daily/weekly/monthly DataFrames
  - Lazy-loads data, syncs with latest prices, caches to disk
  - Multiple time-series views via `span()` context manager
  - Supports both stocks (Yahoo Finance) and cryptocurrencies (CryptoCompare)

- **`models/span.py`**: `Span` context manager that orchestrates the workflow
  - `RSI` mode: Uses RSI(7) with 10-period exponential MA for crossover signals
  - `MACD` mode: Uses MACD histogram for divergence signals
  - `BBands` mode: Bollinger Bands support/resistance levels
  - Automatically chains: `calc` (indicators) → `decide` (buy/sell) → `eval` (backtest) → `plot` (visualization)

- **`models/directors.py`**: `TheDecider` and subclasses implement entry/exit logic
  - Filters raw indicator crossovers to match buy before sell patterns
  - Computes confidence weights based on RSI levels
  - Returns tuples of (buy_indices, sell_indices, confidence_volumes)

### Technical Indicators

All implemented in `util/indicators.py` and `models/indicators.py`:
- **RSI**: 7-period relative strength index with 10-period EMA of RSI
- **MACD**: 12/26 exponential moving average convergence/divergence with signal line
- **Bollinger Bands**: 21-period SMA with ±2 std dev bands
- **Fibonacci Retracement**: Support/resistance levels at Fibonacci ratios
- **Moving Averages**: Simple and exponential MA with configurable periods

### Data Flow

```
ticker → Security.load() → cache check → fetch data → add weekly/monthly → Span context
    ↓
RSIMixin/MACDMixin/BBandsMixin (compute indicators)
    ↓
NumpyDecider/MACDDecider (find crossovers, filter, confidence)
    ↓
TheEvaluator (backtest: buy_price × qty, sell_price × qty, net P&L)
    ↓
PlotMixin (matplotlib SVG/PNG with price, MA, indicators, buy/sell markers)
    ↓
Client (display chart or JSON response)
```

### Relevance Scoring

`sort_securities.py` implements `Relevancy` class that:
- Evaluates multiple securities in parallel using joblib
- Scores each based on daily and weekly indicator performance
- Combines scores with configurable weights (default: 40% daily, 60% weekly)
- Returns ranked list of tickers by trading opportunity strength

## Development Commands

### Running the Web Service

```bash
python webservice.py
# Runs aiohttp server on port 8080
# Endpoints:
#   GET /evaluate?ticker=AAPL&span=daily&start_date=2023-01-01
#   GET /relevance?key=n100&w_daily=0.4&w_weekly=0.6
#   Static files from Frontend/
#   Swagger UI at /api/doc
```

### Analyzing a Single Security

```bash
python evaluate_securities.py AAPL
python evaluate_securities.py --span weekly --save GLD
python evaluate_securities.py --macd NVDA  # Use MACD instead of RSI
python evaluate_securities.py --force TSLA  # Invalidate cache and refetch
```

### Batch Analysis

```bash
python evaluate_securities.py --ndx  # All NASDAQ-100 + favorites
python evaluate_securities.py --crypto BTC ETH  # Cryptocurrencies (ticker prefixed with 'coin')
```

### Visualization Scripts

- `plot_stock_market.py`: Unsupervised learning visualization (market structure via GraphLasso, clustering, manifold embedding)
- `plot_rsi_support.py`: RSI with support/resistance levels
- `plot_predicted_rsi_support.py`: RSI prediction using wavelet transforms and SVM
- `rsi.py`: RSI prediction with 1/3/7-day forward models
- `simple_draw_rsi.py`: Minimal RSI chart

### Testing

```bash
python -m pytest models/security_test.py
# Unit tests for Security class (sync, save/load, versioning)
```

## Key Dependencies

- **Data Fetching**: `pandas-datareader` (Yahoo Finance), `requests` (CryptoCompare API), `gdax` (deprecated crypto)
- **Data Processing**: `pandas`, `numpy`
- **ML/Signal Processing**: `scikit-learn`, `scipy` (wavelets, SVD)
- **Visualization**: `matplotlib`, `seaborn`
- **Web**: `aiohttp`, `aiohttp-swagger` (Swagger UI)
- **Persistence**: `joblib` (Security caching), `sklearn.externals.joblib` (parallel processing)
- **Bayesian**: `pymc3`, `Theano` (unused in current workflows)

## Caching and Data Management

- **Cache Location**: `DataStore/` directory (joblib pickles)
- **Cache Key**: `{coin}{ticker}` (prefix 'coin' for crypto)
- **Staleness Check**: 1 day (resyncs data if > 1 day old)
- **Versioning**: Security class version tracked; auto-upgrade logic on load
- **Cache Invalidation**: Use `--force` flag in CLI or pass `force_fetch=True` to `Security.load()`

## Configuration and Symbols

- **Default Start Date**: June 1, 2017 (Security.STARTDATE)
- **NASDAQ-100**: `finance_ndx.py` (hardcoded list) + custom favorites in `my_faves`
- **S&P 500**: `s-and-p-500-companies/data/constituents.csv` (util/load_symbols.py)
- **Cryptocurrencies**: Top 100 by market cap from `s-and-p-500-companies/coin100.json`

## Common Workflows

### Adding a New Indicator

1. Implement calculation in `util/indicators.py` (standalone function)
2. Create a Mixin class in `models/indicators.py` (RSIMixin pattern)
3. Create Decider subclass in `models/directors.py` (NumpyDecider pattern)
4. Create Span subclass in `models/span.py` (wires indicator + decider + evaluator + plotter)
5. Add to klass_lookup in `Security.span()` method

### Modifying Buy/Sell Logic

- Edit `TheDecider.compute_possible_buysell()` in `models/directors.py` (identify raw crossovers)
- Edit `TheDecider.filter_buysell()` to reject invalid patterns (e.g., sell before buy)
- Edit `buy_confidence()` to adjust risk weighting

### Changing Plotting

- Subclass `PlotBaseMixin` from `models/plotter.py`
- Override `plot_indicator()`, `plot_price()`, `plot_buysell()`, `plot_retracement()` methods
- Register in Span subclass via `self.plot = YourPlotMixin(...)`

## Important Implementation Notes

- **Symbol Prefix for Crypto**: Tickers starting with "coin" are treated as cryptocurrencies (e.g., "coinBTC")
- **DataFrame Columns**: Expected columns are lowercase: `open`, `high`, `low`, `close`, `adj_close`, `volume`
- **Index Requirement**: All DataFrames must have DatetimeIndex for resampling to work
- **Business Days**: `timespan.py` uses CustomBusinessDay with USFederalHolidayCalendar for aggregation
- **Matplotlib Backend**: Ensure matplotlib backend supports rendering to SVG for web service

## Known Limitations and TODOs

- `predict_rsi` endpoint in webservice.py is not implemented (returns NotImplemented)
- Forward-looking predictions (rsi.py) roll data, causing consistency issues on trailing data
- Parallel processing commented out (can enable with joblib.Parallel)
- Some deprecated APIs (gdax, old_coins function) left for compatibility
- pymc3/Theano dependencies unused (kept for potential Bayesian models)
- CryptoCompare API may have rate limits for large batch requests

## File Structure Summary

```
StockSurvey/
├── webservice.py              # aiohttp REST API
├── evaluate_securities.py     # CLI entry point for single/batch analysis
├── sort_securities.py         # Relevancy scoring for multiple securities
├── rsi.py, plot_*.py          # Analysis and visualization scripts
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── models/
│   ├── security.py           # Data loading, caching, sync
│   ├── span.py               # Workflow orchestration (context manager)
│   ├── directors.py          # Buy/sell decision logic
│   ├── indicators.py         # Indicator mixins (RSI, MACD, BBands)
│   ├── plotter.py            # Visualization (matplotlib to SVG/PNG)
│   ├── timespan.py           # Time aggregation (daily→weekly/monthly)
│   └── security_test.py      # Unit tests
├── util/
│   ├── load_ticker.py        # Yahoo Finance, CryptoCompare API
│   ├── load_symbols.py       # Symbol lists (NASDAQ, S&P 500, crypto)
│   └── indicators.py         # Technical indicator functions
├── Frontend/
│   └── index.html            # Slick carousel demo HTML
├── DataStore/                # Cache directory (joblib pickles)
├── Output/                   # Generated plots
└── s-and-p-500-companies/    # Symbol data (submodule)
```

