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

Two parallel subsystems coexist in this repo:
1. **Legacy single-ticker workflow** — `Security` → `Span` → `Decider` → `Evaluator` → `Plot` for one ticker at a time (Yahoo Finance / CryptoCompare). Powers `evaluate_securities.py`, `sort_securities.py`, and `webservice.py`.
2. **Multi-ticker portfolio pipeline** — `backtest_bt.py`, `backtest_ranking.py`, `optimize_regime.py`, `regime/` (standalone package). Operates on a wide CSV matrix of NASDAQ tickers sourced from Kaggle, builds portfolio weights via various ranking strategies, runs realistic backtests with the `bt` library. This is where recent development has been.

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

## Portfolio Backtesting Pipeline (recent work)

### Dataset: `Nasdaq3347/`

Per-ticker CSV dump from Kaggle **svaningelgem/nasdaq-daily-stock-prices** (3347 tickers, 1962–today with monthly updates). Schema: `ticker,date,open,high,low,close` — **no volume, no adj_close**. This is the constraint that shapes the rest of the pipeline: volume-based liquidity filters (ADV, dollar volume, VWAP) are not available, so Corwin-Schultz high/low-based spread estimation is used as a liquidity proxy instead.

After `min_history=504` filtering, effective usable date range is roughly **2013-01-29 → today** with ~1190 liquid tickers.

### Scripts

- **`backtest_bt.py`** — runs full bt-library backtests on the ranking strategies (rsi, scalogram, regime, equal-weight). Uses Corwin-Schultz spread filter, takes `--commission-bps`, `--top-n`, `--rebalance-days`, `--max-spread`. Saves `Output/backtest-bt-stats.txt` and `Output/backtest-bt-comparison.png`. **Must use `integer_positions=False`** in `bt.Backtest(...)` or the rebalance solver diverges on the full universe ("commission fn not smooth" error).

- **`backtest_ranking.py`** — simpler plain-numpy portfolio simulation (no bt library). Good for quick signal exploration before paying bt's overhead.

- **`optimize_regime.py`** — Optuna walk-forward hyperparameter search over `lookback`, `n_tail`, `top_n`, `divergence` metric, and scale subset flags. Splits history into rolling train(5y)/validate(3y)/step(2y) windows, reports best params per window and their stability. Slow: ~5–6 min per window × 6 windows = ~30 min for 50 trials each.

- **`regime/`** — differentiable regime optimizer using **JAX autograd + optax Adam**, packaged as a standalone module (`python -m regime --data-dir ./Nasdaq3347`). Optimizes the 13 CWT scale weights + softmax temperature (14 params total) with real analytic gradients. ~20–25 steps/sec after JIT warmup, converges in 500 steps (~25s) vs Optuna's 30+ min. Precomputes windowed recent/historical power means once in numpy (O(n_scales * n_blocks * n_tickers) only; cumsum trick + block-level subsampling at rebalance boundaries). Reports both train and held-out val Sharpe. Submodules: `data` (CSV loader + Corwin-Schultz), `cwt` (causal wavelet + windowing), `strategy` (JAX KL score + Sharpe-with-costs), `trainer` (Adam loop returning `TrainResult`), `reporting`, `cli`.

### Key findings from recent runs (2013-01-29 → 2025-12-11, 10bps commission, 20-day rebal)

- **Baseline regime (default params: lookback=120, n_tail=20, top_n=20, KL divergence):** CAGR 31.2%, total return 3234%, daily Sharpe **0.63**, max drawdown -59.5%. Daily kurtosis 750 — fat-tailed, dominated by a 150% best-day spike. High return but poor risk-adjusted.
- **Equal-weight baseline:** CAGR 27.6%, Sharpe **1.26**, max drawdown -30.8%. Better Sharpe despite lower return.
- **Optuna walk-forward best params vary wildly across windows** (lookback 144–246, n_tail 3–110, top_n 6–24, divergence bounces cosine/js/kl) — the signal is unstable. Val Sharpe in later windows reaches ~1.36–1.63.
- **JAX differentiable optimizer** (lookback=229, n_tail=106, 500 Adam steps, train 70% / val 30%): train Sharpe +1.22, val Sharpe **+0.80** out-of-sample. Learned scale weights collapse to long horizons: **126d (48%), 90d (18%), 26d (16%), 42d (15%)** — all short scales (≤21d) drop to <1%. Temperature drops to 0.005 (near-hard top-1 concentration).
- **The regime signal works on monthly-to-biannual horizons, not short-term noise.** This is consistent across both Optuna's best lookbacks and JAX's learned scale weights.

### Platform constraints

- Python 3.13.x on macOS **x86_64 (Intel)** Darwin 22.6.0
- **PyTorch unavailable** — wheels dropped after torch 2.2.x, which doesn't support Python 3.13+
- **JAX pinned to `<0.5`** (jax==0.4.38, jaxlib==0.4.38) — jaxlib 0.10+ dropped Intel macOS wheels
- `uv` is the package manager; `uv run python <script>` is how scripts are invoked

### Corwin-Schultz spread estimator

Implemented in `backtest_bt.py:corwin_schultz_spread(highs, lows)`. Returns a DataFrame of spread fractions per (date, ticker) derived from the 2-day vs 1-day high-low range ratio (Corwin & Schultz 2012). Used as the `spread_df` input to all weight builders; tickers where `spread_df.values[t] > max_spread` (default 2%) get `-inf` regime scores at date `t`, pushing them to zero weight via softmax.

## Key Dependencies

- **Data Fetching**: `pandas-datareader` (Yahoo Finance), `requests` (CryptoCompare API), `gdax` (deprecated crypto), `yfinance`
- **Data Processing**: `pandas`, `numpy`
- **ML/Signal Processing**: `scikit-learn`, `scipy` (wavelets, SVD, optimize)
- **Backtesting / portfolio**: `bt` (portfolio backtest library), `optuna` (TPE hyperparameter search)
- **Differentiable optimization**: `jax<0.5`, `jaxlib<0.5`, `optax>=0.2.5` (Intel macOS wheels require pinning)
- **Visualization**: `matplotlib`, `seaborn`
- **Web**: `aiohttp`, `aiohttp-swagger` (Swagger UI)
- **Persistence**: `joblib` (Security caching; also used by backtest scripts for per-ticker CSV loading)
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
├── webservice.py              # aiohttp REST API (legacy workflow)
├── evaluate_securities.py     # CLI entry point for single/batch analysis (legacy)
├── sort_securities.py         # Relevancy scoring for multiple securities (legacy)
├── backtest_bt.py             # Portfolio backtests via bt library (rsi/scalogram/regime/equal)
├── backtest_ranking.py        # Simpler numpy-only portfolio simulation
├── optimize_regime.py         # Optuna walk-forward hyperparameter search
├── regime/                    # JAX-autograd differentiable regime optimizer (standalone package)
│   ├── data.py                #   CSV loader + Corwin-Schultz spread
│   ├── cwt.py                 #   causal Ricker CWT + windowed power means
│   ├── strategy.py            #   regime KL score + portfolio Sharpe with costs
│   ├── trainer.py             #   Adam loop, TrainResult dataclass
│   ├── reporting.py           #   scale-weight printout + training plot
│   └── cli.py                 #   argparse main; `python -m regime`
├── rsi.py, plot_*.py          # Analysis and visualization scripts
├── pyproject.toml             # uv-managed Python deps (requires-python >=3.10)
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
├── Nasdaq3347/                # Kaggle svaningelgem per-ticker CSVs (OHLC, no volume)
├── DataStore/                # Cache directory for legacy Security objects
├── Output/                   # Generated plots and stats
└── s-and-p-500-companies/    # Symbol data (submodule)
```

