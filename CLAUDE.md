# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockSurvey is a uv-workspace monorepo containing trading-strategy research and live execution. Layout:

- `apps/regime/`   — differentiable CWT-regime portfolio strategy. Trains a JAX-Adam model offline, persists a JSON checkpoint, and trades it live via Alpaca. Contains a `research/` sub-package with bt-library backtests and Optuna walk-forward search. Active development.
- `apps/v1/`       — legacy single-ticker workflow (`Security` → `Span` → `Decider` → `Evaluator` → `Plot`) plus the aiohttp web service. Parked. Imports JAX only when calling `ss_indicators` for the first time.
- `apps/notebook/` — Jupyter notebooks for cross-cutting research (e.g. CWT-vision multi-head decoder).
- `packages/loaders/`    (`ss_loaders`)    — Kaggle CSV matrix, Yahoo, CryptoCompare, symbol lists.
- `packages/indicators/` (`ss_indicators`) — JAX matrix-form RSI/MACD/BBands/SMA/EMA, Corwin-Schultz spread, symmetric-KL divergence, Fibonacci levels.
- `packages/wavelets/`   (`ss_wavelets`)   — causal Ricker CWT + windowed power means.
- `packages/portfolio/`  (`ss_portfolio`)  — JAX block-Sharpe with costs, CAGR/drawdown/Sortino/Calmar, water-fill weight cap, masked softmax.
- `packages/plotting/`   (`ss_plotting`)   — training-curve, equity-comparison, scalogram-heatmap helpers.

All packages target JAX (matrix form). The legacy v1 indicators in `v1/util/indicators.py` are preserved untouched for the parked workflow but are not the canonical implementation; new code uses `ss_indicators`.

## Workspace conventions

- **uv workspace**: root `pyproject.toml` declares `apps/*` and `packages/*` as members. `uv sync --all-packages --inexact` installs the whole graph editable. Each member has its own `pyproject.toml` listing its workspace and external deps.
- **Naming**: distribution names use hyphens (`ss-indicators`); import names use underscores (`ss_indicators`). The notebook app's dist name is `stocksurvey-notebook` to avoid colliding with Jupyter's `notebook` package; its import name is `ss_notebook`.
- **App import names**: `regime`, `v1`, `ss_notebook` (the apps don't get an `ss_` prefix on their import name; only packages do).
- **Source layout**: every member uses `src/<importname>/` with `[tool.hatch.build.targets.wheel] packages = ["src/<importname>"]`.
- **Cross-package deps**: declare via `[tool.uv.sources] ss-foo = { workspace = true }` in the consumer's pyproject.
- **Nix devShell**: `flake.nix` at repo root provides Python 3.13 with numba/llvmlite/numpy/scipy/pandas pre-built (PyPI has no Intel-macOS / Py3.13 wheels). One-time setup: `nix develop` → `uv venv --system-site-packages` → `uv sync --all-packages --inexact`. After that, `uv run ...` works from any shell — see README.md.
- **vectorbt**: pinned to `>=1.0,<2.0` in `regime[research]`. The root pyproject's `[[tool.uv.dependency-metadata]]` override strips `numba` from vectorbt's declared deps so uv doesn't try to reinstall the nix-provided one.

## Running things

```bash
uv sync --all-packages                              # install everything editable

# Regime app — training + live trading
uv run regime --help
uv run regime train --data-dir ./Nasdaq3347 --save-params Output/regime-v1.json --save
uv run regime live  --params Output/regime-v1.json --dry-run

# Regime research scripts (bt backtests, Optuna)
uv run python -m regime.research.backtest_bt        --data-dir ./Nasdaq3347 --top-n 20
uv run python -m regime.research.backtest_ranking   --data-dir ./Nasdaq3347 --top-n 10
uv run python -m regime.research.optimize_regime    --data-dir ./Nasdaq3347 --n-trials 50

# Legacy v1
uv run python -m v1.scripts.webservice              # aiohttp on :8080
uv run python -m v1.scripts.evaluate_securities AAPL
uv run python -m v1.scripts.sort_securities

# Notebook
uv run jupyter notebook apps/notebook/notebooks/
```

Run `jupyter notebook` from the workspace root so the notebook's `data_dir = './Nasdaq3347'` resolves.

## Regime app architecture

`apps/regime/src/regime/`:
- `trainer.py`   — JAX-Adam loop returning a `TrainResult`. Calls `ss_wavelets.causal_cwt` + `precompute_windows` once up-front, then iterates a small `(n_blocks, n_tickers)` JIT'd forward pass through `ss_indicators.symmetric_kl_divergence` + `ss_portfolio.block_sharpe_with_costs`.
- `persist.py`   — JSON checkpoint serialization (no pickle). `Checkpoint` dataclass captures params + scales + hyperparams + universe + provenance.
- `inference.py` — pure forward pass that loads a `Checkpoint` and returns soft top-N target weights for the latest bar.
- `broker.py`    — `AlpacaBroker` adapter (account, positions, recent bars, build trades, submit orders). Reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` from env; defaults to paper-trading endpoint.
- `live.py`      — orchestration: load checkpoint → fetch bars → score universe → cap weights via `ss_portfolio.apply_position_cap` → diff vs current positions → submit. Risk rails: kill-switch file, data-freshness check, per-name cap, dry-run by default.
- `cli.py`       — argparse subcommands `train` and `live`.
- `reporting.py` — thin adapters from `TrainResult` to `ss_plotting.plot_training_curves` / `print_scale_weights`.
- `research/`    — pre-trainer experiments (`backtest_bt.py`, `backtest_ranking.py`, `optimize_regime.py`).

## Key findings from past runs (2013-01-29 → 2025-12-11, 10bps commission, 20-day rebal)

- **Baseline regime (default params: lookback=120, n_tail=20, top_n=20, KL divergence):** CAGR 31.2%, total return 3234%, daily Sharpe **0.63**, max drawdown -59.5%. Daily kurtosis 750 — fat-tailed, dominated by a 150% best-day spike. High return but poor risk-adjusted.
- **Equal-weight baseline:** CAGR 27.6%, Sharpe **1.26**, max drawdown -30.8%. Better Sharpe despite lower return.
- **Optuna walk-forward best params vary wildly across windows** (lookback 144–246, n_tail 3–110, top_n 6–24, divergence bounces cosine/js/kl) — the signal is unstable. Val Sharpe in later windows reaches ~1.36–1.63.
- **JAX differentiable optimizer** (lookback=229, n_tail=106, 500 Adam steps, train 70% / val 30%): train Sharpe +1.22, val Sharpe **+0.80** out-of-sample. Learned scale weights collapse to long horizons: **126d (48%), 90d (18%), 26d (16%), 42d (15%)** — all short scales (≤21d) drop to <1%. Temperature drops to 0.005 (near-hard top-1 concentration).
- **The regime signal works on monthly-to-biannual horizons, not short-term noise.**

## Dataset: `Nasdaq3347/`

Per-ticker CSV dump from Kaggle **svaningelgem/nasdaq-daily-stock-prices** (3347 tickers). Schema: `ticker,date,open,high,low,close` — **no volume, no adj_close**. This shapes the rest of the pipeline: volume-based liquidity filters (ADV, dollar volume, VWAP) are not available, so `ss_indicators.corwin_schultz_spread` is the liquidity proxy. After `min_history=504` filtering, effective usable date range is roughly **2013-01-29 → today** with ~1190 liquid tickers.

## Platform constraints

- Python 3.13.x on macOS **x86_64 (Intel)** Darwin 22.6.0
- **PyTorch unavailable** — wheels dropped after torch 2.2.x, which doesn't support Python 3.13+. Notebooks use Flax/JAX instead.
- **JAX pinned to `<0.5`** (jax==0.4.38, jaxlib==0.4.38) — jaxlib 0.10+ dropped Intel macOS wheels.
- `uv` is the package manager.

## Live-trading risk rails

`regime live` enforces four checks before submitting orders, each aborting with a clear reason rather than silently coercing:
1. **Kill-switch file** (`~/.regime-killswitch` by default) — operator can halt without touching cron.
2. **Data freshness** — abort if latest bar is older than `--max-data-age-days` (default 3).
3. **Per-name cap** (`--max-position`, default 0.25) via `ss_portfolio.apply_position_cap` (water-fill).
4. **Dry-run by default** — `--live` is opt-in.

`ALPACA_BASE_URL` defaults to the paper endpoint; set explicitly to `https://api.alpaca.markets` for real money.

## Caching and data management

- **Legacy v1 cache**: `DataStore/` (joblib pickles), 1-day staleness, `coin{ticker}` prefix for crypto.
- **Regime checkpoints**: `Output/regime-v*.json`, plain JSON (portable, inspectable, safe to load).
- **Backtest outputs**: `Output/backtest-bt-{stats.txt,comparison.png}`, `Output/optimize-regime-walkforward.png`.

## Common workflows

### Adding a new indicator
- Implement in `packages/indicators/src/ss_indicators/<name>.py` as a JAX function operating on axis-0 time. Add re-export to `ss_indicators/__init__.py`.
- If the indicator needs a scalar reduction for portfolio scoring, also add a divergence-style function next to `symmetric_kl_divergence`.

### Adding a new ranking strategy
- Add a `weights_<name>` builder in `apps/regime/src/regime/research/backtest_bt.py` and register in `WEIGHT_BUILDERS`. Use `ss_indicators` and `ss_wavelets` primitives.

### Live broker swap
- Subclass or replace `AlpacaBroker` in `apps/regime/src/regime/broker.py`. Keep the public surface (`get_account`, `get_positions`, `get_recent_bars`, `build_trades`, `submit_orders`) so `live.py` doesn't need to change.

## Important implementation notes

- **JAX returns**: `ss_indicators` and `ss_portfolio` JAX functions return `jnp.ndarray`. Cast with `np.asarray(...)` at numpy/pandas boundaries.
- **Symbol prefix for crypto** (legacy v1): tickers starting with `coin` are treated as cryptocurrencies (`coinBTC`).
- **DataFrame columns**: lowercase `open`, `high`, `low`, `close`, `adj_close`, `volume`. Kaggle Nasdaq dataset omits `volume` and `adj_close`.
- **Index**: all DataFrames must have a `DatetimeIndex` for resampling.
- **`integer_positions=False`** is required in `bt.Backtest(...)` or the rebalance solver diverges on the full universe ("commission fn not smooth" error).

## Known limitations and TODOs

- v1 webservice's `predict_rsi` endpoint returns NotImplemented.
- `v1/scripts/evaluate_securities.py` has a pre-existing brittle `from sklearn.externals.joblib import` that requires `v1.models.security` to be imported first (which monkey-patches `sklearn.externals.joblib`). Pre-restructure issue, not regressed by the move.
- v1 indicator implementations in `v1/util/indicators.py` are NOT the canonical version — they predate `ss_indicators` and have slightly different defaults (e.g. RSI n=14 vs ss_indicators' n=7). Use `ss_indicators` for new code.
