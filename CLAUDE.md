# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockSurvey is a uv-workspace monorepo containing trading-strategy research and live execution. Layout:

- `apps/regime/`   — CWT-regime portfolio strategy. Optuna+vectorbt walk-forward search by default; `research/optimize_adam.py` is the JAX-Adam differentiable variant — **parked** since `ss_indicators` was migrated to numpy (its `jax.value_and_grad` no longer flows through `get_divergence`). Persists a JSON checkpoint, trades live via Alpaca. Active development.
- `apps/v1/`       — legacy single-ticker workflow (`Security` → `Span` → `Decider` → `Evaluator` → `Plot`) plus the aiohttp web service. Parked.
- `apps/notebook/` — Jupyter playground + scalogram visualizer CLIs (`ss-scalogram`, `ss-scalogram-video`). Source in `src/ss_notebook/`. The CNN trainer was extracted to `apps/replay/`; the rank-IC scorer to `apps/factor/`.
- `apps/replay/`   — multi-head CNN trainer (`ss-replay`) that reconstructs technical indicators (RSI/MACD/vol/CCI/price) from causal CWT slices, with FiLM conditioning over (n, w) parameter grids. Tinygrad runtime. Produces backbone npz artifacts that `apps/factor/` consumes for downstream rank-IC scoring. `scripts/modal/train_cnn_multihead.py` is the Modal-T4 4-step harness (train + CSCO zero-shot + AAPL FiLM/uncond attention) for cloud runs against the baked-in 21-ticker Stooq subset under `data/stooq_phase2/`.
- `apps/factor/`   — cross-sectional rank-IC scorer (tinygrad). Two input paths share the same head + objective: (1) the SSL-pretrained CNN backbone produced by `ss-replay --decoder cnn`, loaded via `ss_features.load_backbone`; (2) `IndicatorGridConfig` — a wide flat stack of strided RSI/CCI grids, MACD over a fast-period grid, and realized vol over a window grid, fed through `identity_backbone(K=1, F=...)`. Public API: `from factor import ...`.
- `apps/relational/` — sector-relative excess-divergence research (CWT-based). Active.
- `packages/loaders/`    (`ss_loaders`)    — Kaggle CSV matrix, Stooq archive, Yahoo, CryptoCompare, symbol lists.
- `packages/features/`   (`ss_features`)   — shared primitives between apps that needed them: `TickerData` bundle, `load_prices` (Stooq/Kaggle/Yahoo), `realized_vol` / `log_returns`, plus `Backbone` dataclass + `load_backbone` (numpy-only npz I/O for the SSL-pretrained CNN backbone). Tinygrad runtime stays in `factor.backbone`.
- `packages/indicators/` (`ss_indicators`) — **numpy** matrix-form RSI/MACD/BBands/SMA/EMA + CCI, plus stride-w variants (`rsi_strided`, `cci_strided`) for FiLM-conditioned head training, plus Corwin-Schultz spread, KL/JS/cosine/L2 divergences, Fibonacci levels. Pure numpy after the JAX migration; no autograd path.
- `packages/wavelets/`   (`ss_wavelets`)   — causal Ricker CWT + windowed power means. `KERNEL_HALF_EXTENT=3` and `ALL_SCALES` exposed.
- `packages/stream/`     (`ss_stream`)     — point-in-time universe iterator over the Stooq archive (incremental loader for live trading).
- `packages/portfolio/`  (`ss_portfolio`)  — JAX block-Sharpe with costs, CAGR/drawdown/Sortino/Calmar, water-fill weight cap, masked softmax.
- `packages/plotting/`   (`ss_plotting`)   — training-curve, equity-comparison, scalogram-heatmap helpers.

`ss_indicators` is numpy; `ss_portfolio` is still JAX (block-Sharpe needs autograd in the regime trainer's loss). `apps/replay` and `apps/factor` use **tinygrad**. The legacy v1 indicators in `v1/util/indicators.py` are preserved untouched for the parked workflow but are not the canonical implementation; new code uses `ss_indicators`.

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

# Regime app — training + live trading (Stooq is the default --source)
uv run regime --help
uv run regime train --data-dir ./StooqData --save-params Output/regime-v1.json
uv run regime train --data-dir ./StooqData --use-log-returns   # opt-in: see "Key findings"
uv run regime live  --params Output/regime-v1.json --dry-run

# Regime research scripts (bt backtests, Optuna, Adam)
uv run python -m regime.research.backtest_bt        --data-dir ./Nasdaq3347 --top-n 20
uv run python -m regime.research.backtest_ranking   --data-dir ./Nasdaq3347 --top-n 10
uv run python -m regime.research.optimize_regime    --data-dir ./Nasdaq3347 --n-trials 50
uv run python -m regime.research.optimize_adam      --data-dir ./Nasdaq3347

# Scalogram tools (apps/notebook)
uv run ss-scalogram --stooq-dir ./StooqData TSLA                    # static composite figure
uv run ss-scalogram-video --stooq-dir ./StooqData --start 2000-01-01 \
       --start-after-lookback AAPL                                  # day-by-day mp4

# Legacy v1
uv run python -m v1.scripts.webservice              # aiohttp on :8080
uv run python -m v1.scripts.evaluate_securities AAPL
uv run python -m v1.scripts.sort_securities

# Notebooks
uv run jupyter notebook apps/notebook/notebooks/
```

Run `jupyter notebook` from the workspace root so the notebook's `data_dir = './Nasdaq3347'` resolves.

## Regime app architecture

`apps/regime/src/regime/`:
- `trainer.py`   — Optuna+vectorbt walk-forward search. `weights_regime` ranks by recent-vs-historical CWT-power divergence (KL/JS/cosine/L2); `weights_scalogram` ranks by direction−momentum×coherence. Both call `ss_wavelets.causal_cwt` + `precompute_windows`. CWT input is **raw close by default**; `_log_returns` is opt-in via `use_log_returns=True` (preserved as a flag — see "Key findings"). `DEFAULT_PER_WINDOW_MIN_HISTORY = KERNEL_HALF_EXTENT * max(LONG_SCALES) + LOOKBACK_SEARCH_MAX = 630` is the survivorship floor — derived, not hardcoded, so it follows kernel/lookback constants.
- `persist.py`   — JSON checkpoint serialization (no pickle). `Checkpoint` dataclass captures params + scales + hyperparams + universe + provenance + `use_log_returns` (so live inference matches train-time input).
- `inference.py` — pure forward pass that loads a `Checkpoint` and returns soft top-N target weights for the latest bar. Reads `checkpoint.use_log_returns` to mirror the trainer's input mode.
- `broker.py`    — `AlpacaBroker` adapter (account, positions, recent bars, build trades, submit orders). Reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` from env; defaults to paper-trading endpoint.
- `live.py`      — orchestration: load checkpoint → fetch bars → score universe → cap weights via `ss_portfolio.apply_position_cap` → diff vs current positions → submit. Risk rails: kill-switch file, data-freshness check, per-name cap, dry-run by default.
- `cli.py`       — argparse subcommands `train` and `live`. `--use-log-returns` flag exposed on `train` (off by default).
- `reporting.py` — thin adapters from `TrainResult` to `ss_plotting.plot_training_curves` / `print_scale_weights`.
- `research/`    — alternative search/eval (`backtest_bt.py`, `backtest_ranking.py`, `optimize_regime.py`, `optimize_adam.py` — JAX-Adam differentiable optimizer, **parked** since `ss_indicators` migrated to numpy).

## Key findings from past runs (2013-01-29 → 2025-12-11, 10bps commission, 20-day rebal)

- **Baseline regime (default params: lookback=120, n_tail=20, top_n=20, KL divergence):** CAGR 31.2%, total return 3234%, daily Sharpe **0.63**, max drawdown -59.5%. Daily kurtosis 750 — fat-tailed, dominated by a 150% best-day spike. High return but poor risk-adjusted.
- **Equal-weight baseline:** CAGR 27.6%, Sharpe **1.26**, max drawdown -30.8%. Better Sharpe despite lower return.
- **Optuna walk-forward best params vary wildly across windows** (lookback 144–246, n_tail 3–110, top_n 6–24, divergence bounces cosine/js/kl) — the signal is unstable. Val Sharpe in later windows reaches ~1.36–1.63.
- **JAX differentiable optimizer** (lookback=229, n_tail=106, 500 Adam steps, train 70% / val 30%): train Sharpe +1.22, val Sharpe **+0.80** out-of-sample. Learned scale weights collapse to long horizons: **126d (48%), 90d (18%), 26d (16%), 42d (15%)** — all short scales (≤21d) drop to <1%. Temperature drops to 0.005 (near-hard top-1 concentration).
- **The regime signal works on monthly-to-biannual horizons, not short-term noise.**
- **Log-returns CWT input degrades Sharpe (controlled walk-forward eval, Stooq 2010-2024, 20 trials/window, kernel half-extent 3 fixed in both arms):** raw close beats log-returns on val Sharpe in every window. Median val Sharpe `+0.15` (raw) vs `+0.03` (log-returns); mean `+0.07` vs `-0.29`; worst window `-0.41` vs `-1.06`. Per-window: log-returns has **higher train but lower val** Sharpe — overfitting signature. Theory: raw close bleeds price-level trend into long-scale wavelet power, embedding an implicit cross-sectional momentum factor; log-returns purifies trend out, leaving only "vol regime shift" which is not a known cross-sectional return predictor. Default is therefore raw close; `--use-log-returns` flag preserved for non-ranking research (vol forecasting, regime-break detection). Eval artifacts: `Output/regime-eval-{rawclose-kernel3,logreturns}.{log,json}`.

## Datasets

Two on-disk sources, picked via `regime train --source {stooq,kaggle} --data-dir DIR`:

- **`StooqData/`** (default for the trainer) — Stooq daily archive bulk dump, layout `daily/<country>/<exchange>/<bucket>/*.txt`. Has volume, split-/dividend-adjusted prices, and includes delisted tickers. Loaded via `ss_loaders.load_stooq_matrix` (with optional `cache_path` to skip the 12K-file scan after the first run).
- **`Nasdaq3347/`** — per-ticker CSV dump from Kaggle **svaningelgem/nasdaq-daily-stock-prices** (3347 tickers). Schema: `ticker,date,open,high,low,close` — **no volume, no adj_close**. After `min_history=504` filtering, effective usable date range is roughly **2013-01-29 → today** with ~1190 liquid tickers. Volume-based liquidity filters (ADV, dollar volume, VWAP) are not available regardless of source — `ss_indicators.corwin_schultz_spread` is the liquidity proxy used end-to-end.

## Platform constraints

- Python 3.13.x on macOS **x86_64 (Intel)** Darwin 22.6.0
- **PyTorch unavailable** — wheels dropped after torch 2.2.x, which doesn't support Python 3.13+. `apps/notebook/` uses tinygrad for trainers; `apps/regime` + `packages/portfolio` use JAX; `packages/indicators` + `packages/wavelets` are pure numpy.
- **JAX pinned to `<0.5`** (jax==0.4.38, jaxlib==0.4.38) — required by `apps/regime` + `packages/portfolio` (jaxlib 0.10+ dropped Intel macOS wheels). `packages/indicators` was migrated off JAX (numpy-only since the cci+rsi_strided refactor); `apps/notebook` doesn't depend on JAX directly. JAX remains in the env transitively via `ss_portfolio` whenever the regime trainer runs.
- **tinygrad** for `apps/notebook`. Default backend is auto-selected (Metal on macOS, CUDA on NVIDIA, AMD KFD on Linux+ROCm hardware, CPU fallback). bf16 mixed precision is default-on; `--cnn-no-bf16` disables it for backends without bf16 (Metal on Intel macOS) or fp32 reproducibility.
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
- Implement in `packages/indicators/src/ss_indicators/<name>.py` as a numpy function operating on axis-0 time (matrix-form: `(T, ...)` in → `(T, ...)` out). Add re-export to `ss_indicators/__init__.py` plus a test in `packages/indicators/tests/test_indicators.py`.
- For dense supervision of FiLM-conditioned heads, also add a `<name>_strided(prices, n, w)` 1-D variant alongside (see `rsi_strided` / `cci_strided` for the convention).
- If the indicator needs a scalar reduction for portfolio scoring, also add a divergence-style function next to `symmetric_kl_divergence`.

### Adding a new ranking strategy
- Add a `weights_<name>` builder in `apps/regime/src/regime/research/backtest_bt.py` and register in `WEIGHT_BUILDERS`. Use `ss_indicators` and `ss_wavelets` primitives.

### Live broker swap
- Subclass or replace `AlpacaBroker` in `apps/regime/src/regime/broker.py`. Keep the public surface (`get_account`, `get_positions`, `get_recent_bars`, `build_trades`, `submit_orders`) so `live.py` doesn't need to change.

### Visualizing what the trainer sees
- `ss-scalogram TSLA` renders a static composite figure (price strip + scalogram heatmap + RSI/MACD/BBands comparison strips) for one ticker.
- `ss-scalogram-video --start 2000-01-01 AAPL` renders an mp4 walking `t` forward one bar at a time. Each frame shows the causal CWT through day `t` plus three vertical guides marking the current bar, the recent-window left edge (`t - n_tail + 1`), and the historical-window left edge (`t - lookback + 1`). Useful for sanity-checking that the trainer's per-rebalance view matches expectations. Implementation precomputes the full scalogram once and animates only an `axvspan` "fog of war" rectangle masking the future — `causal_cwt` strict causality makes that bit-identical to recomputing per-frame.

## Important implementation notes

- **`ss_indicators` returns numpy** since the JAX migration. `ss_portfolio` JAX functions still return `jnp.ndarray` — cast with `np.asarray(...)` at numpy/pandas boundaries.
- **`ss_wavelets.causal_cwt` is numpy + scipy**, not JAX. It's a one-shot precompute (no autograd flows through wavelet coefficients). Cast to `jnp.asarray(...)` at the JAX boundary if a downstream `ss_portfolio` op needs it.
- **`KERNEL_HALF_EXTENT = 3`** in `ss_wavelets.cwt` truncates the Ricker at |t|=3 in scale-normalized time (under 0.3% energy loss vs |t|=4). Single source of truth — `regime.trainer.DEFAULT_PER_WINDOW_MIN_HISTORY` derives from it.
- **Per-day data dependency for `coeffs[scale, t]`** is `KERNEL_HALF_EXTENT * scale + lookback` bars. At trainer defaults that's `3 * 126 + 252 = 630` for the largest scale — also the universe-filter floor. Smaller scales (and shorter Optuna-chosen lookbacks) saturate sooner per element, but `weights_regime` requires *all* scales to be populated, so the largest scale gates everything.
- **Symbol prefix for crypto** (legacy v1): tickers starting with `coin` are treated as cryptocurrencies (`coinBTC`).
- **DataFrame columns**: lowercase `open`, `high`, `low`, `close`, `adj_close`, `volume`. Kaggle Nasdaq dataset omits `volume` and `adj_close`; Stooq has volume but no separate `adj_close` (close is already split-/dividend-adjusted).
- **Index**: all DataFrames must have a `DatetimeIndex` for resampling.
- **`integer_positions=False`** is required in `bt.Backtest(...)` or the rebalance solver diverges on the full universe ("commission fn not smooth" error).
- **ffmpeg compatibility on Intel macOS**: nix `ffmpeg_7+` links macOS-14 SDK symbols and crashes with a `_AVCaptureDeviceTypeContinuityCamera` dyld error on older hosts. Pin `ffmpeg_6` (works) or fall back to the bundled `imageio-ffmpeg` binary — `ss_notebook.scalogram_video._configure_ffmpeg` probes both.

## Known limitations and TODOs

- v1 webservice's `predict_rsi` endpoint returns NotImplemented.
- `v1/scripts/evaluate_securities.py` has a pre-existing brittle `from sklearn.externals.joblib import` that requires `v1.models.security` to be imported first (which monkey-patches `sklearn.externals.joblib`). Pre-restructure issue, not regressed by the move.
- v1 indicator implementations in `v1/util/indicators.py` are NOT the canonical version — they predate `ss_indicators` and have slightly different defaults (e.g. RSI n=14 vs ss_indicators' n=7). Use `ss_indicators` for new code.
