# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Source of truth.** The `apps/docs` Material for MkDocs site (`uv run ss-docs-serve`)
is the canonical home for findings, walk-forward results, workflows, and the
active backlog. This file stays lean: it is the operational reference Claude
needs to navigate the workspace and avoid known gotchas. When you discover a new
finding worth recording, write it to `apps/docs/docs/findings/`. When you change
a workspace convention or add a new platform-level constraint, update this file.

## Project Overview

StockSurvey is a uv-workspace monorepo containing trading-strategy research and live execution. Layout:

- `apps/regime/`     — CWT-regime portfolio strategy. Optuna+vectorbt walk-forward search; persists a JSON checkpoint, trades live via Alpaca. Active.
- `apps/relational/` — sector-relative + fingerprint-space CWT scorers. Six scoreboard winners (empirical / gmm / analog / farthest / diversified / velocity) wired through `ss-relational live` for paper trading. Active.
- `apps/factor/`     — cross-sectional rank-IC scorer (tinygrad). Two input paths share the same head + objective: SSL-pretrained CNN backbone (`ss_features.load_backbone`) or `IndicatorGridConfig` (74-channel deterministic stack). Walk-forward eval is the primary OOS protocol.
- `apps/replay/`     — multi-head CNN trainer (`ss-replay`) reconstructing technical indicators (RSI/MACD/vol/CCI/price) from causal CWT slices, with FiLM conditioning. Tinygrad. Produces backbone npz artifacts that `apps/factor` consumes.
- `apps/notebook/`   — Jupyter playground + scalogram visualizer CLIs (`ss-scalogram`, `ss-scalogram-video`).
- `apps/docs/`       — Material for MkDocs site; canonical home for findings, workflows, TODO.
- `apps/v1/`         — legacy single-ticker workflow + aiohttp web service. Parked.
- `packages/loaders/`    (`ss_loaders`)    — Kaggle CSV matrix, Stooq archive, Yahoo, CryptoCompare, symbol lists.
- `packages/features/`   (`ss_features`)   — `TickerData`, `load_prices`, `realized_vol` / `log_returns`, `Backbone` + `load_backbone` (numpy npz I/O). Tinygrad runtime stays in `factor.backbone`.
- `packages/indicators/` (`ss_indicators`) — **numpy** matrix-form RSI/MACD/BBands/SMA/EMA + CCI, stride-w variants (`rsi_strided`, `cci_strided`), Corwin-Schultz spread, KL/JS/cosine/L2 divergences, Fibonacci levels, `rolling_pearson_corr`. No autograd path.
- `packages/wavelets/`   (`ss_wavelets`)   — causal Ricker CWT + windowed power means. `KERNEL_HALF_EXTENT=3` and `ALL_SCALES` exposed.
- `packages/stream/`     (`ss_stream`)     — point-in-time universe iterator over the Stooq archive.
- `packages/portfolio/`  (`ss_portfolio`)  — numpy block-Sharpe with costs, CAGR/drawdown/Sortino/Calmar, water-fill weight cap, masked softmax. Houses `ss_portfolio.broker` (canonical Alpaca adapter, `[alpaca]` extra) — both `regime live` and `ss-relational live` share it.
- `packages/plotting/`   (`ss_plotting`)   — training-curve, equity-comparison, scalogram-heatmap helpers.

`ss_indicators`, `ss_portfolio`, and `ss_wavelets` are all pure numpy. `apps/replay` and `apps/factor` use **tinygrad** (the only autograd runtime in the repo). The legacy v1 indicators in `v1/util/indicators.py` are preserved untouched but are not the canonical implementation.

## Workspace conventions

- **uv workspace**: root `pyproject.toml` declares `apps/*` and `packages/*` as members. `uv sync --all-packages --inexact` installs the whole graph editable.
- **Naming**: distribution names use hyphens (`ss-indicators`); import names use underscores (`ss_indicators`). The notebook app's dist name is `stocksurvey-notebook` to avoid colliding with Jupyter's `notebook` package; its import name is `ss_notebook`.
- **App import names**: `regime`, `v1`, `ss_notebook`, `ss_docs` (apps don't get an `ss_` prefix on their import name unless they would otherwise collide; only packages do uniformly).
- **Source layout**: every member uses `src/<importname>/` with `[tool.hatch.build.targets.wheel] packages = ["src/<importname>"]`.
- **Cross-package deps**: declare via `[tool.uv.sources] ss-foo = { workspace = true }` in the consumer's pyproject.
- **Nix devShell**: `flake.nix` provides Python 3.13 with numba/llvmlite/numpy/scipy/pandas pre-built (PyPI has no Intel-macOS / Py3.13 wheels). One-time setup: `nix develop` → `uv venv --system-site-packages` → `uv sync --all-packages --inexact`. After that, `uv run ...` works from any shell.
- **vectorbt**: pinned to `>=1.0,<2.0` in `regime[research]`. The root pyproject's `[[tool.uv.dependency-metadata]]` strips `numba` from vectorbt's declared deps so uv doesn't try to reinstall the nix-provided one.

## Running things

```bash
uv sync --all-packages --inexact                    # install everything editable

# Regime (training + live)
uv run regime train --data-dir ./StooqData --save-params Output/regime-v1.json
uv run regime live  --params Output/regime-v1.json --dry-run

# Relational (six canonical scorers, paper trading)
uv run python apps/relational/scripts/build_canonical_checkpoints.py
uv run ss-relational live --params Output/relational-empirical.json --dry-run

# Scalogram visualizers
uv run ss-scalogram --stooq-dir ./StooqData TSLA
uv run ss-scalogram-video --stooq-dir ./StooqData --start 2000-01-01 --start-after-lookback AAPL

# Docs site
uv run ss-docs-serve   # http://127.0.0.1:8000

# Tests
uv run pytest
```

## App architectures

### Regime (`apps/regime/src/regime/`)
- `trainer.py`   — Optuna+vectorbt walk-forward search. `weights_regime` ranks by recent-vs-historical CWT-power divergence (KL/JS/cosine/L2); `weights_scalogram` ranks by direction−momentum×coherence. Both call `ss_wavelets.causal_cwt` + `precompute_windows`. CWT input is **raw close by default** — see findings page on log-returns. `DEFAULT_PER_WINDOW_MIN_HISTORY = KERNEL_HALF_EXTENT * max(LONG_SCALES) + LOOKBACK_SEARCH_MAX = 630` is the survivorship floor (derived).
- `persist.py`   — JSON checkpoint serialization. `Checkpoint` captures Optuna-search hyperparams + scales + universe + provenance + `use_log_returns`. Legacy adam-mode fields silently dropped on load.
- `inference.py` — pure forward pass. Dispatches on `checkpoint.strategy` ∈ `{regime, scalogram, rsi}`; mirrors `checkpoint.use_log_returns`.
- `broker.py`    — re-export shim for `ss_portfolio.broker.AlpacaBroker`.
- `live.py`      — orchestration with the four risk rails (see "Live-trading risk rails" below).
- `cli.py`       — argparse subcommands `train` and `live`.
- `research/`    — alternative search/eval (`backtest_bt.py`, `backtest_ranking.py`, `optimize_regime.py`).

### Relational (`apps/relational/src/relational/`)
- `persist.py`    — `RelationalCheckpoint` JSON I/O. Strategy ∈ `{empirical, gmm, analog, farthest, diversified, velocity}`. Strategy-specific knobs in `strategy_kwargs`; common fields are `universe`, `lookback`, `top_n`, `scales`, `rebal_days`, `max_spread`, `commission_bps`. Falsified strategies (pair-trade, NN-pair) are intentionally not exposed.
- `inference.py`  — `target_weights(prices, highs, lows, checkpoint)` dispatches on `checkpoint.strategy`, takes the latest-bar row, applies a Corwin-Schultz spread gate, renormalizes. Validator asserts `len(prices) >= KERNEL_HALF_EXTENT*max(scales) + lookback + 1` so callers fail fast.
- `live.py`       — orchestration mirroring `regime/live.py`. Default kill-switch is `~/.relational-killswitch`. Bar fetch derives count from wavelet support: `lookback + KERNEL_HALF_EXTENT*max(scales) + bar_buffer_days`.
- `cli.py`        — `ss-relational live --params <ckpt.json> [--dry-run|--live]` plus `ss-relational head-to-head` research subcommand.
- Scoring modules — `empirical_sectors.py`, `empirical_sectors_gmm.py`, `analog_knn.py`, `farthest.py`, `diversify.py`, `regime_velocity.py` (each exports a `weights_*` builder); `fingerprints.py`, `transitions.py`, `cluster_tracking.py`, `scalogram_cache.py` are shared primitives.
- `scripts/build_canonical_checkpoints.py` — writes `Output/relational-{strategy}.json` for the six scoreboard winners. Phase-2 strategies pin to `PHASE2_TICKERS` (21 mega-cap names); velocity pins to `apps/notebook/data/stooq_us_long`. **Phase-2 wins are mega-cap-specific** (see findings page on universe shift).

### Factor (`apps/factor/src/factor/`)
- `backbone.py`           — tinygrad runtime: `identity_backbone`, `compute_input_stats`, `apply_backbone(_pytree)`, `backbone_to_pytree`. Re-exports `Backbone` + `load_backbone` from `ss_features`.
- `data.py`               — `AlignedTickers` + `align_tickers` (strict date-range intersection — single short-history ticker shrinks the common axis, so callers should pre-filter via `min_history_bars`) + `forward_log_returns`.
- `objectives.py`         — `pearson_rank_ic` (training; Pearson on raw scores), `block_sharpe` (eval-only).
- `scorers.py`            — Linear / MLP head builders + apply.
- `indicator_features.py` — `IndicatorGridConfig` (74 channels at default: 30 RSI + 16 CCI + 6 vol + 18 MACD + 4 coherence) + `build_indicator_features` + `load_ticker_indicators` + `make_indicator_backbone` + `train_scorer_indicators` + `train_scorer_indicators_walkforward`. Default warmup is 820 bars (CCI cell `n=40, w=21` dominates).
- `train.py`              — `train_scorer` (Stage 1 head-only + optional Stage 2 backbone fine-tune), `precompute_inputs`, `predict`.
- `train_walkforward.py`  — `train_scorer_walkforward` + `WalkForwardWindow` + `WalkForwardResult`.
- `scripts/modal/train_indicator.py` — Modal-T4 cloud harness (`train_grid` + `walkforward` entrypoints).

## Datasets

Two on-disk sources, picked via `regime train --source {stooq,kaggle} --data-dir DIR`:

- **`StooqData/`** (default for the trainer) — Stooq daily archive bulk dump, layout `daily/<country>/<exchange>/<bucket>/*.txt`. Has volume, split-/dividend-adjusted prices, includes delisted tickers. Loaded via `ss_loaders.load_stooq_matrix` (with optional `cache_path` to skip the 12K-file scan after the first run).
- **`Nasdaq3347/`** — per-ticker CSV dump from Kaggle **svaningelgem/nasdaq-daily-stock-prices** (3347 tickers). Schema: `ticker,date,open,high,low,close` — **no volume, no adj_close**. After `min_history=504` filtering, effective usable date range is roughly **2013-01-29 → today** with ~1190 liquid tickers. Volume-based liquidity filters are not available regardless of source — `ss_indicators.corwin_schultz_spread` is the liquidity proxy used end-to-end.

## Compute placement — default to Modal for heavy work

**Default rule: anything that takes more than ~2 minutes wall on the local 8-core Intel Mac, or that risks the laptop crashing mid-run, runs on Modal.**

- Modal is invoked via **`uvx modal run apps/<app>/scripts/modal/<entrypoint>.py [...flags]`**. Use `uvx`, not `uv run` — Modal isn't pinned in the project venv. The `local_entrypoint` runs in that isolated env, so it must NOT import project-venv-only deps (no `pandas`, `ss_loaders`, etc.). For runs that need project-venv data prep, do prep in a separate `uv run python <prep>.py` step that pickles the input, then have the `local_entrypoint` read the pickle as raw bytes and ship via Modal RPC. Working pattern: `apps/factor/scripts/modal/{prep_universe_pivot_data.py, universe_pivot_vol_arm.py}`.
- **Inside the remote function**: parallelize feature-build with `mp.Pool(os.cpu_count() or 4)` — Modal T4 instances expose ~24 cores; sequential per-ticker loops can time out. Set `cpu=8` (soft) and `timeout=2*60*60`. Pin tinygrad to CUDA via `os.environ['CUDA']='1'` *before* tinygrad imports, then assert `Device.DEFAULT == 'CUDA'` to fail fast on CPU fallback.
- **What "heavy" means**: walkforward at >300 tickers, any feature build over the full StooqData/ archive, any tinygrad training >2k steps, any image pipeline, any optuna sweep with >20 trials. Single-ticker scripts (scalogram, replay, evaluate) stay local.
- **What stays local**: data prep (`prep_*.py` scripts), interactive Jupyter, smoke tests with `--max-tickers 30 --n-steps 50`, anything single-shot under ~2 min.
- **Reference Modal entrypoints**: `apps/factor/scripts/modal/{train_indicator.py, train_ssl_walkforward.py, universe_pivot_vol_arm.py}`, `apps/replay/scripts/modal/train_cnn_multihead.py`. Image base is `nvidia/cuda:12.4.0-devel-ubuntu22.04` + uv + `add_local_dir` of repo (with StooqData/, Output/, Nasdaq3347/, .git/ ignored). One-time auth: `uvx modal token new`.
- Modal results stream back via `dict[str, bytes]` from the remote function and get written to local `Output/`. Don't try to read large files from a Modal Volume mid-run.

## Platform constraints

- Python 3.13.x on macOS **x86_64 (Intel)** Darwin 22.6.0.
- **PyTorch unavailable** — wheels dropped after torch 2.2.x, which doesn't support Python 3.13+. `apps/replay` and `apps/factor` use **tinygrad** for trainers; the rest of the workspace is pure numpy.
- **No JAX in the workspace.** The `apps/replay/scripts/colab/*` scripts still import JAX, but those run on Colab/GPU instances that provide JAX out of band — not in this venv.
- **tinygrad** for `apps/replay` + `apps/factor`. Default backend auto-selected (Metal on macOS, CUDA on NVIDIA, AMD KFD on Linux+ROCm, CPU fallback). bf16 mixed precision default-on; `--cnn-no-bf16` disables it.
- `uv` is the package manager.

## Live-trading risk rails

Both `regime live` and `ss-relational live` enforce four checks before submitting orders, each aborting with a clear reason:

1. **Kill-switch file** (`~/.regime-killswitch` / `~/.relational-killswitch`).
2. **Data freshness** — abort if latest bar is older than `--max-data-age-days` (default 3).
3. **Per-name cap** (`--max-position`, default 0.25) via `ss_portfolio.apply_position_cap` (water-fill). Distributes among **nonzero-weight** names only — zero-weighted names are not re-introduced. All-zero input short-circuits to all-zero output.
4. **Dry-run by default** — `--live` is opt-in.

Two implicit prerequisites enforced earlier:
- **Wavelet support** — bar fetch covers `KERNEL_HALF_EXTENT*max(scales) + lookback + bar_buffer_days` trading bars. Without this the latest-bar CWT runs against zero-padded history. `_validate_inputs` raises if a caller bypasses the broker.
- **Order rejections surfaced** — `submit_orders` returns `(order_ids, rejections)`. Per-symbol failures (most commonly: non-fractionable rejecting fractional qty) are logged + captured into `LiveRunResult.rejected_orders`.

`ALPACA_BASE_URL` defaults to the paper endpoint. Credentials and broker code are gated by the `ss-portfolio[alpaca]` optional extra.

## Important implementation notes

- **`ss_indicators`, `ss_portfolio`, `ss_wavelets` all return plain numpy.** No JAX anywhere in the workspace; if you need autograd, use tinygrad in an `apps/factor`-style trainer.
- **`KERNEL_HALF_EXTENT = 3`** in `ss_wavelets.cwt` truncates the Ricker at |t|=3 in scale-normalized time (under 0.3% energy loss vs |t|=4). Single source of truth — `regime.trainer.DEFAULT_PER_WINDOW_MIN_HISTORY` derives from it.
- **Per-day data dependency for `coeffs[scale, t]`** is `KERNEL_HALF_EXTENT * scale + lookback` bars. At trainer defaults that's `3 * 126 + 252 = 630` for the largest scale — also the universe-filter floor.
- **Symbol prefix for crypto** (legacy v1): tickers starting with `coin` are treated as cryptocurrencies (`coinBTC`).
- **DataFrame columns**: lowercase `open`, `high`, `low`, `close`, `adj_close`, `volume`. Kaggle Nasdaq dataset omits `volume` and `adj_close`; Stooq has volume but no separate `adj_close` (close is already split-/dividend-adjusted).
- **Index**: all DataFrames must have a `DatetimeIndex` for resampling.
- **`integer_positions=False`** is required in `bt.Backtest(...)` or the rebalance solver diverges on the full universe.
- **ffmpeg compatibility on Intel macOS**: nix `ffmpeg_7+` links macOS-14 SDK symbols and crashes with a `_AVCaptureDeviceTypeContinuityCamera` dyld error on older hosts. Pin `ffmpeg_6` or fall back to the bundled `imageio-ffmpeg` binary — `ss_notebook.scalogram_video._configure_ffmpeg` probes both.

### Operational rules extracted from findings

- **Regime trainer CWT input: raw close (default), not log-returns.** `--use-log-returns` flag preserved for non-ranking research only.
- **Phase-2 relational wins are mega-cap-specific.** `Output/relational-analog.json` is Phase-2-restricted; do not transfer the val Sharpe to wider universes without fresh OOS validation.
- **Do not pin `compression=` on canonical relational checkpoints.** DWT-L1 won single-arm bt but failed walk-forward across all four distance scorers.
- **Replay CNN: `--compress dwt --compress-levels 1 --compress-wavelet haar` is safe** for SSL backbone training (~4× input shrink, no quality loss on RSI/CCI/vol). MACD remains pathological in either arm.
- **The regime signal works on monthly-to-biannual horizons.** Short-scale weights collapse to <1% under JAX-Adam; don't expect short-term predictiveness.

## Common workflows

See `apps/docs/docs/workflows.md` (rendered at `/workflows/` on the docs site).
Covers: adding a new indicator, adding a new ranking strategy, swapping the
broker, authoring relational checkpoints, scalogram visualizers.

## TODO and findings

- **TODO** — `apps/docs/docs/TODO/` (one page per workstream).
- **Findings** — `apps/docs/docs/findings/` (regime baselines, log-returns vs
  raw close, factor indicator-IC baseline, replay DWT compression, relational
  universe-shift, relational DWT-L1 OOS failure).

## Known limitations

- v1 webservice's `predict_rsi` endpoint returns NotImplemented.
- `v1/scripts/evaluate_securities.py` has a pre-existing brittle
  `from sklearn.externals.joblib import` that requires `v1.models.security`
  to be imported first (which monkey-patches `sklearn.externals.joblib`).
- v1 indicator implementations in `v1/util/indicators.py` are NOT canonical —
  they predate `ss_indicators` and have slightly different defaults
  (e.g. RSI n=14 vs ss_indicators' n=7). Use `ss_indicators` for new code.
