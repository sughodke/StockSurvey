# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Source of truth.** The `apps/docs` Material for MkDocs site
(`uv run ss-docs-serve`) is the canonical home for findings,
walk-forward results, workflows, and the active backlog. This file
stays lean: it is the operational reference Claude needs to navigate
the workspace and avoid known gotchas. When you discover a new finding
worth recording, write it to `apps/docs/docs/findings/` — see
"Recording findings in apps/docs" below for the run/experiment/arc
protocol. When you change a workspace convention or add a new
platform-level constraint, update this file.

## Project Overview

StockSurvey is a uv-workspace monorepo containing trading-strategy research and live execution. Layout:

- `apps/regime/`     — CWT-regime portfolio strategy. Optuna+vectorbt walk-forward search; persists a JSON checkpoint, trades live via Alpaca. Active.
- `apps/relational/` — sector-relative + fingerprint-space CWT scorers. Six scoreboard winners (empirical / gmm / analog / farthest / diversified / velocity) wired through `ss-relational live` for paper trading. Active.
- `apps/factor/`     — cross-sectional rank-IC scorer (tinygrad). Two input paths share the same head + objective: SSL-pretrained CNN backbone (`ss_features.load_backbone`) or `IndicatorGridConfig` (74-channel deterministic stack). Walk-forward eval is the primary OOS protocol.
- `apps/replay/`     — multi-head CNN trainer (`ss-replay`) reconstructing technical indicators (RSI/MACD/vol/CCI/price) from causal CWT slices, with FiLM conditioning. Tinygrad. Produces backbone npz artifacts that `apps/factor` consumes.
- `apps/gate/`       — aggregate drawdown forecaster as an EW-exposure gate. Numpy-only OLS predictor on trailing aggregate features (vol, return, trailing DD, breadth) over EW universe. v0 is `partial-OOS` (mean val Pearson r +0.26, mean alpha +0.07 within noise). First test of the prediction-problem pivot off cross-sectional return forecasting.
- `apps/pairs/`      — pair-spread mean reversion. Engle-Granger cointegration screening (per train slice) + classical z-score-crossing trade rules on dollar-neutral pair trades. Numpy + statsmodels. Second test of the prediction-problem pivot. v0 is `confirmed-null` per pre-reg (mean agg val Sh +0.099, 4/6 pos windows, dragged below floor by single dot-com-era window).
- `apps/vol/`        — implied vol surface predictor. Numpy OLS over 10 surface-shape features (skew / smile / multi-horizon IV-HV ratio / OI imbalance / VIX-spread / strike-spread) → forward 20d IV/RV gap, top-quantile gated short-vol. Third test of the prediction-problem pivot. v0 is `inconclusive` (mean per-cell-Sharpe alpha +0.089, 5/5 positive windows — strongest directional consistency in the pivot arc but mean just below +0.10 marginal floor). Reuses `ss_iv` (loaders + short-vol PnL).
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
- `packages/iv/`         (`ss_iv`)         — implied-vol loaders (gauss314 HF + DoltHub) + short-vol PnL accounting (`short_vol_pnl_panel`, `evaluate_short_vol`, `evaluate_universe_short_vol`). Promoted from `apps/relational/src/relational/{iv_data, short_vol}.py` 2026-05-10 when `apps/vol` became a second consumer.
- `packages/macro/`      (`ss_macro`)      — FRED CSV loaders (no auth, public `fredgraph.csv` endpoint, on-disk cache at `.macro-cache/`) + canonical 6-feature regime stack (fed_funds, slope_10y_3m, credit_baa, m2_yoy, real_yield_10y, vix). Built 2026-05-10 after the macro regime diagnostic showed 5/6 features predict pivot-arc window outcomes.

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
- **Keep `forward_log_returns` and any rank-IC target array at f64 internally.** `pearson_rank_ic`'s covariance numerator cancels catastrophically when val IC is small (~0.003); f32 rounding on `log_p` propagates to the cov sum at exactly that magnitude. The Tensor-boundary cast to f32 in `precompute_inputs` is the only place precision should drop. See `findings/factor-f32-precision-cancellation.md` for the regression-and-fix that motivated this rule.
- **Any portfolio-level "win" must clear the passive equal-weight Sharpe on the same universe + window before being treated as shippable.** Use `alpha = model_val_sharpe − passive_val_sharpe` as the load-bearing column in any new leaderboard row that claims live-tradeability — raw val Sharpe is largely market beta of the chosen universe. The 2026-05-10 EW benchmark (`apps/relational/scripts/equal_weight_benchmark.py`, `Output/equal-weight-benchmark.json`) reclassified the three previously "shippable" relational rows: Phase-2 analog cross_ticker alpha = **+0.067** (≈ noise), stooq_us_long Morlet alpha = **−0.133**, ex-Phase-2 cross_ticker alpha = **−0.334**. No current checkpoint clears its passive baseline. See `findings/passive-ew-benchmark.md`.
- **Long-short market-neutral construction does NOT rescue rank-IC heads at the factor-narrow IC scale (~+0.005 to +0.012 mean val IC).** The 2026-05-10 long-short eval (`apps/factor/scripts/long_short_eval.py`, `factor.objectives.{long_short_weights, block_sharpe_long_short}`) on the indicator linear head delivered LS val Sharpe **−0.067** vs LO **+0.278** (alpha −0.345, 2/6 positive windows), failing both pre-registered cuts. The "discarded short signal" hypothesis is falsified: friction (10 bps × 2× LS turnover ≈ 400 bps annualized) eats the +0.005 IC under either constructor. Don't re-test LS variants on a head with mean val IC < +0.02 — the underlying signal is below the friction floor. See `findings/factor-rankic-long-only-mismatch.md`. Next-experiment per `confirmed-null` rule: pivot to a different prediction problem, not a different constructor.
- **Don't train cross-sectional return heads with `block_sharpe` or `block_ir_vs_ew` losses when mean val IC is below ~+0.02.** The 2026-05-10 loss-pivot eval (`apps/factor/scripts/loss_pivot_eval.py`, `factor.objectives.block_ir_vs_ew`) showed both Sharpe-aligned losses *destroy* val Sharpe in the low-IC regime: rank-IC val Sh **+0.278** vs block_sharpe **−0.097** (delta −0.375) vs ir_vs_ew **−0.109** (delta −0.387). Mechanism: trainable temperature collapses to ~0.2 (near-argmax concentration), train IC drops 5×, and concentrated bets on 1-3 names amplify variance without amplifying mean at +0.005 IC. Rank-IC's scale-invariance was acting as inadvertent shrinkage toward EW — the right risk control in low-signal regimes. The Sharpe/IR losses remain in `objectives.py` for future use when IC is large enough to justify concentration. Three independent tests (passive-EW gate, long-short constructor, loss-pivot) now converge: at this IC scale the binding constraint is signal magnitude, not optimization. See `findings/factor-loss-pivot.md`. Next move per `confirmed-null` rule: pivot to a different prediction problem.
- **Pair-spread mean reversion (`apps/pairs`) is regime-conditional, not unconditionally shippable on factor-narrow.** The 2026-05-10 pairs classical v0 walk-forward (`apps/pairs/scripts/run_walkforward.py`) delivered mean agg val Sharpe **+0.099** across 6 windows (below the +0.20 fail floor pre-registered in `TODO/apps-pairs.md`), with 4/6 windows positive but a single catastrophic window (−1.23 in 2005-2007 bull market with dot-com-trained pairs). Mean ex-window-0 = +0.365 (would clear the marginal threshold) — pair trading worked in 2008-2017 mean-reverting markets, failed in trending bull markets. EG-passing-rate per window itself is a regime indicator (high in mean-reverting, low in trending). v2 follow-ups noted: regime gate on EG-passing-rate, stop-loss on widening spreads, sector restriction, ML predictor. **Don't deploy pair trading unconditionally on broad universes — needs regime gating.** Speed: pass `maxlag=1` to `statsmodels.coint()` to skip BIC lag-selection inner loop (~17× faster, 500ms→30ms per pair). See `findings/pairs-classical-v0.md`.
- **Vol surface signal lives in joint multivariate structure, not in any single feature — never pre-screen vol-surface features by univariate Pearson r.** The 2026-05-10 vol surface v0 walk-forward (`apps/vol/scripts/run_walkforward.py`) showed univariate Pearson r per feature ≤ +0.003 (the audit looked dead) but multivariate val Pearson r = **+0.12** (40× larger). The walk-forward wouldn't have run if we'd gated on univariate signal. This is the opposite of cross-sectional return prediction (where univariate ≈ multivariate IC at the +0.005 to +0.012 ceiling). Mean per-cell-Sharpe alpha +0.089 with 5/5 positive windows — `inconclusive` per pre-reg, partially refutes NO_OPTIONS.md's "the IV market efficiently incorporates the dislocation information" (true at ATM-IV level the prior arc tested; leaves residuals at the full-surface level the new feature class taps). Late windows (2022-12 → 2023-06) carry the signal: val r > +0.23. See `findings/vol-surface-v0.md`.
- **Predictions with non-zero multivariate signal but regime-conditional deployment performance need a regime filter, not a richer predictor.** The 2026-05-10 prediction-problem-pivot arc (gate-drawdown-v0, pairs-classical-v0, vol-surface-v0) tested three orthogonal prediction problems and got the same pattern: real multivariate signal (mean val Pearson r in [+0.12, +0.26]), mean alpha within ±0.05 of the marginal threshold, regime-conditional structure (works in some windows, fails in others). The relational arc's `transition-triggered rebal` (NO_OPTIONS.md Phase 9) discovered the same operational rule independently — signal-triggered timing of an existing scorer beat scheduled cadence by +0.21 Sharpe. **Default v1 architecture for any partial-OOS predictor: a regime classifier on top that gates the underlying predictor by recent-window characteristics. Schedule the trigger, not the trade.** See `findings/prediction-problem-pivot-arc.md` and `findings/relational-arc-synthesis.md`.
- **Use the fingerprint embedding for selection and timing; do not use it for hedging.** NO_OPTIONS.md Phases 9/11/12 tested three word2vec-analog uses of the relational fingerprint embedding: cluster transitions as rebal triggers (Sharpe +0.21 over baseline — works clearly), velocity magnitude as a scorer (+0.06 lift — works), and nearest-neighbor pair as a per-pick hedge (Sharpe **−1.12 / max DD −99%** — fails badly). The mechanism: empirical's score is excess-divergence vs cluster aggregate, so the "nearest behavioral peer" is by construction another stock from the same cluster — shorting it doubles the bet rather than hedging it. The embedding has predictive content for *positional dynamics* but negative content for hedge selection. See `findings/relational-arc-synthesis.md`.
- **Factor signal-quality from a 6-window walk-forward is too lagged to outperform a real-time VIX-median meta-gate.** The 2026-05-12 sizing-input v1 retroactive eval (`apps/gate/scripts/macro_meta_gate_eval.py` factor extension, joined to `Output/sizing-input-rank_ic-windows.npz` via most-recent-OOS factor val_start lookup) tested factor-only / VIX-AND-factor / VIX-OR-factor meta-gates against the VIX-only baseline (+0.215 z lift). All factor arms underperformed: factor-only **−0.143 z** (worse than no-gate), VIX AND factor −0.034 z, VIX OR factor +0.106 z (best factor arm, still 0.109 below VIX-only). Mechanism: factor walk-forward val periods average ~2 years; the available factor signal-quality read at pivot val_start is 0–3y stale and reflects the *previous* regime's dispersion, not the current one. Two clean inversions: factor sq +1.128 (calm pre-GFC read) said SUSPEND at 2008-02 when VIX correctly said DEPLOY (gate w1 +0.321, pairs w1 +0.870 alphas lost); factor sq +3.468 (GFC-aware read) said DEPLOY at 2011-03 when VIX said SUSPEND. **Before retraining factor at finer windowing, do the cheap per-bar emission v2** (lag bounded by 20d instead of 780d, ~2-hour wiring fix using existing `signal_quality_per_val_bar` arrays). The factor signal IS the right *concept* (cross-sectional dispersion regime) at the wrong temporal grain. See `findings/factor-sizing-input-v1.md`.
- **Re-purposing apps/factor as a sizing-input model: rank_ic stays as the training loss; mse_alpha calibrates magnitudes but adds zero downstream value when the consumer is a rank-based dispersion stat.** The 2026-05-12 sizing-input v0 head-to-head (`apps/factor/scripts/modal/sizing_input_eval.py`, 6-window factor-narrow walkforward) trained the indicator linear head two ways — `pearson_rank_ic` (scale-invariant) vs `masked_mse` on per-bar cross-sectional alpha targets (scale-calibrated, val MSE-alpha **52× smaller**). On the pre-registered sizing-input emission (per-val-bar top-decile-minus-bottom-decile predicted alpha) **both arms produced identical downstream-useful properties**: Spearman ρ between per-window signal-quality mean and val Sharpe = **+0.486 exactly** in both arms (above the +0.40 marginal threshold), pooled lag-1 autocorrelation +0.82 (mse_alpha) vs +0.91 (rank_ic). Mechanism: top-decile − bottom-decile is rank-invariant by construction — rescaling scores doesn't move it. Don't run a magnitude-aware sizing-input experiment unless the downstream gate explicitly consumes magnitudes (sum-of-top-decile-predicted-alpha, fraction-above-cost-threshold). **The rank_ic head's signal-quality emission already has the temporal stability + val-Sharpe correlation a macro meta-gate needs** — go straight to v1 wiring (factor signal-quality at val_start as a second gate input alongside macro state in `apps/gate/scripts/macro_meta_gate_eval.py`) without re-litigating the loss axis. See `findings/factor-sizing-input-v0.md`.
- **Macro features are a real but graduated regime signal — use them as a continuous deployment scaler, NOT as direct predictor inputs.** The 2026-05-10 diagnostic (`apps/gate/scripts/macro_regime_diagnostic.py`, n=17 across gate/pairs/vol windows) showed 5/6 macro features predict per-app-z-scored alpha (Pearson r `[+0.34, +0.49]`) with a clean VIX-median split (above median: 5/8 wins; below: 1/9). The 2026-05-11 v1 follow-up tested two integration arms: **(v1a) adding macro features to gate's predictor stack made it WORSE** (mean alpha +0.067 → −0.086, COVID window catastrophe alpha −0.78 — train-regime overfit because macro distributions are non-stationary across train/val); **(v1b) binary VIX-above-1y-rolling-median meta-gate** delivered pooled per-app-z-scored lift +0.215 (real but modest) but raw-units lift only −0.010 (binary threshold too crude — vol w4's +0.134 alpha got suspended at VIX 22.8 vs 25.6 median). Don't add macro to within-app feature stacks. Don't fold into `apps/regime` (per-ticker CWT-portfolio is the wrong aggregation level). Default v1: continuous macro-percentile deployment scaler at the meta-level — `apps/gate/scripts/macro_meta_gate_eval.py` is the retroactive harness; v2 should test continuous + composite-macro (Chicago Fed NFCI etc.) before committing to a deployment architecture. See `findings/macro-regime-diagnostic.md` (v1 results section).
- **At the meta-allocator level, the binding constraint is the action menu, not the algorithm.** The 2026-05-12 CFR Phase 1 walk-forward (`apps/cfr/scripts/modal/run_phase1.py`, 6 windows on stooq_us_long, tabular CFR over 16 universe-agnostic actions × 9 vol-dispersion infosets) cleared the pre-registered PASS cut against trailing-best-greedy (mean +0.609 Sharpe lift, 6/6 windows positive vs +0.10 threshold) but **tied naive uniform mix within noise** (Δ +0.002) and undershot passive EW (alpha −0.093, 1/6 positive). The pattern is a clean Cover universal-portfolio result: counterfactual regret matching converges to the uniform-mix limit when no infoset has consistently positive cumulative regret — which is correct no-regret behavior over a menu of universe-agnostic top-K factor exposures (momentum / reversal / vol-rank) at our universe + horizon. See `findings/cfr-phase1.md`.
- **Tabular menu enrichment makes CFR worse, not better — at meta-allocator scale the binding constraint is regret-table sample density.** The 2026-05-12 CFR Phase 2a (28-action menu, +4 documented-alpha modes: 12-1 momentum, 12-month low-vol, trailing-Sharpe top-K, return/MDD top-K) and Phase 2b (31-action menu, +real SEC 13F-HR consensus mode from new `packages/edgar` covering 14 curated funds since 2013) walk-forwards both **failed the menu-enrichment cut**: CFR Sharpe stayed at ~+0.58 across all three phases while naive uniform mix rose monotonically from +0.591 → +0.632 → +0.652. CFR's lift over naive uniform went +0.002 → −0.059 → −0.069. Mechanism: the Cesa-Bianchi & Lugosi O(√(log n)/√T) regret bound predicts a *worse* expected regret as menu size grows at fixed T (here ~6,000 train rebals). Naive uniform is sample-density-free; tabular CFR's per-cell estimator is sample-density-bound. **One genuine positive in Phase 2b**: window 5 (val 2020-2023, the only window with full 13F coverage in train+val) lifted CFR alpha −0.271 → +0.006 — a +0.277 within-window gain from the 13F mode. The 13F signal IS real where coverage exists; tabular-CFR can't extract it because (a) only 1.5 windows have full coverage so the relevant regret-table cells are under-visited and (b) `Top13FConsensusMode` returns cash for pre-2013 bars, contaminating early-window regret-table cells with phantom-cash entries that aren't deduped (gross > 0). **Phase 3 must replace tabular CFR with deep CFR** (`regret_net(state, action_emb) → R` MLP that shares statistical strength across actions) over a learned multi-modal encoder. Don't add more actions to the tabular table; replace the table. See `findings/cfr-phase2.md`.
- **Deep CFR lifts the meta-allocator by ~+0.02 mean Sharpe over tabular — real but architecturally bounded; the binding constraint at this universe + horizon is signal availability, not representation.** The 2026-05-12 CFR Phase 3 walk-forward (`apps/cfr/scripts/modal/run_phase3.py`, tinygrad MLP `regret_net(state→R)` over a 10-feature continuous state vector — 6 universe-internal + 4 FRED macro features — replacing the 9-cell tabular infoset) cleared **part** of the Phase 2 sample-density problem (window 2 alpha flipped −0.111 → +0.127, the cleanest deep-architecture win) but the cumulative Phase 1 → 3 lift was only **+0.021 mean Sharpe** (+0.593 → +0.614, far short of +0.15 PASS floor). CFR vs naive uniform improved from −0.069 (Phase 2b) to −0.038 (Phase 3) — best of any enriched-menu phase, still negative. CFR alpha vs passive EW improved from −0.093 (Phase 1) to −0.071 (32% reduction, still negative). **Across 5 phase variants the architecture-side variance is bounded at ~±0.02, the same magnitude as the no-regret O(√(log n)/√T) theoretical bound at our T=6,000 / n=31.** Cover universal-portfolio guarantees no-regret vs the best fixed-mix-in-hindsight, which on 25-year US equity *is* passive EW; we cannot beat it without genuinely regime-switching alpha that exists in some regime and a state representation that can isolate that regime. Phase 4 should change the prediction problem (different universe — sector ETFs / different horizon — daily rebal / hybrid with macro v1b VIX gate / composite-regime action menu), NOT the meta-allocator's representation. **Common failure pattern with the prediction-problem-pivot apps:** all 5 phases of CFR (and gate / pairs / vol) post their only positive-alpha windows in the GFC + post-COVID regime cluster — the macro-stress regime the macro-regime-diagnostic identified. **Tabular CFR on this universe is at the architectural ceiling.** See `findings/cfr-phase3.md`.
- **Universe shift is the lever, not architecture or menu enrichment — multi-asset deep CFR PASSES where equity-only never does.** The 2026-05-12 CFR Phase 4 sweep tested 4 orthogonal axes on top of Phase 3 deep CFR: 4a (per-bar VIX-above-1y-rolling-median gate), 4b (sector ETF universe), 4c (5-day rebal), and 4d (13-asset multi-asset basket = 9 SPDR sector ETFs + TLT/IEF + GLD/DBC). **4a destroyed Phase 3** (mean CFR +0.614 → +0.383 because bar-level gating suspends 57% of bars and CFR loses compounding more than it saves; window-level gating remains viable, bar-level isn't). **4c traded alpha for friction** (5-day rebal at 10 bps round-trip = 5%/yr friction tax that the equity universe doesn't earn back; notable side observation: training stability dramatically improved with 4× more SGD samples — first phase with finite final loss across all windows). **4b sector-ETF universe partial-OOS** (mean CFR +0.78, first positive mean alpha vs EW at +0.015, 3/5 positive windows). **4d 13-asset multi-asset PASSES** — mean CFR **+0.861** (Phase 1 + 0.27, well over +0.15 PASS floor), CFR vs naive **+0.101** (clears +0.10 cut), mean alpha vs passive EW **+0.056** (positive for the first time across all 9 phase variants), 3/5 positive alpha windows. **Why multi-asset works where equity-only doesn't:** (a) per-action variance — gold ↔ stocks correlation ~0, bonds ↔ stocks ~0; mode portfolios on 13 cross-asset are genuinely orthogonal (vs ~85% correlated on 312 equities), so regret signal per (state, action) is structurally larger; (b) real regime-conditional alpha — cross-asset literature has 60+ years of evidence that stocks/bonds/commodities have a regime-switching optimum (Bridgewater All Weather thesis). **The arc-level lesson:** the meta-allocator is **prediction-problem bound, not representation bound**. Iterating on architecture hit a +0.02 ceiling across 5 variants; iterating on universe broke through it. Phase 4d is the first deployable CFR result; needs `ss-cfr live` integration with the four risk rails before paper trading. Honest caveats: alpha +0.056 is positive but below the +0.15 paper-trade threshold from `findings/passive-ew-benchmark.md`; only 5 windows; w2 (2016-2019, "everything works passive" era) posts catastrophic −0.508 single-window alpha. See `findings/cfr-phase4.md`.

## Common workflows

See `apps/docs/docs/workflows.md` (rendered at `/workflows/` on the docs site).
Covers: adding a new indicator, adding a new ranking strategy, swapping the
broker, authoring relational checkpoints, scalogram visualizers.

## Recording findings in apps/docs — after every run, experiment, and arc

`apps/docs` is append-only and source of truth. Every empirical run lands
somewhere on the docs site before the conversation ends. Three escalating
levels:

### After every run (one experiment, one row of result)

1. Append a row to `apps/docs/docs/leaderboard.md` master table:
   `date | app | experiment | universe | windowing | metric | train | val | delta (val − train) | verdict | artifact`.
   Reuse existing universe / windowing tags where possible (Phase-2,
   factor-narrow, stooq_us_long, regime-3w-optuna, phase-2 split, etc.) —
   they're defined in the same page's "Operating conditions" section.
2. Pick the right verdict label from the predefined vocabulary:
   `confirmed-OOS`, `reversed-OOS`, `partial-OOS`, `confirmed-null`,
   `diagnostic`, `pending`. Don't invent new labels.
3. Append-only — do not rewrite prior rows. If a new run supersedes an
   earlier one, add a new row and reference the prior in the notes
   column. The leaderboard captures history, not the latest opinion.

### After every experiment (a hypothesis tested with one or more arms)

Everything above for each arm, plus:

1. If the result has prose worth keeping (mechanism, surprise, follow-up
   implications) beyond what fits in the row's notes column, write or
   extend `apps/docs/docs/findings/<topic>.md`. The page should:
   - Lead with the operational rule extracted (the "what to do
     differently" takeaway).
   - Give the eval setup so the row is reproducible.
   - Present per-window numbers in tabular form.
   - Briefly explain mechanism (*why* the result happened, not just
     *what* it was).
   - Close with a "Master walk-forward log" pointer linking the
     corresponding leaderboard row(s) and verdict label
     (`[verdict-label-here](../leaderboard.md#verdict-labels)`).
2. Move artifact figures from `Output/` into
   `apps/docs/docs/findings/images/` and embed them with captions that
   point at the *insight*, not the description.
3. Cross-link the operational rule wherever it's referenced in other
   docs (other findings, notes, TODO, app overview pages) so a reader
   hovering the claim is one click from the eval that grounds it.
4. Add the new page to the `Findings` nav in `apps/docs/mkdocs.yml`
   *and* to the listing in `apps/docs/docs/findings/index.md`.

### After every arc (multi-experiment investigation that terminates)

Everything above for each experiment, plus:

1. If the arc produced a singular operational rule, write a closing
   prose page in `apps/docs/docs/findings/` and cross-link it to the
   per-experiment findings pages.
2. If the arc produced a durable *concept* (a framing, not a result —
   e.g. "strategy as a dot product", "search vs optimize"), add or
   extend a section in `apps/docs/docs/notes.md`.
3. Update the "Operational rules extracted from findings" subsection
   above with the actionable rule(s) the arc established. The
   leaderboard row holds the evidence; this list holds the imperative.
4. If the arc closes a TODO entry, mark it as superseded with a pointer
   to the closing finding, or remove it if it was a single-line item.

### Propose what's next — every finding gates the next experiment

Recording the finding is half the loop; proposing the next experiment
is the other half. After the leaderboard row + (optional) findings
page lands, decide what the verdict implies and write it down before
moving on. If you don't, the work decays into a stack of one-off rows
with no forward momentum.

Default question shape per verdict label:

| Verdict | Default-next question |
|---|---|
| `confirmed-OOS` | Where does it stop working? Run the adjacent test that would either confirm scope (different universe / horizon / wavelet / rebal cadence) or break it. A rule that holds in only one regime is *Phase-2-specific* until proven otherwise — see `apps/docs/docs/findings/relational-universe-shift.md`. |
| `reversed-OOS` | What killed val? Distinguish three failure modes — overfit (DOF too high → regularize, shrink the bundle, or wider universe), regime-specific (split the train/val on a different axis and re-run), pipeline bug (compare arms row-by-row) — each implies a different next experiment. |
| `partial-OOS` | Stratify the windows. The "partial" usually means one or two windows carry the signal. What feature distinguishes the surviving windows from the failing ones (vol regime, dispersion, sector concentration)? If a feature splits them, you have a regime gate; if not, the verdict is closer to `reversed-OOS` than the partial label implies. |
| `confirmed-null` | Stop testing variations of the same lever — find an orthogonal one. Different prediction problem (return → vol → drawdown → IV-vs-realized), different feature class (indicators → CWT → pair-spread), different operational use (predictor → sizing → gate). The vol-forecast arc on `apps/factor` is the canonical example: four nulls in a row before the pivot. |
| `diagnostic` | Turn it into a falsifiable hypothesis. A diagnostic without a follow-up test is a curiosity. What experiment would tell us *why* the diagnostic looks the way it does? |
| `pending` | Land the verdict first; rows that sit on `pending` longer than a week become invisible. |

Where the next-experiment hypothesis lands:

1. If a `TODO/<workstream>.md` page already covers the line of work,
   **update it** to reflect what was just learned — add the new
   sub-question, mark superseded sub-bullets, refresh the priority
   order.
2. If the work is new and orthogonal to existing TODOs, **create**
   `TODO/<topic>.md` with the verdict → next-experiment chain stated
   up front. Add it to the `TODO` nav in `apps/docs/mkdocs.yml` and
   the listing in `apps/docs/docs/TODO/index.md`.
3. The hypothesis must be **falsifiable** and have a **test design**
   (universe, windowing, expected delta vs baseline, what counts as a
   positive result). "Try it on the wider universe" is not a
   hypothesis; "if the val Sharpe still loses by ≥0.10 on
   `stooq_us_long`, the effect is mega-cap-specific; otherwise the
   train/val split was the binding constraint" is.

### Executing a TODO end-to-end

When you pick a `TODO/<topic>.md` item and start implementing it,
follow this loop so the TODO converts cleanly into a leaderboard row
+ findings page without intermediate state going missing:

1. **Implement against the TODO's stated test design.** If the TODO is
   thin, push back and refine it before coding — every step below
   assumes a falsifiable hypothesis with universe/windowing/expected
   delta already pinned.
2. **Smoke test locally first** (`--max-tickers 30 --n-steps 50` style;
   see "Compute placement"). Fast feedback on the wiring before
   spending Modal time. **Commit when the smoke test passes** — a
   green scaffold is a checkpoint worth preserving even if the full
   eval later reverses the verdict.
3. **Decide whether the full eval is a leaderboard row.** Not every
   TODO produces one — refactors, plumbing, debug investigations
   don't. The test for "is this a leaderboard row?" is: does the
   eval produce a train/val number against a named universe and
   windowing in [Operating conditions](apps/docs/docs/leaderboard.md#operating-conditions)?
   If yes, run on Modal per the heavy-work rule. If no (e.g. you're
   just verifying a code path), local is fine and skip the
   leaderboard step.
4. **Run the Modal eval, commit the driver/script before kicking off.**
   The remote run is the load-bearing artifact; the local commit is
   the reproducibility anchor.
5. **When the Modal run finishes, convert the TODO to learnings in
   one pass:**
   - Append the row to `apps/docs/docs/leaderboard.md` with the
     verdict label.
   - Write or extend the `apps/docs/docs/findings/<topic>.md` page
     per the "After every experiment" protocol above.
   - Update `TODO/<topic>.md`: if the experiment closes the
     workstream, mark it superseded with a pointer to the finding
     and consider removing the TODO entry from `mkdocs.yml` nav. If
     it spawns a follow-up, rewrite the page to reflect the new
     verdict → next-experiment chain.
   - Commit the docs changes as a single follow-up commit referencing
     the implementation commit.
6. **Do not leave a Modal run unrecorded.** A finished eval without
   a leaderboard row is the worst state — the result decays from
   memory and the next researcher re-runs it.

### Mechanics

- `uv run ss-docs-serve` live-previews at http://127.0.0.1:8000. The
  Material livereload watcher does not always pick up newly-created
  pages or image directories — if a new page 404s, restart the server
  rather than touching `mkdocs.yml` to force a rebuild.
- `uv run ss-docs-build` for a clean static rebuild before committing.
- Verdict labels in prose should link to
  `leaderboard.md#verdict-labels`. Concept mentions should link to
  the relevant `findings/*.md`.
- Image filenames live alongside their parent section
  (`apps/docs/docs/{findings,apps}/images/`). Use descriptive names
  (`replay-zeroshot-tsla-from-19.png`, not the bare `Output/` filename
  with ticker pools embedded).
- Don't put result-bearing prose in `CLAUDE.md`. This file is for
  operational rules, conventions, and gotchas; eval numbers live in
  leaderboard rows and findings pages.

### Where to find things

- **Leaderboard** — `apps/docs/docs/leaderboard.md` (one row per run,
  append-only).
- **Findings** — `apps/docs/docs/findings/` (one page per experiment
  or arc with prose worth keeping; `findings/index.md` lists them).
- **Notes** — `apps/docs/docs/notes.md` (durable concepts, not
  results).
- **TODO** — `apps/docs/docs/TODO/` (one page per workstream).
- **Per-app overviews** — `apps/docs/docs/apps/{regime,relational,factor,replay,notebook}.md`
  (figure-heavy gallery + cross-links into findings).

## Known limitations

- v1 webservice's `predict_rsi` endpoint returns NotImplemented.
- `v1/scripts/evaluate_securities.py` has a pre-existing brittle
  `from sklearn.externals.joblib import` that requires `v1.models.security`
  to be imported first (which monkey-patches `sklearn.externals.joblib`).
- v1 indicator implementations in `v1/util/indicators.py` are NOT canonical —
  they predate `ss_indicators` and have slightly different defaults
  (e.g. RSI n=14 vs ss_indicators' n=7). Use `ss_indicators` for new code.
