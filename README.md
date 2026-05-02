# StockSurvey

A uv-workspace monorepo for trading-strategy research and live execution.

- **`apps/regime/`** — CWT-regime portfolio strategy. Optuna+vectorbt walk-forward search by default; JAX-Adam differentiable variant in `research/`. Persists JSON checkpoints, trades live via Alpaca.
- **`apps/v1/`** — legacy single-ticker workflow (`Security` → `Span` → `Decider` → `Evaluator` → `Plot`) plus the aiohttp web service. Parked.
- **`apps/notebook/`** — research notebooks **plus** runnable CLIs (`ss-scalogram`, `ss-scalogram-video`, `ss-replay`, `ss-replay-optuna`) and the `ss_notebook.scoring` subpackage (frozen-backbone IC head over the replay CNN). Colab scripts for SSL pretrain / frozen-backbone probe / IC scorer live in `apps/notebook/scripts/colab/`.
- **`packages/loaders/`** (`ss_loaders`) — Stooq daily archive (default), Kaggle CSV matrix, Yahoo, CryptoCompare, symbol lists.
- **`packages/indicators/`** (`ss_indicators`) — JAX matrix-form RSI/MACD/BBands/SMA/EMA, Corwin-Schultz spread, KL/JS/cosine/L2 divergences.
- **`packages/wavelets/`** (`ss_wavelets`) — strictly-causal Ricker CWT + windowed power means.
- **`packages/stream/`** (`ss_stream`) — point-in-time universe iterator over the Stooq archive (parquet event store with `active_at(date)` / `bars_between(...)`).
- **`packages/portfolio/`** (`ss_portfolio`) — JAX block-Sharpe with costs, CAGR/drawdown/Sortino/Calmar, water-fill weight cap.
- **`packages/plotting/`** (`ss_plotting`) — training curve, equity comparison, scalogram heatmap helpers.

See **`CLAUDE.md`** for deeper architecture notes, key historical findings, and platform constraints.

## Setup

This repo uses **uv workspaces** for Python package management and a **nix devShell** to provide the C-extension stack (numba/llvmlite) that has no PyPI wheel coverage on the current platform (Intel macOS, Python 3.13).

### Prerequisites

- [nix](https://nixos.org/download.html) with flakes enabled.
- That's it — nix provides Python 3.13, uv, and all C deps.

### First-time setup

```bash
nix develop                              # enters dev shell with python + uv + numba
uv venv --system-site-packages           # venv inherits nix's pre-built numba/llvmlite
uv sync --all-packages --inexact         # installs all workspace members + extras
```

The `--system-site-packages` flag is essential: it lets the workspace venv see the nix-baked `numba`, `llvmlite`, `numpy`, `scipy`, `pandas`, so uv doesn't try to install them from PyPI (which would fail — see "Why nix" below).

### Everyday workflow

You do **not** need to be in `nix develop` to run things — `uv run` works from any shell, because:
- `uv` is on your PATH globally (via `~/.nix-profile`).
- `.venv/bin/python` is a symlink into a `/nix/store/...` path that's immutable and always available.
- `numba` etc. are baked into that nix-store Python's `sys.path`.

```bash
uv run regime --help
uv run regime train --data-dir ./Nasdaq3347 --save-params Output/regime-v1.json
uv run python -m regime.research.optimize_regime --data-dir ./Nasdaq3347
uv run python -m v1.scripts.webservice
uv run pytest
```

### When you DO need `nix develop`

| Operation | Needs `nix develop`? |
|---|---|
| `uv run ...` everyday runs | no |
| `uv pip install <pure-Python pkg>` | no |
| Initial `uv venv --system-site-packages` | **yes** |
| `uv sync` after pyproject changes | **yes** |
| `uv add <pkg-with-C-extensions>` | **yes** (needs LLVM/clang/cmake) |

## Why nix

PyPI dropped Intel macOS x86_64 wheels for the heavy ML/scientific stack starting in 2024-2025 — JAX 0.5+, numba 0.65+, llvmlite 0.47+ all fall back to source builds that fail without a careful LLVM toolchain. nixpkgs still provides binary builds for `x86_64-darwin` (until nixpkgs 26.05; ~12 months from now), so the devShell is the cleanest stopgap.

The `[[tool.uv.dependency-metadata]]` override in the root `pyproject.toml` strips `numba` from `vectorbt`'s declared deps so uv doesn't try to install (and rebuild) it. nix provides numba, vectorbt picks it up at runtime.

## Caveats

- **GC root**: `.venv/bin/python` symlinks into `/nix/store/...-python3-3.13.12-env`. Running `nix-collect-garbage` will break the venv until you re-enter `nix develop`. Pin the env with `nix profile install nixpkgs#python313` if you collect frequently.
- **x86_64-darwin sunset**: nixpkgs 26.05 is the last release supporting Intel macOS. Plan for a migration to Apple Silicon, Linux, or Docker before then.
- **Cold JIT**: vectorbt's first call compiles numba kernels (~20s). Subsequent calls are fast.

## Key commands cheat-sheet

| Task | Command |
|---|---|
| Train regime model | `uv run regime train --data-dir ./StooqData --save-params Output/m.json` |
| Live trade (paper) | `uv run regime live --params Output/m.json --dry-run` |
| Live trade (real) | `ALPACA_BASE_URL=... uv run regime live --params Output/m.json --live` |
| bt-library backtest | `uv run python -m regime.research.backtest_bt --data-dir ./StooqData` |
| Optuna search (legacy) | `uv run python -m regime.research.optimize_regime --data-dir ./StooqData` |
| Static scalogram figure | `uv run ss-scalogram --stooq-dir ./StooqData TSLA` |
| Day-by-day scalogram mp4 | `uv run ss-scalogram-video --stooq-dir ./StooqData --start 2000-01-01 --start-after-lookback AAPL` |
| Replay reconstruction probe | `uv run ss-replay AAPL --val-ticker TSLA --window-cols 64 --include-zscore-stats --decoder cnn` |
| Stream ingest (parquet) | `uv run ss-stream ingest --src ./StooqData --dst ./Output/stream` |
| v1 web service | `uv run python -m v1.scripts.webservice` |
| All tests | `uv run pytest` |
| Just package tests | `uv run pytest packages/` |
