# ss_notebook

Runnable CLIs that live alongside the research notebooks in
`apps/notebook/notebooks/`. Each module is invoked through a
`uv run ss-*` console script defined in `apps/notebook/pyproject.toml`.

All scripts share `load_prices` from `scalogram.py` (Stooq loader by
default, Kaggle Nasdaq3347 slice via `--kaggle-dir`) and write
artifacts to `Output/` without opening an interactive window. The
scalogram/video CLIs are single-ticker; `ss-replay` accepts a pooled
train list plus an optional held-out val ticker.

## Modules

### `scalogram.py` → `ss-scalogram`

Static composite figure: price strip + causal CWT heatmap +
RSI/MACD/BBands strips, all aligned on the same time axis. Useful for
EDA — the heatmap is the same `causal_cwt` view the regime trainer
sees.

```bash
uv run ss-scalogram TSLA
uv run ss-scalogram --kaggle-dir ./Nasdaq3347 AAPL
```

### `scalogram_video.py` → `ss-scalogram-video`

Day-by-day mp4 of the causal scalogram. Three vertical guides mark
the current bar, the recent-window left edge (`t - n_tail + 1`),
and the historical-window left edge (`t - lookback + 1`) — i.e.
the exact slices the trainer's divergence score compares per
rebalance. Implementation precomputes the full scalogram once and
animates an `axvspan` "fog of war" rectangle masking the future;
`causal_cwt`'s strict causality makes that bit-identical to
recomputing per frame.

```bash
uv run ss-scalogram-video --start 2000-01-01 --start-after-lookback AAPL
```

### `replay/` → `ss-replay`

CWT-slice reconstruction probe. Split into focused submodules:
`cli` (argparse + figure I/O), `features` (CWT + lag-window builder
+ `TickerData` + `load_ticker`), `decoders` (OLS / tinygrad MLP /
tinygrad Conv1D / tinygrad masked-AE), `metrics` (R²/RMSE/max-|Δ|),
`plot` (3-panel figure), and `reconstruct` (`fit_and_evaluate`
orchestrator + a single-ticker `reconstruct_indicators` wrapper).

For each bar `t` we extract a trailing window of K columns of the
causal CWT (`coeffs` and `power`, 26 channels per lag with the
default 13 scales) and fit a decoder predicting RSI(7) /
MACD(12,26,9) / close at the same bar. R², RMSE, and max-|Δ| are
rendered onto the saved figure as right-aligned subplot titles;
the suptitle records the decoder, window size, and feature count.

Three knobs control how close reconstruction can approach the
information-theoretic ceiling of "full CWT is invertible":

- `--window-cols K` — trailing-window size (default 1, single
  column; K=64 captures roughly the indicator lookback).
- `--include-zscore-stats` — append the causal rolling μ, σ that
  `causal_cwt` strips out before convolution. Restores the price
  level the wavelet bandpass filter discards. Incompatible with
  `--decoder cnn` (the stats aren't lag-windowed and would break
  the CNN reshape).
- `--decoder {linear, mlp, cnn, masked-ae}` — `linear` = OLS via
  `np.linalg.lstsq`; `mlp` = small tinygrad MLP (Adam, hidden=128,
  layers=2, steps=2000 by default); `cnn` = 1-D Conv1D over the
  trailing-K window with shared weights across lags; `masked-ae` =
  self-supervised pretrain (mask `--mask-ratio` of input cells, MSE-
  reconstruct via a small MLP decoder, save backbone only — see
  `replay/README.md` for the SSL motivation). CNN/`masked-ae` require
  `--window-cols > cnn_kernel * cnn_layers`. CNN/`masked-ae` default
  to bf16 mixed precision and FiLM-grid streaming (re-samples grid
  cells per minibatch instead of materializing the replicated tile)
  for sub-16-GB GPU/iGPU targets; `--cnn-no-bf16` falls back to fp32
  on backends without bf16 (Metal Intel macOS), and
  `--cnn-microbatch-size` enables gradient accumulation for tighter
  VRAM budgets.

Two further knobs control the train/val split across tickers:

- `--train-tickers CSV` — pool extra tickers' valid feature rows
  with the primary ticker's into a single decoder fit. No figures
  are saved for these; only the primary ticker's reconstruction is
  plotted on the train side.
- `--val-ticker SYMBOL` — apply the pooled-train decoder zero-shot
  to a held-out ticker, report val R²/RMSE/max-|Δ|, and save
  `<output-dir>/<val-ticker>-replay-zeroshot-from-<train-tag>.png`.

Empirical headlines (AAPL 2013-01-29 → 2025-12-11, K=64,
`--include-zscore-stats`, `--decoder mlp`):

- **In-sample (single-ticker fit, no val):** price R² 0.9997, RSI
  R² 0.987, MACD R² 0.9999. With single-column linear OLS the same
  targets land at 0.04 / 0.21 / 0.15 — the ceiling is high but the
  bottleneck is real.
- **Zero-shot val on TSLA, AAPL-only train:** price R² −0.075,
  RSI R² −2.35, MACD R² 0.132. The "scalogram encodes everything"
  claim is in-sample memorization, not a portable encoder. With
  1666 features × ~3000 rows × one ticker, the MLP fits AAPL-
  specific feature distributions; TSLA's CWT magnitudes (and
  appended μ, σ) sit outside that distribution.

The decoder is fit globally over the full valid history of the
train pool (not walk-forward). This is an in-sample expressivity
probe — it answers "can the CWT slice encode the indicator at
all," not "could a model trained on past data forecast it OOS."
Use `--val-ticker` (and ideally `--train-tickers` to broaden the
distribution) when the question is generalization rather than
expressivity.

```bash
# In-sample expressivity (single ticker, no val):
uv run ss-replay AAPL                                       # OLS, K=1
uv run ss-replay AAPL --window-cols 64 --include-zscore-stats \
    --decoder mlp                                           # max in-sample fit

# Zero-shot generalization (pooled train, held-out val):
uv run ss-replay AAPL --val-ticker TSLA --window-cols 64 \
    --include-zscore-stats --decoder mlp                    # 1-train, 1-val
uv run ss-replay AAPL --train-tickers MSFT,GOOGL,AMZN \
    --val-ticker TSLA --window-cols 64 --include-zscore-stats \
    --decoder mlp                                           # 4-train, 1-val
```

### Cross-sectional IC scorer — moved to `apps/factor/`

The frozen-backbone scoring pipeline that used to live under
`ss_notebook.scoring/` now lives in its own app: `apps/factor/`,
import as `from factor import ...`. The `Backbone` dataclass + npz I/O
(`load_backbone`) moved one step further into `packages/features/`
(`ss_features`) so both apps can read the SSL pretrain output without
depending on each other. See `apps/factor/src/factor/__init__.py` for
the full public surface, including the deterministic-indicator
alternative (`IndicatorGridConfig`, `train_scorer_indicators`).

The backbone loader filters out per-target heads / FiLM weights /
target standardizers from the replay npz and verifies the remaining
backbone tensors are byte-identical across all per-target prefixes,
so the IC head sees a parameter-agnostic latent regardless of which
multi-head pretrain produced the npz.

### `replay_optuna.py` → `ss-replay-optuna`

Optuna TPE study over `replay.reconstruct_indicators` for the MLP
decoder. Maximizes mean R² across {price, RSI, MACD}. Search
space: `window_cols ∈ {1,4,8,16,32,64,96,128}`, `include_zscore_stats`,
`mlp_hidden ∈ {32,64,128,256,512}`, `mlp_layers ∈ [1,4]`,
`mlp_steps ∈ {500,1000,2000,4000}`.

Per-trial progress prints live via a callback; final markdown
table sorted by objective lists every trial with R² breakdown
and wall time.

```bash
uv run ss-replay-optuna AAPL --start 2013-01-29 --end 2025-12-11 \
    --n-trials 40
```

## Common flags

All four scripts accept:

- `--stooq-dir DIR` (default `./StooqData`) or `--kaggle-dir DIR`.
- `--start YYYY-MM-DD` / `--end YYYY-MM-DD` for date trimming.

`scalogram` and `scalogram_video` write to `Output/`; `replay` and
`replay_optuna` accept `--output-dir` (default `Output`). None of
these scripts call `plt.show()` — they save and exit.

## Design notes

### Why CNN ≠ "strictly worse MLP" for the replay decoder

RSI, MACD, and EMA are all sliding-window linear filters of the form
`y_t = Σ_k w_k · x_{t-k}`. The Conv1D decoder has shared weights
across lags, which is exactly that structure — it gets translation
equivariance over the lag axis for free, while the MLP has to learn
that lag-i and lag-(i+1) play similar roles independently. Layer-1
parameter counts at K=64 / 13 scales: MLP hidden=128 ≈ 213k params;
CNN kernel=5 hidden=64 ≈ 8k params. The CNN is forced to find the
filter structure rather than memorize, which should help cross-ticker
generalization on the indicator targets specifically.

The catch: `--include-zscore-stats` is incompatible with
`--decoder cnn` (features.py raises) — the appended μ, σ aren't
lag-windowed and would break the reshape. Without those stats, the
CWT has stripped the price level out and price R² is bounded near 0
no matter how good the decoder is. So the meaningful comparison is
**CNN vs MLP without zscore-stats** on RSI/MACD; if the question is
"how high can price R² climb in-sample," that's the MLP+zscore
regime and CNN can't enter it.

### Why the regime trainer's CWT input is raw close, not log-returns

A natural-feeling alternative: CWT of log-returns, since Stooq is
already split-/dividend-adjusted so log-returns has no artificial
discontinuities and is the textbook stationary transform. A
controlled walk-forward eval (Stooq 2010–2024, 20 trials/window,
kernel half-extent 3 fixed in both arms) showed raw close beats
log-returns on val Sharpe in every window — log-returns has higher
*train* and lower *val* Sharpe across the board, the canonical
overfitting signature.

Mechanism: `causal_cwt` does a rolling z-norm before the Ricker
convolution. With raw close that z-norm centers but doesn't detrend
within the window — a steadily rising price has a steadily rising
z-score, so long-scale Ricker power is essentially smoothed
momentum. With log-returns the z-norm gives "today's return divided
by recent return std," and long-scale power on that captures vol-
regime concentration, which is not a known cross-sectional return
predictor. The "dirtiness" of using raw close is load-bearing — it's
what bleeds the momentum factor into the divergence score the
trainer ranks on. Log-returns is the right input if the goal ever
shifts to vol forecasting or regime-break detection, but not for
the cross-sectional ranking this trainer does. The
`--use-log-returns` flag stays on `regime train` for that future
use case; default is raw close.

## Where to look next

- Notebooks in `apps/notebook/notebooks/`:
  - `causal_cwt_walkthrough.ipynb` — derivation of the `causal_cwt`
    machinery the CLIs all share.
  - `cwt_vision_multihead.ipynb` — Flax/JAX vision multi-head over
    scalogram tiles, including a Ridge linear-probe sanity baseline.
- Colab scripts in `apps/notebook/scripts/colab/`:
  - `train_cnn_multihead.sh` / `train_cnn_signreturn.sh|py` —
    supervised multi-head CNN pretrain (FiLM-conditioned RSI head).
  - `train_ssl.sh` — SSL pretrain via `--decoder masked-ae`
    (canonical "broad encoding" path; see `replay/README.md`).
  - `probe_ssl.sh` — supervised heads on a frozen SSL backbone, the
    diagnostic that reads off per-indicator R² from the SSL latent.
  - `zeroshot_eval.py` — uncond zero-shot indicator-decoding stats
    on a held-out ticker.
  - `attention_macd_vol.py` / `film_attention.py` — diagnostic
    attention plots over channels / FiLM bandwidth.
  - **Cross-sectional IC scorer scripts moved to `apps/factor/scripts/`**
    (`stage1_ic_scorer.py`, `ssl_ic_scorer.py`,
    `no_backbone_baseline.py`, `no_backbone_baseline_matched.py`) —
    frozen-backbone IC heads against forward returns. Lives with
    the `factor` app.
- Production trainer that consumes `ss_wavelets.causal_cwt` output:
  `apps/regime/src/regime/trainer.py`.
