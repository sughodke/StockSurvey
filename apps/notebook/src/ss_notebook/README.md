# ss_notebook

Runnable CLIs that live alongside the research notebooks in
`apps/notebook/notebooks/`. Each module is invoked through a
`uv run ss-*` console script defined in `apps/notebook/pyproject.toml`.

All scripts are single-ticker, share `load_prices` from `scalogram.py`
(Stooq loader by default, Kaggle Nasdaq3347 slice via `--kaggle-dir`),
and write artifacts to `Output/` without opening an interactive
window.

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

### `replay.py` → `ss-replay`

CWT-slice reconstruction probe. For each bar `t` we extract a
trailing window of K columns of the causal CWT (`coeffs` and
`power`, 26 channels per lag with the default 13 scales) and fit a
decoder predicting RSI(7) / MACD(12,26,9) / close at the same bar.
R², RMSE, and max-|Δ| are rendered onto the saved figure as
right-aligned subplot titles; the suptitle records the decoder,
window size, and feature count.

Three knobs control how close reconstruction can approach the
information-theoretic ceiling of "full CWT is invertible":

- `--window-cols K` — trailing-window size (default 1, single
  column; K=64 captures roughly the indicator lookback).
- `--include-zscore-stats` — append the causal rolling μ, σ that
  `causal_cwt` strips out before convolution. Restores the price
  level the wavelet bandpass filter discards. Incompatible with
  `--decoder cnn` (the stats aren't lag-windowed and would break
  the CNN reshape).
- `--decoder {linear, mlp, cnn}` — `linear` = OLS via
  `np.linalg.lstsq`; `mlp` = small JAX MLP (Adam, hidden=128,
  layers=2, steps=2000 by default); `cnn` = 1-D Conv1D over the
  trailing-K window with shared weights across lags. CNN requires
  `--window-cols > cnn_kernel * cnn_layers`.

Empirical headline (AAPL 2013-01-29 → 2025-12-11, K=64
+ `--include-zscore-stats` + `--decoder mlp`): price R² 0.9997,
RSI R² 0.987, MACD R² 0.9999. With single-column linear OLS the
same targets land at 0.04 / 0.21 / 0.15 — the ceiling is high but
the bottleneck is real.

The decoder is fit globally over the full valid history (not
walk-forward). This is an in-sample expressivity probe — it
answers "can the CWT slice encode the indicator at all," not
"could a model trained on past data forecast it OOS."

```bash
uv run ss-replay AAPL                                       # baseline
uv run ss-replay AAPL --window-cols 64                      # +window
uv run ss-replay AAPL --window-cols 64 --include-zscore-stats
uv run ss-replay AAPL --window-cols 64 --include-zscore-stats \
    --decoder mlp                                           # all three
```

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

## Where to look next

- Notebooks in `apps/notebook/notebooks/`:
  - `causal_cwt_walkthrough.ipynb` — derivation of the `causal_cwt`
    machinery the CLIs all share.
  - `cwt_vision_multihead.ipynb` — Flax/JAX vision multi-head over
    scalogram tiles, including a Ridge linear-probe sanity baseline.
- Production trainer that consumes `ss_wavelets.causal_cwt` output:
  `apps/regime/src/regime/trainer.py`.
