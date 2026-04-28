# `ss_notebook.replay` — CWT-slice indicator decoder

Train a small per-bar decoder that reconstructs classical TA indicators
(RSI, MACD line, realized vol, raw close) from the trailing-window
**causal CWT** of price. Used as a **backbone-pretraining** stage: the
saved npz holds a shared 1-D conv backbone whose latent is then frozen
and reused by `ss_notebook.scoring/` for forward-return scoring.

The point of training on indicator targets isn't the indicators
themselves — it's the regularizer they impose on the latent. Period-
conditioned heads (e.g. RSI(n) for arbitrary n) push the network to
encode the **underlying primitives** (smoothed gain/loss series, return-
distribution moments) instead of memorizing one parametrization, which
gives the downstream linear scorer a more stable feature space.

## File map

| File | Role |
| --- | --- |
| `features.py` | Build per-bar feature stack (CWT coeffs + power, optional rolling z-score stats, optional log-returns) and the four ground-truth targets (`price`, `rsi`, `macd`, `vol`). Holds `realized_vol` and the period-grid plumbing for RSI conditioning. |
| `decoders.py` | Decoder fits: `fit_ols`, `fit_mlp`, `fit_cnn` (single-target), `fit_cnn_multihead` (shared backbone + per-target heads, optional per-head conditioning concat). |
| `reconstruct.py` | `fit_and_evaluate` — pool train tickers into one fit, predict per ticker. Owns the **training-pool augmentation** for the RSI period grid (each row replicated per grid value) and threads the per-head conditioning into the trainer. |
| `cli.py` | `ss-replay` entrypoint. Parses flags, loads tickers (Stooq / Kaggle / Yahoo), runs the fit, writes the reconstruction figure(s) and the weights npz. |
| `plot.py` | Per-ticker reconstruction figure (one row per target). |
| `metrics.py` | R² / RMSE / max-\|Δ\| stats vs ground truth. |

## Targets

`TARGET_NAMES = ('price', 'rsi', 'macd', 'vol')`. Each emits one decoder
fit (or one head when `--decoder cnn`):

- **`price`** — raw close. Recovers the rolling-z-norm level info that
  `causal_cwt` strips out. Only useful when `--include-zscore-stats`
  feeds the rolling mean/std back as input channels.
- **`rsi`** — Wilder's RSI at `--rsi-n` (default 7). With
  `--rsi-n-grid` set, the head is **period-conditioned** and trains on
  every n in the grid simultaneously.
- **`macd`** — MACD line at `--macd-fast/--macd-slow/--macd-signal`.
  (Currently single-parameter; see TODO below.)
- **`vol`** — realized vol = causal rolling std of daily log returns
  over `--vol-window` bars (default 20, matches the regime trainer's
  rebalance horizon). Replaces the originally-planned forward-return
  head: vol is the part of the return distribution the rolling-z-normed
  scalogram is best positioned to recover, and it is a known cross-
  sectional return predictor in its own right — making it a low-noise
  pretraining anchor for downstream forward-return scoring.

## Multi-head CNN architecture

`fit_cnn_multihead` (in `decoders.py`):

```
input  (n, K, F)
        │       K = window_cols, F = channels per lag
        ▼
  z-norm  (feat_mu, feat_sd)              ─┐
        ▼                                   │   shared backbone:
   [Conv1D + ReLU] × n_layers, VALID padding│  saved as feat_mu/sd +
        ▼                                   │  conv{i}_W/b — frozen and
   flatten -> (n, K_post * hidden) latent  ─┘  reused by ss_notebook.scoring
        │
        ├── target 'price':  Linear(latent_dim       -> 1)
        ├── target 'macd' :  Linear(latent_dim       -> 1)
        ├── target 'vol'  :  Linear(latent_dim       -> 1)
        └── target 'rsi'  :  Linear(latent_dim + p_dim -> 1)
                              ▲
                              │ concat(latent, conditioning_vector)
                              │ conditioning = [n / max(grid)] (p_dim=1)
```

### Period conditioning — Option B (latent-then-concat)

Choices considered for "make the model handle arbitrary RSI(n)":

- **Option A — append `n` as an input channel** to the backbone. Easy
  to wire but contaminates the backbone latent with the parameter,
  which defeats the whole point: the scoring pipeline reads the frozen
  latent and would inherit a parameter axis it doesn't care about.
- **Option B — concat `n` to the flattened latent right before the
  head.** Backbone stays parameter-agnostic (its output shape is
  unchanged); only the conditioned head's input widens by `p_dim`.
  This is what is implemented.

Implementation:

1. `features.py::build_features_and_targets` accepts `rsi_n_grid` and,
   when non-empty, computes RSI at every n in the grid as a `(n_grid,
   n_dates)` array on the side (anchor n still produces the 1-D
   `targets['rsi']` used for plotting/stats).
2. `reconstruct.py::fit_and_evaluate` replicates the pooled training
   rows once per grid value (`n_pooled` → `n_pooled * n_grid`),
   overrides `y_train['rsi']` with the grid-indexed target values,
   tiles the other targets for alignment, and builds
   `head_conditioning_train['rsi']` of shape `(n_pooled * n_grid, 1)`
   holding `n / max(grid)`.
3. `decoders.py::fit_cnn_multihead` widens that head's input weight
   matrix by `p_dim`, concats the conditioning vector to the latent
   inside `forward`, and persists `cond_dim` in the per-target params
   dict so inference knows what to feed.
4. For the prediction pass, `head_conditioning_predict['rsi']` is a
   constant column of `anchor_n / max(grid)` (anchor = `--rsi-n`,
   defaults to median of the grid) so the reconstruction figure
   compares against the same n the ground-truth panel uses.

The pattern generalizes: MACD `(fast, slow, signal)` would use
`p_dim=3`, BB `(period, k)` would use `p_dim=2`. Currently only RSI is
wired.

## CLI

```bash
# Single ticker, no conditioning, all default targets.
uv run ss-replay AAPL --stooq-dir ./StooqData --window-cols 64 \
    --decoder cnn --targets price,rsi,macd,vol

# Multi-ticker pool, RSI period conditioning, vol head.
uv run ss-replay AAPL --train-tickers MSFT,GOOGL,AMZN --val-ticker TSLA \
    --stooq-dir ./StooqData --window-cols 64 --decoder cnn \
    --rsi-n-grid 5,7,9,13,21 --vol-window 20 \
    --targets rsi,vol --cnn-steps 2000

# Colab path (no on-disk archive).
uv run ss-replay AAPL --val-ticker MSFT --yahoo \
    --start 2018-01-01 --end 2024-12-31 --window-cols 64 \
    --decoder cnn --rsi-n-grid 5,7,9,13,21 --targets rsi,vol
```

Key flags:

| Flag | Meaning |
| --- | --- |
| `--window-cols K` | Trailing-window size in lags. CNN needs `K > kernel * n_layers`. |
| `--rsi-n N` | Anchor RSI period — used for the 1-D ground-truth panel and as the conditioning value at prediction time. |
| `--rsi-n-grid 5,7,9,13,21` | Enables period conditioning for RSI. CNN-only. |
| `--vol-window 20` | Realized-vol window. |
| `--include-zscore-stats` | Append rolling z-norm mean/std as 2 extra input channels. Required to recover `price`. |
| `--include-returns` | Append per-bar log returns as 1 extra input channel. Closes the high-frequency CWT band-limit gap. |
| `--targets` | Comma-separated subset of `price,rsi,macd,vol`. Each adds one head. |

## Outputs

```
Output/<train>-<targets>-<decoder>-<git-sha>.npz   # weights + metadata
Output/<train>-replay.png                           # train ticker recon figure
Output/<val>-replay-zeroshot-from-<train>.png      # val ticker recon figure (if --val-ticker)
```

The npz layout is `{target}__{key}` per target with shared backbone
weights duplicated under each prefix:

```
{target}__feat_mu, {target}__feat_sd
{target}__conv{i}_W, {target}__conv{i}_b      # shared backbone, repeated
{target}__head_W, {target}__head_b            # per-target head
{target}__head_cond_dim                       # 0 if unconditioned, else p_dim
{target}__target_mu, {target}__target_sd      # per-target output unstandardizer
_meta                                         # JSON blob of CLI args + train/val stats
```

`ss_notebook.scoring.backbone.load_backbone` extracts only the shared
backbone (drops every `head_*` / `target_*` / `cond_dim` key) and
verifies the per-target prefixes carry identical backbone tensors. So
period-conditioned heads are written to disk for completeness and
reproducibility but the scoring pipeline is unaffected by them — it
sees only the parameter-agnostic latent.

## TODO

- Extend conditioning to MACD `(fast, slow, signal)` — `p_dim=3`,
  same pattern as RSI. Requires a `macd_param_grid` in `features.py`
  and the matching pool-augmentation in `fit_and_evaluate`.
- Optional: extend conditioning to Bollinger Bands `(period, k)`. Add
  a `bbands` target first.
- The realized-vol head currently uses a fixed `vol_window`; if the
  scoring pipeline ends up wanting predictions at multiple horizons,
  the same conditioning pattern applies.
