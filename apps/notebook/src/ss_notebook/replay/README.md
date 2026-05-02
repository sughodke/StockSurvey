# `ss_notebook.replay` — CWT-slice indicator decoder

Train a small per-bar decoder that reconstructs classical TA indicators
(RSI, MACD line, realized vol, raw close) from the trailing-window
**causal CWT** of price. Used as a **backbone-pretraining** stage: the
saved npz holds a shared 1-D conv backbone whose latent is then frozen
and reused by `ss_notebook.scoring/` for forward-return scoring.

The intent of training on indicator targets isn't the indicators
themselves — it's the regularizer they impose on the latent. Period-
conditioned heads (e.g. RSI(n, w) for arbitrary `(n, w)` in a trained
grid) push the network to encode the **underlying primitives** (smoothed
gain/loss series, return-distribution moments) instead of memorizing one
parametrization, which is meant to give the downstream linear scorer a
more stable feature space. See "What the backbone actually learns"
below for an honest read on how much of this hope is realized in
practice — short version: window-invariant on the conditioned axis,
period-coupled on the unconditioned heads, and weakly indicator-
invariant overall (the linear-head architecture only puts gradient
pressure on a few directions of the latent).

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
        ├── target 'price':  Linear(latent_dim -> 1)               (unconditioned)
        ├── target 'macd' :  Linear(latent_dim -> 1)               (unconditioned)
        ├── target 'vol'  :  Linear(latent_dim -> 1)               (unconditioned)
        └── target 'rsi'  :  FiLM-modulated Linear(latent_dim -> 1)
                              ▲
                              │ cond = [n / max(n_grid), w / max(w_grid)]   p_dim = 2
                              │ MLP_gamma(cond), MLP_beta(cond) -> per-latent (gamma, beta)
                              │ latent' = gamma * latent + beta
                              │ y_hat   = latent' @ head_W + head_b
```

### Period conditioning — FiLM-modulated Option B

Three choices considered for "make the model handle arbitrary RSI(n, w)":

- **Option A — append `(n, w)` as input channels** to the backbone.
  Easy to wire but contaminates the backbone latent with the parameter,
  which defeats the whole point: the scoring pipeline reads the frozen
  latent and would inherit a parameter axis it doesn't care about.
- **Option B (additive concat) — concat `(n, w)` to the flattened
  latent right before the head.** Backbone stays parameter-agnostic
  (its output shape is unchanged); only the conditioned head's input
  widens by `p_dim`. The head's weight matrix gets `p_dim` extra
  columns that absorb the cond contribution as a pure additive bias —
  no `latent × cond` interaction.
- **Option B + FiLM (current default).** Same backbone-stays-clean
  property, but the conditioning vector drives two small MLPs that
  produce per-latent `(gamma, beta)` so `latent' = gamma * latent +
  beta`. This gives true `latent × cond` interaction (each cond value
  can amplify or suppress different latent dimensions), at the cost of
  two extra weight tensors per conditioned head. Falls back to additive
  concat when `film_hidden = 0`.

Implementation:

1. `features.py::build_features_and_targets` accepts `rsi_n_grid` and
   `rsi_w_grid` and, when non-empty, computes RSI at every `(n, w)`
   pair in the cross-product as a `(n_cells, n_dates)` array on the
   side (anchor n/w still produces the 1-D `targets['rsi']` used for
   plotting/stats).
2. `reconstruct.py::fit_and_evaluate` replicates the pooled training
   rows once per grid cell (`n_pooled` → `n_pooled * |n_grid| *
   |w_grid|`), overrides `y_train['rsi']` with the grid-indexed target
   values, tiles the other targets for alignment, and builds
   `head_conditioning_train['rsi']` of shape `(n_replicated, 2)`
   holding `[n / max(n_grid), w / max(w_grid)]`.
3. `decoders.py::fit_cnn_multihead` initializes FiLM gamma/beta MLPs
   per conditioned head (with `film_hidden=32` by default; the final
   gamma layer is zero-init so the head is identity wrt cond at step
   0). Persists FiLM weights under `{target}__head_film_*` and the
   cond width under `{target}__head_cond_dim`.
4. For the prediction pass, `head_conditioning_predict['rsi']` is a
   constant column of `[anchor_n / max(grid), anchor_w / max(grid)]`
   (anchor = `--rsi-n` / `--rsi-anchor-w`) so the reconstruction
   figure compares against the same `(n, w)` the ground-truth panel
   uses.

The pattern generalizes: MACD `(fast, slow, signal)` would use
`p_dim=3`, BB `(period, k)` would use `p_dim=2`. Currently only RSI is
wired — see TODO and "What the backbone actually learns" below for why
extending this is the cheapest path to broader window-invariance.

## What the backbone actually learns

The intent stated at the top of this README ("indicator targets as a
regularizer that pushes the latent toward underlying primitives") is
partly realized but narrower than the phrasing suggests. Concretely:

**Window-invariance — partial.** Only the `rsi` head is FiLM-conditioned,
so the backbone is gradient-pressured to support arbitrary RSI(n, w)
within the trained grid. The other three heads (`macd`, `price`, `vol`)
are unconditioned and pull the backbone toward representations
specifically optimized for their fixed periods (MACD(12, 26, 9),
vol(20), raw close). Net: window-invariant on the RSI axis, period-
coupled on the rest.

**Indicator-invariance — weak.** Each unconditioned linear head reads
the flat latent (`K_post * hidden = 5632` dims at K=96, hidden=64,
4-layer kernel-3 conv stack) as a single projection vector. Four heads
→ at most a 4-rank usage of the 5632-d latent. The other ~5600
dimensions get zero gradient signal from any head; Adam's implicit
regularization (and any weight decay) shrinks the conv weights that
produce those unused dimensions toward zero. So the backbone is not a
generic CWT reader — it's a 4-indicator-family linearly-decodable
representation, with decayed-toward-zero noise in the unused subspace.

**What does transfer.** The CSCO zero-shot R² of 0.972 from a single-
head RSI(7) bundle run demonstrates the indicator-decoding subspaces
the backbone learned are not ticker-specific — they generalize across
the universe. That's real, but its scope is "how to read CWT for
RSI/MACD/vol/price *in any ticker*," not "how to read CWT in general."

**Implication for downstream IC scoring** (`ss_notebook.scoring`). A
linear IC head reading the frozen latent inherits the same narrow scope.
It can produce scores proportional to RSI/MACD level, but rank IC of
indicator level against forward returns is ~0 (the alpha lives in
nonlinear functions of indicators — thresholds, divergences, regime
gating — none of which a linear head over a 4-indicator-shaped latent
can express). This is the structural reason the encoder-vs-raw IC
comparison ties at the noise floor (NOTES.md, 2026-04-30) even though
the backbone provably encodes rich indicator structure (replay R² ≥
0.95).

**To make the backbone genuinely indicator-invariant**, the gradient
signal has to put pressure on the *whole* latent, not just on a few
projection directions. Two paths, in increasing cost / strictness:

1. **FiLM-condition every head over its period grid.** Each head
   becomes a many-windows-of-its-family decoder; the gradient still
   only spans 4 indicator families but spans all windows of each.
   Window-invariant within each family, not generic. Cheap — same
   trainer, more conditioning vectors threaded through.
2. **SSL pretrain** (`fit_cnn_masked_ae`, now wired as `--decoder
   masked-ae` with `--mask-ratio` / `--ssl-decoder-hidden` /
   `--ssl-decoder-layers`; Colab pipeline lives at
   `apps/notebook/scripts/colab/train_ssl.sh` →
   `probe_ssl.sh` → `ssl_ic_scorer.py`). Loss = MSE on masked
   cells of the bundle. Every output cell of the decoder backprops
   through the backbone, forcing broad encoding instead of
   N-direction encoding. No indicator-shaped bias at all. Strictly
   cleaner; more compute.

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
| `--rsi-n-grid 5,7,9,13,21` | Enables period conditioning for RSI on the n axis. CNN-only. |
| `--rsi-w-grid 1,5,10,21` | Enables stride conditioning for RSI on the w axis (cross-product with `--rsi-n-grid`). w=1 is canonical daily RSI; w>1 is RSI evaluated on prices `w` bars apart. Requires FiLM-modulated head (default). |
| `--rsi-anchor-w W` | Anchor stride for the 1-D plotting / stats panel (default 1). |
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
{target}__head_W, {target}__head_b            # per-target head (latent_dim -> 1)
{target}__head_cond_dim                       # 0 if unconditioned, else p_dim
{target}__head_film_gamma_W0/b0/W1/b1         # FiLM gamma MLP (conditioned heads only,
{target}__head_film_beta_W0/b0/W1/b1          #   when film_hidden > 0)
{target}__target_mu, {target}__target_sd      # per-target output unstandardizer
_meta                                         # JSON blob of CLI args + train/val stats
```

`scoring.backbone.load_backbone` filters out everything matching
`{target}__head_*` and `{target}__target_*` (so all the head FiLM
weights, head_W/b, cond_dim, and target standardizer fields stay
behind), and verifies the remaining backbone tensors are byte-identical
across all per-target prefixes. The scoring pipeline therefore sees a
parameter-agnostic latent — period-conditioned head bookkeeping is
written to disk for completeness and reproducibility but does not
affect downstream IC fitting.

## TODO

- **FiLM-condition `macd` over `(fast, slow, signal)`** — `p_dim=3`,
  same pattern as RSI. Requires a `macd_param_grid` in `features.py`
  and the matching pool-augmentation in `fit_and_evaluate`. This is
  the cheapest step toward broader window-invariance per "What the
  backbone actually learns" above.
- **FiLM-condition `vol` over `vol_window`** — `p_dim=1`. Same plumbing.
  Drops the fixed-window coupling on the vol axis.
- **FiLM-condition `price` over stride** if/when the scoring pipeline
  wants multi-horizon level prediction.
- Optional: extend conditioning to Bollinger Bands `(period, k)`. Add
  a `bbands` target first.
- **Mask-ratio + decoder-capacity sweep for SSL.** `--decoder
  masked-ae` is wired and the `train_ssl.sh` → `probe_ssl.sh` →
  `ssl_ic_scorer.py` Colab loop is operational, but the first run
  reported `train_mse_masked ≈ 0.89` (z-norm space) with masked vs
  unmasked MSE barely differing — symptom of decoder under-capacity,
  not future leakage. Current defaults (`--mask-ratio 0.25`,
  `--ssl-decoder-hidden 1024`, `--cnn-steps 20000`) are the second-
  attempt point; sweep mask_ratio ∈ {0.15, 0.25, 0.40} ×
  decoder_hidden ∈ {512, 1024, 2048} and report frozen-probe
  per-indicator R² + downstream IC.
- **Decision**: if SSL probe R² lands ≥ 0.85 on RSI/MACD with
  meaningful uplift on `ssl_ic_scorer` rank IC vs the
  `identity_backbone` no-encoder baseline (`scripts/no_backbone_
  baseline.py`), promote SSL to the canonical backbone-pretrain
  stage in this README's pipeline diagram and demote the multi-head
  supervised path to "diagnostic / lighter-weight alternative."
