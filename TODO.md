# TODO

## Extract shared CLI args into `packages/cli` (`ss_cli`)

CLI flags for "where's the data + when + save where" are duplicated across
~7 scripts in two apps. Two distinct flag groupings exist; both should
live in a single shared package so future scripts opt-in by one
function call instead of copying argparse blocks.

**Distribution name:** `ss-cli` &nbsp;**Import name:** `ss_cli`
**Layout:** `packages/cli/src/ss_cli/__init__.py` (mirrors the
`ss-loaders` / `ss-indicators` convention).

**API to export:**

```python
add_single_ticker_loader_args(parser)
    # --stooq-dir, --kaggle-dir, --start, --end
add_universe_loader_args(parser)
    # --data-dir (required), --start, --end
add_save_args(parser, *, default_output_dir='Output')
    # --save, --output-dir
```

The two loader helpers stay separate because the flag names actually
differ (single-ticker scripts can pick one of two sources;
universe scripts have a single `--data-dir`). `add_save_args` is
shared by both groupings.

**Call sites to migrate:**

Single-ticker (notebook app):
- `apps/notebook/src/ss_notebook/scalogram.py`
- `apps/notebook/src/ss_notebook/scalogram_video.py`
- `apps/notebook/src/ss_notebook/replay.py`

Universe (regime app):
- `apps/regime/src/regime/cli.py`
- `apps/regime/src/regime/research/backtest_bt.py`
- `apps/regime/src/regime/research/optimize_regime.py`
- `apps/regime/src/regime/research/backtest_ranking.py`

**Drift to normalize during migration:**
- `backtest_ranking.py` uses `--start-date` / `--end-date`; everything
  else uses `--start` / `--end`. Standardize on `--start` / `--end`
  and keep `--start-date` / `--end-date` as deprecated aliases for
  one release if any external scripts call it.

**Workspace wiring:**
- `packages/cli/pyproject.toml` with `[tool.hatch.build.targets.wheel]
  packages = ["src/ss_cli"]`.
- Root `pyproject.toml` already includes `packages/*` in the workspace
  members glob, so `uv sync --all-packages --inexact` picks it up.
- Each consumer adds `ss-cli` to its `dependencies` and
  `[tool.uv.sources] ss-cli = { workspace = true }`.

**Out of scope:**
- Legacy `apps/v1/scripts/*` `--save` flags. Parked workflow.
- The `regime live` arg block (`--params`, `--dry-run`, `--max-position`,
  `--killswitch`, `--max-data-age-days`) — single call site, no reuse.

## Add a realized-volatility (and/or autocorrelation) head to ss-replay

Once multi-head CNN is shaken out on rsi/macd/price, add another head
that tests a *higher-order* statistic — something the bundle doesn't
expose as a literal input feature. Right now every target is either
level-flavored (price) or a near-linear function of recent returns
(RSI, MACD).

**Why:** the encoder has CWT power per scale + raw returns + (mu, sigma).
RV is `std(returns_{t-19:t})` — variance of recent returns. CWT power
explicitly factorizes power by scale so summing per-scale power *should*
recover aggregate RV well. If multihead nails RV, the encoder genuinely
captures vol structure, not just first-order direction. If it underfits,
the bundle is missing variance-style information.

**ADX is not a viable substitute for RV here:** ADX measures trend
*consistency* (sign-agnostic but direction-aware), needs OHLC (high,
low, close) for `+DM/-DM/TR`. Yahoo close-only blocks it. RV measures
*magnitude* (sign-agnostic, direction-agnostic), needs only close.
Different signals, different dimensions of "market activity." A
close-only proxy for ADX-style trend strength would be the
autocorrelation of returns over a window (or Hurst exponent) — also
higher-order, also feasible.

**Implementation:**
- Add `realized_vol_n` and (optionally) `return_autocorr_n` to
  `TARGET_NAMES` in `apps/notebook/src/ss_notebook/replay/features.py`.
- Compute `realized_vol[t] = std(log_returns[t-n+1:t+1])` over n=20.
- Multi-head CNN picks them up automatically; no decoder-side change.

**Out of scope:**
- True ADX (would need OHLC source — change to Stooq daily archive
  which has high/low, breaks the Yahoo cross-source path).
- Williams %R, Stochastic, OBV — same OHLCV-needing blocker.
- Bollinger bands — trivially recoverable from `--include-zscore-stats`
  (BBands middle = mu, edges = mu ± k*sigma, both literal inputs).
  Sanity check, not a research lever.

## Ablation — disentangle why long-period RSI underperforms

The CSCO zero-shot RSI(n) sweep on the 30-ticker / `n_grid={5,7,9,13,21}`
/ K=64 run showed a sharp degradation at the long end:

| n  | in-grid | R²    |
|----|---------|-------|
| 9  | yes     | 0.964 |
| 13 | yes     | 0.902 |
| 18 | no      | 0.690 |
| 21 | yes     | 0.520 |

Two factors were proposed (see chat 2026-04-27):
1. **Grid spacing** — n=21 sits at the conditioning maximum with no
   right-neighbor; gap to its left-neighbor n=13 is 8 (vs spacings of
   2 below). The linear conditioning has fewer interpolation pairs
   here.
2. **Effective lookback** — Wilder RSI(n) has effective memory ~3×n
   bars. RSI(21) ≈ 63–84 bars; K=64 is at the edge. The model has
   long-horizon info via the rolling z-score stats and long CWT scales,
   but the *direct per-lag* path is window-bounded.

Three runs to disentangle (each is one CLI flag tweak from the
existing `ss-replay … --rsi-n-grid 5,7,9,13,21 …` cell):

| Run | `--rsi-n-grid`              | `--window-cols` | tests       |
|-----|-----------------------------|-----------------|-------------|
| A   | `5,7,9,13,17,21,25`         | 64              | factor 1    |
| B   | `5,7,9,13,21`               | 96              | factor 2    |
| C   | `5,7,9,13,17,21,25`         | 96              | combined    |

If A recovers RSI(21) R², factor 1 is dominant; spacing matters more
than lookback. If B recovers it, factor 2 is dominant; longer K is the
fix. C is the upper bound.

Beyond fixing one ticker's RSI numbers, this informs grid-design
heuristics for any future parameter-conditioned head — both grid
density and the input-bundle's effective lookback need to be
matched to the longest target parametrization.

**Out of scope** for the same diagnostic:
- Non-linear conditioning (sin/cos of n, or a small MLP on n). If A+B
  both fail, that's the next architectural lever.
- Re-running with the (w, n) 2D conditioning — that's a separate
  capability test, not a disentanglement of the existing failure.

## No-backbone IC baseline — does the CNN encoder help or hurt?

The SSL pretrain path (masked CWT autoencoding → frozen-backbone probe →
IC scorer) tests whether *better pretraining* lifts the scorer. It does
not test whether the CNN encoder is helping at all. If the CWT bundle
has return-predictive structure but the CNN is killing it during
encoding (lossy compression, wrong inductive bias, etc.), bypassing the
encoder entirely could win.

**Baseline:** linear scorer on the flattened raw CWT bundle (shape
`K * F = 64 * 29 ≈ 1856` per bar). Linear because capacity then matches
supervision (~13.5k cross-sectional cells) — no overfitting risk the
way the 5632-d head on the SSL latent had. Pearson IC objective, same
training loop as `scoring.train`.

**Three outcomes, each diagnostic:**

| Outcome                     | Interpretation                              | Next step                                                                 |
|-----------------------------|---------------------------------------------|---------------------------------------------------------------------------|
| `linear-raw > SSL+linear`   | Encoder kills useful info even with SSL.    | Drop the CNN; try transformer over scales, deep set over lags, or flat.   |
| `linear-raw ≈ SSL+linear`   | Encoder neither helps nor hurts.            | Supervision is the bottleneck — more data, better objective (Spearman).   |
| `linear-raw < SSL+linear`   | SSL latent is a genuinely better basis.     | Continue SSL, push capacity.                                              |

**Implementation:**
- Cleanest: build a synthetic `Backbone` in `scoring/backbone.py` whose
  `apply_backbone` is identity-flatten — `K_post * hidden = K * F`.
  Loaded via `Backbone.identity(meta_dict)` constructor that takes the
  same metadata blob a real npz would carry. Zero changes to
  `scoring/train.py` or `scoring/scorers.py`.
- Or: add `--no-backbone` flag to the scoring training entry point
  that swaps `apply_backbone(bb, X)` for `X.reshape(n, -1)` and sets
  `hidden_flat = K * F`. ~30 lines.

**Out of scope:**
- MLP scorer on raw CWT — re-introduces the same head-capacity / IC-noise
  overfitting we already documented; tells us nothing about the encoder.
- Full hyperparameter sweep on the linear baseline; only one number is
  needed (val IC) and the standard scorer setup applies.
- Replacing the encoder with a different architecture (transformer over
  scales, deep set over lags) — only worth doing if `linear-raw` wins
  decisively over `SSL+linear`, since architecture-search is expensive.

## Diagnose why w=1 row underperforms in the FiLM (w, n) head

The FiLM-conditioned (w, n) RSI head trained on
`n ∈ {5,7,9,13,17,21,25} × w ∈ {1,5,10,21}` zero-shot on CSCO produces
a smooth surface where every cell hits R² ≥ 0.65, but the w=1 row
plateaus at 0.70–0.89 while w=7 (off-grid interp between trained 5
and 10) reaches 0.86–0.95 — the *off-grid* row beats every in-grid
row at small n. Sweep snapshot (R² for n=7 across w):

| w  | in-grid | R²    |
|----|---------|-------|
| 1  | yes     | 0.80  |
| 3  | no      | 0.84  |
| 5  | yes     | 0.89  |
| 7  | no      | 0.93  |
| 10 | yes     | 0.93  |
| 21 | yes     | 0.92  |
| 25 | no      | 0.87  |

This is striking because RSI(n=7, w=1) ≡ canonical RSI(7), and a
prior single-target run on the same backbone hit R²=0.97 for that
exact target. Two compounding hypotheses (see chat 2026-04-29):

1. **Latent-frequency mismatch.** The bundle's CWT power is dominated
   by long scales (the regime-app finding); the w=7 sweet spot is
   exactly where the target's smoothing matches the latent's dominant
   representation. Daily RSI(w=1) is high-frequency oscillation that
   reads off the *minority* latent channels (small CWT scales 1/2/3 +
   the single log-returns channel).
2. **MLP smoothness penalty / cond-space asymmetry.** With 4 trained
   w-points normalized to `w/max(w)` = `[0.048, 0.238, 0.476, 1.0]`,
   w=1 sits as the spatial outlier near zero. The FiLM γ/β MLPs fit a
   smooth function across cond-space; smoothness means w=1 can't get
   a sharp local specialization without hurting other cells.

Two cheap experiments to disentangle (each is one knob, no
architecture change):

| Run | knob                                                      | tests       |
|-----|-----------------------------------------------------------|-------------|
| L   | Switch w_norm to `log(w)/log(max(w))` (instead of `w/max`). New trained points: `[0, 0.529, 0.756, 1.0]` — distributes spacing more evenly toward the small-w end. | factor 2 only |
| S   | Train with `--rsi-w-grid 1` and the same n-grid (degenerates to 1-D n-only conditioning, FiLM still on). | factor 1 only |

If L lifts (w=1) toward the w=5+ range while leaving the w=5/10/21
rows roughly intact, the cond-spacing was the dominant cause and the
fix is just a normalization tweak. If L doesn't help much but S
recovers RSI(7) to ≥0.95, the latent-frequency cause dominates and
the only paths forward are (a) accept that the multi-w head pays a
fixed tax on daily RSI, or (b) bias the latent toward higher
frequencies (e.g. drop long CWT scales, shorten the rolling z-score
lookback, or oversample short scales in the bundle).

**Implementation:**
- L is a one-line change in `apps/notebook/src/ss_notebook/replay/reconstruct.py`
  around line 148 (`w_max = float(max(rsi_w_grid)); w_values = ...`).
  Add `--rsi-w-norm {linear,log}` flag (default `linear` to preserve
  current behavior) and thread through `fit_and_evaluate`. Inference
  cell needs the matching change in the cond_vec construction, so
  the npz should record the chosen normalization in `_meta` for
  loaders to mirror.
- S is just a CLI flag tweak — no code change.

**Out of scope** for the same diagnostic:
- Larger FiLM hidden width (`--cnn-film-hidden 64+`). If both L and S
  fail, that's the next lever — gives the cond MLPs more capacity to
  represent sharper per-cell specialization.
- Multi-resolution latent (separate small-scale-only pretraining
  branch concat'd with the long-scale latent). Architectural; tackle
  only if S confirms latent-frequency is the dominant cause.
