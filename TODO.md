# TODO

## Use self-supervised learning to forecast the CWT

We are use patches hold out on the CWT and asking the SSL to guess what they
are. This should now apply to future patches as well. Our val scores of SSL were
okay, not amazing. But it does mean that we have learned some structure of the CWT.

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

## Backbone architecture — push toward broader window/indicator coverage

Three architectural levers below SSL pretrain, ordered by cost.
Goal: relax the current backbone's narrow indicator-shaped bias (only
4 unconditioned linear heads = at most 4-rank usage of the 5632-d
latent; the other ~5600 dims get zero gradient and decay toward noise).
See `apps/notebook/src/ss_notebook/replay/README.md` "What the backbone
actually learns" for the diagnosis.

### Option A — FiLM-condition all four heads over their period grids

Currently only the rsi head is FiLM-conditioned (over `(n, w)`). Extend
the same machinery to the other three:
- macd: `p_dim=3`, grid over `(fast, slow, signal)` — e.g.
  `{8,12,16} × {21,26,34} × {7,9,11}` = 27 cells.
- vol: `p_dim=1`, grid over `vol_window` — e.g. `{5,10,20,40,60}` = 5
  cells.
- price: `p_dim=1`, grid over stride — e.g. `{1,5,10,21}` = 4 cells.

**Why:** the backbone gradient currently sees ~31 effective heads (3
unconditioned + 28 RSI cells). With all four FiLM-conditioned over
wide grids, that climbs to ~64 effective heads (28 RSI + 27 MACD + 5
vol + 4 price). Same indicator-shaped bias, but spread across many
more directions of the latent. Window-invariant within each indicator
family.

**Doesn't fix:** indicator-shaped bias overall — the gradient still
only spans 4 indicator families, just with more periods of each.

**Cost:** medium. New grid plumbing in `features.py`
(`macd_param_grid`, `vol_window_grid`, `price_stride_grid`), matching
pool-augmentation in `reconstruct.py::fit_and_evaluate`, CLI flags in
`cli.py`. ~150 lines.

**Test:** does the IC scorer's val IC move off the noise floor (val
IC ≈ 0, val Sharpe +0.55..+0.63 from `NOTES.md` 2026-04-30) when
trained on this richer backbone? If yes, broader window coverage was
enough. If no, the bottleneck is structural — go to option C or SSL.

### Option B — Add 20+ diverse indicator heads

Throw the kitchen sink: BB, ATR, OBV, ADX, ROC, CCI, Stochastic,
Williams %R, MFI, EMA crossovers, etc. Each adds one unconditioned
linear head → one more direction of gradient pressure on the latent.

**Why:** more heads = more directions of gradient = more of the latent
gets used. Cheaper than option A (no conditioning grids needed) but
also less rich (each head is just one direction; FiLM-cond grids give
many directions per indicator).

**Doesn't fix:** still indicator-shaped — just `N`-indicator-shaped
instead of 4. If standard TA indicators are highly correlated (RSI/CCI/
Williams %R all measure overbought/oversold from different angles),
adding correlated indicators doesn't actually span more of the latent.

**Cost:** low per indicator. Compute the target in `features.py`,
register in `TARGET_NAMES`, that's it. ~30 lines per indicator. Pick a
diverse set (momentum / trend / volume / volatility) to avoid
correlated redundancy.

**Order vs A:** A is structurally richer (each FiLM head spans many
periods); B is volume-richer (more independent heads). Try A first if
you want depth, B first if you want breadth. Not mutually exclusive.

### Option C — MLP heads, transfer head hidden layer to scoring

Replace each `Linear(latent → 1)` head with `Linear(latent → h) →
ReLU → Linear(h → 1)` (default `h=64`). Save the per-head first-layer
weights (`Linear(latent → h)`) into the npz. Modify
`scoring/backbone.py::load_backbone` to optionally also extract the
per-head hidden-layer projections; modify `scoring/scorers.py` so the
IC head reads `concat([head_h_rsi(latent), head_h_macd(latent),
head_h_vol(latent), head_h_price(latent)])` (4 × h = 256-d) instead
of the raw 5632-d latent.

**Why:** the per-head hidden layer + ReLU encodes the implicit
**threshold/regime nonlinearities** the indicator computes (Wilder
smoothing, MACD sign-flips, vol regime bands). The IC scorer reading
these threshold-aware features can express "RSI<30 → long" type
signals as a *linear* combination — exactly the gap between
"deterministic indicator strategies produce positive Sharpe" and
"linear IC head sees nothing" (per the chat thread that produced this
TODO entry). Also drops the IC head's input dim 22× (5632 → 256),
direct attack on the noise-floor overfitting symptom.

**Bonus:** the rsi head's FiLM gamma/beta MLPs are themselves a
transferable invariance machinery. Scoring could query the rsi head
at multiple `(n, w)` cond values and concat the modulated outputs,
giving IC access to "RSI-tuned features at n=7, n=14, n=21"
simultaneously. Pure code change at scoring time, no retraining.

**Doesn't fix:** backbone is still indicator-coupled (arguably more
coupled because MLP heads can compress the backbone's job further).
What this does is repackage the indicator-shaped bias as transferable
threshold-aware features, instead of fighting it. If the IC alpha
lives outside the {RSI, MACD, vol, price} family, this still doesn't
help — only SSL catches that case.

**Cost:** ~80 lines. `decoders.py`: MLP heads + save head_h_W/b.
npz writer: extend per-target prefix to include `head_h_*`.
`scoring/backbone.py`: extend `Backbone` to optionally carry per-head
hidden weights; `apply_backbone` returns either raw latent or
concat-of-head-projections depending on a flag. `scoring/scorers.py`:
no change needed if the backbone returns the right shape.

**Test:** as in option A — does val IC move off the noise floor? This
is the most targeted intervention if you believe the IC failure is the
linear-head-can't-express-thresholds story.

### Decision order

1. Run option A first as the diagnostic — broader window coverage
   without architectural change to scoring. If val IC clears noise
   floor, declare victory and stop.
2. If A doesn't move the needle, run C — the targeted fix for the
   "linear head can't see thresholds" hypothesis.
3. If C still doesn't move it, the bias is fundamental → run full
   SSL pretrain (`fit_cnn_masked_ae`, already implemented; needs
   a CLI hookup, mask-ratio sweep, probe protocol).
4. Option B is a "free" addition at any stage — adding diverse
   indicator targets costs ~30 lines per indicator and is orthogonal
   to A/C.

## Streaming feature pipeline (replace bulk pre-compute, lift the OOM ceiling)

Today's `fit_and_evaluate` concatenates every train ticker's full
feature matrix into one giant `(n_pool, K * F)` array before training.
At 19 tickers × ~3000 valid bars × K=96 × F=33 that's ~720 MB
(float32) on host RAM, plus a JAX device copy, plus FiLM-augmented
auxiliary arrays (Y_std, cond, train_pool_idx). Combined with a 12 GB
Colab CPU runtime this OOMs (2026-05-01 attempt killed at the supervised
sign-of-return run before training started).

The float32 cast (committed 2026-05-01) lifts the ceiling 2x but
doesn't change the asymptote — at K > 128 or train pool > 30 tickers
we'll hit it again. Streaming is the structural fix.

**Goal:** never materialize more than `batch_size` feature rows at
once. Per-step pseudocode:
```
for step in range(n_steps):
    batch = sample_batch(per_ticker_handles, batch_size, train_pool_idx)
    # `batch` is a fresh (batch_size, K, F) gather assembled on demand
    # from per-ticker on-disk / lazy feature arrays.
    loss, grads = value_and_grad(loss_fn)(params, batch, ...)
    params = update(params, grads)
```

**Two implementation candidates, ranked by effort:**

1. **`np.memmap` per-ticker features.** Save each ticker's
   `(n_dates, K * F)` feature matrix to a `.npy` on disk during
   `load_ticker`. The trainer's pool then holds an index of
   `(ticker_id, local_bar_idx)` tuples; the batch sampler maps each
   logical row to a memmap'd ticker file and reads the K-row slice on
   demand. JAX still materializes only `batch_size` worth of rows.
   Cost: ~50 lines (a `LazyTicker` class wrapping memmap + index, plus
   a batch sampler in `decoders.py::fit_cnn_multihead`). No change to
   the model or loss. Disk I/O per step is O(batch_size * K * F) —
   trivial on local SSD, slow on Colab's network disk but probably
   tolerable.

2. **Pure in-memory streaming via `tf.data` / `jax.experimental.shard`.**
   Heavier framework dependency; only worth it if (1) is also too slow.

**Why not just shrink the universe / K?** That works for individual
experiments but doesn't scale. We're going to want to push to
50+ tickers, K=128+, longer FiLM grids — all of which exceed Colab's
12 GB CPU budget AND a single A100's 80 GB if the float32 ceiling
keeps growing.

**Adjacent improvement worth folding in**: the FiLM augmentation
already uses `train_pool_idx` lazily (per `reconstruct.py:99`) but the
trainer still gathers the full augmented set into `Xj` on every step
in the full-batch path (`decoders.py:792`). The streaming refactor
should remove that path entirely — there's no scenario where holding
`n_pool * n_replicas` rows of features in memory is the right answer.

**Test for it:** a "scaling" smoke test in `apps/notebook/tests/`
that runs a 100-ticker / K=128 fit with cnn-batch-size=512 and
asserts peak RSS stays under 4 GB. Fails today; should pass after
streaming.

## Port `ss_portfolio.sharpe.block_sharpe_with_costs` to tinygrad

Last load-bearing JAX site in the workspace. After the
ss_indicators numpy migration + replay/factor on tinygrad,
`packages/portfolio/src/ss_portfolio/sharpe.py` is the only file
that genuinely needs JAX (for `jax.grad` over a differentiable
Sharpe-with-costs loss). Everything else that imports `jax` /
`jnp` does so as residue or as a parked-research dep.

**Live consumers** (full grep, post-extraction):
  - `apps/regime/research/optimize_adam.py` — calls
    `block_sharpe_with_costs` via `jax.value_and_grad`. **Parked**
    per CLAUDE.md (gradient flow is already broken at the
    `get_divergence` boundary since `ss_indicators` went numpy, so
    `jax.grad` produces zero/NaN through the divergence call).
  - `apps/factor/src/factor/objectives.py` — docstring reference
    only ("matching the JAX `ss_portfolio.block_sharpe_with_costs`
    definition"); does not actually call the function.
  - `packages/portfolio/tests/test_portfolio.py` — exercises with
    `jax.grad` (1 shape test + 1 differentiability test).

So the live consumer is one parked file whose gradient is already
broken. Porting `sharpe.py` alone "kills JAX" in the sense that no
non-parked code imports it; but `optimize_adam.py` will fail at
import (it `import jax` + calls `jax.value_and_grad` on a now-
tinygrad function — pure-functional autograd doesn't compose with
tinygrad's stateful `Tensor.backward()`).

Three honest paths, in increasing scope:

1. **B (recommended): port + delete `optimize_adam.py`** (~1-2 h).
   Parked-and-broken-anyway gets removed; the deletion is the most
   honest acknowledgement of CLAUDE.md's status. Tests in
   `test_portfolio.py` switch to tinygrad's `requires_grad=True`
   + `loss.backward()` pattern.
2. **C: port + skeleton stub `optimize_adam.py`** (~1.5 h). Same
   as B but leaves a 10-line file pointing at the last working JAX
   commit so the historical context survives a `git log` search.
3. **A: port everything including a tinygrad rewrite of the
   JAX-Adam optimization loop** (~3-4 h). Most thorough; preserves
   the differentiable-regime-trainer story end-to-end. Requires
   replacing `jax.value_and_grad` + `optax.adam` with tinygrad's
   `Tensor.backward()` + `tinygrad.nn.optim.Adam`. Worth doing
   only if the differentiable optimizer is actually going to be
   used again — otherwise B is honester.

**Mechanical port of `sharpe.py` itself** (independent of which
optimize_adam path is chosen):
  - Replace `jax.Array` annotations with `tinygrad.Tensor`.
  - Replace `jnp.{exp, log, abs, concatenate, sqrt}` with the
    tinygrad equivalents (mostly identical names, all on `Tensor`).
  - `s - s.max(axis=1, keepdims=True)` already a tensor op in both.
  - Soft-top-N math is unchanged; only the framework changes.
  - Tests that did `jax.grad(loss)(jnp.log(jnp.asarray(0.5)))`
    become:
    ```python
    log_t = Tensor(np.log([0.5]).astype(np.float32),
                   requires_grad=True)
    loss = -block_sharpe_with_costs(..., log_temperature=log_t, ...)
    loss.backward()
    grad = log_t.grad.numpy()
    assert np.isfinite(grad).all()
    ```
  - The `jnp.sqrt(TRADING_DAYS / rebal_days)` constant should be
    pre-computed at module load (it's a Python int → Python float;
    no need to wrap in any tensor type).

**Why this isn't done yet:** the only live consumer is parked +
broken. Doing the port without a path B/C/A choice strands
`optimize_adam.py` in import-error territory (worse than parked).
The port is mechanical; the prerequisite is a decision on what
happens to `optimize_adam.py`.

**Side cleanup that should ride along** (not blocking; pick up
during B): the `regime/trainer.py` + `inference.py` + `persist.py`
+ `reporting.py` files import `jax.numpy as jnp` for handful of
type hints / `jnp.zeros` / `jnp.asarray` calls that became
no-ops once `ss_indicators` went numpy. Each is a 2-line cleanup;
together they remove the residual JAX imports across `apps/regime`
that aren't `optimize_adam.py`.

