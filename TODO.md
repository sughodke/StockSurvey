# TODO

## Recently shipped (and dropped from this list)

- `ss-portfolio.broker` (Alpaca adapter shared between regime + relational) +
  `ss-relational live` paper-trade stack with the four risk rails.
- `apps/relational/scripts/build_canonical_checkpoints.py` — six canonical
  RelationalCheckpoint JSONs covering the scoreboard winners.
- Three critical live-trading code-review fixes: `apply_position_cap` no
  longer re-introduces zero-weight names; live bar fetch covers full CWT
  kernel support; `submit_orders` surfaces per-symbol rejections to
  `LiveRunResult.rejected_orders`. (commits a1beead / 8c21c9b / 4ee4d0d)
- `ss_cli` (shared CLI flag groups) + `ss_portfolio.bt_helpers` (shared
  bt.Strategy template) + `packages/tg_ops/` (shared `_conv1d` permute) +
  block-windows generator → `ss_features.walkforward`. (commit ddfadce)
- Polar Morlet + Gaussian + log-L2 amplitude input bundle (commit 954a88a
  + streaming-predict refactor) replaces the
  `--include-zscore-stats / --include-returns / --include-return-sign`
  optional channels. Canonical channel layout per scale: `(|c|, |c|^2,
  cos(arg), sin(arg), g, g^2, log_L2_amp)` = `7 * n_scales` per lag,
  exposed as `ss_features.CHANNELS_PER_SCALE`. CSCO zero-shot R² at the
  canonical 295-ticker pool: vol 0.48 → 0.64 (+0.16), cci 0.85 → 0.89
  (+0.04), rsi 0.90 → 0.87 (-0.03). MACD head pathology unchanged
  (pre-existing; see "MACD head pathology" below).

## Migrate non-research apps to the polar Morlet input bundle

The polar Morlet + Gaussian + log-L2 amplitude bundle is the canonical
SSL/CNN input for **`apps/replay`** (already on it as of 954a88a) and
**`apps/factor`** (consumes the resulting backbone npz via
`ss_features.load_backbone` — no code change needed, the new npz
schema is backward-compatible at the loader layer). The remaining
non-research consumers of `ss_wavelets.causal_cwt` (real Ricker)
should migrate so the workspace converges on a single wavelet family.

Two important caveats up front:

1. **"Bundle" is a CNN-input concept.** It's a `(K, F=7*n_scales)` lag-
   windowed feature stack consumed by a 1-D conv backbone. Apps that
   use CWT as a *scoring primitive* (regime divergences, relational
   kNN distance) don't lag-window — they compute one-shot per-bar
   scalogram outputs and apply scalar reductions. Their migration is
   "swap the wavelet, not the bundle": replace `causal_cwt` with
   `causal_cwt_morlet`, take `np.abs(coeffs)**2` as the power signal,
   optionally add `causal_cwt_gaussian` over cumulative log-returns
   as a trend channel.
2. **Each migration is a strategy change, not a refactor.** The
   prior canonical Sharpe numbers (regime walk-forward 0.6, the six
   relational scoreboard winners 1.07–1.13 on Phase-2) were tuned
   against Ricker-based scalograms. Switching to Morlet `|c|^2`
   shifts the per-scale frequency response (Morlet at `omega0=6` is
   narrowband at `1/scale`; Ricker is broadband around `1/scale`).
   **Every migrated checkpoint must re-pass walk-forward before
   live deploy.**

Public API the migrations should consume:

```python
from ss_features import (
    TickerData,                    # per-ticker container
    build_features_and_targets,    # CNN-input bundle (replay only)
    compute_scalogram_polar,       # 4-tuple (|c|, cos, sin, g)
    load_ticker,                   # one-shot loader
    CHANNELS_PER_SCALE,            # = 7
    channels_per_lag,              # n_scales -> 7 * n_scales
)
from ss_wavelets import (
    causal_cwt,                    # real Ricker (kept; do not delete)
    causal_cwt_morlet,             # complex Morlet, bandpass + phase
    causal_cwt_gaussian,           # real Gaussian, lowpass / trend
    DEFAULT_MORLET_OMEGA0,         # = 6
    KERNEL_HALF_EXTENT,            # = 3
)
```

Real Ricker `causal_cwt` stays in `ss_wavelets` as the legacy primitive
— don't delete it. Research scripts and the parked v1 app still
reference it.

### `apps/regime` (live trading) — STRATEGY CHANGE

Files:
- `apps/regime/src/regime/trainer.py:67` — `from ss_wavelets import
  KERNEL_HALF_EXTENT, causal_cwt, precompute_windows`. Used by
  `weights_regime` and `weights_scalogram` for per-scale CWT power
  (recent-vs-historical window divergence).
- `apps/regime/src/regime/inference.py:28` — same import; used by
  the live scoring path.
- `apps/regime/src/regime/persist.py` — `Checkpoint` dataclass should
  gain a `wavelet: str` field (default `"ricker"` for back-compat;
  set to `"morlet"` after migration). Consume in `inference.py` so
  live can mirror train-time wavelet choice.
- `apps/regime/research/optimize_regime.py`,
  `apps/regime/research/backtest_bt.py`,
  `apps/regime/research/backtest_ranking.py` — same import; wire the
  wavelet flag through.

Migration:
- Replace `causal_cwt(prices, scales, lookback)` with
  `np.abs(causal_cwt_morlet(prices, scales, lookback)) ** 2` to get
  Morlet power. The downstream `precompute_windows` signature is
  unchanged.
- Add `--wavelet {ricker,morlet}` CLI flag on `regime train` and
  thread to the trainer / persist into the checkpoint.

Validation gate:
- Run the existing controlled walk-forward eval (`Output/regime-eval-
  rawclose-kernel3.{log,json}` template) on Stooq 2010-2024, 20
  trials per window, both wavelet arms. The Morlet arm must match or
  beat the Ricker arm's median val Sharpe (currently +0.15, mean
  +0.07). If Morlet regresses, the migration is rejected for live —
  Morlet stays research-only.
- After validation, regenerate `Output/regime-v1.json` with the
  Morlet checkpoint and confirm `regime live --dry-run` produces
  weights consistent with the chosen target portfolio.

Risk: the regime trainer's strongest signal is on long scales (126d
win 48% scale weight per the JAX-Adam finding in CLAUDE.md). Morlet
narrowband behaviour at long scales may either sharpen or noise that
signal — empirical question.

### `apps/relational` (live trading) — STRATEGY CHANGE, all six checkpoints

Files (every CWT-touching scoring module):
- `apps/relational/src/relational/scalogram_cache.py:30,66` — the
  Modal-volume-cached `causal_cwt` wrapper. **Migration entry point**:
  add a `wavelet` parameter (`"ricker"` | `"morlet"`) and key the
  cache by it so the existing Ricker cache doesn't get clobbered.
- `apps/relational/src/relational/regime_velocity.py:46,84` —
  `from ss_wavelets import causal_cwt`; thin wrapper over scalogram_cache.
- `apps/relational/src/relational/scoring.py:22` —
  `from ss_wavelets import causal_cwt, precompute_windows`. Core
  fingerprint + power computation.
- `apps/relational/src/relational/fingerprints.py:43` (docstring
  reference + actual consumer downstream) — fingerprint primitives
  consume scalogram cache output.
- `apps/relational/src/relational/empirical_sectors.py:50`,
  `empirical_sectors_gmm.py:58`, `analog_knn.py`, `farthest.py:?`,
  `diversify.py:34`, `regime_velocity.py` — each `weights_*` builder
  consumes `precompute_windows` output. No direct `causal_cwt`
  imports here, but each transitively depends on the scalogram via
  the cache.

Migration:
- Add `wavelet: str = "ricker"` to `RelationalCheckpoint` (in
  `apps/relational/src/relational/persist.py`) so live mirrors
  train-time choice.
- Plumb `wavelet` from `RelationalCheckpoint` through
  `scalogram_cache.compute_or_load` → all six `weights_*` builders.
- For Morlet: power = `np.abs(coeffs) ** 2` (signed `coeffs` is
  meaningless for Morlet — phase lives in `arg`, magnitude in `|c|`).
  Fingerprints currently use signed Ricker coefficients; the Morlet
  equivalent is `np.stack([|c|, cos(arg), sin(arg)], axis=...)` per
  scale, which roughly triples fingerprint dim. Decide if the
  fingerprint should compress this back via `Compression(kind='dwt')`
  or stay full-resolution.

Validation gate:
- Re-run the 8-arm walk-forward Modal entrypoint
  (`apps/relational/scripts/modal/relational_dwt_phase2.py`) on each
  of the six canonical strategies, both Ricker and Morlet arms, on
  the Phase-2 21-ticker pool. The val Sharpe must match or beat the
  current canonical (1.07–1.13).
- After validation, regenerate
  `apps/relational/scripts/build_canonical_checkpoints.py` outputs
  with the Morlet variants. Existing
  `Output/relational-{empirical,gmm,analog,farthest,diversified,velocity}.json`
  stay on Ricker until the per-strategy walk-forward signs off.

Risk: this is six independent strategy revalidations. Almost certainly
some will regress — the Phase-2 wins are mega-cap-specific and
narrow; Morlet's narrowband response may erase whatever specific
spectral feature each scoring family was picking up. Plan for at
least one or two of the six to stay on Ricker permanently.

### `apps/notebook` scalogram visualizers — VIZ ONLY, low priority

Files:
- `apps/notebook/src/ss_notebook/scalogram.py:45` — `ss-scalogram`
  CLI's static composite figure.
- `apps/notebook/src/ss_notebook/scalogram_video.py` — `ss-scalogram-
  video` CLI's day-by-day mp4. (Imports `compute_scalogram_power`
  from the sibling module; no direct `causal_cwt` here.)

Migration:
- Add `--wavelet {ricker,morlet}` flag, default `morlet` once the
  regime/relational migrations land. For Morlet, plot `|c|^2` as the
  heatmap (matches the bandpass-power semantics Ricker plots use).
- Keep Ricker as a fallback so prior scalogram figures in
  `Output/` can be reproduced verbatim.

Risk: low. No live consumer; figures are diagnostic. Defer until
after regime + relational migrations confirm Morlet is the workspace
default.

### Out of scope for this migration

- `apps/v1/` — parked legacy app; its own `v1/util/indicators.py` is
  not the canonical implementation. Leave on Ricker.
- `apps/lie/` — research scaffolding; mostly indicator-shape based,
  doesn't use the CWT bundle.
- `apps/factor/src/factor/indicator_features.py` — uses the
  hand-crafted indicator path (`IndicatorGridConfig`), not CWT.
  Already a separate input pipeline; no migration needed.

### Decision order

1. **Regime first** (simpler, one strategy, one checkpoint). If the
   Morlet arm clears the existing eval bar, we have evidence the
   migration won't kill production.
2. **Relational second**, but expect heterogeneous results across the
   six strategies. Migrate them one at a time; let each pass its own
   walk-forward before swapping the canonical JSON.
3. **Notebook scalograms last** — viz, not blocking anything else.

If regime fails the gate, halt the migration and treat polar Morlet
as a research-only primitive available via direct
`ss_wavelets.causal_cwt_morlet` import. The new bundle stays
canonical only for the SSL trainer in that scenario.

## Review follow-ups — paper-trade can proceed without these

Inline `TODO(review #N)` markers point to the file:line. Grep
`TODO(review` to surface the full backlog.

- **#4** — walk-forward train/val slices double-count the boundary bar
  via end-inclusive pandas `.loc` (`apps/regime/src/regime/trainer.py:522`).
  Research-only, no live impact.
- **#5** — `submit_orders` swallows transport-layer (5xx / connection)
  errors the same way it swallows fractionability rejections. Distinguish
  4xx (skip+log, current) from 5xx (re-raise+abort) so an Alpaca outage
  doesn't silently zero every order. (`ss_portfolio/broker.py:submit_orders`)
- **#6** — `gmm_cluster_pair_weights` produces signed long/short. Not
  exposed via the relational dispatch but importable; document /
  assert long-only invariant at the inference boundary if anyone wires
  it in. (`relational/empirical_sectors_gmm.py:409`)
- **#7** — `rsi` (matrix) and `rsi_strided` use different lag conventions
  (`up[t-1]` vs `up[t]`). Both causal but not interchangeable.
  (`ss_indicators/rsi.py:17`)
- **#10** — `precompute_windows` per-ticker mean is over ALL TIME, not
  causal. Safe under scale-axis-normalized divergences (KL/JS/cosine/L2)
  only — assert that at call sites, or refactor to a causal rolling
  mean before exposing to non-scale-invariant downstream ops.
  (`ss_wavelets/windowing.py:42`)
- **#3 follow-up** — `min_notional` gate in `build_trades` uses full-
  precision notional but ships `round(qty_diff, 6)` qty. Penny-stock
  edge case; surfaced via `rejected_orders` already.
  (`ss_portfolio/broker.py:build_trades`)

## Different prediction problem — pair-spread / drawdown / IV-vs-realized

The +0.012 ceiling is for *cross-sectional return direction* at 297 tickers / 20d. Other targets may carry more signal:

- **Pair-spread mean reversion** — high-IC, low-cap. Pick correlated pairs, predict spread reversion.
- **Drawdown forecasting** — directly relevant to sizing; positive signal here would ship as a risk overlay even at modest IC.
- **IV-vs-realized** — DoltHub IV data is on hand from the relational arc. Predict whether implied vol over-/underestimates realized.

Different prediction problems have different data ceilings; not all are bounded by the +0.012 cross-sectional return-IC limit we hit on indicators / CWT / wider universe / longer horizon.

## Use self-supervised learning to forecast the CWT

We are use patches hold out on the CWT and asking the SSL to guess what they
are. This should now apply to future patches as well. Our val scores of SSL were
okay, not amazing. But it does mean that we have learned some structure of the CWT.

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

## DWT-compression follow-ups (post 2026-05-07 breakthrough)

The 2D DWT keep-LL compression of CWT tiles + relational fingerprints
landed for replay (`ss_features.Compression` + `ss-replay --compress
dwt`) and relational (`extract_fingerprints(..., compression=...)` +
`compress_levels` in `strategy_kwargs`). Phase-2 analog-kNN val
Sharpe moved from 1.07 → 1.11 with a 168→44 fp_dim shrink; canonical
`Output/relational-analog.json` now pins `compress_levels=1`. Open
threads:

### DCT zigzag-keep-top-k variant

Originally specified by the user but deferred — zigzag top-k yields a
flat coefficient vector (loses the 2D `(K, C)` tile shape the CNN
reshape relies on), so a flat-input decoder branch is needed. Stub
already in `Compression.kind='dct'` raises NotImplementedError. Plan:

- Add a `dct_zigzag_keep_top_k` helper to
  `ss_features.compression`. Use `scipy.fft.dctn` over the per-bar
  `(K, S)` tile, traverse coefficients in standard JPEG zigzag order,
  retain the first `keep_top_k`. Output is a flat
  `(n_dates, keep_top_k)` array per CWT-derived stack.
- Replay decoder: add a `--decoder-flat-input` mode (or new decoder
  type `cnn-flat`) that takes the concatenated flat coefficients and
  runs an MLP rather than reshape-then-conv. Disable the K/C reshape
  validation when this mode is active.
- Relational consumer: a flat-vector fingerprint plays nicely with
  the kNN code as-is — `extract_fingerprints` already returns `(n_dates,
  n_tickers, fp_dim)` regardless of how `fp_dim` was assembled. So
  the DCT path is *cheaper to wire into relational than into replay*.
- Test: the same Phase-2 head-to-head harness (`idea_b_analog_knn_dwt`)
  with arms `analog`, `analog-dwt-L1`, `analog-dct-k20`, `analog-dct-k40`.
  If DCT-top-k beats DWT-L1 on Sharpe, the win was about
  energy-concentration (DCT is closer to optimal for piecewise-smooth
  signals) rather than the multiresolution structure of the LL band.

### MACD head pathology (replay)

The 2026-05-07 Modal A/B (cwt-only bundle, baseline + dwt-L1) showed
MACD reconstruction R² ≈ −400 to −1200 across the (n,w) grid in
**both** arms. RSI / CCI / vol heads behaved sensibly (R² in the
0.45–0.92 range zero-shot on CSCO), but MACD alone exploded. Same
training config produced a usable MACD head in earlier replay runs.

Hypotheses to check (no need for Modal — should reproduce locally on
one ticker):

1. Scaling. MACD is unscaled price-difference, can range ±5..±50.
   The other heads are bounded ([0,100] for RSI/CCI, ~[0,0.05] for
   vol). When the multi-head loss sums un-normalised per-target
   MSEs, MACD's term dominates and the optimiser can blow up by
   over-correcting on a few outlier bars.
2. The `macd_fast_grid={8,12,16,24}` introduced in the multi-head
   training expects `slow=2*fast` and `signal=int(fast*3/4)`, but
   the un-conditioned MACD head reads the canonical `(fast=12,
   slow=26, signal=9)` triple. Possible mismatch between target
   computation and head conditioning.
3. Anchor-target wiring: the head supervised on the canonical target
   may be reading the FiLM-conditioned MACD line instead, which uses
   a different slow/signal ratio. The plot title says "MACD" but the
   target may be off-anchor.

Test plan: run `ss-replay AAPL --decoder cnn --targets macd
--cnn-steps 200` (single target, no FiLM grid) → confirm sensible R².
Then add `--macd-fast-grid 8,12,16,24` → reproduce the explosion. If
yes, the bug is in the grid conditioning path.

This is **blocking** on declaring the apps/replay → apps/factor SSL
backbone pipeline trustworthy for live use. Relational doesn't depend
on the backbone, so live trading on the relational checkpoints is
unaffected — but anyone consuming the replay-trained backbone via
`ss_features.load_backbone` for downstream factor scoring should be
aware that the MACD prefix of the `_meta` is currently unreliable.

### Wider-universe DWT validation

CLAUDE.md records that Phase-2 wins for ideas A/B/C/D drop from
Sharpe ~1.1 to ~0.4 when the same code runs on the wider 312-ticker
`stooq_us_long` universe. The DWT-L1 finding (Sharpe 1.07 → 1.11) was
measured *on top of* the Phase-2-specific strategy. We don't know
whether DWT helps, hurts, or is neutral on the wider universe.

Plan: run the same 8-arm Modal entrypoint
(`relational_dwt_phase2.py`) but loaded against
`stooq_us_long` (or a min_history-filtered subset). The kNN inner
loop scales as O(n_dates × n_tickers × cand_pool); on 312 tickers
that's ~15× the Phase-2 work — each arm becomes ~30-45 min. Bump the
function timeout to 4h or split into separate function calls per
arm. Phase-2 entrypoint is the template; only the prep step
(`prep_phase2_prices.py`) needs to be replaced with a wide-universe
loader.

If DWT-L1 *also* wins on the wide universe, this becomes the first
finding in this codebase that beats the Phase-8 universe-degradation
result — would justify a wide-universe canonical checkpoint, not just
Phase-2. If it ties or loses, the result is mega-cap-specific and the
canonical checkpoints stay Phase-2-only.

### Non-Haar wavelet sweep

Haar is the shortest filter (length 2) and produces the blockiest LL
band. Smoother wavelets (db2, sym4, coif1) have longer filters that
blur the LL across more neighbouring time/scale cells before
downsampling — could either preserve more signal (if the
discontinuities the Haar LL captures are noise artifacts) or destroy
the signal Haar was usefully picking up.

One-arm follow-up on Phase-2: sweep `wavelet ∈ {haar, db2, sym4,
coif1}` at L=1 with the same harness. Cheap (~7 min for 4 arms with
the CWT cached). Only worth doing once the wide-universe result and
the rebal-days sweep have run, since those are higher-information
experiments.

## Rebal-days sweep (gates the event-driven trade trigger)

Open question: is the analog-kNN signal genuinely monthly (per the
"regime signal works on monthly-to-biannual horizons" finding in
CLAUDE.md), or does the DWT-L1 daily-Sharpe edge mean the compressed
fingerprint can act on a faster cadence?

Sweep `rebal_days ∈ {5, 10, 20, 40}` on the same Phase-2 universe,
both baseline and DWT-L1 arms. Two outcomes:

1. **Shorter rebal wins** (rebal_days=5 Sharpe ≥ 20-day): the signal
   supports faster action. Then run a divergence-trigger variant —
   daily compute weights, only act if `max(|target - current|) > θ`,
   sweep θ. If trigger beats both fixed-cadence arms net of
   commission, deploy event-driven.
2. **20-day wins or ties**: the underlying signal is monthly. Daily
   cron is for monitoring only; trades stay 20-day fixed.

This experiment **gates** the Modal-cron-event-driven design. Without
it, deploying event-driven is shipping a new strategy with new
hyperparameters and zero backtest evidence.

Cost: 4 rebal-days values × 2 arms = 8 backtests, ~20 min total with
the CWT cached. Run after the current 8-arm and walk-forward results
land.

## Modal-cron live deployment for ss-relational

Goal: replace the laptop-or-VPS execution model with a Modal cron
that fires daily, with monthly trade actions (subject to the
rebal-days sweep above). Deploy plan, in order:

1. **Wait for walk-forward eval to confirm OOS edge.** If the
   2021-2025 val-period Sharpe collapses, none of the rest matters.
2. **Cloud-native kill-switch.** Replace the `~/.relational-killswitch`
   file rail with a `modal.Dict["killswitch_active"]` boolean. Add a
   tiny side-CLI (`ss-relational-killswitch {on,off,status}`) that
   reads/writes the Dict. Without this rail you've removed the
   operator override.
3. **Secrets via `modal.Secret`.** Create `modal.Secret.from_name(
   "alpaca-keys")` containing `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL`. The cron function reads
   them at runtime. Default `BASE_URL` to paper until the pilot
   completes.
4. **Idempotency guard.** `modal.Dict["last_run_date"]` — abort at
   function start if today already ran successfully. Protects against
   Modal retry storms.
5. **Failure webhook.** Slack / Discord / email URL in another
   `modal.Secret`. Fire on: any uncaught exception, kill-switch hit,
   non-empty `rejected_orders`, or weight-diff above some sanity
   threshold (e.g. all-new top-N versus yesterday's top-N — could
   indicate a feed glitch).
6. **Run record persistence.** Each cron invocation appends
   `(date, target_weights, executed_orders, rejections)` to a
   `modal.Volume`-backed parquet so you can reproduce decisions for
   compliance/audit. Modal's stdout logs are useful but ephemeral.
7. **Schedule.** `Cron("30 21 * * 1-5")` — 21:30 UTC fires after
   NYSE close in both winter (21:00) and summer (20:00) DST regimes.
   Don't use a wall-clock-naive cron unless you want to trade an hour
   before close half the year.
8. **Phased rollout.**
   - Week 1-2: cron in `--dry-run`, log decisions, no orders.
     Compare logged decisions against backtest expected positions.
   - Week 3-6: `--live` with `--max-position 0.05` (5% per name)
     pilot. Limit damage from latent bugs.
   - Week 7+: full size (`--max-position 0.25`).

The non-trade Modal cron (daily checks: kill-switch, data freshness,
position drift vs target) is a separable, lower-risk first deploy —
ship it before the trade-submitting cron lands.

Cost: ~1 day of code + ~1 day of testing. The minimum-viable scaffold
above is the deliverable.

