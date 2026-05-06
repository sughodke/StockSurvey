## Strategy Weights

interesting observation, we can represent the trading strategy as a dot product.

Per-rebalance-bar shapes (cleanest form):

```
X  ∈ ℝ^{N × K × F}     # features:  N tickers, K scales, F per-(ticker,scale) channels
W  ∈ ℝ^{K × F}         # strategy:  shared across the universe
s  = einsum('nkf,kf->n', X, W)   ∈ ℝ^N      # cross-sectional scores
r  = π(s)              ∈ ℝ^N     # rebal vector: π = softmax_τ → top-N → water-fill cap
```

`N = n_universe`, `K = len(ALL_SCALES)` for the CWT view (or 1 for the flat indicator
stack), `F` = per-(ticker,scale) channels (recent power, hist power, divergence, etc.).

How the existing strategies map onto `(K, F, W)`:

| Strategy | K | F | W |
|---|---|---|---|
| `weights_regime` (KL/JS/cos/L2) | `|ALL_SCALES|` | 1 (precomputed divergence per scale) | uniform mean over K |
| `optimize_adam` | `|ALL_SCALES|` | 1 | learned softmax-over-K (collapsed to 126d≈48%) |
| `factor` linear head, indicator grid | 1 | 74 (RSI+CCI+MACD+vol+coh) | learned, rank-IC objective |
| `factor` linear head, CNN backbone | — | F_backbone | learned over backbone embedding |

The unified pipeline is just: pick a featurizer that emits `X[N,K,F]`, learn `W[K,F]`
against rank-IC, project through `π`. Everything else is a special case
(fixed W, F=1, or K=1).

**Where the framing breaks:** `π` is *not* linear — softmax temperature, top-N, and
water-fill position cap are all non-linear. The dot product gives cross-sectional
scores per ticker; the universe → weights map is a separate non-linear projection.
The regime trainer's "temperature collapsed to 0.005, weight piled into 126d" finding
lives entirely in `π`, not in `W`.

In this lens we actually have two possible loss functions:

1) Sharpe on the rebal positions
- Rank IC for differentiable and more learnable signals

2) Best Rebal vector calculation
- Cross-entropy(?) loss between proposed rebal vector to best rebal?


## No-backbone IC baseline — encoder vs raw at matched setup (2026-04-30)

Built `identity_backbone(K, F)` in `scoring/backbone.py` — a synthetic
`Backbone` whose `apply` is z-norm + flatten (empty `conv_params`). Lets
`train_scorer` ride directly on flat raw CWT bundles, no encoder. Tested
against the Colab supervised-pretrained backbone at matched topology and
date window (the apples-to-apples comparison the encoder-vs-raw question
demanded).

**Setup (identical between Colab encoder run and local raw run):**
- K=96 lag window, F=33 channels (`ALL_SCALES + extras [1,2]` →
  15 scales × 2 + zscore_stats(2) + returns(1)).
- Date range: 2013-01-29 to 2025-12-11.
- 30-ticker universe (Yahoo close-only).
- `train_frac=0.7`, `rebal_days=5`, `scorer='linear'`, `n_steps=500`,
  `learning_rate=1e-3`, `weight_decay=0`.
- 452 train / 195 val rebalance blocks (exact match — confirms the
  date filtering is identical).

**Results:**

| Metric                       | Encoder (5632-d head) | Raw (3168-d head) |
|------------------------------|----------------------:|------------------:|
| Initial train IC             | -0.0395               | +0.0223           |
| Initial val IC               | +0.0143               | +0.0404           |
| Initial val Sharpe           | +0.645                | +1.190            |
| **Final train IC**           | +0.7226 (overfit)     | +0.3165           |
| **Final val IC**             | +0.0039               | -0.0050           |
| **Final val Sharpe**         | +0.554                | +0.628            |
| Peak val IC during run       | not tracked           | +0.038 @ step 0   |

**Read:** at matched setup, **encoder and raw are essentially tied at
the noise floor** — both end with val IC ≈ 0 and val Sharpe in the
+0.55..+0.63 band. The encoder has more parameters (5632 vs 3168) and
overfits train harder (+0.72 vs +0.32) but neither converts that to
val signal. The raw run's "best" peak is at step 0 — random init
projecting z-normed features happens to correlate slightly with
returns; training degrades it. Not a real learned signal.

**An earlier local run had raw beating encoder by 46% on val Sharpe**
at K=64/F=29/2010-2024 (val Sharpe +0.81 vs Colab's +0.55) — that
gap **does not survive matching** to K=96/F=33/2013-2025. The 2010-2024
result was specific to that date range / smaller feature count. Don't
re-cite it as evidence the encoder is harmful.

**Reusable conclusions:**
- Cross-sectional IC supervision at this scale (30 tickers × ~450
  train rebal bars) is the binding constraint, not the encoder choice.
  Both methods plateau at the noise floor regardless of head capacity.
- Floor at this setup: val IC ≈ 0, val Sharpe ≈ 0.55..0.63 with linear
  scoring. Future architectural changes (SSL pretrain, MLP scorer,
  different conditioning) need to clearly beat this floor to count as
  real progress.
- Don't conclude "the encoder is harmful" or "skip the SSL plan" from
  these numbers — they say neither encoder nor raw helps, which is
  consistent with the supervision being the bottleneck.

**Cheaper levers to try before more architecture work:**
- Larger universe (50-100 tickers, more cross-sectional samples per bar).
- Longer rebal horizon (5d → 21d, closer to classical alpha horizons).
- Cross-sectional demean of forward returns before IC fit
  (Pearson-on-residuals strips out the market beta that no per-ticker
  scorer can predict).

Local script archived at `/tmp/no_backbone_baseline_matched.py`;
reproducing costs ~2 min on CPU (no GPU needed for any of these runs).

## Self Supervised

Yes — and this is where the multi-head replay infrastructure shifts from "the pretrain task" to "the probe that validates the pretrain task." That's actually a cleaner separation of concerns.

## Why SSL generalizes where supervised reconstruction doesn't

Supervised reconstruction (current): `loss = MSE(decoded_RSI, true_RSI) + MSE(decoded_MACD, true_MACD) + ...`. The latent's capacity is rationed *to whatever's needed to reproduce those four targets*. Information in the CWT input that doesn't help reconstruct those four indicators consumes capacity without reducing loss → gets actively suppressed by gradient descent. The latent commits to a specific projection of the CWT before any downstream task gets to vote.

Self-supervised (proposed): `loss = MSE(decoded_input, original_input)` where the encoder sees only a partial view of the input and the decoder must reconstruct the full thing. No external labels. The latent has to encode *enough of the input's full structure* that any masked region can be filled in from the visible context. That structure has to include the multi-scale temporal correlations, cross-scale phase relationships, and per-scale regime dynamics — the exact patterns RSI/MACD are crude scalar summaries of, plus everything else those scalars throw away.

The latent isn't *trying* to be useful for downstream tasks. But because it's encoding the input's full conditional structure, it's *available* for downstream tasks to project onto — without the per-task supervision having to discover the projection from scratch on noisy IC signal.

## Concrete method for our setup

The strongest fit is **masked CWT autoencoding** (the time-series analog of MAE):

```
Input bundle: (K=96 lags, F=33 channels) per bar
              [coeffs per scale | power per scale | mu | std | log-return]

Pretrain forward:
  1. Mask ~40% of (lag, channel) cells — replace with 0 or learned mask token
  2. Encoder (the existing CNN backbone) processes masked input → latent
  3. Decoder (small MLP or transposed-conv stack) predicts the FULL bundle
  4. Loss = MSE on the masked positions only (visible positions don't contribute)
```

The encoder must learn to recover masked CWT power at scale 7 from visible context at scales 3, 12, 21 + the surrounding lags + the rolling z-norm stats. That's *literally* learning multi-scale composition. RSI, MACD, vol fall out of the learned latent as projections you can train probes for after the fact.

Why mask both lag and channel axes (not just one): masking only along the lag (time) axis would let the encoder cheat by interpolating from temporal neighbors of the same scale. Masking across channels too forces it to use cross-scale information.

Why ~40%: empirical sweet spot from MAE literature. Lower mask = too easy, encoder doesn't have to compress; higher mask = too hard, decoder can't recover and gradient is noisy. 40-75% is the typical range for image MAE; for time series 30-50% is more common.

## Validation via the multi-head replay probe

This is where it gets clean. Today's `fit_cnn_multihead` is doing two jobs at once: (a) training the backbone, (b) showing what the backbone encodes via per-target heads. Splitting them gives:

**Probe protocol** (no architecture change to multi-head trainer):
1. Pretrain backbone with masked CWT-AE → save backbone-only npz (no per-target heads).
2. Load that backbone in `fit_cnn_multihead` with `--freeze-backbone` (new flag), train ONLY the per-target heads (FiLM or linear) on top of frozen latent.
3. Read off per-target R² for RSI / MACD / vol / price. These now mean *"how much of indicator X is linearly recoverable from the SSL latent?"* rather than *"how well did we reconstruct X during training?"*

The diagnostic readout from probe R²:
- **High R² across the board** (≥0.85 for RSI/MACD, ≥0.95 for price): the SSL latent captures everything the supervised latent did, *and* probably extra structure those indicators don't summarize. Best case.
- **Indicator R² drops modestly** (0.7–0.85 for RSI): the latent reallocated some capacity away from RSI-specific features into other patterns. Whether that's good depends on whether those other patterns help downstream IC.
- **Indicator R² collapses** (<0.5): the SSL latent doesn't preserve enough structure for known-good signals. Either the mask ratio is too aggressive, the encoder is too small, or pretraining didn't run long enough. Diagnostic, not catastrophic — points at a hyperparameter.

The IC scorer experiment then runs on top of the same SSL-pretrained latent (same `load_backbone` + `train_scorer` plumbing). If val IC goes from "+0.018 inside noise" to "+0.03–0.05 outside noise" while indicator probe R² stays high, we've shown the SSL latent contains the same indicator information *plus* return-predictive structure the supervised latent killed. That's the whole thesis validated.

## What the implementation actually costs

Roughly in increasing order of effort:

1. **`replay/features.py`**: factor masking out as a function `mask_bundle(features, mask_ratio, key) -> (masked_features, mask)`. ~20 lines.
2. **`replay/decoders.py`**: add `fit_cnn_masked_ae(...)` — same backbone init, new decoder head (small MLP that maps backbone output → reconstructs full `(K, F)`), MSE loss on masked positions only. ~80 lines, parallels `fit_cnn_multihead`.
3. **`replay/cli.py`**: new `--decoder masked-ae --mask-ratio 0.4` flags, mutually exclusive with the per-target multi-head path. The npz produced has only backbone weights + `_meta`. ~30 lines.
4. **`replay/cli.py` (probe)**: `--freeze-backbone <npz_path>` flag for the existing multi-head decoder, loads backbone weights and disables backbone gradients. ~20 lines.
5. **Compute**: SSL typically needs 5–10× more pretraining steps than supervised. The current run was 2,000 steps — budget 10–20k for the SSL run. On TPU v5e-1, that's probably 30 min–1 hr instead of 2 min. Manageable.

## Honest risks

- **Mask ratio is a real hyperparameter** and we have to sweep it (40% is a starting guess, not a known-good for CWT bundles specifically).
- **Decoder choice matters** — too weak and the encoder doesn't get useful gradient; too strong and the encoder doesn't have to learn much. Symmetric to encoder is the standard default.
- **SSL can fail silently** — produce a latent that's perfectly self-consistent but useless for downstream. The probe protocol catches that, which is why it's important.
- **Compute budget** — if v5e quota is tight, maybe one full SSL pretrain run per day, not five.

Want me to sketch the masked AE decoder + the `--freeze-backbone` flag as a concrete plan with file/line targets, before writing any code?




## Differences in Search vs Optmize implementations
  Where they fundamentally differ (not just parameter values):
  - Search vs optimize: Optuna treats hyperparameters as a discrete combinatorial space and samples; JAX takes hyperparameters as given and finds an interior
   optimum of a continuous loss surface.
  - Hard vs soft selection: this is the largest source of result divergence. Optuna's hard top-N=5 puts 20% on each of 5 names; JAX's soft top-N over ~1000
  tickers spreads weight across dozens even at low temperature.
  - Sharpe definition: Optuna's bt Sharpe captures intra-block drift; JAX's block Sharpe collapses each rebalance period to a single number. They diverge for
   noisy / fat-tailed strategies.

  ┌─────────────────────┬────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │       Aspect        │            Optuna (research/optimize_regime.py)            │                  JAX-Adam (regime/trainer.py)                  │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Optimizer           │ TPE Bayesian search                                        │ Gradient descent (optax Adam)                                  │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ What's optimized    │ 7 discrete hyperparams (search)                            │ 14 continuous floats (gradient)                                │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Lookback            │ searched, int[40, 252]                                     │ fixed by user                                                  │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ n_tail              │ searched, int[3, lookback//2]                              │ fixed by user                                                  │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Top-N count         │ searched, int[5, 30] (hard pick)                           │ implicit; controlled by learned temperature                    │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Divergence          │ searched, {kl, js, cosine, l2}                             │ fixed by user (--divergence, added today)                      │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Scale set           │ searched: 3 booleans (short/mid/long groups) → 8 combos    │ always all 13 in ALL_SCALES                                    │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Per-scale weighting │ none — chosen scales contribute equally                    │ learned 13-vector via softmax (scale_log_weights)              │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Allocation          │ hard top-N equal-weight (1/top_n each)                     │ soft via softmax(score/temp + log(mask))                       │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Temperature         │ n/a                                                        │ learned single scalar (log_temperature)                        │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Sharpe computation  │ daily-return Sharpe via bt-library backtest                │ block-Sharpe mean/std × √(252/rebal_days) (assumes iid blocks) │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Costs               │ bt commission_fn per side, applied to actual share moves   │ commission_frac × (init_cost + 0.5·L1(Δw)) per block           │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Spread mask         │ NaN-out illiquid scores → excluded from top-N              │ log(mask) added to score → driven to ~0 weight in softmax      │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Train/val structure │ rolling walk-forward (default 5y train / 3y val / 2y step) │ single split via train_frac                                    │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Trains per run      │ n_trials × n_windows (e.g. 50 × 3 = 150)                   │ one                                                            │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Wall time           │ ~30 min for 50 trials × 3 windows                          │ ~25 s for 500 Adam steps after CWT precompute                  │
  ├─────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Output              │ per-window best hyperparams + Sharpe                       │ learned 14-param model + checkpoint JSON                       │
  └─────────────────────┴────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────┘

### Summary
  Search wins for most stock strategies — including this one. The reason is structural, not preference: trading rules are inherently discrete (top-N,
  rebalance frequency, divergence choice), the objective (Sharpe / Calmar) is non-differentiable through realistic costs and selection, and noisy returns
  mean overfitting risk dwarfs gradient efficiency. Walk-forward search is also the closest match to how the strategy will actually fail in production —
  params that win a 3y window are at least independently validated, while a single gradient-optimized model is one continuous interpolation that can hide
  instability.

  Gradient optimization (JAX) earns its keep in two specific cases: (1) the parameter is genuinely continuous and high-dimensional — neural-net weights,
  attention scales, embedding tables — where search can't enumerate; (2) you need end-to-end backprop into a downstream model (the notebook's CNN-on-CWT is
  the right use case). For this regime strategy, the JAX trainer is mostly scaffolding to enable future feature-learning work; the +0.46 result that matters
  came from Optuna and you should treat that as the canonical pipeline.

  Practical recommendation: keep Optuna as the production search; use JAX-Adam only when you start layering learned features on top of the regime score.

## Fundamentals

  The framing is "any combinatorial search problem can be expressed as continuous optimization", which is the bedrock
  of differentiable programming and neural architecture search. For optimize to be a strict superclass of search, four things have to hold simultaneously:

  1. Continuous relaxation of every discrete choice. Each enum-like decision needs a differentiable surrogate that recovers the hard choice in some limit.
  The standard tools:
  - divergence ∈ {kl, js, cosine, l2} → 4 logits → Gumbel-softmax ([Jang/Gu/Poole 2017]) producing a soft mixture that hardens as temperature → 0.
  - top_n = 5 (hard pick) → differentiable top-K (SinkhornSort, OT-based selection, or perturbed optimizers à la Berthet 2020). Soft selection vector that
  becomes one-hot on the top-K at low temperature.
  - lookback ∈ ℤ[40,252] → real-valued + linear interpolation between adjacent integer CWTs, OR Gumbel-softmax over a discretization.
  - use_short/mid/long_scales (booleans) → continuous gates in [0,1] with sparsity prior (L0 regularization à la Louizos 2018).

  2. The objective remains differentiable through the relaxation. Sharpe through realistic costs has kinks at zero-turnover, fees, masking. Either accept a
  smooth surrogate (what ss_portfolio.block_sharpe_with_costs does — assumes iid block returns, smooths the cost) or use straight-through estimators.

  3. The relaxation is tight — at convergence, the soft solution approaches a real discrete one. This requires a temperature anneal schedule. Get the
  schedule wrong and you converge to a "soft mixture of divergences" that has no integer interpretation.

  4. Walk-forward validation has to be baked in. Search naturally gets it from outer loop windowing. To bake it into a single gradient-descent objective,
  you'd need meta-learning (MAML-style: optimize for "post-finetune" performance on val splits) or just an outer-loop search over window splits — at which
  point you've reinvented Optuna.

  The duality: search is "broad sampling, no gradient signal, robust to non-smooth surfaces"; optimize is "focused stepping, full gradient signal, requires
  smooth surfaces". With enough relaxation, either simulates the other — but at a cost. Optimize-as-search needs all four conditions above + a stable
  training regime. Search-as-optimize is just zeroth-order optimization (e.g. CMA-ES, evolutionary strategies) and naturally falls out.

  For our regime strategy specifically: you could in principle replace regime.trainer with a single-shot Adam loop that learns (divergence_logits,
  lookback_continuous, top_k_softrank, scale_gates) with annealing. It would be ~5× longer code, fragile to schedule choices, and almost certainly recover
  worse Sharpe in practice because the optimization landscape from those four relaxations is brutally non-convex — even though the optimum, in theory, is the
   same point.

  The stronger framing isn't "is optimize a superclass" — it's "what's the right tool for this problem geometry". Discrete + non-smooth + noisy + low-dim →
  search. Continuous + smooth + clean + high-dim (e.g. neural net weights) → optimize. The notebook's CNN-on-CWT is the latter; the regime hyperparameter
  selection is the former.

## CWT beyond one stock

  ---
  CWT of the market (e.g., SPY):

  Tells you what the macro volatility regime looks like at every timescale. Concretely:

  - Power across scales = "how much of recent market variance lives at 1-week vs 1-month vs 1-quarter horizons." High short-scale power + low long-scale
  power = choppy trendless tape. The reverse = clean trending market.
  - Recent vs historical divergence (same math we use per-stock) = "the market just changed character." This is essentially what VIX measures, but VIX is one
   number; CWT gives you the per-scale breakdown.
  - Use cases: regime gate for sizing (trade smaller in changing-character periods), abstain rules (skip rebalances when market CWT is in flash-crash
  territory), market-vol-regime conditioning of strategy selection.

  The market CWT is essentially a macro overlay. It tells you when the environment favors your strategy.

  ---
  CWT of a sector (e.g., XLK for tech):

  Tells you about sector-level activity independent of individual names:

  - A sector whose CWT power has just shifted (regime divergence at the sector level) is undergoing rotation — money flowing in or out, sector-wide.
  - High short-scale + low long-scale power within a sector = intra-sector chop (smart money repositioning). Often precedes a sector drawdown.
  - Use cases: sector rotation detection, "where is the action right now" allocation across sectors, cross-sector coherence (pairwise sector CWT correlation
  tells you which sectors are moving together → diversification map).

  ---
  Where the real alpha lives — relational CWTs:

  These are the ones the regime/scalogram strategies don't use today, and they're what a real quant shop would build:

  1. Stock minus sector (excess CWT).

  A stock's CWT divergence is interesting; a stock's CWT divergence net of its sector's CWT divergence is far more interesting. The first might be "AAPL's
  regime shifted because all of XLK shifted." The second isolates the idiosyncratic shift — AAPL doing something its sector isn't. That's a cleaner signal
  because you've controlled for the sector-wide move.

  score[stock, t] = divergence(stock_recent, stock_hist)
                  − divergence(sector_recent, sector_hist)

  This single change could meaningfully improve regime/scalogram by removing the "we just bought 5 tech names because tech moved most" failure mode.

  2. CWT of cross-sectional dispersion.

  Instead of CWT on a price series, CWT on cross_section_std(daily_returns). Measures how spread out stock returns are over time. The output tells you when
  the market is in a stock-pickers' regime vs a macro-driven regime:

  - High dispersion power → lots of name-specific variance → stock-picking strategies (regime, scalogram) should work
  - Low dispersion power → everything moves with SPY → no edge for stock-pickers, sit out

  This is a beautiful strategy-on/off signal that the regime app currently lacks. You'd compute it once on the universe and use it to gate or size all
  per-stock strategies.

  3. CWT of cross-sectional correlation.

  mean(rolling_corr(every_stock, market)) over time, then CWT it. Captures the correlation regime — are we in a "correlations-go-to-1 sell-everything" regime
   (low CWT power, stable high corr) or a "stock-pickers' paradise" (low correlations, high CWT power in the corr series itself)?

  This is the macro analog of the dispersion signal — same idea, different framing.

  4. Cross-sector coherence (sector-pair CWT).

  For each pair of sectors, the CWT-coherence we just talked about for scalogram, applied to sector indices instead of single stocks. Tells you which sectors
   are coupled vs decoupled at which timescales. Useful for:

  - Avoiding correlated bets (don't pick top-divergence in 5 sectors that are all coherent right now — they'll move together)
  - Detecting sector rotation (a previously-coherent sector decoupling)

## Other CWT comments

include CWT of volume
each day has a vector, it is 2d because (n_samples, n_days)

## Forecast-target probe — sign-of-demeaned ties/loses to raw-return target (2026-05-04)

Tested whether replacing the rank-IC training target with a sign-reduced
forecast target lifts val IC over the documented +0.0120 baseline on the
297-ticker stooq_us_long walk-forward universe (`min_history_bars=6500`,
rebal=20d, train=63 / val=39 / step=39 blocks, AdamW lr=1e-2 wd=1e-3,
n_steps=200, linear head). Pearson IC already subtracts per-bar
cross-sectional means inside the correlation, so the demean prescription
from 2026-04-30 is **already structurally present** in the loss — the
new variable is the target *reduction*.

**Setup:**
- Control: `forward_target_kind='log_return'` (existing pipeline).
- Probe:   `forward_target_kind='sign_demeaned'` — `sign(fwd_log_ret −
  cross_sectional_mean(fwd_log_ret))` per bar against the same liquid
  peer set the IC eval mask uses. ±1/0 target.
- Loss stays Pearson IC; Sharpe eval stays on actual block returns.
- Driver: `apps/factor/scripts/forecast_probe_walkforward.py`.

**Results:**

| Metric | Control (log_return) | Probe (sign_demeaned) | Δ |
|---|---:|---:|---:|
| Mean val IC | **+0.0120** | +0.0088 | **−0.0032** (−27%) |
| Median val IC | +0.0168 | +0.0166 | −0.0002 (tied) |
| Mean val Sharpe | +0.440 | +0.379 | −0.060 |
| Pos-val-IC fraction | 5/6 | 5/6 | tied |

Per-window val IC (Δ vs control): w0 −0.009, w1 +0.016, w2 −0.002,
w3 −0.008, w4 −0.014, w5 −0.002. 5/6 windows degrade; only w1
improves. Loss windows lose more (worst −0.014) than the winning
window wins (+0.016).

**Read.** The control reproduces the +0.0120 documented baseline
exactly, validating the wiring. The probe loses cleanly. Mechanism:
Pearson IC on raw forward returns weighs each ticker's contribution by
its squared deviation from the per-bar mean — so big movers (whose
signs are most predictable due to fat-tailed return distributions)
carry more weight than middle-of-the-pack noise. Reducing the target
to ±1 flattens this, treating a +5% mover and a +0.5% mover equally
and discarding the heteroskedasticity-driven asymmetry that was
carrying the +0.012 IC. Median is preserved (the typical window's
signal lives where magnitudes are similar); mean is hurt because the
*strong* windows lost their tails.

**Implication.** +0.0120 is not the floor of this feature/head combo —
it's near the **ceiling**. Throwing away magnitude moves down, not up.
Combined with the 2026-04-30 finding (encoder-vs-raw ties at noise
floor), this strengthens the read that the deterministic indicator
stack saturates the predictable signal in its own descriptive scope.
Lifting past +0.0120 needs a genuinely orthogonal target (e.g. vol
innovation — different prediction problem entirely, not a reduction
of the return target) or a different feature class.

Artifacts: `Output/forecast-probe-{control,probe}-windows.npz`,
`Output/forecast-probe-summary.json`. Reproduce with `uv run python
apps/factor/scripts/forecast_probe_walkforward.py` (~5 min wall on
8-core CPU; feature build ~50 s, two arms ~130 s each).

Don't re-cite "Pearson-on-residuals demean" in isolation as an
unimplemented lever — Pearson IC already does it. The actionable form
of that prescription is target *redefinition* (sign reduction here,
falsified) plus orthogonal target choice (vol innovation, pending).

## Forecast-target probe — vol innovation hits +0.47 val IC but doesn't transfer to Sharpe (2026-05-04)

Follow-up to the sign-demeaned probe. Same 297-ticker walk-forward
config (min_history_bars=6500, rebal=20d, train=63 / val=39 /
step=39 blocks, AdamW lr=1e-2 wd=1e-3, n_steps=200, linear head).
Third arm added: `forward_target_kind='vol_innovation'` —
`log(σ_fwd / σ_trail)` where both vols are realized over `rebal_days`
of squared log returns. Genuinely orthogonal prediction problem (vol
regime change rather than directional return); the trivial
vol-persistence piece is structurally subtracted by the ratio form.

**Three-arm leaderboard:**

| arm           | target           | mean_ic   | median_ic | mean_sh | posfrac | Δ_ic    |
|---------------|------------------|----------:|----------:|--------:|--------:|--------:|
| control       | log_return       | +0.0120   | +0.0168   | +0.440  | 5/6     | +0.0000 |
| probe-sign    | sign_demeaned    | +0.0088   | +0.0166   | +0.379  | 5/6     | -0.0032 |
| **probe-vol** | **vol_innovation** | **+0.4743** | **+0.4735** | +0.515  | **6/6** | **+0.4622** |

Per-window val IC for vol_innovation: +0.4008, +0.4386, +0.4398,
+0.5303, +0.5071, +0.5289. Range 0.40–0.53, basically flat across
windows; train IC range 0.435–0.536 — train and val are tight, so it
isn't overfit. ~40× control on IC.

**This is a real signal, but it's *not* return alpha.** Mean val
Sharpe only nudges +0.075 (0.515 vs 0.440) despite IC blowing up
40×. The reason is that Sharpe is computed against actual block log
returns regardless of what the loss optimizes; the vol_innovation arm
trains a head whose scores correlate strongly with future vol-regime
change but only weakly with future return direction. Picking
top-N-by-vol-expansion produces a portfolio with mediocre risk-adjusted
returns — high-vol expanders go up *or* down with about equal frequency.

**Mechanism is vol clustering, not novel insight.** The
`IndicatorGridConfig` features include rolling vol channels at
multiple windows (`vol_n5, vol_n10, vol_n20, vol_n60, vol_n120,
vol_n252`). The head's task on the vol_innovation target reduces to
"given the ratio of short-window to long-window realized vol at t,
predict log(σ_fwd / σ_trail) over the next 20 bars." This is
fundamentally vol-of-vol persistence — one of the strongest empirical
regularities in finance, well-known since Engle/Bollerslev. The
+0.47 IC is the deterministic indicator stack saturating that
predictability ceiling, not finding new signal.

**What this says about the original question (forecast SSL).** The
result clears the question this whole arc was built around: *can
deterministic indicator features encode a forecast signal at all?*
Yes — for vol forecasting, plainly. The indicator stack is not stuck
at the +0.012 ceiling generically; it's stuck there *for return
forecasting*. Return prediction at this universe is bounded by ~+0.012
val IC because returns are autocorrelation-poor; vol prediction is
bounded by ~+0.47 because vol clusters. The 40× gap is the
fundamental tractability gap between these two prediction problems,
not a feature-engineering deficit.

**Implications for SSL pretrain plan.** A forecast-style SSL using
vol_innovation as the target *would* learn predictive structure (the
loss has signal, unlike a vol-persistence-only sign-of-return SSL).
But the resulting embedding's value to a *return scorer* downstream
is bounded by whether vol forecast information transfers to return
prediction — and our Sharpe nudge says it transfers only weakly
(+0.07 Sharpe). The honest path forward is two-stage:

1. **Use vol forecast as a risk-targeting overlay**, not a direct
   return predictor. Vol-target the existing return scorer's portfolio
   so position sizes shrink ahead of forecast vol expansion. This is
   the natural use of a +0.47-IC vol forecast and is one-script-away
   in `apps/factor` (apply `vol_target_weights` from
   `apps/relational/src/relational/sizing.py` with the head's score
   as the input).
2. **Use vol forecast as a regime gate.** When forecast cross-sectional
   vol is *high*, return signals are noisier; when it's *low*, signals
   may be cleaner. Gate strategy on/off based on forecast vol regime.
   See the dispersion-CWT discussion above — same intuition, with the
   forecast head as the operational handle.

A forecast-SSL pretrain on vol_innovation targets is interesting only
if it improves either of these, not for IC chasing in isolation.

Artifacts: `Output/forecast-probe-{control,probe-sign,probe-vol}-windows.npz`,
`Output/forecast-probe-summary.json`. Reproduce with `uv run python
apps/factor/scripts/forecast_probe_walkforward.py` (~7 min wall on
8-core CPU; feature build ~50 s, three arms ~130 s each).

**Don't celebrate the +0.47 IC as alpha.** Cite it correctly: "the
indicator stack hits +0.47 IC on cross-sectional vol prediction at
20-day horizon, near the well-known vol-clustering ceiling." It
validates the feature pipeline as forecast-capable on a tractable
target; it does not solve the return-prediction problem the +0.012
control was bounded by.

## Vol-target overlay — +0.47 forecast IC does not transfer to Sharpe (2026-05-04)

Operational follow-up to the +0.4743 val-IC vol forecast: use it to
size the existing return scorer's top-N basket via diagonal-cov
vol-target overlay (`σ_p ≈ √Σ (w_i σ_i)²`, scale to `target_vol /
σ_p` clipped at `max_leverage=2.0`). Three portfolio variants on
the same return-scorer top-10 basket per walk-forward window:
EW (1/N), trail-VT (σ from `vol_n20` channel), fcst-VT (σ from
calibrated vol head). Calibration is a 1-D linear regression on the
train slice of `head_score → log(σ_fwd / σ_trail)` since Pearson IC
loss is scale-invariant; mean train calibration `r=+0.41`.

**Aggregate over 6 windows (linear head, n_steps=200, top_n=10,
target_vol=0.15, commission_bps=10, 297-ticker / rebal=20d):**

| variant | mean Sharpe | median | pos-frac | mean gross | Δ vs EW |
|---|---:|---:|---:|---:|---:|
| EW         | +0.215 | +0.270 | 5/6 | 1.00 | — |
| trail-VT   | +0.231 | +0.318 | 4/6 | 1.47 | +0.016 |
| fcst-VT    | +0.232 | +0.219 | 3/6 | 1.48 | +0.017 |

**`fcst-VT − trail-VT = +0.001`. The forecast adds nothing over
trailing vol for sizing.** Per-window deltas are ±0.17 — pure noise
around zero with one big winner (window 2, +0.167) offsetting one
big loser (window 3, −0.171). Mechanism: the forecast head predicts
the *change* `log(σ_fwd / σ_trail)`; for vol-target sizing you need
the absolute *level* of forward σ, and trailing vol already gives
that level (vol clusters → σ_fwd ≈ σ_trail in expectation). The
+0.47 IC measures something orthogonal to what the overlay needs.

EW Sharpe (+0.215) is well below the +0.440 reported by the original
walkforward driver because that driver uses a softmax-temperature
soft top-N (universe-weighted, much less concentrated); this
test uses hard top-10 + commission, which is more turnover-heavy and
small-basket-noisier.

Verdict: vol forecast is real signal but **not operationally useful
as a sizing input** at this universe / horizon. The signal would need
a different operational use (regime gate, options pricing, or
downstream feature for a return scorer) to be tradeable. Driver:
`apps/factor/scripts/vol_overlay_walkforward.py`. Artifacts:
`Output/vol-overlay-{summary.json,windows.npz}`.

## Pure-CWT bundle vs IndicatorGridConfig at matched setup (2026-05-04)

The earlier "indicators forecast, raw CWT doesn't" framing rested on
the 2026-04-30 No-backbone IC baseline at 30 tickers / rebal=5d. The
indicator probe ran at 297 tickers / rebal=20d. Different universe
and horizon — the comparison wasn't apples-to-apples. This driver
re-runs raw CWT at the indicator's setup.

**Featurizer**: pure CWT bundle = `[coeffs, power]` per scale, no
prices, no close, no raw or log returns, no z-norm stats. Confirmed
at the channel-build site (`ss_features.compute_scalogram` → only
those two channel families). The `coeffs` come out of `causal_cwt`,
which z-norms prices over the lookback before the Ricker convolution
— absolute price level is stripped at the wavelet stage.

**Setup**: `K=1` (point-in-time CWT, matching IndicatorGridConfig's
K=1 framing exactly), F=2×13=26 channels (13 scales = ALL_SCALES),
hidden_flat=26 (vs indicator's 74). Same identity backbone (z-norm +
flatten only), same linear head, same walk-forward config, same 297
tickers, rebal=20d. Earlier attempt at K=96 OOM'd a 32GB Mac at the
`(D, 297, 96, 26)` aligned tensor — K=1 stays under 200 MB.

**Results:**

| arm                | mean val IC | median | mean Sharpe | pos-frac |
|--------------------|----------:|----------:|----------:|---------:|
| **cwt-return**     | +0.0091   | +0.0059   | **+0.461** | 4/6      |
| **cwt-vol**        | +0.2165   | +0.2131   | +0.453    | 6/6      |
| indicator-control  | +0.0120   | +0.0168   | +0.440    | 5/6      |
| indicator-vol      | **+0.4743** | +0.4735 | +0.515    | 6/6      |

**Read.**

1. **CWT and indicators are statistically tied on return prediction**
   (+0.0091 vs +0.0120, both at the noise-floor `+0.012` we've been
   citing as the "indicator ceiling"). CWT's *Sharpe* is actually
   slightly higher (+0.461 vs +0.440). The +0.012 IC ceiling is a
   **return-prediction ceiling at this universe / horizon**, not an
   indicator-specific ceiling — different bases get to roughly the
   same place.
2. **Indicators dominate CWT on vol prediction by ~2×** (+0.4743 vs
   +0.2165). CWT power *is* a multi-horizon vol signature, and a
   linear head should approximate `vol_n20` from scale combinations
   `(15, 21, 26)` — but it gets ~half the IC. CWT power and realized
   vol of returns are not the same object: CWT applies causal Ricker
   convolution to *z-normed* prices, not squared returns directly.
   The mapping is related but lossy. Explicit `vol_n{k}` channels at
   the right window are strictly stronger as a linear feature for
   forecasting realized vol.
3. **6/6 windows positive on vol for both bases** — both featurizers
   carry robust vol signal; only the magnitude differs.

**Implications for prior work and SSL plan:**

- Don't relabel earlier work as "should have used CWT" or "should
  have used indicators" — for *return* prediction (the load-bearing
  target) they are equivalent within noise.
- For SSL pretraining, raw CWT input has **no advantage over
  indicators** at the linear-head capacity. The only argument for
  CWT-based SSL is that a learned encoder might extract nonlinear
  structure that linear-on-CWT misses (where linear-on-indicators
  cannot, because indicators are already nonlinear summaries). That
  is a thesis to test, not an established result.
- The 2026-04-30 "encoder vs raw both at noise floor" finding now
  has a counterpart: at 297 tickers / 20d, raw CWT *does* lift off
  the noise floor on vol (+0.22 IC) and ties indicators on returns.
  Universe size mattered, not the encoder.

Driver: `apps/factor/scripts/cwt_bundle_walkforward.py`. Artifacts:
`Output/cwt-bundle-{summary.json,cwt-return-windows.npz,cwt-vol-windows.npz}`.

## Feature augmentation — vol forecast as 75th feature is ignored by the return head (2026-05-04)

Third operational test of the +0.4743 vol forecast: feed it as a
deterministic feature to the return scorer. Per walk-forward window:
train vol head on train slice → calibrate score to log(σ_fwd / σ_trail)
via lstsq → apply across full date range → concat as a 75th channel
to the IndicatorGridConfig 74-channel base → train two return heads
from scratch with the same seed (one on 74 base channels, one on the
75 augmented channels) → compare val IC + val Sharpe.

**Aggregate over 6 windows** (linear head, n_steps=200, AdamW lr=1e-2
wd=1e-3, 297-ticker / rebal=20d):

| metric | base (74ch) | aug (75ch) | Δ |
|---|---:|---:|---:|
| mean val IC      | +0.0120 | +0.0117 | **−0.0004** |
| median val IC    | +0.0168 | +0.0164 | −0.0005 |
| pos-val-IC frac  | 5/6     | 5/6     | 0 |
| mean val Sharpe  | +0.446  | +0.442  | −0.004 |

Per-window Δ ranges −0.0032 to +0.0015 — pure noise around zero.
Mean cal corr (vol head score → log-vol-ratio target on train) =
+0.415, so the calibration is good and the forecast feature is
genuinely informative about vol — the head just doesn't find it
useful for *return* direction.

**Smoking gun: forecast-feature L1 share = 0.014, exactly uniform
allocation (1/75 = 0.0133).** Gradient descent looked at the +0.47-IC
vol forecast channel and put no more weight on it than any other
random direction. This is the strongest possible null — the head
isn't even *trying* to use the forecast; it's actively neutral on it.

**This closes the operational test arc on the +0.47 vol forecast.**
Three pathways tested (sign reduction, sizing overlay, feature
augmentation), all null. The signal is real but **operationally
orthogonal** to cross-sectional return prediction at 297 tickers /
20d. Vol clustering and return direction live on different cross-
sectional axes.

**Implication for the broader research program.** The original arc
(deterministic indicators → forecast SSL → predictive embedding →
vector-relational ops) is bounded above by the +0.012 return-IC
ceiling that holds across CWT, indicators, and indicators-augmented-
with-vol-forecast. SSL pretraining on this universe/horizon would
inherit that ceiling. **The data is the bottleneck, not the
architecture.**

Honest pivots from here (not continuations of this arc):
1. **Different horizon** — `rebal=63d` (quarterly). Cross-sectional
   return autocorrelation is documented to be larger at classical
   alpha horizons than 20d. Cheapest screen of "is +0.012 horizon-
   bound or universe-bound?".
2. **Wider universe** — drop `min_history_bars` to ~2500 (~10y),
   admit smaller-cap less-arbed names. Trades date axis for cross-
   section.
3. **Different prediction problem** — pair-spread mean reversion,
   drawdown forecasting, options IV-vs-realized (DoltHub IV from the
   relational arc is on hand).
4. **Regime gate** — the one operational use of the +0.47 vol
   forecast not yet tested. Use aggregate forecast vol to turn
   return strategies on/off rather than resize them.

Driver: `apps/factor/scripts/feature_aug_walkforward.py`. Artifacts:
`Output/feature-aug-summary.json`. Reproduce ~10 min wall.

## Horizon pivot — quarterly is worse, not better (2026-05-04)

First of two pivots from the dead-end indicator/vol arc: try `rebal=63d`
(quarterly, classical alpha horizon) at the same 297-ticker universe.
Window blocks scaled by `20/63` so train / val durations stay
comparable in years (train=20 blocks ≈ 5y, val=12 ≈ 3y, step=12 ≈ 3y).
6 walk-forward windows fit. Same `IndicatorGridConfig` (74 channels),
same linear head, same n_steps=200 / lr=1e-2 / wd=1e-3.

**Results:**

| arm           | rebal | target           | mean_ic   | median_ic | mean_sh   | posfrac |
|---------------|------:|------------------|----------:|----------:|----------:|--------:|
| q-return      |   63d | log_return       | **−0.0019** | +0.0062  | +0.654    | 4/6     |
| q-vol         |   63d | vol_innovation   | +0.3439   | +0.3458   | +0.631    | 6/6     |
| control (20d) |   20d | log_return       | +0.0120   | +0.0168   | +0.440    | 5/6     |
| vol (20d)     |   20d | vol_innovation   | +0.4743   | +0.4735   | +0.515    | 6/6     |

**Read.**

1. **Return-prediction skill *falls* at quarterly**, from +0.012 mean
   val IC to −0.002 (basically zero). 4/6 windows positive vs 5/6 at
   20d. The +0.012 ceiling is **not horizon-bound** in the direction
   we hoped — going longer makes it worse, not better. The hypothesis
   that "classical alpha horizons carry more cross-sectional return
   signal" doesn't survive on this universe.
2. **Vol IC drops too** (+0.47 → +0.34, −28%). Vol clustering
   autocorrelation decays with horizon; predicting 63-day forward vol
   regime is harder than 20-day. Still 6/6 positive — vol is just less
   sharply forecast-able at longer windows.
3. **Sharpe rises for both arms despite IC falling.** This is
   mechanical, not skill: quarterly rebal pays one-sided commission
   1/3 as often as 20d, so even a near-zero-IC portfolio gains Sharpe
   from lower turnover. The +0.654 q-return Sharpe at IC ≈ 0 is *not*
   a return-prediction success — it's the cost of trading falling.

**Implication.** The +0.012 ceiling is a property of cross-sectional
return-prediction signal at this universe, not at this horizon. The
remaining diagnostic lever is universe size: does the same indicator
stack at the same 20-day horizon hit higher IC on a wider universe?
(NOTES 2026-04-30 listed "larger universe (50-100 tickers)" as the
top supervision-side lever.)

Driver: `apps/factor/scripts/horizon_pivot_walkforward.py`. Artifacts:
`Output/horizon-pivot-{summary.json,q-return-windows.npz,q-vol-windows.npz}`.
Reproduce ~3 min wall.

## Universe pivot — 7× wider universe doesn't lift the ceiling either (2026-05-06)

Second of the two pivots. The curated `stooq_us_long` subset is
pre-filtered for long histories — dropping `min_history_bars` from
6500 to 2500 only adds 14 tickers (297 → 311). Real wider-universe
test has to come from the full `StooqData/` archive (12K tickers
incl. delisted).

**Universe construction.** `load_stooq_matrix(StooqData,
min_history=3500)` returns 2404 raw US tickers. The first-valid-date
distribution shows a hard gap: 305 tickers start by 2000-01-01, then
only 9 in 2001-2004, then 1858 in 2005-2009. To capture the 2005-2009
cohort we set `start_grace_days=3650` (10y), giving a target start
date of 2010-01-01 and **2162 keep-tickers**. The common date axis
stays at 2000-2026 because `load_stooq_matrix` returns the full
panel with NaN where untraded — late-listing tickers' valid mask
just kicks in later, and per-bar cross-section grows over time.

**Critical bug fix (drove a 14% build success rate to 96%).** The
naive feature-build path had `valid.sum() == 0` for late-listing
tickers because `ss_indicators.macd` seeds its EMA on the first
sample — if `prices[0] == NaN`, the EMA is NaN forever and all 18
MACD channels are NaN, which AND-fails the per-bar valid mask. Fix
in `_build_one_ticker_args`: trim leading NaN before
`build_indicator_features`, then pad features and valid mask back
onto the full date axis so `align_tickers` sees a single common
axis. Without this, 1868 / 2162 tickers got dropped silently.

**Recovery via Modal.** The local 297→2073 walkforward crashed
mid-vol-arm (laptop crash). Re-ran the vol arm via Modal-T4 with the
parallel `mp.Pool` feature-build pattern from
`apps/factor/scripts/modal/train_indicator.py`. Local prep step
pickles the 2162-ticker close DataFrame (~109 MB) and ships it
through Modal RPC; the remote function builds TickerData per column
in parallel (24 workers on T4 instance, 53s for 2162 tickers) and
runs the walkforward (~87s on T4). Driver:
`apps/factor/scripts/modal/universe_pivot_vol_arm.py` + prep helper
`prep_universe_pivot_data.py`. First Modal run timed out at 60min
because the feature-build was sequential — bumped `cpu=8`,
`timeout=2*60*60`, used `mp.Pool` inside the remote.

**Results (vs documented 297-ticker baseline at same setup):**

| arm           | universe | mean_ic    | median_ic | mean_sh   | posfrac |
|---------------|---------:|-----------:|----------:|----------:|--------:|
| wide-return   |     2073 | **+0.0106** | +0.0092   | +0.205    | 3/6     |
| wide-vol      |     2073 | **+0.4091** | +0.4619   | +0.366    | 6/6     |
| narrow-return |      297 | +0.0120    | +0.0168   | +0.440    | 5/6     |
| narrow-vol    |      297 | +0.4743    | +0.4735   | +0.515    | 6/6     |

Per-window val IC for wide-vol: `[0.449, 0.135, 0.438, 0.479, 0.479,
0.475]` — 5 of 6 windows in the +0.44..+0.48 range, w1 outlier at
+0.135. Stable signal across the late-period windows.

**Read.**

1. **Return IC ties at noise floor.** +0.0106 (wide) vs +0.0120
   (narrow) — within 0.003 of each other, both at the documented
   +0.012 ceiling. **7× the universe doesn't lift return prediction.**
   The ceiling is data-side, not supervision-side. The 2026-04-30
   prescription "larger universe (50-100 tickers)" was right
   directionally for sharpening the *measurement* but doesn't
   translate to lifting the *signal* ceiling — the underlying
   cross-sectional return-predictability at 20d horizon on US equities
   simply isn't there to extract.
2. **Vol IC drops slightly** (+0.41 vs +0.47, −13%). Possible cause:
   the wider universe includes more late-listing tickers whose
   per-bar mask kicks in later, making early-window cross-sections
   more variable in size. The signal is still robust (6/6 positive)
   but slightly noisier.
3. **Sharpe drops** for both arms in the wider universe (return
   +0.205 vs +0.440; vol +0.366 vs +0.515). Likely commission /
   per-bar mask variation effect on softmax-temperature top-N
   weighting at the larger panel — when the universe size varies
   per-bar (late-listing names invalid in early bars), turnover
   structure changes.

**Implication: both pivots from the dead-end arc closed.** Quarterly
horizon (NOTES above): IC fell to ~zero. Wider universe: IC tied at
+0.012. The +0.012 return-prediction ceiling is a property of the
data — US equity 20d cross-sectional return signal at this scale —
not a property of the indicator stack, the universe filter, the
horizon choice, or the SSL pretraining state. Honest move: accept
this and pivot to a different prediction problem (pair-spread,
drawdown, IV-vs-realized) or a different operational use of the +0.41
vol forecast (regime gate, options pricing).

Drivers: `apps/factor/scripts/universe_pivot_walkforward.py`
(local), `apps/factor/scripts/modal/universe_pivot_vol_arm.py` (Modal
T4 recovery). Artifacts: `Output/universe-pivot-{summary.json,
wide-return-windows.npz, wide-vol-windows.npz, close.pkl}`.

## Regime gate — aggregate forecast vol does not gate the return signal (2026-05-06)

Fourth and final operational test of the +0.4743 vol forecast. Three
prior pathways closed (sign-of-demeaned `def3ac9`, vol-target overlay
`9cbf8bb`, feature augmentation `75839f8`) — all *return-prediction-
via-vol* mechanisms. The remaining untested pathway is the **regime
gate**: instead of resizing positions continuously off the forecast,
flip the strategy ON/OFF per rebal bar based on aggregate forecast vol.

Hypothesis: when aggregate forecast vol is high, the cross-sectional
return signal is noisier (returns dominated by macro / surprise events
not anticipated by the indicator stack). When forecast vol is low,
signals are cleaner. Sit out the high-forecast-vol bars → fewer losing
periods → higher Sharpe even if mean return drops.

Per walk-forward window: train return + vol heads; calibrate vol head
on train slice via lstsq → `(intercept, slope)`; apply to val to get
per-(bar, ticker) `σ_fcst[t,i] = σ_trail[t,i] · exp(calibrated_log_ratio)`;
aggregate per bar = mean over active tickers. Threshold = 70th / 80th /
90th percentile of train-slice aggregate forecast vol. On val: weights
= top-10 return basket if `agg_val[t] < threshold`, else 0 (cash).

**Aggregate over 6 windows (linear head, n_steps=200, top_n=10,
commission_bps=10, 297-ticker / rebal=20d):**

| variant | mean Sharpe | median | pos-frac | mean off | mean gross | Δ vs always |
|---|---:|---:|---:|---:|---:|---:|
| always-on | +0.215 | +0.270 | 5/6 | 0.00 | 1.00 | — |
| gate-70   | +0.015 | +0.192 | 4/6 | 0.44 | 0.56 | −0.200 |
| gate-80   | +0.195 | +0.145 | 5/6 | 0.31 | 0.69 | −0.020 |
| gate-90   | +0.240 | +0.292 | 5/6 | 0.21 | 0.79 | +0.025 |

Mean train calibration r=+0.414 (consistent with the +0.47 head IC).
Per-window deltas across gates are large (±1.0), driven by win 1 (gate-70
sits out 69% and crashes a +1.04 Sharpe to +0.31) and win 4 (gate-70
sits out 87% and drops a +0.14 always-on to −1.02). Gate-90 fires
rarely (mean 21% off) and its lift is within noise.

**Regime gate either ties or degrades the always-on baseline.**
Strictest threshold (gate-70, 44% off) is the worst arm — sitting out
the most bars destroys the signal rather than concentrating it. The
two windows where gate-70 most aggressively triggers (win 1, win 4)
are precisely the windows where always-on did *fine* — high forecast
vol does not coincide with bad-return regimes for this signal. Gate-90
nudges +0.025 with 21% off, but per-window pos-frac stays at 5/6 just
like always-on, and the median Sharpe lift (+0.292 vs +0.270) is well
inside ±0.17 noise envelope from the vol-overlay null. Median across
all four variants stays in the +0.15..+0.29 band.

**This closes the operational test arc on the +0.47 vol forecast for
the second time.** Four pathways tested (sign reduction, sizing
overlay, feature augmentation, regime gate), all null. The signal is
real but **operationally orthogonal** to cross-sectional return
prediction at 297 tickers / 20d. Forecast vol does not align with
return-signal quality regimes. Vol forecast remains a candidate for
options pricing or non-cross-sectional uses (drawdown forecasting,
single-name vol-trading), but it does not gate, size, augment, or
direction-flip the return scorer at this universe / horizon.

Driver: `apps/factor/scripts/regime_gate_walkforward.py`. Artifacts:
`Output/regime-gate-{summary.json,windows.npz}`. Reproduce ~5 min wall.
