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
