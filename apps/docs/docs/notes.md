# Notes

Durable conceptual notes that don't fit the per-experiment shape of
the [Leaderboard](leaderboard.md) or [Findings](findings/index.md).
Most of these are framing — design intuitions, hypothesis sketches,
and theoretical context — that should still travel with the project
even after the specific experiments they motivated have run.

## Strategy as a dot product

Per-rebalance-bar shapes (cleanest form):

```
X  ∈ ℝ^{N × K × F}     # features:  N tickers, K scales, F per-(ticker,scale) channels
W  ∈ ℝ^{K × F}         # strategy:  shared across the universe
s  = einsum('nkf,kf->n', X, W)   ∈ ℝ^N      # cross-sectional scores
r  = π(s)              ∈ ℝ^N     # rebal vector: π = softmax_τ → top-N → water-fill cap
```

`N = n_universe`, `K = len(ALL_SCALES)` for the CWT view (or 1 for
the flat indicator stack), `F` = per-(ticker,scale) channels (recent
power, hist power, divergence, etc.).

How the existing strategies map onto `(K, F, W)`:

| Strategy | K | F | W |
|---|---|---|---|
| `weights_regime` (KL/JS/cos/L2) | `|ALL_SCALES|` | 1 (precomputed divergence per scale) | uniform mean over K |
| `optimize_adam` (deleted) | `|ALL_SCALES|` | 1 | learned softmax-over-K (collapsed to 126d≈48%) |
| `factor` linear head, indicator grid | 1 | 74 (RSI+CCI+MACD+vol+coh) | learned, rank-IC objective |
| `factor` linear head, CNN backbone | — | F_backbone | learned over backbone embedding |

The unified pipeline is just: pick a featurizer that emits `X[N,K,F]`,
learn `W[K,F]` against rank-IC, project through `π`. Everything else is
a special case (fixed W, F=1, or K=1).

**Where the framing breaks:** `π` is *not* linear — softmax temperature,
top-N, and water-fill position cap are all non-linear. The dot product
gives cross-sectional scores per ticker; the universe → weights map is
a separate non-linear projection. The
[deleted JAX-Adam regime trainer](https://github.com/sughodke/StockSurvey/commit/68ee595)'s
"temperature collapsed to 0.005, weight piled into 126d" finding lived
entirely in `π`, not in `W`.

Two possible loss functions follow naturally:

1. **Sharpe on the rebal positions** — what `block_sharpe_with_costs`
   approximates. Differentiable through the costs and selection if you
   relax everything (see [search vs optimize](#search-vs-optimize-picking-the-right-tool)
   below).
2. **Rank IC on the cross-sectional scores `s`** — what
   `apps/factor` actually trains on. Differentiable, more learnable on
   noisy returns because it sidesteps the non-linear `π`.

## Self-supervised pretrain — why and how

### Why SSL generalises where supervised reconstruction doesn't

Supervised reconstruction (the original replay-CNN setup):
`loss = MSE(decoded_RSI, true_RSI) + MSE(decoded_MACD, true_MACD) + ...`.
The latent's capacity is rationed *to whatever's needed to reproduce
those four targets*. Information in the CWT input that doesn't help
reconstruct those four indicators consumes capacity without reducing
loss → gets actively suppressed by gradient descent. The latent
commits to a specific projection of the CWT before any downstream task
gets to vote.

Self-supervised: `loss = MSE(decoded_input, original_input)` where the
encoder sees only a partial view and the decoder must reconstruct the
full thing. No external labels. The latent has to encode *enough of
the input's full structure* that any masked region can be filled in
from visible context. That structure has to include multi-scale
temporal correlations, cross-scale phase relationships, and per-scale
regime dynamics — the exact patterns RSI/MACD are crude scalar
summaries of, plus everything else those scalars throw away.

The latent isn't *trying* to be useful for downstream tasks. But
because it's encoding the input's full conditional structure, it's
*available* for downstream tasks to project onto — without the
per-task supervision having to discover the projection from scratch on
noisy IC signal.

### Concrete method for our setup

The strongest fit is **masked CWT autoencoding** (the time-series
analog of MAE):

```
Input bundle: (K=96 lags, F=33 channels) per bar
              [coeffs per scale | power per scale | mu | std | log-return]

Pretrain forward:
  1. Mask ~40% of (lag, channel) cells — replace with 0 or learned mask token
  2. Encoder (the existing CNN backbone) processes masked input → latent
  3. Decoder (small MLP or transposed-conv stack) predicts the FULL bundle
  4. Loss = MSE on the masked positions only (visible positions don't contribute)
```

The encoder must learn to recover masked CWT power at scale 7 from
visible context at scales 3, 12, 21 + the surrounding lags + the
rolling z-norm stats. That's *literally* learning multi-scale
composition. RSI, MACD, vol fall out of the learned latent as
projections you can train probes for after the fact.

Why mask both lag and channel axes (not just one): masking only along
the lag (time) axis would let the encoder cheat by interpolating from
temporal neighbors of the same scale. Masking across channels too
forces it to use cross-scale information.

Why ~40%: empirical sweet spot from MAE literature. Lower mask = too
easy, encoder doesn't have to compress; higher mask = too hard,
decoder can't recover and gradient is noisy. 40-75% is the typical
range for image MAE; for time series 30-50% is more common.

### Validation via the multi-head replay probe

Today's `fit_cnn_multihead` does two jobs at once: (a) trains the
backbone, (b) shows what the backbone encodes via per-target heads.
Splitting them gives:

**Probe protocol** (no architecture change to multi-head trainer):

1. Pretrain backbone with masked CWT-AE → save backbone-only npz
   (no per-target heads).
2. Load that backbone in `fit_cnn_multihead` with `--freeze-backbone`,
   train ONLY the per-target heads (FiLM or linear) on top of frozen
   latent.
3. Read off per-target R² for RSI / MACD / vol / price. These now mean
   *"how much of indicator X is linearly recoverable from the SSL
   latent?"* rather than *"how well did we reconstruct X during
   training?"*

The diagnostic readout from probe R²:

- **High R² across the board** (≥0.85 for RSI/MACD, ≥0.95 for price):
  the SSL latent captures everything the supervised latent did, *and*
  probably extra structure those indicators don't summarize.
- **Indicator R² drops modestly** (0.7–0.85 for RSI): the latent
  reallocated some capacity. Whether that's good depends on whether
  those other patterns help downstream IC.
- **Indicator R² collapses** (<0.5): SSL latent doesn't preserve enough
  structure for known-good signals. Diagnostic, not catastrophic —
  points at a hyperparameter (mask ratio, encoder size, run length).

The IC scorer experiment then runs on top of the same SSL-pretrained
latent (same `load_backbone` + `train_scorer` plumbing). If val IC
goes from "+0.012 inside noise" to "+0.03–0.05 outside noise" while
indicator probe R² stays high, we've shown the SSL latent contains the
indicator information *plus* return-predictive structure the
supervised latent killed. That's the whole thesis validated.

### Honest risks

- **Mask ratio is a real hyperparameter** — 40% is a starting guess,
  not known-good for CWT bundles specifically.
- **Decoder choice matters** — too weak and the encoder doesn't get
  useful gradient; too strong and the encoder doesn't have to learn
  much. Symmetric to encoder is the standard default.
- **SSL can fail silently** — produce a latent that's perfectly
  self-consistent but useless for downstream. The probe protocol
  catches that.
- **Compute budget** — SSL typically needs 5–10× more pretraining
  steps than supervised. Plan accordingly.

### What we already know about supervision being the binding constraint

Before SSL was implemented, a controlled apples-to-apples test pitted
the Colab supervised-pretrained backbone against an `identity_backbone`
(z-norm + flatten, no encoder) at matched topology / date window /
universe. **Both arms ended at val IC ≈ 0 and val Sharpe in the
+0.55..+0.63 band** — encoder and raw tied at the noise floor. The
encoder overfit train harder (+0.72 vs +0.32) but that didn't
transfer.

Read: cross-sectional IC supervision at the original 30-ticker scale
is the binding constraint, not the encoder choice. Both methods
plateau at the noise floor regardless of head capacity. Don't conclude
"the encoder is harmful" or "skip the SSL plan" from those numbers —
they say neither encoder nor raw helps, which is consistent with the
supervision being the bottleneck. The Leaderboard's later universe-
pivot row (2073-ticker / rebal=20d, val IC tied at
[+0.012](findings/factor-indicator-baseline.md)) confirmed the IC
ceiling is data-side at this prediction problem, not supervision-side
at small `N`.

## Search vs optimize — picking the right tool

### Where they fundamentally differ

| Aspect | Optuna (`research/optimize_regime.py`) | JAX-Adam (deleted) |
|---|---|---|
| Optimizer | TPE Bayesian search | Gradient descent (optax Adam) |
| What's optimized | 7 discrete hyperparams (search) | 14 continuous floats (gradient) |
| Lookback | searched, int[40, 252] | fixed by user |
| n_tail | searched, int[3, lookback//2] | fixed by user |
| Top-N count | searched, int[5, 30] (hard pick) | implicit; controlled by learned temperature |
| Divergence | searched, {kl, js, cosine, l2} | fixed by user |
| Scale set | searched: 3 booleans (short/mid/long groups) → 8 combos | always all 13 in `ALL_SCALES` |
| Per-scale weighting | none — chosen scales contribute equally | learned 13-vector via softmax (`scale_log_weights`) |
| Allocation | hard top-N equal-weight (1/top_n each) | soft via `softmax(score/temp + log(mask))` |
| Temperature | n/a | learned single scalar (`log_temperature`) |
| Sharpe computation | daily-return Sharpe via bt-library backtest | block-Sharpe `mean/std × √(252/rebal_days)` (assumes iid blocks) |
| Costs | `bt` commission_fn per side, applied to actual share moves | `commission_frac × (init_cost + 0.5·L1(Δw))` per block |
| Spread mask | NaN-out illiquid scores → excluded from top-N | `log(mask)` added to score → driven to ~0 weight in softmax |
| Train/val structure | rolling walk-forward (default 5y train / 3y val / 2y step) | single split via `train_frac` |
| Trains per run | n_trials × n_windows (e.g. 50 × 3 = 150) | one |
| Wall time | ~30 min for 50 trials × 3 windows | ~25 s for 500 Adam steps after CWT precompute |
| Output | per-window best hyperparams + Sharpe | learned 14-param model + checkpoint JSON |

### Summary

Search wins for most stock strategies — including this one. The reason
is structural, not preference: trading rules are inherently discrete
(top-N, rebalance frequency, divergence choice), the objective
(Sharpe / Calmar) is non-differentiable through realistic costs and
selection, and noisy returns mean overfitting risk dwarfs gradient
efficiency. Walk-forward search is also the closest match to how the
strategy will actually fail in production — params that win a 3y
window are at least independently validated, while a single
gradient-optimized model is one continuous interpolation that can hide
instability.

Gradient optimization earns its keep in two specific cases: (1) the
parameter is genuinely continuous and high-dimensional — neural-net
weights, attention scales, embedding tables — where search can't
enumerate; (2) you need end-to-end backprop into a downstream model
(`apps/replay`'s CNN-on-CWT is the right use case). For the regime
strategy, the JAX trainer was mostly scaffolding; it has been
[deleted](https://github.com/sughodke/StockSurvey/commit/68ee595)
(and the workspace's
[JAX dependency dropped](https://github.com/sughodke/StockSurvey/commit/191d787))
— Optuna is the canonical pipeline.

### Why "optimize is a strict superclass of search" is misleading

The framing "any combinatorial search problem can be expressed as
continuous optimization" is the bedrock of differentiable programming
and neural architecture search. For optimize to be a strict superclass
of search, four things have to hold simultaneously:

1. **Continuous relaxation of every discrete choice.** Each enum-like
   decision needs a differentiable surrogate that recovers the hard
   choice in some limit:
    - divergence ∈ {kl, js, cosine, l2} → 4 logits → Gumbel-softmax
      hardening as temperature → 0.
    - top_n = 5 (hard pick) → differentiable top-K (SinkhornSort,
      OT-based selection, or perturbed optimizers à la Berthet 2020).
    - lookback ∈ ℤ[40,252] → real-valued + linear interpolation
      between adjacent integer CWTs, OR Gumbel-softmax over a
      discretization.
    - use_short/mid/long_scales (booleans) → continuous gates in [0,1]
      with sparsity prior (L0 regularization).
2. **The objective remains differentiable through the relaxation.**
   Sharpe through realistic costs has kinks at zero-turnover, fees,
   masking. Either accept a smooth surrogate or use straight-through
   estimators.
3. **The relaxation is tight** — at convergence, the soft solution
   approaches a real discrete one. This requires a temperature anneal
   schedule. Get it wrong and you converge to a "soft mixture of
   divergences" with no integer interpretation.
4. **Walk-forward validation has to be baked in.** Search naturally
   gets it from outer loop windowing. To bake it into a single
   gradient-descent objective you'd need meta-learning (MAML-style) or
   an outer-loop search over window splits — at which point you've
   reinvented Optuna.

The duality: search is "broad sampling, no gradient signal, robust to
non-smooth surfaces"; optimize is "focused stepping, full gradient
signal, requires smooth surfaces". With enough relaxation, either
simulates the other — but at a cost.

The stronger framing isn't "is optimize a superclass" — it's "what's
the right tool for this problem geometry". Discrete + non-smooth +
noisy + low-dim → search. Continuous + smooth + clean + high-dim
(e.g. neural net weights) → optimize. The replay-CNN is the latter;
regime hyperparameter selection is the former.

## Multi-stock CWT framings

CWT applied to a single stock gives per-scale power for that name.
Many natural extensions follow once you stop thinking "one ticker at a
time."

### CWT of the market (e.g. SPY)

Tells you what the macro volatility regime looks like at every
timescale:

- **Power across scales** = "how much of recent market variance lives
  at 1-week vs 1-month vs 1-quarter horizons." High short-scale +
  low long-scale = choppy trendless tape. The reverse = clean
  trending market.
- **Recent vs historical divergence** (same math we use per-stock) =
  "the market just changed character." Essentially what VIX measures,
  but VIX is one number; CWT gives you the per-scale breakdown.
- Use cases: regime gate for sizing (trade smaller in changing-
  character periods), abstain rules (skip rebalances when market CWT
  is in flash-crash territory), market-vol-regime conditioning of
  strategy selection.

### CWT of a sector (e.g. XLK for tech)

- Sector whose CWT power has just shifted = sector undergoing rotation.
- High short-scale + low long-scale within a sector = intra-sector chop
  (smart money repositioning); often precedes a sector drawdown.
- Use cases: sector rotation detection, "where is the action" allocation
  across sectors, cross-sector coherence (pairwise sector CWT
  correlation → diversification map).

### Where the real alpha may live — relational CWTs

These are the framings the regime/scalogram strategies don't use today;
several are partially implemented in `apps/relational/`.

1. **Stock minus sector (excess CWT).** A stock's CWT divergence net
   of its sector's. Isolates the idiosyncratic shift — AAPL doing
   something its sector isn't. Cleaner than raw stock divergence
   because you've controlled for the sector-wide move.

    ```
    score[stock, t] = divergence(stock_recent, stock_hist)
                    − divergence(sector_recent, sector_hist)
    ```

    Removes the "we just bought 5 tech names because tech moved most"
    failure mode.
2. **CWT of cross-sectional dispersion.** CWT on
   `cross_section_std(daily_returns)` rather than on a price series.
   Tells you when the market is in a stock-pickers' regime vs a
   macro-driven regime:
    - High dispersion power → lots of name-specific variance →
      stock-picking strategies should work.
    - Low dispersion power → everything moves with SPY → no edge for
      stock-pickers, sit out.
3. **CWT of cross-sectional correlation.** `mean(rolling_corr(every_stock,
   market))` over time, then CWT it. Captures the correlation regime —
   are we in a "correlations-go-to-1 sell-everything" regime (low CWT
   power, stable high corr) or a "stock-pickers' paradise" (low
   correlations, high CWT power in the corr series itself)?
4. **Cross-sector coherence.** For each pair of sectors, the
   CWT-coherence applied to sector indices instead of single stocks.
   Useful for avoiding correlated bets and detecting sector rotation
   (a previously-coherent sector decoupling).

### Other primitives worth folding in

- **CWT of volume.** Each day has a 2D vector `(n_samples, n_days)` —
  volume regime as a scalogram alongside price.

## Master target — there isn't one (and what to use instead)

"What target do firms like Jane Street use as a master?" is a
category error in the same way that asking it of our own stack is.
Top quant firms do **not** have a single master prediction target;
they have a hierarchy with very different things at different
levels, and the answer to "what's the master?" depends on which
level you're asking about.

The category error matters because it leaks into research design:
new researchers (and AIs) routinely look for "the right thing to
predict" as if a single answer exists, then either chase a target
that's not tradeable at the firm's horizon or paper-over the
portfolio-construction layer as an afterthought. The honest framing
is that **prediction targets live one level below the master objective,
and a serious shop has many of them**.

### What "master target" decomposes into by firm class

| Firm class | Examples | Master objective | Decomposed into |
|---|---|---|---|
| Market-maker | Jane Street, Citadel Securities, Virtu, IMC, Jump | Expected revenue per unit of risk capacity used | Spread captured − adverse selection − inventory carry − hedging cost |
| Systematic alpha (medium horizon) | Two Sigma, DE Shaw, AQR | Portfolio Sharpe / IR vs benchmark | Many forward-return forecasts × portfolio-construction layer |
| HFT stat-arb | Renaissance Medallion, HRT, XTX | Forward price move at signal's natural horizon | Per-signal microsecond-to-minute point forecasts |
| Discretionary multi-strat | Citadel, Millennium, Point72 | Per-PM Sharpe + drawdown-constrained P&L | Bespoke per pod; allocator allocates based on rolling Sharpe |
| Risk-parity / factor | Bridgewater, AQR factor side | Vol-targeted return per asset class | Forecast covariance + factor premia → optimal portfolio under risk parity |

The key insight: **for market-makers, "what direction will price
move?" is one input among many**, and not the most important one.
Inventory cost dominates at microsecond scales because spreads are
tight and price moves are noisy. The decision-time objective at
Jane Street is roughly:

```
maximize  Σ_quote  [ probability_of_fill × expected_revenue_per_fill
                  − inventory_penalty(current_inventory + Δ)
                  − adverse_selection_cost ]
```

Each term has its own predictor (microstructure model, short-horizon
mid forecast, queue-position model, adverse-selection model). The
"master" is the combined EV. The famous Jane Street EV-puzzle
interview style signals exactly for this composition.

For systematic alpha at our horizon (daily bars, ~20-day holding):
the master is **portfolio Sharpe** decomposed as

```
portfolio_Sharpe = portfolio_construction(  many_alpha_signals
                                          + risk_model
                                          + cost_model
                                          + capacity_constraints )
```

Each signal predicts forward returns at its natural horizon. None
individually is "the master."

### Three questions that determine the right target

Instead of asking "what's the master target?", a research-design
discussion should answer:

1. **At what horizon are you trying to make money?** Microseconds
   (microstructure), days (medium-horizon stat-arb), quarters
   (factor / fundamental). The horizon dictates which signals are
   tradeable.
2. **Are you a price-taker or a price-maker?** Alpha shop vs
   market-maker — fundamentally different objectives. Most of the
   academic finance literature is written from the price-taker
   side, which is why "predict forward returns" is the default
   answer; it's only the right answer for one of the five firm
   classes above.
3. **What's the risk capacity and capital constraint?** Sets the
   λ in `E[return] − λ · risk`. A small fund with high turnover
   and tight risk budget cares about Sharpe; a large allocator
   with multi-decade horizons cares more about expected return
   given an acceptable drawdown.

### How this maps to our stack

We're closest to the **systematic-alpha medium-horizon** class.
Concretely:

| Industry pattern | Our analog | Status |
|---|---|---|
| Many alpha signals at the same horizon | [`factor`](apps/factor.md) (cross-sec return), [`lie`](apps/index.md) v3 (per-ticker shape kNN), eventually more | Have ~2 signals at 20d; need more |
| Risk model (covariance, factor exposures) | `ss_portfolio.apply_position_cap` only | **Missing** the covariance / factor-risk layer |
| Cost model | 10 bps round-trip baked into block-Sharpe eval | Crude — single number, not dynamic |
| Portfolio construction combining signals | None — each scorer outputs weights independently | **Missing** — proposed [`apps/cfr`](TODO/apps-cfr.md) is exactly this layer |
| Master scoring metric | Walk-forward Sharpe minus passive-EW Sharpe (alpha) | Have this — [`passive-ew-benchmark`](findings/passive-ew-benchmark.md) made it load-bearing |
| Risk-side predictors | [`gate`](apps/gate.md) forward-DD; [`lie`](apps/index.md) market-state regime | Have early versions |

The thing we lack that every serious systematic shop has is the
**portfolio-construction layer** — the piece that takes many alpha
signals + a risk model + a cost model + capacity constraints and
produces a target portfolio. Right now each scorer in our stack is a
complete strategy from signal to weights; there is no separation
between **alpha generation** and **portfolio construction**. That
separation is the architectural step the [`apps/cfr`](TODO/apps-cfr.md)
TODO proposes — using existing scorers as the alpha menu, CFR as
the meta-allocator.

### Our master metric, named explicitly

For us, at our horizon and capital constraint, the closest analog
to a "master target" is

> **mean per-window val Sharpe minus passive-EW Sharpe on the same universe**

That's our alpha; everything we predict is in service of pushing
this number up. It was made load-bearing by the
[passive-EW benchmark finding](findings/passive-ew-benchmark.md),
which reclassified three previously "shippable" rows after raw val
Sharpe turned out to be ~entirely market beta of the chosen
universe. Per-window numbers, not aggregate; alpha vs passive, not
raw Sharpe; same universe + window for both arms, not cherry-picked
baselines.

Different from any single prediction target the apps train against
(rank-IC, OLS R², forward-DD Pearson r), and intentionally so —
those targets exist to *serve* this metric, not to be it.

## Where the result-bearing sections went

Earlier drafts of this notes file carried full prose for each
forecast/horizon/universe/regime-gate experiment. Those have since
been condensed into one row each on the
[Leaderboard](leaderboard.md), with the operational verdict
([verdict-label vocabulary](leaderboard.md#verdict-labels)),
per-window stats, and artifact paths. Adjacent prose findings
covering the same arc:
[factor indicator-IC baseline (the +0.012 ceiling)](findings/factor-indicator-baseline.md),
[log-returns vs raw close](findings/log-returns-vs-raw-close.md),
[regime baselines](findings/regime-baselines.md). The relevant rows
are:

- `sign_demeaned target reduction probe` — `reversed-OOS`.
- `vol_innovation forecast target` — `confirmed-OOS for vol IC`.
- `Vol-target overlay` — `confirmed-null` (forecast adds nothing over
  trailing vol for sizing).
- `Pure CWT bundle vs IndicatorGridConfig` — `confirmed-null` on
  returns, `partial-OOS` on vol.
- `Feature augmentation: vol forecast as 75th feature` —
  `confirmed-null`.
- `Horizon pivot rebal=63d (quarterly)` — `reversed-OOS`.
- `Universe pivot 297 → 2073` — `confirmed-null`.
- `Regime gate via aggregate forecast vol` — `confirmed-null`.

The aggregate read across all eight: the +0.012 cross-sectional
return-IC ceiling is **data-side**, not supervision-, encoder-,
horizon-, or universe-side. Honest pivots from here are
prediction-problem changes (pair-spread, drawdown, IV-vs-realized) —
tracked in the [TODO](TODO/different-prediction-problem.md).

## Quote-availability is a deployability gate

A deployability check that bites *before* transaction cost, and that
the alpha-vs-EW backtest construction structurally hides: **can you
get a price — historical or live — for the exact instruments the
signal selects?**

The vol small-capacity re-frame
([`vol-borrow-illiquid-vrp-falsified`](findings/vol-borrow-illiquid-vrp-falsified.md))
is the worked example. The signal was *real* — a +0.14-of-premium
short-vol edge per cycle that survived artifact-cleaning. It still
failed, because the cohort carrying it (microcap single-name options)
is absent from every free option-quote source: 92.5% of the predicted
picks could not be priced at all, and the 7.5% that could had
*negative* edge. No spread number was ever needed.

The general principle, in order of when each gate bites:

1. **Quote-availability** — is the cohort priced by any source you
   can access (free or affordably paid), historically *and* live? If
   not, a real signal there is worth zero and the backtest cannot even
   measure honest cost.
2. **Transaction cost** — only meaningful *after* (1) passes; the
   breakeven-spread question.
3. **Capacity** — only meaningful after (1) and (2).

The trap is structural, not incidental: edge that survives because it
lives in an illiquid corner ("illiquidity is the moat", see
[research-strategy: capacity-constrained edges](#)) is *co-extensive*
with the corner no vendor bothers to quote — coverage economics track
liquidity economics for the same reason the premium exists. So the
"moat" inverts into an un-priceable wall. **Verify quote-availability
of the target cohort before building anything**; it is the cheapest
gate and it fails the most seductive ideas (real-signal-in-an-
illiquid-cohort) fastest. The alpha-vs-EW metric hides this because
both arms are notional — neither has to be quotable to difference.

## Learner layer matters more than learner complexity

Across the repo there are several "cleverer-model-on-standard-data is
confirmed-null 49×" lines (see
[`TODO/vol-borrow-liquid-universe`](TODO/vol-borrow-liquid-universe.md),
[`TODO/factor-cwt-return-coupled`](TODO/factor-cwt-return-coupled.md),
[`TODO/factor-reinforce-target-side`](TODO/factor-reinforce-target-side.md),
[`findings/factor-endogenous-horizon-mixture`](findings/factor-endogenous-horizon-mixture.md),
the 2026-05-17 factor-cwt row of the
[`leaderboard`](leaderboard.md)). Those claims are correct **within
their narrow scope**: learners trained at the *prediction layer*
(forecast next-period cross-sectional returns; train rank-IC, eval
Sharpe; softmax weights on a probability simplex) have indeed lost
49× across the leaderboard. That story remains operationally true.

**The narrow scope is the whole point.** Three closing arcs on
2026-05-28 sharpened where learners DO work and where they don't:

1. **At the prediction layer (forecast cross-arc / cross-sectional
   returns)** — still confirmed-null. See
   [`findings/meta-allocator-internal-features`](findings/meta-allocator-internal-features.md):
   ridge / kernel-ridge / 2-PC predictive regression on 14 strategy-
   internal features at quarterly cadence all underperformed B3
   inverse-vol AND the canonical (DCA + 2×vol_v3) by ΔSR margins
   that excluded zero negative. Same data, same arcs, wrong layer.
2. **At the sizing / allocation layer (direct portfolio Sharpe
   loss, unconstrained weights)** — **confirmed-OOS**. See
   [`findings/learned-ensemble-beats-deterministic`](findings/learned-ensemble-beats-deterministic.md):
   a 2-parameter mean-variance learner (closed-form + gradient
   ascent on Sharpe both converge) over the joint (DCA daily,
   vol_v3 daily-aligned) return stream beats the deterministic
   `(1, 2)` recipe on every OOS split with ΔSR_ann +3.0 to +4.9,
   every CI excluding 0, and tighter max-DD.
3. **At the end-to-end paradigm layer (direct Sharpe loss on raw
   inputs, deep encoder)** — paradigm directionally confirmed on
   fold-3 unseen 2024+ (Sharpe +1.21–1.25), pooled CI-includes-zero
   vs DCA. See [`findings/e2e-portfolio-v1`](findings/e2e-portfolio-v1.md)
   and [`findings/e2e-portfolio-v2`](findings/e2e-portfolio-v2.md).
   The vol_position output head in v2 learned exactly the right
   regime gate (mean +0.52 on fold-3, ~0 in no-IV folds) without
   any meta-layer alpha streams as inputs.

The unifying frame: **a learner's failure to match a deterministic
benchmark that extracts real alpha is a learner-objective failure,
not a no-signal verdict on the data.** The deterministic recipe
`DCA + 2×vol_v3` is *itself* a learned answer (it captures the σ/α
mismatch of two alpha streams via fixed mean-variance reasoning); a
learner that can't represent gross > 1.0 (softmax on simplex)
physically can't represent the recipe, regardless of how clever the
predictor.

**Operational implication for choosing the lever.** When a learner
arc lands `confirmed-null` vs a deterministic benchmark that extracts
real alpha:

- First check the **action space**: does it include the deterministic
  answer as a representable policy?
- Then check the **objective**: is it portfolio Sharpe directly, or
  a forecast loss that's only a proxy for portfolio Sharpe?
- Only after both are correct does "the model isn't extracting
  signal" become the load-bearing claim.

The "cleverer-model-on-standard-data is confirmed-null 49×"
shorthand survives — but its narrow scope is the prediction layer
with forecast loss. Sizing-layer learners with direct-Sharpe loss and
appropriate action spaces are a different family; they don't have a
49× null count, they have a `confirmed-OOS` result.
