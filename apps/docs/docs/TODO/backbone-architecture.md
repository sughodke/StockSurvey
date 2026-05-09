# Backbone architecture — push toward broader window/indicator coverage

Three architectural levers below SSL pretrain, ordered by cost.
Goal: relax the current backbone's narrow indicator-shaped bias (only
4 unconditioned linear heads = at most 4-rank usage of the 5632-d
latent; the other ~5600 dims get zero gradient and decay toward noise).
See `apps/notebook/src/ss_notebook/replay/README.md` "What the backbone
actually learns" for the diagnosis.

## Option A — FiLM-condition all four heads over their period grids

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
IC ≈ 0, val Sharpe +0.55..+0.63 from
[Notes — supervision is binding](../notes.md#what-we-already-know-about-supervision-being-the-binding-constraint)) when
trained on this richer backbone? If yes, broader window coverage was
enough. If no, the bottleneck is structural — go to option C or SSL.

## Option B — Add 20+ diverse indicator heads

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

## Option C — MLP heads, transfer head hidden layer to scoring

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

## Decision order

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
