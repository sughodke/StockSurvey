---
tags:
  - factor-narrow
  - confirmed-null
---

# Pairs classical v0 — `confirmed-null` per pre-registration, regime-conditional partial signal

!!! note "Update 2026-05-14 — EG-passing-rate gate falsified, but operational framing softened"

    The "EG-passing-rate per window is itself a regime indicator"
    hypothesis was tested directly in
    [`pairs-eg-gate-falsified`](pairs-eg-gate-falsified.md) and
    **falsified at all three pre-registered thresholds**: best-case
    gated alpha is +0.104 full-panel (+0.156 fired-only), 3-5×
    short of the audit's predicted +0.5 lift. The deeper finding:
    w0 (worst val Sharpe at −1.233) has train EG-pass = 3918, the
    **third-highest** of the 6 windows, sitting cleanly in the
    "working" band by EG-pass count yet failing badly OOS. Train-
    side regime indicators inherit the regime that produced the
    training data, not the regime that will receive the val
    deployment.

    The `confirmed-null` verdict label below is unchanged
    (technically correct by the +0.20 alpha pre-reg cut), but the
    operational claim "pair-spread mean reversion is not an alpha
    source on this universe at this horizon" is **overstated**: 4/6
    windows post positive val Sharpe; the mean is dragged below the
    threshold by one large negative window (w0, dot-com-trained).
    A more honest description is "alpha exists in 4/6 windows,
    mean below the +0.20 marginal floor". Pair trading stays parked,
    but the audit-flagged "missed deployment lever" (EG-passing-rate
    gate) is now confirmed to not be one.

Second test of the
[different-prediction-problem](../TODO/different-prediction-problem.md)
pivot. Engle-Granger cointegration screening on per-window train
slices + classical z-score-crossing trade rules on the screened
survivors. Numpy + statsmodels, no ML.

Verdict: [`confirmed-null`](../leaderboard.md#verdict-labels) per
the pre-registered cuts (mean agg val Sharpe **+0.099**, below
the +0.20 fail threshold). 4/6 windows positive — directionally
consistent — but a single catastrophic window (−1.23 in the
2005-2008 bull market) drags the mean well below the noise band.
The regime-conditional signal (works in 2008-2017
mean-reverting markets, fails in 2005-2007 + 2020-2023 trending
bull markets) is real but doesn't make the unconditional
strategy deployable.

Per `confirmed-null` next-move rule: pivot to
[`apps/vol`](../TODO/apps-vol.md). The regime conditioning is
parked as a v2 follow-up — not blocking the broader pivot.

## Setup

Universe: factor-narrow (297 stooq_us_long names with
`min_history_bars=6500`). Span: 2000-01-03 → 2025-12-11 (6526
daily bars, 297 × 6526 panel).

Walk-forward: 6 rolling windows, train=1260 bars (~5y) /
val=780 (~3y) / step=780 (no overlap). Same shape as
[`gate-drawdown-v0`](gate-drawdown-v0.md) for direct
comparability.

Per-window screening pipeline:

1. **Min-overlap** ≥ 1008 bars (80% of train window) of joint
   non-NaN history on the train slice. Drops survivorship bias
   from late-listing tickers.
2. **Correlation prefilter** `|corr(log_p_a, log_p_b)| ≥ 0.7`
   on the train slice. Cuts ~78% of the C(297,2) ≈ 44k
   candidate pairs to ~10-25k. The high-corr restriction is
   standard quant-lit preprocessing for cointegration screens.
3. **Engle-Granger** ADF p-value < 0.05 on the OLS residuals
   `log(P_A) − β · log(P_B) − α`. Per-window EG passing rates
   ranged from 2249 (window 4) to 4755 (window 3); the rate
   itself is a regime indicator (more cointegrated pairs in
   mean-reverting eras).
4. **Top-50** by ascending EG p-value. Same K across windows.

Trading: classical z-score state machine — flat → long-spread
when `z < −2σ`, → short-spread when `z > +2σ`, → flat when `|z|`
crosses back past `0.5σ`. Z-score uses train-set mean and
stdev (no peeking).

Costs: 10 bps × 2 legs per state transition (each leg pays
on its open / close / flip).

Aggregation: equal-weight `1/N` across the 50 top pairs;
per-bar PnL is the mean of per-pair PnLs.

Implementation hot path: `coint(log_p_a, log_p_b, maxlag=1)` —
explicit `maxlag=1` skips statsmodels' BIC lag-selection inner
loop, ~17× speedup (~500ms/pair → ~30ms/pair on 1260 bars).
Daily-bar pairs over 5-year windows don't need lag augmentation
to handle residual autocorrelation; the speedup is the
difference between a 30-second arm and a 2-hour arm.

Total wall: ~5-7 minutes on local 8-core for the full
6-window run, dominated by EG screening of ~80k total pairs
(across windows) at ~30ms each with `mp.Pool(8)`.

## Result (2026-05-10)

| win | val period | EG passing | agg Sharpe | mean pair Sh | pos pair frac | maxDD% |
|---|---|---:|---:|---:|---:|---:|
| 0 | 2005-01 → 2008-02 | 3918 | **−1.233** | −0.258 | 0.30 | −14.5 |
| 1 | 2008-02 → 2011-03 | 3522 | **+0.870** | +0.403 | 0.82 | −11.4 |
| 2 | 2011-03 → 2014-04 | 3118 | +0.593 | +0.164 | 0.54 | −1.8 |
| 3 | 2014-04 → 2017-05 | 4755 | +0.392 | +0.263 | 0.66 | −7.7 |
| 4 | 2017-06 → 2020-07 | 2249 | +0.080 | +0.098 | 0.54 | −7.2 |
| 5 | 2020-07 → 2023-08 | 2857 | −0.109 | +0.021 | 0.52 | −6.7 |
| **mean** | | | **+0.099** | **+0.115** | **0.56** | |

Pre-registered verdict cuts (per
[`TODO/apps-pairs.md`](../TODO/apps-pairs.md)):

| Cut | Threshold | Observed | Verdict |
|---|---|---|---|
| Pass | mean ≥ +0.50, ≥ 4/6 pos | mean +0.099 | fail |
| Marginal | mean +0.20-0.50, ≥ 3/6 pos | mean +0.099 < +0.20 | fail |
| **Fail** | **mean < +0.20 *or* ≤ 2/6 pos** | **+0.099 < +0.20** | **fail** |

Strict pre-registration says `confirmed-null`. Honor it.

## Why the headline is misleading without the per-window read

Looking only at the mean (+0.099) hides the regime structure.
The arithmetic mean is dominated by window 0's −1.23 — without
that single outlier the mean of the remaining 5 is **+0.365**
(passes the marginal threshold) and 4/5 are positive.

That's not a license to drop window 0 — it's a deployment
risk diagnosis. **A strategy that earns +0.5 Sharpe most of the
time and loses −1.2 Sharpe occasionally is not deployable**
unless you can detect the bad regime *in advance*. The
alternative readings:

- **In 2008-2017 (windows 1-3), pair trading delivered
  consistent positive Sharpe** of +0.39 to +0.87. The
  post-GFC era was the cointegration sweet spot — markets
  mean-reverted, sector rotations preserved the cointegration
  relationships in the train slice through the val slice.
- **In 2005-2008 (window 0), pair trading was catastrophic**
  (−1.23 agg Sharpe). The pairs were trained on 2000-2005
  (dot-com aftermath, mean-reverting). When deployed into the
  2005-2007 bull market, divergent trends widened the spreads
  faster than they reverted; the −2σ entry got +3σ wider, and
  the strategy held losses while waiting for reversion that
  never came in val.
- **In 2017-2023 (windows 4-5), pair trading flatlined**
  (+0.08 → −0.11). The 2017-2019 bull + 2020 COVID + 2021-2023
  AI / mega-cap dispersion eroded cointegration — fewer pairs
  passed EG (2249-2857 vs 4755 in window 3), and even the
  surviving pairs traded as drift not reversion in val.

## The regime classifier

The most natural follow-on (parked, see "v2 follow-ups" below):
the EG-passing-rate per window is itself a regime indicator.
Windows 0, 4, 5 all had < 4000 EG-passing pairs and produced
agg Sharpe ≤ +0.08. Windows 1, 2, 3 had ≥ 3500 with one outlier
(window 1's GFC has 3522 — driven by a *concentration* effect,
spreads stay narrow when correlations spike during crisis).

This is the same regime-conditioning lesson as
[`gate-drawdown-v0`](gate-drawdown-v0.md): the prediction
problem has signal in some regimes, none in others, and the
unconditional deployment averages them out below the noise
band. A v2 that gates the strategy by EG-passing-rate-of-train
would deploy only in cointegration-favorable regimes — but
this stacks regime-classification on top of the underlying
trade rule, doubling the model surface and risking overfit.

## Mechanism — why the unconditional fails

Three observations:

### 1. Cointegration is regime-specific.

The relational analog scorer
([`relational-universe-shift`](relational-universe-shift.md))
already taught us this lesson on a different problem class.
Per-window screening on train avoids the obvious failure mode
(re-using a pair list across regimes) but doesn't avoid the
deeper one: a pair that cointegrated in train can stop
cointegrating in val even with the same screening protocol.
Window 0 is the canonical example — dot-com-era cointegration
relationships didn't survive the 2005-2007 bull market.

### 2. The trade rule has no concept of "this regime is wrong."

The classical state machine enters a position whenever `|z| > 2`
and holds until reversion or exit. There's no stop-loss, no
"if z is still widening at t+30, give up." Window 0's
catastrophe is the textbook failure of this rule: spreads
widen in val because the underlying cointegration broke,
positions accumulate losses while waiting for a reversion
that doesn't come, and exits eventually fire at much wider
spreads than entry.

A `stop` parameter exists in `predictor.py` (default `inf`,
disabled). v2 could test `stop=4σ` or similar. But that's a
band-aid — the deeper issue is that the rule has no way to
detect cointegration breakdown.

### 3. Mean-reverting markets vs trending markets.

Pair trading is a long-vol-of-spread, short-trend-in-spread
strategy. Mean-reverting eras (2008-2017) reward it; trending
eras (2005-2007, 2020-2023) punish it. The 6-window walk-forward
spans both eras roughly equally — net outcome is
unconditionally null, regardless of how good the per-pair
selection is.

## Connection to the broader pivot

The arc:

1. [`gate-drawdown-v0`](gate-drawdown-v0.md) — drawdown
   prediction has signal (Pearson r +0.26 ~25-50× the
   cross-sectional return IC) but mean alpha within noise
   band. `partial-OOS`.
2. **This finding** — pair-spread mean reversion has signal in
   some regimes (mean +0.365 ex-window-0) but mean below
   threshold unconditionally. `confirmed-null` per
   pre-registration.
3. Next: [`apps/vol`](../TODO/apps-vol.md) — IV-vs-realized
   vol mispricing.

The pattern across the two new prediction problems is the
same: **non-zero raw signal per the cleanest predictive
metric (Pearson r for gate, mean-positive for pairs), but
deployment-as-shippable-strategy fails when single bad
windows / regimes overpower the average alpha**. Both findings
suggest we have models that genuinely predict their target,
but the underlying targets don't *consistently* compensate for
friction + risk control gaps.

If `apps/vol` produces a similar pattern (signal but not
shippable), the systemic conclusion is: **all three
orthogonal prediction problems have small, regime-conditional
signals — but no single problem has enough signal to ship
as an unconditional strategy**. The right v2 architecture
would *combine* the three weak signals (drawdown + pairs + vol)
into a composite portfolio that hedges their regime
specificities. But that's a multi-experiment design, only
worth committing to once `apps/vol` lands its standalone
verdict.

## v2 follow-ups (parked)

Filed in the same TODO neighborhood as
[gate v2 follow-ups](gate-drawdown-v0.md):

- **EG-passing-rate regime gate.** Run pair trading only when
  the train-window EG-passing-rate exceeds a threshold (would
  have skipped windows 0, 4, 5 in this run, lifting mean to
  ~+0.6). Risk: overfitting the threshold to this dataset.
- **Stop-loss on widening spreads.** Add `stop=4σ` to the
  trade rule. Reduces window 0's drawdown but costs some
  legitimate-reversion captures elsewhere.
- **Half-life-conditional sizing.** Pairs with shorter
  half-lives revert faster; size positions inversely to
  expected holding time. Reduces capital tied up in pairs
  that revert slowly.
- **Sector-restricted pairs.** Cross-sector cointegration is
  statistically noisier (more likely a regression-on-noise
  artifact). Adding sector metadata (separate source) and
  restricting candidates to within-sector would tighten
  screening at some loss of universe.
- **ML predictor.** Linear / MLP head over `(z_{t-k}, ...,
  z_t, half_life, vol_of_spread)` predicting forward 20-day
  spread move. v2 if regime gating makes the unconditional
  pass first.

These are not blockers for the prediction-problem pivot.
[`apps/vol`](../TODO/apps-vol.md) is the next test per the
"tackle all three" plan.

## Smoke validation

`apps/pairs/scripts/smoke_kopep.py` ran the pipeline on the
famous KO+PEP cointegration over 2010-2020 train. EG p-value =
0.086 (above the 0.05 threshold), hedge β = +0.726, intercept
= +0.338, n_obs = 2769. Pipeline produces sensible numerical
output even on the (in this window) non-cointegrated pair —
the infrastructure was validated independently of the
walk-forward outcome. KO+PEP cointegration was strongest in
the 1990s-2000s; the 2010-2020 era saw it loosen, consistent
with the broader regime-conditioning observation here.

## Master walk-forward log

[2026-05-10 pairs classical v0 row](../leaderboard.md) —
[`confirmed-null`](../leaderboard.md#verdict-labels) per
pre-registered cuts, regime-conditional signal noted.
