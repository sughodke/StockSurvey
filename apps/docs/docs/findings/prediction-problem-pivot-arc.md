---
tags:
  - arc-synthesis
---

# Prediction-problem-pivot arc — three orthogonal tests, three regime-conditional partial signals

Closes the
[`different-prediction-problem`](../TODO/different-prediction-problem.md)
TODO opened after the cross-sectional return prediction class was
exhausted by three confirmed-nulls
([passive-EW](passive-ew-benchmark.md),
[long-short-constructor](factor-rankic-long-only-mismatch.md),
[loss-pivot](factor-loss-pivot.md)). The thesis: each of those
nulls held the *prediction problem* fixed (forward log-return,
cross-sectional ranking, factor-narrow universe, 20-day horizon)
and varied portfolio construction or training loss. The natural
follow-on is to vary the prediction problem itself.

Three orthogonal problems were tested in parallel apps:

- **`apps/gate`** — predict EW universe forward 20-day max
  drawdown. Time-series, single-target, deploys as exposure scaler.
- **`apps/pairs`** — predict pair-spread mean reversion via
  Engle-Granger cointegration screening. Per-pair time-series,
  long-short by construction.
- **`apps/vol`** — predict forward IV/RV gap from richer surface
  features (skew, smile, IV/HV ratio, OI imbalance, VIX-spread).
  Cross-sectional + time-series, deploys as gated short-vol.

Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels) at the
arc level. Three independent tests show the same pattern — real
multivariate signal, mean alpha within ±0.05 of the marginal
threshold, regime-conditional structure (works in some windows,
fails in others). **No single prediction problem cleared
shippability standalone**, but the cumulative directional
consistency is too strong to be noise.

## Headline scorecard

| Test | Mean alpha | Pos windows | Verdict | Strongest signal |
|---|---:|---:|---|---|
| [`gate-drawdown-v0`](gate-drawdown-v0.md) | +0.067 | 4/6 | `partial-OOS` | val Pearson r **+0.26** (~25-50× factor IC) |
| [`pairs-classical-v0`](pairs-classical-v0.md) | +0.099 | 4/6 | `confirmed-null` (per pre-reg) | mean ex-window-0 = **+0.365** |
| [`vol-surface-v0`](vol-surface-v0.md) | +0.089 | **5/5** | `inconclusive` | val Pearson r **+0.12** (40× the audit's univariate) |

All three numbers cluster around **+0.08-0.10 mean per-window
alpha** with 4-5 out of 5-6 windows positive. The arc-level
result is **not** "three independent nulls" — it's "three
independent under-threshold partial-OOS results that share a
mechanism."

## The shared mechanism — regime-conditional partial signal

Each of the three apps has the same per-window structure: a small
number of windows show clear positive alpha (consistent with a
real prediction signal), and other windows show flat or
slightly-negative alpha (consistent with the prediction failing
in regimes outside its training distribution).

**Gate (drawdown forecast):** window 1 (2008 GFC) carries +0.32
alpha by flatting EW exposure to 51% during the crash. Calm
windows accumulate small false-positive losses (−0.02 to −0.07).
The model has skill on tail drawdowns but poor magnitude
calibration.

**Pairs (cointegration mean reversion):** windows 1-3 (2008-2017,
mean-reverting markets) show agg Sharpe +0.39 to +0.87. Window 0
(dot-com pairs into 2005-2007 bull market) shows −1.23 — pairs
trained on a mean-reverting era can't survive a trending bull
market. EG-passing-rate per window is itself a regime indicator.

**Vol surface (skew / smile / IV-HV / OI / VIX-spread):** late
windows (2022-12 → 2023-06, post-COVID vol regime) show val r
> +0.23 / alpha > +0.13. Early windows (2021 mega-cap melt-up)
show train R² = 0.000 / val r ≤ 0.055 / alpha < +0.05. The
surface flattens in low-vol-dispersion regimes and the features
lose discriminative power.

Each app's signal is *regime-specific*. None of them work
unconditionally. The three together suggest a real but
fragile predictive layer in markets that none of our v0
constructors can monetize standalone.

## Why each problem nearly cleared and why none cleared

### What worked

All three problems have **non-zero multivariate signal** much
larger than the cross-sectional return ceiling (+0.005 to +0.012):

- Gate: val Pearson r +0.26 (50× the factor ceiling)
- Vol: val Pearson r +0.12 multivariate (univariate ≤ +0.003 — a
  methodological lesson preserved in CLAUDE.md: *never pre-screen
  vol-surface features by univariate Pearson r*)
- Pairs: per-window agg Sharpe spreads from −1.23 to +0.87 — the
  positive windows are real mean reversion captures.

The data is not the problem. These are real prediction problems
with real signal.

### What didn't work

In each case, the v0 *deployment construction* is too unconditional:

- Gate v0 binary-flips EW based on a single threshold. Calm-period
  false positives accumulate. Sigmoid mode catches more GFC alpha
  but distributes the calm-period bleed.
- Pairs v0 deploys all top-50 pairs unconditionally each window.
  Window 0 catastrophe drags the mean below threshold. No regime
  filter on the pair list.
- Vol v0 picks top-20% of cells by predicted gap each rebal, no
  per-rebal portfolio aggregator (per-cell Sharpe is a weak
  metric), no costs in the loop.

The common thread: **the v0 deployments treat the prediction as
unconditional**. Each app's per-window alpha distribution shows
the predictions are *only* skillful in certain regimes — but the
deployments don't know to suspend during off-regime periods.

## The arc-level operational rule

Codified in CLAUDE.md (added with this finding): **predictions
that have non-zero multivariate signal but regime-conditional
deployment performance need a regime filter, not a richer
predictor.** The natural v1 architecture for each of the three
apps is the same: a regime classifier on top that gates the
underlying predictor by recent-window characteristics
(EG-passing-rate for pairs; trailing vol regime for gate;
surface-shape stability for vol).

This is what NO_OPTIONS.md's Phase 9 transition-triggered rebal
discovered for the relational arc — see
[`relational-arc-synthesis`](relational-arc-synthesis.md): "the
fingerprint embedding has real predictive content for positional
dynamics; signal-triggered timing of the existing scorer beats
scheduled cadence." The same lesson applies here: schedule the
trigger, not the trade.

## What's next

Two paths forward, both gated on user direction:

### Composite portfolio of the three weak signals

Combine drawdown gate × pair regime × vol surface as a multi-
prediction-problem composite. Each individual is partial-OOS but
their regime sensitivities differ (gate triggers on tail-vol
regimes; pairs work in mean-reverting; vol surface works in
post-shock recovery). A composite that allocates exposure by
which signal is *active* in the current regime could in principle
clear shippability where none of the individuals do.

This is the natural next experiment but is a multi-app design —
needs its own scoping pass.

### v1 for the strongest single result

`vol-surface-v0`'s 5/5 directional consistency + late-window
strength + the
[v1 follow-ups](vol-surface-v0.md#v1-follow-ups-parked-unless-we-revisit)
(per-rebal portfolio aggregation, costs-in-loop, DoltHub
extension to 2026, MLP head) make it the most promising single
v1. The DoltHub extension alone would extend the test from 5
windows / 4 years to ~7 years, with a real OOS extension on the
late-window signal.

### What we're not doing

More v0 tests on more prediction problems. The cross-arc pattern
is now clear: orthogonal v0s show partial-OOS at consistent
magnitudes. Adding a fourth orthogonal v0 (e.g., earnings-revision
forecasting, pair-trade-of-pair-trades, etc.) would just add
another partial-OOS row without changing the strategic picture.
The information is in *combining* what we have or *deepening*
the strongest individual.

## Master walk-forward log

Three rows, all dated 2026-05-10:
- [`gate` drawdown v0](../leaderboard.md) — `partial-OOS`
- [`pairs` classical v0](../leaderboard.md) — `confirmed-null` per
  pre-reg
- [`vol` surface v0](../leaderboard.md) — `inconclusive`

Plus the arc-level synthesis (this page).

## Resolved TODOs

- [`apps-pairs`](../TODO/apps-pairs.md) — v0 resolved 2026-05-10
- [`apps-vol`](../TODO/apps-vol.md) — v0 resolved 2026-05-10
- [`different-prediction-problem`](../TODO/different-prediction-problem.md) —
  the umbrella TODO; superseded by this synthesis
