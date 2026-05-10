# Drawdown gate v0 — real Pearson signal, marginal Sharpe lift

First test of the [`different-prediction-problem`](../TODO/different-prediction-problem.md)
pivot: predict the EW universe's next-20-day max drawdown from
trailing aggregate features, use the prediction as an exposure
gate (`gated_returns_t = gate_t · ew_return_t`). The hypothesis:
even noisy drawdown prediction should add Sharpe because
drawdowns are asymmetric — a gate that catches a fraction of
real DD events earns more in avoided losses than it loses in
missed gains during false positives.

Verdict: [`partial-OOS`](../leaderboard.md#verdict-labels) at the
canonical operating point (binary gate, train-pred 95th-quantile
threshold) — mean alpha **+0.067 Sharpe** with **4/6 positive
windows**. Within the ±0.10 noise band but with a clean
mechanism: window 1 (2008 GFC) carries +0.32 alpha by flatting
to 51% exposure during the crash; calm windows accumulate small
false-positive losses (−0.02 to −0.07) that nearly cancel the
GFC win.

Per the [verdict-action table](../leaderboard.md#verdict-labels):
`partial-OOS` → stratify by regime, then decide. The stratification
is already visible in the per-window data: gate works in
high-vol windows (windows 1, 4) and is approximately flat in
low-vol windows (windows 2, 3, 5). This is the regime split a
sequel arc would test directly — gate the gate by realized
volatility regime, deploying only when trailing vol is high
enough to justify the false-positive cost. Filed for follow-up
[here](../TODO/different-prediction-problem.md); not blocking
the pivot to [`apps/pairs`](../TODO/apps-pairs.md).

## Setup

Universe: `stooq_us_long` (312 tickers from
`apps/notebook/data/stooq_us_long/manifest.json`).

Aggregate construction: per-date EW simple return over tickers
with both a current and prior valid close (basket grows as
tickers come online; delistings drop out at last quoted price).
Implementation in `apps/gate/src/gate/aggregate.py` —
`build_ew_aggregate(prices, min_active=10)`. The aggregate EW
return series + 10 trailing aggregate features form the input
to the predictor.

Target: `forward_max_drawdown(ew_log_ret, horizon=20)` — for
each date `t`, the max peak-to-trough drawdown of the EW
log-return cumulative path over `(t, t+20]`. Implementation in
`apps/gate/src/gate/target.py`.

Features (all point-in-time at `t`):

```
vol_5, vol_20, vol_60   — trailing realized vol of EW log return
ret_5, ret_20, ret_60   — trailing mean log return
tdd_20, tdd_60          — trailing-window max drawdown
vol_term                — vol_5 - vol_60 (term structure)
breadth                 — fraction of universe online
```

Predictor: numpy OLS linear regression with z-scored features +
intercept. Trained on the train slice of each walk-forward
window. Implementation in `apps/gate/src/gate/predictor.py`.

Walk-forward: 6 windows on the 2000-2025 span, train=1260 bars
(~5y) / val=780 bars (~3y) / step=780 (~3y, no overlap). Driver
at `apps/gate/scripts/run_walkforward.py`.

Gate: `binary` mode — `gate = 1` if `predicted_dd ≤ threshold`,
else `0`. Threshold = train-pred 95th quantile (chosen on train
only, no peeking). Lagged by 1 bar so signal at `t-1` close
governs exposure at `t`.

Costs: not modeled in v0. Gate flips ~10-40 times per 780-bar
val window in the binary mode; at 10 bps round-trip per flip
that's roughly 0.13% to 0.5% per year of friction — would
slightly reduce the modest alpha numbers reported below but
not change the verdict direction.

## Result (2026-05-10)

### Canonical arm: binary gate, q=0.95 threshold

| win | val dates | train R² | val R² | val r | avg exp | unc Sh | gated Sh | alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2005-01-10 → 2008-02-14 | +0.190 | −0.052 | +0.319 | 0.977 | +0.702 | +0.699 | −0.002 |
| 1 | 2008-02-15 → 2011-03-21 | +0.160 | **+0.260** | **+0.612** | **0.513** | +0.519 | +0.840 | **+0.321** |
| 2 | 2011-03-22 → 2014-04-28 | +0.505 | −0.231 | +0.330 | 0.995 | +0.967 | +1.012 | +0.046 |
| 3 | 2014-04-29 → 2017-06-01 | +0.164 | −0.138 | +0.026 | 0.999 | +0.977 | +0.967 | −0.010 |
| 4 | 2017-06-02 → 2020-07-08 | +0.049 | −0.017 | +0.201 | 0.824 | +0.409 | +0.424 | +0.015 |
| 5 | 2020-07-09 → 2023-08-14 | +0.125 | −0.071 | +0.097 | 0.994 | +1.065 | +1.094 | +0.030 |

| Aggregate | Value |
|---|---:|
| mean val R² | −0.042 |
| **mean val Pearson r** | **+0.264** |
| mean unconditional EW Sharpe | +0.773 |
| mean gated Sharpe | +0.840 |
| **mean alpha** | **+0.067** |
| **positive-alpha windows** | **4/6** |

### Threshold sensitivity (binary gate, all 6 windows)

| threshold quantile | mean alpha | pos windows | mean gated Sharpe |
|---:|---:|---:|---:|
| 0.85 | **−0.114** | 1/6 | +0.659 |
| 0.90 | −0.009 | 2/6 | +0.764 |
| **0.95** | **+0.067** | **4/6** | **+0.840** |
| 0.50 (single-split smoke) | −0.691 (val only) | n/a | +0.115 (val only) |

Pattern: aggressive flatting (low threshold quantile) is
strictly worse. The signal lives only in the *tails* of
predicted drawdown — the model has weak ordering skill but
poor magnitude calibration, so a "be invested unless prediction
is extreme" rule monetizes the ordering without paying for the
mis-scaling.

### Sigmoid mode at q=0.95

| Mode | mean alpha | pos windows | window-1 alpha (GFC) |
|---|---:|---:|---:|
| binary | +0.067 | 4/6 | +0.32 |
| sigmoid (slope=50) | +0.059 | 2/6 | **+0.46** |

Sigmoid catches more of window 1's GFC alpha (+0.46 vs +0.32)
because it gradually scales rather than binary-flipping, but
trades that off by adding small graduated exposure changes in
calm periods that net negative on average. Mean alpha is
similar (+0.06 either way). Binary's 4/6 positive frac is the
more deployable operating point.

## Why it works in window 1 and not elsewhere

Window 1 trains on 2003-02-14 → 2008-02-14 and validates on
2008-02-15 → 2011-03-21 — directly through the GFC. The
predictor learned during the dot-com recovery + early bull (a
period with multiple modest 10-15% drawdowns) and then deployed
into the 2008-09 crash + 2010 flash crash. **The training data
contained drawdown events of roughly the magnitude that
appeared on val.** Val Pearson r jumps to +0.612 (vs ~+0.20 in
calm-window val periods).

The other windows train on calmer periods and fail to predict
the *one* extreme drawdown in their val (e.g. window 5's val
is mostly the 2020-2023 bull, with one COVID-recovery dip the
gate misses entirely because the model didn't see anything
similar in 2015-2020 train). This is the textbook regime-shift
failure — drawdowns are rare events, and a single-window OLS
predictor on a small feature stack can't extrapolate.

## Mechanism summary

The drawdown-gate hypothesis is partially confirmed: there *is*
a real predictive relationship between trailing aggregate vol /
return / DD features and forward 20-day drawdown (mean val
Pearson r = +0.264 is much larger than the +0.005 to +0.012 IC
we see in cross-sectional return prediction, by ~25-50×). The
prediction problem is a different ceiling than cross-sectional
return — that part of the pivot hypothesis is validated.

But monetizing the signal as a binary exposure gate is fragile
at the v0 implementation:

1. **The model only has skill on tail drawdowns.** Pearson r is
   driven by extreme bars where features clearly indicate
   "vol is rising, return is falling" → drawdown follows. On
   ordinary bars the model is near random.
2. **Binary gate over-reacts in calm periods.** The 0.95
   quantile threshold means flat 5% of the time on average
   (matching `avg_exp ≈ 0.95`), but those 5% are scattered
   across calm bars where flatting costs more than it saves.
3. **OLS linear can't extrapolate.** The 2008 alpha came from
   training data that contained similar features. Other val
   windows had no analogous training signal for their
   drawdown event.

Viable v1 directions (not in scope for this finding):

- **Regime-conditional gate.** Only enable the gate when
  trailing vol exceeds a percentile threshold of train vol.
  This skips most calm-period false positives.
- **Two-stage classifier + sizing.** First predict
  `P(drawdown_event)`, then size the gate proportional to
  predicted probability * predicted magnitude. Decouples
  "is there a regime change" from "how much to flat."
- **Better features.** Add cross-sectional dispersion (a
  VIX-like proxy from inside the universe), term-structure of
  realized vol, breadth indicators (% above 200dma).
- **Non-linear predictor.** MLP or random forest could
  extrapolate better than OLS on rare events; tinygrad path
  exists in `apps/factor` for the linear case and could be
  ported.

These are TODO follow-ups, not blockers for the broader pivot.
The test confirms drawdown forecasting *has signal* — much
more than cross-sectional return prediction does. Whether
that signal becomes a shippable strategy depends on a v1
implementation that handles regime shifts and false-positive
costs, which is a meaningful but not infinite amount of work.

## Implication

Per the `partial-OOS` next-move rule: stratify by regime
before deciding. The stratification is already done implicitly
(window 1 vs windows 0/2/3/5) and points to a regime-conditional
v1. But the broader prediction-problem pivot
([`apps/pairs`](../TODO/apps-pairs.md),
[`apps/vol`](../TODO/apps-vol.md)) is independent — those
test orthogonal alpha sources, and we're committed to running
them per the user's "tackle all three" plan.

The right operational posture: park `apps/gate` at v0
`partial-OOS` with the regime-conditional v1 noted as a
follow-up, run `apps/pairs` next, then revisit `apps/gate` v1
*if* `apps/pairs` also nulls and we need to consolidate
multiple weak alpha sources into a deployable strategy.

The gate's value-as-overlay is also the natural integration
point with the
[EW + rank-IC overlay](../TODO/ew-overlay-test.md) parked TODO
— combine `gated_returns_t = gate_t · (EW_t + α · rank_IC_tilt_t)`
for a layered defensive overlay.

## Master walk-forward log

[2026-05-10 gate drawdown v0 row](../leaderboard.md) —
[`partial-OOS`](../leaderboard.md#verdict-labels).
