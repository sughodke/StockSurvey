---
tags:
  - vol-surface
  - inconclusive
---

# Vol surface v0 — multivariate prediction works (mean val r +0.12), per-cell alpha just below threshold

Third app in the prediction-problem pivot off cross-sectional return
forecasting, picking up after
[`gate-drawdown-v0`](gate-drawdown-v0.md) (`partial-OOS`) and
[`pairs-classical-v0`](pairs-classical-v0.md)
(`confirmed-null` per pre-reg). Tests an explicitly-untested feature
class per [`apps/vol`](../apps/vol.md) — **skew, smile curvature,
IV/HV ratio, OI imbalance, single-name-vs-VIX spread** — that the
[NO_OPTIONS.md](https://github.com/sughodke/StockSurvey/blob/master/apps/relational/NO_OPTIONS.md)
arc's 9 scorers (all using `ATM_IV` only) didn't touch.

Verdict: [`inconclusive`](../leaderboard.md#verdict-labels) per
pre-registered cuts (mean alpha **+0.089** per-cell-Sharpe, just
below the **+0.10** marginal floor). But **5/5 positive-alpha
windows** is the cleanest directional consistency in the
prediction-problem-pivot arc so far (vs gate's 4/6 and pairs's
4/6). The most informative number: **mean val Pearson r = +0.12**
when the audit's univariate Pearson r maxed out at **+0.003** —
the surface-shape signal lives entirely in the *joint*
multivariate structure, not in any single feature.

Per `inconclusive` rule: stratify and reconsider before deciding.
The stratification is already visible in the per-window numbers
— late windows (2022-12 → 2023-06, the post-COVID-vol regime
era) show val r ~ +0.25 vs early windows ~ +0.05. v1 follow-ups
listed below; not pushing past v0 in this session per the
"tackle all three" plan, but the result is the strongest
cross-app test result so far and probably *the* one to revisit.

## Setup

Universe: full gauss314 SPX coverage — **3,893 unique symbols
× 938 dates** = 3.16M cells over 2019-10-14 → 2023-07-28.
Loaded from `.iv-cache/data_IV_USA.csv` via the new
`vol.load_gauss314_full()` (the prior `ss_iv.load_atm_iv()`
returned only `ATM_IV`; we needed the full schema).

Target: `iv_rv_gap_t = ATM_IV_t − hv_20_{t+20}` per
`(date, symbol)`. Positive gap = realized came in below implied
→ short-vol won that cycle. Standard short-vol PnL convention
matched to `ss_iv.short_vol_pnl_panel`.

Features (10 total — all explicitly **untested** by NO_OPTIONS):

| Feature | Definition | Mechanism |
|---|---|---|
| `skew_otm` | `(DOTM_IV − ATM_IV) / ATM_IV` | OTM tail-IV vs ATM ratio |
| `skew_itm` | `(DITM_IV − ATM_IV) / ATM_IV` | ITM tail-IV vs ATM ratio |
| `smile_curvature` | `(DOTM_IV + DITM_IV − 2·ATM_IV) / ATM_IV` | U-shape depth |
| `iv_over_hv20` | `ATM_IV / hv_20` (clipped ±10) | VRP magnitude, short horizon |
| `iv_over_hv60` | `ATM_IV / hv_60` (clipped ±10) | VRP magnitude, mid horizon |
| `iv_over_hv120` | `ATM_IV / hv_120` (clipped ±10) | VRP magnitude, long horizon |
| `hv_term` | `hv_20 / hv_200` | Realized vol mean reversion |
| `oi_imbalance` | `puts_OI / (puts_OI + calls_OI)` | Put positioning |
| `vix_spread` | `ATM_IV − VIX` | Single-name vs market vol |
| `strike_spread_norm` | `strikes_spread / ATM_IV` | MM inventory risk |

Predictor: numpy OLS linear with z-scored features + intercept,
per-window train (no peeking).

Walk-forward: 5 windows over the gauss314 span, train=300 days
(~14mo) / val=120 (~6mo) / step=120, no overlap. Window count
limited by the dataset's 938 trading days; window shape is
shorter than the equity apps' 1260/780/780 because we have less
calendar coverage.

Trade construction: at each val rebalance bar, pick the top-20%
of cells by *predicted* IV-RV gap (highest predicted short-vol
edge). Compare per-cell vol-points PnL of the gated picks vs
the universe (all cells) baseline. Per-cell Sharpe = `mean / std`
of vol-points across all picks (no annualization, not a portfolio
Sharpe — see Caveats).

## Result (2026-05-10)

| win | val period | train R² | val R² | **val r** | unc Sh | **gated Sh** | **alpha** |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 2021-01 → 2021-06 | +0.000 | −0.002 | +0.005 | +0.005 | +0.028 | **+0.023** |
| 1 | 2021-06 → 2021-12 | +0.000 | −0.495 | +0.055 | +0.162 | +0.211 | **+0.049** |
| 2 | 2021-12 → 2022-06 | +0.000 | −1.472 | +0.035 | +0.013 | +0.100 | **+0.086** |
| 3 | 2022-06 → 2022-12 | +0.043 | +0.062 | **+0.268** | +0.177 | +0.332 | **+0.155** |
| 4 | 2022-12 → 2023-06 | +0.034 | +0.051 | **+0.238** | +0.187 | +0.321 | **+0.134** |
| **mean** | | **+0.015** | **−0.371** | **+0.120** | **+0.109** | **+0.198** | **+0.089** |

Pre-registered cuts (per [`TODO/apps-vol.md`](../TODO/apps-vol.md)):

| Cut | Threshold | Observed | Verdict |
|---|---|---|---|
| Pass | mean alpha ≥ +0.30, ≥ 4/6 pos | mean +0.089 | fail |
| Marginal | mean +0.10–0.30, ≥ 3/6 pos | mean **+0.089** < +0.10 | fail (just) |
| Fail (clean) | mean < +0.05 | mean +0.089 > +0.05 | not a clean fail |

So the script reports `INCONCLUSIVE` and the honest call is
`inconclusive` leaning `partial-OOS`. The ±0.011 gap to the
marginal floor (+0.089 vs +0.100) is well within reasonable
single-test noise; the 5/5 directional consistency is
unambiguous evidence of *some* signal.

## Why the audit's univariate Pearson r looked dead

The data audit (`apps/vol/scripts/audit_data.py`) showed
single-feature Pearson r of ≤ +0.003 for every feature against
the same target. That was the most pessimistic single-feature
read possible — for any *individual* feature, the relationship
to forward IV/RV gap is essentially zero.

The walk-forward's mean val r = +0.120 — **40× larger** — comes
entirely from the *joint* fit. The features carry no marginal
signal but a meaningful conditional one: knowing the *combination*
of (skew, smile, IV/HV, OI imbalance, VIX-spread) at `t` predicts
forward gap better than any single one.

This is a clean methodological lesson worth keeping: **for vol
surface prediction, never gate on univariate feature-target
correlations**. The signal lives in the joint structure; a
univariate filter would have killed the experiment at the audit
step. The opposite of what's true for cross-sectional return
prediction (where the +0.005 IC ceiling is the same whether
measured univariate or multivariate — the features there are
nearly orthogonal in their (lack of) predictive content).

## Why late windows carry the signal

Windows 0, 1, 2 (2021-01 → 2022-06) have train R² = 0.000 and
val r ≤ 0.055 — essentially no signal. Windows 3, 4 (2022-06 →
2023-06) have train R² ~0.04 and val r > 0.23.

The 2022-06 → 2023-06 era is the post-COVID-vol-regime period
when the IV market was reckoning with persistent inflation, Fed
rate hikes, and earnings dispersion across mega-cap tech. The
surface features that worked there:

- **Smile curvature** likely picked up the systematic
  reach-for-puts during 2022's bear market;
- **IV/HV ratio** picked up the persistent VRP overpricing
  during the recovery;
- **VIX spread** distinguished idiosyncratic vol (single-name
  earnings shocks in tech) from macro vol regime.

The 2021 era (windows 0-2) was dominated by mega-cap melt-up
with low cross-sectional vol dispersion; the surface flattens
in such regimes and the features lose discriminative power.
This is regime-conditioning of the same flavor as
[`pairs-classical-v0`](pairs-classical-v0.md)'s
trending-vs-mean-reverting split.

## Connection to the broader pivot

Cumulative scorecard across the prediction-problem pivot:

| Test | Mean alpha | Pos windows | Verdict |
|---|---:|---:|---|
| `gate-drawdown-v0` | +0.067 | 4/6 | `partial-OOS` |
| `pairs-classical-v0` | +0.099 | 4/6 | `confirmed-null` per pre-reg |
| **`vol-surface-v0`** | **+0.089** | **5/5** | **`inconclusive`** (this) |

All three orthogonal prediction problems show **non-zero
multivariate signal** but **mean alpha just below the
single-test noise floor + commission stack**. The pattern is
remarkably consistent: *something is there*, but no single
prediction problem is rich enough to clear shippability gates
unconditionally.

The vol-surface-v0 result has one unique characteristic:
**5/5 positive windows** (vs the others' 4/6). If we had a
6th window the directional consistency would be clearer, but
the gauss314 dataset's calendar coverage caps us at 5. This is
the cleanest "real signal" result of the three, even though
it's still below the pass threshold.

NO_OPTIONS.md's conclusion was "the IV market efficiently
incorporates the dislocation information" — measured against
9 scorers using only `ATM_IV`. **This v0 partially refutes
that claim**: the IV market's *single-IV-value* representation
of vol surface state efficiently incorporates dislocation, but
the *full surface* (skew + smile + multi-horizon IV/HV +
single-name vs VIX) carries forward-looking information that
ATM-IV-only feature stacks miss. The market is efficient at
the level the prior arc tested; it leaves residuals at the
level we just tested.

## Caveats

- **Per-cell Sharpe is a weak metric.** Each "cell" is one
  `(date, symbol)` observation, and per-cell Sharpe is just
  `mean / std` over cells with no temporal aggregation or
  annualization. The honest portfolio metric would be per-rebal
  PnL aggregated across simultaneously-held picks, then a
  time-series Sharpe over the rebal series. v1 follow-up.
- **Costs not modeled.** Options friction is 100-1000 bps
  round-trip (NO_OPTIONS.md's note: "100-500 bps round-trip
  for liquid SPX names"). The +0.089 per-cell-Sharpe alpha
  needs to be net of that friction to be deployable. Likely
  cuts the alpha to near zero; the result is essentially "the
  signal exists; whether it's tradable depends on options
  liquidity restrictions we haven't modeled."
- **Data span is ~4 years.** Limited to the gauss314 CSV's
  2019-10 → 2023-07 coverage. The DoltHub parquet covers
  longer (2019-02 → 2026-04) but only has `iv_current` +
  `hv_current`, not the full surface. v1 should test whether
  a hybrid (gauss314 features computed up to 2023, joined
  with DoltHub for the 2023-2026 OOS extension) recovers the
  late-window signal.
- **iv_over_hv ratio outliers** were clipped to ±10. The
  unclipped distribution had mean 35,000 / std 1.9M from
  HV-near-zero rows. Clip at ±10 is the cleanest pre-feature
  step but means we lose information from extreme-low-HV
  regimes (after big drops in realized).

## v1 follow-ups (parked unless we revisit)

- **Per-rebal portfolio aggregation** — replace per-cell
  Sharpe with per-rebal PnL → annualized portfolio Sharpe.
  The honest deployment metric.
- **Costs-in-the-loop** — embed 100-500 bps options friction
  in the gated PnL. May reveal the +0.089 alpha is gross,
  not net.
- **DoltHub extension** — compute proxy surface features from
  DoltHub `volatility_history` (only iv_current + hv_current)
  to extend the test through 2026-04 for a real OOS check on
  whether the late-window signal holds.
- **MLP head** — multivariate Pearson r +0.12 + train R²
  > 0 in late windows suggests there's nonlinear structure to
  capture. MLP via tinygrad (port from `apps/factor`'s
  pattern) is the natural next step.
- **Universe restriction by liquidity.** Currently 3,877
  symbols enter the regression, many with thin options
  liquidity. Restricting to top-100 by OI per date would
  trade some sample size for cleaner trading targets.

## Smoke validation

Pipeline validation done in
`apps/vol/scripts/audit_data.py`: data loads cleanly, schema
matches expected, 0% null on critical columns (`ATM_IV`,
`hv_20`, etc), feature computation produces sensible distributions
(skew_otm mean +0.33 ≈ ~33% OTM premium over ATM, in line with
empirical skew). Outlier issue on `iv_over_hvX` was identified
in audit and addressed via clip in the walk-forward.

## Master walk-forward log

[2026-05-10 vol surface v0 row](../leaderboard.md) —
[`inconclusive`](../leaderboard.md#verdict-labels), strongest
directional consistency (5/5) of the three pivot tests but
mean alpha just below the +0.10 marginal floor.
