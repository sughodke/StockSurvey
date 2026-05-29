# Meta-allocator with strategy-internal features — pre-registered

**Status: `DONE` — eval ran 2026-05-28; verdict `confirmed-null` per
the locked pre-reg bar. See [`findings/meta-allocator-internal-features`](../findings/meta-allocator-internal-features.md)
for the writeup and the 2026-05-28 row of [`leaderboard`](../leaderboard.md).
Closes the meta-allocator arc — cadence × input-class × learner levers
all exhausted. Deployment recipe remains canonical DCA + 2×vol_v3
sleeve.**

---

Original pre-registration (kept verbatim below for the record):
Direct follow-up to
[`meta-allocator-regime-forecasting`](../findings/meta-allocator-regime-forecasting.md)
(2026-05 `confirmed-OOS` for B3 inverse-arc-vol; 5/5 macro-based
forecasters `reversed-OOS`) and
[`meta-allocator-no-vol-v3`](../findings/meta-allocator-no-vol-v3.md)
(B3 falsified when vol_v3 dropped from the panel). The 6-feature
macro stack (`fed_funds, slope_10y_3m, credit_baa, m2_yoy,
real_yield_10y, vix`) did not predict cross-arc relative returns at
monthly cadence. This pre-reg locks the orthogonal lever: replace
external macro features with **strategy-internal valuation /
crowding / correlation features at the cadence the literature
supports.**

---

## Why this pre-reg exists

The prior arc's `confirmed-null` verdict on every learned forecaster
isolated the failure to **feature category**, not learner capacity.
Five learners (Markov, turbulence, meta-labeling, CUSUM, combo) over
the same 6 macro features all lost. Per the
[`confirmed-null` decision branch](../leaderboard.md#verdict-labels)
("stop testing variations of the same lever — find an orthogonal
one"), the orthogonal lever is the **input space**, not the model
class or loss function.

The Haddad-Kozak-Santosh 2020 result (*Review of Financial Studies*,
NBER 26708) is the load-bearing literature pointer: **dominant PCs
of long-short factor returns are predictable from the BM-spread of
the factor's own legs, not from macro state.** Macrosynergy and
Mulliner-Harvey-Xia-Fang 2025 (Swedroe summary) converge on the
same operational point: macro features matter at business-cycle
frequency (quarters), not monthly. Alpha-decay literature
(Pedersen, AQR; arXiv 2512.11913) adds a third channel: capacity-
utilization proxies (rank-IC trend, spread compression) measurably
predict per-strategy alpha decay.

This pre-reg combines all three into one feature set and locks the
bar.

---

## Mechanism — the steel-man

Why this could clear the bar:

1. **Internal-spread features are mechanically linked to the
   strategy's own alpha.** HKS showed BM-spread of factor legs
   predicts factor returns directly because high spread = factor
   is "cheap" by its own valuation. For our arcs:
   - **DCA / dca_winner_4etf**: P/E of the held basket vs trailing
     10y history is a direct buy/sell-side spread.
   - **vol_v3**: percentile of trailing IV-vs-RV gap on the held
     underlyings (already computed internally; never exposed as
     meta-allocator feature).
   - **relational / pairs / gate**: each has a natural internal
     valuation analogue (correlation-cluster dispersion, pair
     z-score widening, EW dispersion).

2. **IC-trend per arc is a direct alpha-decay measurement.**
   Trailing-252d rank-IC on each arc's underlying signal monotonically
   tracks whether the strategy is decaying. A meta-allocator that
   under-weights decaying arcs by reading their own IC trend has a
   measurable, mechanism-grounded reason to deviate from inverse-vol.

3. **Cross-arc correlation effective rank captures the regime where
   inverse-vol fails.** B3 over-concentrated in vol_v3 precisely
   because vol_v3 had anomalously low vol. When cross-arc
   correlations spike (crisis lockstep), low-vol = low-info, not
   low-risk. A measured `eff_rank` of the rolling correlation matrix
   tells the allocator when to compress toward 1/N.

4. **Quarterly cadence respects the data's signal-to-noise.**
   137 monthly decisions is the binding sample-size problem from
   the prior arc. Going to quarterly cadence (~46 decisions across
   2015-2025) **reduces** training data but **increases per-point
   SNR** because the macro features actually update at that
   frequency. Net effect on regression CI is ambiguous but the
   bias term (using slow features at fast cadence) is removed.

If internal-spread features + IC trend + cross-arc effective rank
beat B3 at quarterly cadence on the locked bar, the meta-allocator
problem is solvable; we just had the wrong features. If they
don't, the meta-allocator arc closes for good (every reasonable
input class tried).

---

## Feature set (locked)

**14 features**, all point-in-time at trailing close, lagged 1 day
to avoid look-ahead. For each of the 6 arcs (where defined; missing
arcs masked at the rebal date):

| feature | computation | n |
|---|---|---:|
| arc realized vol (252d) | `std(returns).annualize()` | 6 |
| arc rank-IC trend (252d) | trailing rank-IC of arc's primary signal, slope of linear fit | 6 |
| **strategy-internal valuation spread** | per-arc (see table below) | 1 per arc |
| cross-arc correlation eff-rank (60d) | effective rank of trailing 60d return-correlation matrix | 1 (cross-arc) |
| portfolio-of-arcs realized vol (60d) | for normalisation only | 1 (cross-arc) |

Per-arc valuation spreads:

| arc | spread definition |
|---|---|
| `dca` | weighted P/E of 13-ETF basket vs its trailing 10y median |
| `dca_winner_4etf` | weighted P/E of 4-ETF basket vs trailing 10y median |
| `vol_v3` | percentile of trailing 60d mean IV-vs-RV gap on held underlyings (already in `ss_iv`) |
| `relational` | dispersion of CWT-fingerprint distances at last rebal (already in `relational.fingerprints`) |
| `pairs` | mean abs(z-score) of all candidate cointegrated pairs at last rebal |
| `gate` | trailing 60d aggregate drawdown vs its trailing 10y median |

Total feature count: **14**. With 6 arc outputs, max effective DOF
is 6 × 14 = 84 — over a quarterly sample of ~46 points this is
hostile, so **dimensionality reduction is part of the locked
design** (see below).

---

## Search space (locked)

Single locked grid:

| dimension | values | rationale |
|---|---|---|
| cadence | quarterly (63 trading days) | matches macro/valuation feature update rate |
| model family | ridge regression (D1), random forest depth-3 (D2), 2-PC predictive regression (D3) | D3 follows HKS exactly; D1 is the linear baseline; D2 tests nonlinearity |
| target | each arc's next-quarter return | direct per-arc forecast, NOT cross-arc rank |
| normalization | z-score within trailing 5y window (point-in-time) | prevents look-ahead via training-set normalisation |
| weight transform | softmax of expected return / B3 vol weights, normalized | softmax with the σ-scaling makes the predictor compete on a per-vol-adjusted basis with B3 |

Grid: 3 model families × 1 cadence × 1 target = **3 trials**.
n_trials for deflation = 3 (smaller than prior arc; conservative).

Plus the two locked baselines (carried from the prior arc):
- **B2 1/N** equal-weight
- **B3 inverse-arc-vol** (the prior winner)

---

## Datasets + windowing (locked)

| field | value |
|---|---|
| panel | `meta-alloc-arcs-6` — exactly the 6 arcs of the prior finding |
| span | 2015-01-01 → 2025-10-16 |
| folds | 3 contiguous OOS folds — 2015-2018, 2019-2022, 2023-2025-Q3 (identical to the prior arc) |
| cadence | 63 trading days (~quarterly), 10 bps switching cost on `\|Δw\|/2` |
| availability mask | vol_v3 from 2024-04-12 only; arcs unavailable at a rebal are masked out and weights renormalize |
| metric | pooled-OOS daily-return Sharpe, Ledoit-Wolf studentized ΔSR CI vs B3, DSR-t |
| deflation | n_trials = 3 (the 3 model families) |

---

## Pre-locked verdict bar

The 3-trial grid produces an in-search winner `model*` on the **first
2 folds**. The verdict is locked on the **fold-3 OOS** behaviour at
`model*`:

| condition | verdict |
|---|---|
| OOS ΔSR_ann ≥ +0.30 vs **B3** AND Ledoit-Wolf 95% CI excludes 0 AND DSR-t > +3.0 | **confirmed-OOS** — promote internal-feature meta-allocator over B3 |
| OOS ΔSR_ann ≥ +0.15 vs B3 AND CI excludes 0 on the positive side AND DSR-t > +1.5 | **partial-OOS** — record the feature lift; do not deprecate B3 |
| OOS ΔSR_ann ≥ +0.10 vs B2 1/N but does not beat B3 | **partial-OOS-vs-1/N-only** — the features carry signal but not enough to beat the vol-only baseline |
| OOS ΔSR_ann < +0.05 vs B3 OR `model*` collapses to B3-equivalent weights in-search | **confirmed-null** — meta-allocator arc closes; every reasonable feature class tested; deploy B3 forever |

**Sample-size honesty.** ~46 quarterly points across 3 folds is
brutal. If the fold-3 stationary-bootstrap 95% CI on ΔSR vs B3 is
wider than **±0.40**, the verdict is **automatically downgraded one
tier**. The prior arc had n=137 monthly points and still couldn't
support confident ΔSR claims for the forecasters; this arc has 1/3
the data and adds an internal-features structural prior. The
downgrade rule is the load-bearing falsifier against false-positive
confirmation.

---

## What out-of-scope means here

- **New model families beyond ridge / RF-depth-3 / 2-PC regression.**
  These three span the relevant capacity hypotheses (linear /
  nonlinear / dimensionality-reduced). Adding XGBoost, LSTM, etc.
  inflates deflation against a sample that can't support it.
- **New arcs.** This is a re-test of the existing 6-arc panel under
  a feature change; not a panel expansion. Panel expansion is a
  separate pre-reg.
- **Monthly cadence.** The prior arc tested monthly and lost.
  Re-testing the same cadence with internal features confounds the
  cadence change with the feature change.
- **Per-arc weight bounds.** Standard softmax + renormalisation;
  no `min_weight` or `max_weight` constraints (which would mask
  the model's true allocation signal).
- **Reinforcement learning.** Same data, more capacity. RL over 46
  quarterly decisions is not a real experiment.

---

## Acceptance criteria

1. Driver script `apps/meta_allocator/scripts/run_internal_features.py`
   computes the 3-trial grid over the 3-fold walk-forward, picks
   `model*` on folds 1+2, reports fold-3 OOS Sharpe + Ledoit-Wolf
   CI vs B3 and B2.
2. Per-arc feature data prep wires the strategy-internal spreads
   from existing modules (`ss_iv`, `relational.fingerprints`, etc.)
   with point-in-time discipline (no look-ahead on universe selection
   or normalisation).
3. Verdict label lands in `apps/docs/docs/leaderboard.md` per the
   table above.
4. Finding writes to
   `apps/docs/docs/findings/meta-allocator-internal-features.md`
   regardless of verdict (the null is informative — every feature
   class will have been tested).
5. If `confirmed-OOS`, update
   [`meta-allocator-regime-forecasting`](../findings/meta-allocator-regime-forecasting.md)
   with a closing note that the macro-features arc was superseded
   by internal-features. If `confirmed-null`, append a closing
   paragraph that the meta-allocator arc is closed and B3 is the
   permanent default.

---

## Literature grounding

- **Haddad-Kozak-Santosh 2020** — [*Factor Timing*, RFS](https://academic.oup.com/rfs/article-abstract/33/5/1980/5753962) /
  [NBER 26708](https://www.nber.org/system/files/working_papers/w26708/w26708.pdf):
  the load-bearing result. Long-short factor BM-spread predicts factor
  returns directly; macro features don't. Our internal-valuation-spread
  features are the direct analogue on our 6-arc panel.
- **Macrosynergy** — [factor timing research note](https://research.macrosynergy.com/factor-timing/):
  point-in-time macro at business-cycle frequency, not monthly. Our
  quarterly-cadence design honors this.
- **Mulliner-Harvey-Xia-Fang 2025** — [Swedroe summary](https://larryswedroe.substack.com/p/a-new-approach-to-regime-detection):
  high-yield spreads + stock-bond correlation + yield curve are the
  *macro* features that work — but quarterly, not monthly. Out of
  scope here (this pre-reg is internal-features-only); the macro
  features at quarterly cadence could be a separate sister arc if
  this one clears.
- **Alpha decay** — [Not All Factors Crowd Equally, arXiv 2512.11913](https://arxiv.org/pdf/2512.11913):
  IC trend per strategy directly measures alpha decay; we include
  this as a per-arc feature.

---

## Pointers

- Parent finding: [`meta-allocator-regime-forecasting`](../findings/meta-allocator-regime-forecasting.md) — locks the prior baseline.
- Sister finding: [`meta-allocator-no-vol-v3`](../findings/meta-allocator-no-vol-v3.md) — establishes the B3-without-vol_v3 fragility we must beat to be deployable.
- Verdict vocab: [`leaderboard.md#verdict-labels`](../leaderboard.md#verdict-labels).
- Existing per-arc data sources: `apps/meta_allocator/` directory + `ss_iv` + `relational.fingerprints` + `ss_macro`.
