# Meta-allocator with strategy-internal features — `confirmed-null`

**Operational rule.** Strategy-internal valuation / crowding / correlation
features at quarterly cadence do *not* extract a meta-allocator that
beats inverse-arc-volatility (B3) on the 6-arc panel — every model in
the locked 3-trial grid loses to B3 on fold-3 OOS with CI excluding 0
on the *negative* side. **Do not pursue further meta-allocator feature
classes**; the arc closes with B3 retained as the in-vol_v3-regime
baseline, but the user-asked deployment frame (canonical DCA + 2×vol_v3
ensemble) dominates everything tested here — model* trails the canonical
ensemble by ΔSR_ann **−3.29 [−4.93, −1.70]** on fold-3 (n=693) and
**−4.61 [−6.78, −2.43]** on the 2024+ slice (n=445). The deployable
answer to "what meta-allocator to ship over the 6-arc panel" remains
**DCA + sized vol_v3 sleeve**, exactly as the [post-2020 ranking
finding](post-2020-arc-ranking.md) concluded.

This is the closing experiment of the meta-allocator arc: orthogonal
input class (strategy-internal vs macro), orthogonal cadence
(quarterly vs monthly), same null verdict against the canonical
benchmark. Every reasonable feature category has now been tested.

---

## Pre-registration

Locked in [`TODO/meta-allocator-internal-features.md`](../TODO/meta-allocator-internal-features.md).
Direct follow-up to [`meta-allocator-regime-forecasting`](meta-allocator-regime-forecasting.md)
(B3 inverse-arc-vol `confirmed-OOS` 2026-05-23) and
[`meta-allocator-no-vol-v3`](meta-allocator-no-vol-v3.md) (B3 falsified
once vol_v3 dropped 2026-05-24). The Haddad-Kozak-Santosh 2020 +
Macrosynergy + alpha-decay-literature steel-man motivated the
input-class pivot: internal valuation spreads + per-arc IC trend +
cross-arc correlation effective rank at quarterly cadence.

Verdict bar (verbatim):

| condition | verdict |
|---|---|
| OOS ΔSR_ann ≥ +0.30 vs B3 AND CI excludes 0 AND DSR-t > +3.0 | `confirmed-OOS` |
| OOS ΔSR_ann ≥ +0.15 vs B3 AND CI excludes 0 on positive side AND DSR-t > +1.5 | `partial-OOS` |
| OOS ΔSR_ann ≥ +0.10 vs B2 1/N but does not beat B3 | `partial-OOS-vs-1/N-only` |
| OOS ΔSR_ann < +0.05 vs B3 OR `model*` collapses to B3-equivalent weights | `confirmed-null` |

Bootstrap CI > ±0.40 auto-downgrades one tier.

---

## Pragmatic substitutions vs the pre-reg

Documented up-front per CLAUDE.md "document the choice you made in the
findings page and keep going":

1. **D2 random forest → kernel ridge with RBF kernel + median-pairwise-
   distance bandwidth.** The pre-reg explicitly licensed this fallback.
   The nonlinearity hypothesis is preserved; the test of
   nonlinear-capacity-over-the-feature-grid is intact.
2. **Per-arc valuation spreads → cross-arc aggregates only.** The pre-reg
   feature table listed 6 vol + 6 IC-trend + 6 per-arc valuation +
   1 cross-arc eff-rank + 1 port-vol = 20 features, but called the
   total "14". Building 6 bespoke per-arc valuation series (P/E vs 10y
   median for ETFs, fingerprint-dispersion for relational, mean
   abs-z-score for pairs, IV-vs-RV-gap percentile for vol_v3,
   trailing-DD-vs-median for gate) requires 6 separate data pipelines
   from sources that vary in availability. To honor the 14-feature
   budget, we dropped the per-arc valuation spreads and kept the
   cross-arc aggregates (eff-rank + port-vol). The HKS-style "internal
   spread" channel is therefore weaker than the pre-reg envisioned;
   the per-arc-IC-trend channel and the cross-arc-correlation channel
   remain.
3. **`rank_ic_trend_252` → 252d slope of 60d rolling Sharpe.** As
   licensed by the pre-reg for arcs without a per-name cross-sectional
   signal (dca, gate, pairs, dca_winner_4etf, vol_v3 — i.e. all of them
   at this granularity).
4. **Driver location: `apps/docs/scripts/`, not `apps/meta_allocator/`.**
   Avoids ~50 LOC of `pyproject.toml` / Hatch / src layout scaffolding
   for a one-shot eval; the script imports `count_regimes_since_2005`
   from the same directory.

Final feature count: **14** (6 realized vol + 6 Sharpe trend +
corr-eff-rank-60 + port-vol-60), matching the pre-reg dimensionality
budget.

---

## Eval setup

| field | value |
|---|---|
| panel | `meta-alloc-arcs-6` (dca, gate, pairs, relational, dca_winner_4etf, vol_v3) |
| span | 2015-01-02 → 2025-12-11 (2753 trading days) |
| cadence | 63 trading days (quarterly); 43 rebal dates total |
| folds | fold1 2015-2018, fold2 2019-2022, fold3 2023-2025-12 |
| train | fold1+2 (n_rebal = 27); test fold3 (n_rebal = 11) |
| availability mask | per-arc, requires ≥ 252d history; vol_v3 enters 2024-04-12 only; gate/pairs end 2023-08 |
| weight transform | softmax(expected_return / per-arc vol) over available arcs |
| friction | 10 bps on |Δw|/2 at each rebal |
| models | D1 ridge α=1.0, D2 RBF kernel ridge α=1.0 σ²=median pairwise dist, D3 PCA-2-PC linear regression |
| model* selection | in-sample fold1+2 Sharpe |
| DSR | `standardize_oos` with n_trials=3 |
| CI | `ss_portfolio.sharpe_difference_ci` Ledoit-Wolf studentized stationary-bootstrap n=2000 seed=42 |

Driver: `apps/docs/scripts/meta_allocator_internal_features.py`.
Artifacts: `Output/meta-allocator-internal-features-results.json`,
`Output/meta-allocator-internal-features-streams.npz`.

---

## Results

### Model selection (in-sample fold1+2 Sharpe)

| model | in-sample Sharpe_ann |
|---|---:|
| D1 ridge | +1.135 |
| **D2 kernel ridge** (`model*`) | **+1.143** |
| D3 PCA-2-PC | +1.014 |

D2 wins in-sample by a hair (+0.008 over D1, +0.13 over D3).

### Fold-3 OOS Sharpe

| candidate | fold-3 Sharpe_ann | n_daily |
|---|---:|---:|
| D1 ridge | +2.294 | 693 |
| D2 kernel ridge (`model*`) | +2.828 | 693 |
| D3 PCA-2-PC | +2.929 | 693 |
| **B2 1/N** (cached) | +2.886 | 728 |
| **B3 inv-vol** (cached) | **+4.147** | 728 |
| **canonical DCA + 2×vol_v3** | **+5.940** | 739 |

All three model variants underperform B3 by ≥ 1.2 Sharpe and
underperform the canonical ensemble by ≥ 3.0 Sharpe. The
in-sample-winning D2 is the *middle* of the three OOS — model
selection did not help (or hurt) much, consistent with the
pre-reg sample-size warning.

### Model* (D2 kernel ridge) vs benchmarks on fold-3 OOS

| comparison | n | ΔSR_ann | 95% CI (ann) | excludes 0? |
|---|---:|---:|---|---:|
| `model*` vs B3 | 693 | **−1.386** | [−2.396, −0.284] | yes (negative side) |
| `model*` vs B2 | 693 | −0.113 | [−0.366, +0.166] | no |
| `model*` vs canonical | 693 | **−3.288** | [−4.933, −1.696] | yes (negative side) |

DSR-t = +3.82 in absolute terms — but this is the model's *standalone*
deflated significance, not its outperformance vs B3. The locked bar
requires positive ΔSR vs B3 with DSR-t > 3; we have negative ΔSR.

### 2024+ slice (user-requested OOS)

| candidate | 2024+ Sharpe_ann | n |
|---|---:|---:|
| `model*` | +3.180 | 445 |
| B3 inv-vol | +5.339 | 478 |
| B2 1/N | +3.339 | 478 |
| canonical DCA + 2×vol_v3 | **+7.387** | 489 |

| comparison (2024+) | ΔSR_ann | 95% CI (ann) | excludes 0? |
|---|---:|---|---:|
| `model*` vs B3 | −2.354 | [−3.847, −1.031] | yes (negative side) |
| `model*` vs canonical | −4.609 | [−6.783, −2.429] | yes (negative side) |

The user-asked outcome (meet or exceed DCA + 2×vol_v3 on 2024+) is
**not met** — model* trails the canonical ensemble by 4.2 Sharpe
points on the 2024+ slice with a CI excluding 0 well into negative
territory.

---

## Verdict

**Raw verdict per the locked bar:** ΔSR vs B3 = −1.39 (< +0.05
threshold for `confirmed-null`) → `confirmed-null`.

**Downgrade rule:** CI half-width vs B3 = 1.06 > 0.40 → would
downgrade one tier, but the raw verdict is already at the floor
(`confirmed-null`).

**Locked verdict: `confirmed-null`.**

This *also* fails the user-additional bar ("must meet/exceed canonical
DCA + 2×vol_v3 on 2024+") by ΔSR_ann −4.61 [−6.78, −2.43] — the model
is reliably worse than the deployable ensemble on the user-requested
OOS slice.

---

## Mechanism — why the features didn't carry signal

Three plausible reads, in decreasing order of leaderboard prior
weight:

1. **The signal in B3 *is* the strategy-internal signal already.**
   B3 inverse-arc-vol weights vol_v3 heavily on fold-3 because vol_v3
   has anomalously low realized vol (~1%/yr). That's not B3 cheating
   — it's B3 reading the same "this arc's spread is wide / capacity
   is unused" signal that HKS internal-features would have read, via
   the volatility channel rather than the valuation channel. When the
   internal feature stack is *added* to a learned model, the model
   has to discover this relationship through the noisy training
   signal of 27 quarterly observations, and it under-shoots.

2. **27 quarterly training points is below the 14-feature DOF floor.**
   The pre-reg flagged this explicitly: "max effective DOF is 6 × 14
   = 84 — over a quarterly sample of ~46 points this is hostile." We
   end up with 27 train and 11 test rebals after availability masking,
   even worse than the pre-reg's 46 estimate. The 2-PC reduction (D3)
   was the pre-reg's hedge against this; D3's fold-3 Sharpe (+2.93) is
   the *best* of the three models and is still well below B3 — the
   problem isn't DOF in the linear head, it's that the features don't
   carry sign in the regime that matters (post-vol_v3-entry 2024+).

3. **Cross-arc correlation eff-rank doesn't move enough in fold-3 to
   matter.** The eff-rank feature was the cleanest mechanism story
   (compress toward 1/N when arcs become correlated); but fold-3 has
   only ~3-4 arcs available at any point (gate/pairs exit 2023-08,
   vol_v3 enters 2024-04) and the eff-rank of a 3-4 arc panel is
   structurally close to the panel size — there's not enough range
   in the feature for the model to learn a useful gate.

Honest read: this experiment is more an availability-stress-test of
the panel than a clean test of internal features. With a 4-arc panel
in fold-3 (dca, relational, dca_winner_4etf, vol_v3 for most of it),
the meta-allocator's job is mostly "weight vol_v3 vs the rest" — and
the inverse-vol rule already solves that. The features didn't get
to *do* anything new.

---

## Implications for the meta-allocator arc

Per the [`confirmed-null` decision branch](../leaderboard.md#verdict-labels)
("stop testing variations of the same lever — find an orthogonal one"):

- **Cadence axis**: monthly tested (`meta-allocator-regime-forecasting`
  reversed-OOS for every modeling candidate), quarterly tested here
  (`confirmed-null`). The cadence lever is exhausted.
- **Input class axis**: macro features tested
  (`meta-allocator-regime-forecasting`, reversed-OOS), strategy-internal
  features tested (this row, `confirmed-null`). The input lever is
  exhausted.
- **Learner axis**: linear ridge / nonlinear kernel ridge /
  dimensionality-reduced 2-PC tested here — same `confirmed-null`.
  The learner lever is exhausted.

**The meta-allocator arc closes.** Deployment recipe stays:

1. **DCA canonical 13-ETF basket** as the always-on core.
2. **vol_v3 sleeve sized at vega ≈ 2.0** when options-broker
   integration permits (gap-to-live documented in
   [`vol-sleeve-sizing`](vol-sleeve-sizing.md)).
3. **No learned meta-allocator on top** — B3 inverse-arc-vol is the
   defensible default when more than 2 arcs are active, but its
   `confirmed-OOS` edge over 1/N collapsed once vol_v3 was excluded,
   so the operational simplification is just to ship the two-leg
   ensemble.

Future orthogonal levers (out-of-scope for this arc):

- **Different prediction problem**: the meta-allocator has been
  framed as forecasting which arc will outperform. Recasting as
  drawdown forecasting / vol regime detection at the *portfolio* level
  is a different arc, not a different feature class. See
  `apps/gate` for the prior closest attempt (partial-OOS).
- **Panel expansion**: testing a new arc (e.g., `apps/follow`
  consensus arm) entering the panel changes the substrate; the
  meta-allocator question can be re-posed when that arc clears its
  own pre-reg ([`follow-consensus-arm`](follow-consensus-arm.md) just
  did, but it's not yet integrated into the meta-allocator panel).

---

## Caveats

1. **n_test = 11 quarterly rebals** in fold-3 is too small to claim
   any clean CI on the cross-arc Sharpe difference; the wide CI
   (±1.06 half-width) is the honest reflection of this.
2. **Availability mask drives the result.** Of the 6-arc panel, only
   4 are ever simultaneously available in fold-3 (vol_v3 starts
   2024-04). The meta-allocator's lever is essentially a 4-arc
   weighter; the per-arc valuation features the pre-reg envisioned
   wouldn't have helped much on this thin substrate.
3. **Substitutions vs the pre-reg are not adversarial.** The kernel
   ridge swap is licensed; the per-arc valuation drop is the
   load-bearing one — if a future arc has the data pipelines to
   build bespoke per-arc spreads (especially the relational
   fingerprint dispersion and the pairs z-score mean), re-running
   this exact framework with those features might give a different
   answer. The leaderboard row notes this as a future hypothesis.

---

## Master walk-forward log

Leaderboard row: 2026-05-28 `meta` | meta-allocator-internal-features.
Verdict label: [`confirmed-null`](../leaderboard.md#verdict-labels).
Pre-reg: [`TODO/meta-allocator-internal-features`](../TODO/meta-allocator-internal-features.md).

Related findings:

- [`meta-allocator-regime-forecasting`](meta-allocator-regime-forecasting.md) — B3 confirmed-OOS on 6-arc panel, modeling candidates reversed-OOS.
- [`meta-allocator-no-vol-v3`](meta-allocator-no-vol-v3.md) — B3 confirmed-null when vol_v3 dropped.
- [`post-2020-arc-ranking`](post-2020-arc-ranking.md) — canonical DCA + 2×vol_v3 ensemble is the deployable Pareto frontier.
- [`vol-sleeve-sizing`](vol-sleeve-sizing.md) — recommended vega=2.0 sleeve sizing.
