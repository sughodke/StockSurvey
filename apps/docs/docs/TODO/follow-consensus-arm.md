# Follow-the-leader v1 — cross-sectional consensus arm

## Status — DONE 2026-05-28

Executed 2026-05-28. Verdict
[`confirmed-OOS`](../leaderboard.md#verdict-labels) vs the locked TODO
pre-reg bar (`partial-OOS` under the alternative user-brief CI-based
reading). See
[`findings/follow-consensus-arm`](../findings/follow-consensus-arm.md)
for the eval, fold-3 OOS verdict, and operational rule.

Headline: fold-1+2 reproduced v0 (Sh +1.020 vs +1.006; α +5.45 vs
+5.25 pp/yr). Fold-3 (unseen 2025-01 → 2025-10, n=198) posted
Sh +0.863 / α vs SPY +5.13 pp/yr / pos-Q 100%. Pooled standalone
defl-t +2.58 clears, but pooled edge-vs-SPY defl-t +1.37 sub +2.0
and fold-3 stationary-bootstrap 95% CI [−18.28, +25.83] does not
exclude 0 (n=198 too small).

Operational rule: ship as ensemble constituent with SPY β-hedge;
re-evaluate standalone deployment after xlsx refresh grows fold-3
past ~500 days.

---

## Original pre-registration (kept for audit)

Spawned by the v0 follower's `confirmed-null` verdict
([`findings/follow-leadership-disclosure`](../findings/follow-leadership-disclosure.md))
on 2026-05-25.

## Mechanism (why this could succeed)

The v0 eval surfaced a post-hoc grid cell on the **all-members
baseline arm** that posted ann Sh +1.006 / α vs SPY +5.25pp/yr at
`h=30, k=10, filter=frequency` on a 2-fold walk-forward
2019-21 ∪ 2022-24, entered at `filed + 1` trading day. The cell
clears the *partial-OOS* bar in isolation; it is recorded as
hypothesis here because it was not the pre-registered cell of the v0
arc.

The mechanism distinguishing this arm from the failed leadership
arm:

* **Aggregation primitive**: top-K by trailing-90d-distinct-member-
  buy count (cross-sectional consensus) rather than top-K by
  most-recent-buy on a leadership cohort cut.
* **Less arbitrage attention per name**: leadership trades are
  ranked highest on every aggregator feed; the broader cohort's
  consensus signal sits in the long tail of attention and is
  plausibly still informative at `filed + 1`.
* **Higher-N selection**: 273 disclosing members vs 64 leadership-
  cohort members means the cross-section of disclosure stream is
  ~4× wider; consensus structure is more meaningful when N is large.

## Pre-registered design

* **Entry rule**: `filed + 1` trading day (unchanged from v0 — the
  Bowne-2024 disclosure-lag honesty is non-negotiable).
* **Filter**: `frequency` — rank candidate tickers by count of
  distinct members buying that ticker over the trailing 90 trading
  days.
* **Cohort**: all-members (no leadership filter).
* **Cell**: `h=30, k=10, frequency`. Locked.
* **Folds**:
    | fold | train | val |
    |---|---|---|
    | fold-1 | 2014-2018 | 2019-2021 |
    | fold-2 | 2016-2020 | 2022-2024 |
    | fold-3 | 2018-2022 | **2025-2026** (requires fresh xlsx pull) |
  Fold-3 is the OOS verdict slice — folds 1+2 are reproductions of
  the v0 post-hoc grid result.

## Pre-registered bar (LOCK BEFORE EVAL)

* `confirmed-OOS` — across folds 1+2+3 pooled: ann Sh ≥ +0.85 AND
  α vs SPY ≥ +3pp/yr AND fold-3 (the unseen slice) α ≥ +1pp/yr.
* `partial-OOS`   — fold-1+2 reproduce v0 numbers AND fold-3 α ≥ 0
  AND pooled defl-t > +1.0.
* `confirmed-null` — fold-3 α vs SPY < +1pp/yr OR pooled defl-t < 0.
* `reversed-OOS`  — fold-3 α vs SPY < 0.

The crucial cut: **fold-3 is the only truly unseen slice**; folds 1+2
were grid-searched in v0 to find this cell. If fold-3 falls below
+1pp/yr it would establish that the +5.25pp/yr v0 result was a 2-fold
grid-search artifact, not a deployable edge.

## Extensions queued

* **Market-neutral L/S construction**: long top-K most-bought minus
  short top-K most-sold by the same frequency primitive. Should
  neutralize the cross-fold β-variance that drove v0's fold-1 →
  fold-2 swing. Borrow-cost 50 bps/yr charged on short notional.
* **Size-weighting by `Trade_Size_USD`**: trade-size ranges are
  coarse, but a $50k-$100k purchase plausibly carries more
  information per disclosure than a $1k-$15k purchase. Pre-reg
  weights ∝ midpoint(reported range).
* **Sub-cohort cuts on the all-members baseline**: party (D vs R),
  chamber (House vs Senate). Whether bipartisan consensus carries
  more info than partisan; whether Senate (with more committee
  briefings per member) beats House on the consensus signal.

## Driver / artifact plan

* Driver: `apps/follow/scripts/run_consensus_arm.py` (new). Reuse
  `follow.backtest.run_backtest` with `leadership_only=False`,
  `filter_mode='frequency'`, locked cell `h=30, k=10`. Three folds.
* Artifacts: `Output/follow-consensus-arm-walkforward.{npz,json}`
  with the locked `pre_registered_bar` field.
* ArcSpec keyed `follow-consensus-disclosure` in `compute_dsr.py`,
  same `mode='overlay'` vs SPY, `n_trials=1` (single locked cell).
* Leaderboard row + findings page per CLAUDE.md "after every
  experiment" protocol.

## Compute placement

Local (no Modal). The v0 driver runs ~2 min on the full panel.
Three folds × one cell ≈ 1 min.

## Parent

[`findings/follow-leadership-disclosure`](../findings/follow-leadership-disclosure.md)
— v0 `confirmed-null` verdict that spawned this arc per the
[verdict-→-next-experiment table's `confirmed-null` branch
("find an orthogonal lever")](../leaderboard.md#verdict-labels).
