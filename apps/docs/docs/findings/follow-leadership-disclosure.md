# Follow-the-leader congressional-disclosure follower

**Operational rule:** *do not deploy* the leadership-tier
congressional-disclosure follower as a long-only US-equity strategy.
At the disclosure-lag-honest entry (`filed + 1` trading day) the
leadership filter produces **−0.09pp/yr alpha vs SPY** at the
pre-registered deployable cell (h=60 / k=25 / recency) — sub-passive,
and dominated on Sharpe by the all-members baseline with the
`frequency` filter (+1.01 ann Sh, +5.25pp/yr alpha) it was supposed
to beat. Bowne 2024's prediction that the disclosure-lag would
collapse the apparent alpha is **confirmed** here.

| arm | best mean ann Sharpe (folds) | best mean alpha pp/yr | pos-quarter | DSR-t |
|---|---:|---:|---:|---:|
| **leadership-only**     | +0.796 (h60/k25/recency) | **−0.09** | 0.77 | −1.91 |
| **all-members**         | +1.006 (h30/k10/freq)    | **+5.25** | 0.73 | (not arc-keyed) |
| SPY pooled-val (2019-21 ∪ 2022-24) | +0.897 | 0.00 (def) | — | — |

Verdict on the pre-registered deployable cell vs the locked bar:
[`confirmed-null`](../leaderboard.md#verdict-labels) — ann_sharpe_net
+0.86 ≥ the 0.5 floor but alpha vs SPY < +1pp/yr is the killing
condition. The `confirmed-null` label is the right one because the
edge mechanism (information asymmetry) is *real* in the literature
but the *disclosure-lag-and-public-filter combination* leaves no
deployable residue.

## Eval setup (locked before eval)

* **Disclosure source** — Quiver Quantitative bundle from
  ``ilya2026/congressional-alpha`` (`Congressional Trades.xlsx`,
  106,331 raw rows 2012-02 → 2025-10 across House + Senate, of which
  41,408 are equity-stock rows usable for the follower). Cross-checked
  against `timothycarambat/senate-stock-watcher-data` (Senate-only,
  no disclosure-date column in the aggregate slice — used for
  spot-check only). Loaders: `ss_loaders.load_congressional_trades_xlsx`,
  `ss_loaders.load_senate_stock_watcher`.
* **Legislator metadata** — `unitedstates/congress-legislators`
  project (`legislators-current.json` + `legislators-historical.json`).
  Loader: `ss_loaders.load_legislator_metadata`.
* **Leadership filter** — `ss_loaders.LeadershipFilter`, point-in-time.
  A member qualifies at disclosure date *t* if EITHER:
    1. they appear in a hand-curated roster of Speaker / Majority+Minority
       leaders / chair-or-ranking-member of Intelligence, Armed
       Services, Financial Services / Banking, Ways and Means / Finance,
       Appropriations (each row tenured `[start, end]`); or
    2. they have ≥10 years cumulative term-time as of *t*.
  64 unique bioguides survive in the 2014-2024 span out of 273
  members with disclosed equity trades (24% of disclosing members
  qualify as leadership).
* **Entry rule** — `filed + 1 trading day`. **NOT** the transaction
  date. STOCK Act 2012 lag distribution in this panel: median 24-27
  days, mean 36-76 days, p90 ~117 days. Bowne 2024 specifically
  predicts the entry-on-disclosure timing collapses the apparent
  alpha; this is the falsification test.
* **Exit rule** — fixed-horizon hold; carry into delisting (NaN
  forward-treated as zero return inside the basket).
* **Friction** — 10 bps round-trip charged on `|Δw|` per day.
* **Filters** — purchases only (`Transaction in {Purchase, PURCHASE}`).
  No short basket on sales.
* **Folds**:
  | fold | train | val |
  |---|---|---|
  | fold-1 | 2014-01-01 → 2018-12-31 | 2019-01-01 → 2021-12-31 |
  | fold-2 | 2016-01-01 → 2020-12-31 | 2022-01-01 → 2024-12-31 |
  Train is informational only — the strategy has no fitted
  parameters; the grid is pre-registered. Val is the OOS slice.
* **Grid** — `hold_days ∈ {30, 60, 90} × top_k ∈ {10, 25, 50} ×
  filter_mode ∈ {recency, frequency}` = 18 cells per arm.
* **Pre-registered bar** (locked into the NPZ `pre_registered_bar`
  field BEFORE the eval):
  | label | criteria |
  |---|---|
  | confirmed-OOS  | ann_sh ≥ +1.0 AND defl-t > +2.0 AND α vs SPY ≥ +5pp/yr AND pos-Q ≥ 0.60 |
  | partial-OOS    | ann_sh ≥ +0.5 AND defl-t > +1.0 AND α vs SPY ≥ +2pp/yr |
  | confirmed-null | ann_sh < +0.3 OR α vs SPY < +1pp/yr |
  | reversed-OOS   | α vs SPY < 0 |

## Per-fold + per-grid-cell results

### Leadership-only arm

Best cell within each fold:

| fold | hold | k | filter | val ann Sharpe | val α vs SPY pp/yr | pos-Q | max-DD |
|---|---:|---:|---|---:|---:|---:|---:|
| fold-1 (2019-21) | 60 | 25 | recency   | **+1.209** | +3.45 | 0.91 | −0.339 |
| fold-1 (2019-21) | 90 | 25 | recency   | +1.194     | +3.31 | 0.91 | −0.342 |
| fold-1 (2019-21) | 30 | 25 | recency   | +1.148     | +2.12 | 0.91 | −0.349 |
| fold-2 (2022-24) | 30 | 10 | frequency | +0.449 | −1.68 | 0.64 | −0.261 |
| fold-2 (2022-24) | 60 | 25 | recency   | +0.383 | −3.63 | 0.64 | −0.210 |
| fold-2 (2022-24) | 30 | 25 | recency   | +0.371 | −3.75 | 0.64 | −0.229 |

**Fold-1 looked like a partial-OOS-borderline win** (best-cell α
+3.45pp/yr, all top cells α > 0). **Fold-2 collapsed** (every
top-3 cell α < 0, best at −1.7pp/yr). Mean over folds (best cell):
α vs SPY −0.09pp/yr — the fold-1 lift and the fold-2 loss net to
zero alpha.

### All-members baseline (the apples-to-apples comparator)

| fold | hold | k | filter | val ann Sharpe | val α vs SPY pp/yr | pos-Q | max-DD |
|---|---:|---:|---|---:|---:|---:|---:|
| fold-1 (2019-21) | 30 | 25 | frequency | +1.106 | +0.41 | 0.73 | −0.299 |
| fold-1 (2019-21) | 60 | 25 | frequency | +1.104 | +0.00 | 0.82 | −0.298 |
| fold-1 (2019-21) | 60 | 10 | frequency | +1.097 | +1.50 | 0.82 | −0.321 |
| fold-2 (2022-24) | 30 | 10 | frequency | **+0.915** | **+8.85** | 0.73 | −0.261 |
| fold-2 (2022-24) | 60 | 10 | frequency | +0.734 | +5.77 | 0.73 | −0.310 |
| fold-2 (2022-24) | 30 | 25 | frequency | +0.647 | +1.59 | 0.73 | −0.233 |

The best all-members cell on mean-over-folds is **h=30 / k=10 /
filter=frequency** at ann Sh +1.006, α +5.25pp/yr — would clear the
*partial-OOS* bar in isolation. Leadership-only's best cell loses
to it by **+1.1pp/yr Sharpe and ~+5pp/yr alpha**.

### DSR ladder placement

Per `apps/docs/scripts/compute_dsr.py` (deployable cell h=60/k=25/
recency, overlay vs SPY, n_trials=18, sharpe_std_ann=0.072):

| key | mode | trials | annSh (excess) | defl-t | DSR |
|---|---|---:|---:|---:|---:|
| follow-leadership-disclosure | overlay | 18 | −0.010 | **−1.905** | 0.028 |

Sits in the "below-noise overlay" tier next to `gate-v0-drawdown`
(defl-t −1.79) and `regime-velocity-universe-agnostic` (defl-t
−2.15). Far below the t=+2.0 confirmed-OOS bar, below the t=+1.0
partial-OOS bar. The deflation here is generous (n=1510 daily obs
makes the structural-noise term dominate the deflation); the raw
fact is the strategy doesn't outperform SPY at the deployable cell.

## Mechanism

Why does the literature claim alpha here and the eval find none?

* **Information edge is real but mostly pre-disclosure.** Members
  receive non-public information through committee briefings and
  lobbyist meetings; the alpha that exists is *captured by the
  member at the transaction date*, weeks before the public sees the
  PTR. Bowne 2024's clean point: the disclosure-lag is a structural
  feature of the STOCK Act, not a fixable timing artifact.
* **The crowd has been priced in.** Quiver Quantitative, Unusual
  Whales, and a half-dozen other commercial trackers have been
  publishing leader trades within hours of filing since ~2020.
  Fold-1 (2019-21) is on the boundary where the data was just
  becoming widely watched (and the strategy posts α +3.45pp/yr);
  fold-2 (2022-24) is the regime where the disclosure-feed alpha is
  fully arbitraged (α −1.68 to −3.75pp/yr). The collapse from
  fold-1 to fold-2 is the same story `findings/factor-crypto-venue.md`
  tells about retail-traded venues.
* **Leadership filter is a 24% cohort cut.** It restricts the
  universe but does not lift the per-trade information content
  enough to overcome the lag. The all-members frequency-filter
  cell (+5.25pp/yr) suggests a *cross-sectional consensus signal*
  — when many members buy a name independently — carries more
  weight than the named-leadership tier alone, but that arm wasn't
  pre-registered as the deployable cell, so its +5.25pp/yr is a
  diagnostic / next-arc hypothesis, not a deployable claim.
* **Capacity is constrained by retail attention, not by member
  wealth.** A leadership member's $50k purchase doesn't move
  prices; the *public follow-the-trade signal* does. Once the
  follow-trade aggregators amplify the disclosure, the price has
  already absorbed the news by the time `filed + 1` trading day
  arrives. Bid-ask drift on PTR-day news is ~30-60 bps measured by
  Unusual Whales / Quiver writeups; that swallows the alpha.

## Three honest surprises

1. **Leadership filter is *worse* than all-members at every grid
   cell on fold-2.** The pre-registered hypothesis assumed the
   leadership subset would dominate on information content. In the
   2022-24 sample it underperforms the wider cohort by 1.1 ann
   Sharpe (+0.40 vs +0.92) and 8-10pp/yr in alpha. **Mechanism
   speculation**: leadership members are also *most-followed* —
   their trades get the most aggregator attention, hence the most
   front-running. Below-the-fold members' trades are noisier but
   less arbitraged.
2. **Fold-1 (2019-21) looked like a real edge.** Leadership ann Sh
   +1.21, α +3.45pp/yr at the best cell, 91% positive quarters. If
   we'd stopped there, the headline would be *partial-OOS*. The
   fold-2 collapse is what changes the verdict. This is a textbook
   example of the brief's pre-reg-design pattern: two folds, not
   one, *exactly to prevent the fold-1 win from being mistaken for
   the answer*.
3. **The `frequency` filter dominates `recency` for all-members,
   but the opposite is true for leadership.** At leadership-only,
   `recency` (rank top-K by most recent open-event) wins; at
   all-members, `frequency` (rank top-K by trailing-90d count of
   opens on this ticker) wins by a wide margin. **Interpretation**:
   for a small leadership cohort, single-name purchases by senior
   members carry information; for the broad cohort, **consensus
   across many members** is the real signal — which is the cross-
   sectional buy-pressure signal Unusual Whales publishes openly.

## Implementation gaps

* The `Congressional Trades.xlsx` source is **a third-party Quiver
  bundle**, not the canonical EFD + Office-of-the-Clerk PTR feed.
  Quiver scrapes both, but their classification of `Filed` vs
  `Quiver_Upload_Time` is opaque (we used `Filed` as the disclosure
  date per the column name). A canonical re-fetch via the
  House.gov FD JSON feeds + Senate efdsearch.senate.gov would
  upper-bound any classification error; not done here.
* Hand-curated leadership roster has ~50 member-roles; the
  ≥10-year-tenure heuristic backfills broader senior-member
  coverage. Missing: committee membership (non-chair) for the six
  named committees. The `committee-membership-current.json` feed is
  current-Congress-only, so a clean point-in-time non-chair
  committee filter would require pulling per-Congress
  historical-committee data from govtrack — *bounded extra work
  for a future arm, not done here*.
* No long-only ETF beta-hedge variant. The strategy is +β long
  basket vs SPY; a market-neutral construction would isolate the
  information-edge cleanly. **The all-members frequency-filter
  arm posting +5.25pp/yr α even without a hedge is a stronger
  positive signal than the leadership arm — that's where a
  follow-up arc would re-pre-register.**
* No size-weighting by `Trade_Size_USD`. The brief's v0 rule
  (equal-weight) is preserved; the trade-size ranges are coarse
  ($1k-$15k, $15k-$50k, etc.) and learning from them is a separate
  arc.

## Proposed next experiment

Per the [`confirmed-null` row of CLAUDE.md's verdict → next-
experiment table](../leaderboard.md#verdict-labels): *"Stop testing
variations of the same lever — find an orthogonal one."* The
candidate orthogonal lever is **not "different leadership cut"**
but **"different aggregation primitive"**:

> *Hypothesis*: a **cross-sectional consensus** disclosure-follower
> (top-K by trailing-90-day-distinct-members-buying count, NOT
> top-K by recency on a leadership sub-cohort) is the deployable
> arm. Pre-reg test: same folds, same friction, same `filed + 1`
> entry rule; the all-members frequency-filter cell at h=30 / k=10
> on the SAME data posted ann Sh +1.01 / α +5.25pp/yr / pos-Q 0.73
> in the post-hoc grid. Lock it as the pre-registered cell and
> re-run on fold-3 (2025-2026 OOS, requires fresh xlsx).

Sub-test: market-neutral long/short version of the consensus arm
(long top-K most-bought, short top-K most-sold by frequency).
Predicted to *neutralize the cross-fold β-variance* that drove the
fold-1 vs fold-2 swing here.

## Master walk-forward log pointer

See the [leaderboard](../leaderboard.md) row dated 2026-05-25,
`follow | leadership-disclosure`, verdict
[`confirmed-null`](../leaderboard.md#verdict-labels). Driver:
`apps/follow/scripts/run_walkforward.py`. Artifacts:
`Output/follow-walkforward.{npz,json}`. DSR row keyed on
`follow-leadership-disclosure` in `apps/docs/scripts/compute_dsr.py`.

The arc closes with one positive byproduct (the cross-sectional
consensus signal posts a top-of-class fold-2 number) — recorded
here as the next-arc hypothesis rather than promoted to a deployable
verdict, because it was not the pre-registered cell.
