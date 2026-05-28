# Follow-the-leader v1 — all-members consensus arm, fold-3 OOS verdict

**Operational rule.** *Deploy the all-members congressional-consensus
follower (cell `h=30, k=10, filter=frequency`, entry `filed + 1`, equal-
weight long-only, 10 bps friction) as an **ensemble constituent paired
with a SPY beta-hedge**, NOT as a standalone single-arm bet. The point-
estimate OOS verdict on the unseen 2025 fold is strong (α +5.13 pp/yr,
Sh +0.86, pos-Q 100%), but the 198-day stationary-bootstrap CI on
fold-3 α is wide ([−18, +26] pp/yr) and the pooled edge-vs-SPY
deflated-t is +1.37 (not the +2.0 user-brief threshold). Refresh the
disclosure xlsx weekly; re-tighten the CI as fold-3 grows from 198 to
~500+ days before a standalone deployment.*

## Status

Pre-registered follow-up to the v0 leadership-arm
[`confirmed-null`](follow-leadership-disclosure.md) verdict. Spawned
2026-05-25 in
[`TODO/follow-consensus-arm.md`](../TODO/follow-consensus-arm.md);
executed 2026-05-28. Verdict per the locked TODO bar:
[`confirmed-OOS`](../leaderboard.md#verdict-labels).

## Why this arc existed

The v0 row's all-members baseline cell `h=30, k=10, filter=frequency`
posted ann Sh +1.006 / α vs SPY +5.25 pp/yr on the post-hoc 2-fold
grid (2019-21 ∪ 2022-24). That cell was the load-bearing surprise of
the v0 eval — it cleared the partial-OOS bar *in isolation* but was
**not pre-registered** as the load-bearing cell of v0 (the leadership-
filter hypothesis was). Per CLAUDE.md `confirmed-null` decision branch
("find an orthogonal lever"), the v1 arc locked the consensus cell
ahead of test data, then waited for a fold the post-hoc search had
never seen.

## Eval setup

| field | value |
|---|---|
| universe | all-members US House+Senate disclosed purchases (no leadership filter) |
| disclosure source | Quiver-bundled `Congressional Trades.xlsx` (33,526 disclosures in span; 15,997 purchases; 13,714 surviving Stooq join; 175 disclosing bioguides) |
| price panel | Stooq daily archive, 1623 unique tickers, ETFs included |
| entry rule | `filed + 1` trading day (Bowne 2024 disclosure-lag-honest) |
| filter | `frequency` — count of distinct members buying ticker in trailing 90 trading days |
| top-K | 10 (locked) |
| hold horizon | 30 trading days (locked) |
| friction | 10 bps per `|Δw|` |
| folds | fold-1 val 2019-2021 (n=757), fold-2 val 2022-2024 (n=753), **fold-3 val 2025-01-01 → 2025-10-16 UNSEEN (n=198, bounded by xlsx end)** |
| benchmark | SPY (overlay framing for ladder) |
| deflation | n_trials=1 (single locked cell, no within-row search) |

## Fold-1+2 confirmation — did v0 +5.25 pp/yr replicate?

| metric | v0 reported | v1 measured | match? |
|---|---|---|---|
| ann Sharpe (net) | +1.006 | **+1.020** | ✓ within MC noise |
| α vs SPY (excess, pp/yr) | +5.25 | **+5.45** | ✓ within MC noise |
| α vs SPY (β-adj, pp/yr) | (not recorded) | +4.98 | (new) |

The v0 post-hoc result reproduces cleanly on the same 2019-2024 span
with the v1 driver. Sub-pp-per-year drift comes from refresh of the
xlsx between v0 and v1 (the xlsx is continually updated; same 2019-
2024 calendar window but disclosure rows for that window can grow as
late filings land).

## Fold-3 OOS verdict — the load-bearing result

| metric | fold-3 (2025-01-01 → 2025-10-16, n=198d) |
|---|---|
| ann Sharpe (net, strat) | **+0.863** |
| ann Sharpe (SPY) | +0.847 |
| α vs SPY (excess, pp/yr) | **+5.13** |
| α vs SPY (β-adj, pp/yr) | +2.18 |
| pos-quarter fraction | **1.00 (3/3 quarters positive)** |
| max DD | −25.5% |
| stationary-bootstrap 95% CI on α_excess (n_boot=2000, seed=42, avg_block=20) | **[−18.28, +25.83] pp/yr** |
| CI excludes 0? | **no** |
| mean basket size | 10.0 names |

| metric | pooled 1+2+3 (n=1708d) |
|---|---|
| ann Sharpe (net, strat) | +0.998 |
| α vs SPY (excess, pp/yr) | +5.41 |
| α vs SPY (β-adj, pp/yr) | +4.63 |
| pooled deflated-t (edge vs SPY) | +1.366 |
| pooled deflated-t (standalone strat) | +2.576 |

Per the TODO-locked bar (pooled Sh ≥ +0.85 AND pooled α ≥ +3 pp/yr
AND fold-3 α ≥ +1 pp/yr), all three thresholds clear → **confirmed-
OOS**. Under the user-brief alternative reading (fold-3 CI excludes
0 AND pooled defl-t > +2.0), the verdict is partial-OOS; the
standalone-strategy defl-t +2.58 clears but the edge-vs-SPY defl-t
+1.37 does not, and the CI is wide at n=198. **Per CLAUDE.md, the
TODO is the locked source of truth and wins.** The CI tension is
recorded as the load-bearing nuance for the operational rule.

## Per-fold mechanism breakdown

| fold | strat Sh | SPY Sh | α_excess (pp/yr) | α_β-adj (pp/yr) | pos-Q | MDD | regime |
|---|---|---|---|---|---|---|---|
| fold-1 (2019-21) | +1.112 | +1.163 | +1.98 | +1.76 | 0.73 | −31.8% | bull (SPY ripping) |
| fold-2 (2022-24) | +0.919 | +0.574 | +8.94 | +8.38 | 0.73 | −26.2% | bear → recovery |
| fold-3 (2025) | +0.863 | +0.847 | +5.13 | +2.18 | 1.00 | −25.5% | post-election vol |

The strategy is a **relative-strength-in-not-bull-regimes** primitive:
when SPY's own Sharpe is high (fold-1 +1.16), the strategy's excess
alpha compresses; when SPY's own Sharpe is moderate (fold-2 +0.57,
fold-3 +0.85), the strategy delivers its full +5-9 pp/yr alpha. This
matches the cross-sectional-consensus interpretation: in raging bull
regimes the broad market is the rising tide; in choppier regimes the
consensus aggregator picks out the names insiders had reason to
believe in.

## Mechanism (Bowne 2024 inverted on the consensus arm)

The v0 finding established that **Bowne 2024's disclosure-lag killed
the leadership-arm alpha**: senior-member trades are the most-
arbitraged segment because aggregators (Quiver / Unusual Whales /
Capitol Trades) publish them within hours of filing. By `filed + 1`
the price has absorbed the disclosure.

The all-members consensus arm flips this: instead of betting on
*who* disclosed, it bets on *how many distinct members converged
on the same ticker* in the trailing 90 days. This:

* **Captures the long-tail of disclosure attention.** A single back-
  bencher's PTR doesn't move price (below the aggregator-front-
  running threshold), but 3-5 distinct back-benchers converging on
  the same name *as a cross-sectional consensus* still carries
  information that the aggregator pipeline isn't ranking on.
* **Is robust to the disclosure lag.** The signal is "trailing-90d-
  consensus", not "most-recent-buy" — by construction it expects to
  enter after the lag.
* **Has higher information density per row.** 1623 unique all-members
  tickers vs the v0 leadership arm's 762 — the broader cohort gives
  the frequency rank more denominator to discriminate against.

The standalone-strategy defl-t clears +2.0 (+2.58) but the edge-vs-
SPY defl-t does not (+1.37). This is because the strategy's own
Sharpe is robust (+1.0 on the pooled stream) but SPY's Sharpe on the
same 2019-2025 span is also ~0.9 — the excess-return alpha is large
on a per-year basis (+5.4 pp/yr) but the per-day excess-return
variance is high relative to the per-day mean, so studentizing the
excess requires more years to tighten than studentizing the
standalone return.

## Three honest surprises

1. **Fold-1 is the WEAKEST OOS slice (α +1.98 pp/yr)** — even though
   fold-1 is the v0 grid window. SPY's own Sh in fold-1 is +1.163
   (raging bull), so relative outperformance is hard. The v0
   +5.25 pp/yr cross-fold mean was driven by fold-2's +8.94 pp/yr.
   The strategy is regime-conditional.

2. **Fold-3 posts 100% positive quarters (3/3)** including 2025-Q1
   (the post-election vol window). The aggregator did NOT roll over
   on regime change — weakly falsifies a "this was a 2022-Trump-
   cycle-specific artifact" alternative explanation.

3. **Standalone-strategy defl-t +2.58 clears the user-brief +2.0
   bar, but the edge-vs-SPY defl-t +1.37 does not.** The pooled
   strategy IS a deflated +2σ standalone book; the EDGE over SPY
   needs more years to studentize because the per-day excess-return
   variance is high.

## Operational recommendation

**Ship as ensemble-constituent paired with a SPY beta-hedge. Do NOT
ship as standalone single-arm bet yet.**

Deployment recipe:

1. **Cell**: `h=30, k=10, filter='frequency', commission=10bps`.
2. **Entry rule**: `filed + 1` trading day on Quiver-bundled
   `Congressional Trades.xlsx` (refresh weekly).
3. **Universe**: all disclosing members (no leadership filter).
4. **Construction**: equal-weight long basket of top-10 most-
   consensus-bought names.
5. **β-hedge**: short SPY at 1× notional to neutralize the +β
   exposure (fold-1 demonstrated the un-hedged arm under-Sharpes
   SPY in bull regimes; the α is the load-bearing edge, not the +β
   carry).
6. **Risk rails**: per-name 25% cap via `ss_portfolio.
   apply_position_cap` (k=10 already enforces 10% nominal — cap is a
   safety rail); dry-run-by-default; kill-switch file.
7. **Re-evaluate after xlsx grows fold-3 to ~500+ days** — the
   bootstrap CI on α should tighten meaningfully and either confirm
   or downgrade the standalone-deployment readiness.

## Master walk-forward log

Verdict
[`confirmed-OOS`](../leaderboard.md#verdict-labels) on the locked
TODO pre-reg bar; **partial-OOS** under the alternative user-brief
CI-and-defl-t reading. Parent v0
[`findings/follow-leadership-disclosure`](follow-leadership-disclosure.md).
TODO closed: [`TODO/follow-consensus-arm`](../TODO/follow-consensus-arm.md).
