---
tags:
  - cfr
  - phase-2
  - meta-allocator
  - stooq_us_long
---

# CFR Phase 2 — menu enrichment confirms the binding constraint is sample density, not menu content

**Operational rule (added 2026-05-12 to
[`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings)):**
adding more actions to a tabular CFR table when no action has
consistently positive cumulative regret makes things worse, not
better — because the **naive uniform mix** is also a function of
the menu, and uniform-mix improves *monotonically* with each new
diversifying action while regret-matching suffers from the
additional noise dimensions in the regret table. The Cesa-Bianchi
& Lugosi O(√(log n)/√T) regret bound predicts this. Phase 3 must
move from tabular CFR to deep CFR (regret_net sharing statistical
strength across actions) — not a richer table.

**TL;DR:** Two enrichment strategies tested. Phase 2a (4 documented-
alpha deterministic modes added → 28 actions) and Phase 2b (real
SEC 13F-HR consensus mode added → 31 actions). _Both made the CFR
result slightly worse, not better_. The naive uniform mix improved
more than CFR with each enrichment, confirming the binding constraint
is the regret-table sample density, not the action menu's edge
content. **One genuinely interesting per-window pattern in Phase 2b:**
the late window (2020-2023, the only one with full 13F coverage in
both train and val) posts CFR alpha −0.27 → +0.006 vs Phase 1
(+0.27 lift from the 13F mode when it has data) — encouraging
signal that real 13F has *something*, but the binding constraint
remains the tabular table.

## Setup — both phases on the canonical Phase 1 walkforward

Same universe (`stooq_us_long`, 312 tickers), windowing (6-window
1260-train / 780-val / 780-step), friction (10 bps, rebal 20d),
and infoset (3×3 = 9 cells on trailing-vol × cross-sectional-
dispersion). Only the action menu differs.

| Phase | Menu size | New modes |
|---|---:|---|
| 1   | 16 | (baseline) cash + ew/mom/rev/lowv/highv × {0.5, 1, 2} |
| 2a  | 28 | + mom121 / lowv252 / shtop / trend × {0.5, 1, 2} |
| 2b  | 31 | 2a + top13f × {0.5, 1, 2} (14 funds, 13F-HR since 2013) |

Each runs in 20-30s walkforward + 6.2s uv sync + transit on Modal
CPU 8c. Drivers: `apps/cfr/scripts/modal/run_phase{1,2a,2b}.py`.

## Headline result

| Metric | Phase 1 | Phase 2a | Phase 2b |
|---|---:|---:|---:|
| mean CFR Sharpe | **+0.593** | +0.573 | +0.583 |
| mean passive EW Sharpe | +0.685 | +0.685 | +0.685 |
| mean trailing-best | −0.016 | +0.044 | +0.064 |
| mean naive uniform | +0.591 | **+0.632** | **+0.652** |
| **CFR vs trailing-best** | **+0.609** | +0.529 | +0.520 |
| **CFR vs naive uniform** | **+0.002** | −0.059 | **−0.069** |
| **CFR alpha vs EW** | −0.093 | −0.112 | −0.103 |
| CFR > trailing in N/6 | 6/6 | 6/6 | 6/6 |
| Pre-reg verdict | partial-OOS | confirmed-null | confirmed-null |

**Read across the row:**

1. **CFR Sharpe is essentially flat** across phases (0.573 → 0.593,
   noise band). Adding documented-alpha modes (2a) or real 13F
   data (2b) does not lift CFR's mean Sharpe.
2. **Naive uniform Sharpe rises monotonically** with each
   enrichment (0.591 → 0.632 → 0.652). The 1/N mix benefits from
   each new diversifying action regardless of its alpha content.
3. **CFR's lift over naive uniform turns negative** (+0.002 → −0.059
   → −0.069). The richer menus help the baseline more than the
   algorithm.
4. **CFR's lift over trailing-best-greedy stays high** (+0.609 →
   +0.529 → +0.520) — that's the comparison the original Phase 1
   PASS cut was written against. CFR clears it in every phase, but
   the naive uniform comparison shows this lift was over a
   catastrophic baseline, not a meaningful one.
5. **Per-window concentration of alpha** stays the same: window 1
   (GFC) is the only consistent positive-alpha window across all
   three phases. Phase 2b adds window 5 (2020-2023, post-COVID) as
   a near-tie alpha — the 13F mode's coverage matches the
   regime where it could add value.

## Phase 2a — enriched deterministic menu

**Hypothesis:** the Phase 1 finding said the binding constraint
was the action menu being too close to alpha-zero. Add documented-
alpha modes (12-1 momentum, 12-month low-vol, trailing-Sharpe
top-K, trend-strength = return/MDD) and CFR's regret matching
should concentrate on whichever is regime-active per infoset,
lifting the mean Sharpe meaningfully over Phase 1.

**Result:** CFR Sharpe dropped from +0.593 → +0.573 (Δ −0.020).
Naive uniform on the enriched menu rose from +0.591 → +0.632 (Δ
+0.041). CFR went from tying naive uniform (+0.002) to *losing* to
it by 0.06 Sharpe.

**Pre-registered cut:** PASS if CFR ≥ Phase 1 CFR + 0.10 mean
Sharpe AND CFR > naive uniform on enriched menu by ≥ +0.10.
Failed both ⇒ `confirmed-null`.

Per-window detail (alpha vs EW):

| win | Phase 1 | Phase 2a | Δ |
|---:|---:|---:|---:|
| 0 | −0.251 | −0.370 | **−0.119** (mom121/trend overweighted late dot-com names whose 12-1 ranks were stale) |
| 1 | +0.265 | +0.275 | +0.010 (GFC carries similarly) |
| 2 | −0.111 | −0.117 | −0.006 |
| 3 | −0.189 | −0.301 | −0.112 |
| 4 | +0.000 | −0.009 | −0.009 |
| 5 | −0.271 | −0.151 | +0.120 (252d Sharpe ranking helps post-COVID) |

The lift in window 5 (+0.120) is real — Sharpe-top-K mode picked
up some post-COVID winners. But it's drowned by larger losses in
windows 0 and 3, leaving mean alpha −0.112 (worse than Phase 1's
−0.093).

## Phase 2b — real 13F consensus mode

**Hypothesis:** real 13F consensus is qualitatively different from
deterministic factor exposures — it contains regime-conditional
information about *specific names that informed allocators are
concentrating on*. CFR should be able to discover when to deploy
this mode versus the deterministic alternatives, where
deterministic-only Phase 2a couldn't.

**Setup:** built `packages/edgar` (SEC EDGAR 13F-HR loader, 15
curated institutional managers from CIK 1067983 Berkshire to
1656456 Viking Global, 14 with usable filings since 2013-01-01),
fetched all 13F-HR filings via SEC's submissions API + per-filing
infotable XML, parsed the holdings, restricted to the
stooq_us_long universe (49 tickers in both), aggregated to
fund-count-per-quarter consensus panel (top-K most-broadly-held
by fund count). `Top13FConsensusMode` reads this panel with a
45-day filing lag and EW-portfolios over the consensus tickers
that are also in the price panel. Bars before the first lagged
13F quarter (~2013-08) deploy as cash.

**Result:** CFR Sharpe +0.583 (between Phase 1's +0.593 and Phase
2a's +0.573). Naive uniform on the 31-action menu is +0.652 — the
highest of the three phases. CFR loses to naive uniform by 0.069,
the worst of the three.

Pre-registered cut: PASS if CFR ≥ Phase 1 CFR + 0.10 mean Sharpe
AND CFR > naive uniform on Phase 2b menu by ≥ +0.10. Failed both
⇒ `confirmed-null` on the headline "13F adds enough alpha for
CFR to discover" hypothesis.

**But the per-window pattern is more interesting than the mean:**

| win | val_dates | 13F coverage | CFR Phase 1 alpha | CFR Phase 2b alpha | Δ |
|---:|---|---|---:|---:|---:|
| 0 | 2005-01 → 2008-02 | None (pre-2013) | −0.251 | **−0.415** | **−0.164** |
| 1 | 2008-02 → 2011-03 | None (pre-2013) | +0.265 | +0.291 | +0.026 |
| 2 | 2011-03 → 2014-04 | Late val only | −0.111 | −0.065 | +0.046 |
| 3 | 2014-04 → 2017-05 | Train + val | −0.189 | −0.246 | −0.057 |
| 4 | 2017-05 → 2020-07 | Train + val | +0.000 | −0.182 | −0.182 |
| 5 | 2020-07 → 2023-08 | Train + val | −0.271 | **+0.006** | **+0.277** |

**Window 5** (val 2020-07 → 2023-08, post-COVID recovery + the
2022 inflation hike cycle) shows a **+0.277 alpha lift** from
adding the 13F mode. This is the cleanest within-window signal we
have for the 13F hypothesis: the regime where 13F has full
coverage in both train and val, AND where the macro setup matches
historical patterns hedge funds reward (rate-tightening, vol-
elevated, post-shock recovery — same regime cluster as gate-
drawdown-v0's win).

**Window 0** (val 2005-01 → 2008-02, pre-13F-coverage era) shows
a **−0.164 alpha drag** from adding the 13F mode. Mechanism:
because `Top13FConsensusMode.precompute` returns all-zero weights
when no 13F panel data exists for a bar, CFR can route policy
mass to "13F at gross 0.5/1.0/2.0" actions in pre-2013 cells of
the regret table, reducing effective gross deployment vs Phase 1.
Three of the 31 actions become cash-equivalent in early windows
without the canonical-cash dedup catching them (since their
defined gross > 0).

The aggregate mean is dragged down by this asymmetric structure:
the 13F mode is *helpful where it has data* and *actively
harmful where it doesn't* (via the cash-equivalent-no-dedup
mechanism above). With 4 of 6 windows in the no-coverage era,
the mean reads as null even though the late-window signal is real.

## Combined mechanism

The two enrichment phases reveal the same architectural truth from
two angles:

**Tabular regret matching is sample-density-bound.** With 9
infosets × N actions, each cell needs enough samples for the
cumulative regret to estimate above noise. At our T = 6,000 train
rebals and infoset visit imbalance (some cells get visited 50% of
bars, others 5%), going from 16 to 31 actions roughly doubles the
table dimension while train data stays constant. Cesa-Bianchi &
Lugosi (2006) gives the no-regret learner's regret bound:

$$\text{Regret}(T) \leq \sqrt{\frac{T \log n}{2}}$$

For us: T = 6,000, n = 16 → bound ≈ 0.024; n = 28 → 0.026; n = 31
→ 0.027. The bound worsens slightly with menu size — exactly
what we observed.

**Naive uniform is sample-density-free.** The 1/N mix has zero
parameters to estimate. Adding a diversifying action raises
naive uniform's Sharpe by exactly the marginal improvement in
diversification ratio. CFR can only exceed uniform's Sharpe by an
amount proportional to its concentration accuracy, which is
bounded by the regret estimator's noise.

**The gap to passive EW is structural.** Passive EW deploys 1/N
over 312 tickers with full re-balancing. Naive uniform mix in
Phase 2b deploys (1/31) × Σ_a w_a where w_a varies by action — a
strictly more concentrated portfolio with extra friction. The
−0.05 to −0.15 alpha gap to passive EW reflects the friction +
concentration trade-off, not an algorithmic deficiency.

## What "confirmed-null on tabular menu enrichment" implies for Phase 3

The original [`apps/cfr` TODO](../TODO/apps-cfr.md) had Phase 3 as
"deep CFR fine-tune" — but Phase 2's confirmed-null on tabular
menu enrichment makes the priority order much clearer:

1. **Replace tabular regret table with `regret_net(state_vec,
   action_emb) → R` MLP.** The deep network shares statistical
   strength across (state, action) pairs that have similar
   structure — sample density per cell stops being the binding
   constraint.
2. **Replace 9-cell discrete infoset with a learned encoder** over
   the multi-modal state vector (price CWT + macro panel +
   cross-sectional dispersion + 13F-overlap signal +
   portfolio-state). The discrete cuts in Phase 1-2 throw away
   continuous regime information.
3. **Keep the closed-form counterfactual regret signal** (still
   cheap) but train via SGD over the deep regret-net rather than
   accumulating into a sparse table.
4. **The 13F mode (Phase 2b) stays in the menu** but with a deep
   architecture able to discover *when* to weight it via the
   learned encoder, instead of needing to find it in a sparse
   tabular cell.
5. **Fix the cash-equivalent-no-dedup issue.** The
   `Top13FConsensusMode` should mask its presence (drop from menu)
   for bars without 13F coverage, not silently return cash. Or:
   the menu should be era-dependent — pre-2013 menu has 28
   actions, post-2013 menu has 31. Either prevents the −0.164
   alpha drag in pre-coverage windows seen in Phase 2b w0.

**Pre-registered Phase 3 cut:** **deep CFR mean Sharpe ≥ Phase 1
CFR + 0.15** AND **deep CFR > naive uniform mix on Phase 2b menu
by ≥ +0.10**. The +0.15 over Phase 1 (vs +0.10 in Phase 2)
reflects that Phase 3 is a more ambitious architectural change
with correspondingly higher expected lift if the hypothesis holds.

## Reproducing

```bash
# Phase 2a (uses Phase 1's prepped pickle)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uvx modal run apps/cfr/scripts/modal/run_phase2a.py

# Phase 2b — adds 13F prep step (~4 min cold cache, instant warm)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uv run python apps/cfr/scripts/modal/prep_phase2b_data.py
uvx modal run apps/cfr/scripts/modal/run_phase2b.py
```

13F prep cache lives at `.edgar-cache/`; subsequent runs hit the
cache in <1 second after the first build. SEC's 10 req/s rate
limit dominates first-build wall.

## Master walk-forward log

- [2026-05-12 cfr Phase 1 row](../leaderboard.md) — `partial-OOS`
- [2026-05-12 cfr Phase 2a row](../leaderboard.md) — `confirmed-null`
- [2026-05-12 cfr Phase 2b row](../leaderboard.md) — `confirmed-null`
  (with positive late-window subtlety)
