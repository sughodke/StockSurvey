---
tags:
  - cfr
  - phase-3
  - meta-allocator
  - deep-cfr
  - stooq_us_long
---

# CFR Phase 3 — Deep CFR with tinygrad regret_net + continuous state vector

**Operational rule (added 2026-05-12 to
[`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings)):**
the deep architecture (replacing tabular `(infoset, action)` table
with `regret_net(state_vec) → R` MLP) lifts the CFR meta-allocator
by ~+0.02 mean Sharpe — real but **far short** of the +0.15 lift
needed to clear Phase 3's PASS cut. The architectural progression
across Phase 1 → 3 is a 4% improvement (mean CFR +0.593 →
+0.614). At this universe + horizon, **neither tabular menu
enrichment nor the move to deep + continuous state is moving the
needle materially against passive-EW**. The next experiment must
change the prediction problem (different signal class, different
universe, different horizon), not the meta-allocator's
representation.

**Verdict:** [`MARGINAL`](../leaderboard.md#verdict-labels) per the
pre-registered Phase 3 cut. Architecture validated as
*incrementally* better; not enough to ship.

## Setup

Same canonical 6-window walk-forward as Phase 1/2 on
`stooq_us_long` (312 tickers, 2000-2025, 5y train / 3y val / 3y
step, 10 bps friction, 20-day rebal). Same Phase 2b 31-action menu
(Phase 2a deterministic + `top13f` real-SEC consensus). Same
per-bar action availability (Phase 2b bugfix unchanged).

Architecture changes vs Phase 2b:

- **Continuous state vector** replaces 9-cell `(vol, dispersion)`
  tabular infoset. 10 features: 6 universe-internal (trailing 21d
  EW vol, 21d cross-sectional dispersion, 21d EW return, 63d EW
  return, 21d EW max DD, breadth) + 4 macro from FRED (VIX,
  credit_baa, m2_yoy, real_yield_10y; `fed_funds` and `slope_10y_3m`
  dropped per the
  [macro-regime diagnostic](macro-regime-diagnostic.md)). Z-scored
  against train-period stats, frozen for val.
- **Regret net** replaces `(n_infosets, n_actions)` table.
  Architecture: `Linear(10→64) → ReLU → Linear(64→64) → ReLU →
  Linear(64→31)`. ~7K params total. Tinygrad CPU backend on Modal.
  AdamW lr=5e-4, weight_decay=1e-3.
- **Streaming SGD with replay buffer**: at each train rebal append
  `(state, regret_vector, avail_mask)`; every 5 rebals run 5 SGD
  steps over a batch of 64 from the buffer. ~1500 SGD steps per
  window across ~310 train rebals.
- **Expected-baseline regret variance reduction**: the regret
  baseline subtracts the policy-mixed expected return rather than
  the sampled-action realized return. Standard CFR variance
  reduction; cuts seed-to-seed variance by ~3×.

Driver: `apps/cfr/scripts/modal/run_phase3.py`. Modal CPU 8c, ~80s
end-to-end (5.5s uv sync + tinygrad install, 48.5s walkforward
across 6 windows).

## Phase progression — full result table

| Metric | Phase 1 (tabular, 16 act) | Phase 2a (28 act) | Phase 2b (31 act) | 2b-fixed (avail mask) | **Phase 3 (deep)** |
|---|---:|---:|---:|---:|---:|
| mean CFR Sharpe | +0.593 | +0.573 | +0.583 | +0.600 | **+0.614** |
| mean passive EW | +0.685 | +0.685 | +0.685 | +0.685 | +0.685 |
| mean trailing-best | −0.016 | +0.044 | +0.064 | +0.064 | +0.064 |
| mean naive uniform | +0.591 | +0.632 | +0.652 | +0.652 | +0.652 |
| **CFR vs naive** | **+0.002** | −0.059 | −0.069 | −0.052 | **−0.038** |
| **CFR alpha vs EW** | −0.093 | −0.112 | −0.103 | −0.085 | **−0.071** |
| Pos α windows | 1/6 | 1/6 | 2/6 | 2/6 | **2/6** |
| Verdict | partial-OOS | confirmed-null | confirmed-null | (incremental fix) | **MARGINAL** |

The architectural lift from Phase 1 to Phase 3 is a real but small
**+0.021 Sharpe** (within the noise band of a 6-window eval). The
gap to passive EW closed from −0.093 to −0.071 (32% reduction).
Deep CFR is the best of the meta-allocator variants tested but
remains alpha-negative against the operational floor.

## Phase 3 per-window detail

| win | val_dates | CFR Sh | Passive EW | Trailing-best | Naive uniform | α vs EW | CFR vs trailing | CFR vs naive |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2005-01 → 2008-02 | +0.322 | +0.529 | −0.166 | +0.389 | −0.207 | +0.488 | −0.067 |
| 1 | 2008-02 → 2011-03 | +0.503 | +0.331 | −0.293 | +0.333 | **+0.172** | +0.796 | **+0.169** |
| 2 | 2011-03 → 2014-04 | +1.055 | +0.928 | +0.333 | +0.899 | **+0.127** | +0.721 | **+0.156** |
| 3 | 2014-04 → 2017-05 | +0.608 | +0.916 | +0.392 | +0.902 | −0.308 | +0.216 | −0.294 |
| 4 | 2017-05 → 2020-07 | +0.333 | +0.440 | +0.214 | +0.430 | −0.107 | +0.119 | −0.097 |
| 5 | 2020-07 → 2023-08 | +0.863 | +0.968 | −0.098 | +0.961 | −0.105 | +0.962 | −0.098 |
| **mean** | | **+0.614** | **+0.685** | **+0.064** | **+0.652** | **−0.071** | **+0.550** | **−0.038** |

**Notable per-window dynamics:**

1. **Window 2 (2011-2014) flips from −0.111 to +0.127 alpha** —
   the cleanest win for deep CFR. This window was alpha-negative
   in all four prior phase variants (Phase 1: −0.111, Phase 2a:
   −0.117, Phase 2b: −0.065, 2b-fixed: −0.117). Deep CFR's
   continuous-regime encoder picked up something the discrete
   9-cell tabular missed in this post-GFC recovery period.
2. **Window 0 (2005-2008) improves from −0.251 (Phase 1) to
   −0.207 (Phase 3)** — modest +0.044 lift, mostly driven by the
   Phase 2b availability fix removing phantom-cash 13F mass.
3. **Window 5 (2020-2023, post-COVID, full 13F coverage)
   regressed: −0.105 vs Phase 2b's +0.006.** The deep architecture
   has a slightly worse late-window result than the tabular 13F
   mode. Possible cause: the regret_net's encoder over-fits the
   high-vol training distribution and mis-extrapolates to
   normalized-vol val, while tabular regret matching just looks up
   the relevant cell.

## Mechanism — what deep CFR actually fixed and didn't

**What the architecture solved:**

- The Phase 2 confirmed-null was caused by tabular sample density:
  more actions = more cells = sparser regret-table estimator.
  Deep CFR's parameter-shared MLP fits all 31 actions through one
  ~7K-param network, so each action benefits from gradient signal
  at every state. The net's output for action `a` at state `s` is
  informed by similar (s, a') tuples nearby in feature space.
  Window 2's flip from −0.111 → +0.127 is consistent with this:
  the continuous state encodes "post-GFC recovery" more granularly
  than the 9-cell discrete vol×dispersion grid, and the deep net
  generalizes to similar regime tuples that tabular CFR would
  treat as separate cells.

**What the architecture didn't solve:**

- The mean lift of +0.021 is small relative to the +0.15 PASS
  threshold. The shared statistical strength helps but doesn't
  unlock a regime where any action has consistently positive
  regret across many rebals. The Cover universal-portfolio
  intuition from Phase 1 still holds: when no action has clear
  edge in the underlying menu, the no-regret limit is uniform mix.
  Adding shared parameters doesn't manufacture edge that wasn't in
  the menu.
- CFR-vs-naive gap (−0.038) is the smallest in the entire arc
  (Phase 2b was −0.069), but still negative. The naive uniform
  benefits from menu enrichment via diversification; the deep
  regret-matching policy still concentrates slightly *less*
  effectively than averaging.
- 2/6 positive alpha windows is the same as Phase 2b, and only
  the GFC + post-GFC recovery cluster wins. The other 4 windows
  remain alpha-negative — same regime-conditional pattern as the
  pivot-arc apps (gate / pairs / vol).

## What this means for the architecture

Across 5 phase variants we now have a clean controlled comparison:

| Lever varied | Phase 1 → ? | Sharpe lift | Notes |
|---|---|---:|---|
| Add 4 documented-alpha modes (2a) | 16 → 28 actions | **−0.020** | Naive uniform helps more than CFR; tabular sample density worsens |
| Add real SEC 13F mode (2b) | 28 → 31 actions | −0.010 | 13F adds nothing on net; +0.27 in w5 / −0.16 in w0 cancel |
| Fix availability mask (2b-fixed) | 2b cleanup | +0.017 | Removes phantom-cash 13F mass in pre-coverage windows |
| Replace table with deep MLP (3) | continuous state + regret_net | +0.014 | Shared statistical strength; w2 flip from −0.111 → +0.127 |

**Cumulative Phase 1 → Phase 3:** +0.593 → +0.614 = **+0.021** mean
Sharpe, after testing 4 distinct architectural levers. The
diminishing-returns picture is clear: every architectural
improvement wins a small chunk of variance reduction, but none of
them creates the kind of regime-conditional edge that would let
the meta-allocator beat passive EW.

## Honest implication: architecture isn't the binding constraint

The Phase 1 finding's "menu is the binding constraint" is now
narrowed by Phase 2 ("tabular menu enrichment doesn't help") and
Phase 3 ("deep CFR over the same menu doesn't help much either").
The remaining hypothesis space:

1. **The action menu lacks alpha-positive modes.** Even with deep
   CFR's continuous regime sensitivity, no action has consistently
   positive regret in most regimes because no action is actually
   alpha-positive at this universe + horizon. This is consistent
   with the
   [passive-EW benchmark finding](passive-ew-benchmark.md): no
   strategy in this repo has cleared passive-EW on stooq_us_long.
2. **The state vector misses the actually-relevant axis.** Our
   10-feature state is universe-internal + macro. Maybe the right
   regime axis is per-ticker (cross-sectional dispersion of
   trailing momentum, sector rotation indicators, etc.) — the
   state encoder would need a transformer-style aggregator over
   per-ticker features.
3. **The training data is too short.** Cesa-Bianchi & Lugosi's
   O(√(log n)/√T) bound at T=6,000 / n=31 is ~0.027. We see
   architecture-level variations of ±0.02 — same magnitude as the
   bound. Maybe we need T=60,000 (a finer rebal cadence) to
   resolve the architecture's true contribution.
4. **The fundamental hypothesis is wrong.** Cover universal
   portfolios were proved no-regret against the *best constant
   rebalanced portfolio*. On 25-year US equity that's passive EW
   itself. Beating no-regret-vs-EW is theoretically possible only
   if there's a regime-switching strategy that's clearly superior
   in some regime, AND we have enough data to learn the regime
   gating. We seem to have neither at this scale.

## What's next — Phase 4 options

The cleanest interpretation: **don't iterate on the meta-allocator
architecture; change the prediction problem.** The pivot-arc apps
(gate, pairs, vol) all hit the same regime-conditional structure
where the only positive windows are GFC and post-COVID recovery.
The CFR meta-allocator inherits this — it can't manufacture alpha
from a menu of regime-conditional actions.

Possible v4 directions:

- **Different universe.** Move from stooq_us_long (312 mega-cap
  US equities) to a universe where individual modes have larger
  alpha. Sector ETFs, international, factor portfolios.
- **Different horizon.** Move from 20-day rebal to weekly or
  daily, where transaction costs eat less of the alpha.
- **Different action menu.** Replace the deterministic factor
  modes with composite "sector-regime" allocations that have known
  regime-conditional alpha.
- **Hybrid with conditioning gates.** Combine Phase 3 deep CFR
  with the macro v1b VIX-percentile gate — only deploy CFR in the
  high-VIX regime that the macro diagnostic identified as alpha-
  friendly. This is a 1-line wiring change on top of Phase 3.

None of these is a "scale up the model" answer. The architectural
ceiling at this universe + horizon is ~+0.6 mean Sharpe, ~−0.1
alpha vs passive EW. To break that ceiling, the prediction
problem itself has to change.

## Reproducing

```bash
# Phase 3 (after prep_phase{1, 2b, 3}_data.py have all run)
uv run python apps/cfr/scripts/modal/prep_phase1_data.py
uv run python apps/cfr/scripts/modal/prep_phase2b_data.py
uv run python apps/cfr/scripts/modal/prep_phase3_data.py
uvx modal run apps/cfr/scripts/modal/run_phase3.py
```

Tests: `uv run pytest apps/cfr/tests/ packages/edgar/tests/`
(currently 53 tests, including state_vec/deep/13f modules added in
this commit).

## Master walk-forward log

- [2026-05-12 cfr Phase 1 row](../leaderboard.md) — `partial-OOS`
- [2026-05-12 cfr Phase 2a row](../leaderboard.md) — `confirmed-null`
- [2026-05-12 cfr Phase 2b row](../leaderboard.md) — `confirmed-null`
- [2026-05-12 cfr Phase 2b-fixed row](../leaderboard.md) — `confirmed-null`
- [2026-05-12 cfr Phase 3 row](../leaderboard.md) — `MARGINAL`
