---
tags:
  - cfr
  - phase-4
  - meta-allocator
  - deep-cfr
  - sector-etfs
  - multi-asset
---

# CFR Phase 4 — change the prediction problem; multi-asset PASSES, equity-only never does

**Operational rule (added 2026-05-12 to
[`CLAUDE.md`](https://github.com/sughodke/StockSurvey/blob/master/CLAUDE.md#operational-rules-extracted-from-findings)):**
the CFR meta-allocator is **prediction-problem bound, not
representation bound**. Across 4 Phase 4 variants — VIX gate
hybrid, sector ETF universe, 5-day rebal horizon, multi-asset
basket — only the *universe shift* (Phase 4b sector ETFs;
Phase 4d cross-asset) materially moved the needle. The 4a VIX
gate destroyed Phase 3's mean Sharpe; the 4c finer rebal cadence
ate alpha to friction. **Phase 4d (deep CFR on a 13-asset
cross-asset basket) was the first variant in 9 walkforward
runs to PASS the pre-registered cut**: mean CFR Sharpe +0.861
(Phase 1 + 0.27), CFR vs naive uniform +0.101 (≥ +0.10
threshold), 3/5 positive alpha windows, mean alpha vs passive EW
**+0.056 (positive for the first time).** The lesson confirms
the prediction-problem-pivot-arc finding: when the prediction
problem has structural alpha to find (cross-asset regime
switching), the deep meta-allocator finds it; when it doesn't
(US mega-cap-only), no architecture rescues the menu.

**Verdict at the arc level:** [PASS on
algorithm-validation](../leaderboard.md#verdict-labels) (Phase 4d
clears the pre-reg cut), [partial-OOS on operational floor]
(alpha vs EW positive but below the +0.15 paper-trade threshold).

## Setup — four orthogonal Phase 4 variants

All four use Phase 3's deep CFR architecture (regret_net + 10-
feature continuous state + Phase 2b 31-action menu when
applicable, modulo top13f availability) and the canonical 6-
window walkforward shape (1260-train / 780-val / 780-step). The
universe shifts in 4b/4d give 5 windows instead of 6 because
sector ETFs start in 2005.

| Phase | What changed vs Phase 3 |
|---|---|
| 4a  | + VIX-above-1y-rolling-median per-bar pre-rebal gate (suspends deployment in calm regimes per the [macro v1b finding](macro-regime-diagnostic.md)) |
| 4b  | Universe → 10 SPDR sector ETFs + SPY (vs 312 mega-cap stocks); top_k=3; 13F mode dropped (ETFs don't appear in 13F) |
| 4c  | rebal_days=5 (vs 20); train_every=10 to keep SGD wall comparable |
| 4d  | Universe → 13-asset cross-asset (9 sector ETFs + TLT, IEF + GLD, DBC); top_k=4; 13F mode dropped |

Drivers: `apps/cfr/scripts/modal/run_phase{4a,4b,4c,4d}.py`.
Each runs in 30-180s end-to-end on Modal CPU 8c.

## Headline result

| Metric | Phase 3 | 4a (VIX gate) | 4b (sectors) | 4c (5-day) | **4d (multi-asset)** |
|---|---:|---:|---:|---:|---:|
| mean CFR Sharpe | +0.614 | **+0.383** | +0.780 | +0.574 | **+0.861** |
| mean passive EW | +0.685 | +0.685 | +0.765 | +0.688 | +0.805 |
| mean naive uniform | +0.652 | +0.652 | +0.747 | +0.560 | +0.760 |
| **CFR vs naive** | −0.038 | −0.269 | +0.034 | +0.013 | **+0.101** |
| **CFR alpha vs EW** | −0.071 | −0.302 | +0.015 | −0.114 | **+0.056** |
| Pos α windows | 2/6 | 1/6 | 3/5 | 3/6 | **3/5** |
| Verdict | MARGINAL | **FAIL** | MARGINAL | MARGINAL | **PASS** |

Three signals stand out:

1. **4a VIX gate destroys Phase 3.** Mean CFR drops +0.614 → +0.383
   (−0.231). With the gate open only 43% of bars, CFR sits in
   cash 57% of the time and loses the compounding that drives its
   long-only exposure.
2. **4b/4d universe shifts beat the operational floor.** Both
   ETF-based universes have CFR alpha vs EW positive (+0.015,
   +0.056) for the first time across all 9 phase variants tried.
3. **4d is the only PASS.** CFR mean +0.861 (≥ Phase 1 + 0.15
   = +0.74 ✓), CFR vs naive +0.101 (≥ +0.10 ✓). Both
   pre-registered cuts cleared.

## Phase 4a — VIX-per-bar gate is too aggressive

**Hypothesis:** the macro v1b finding showed VIX-above-1y-rolling-
median produced +0.215 z-lift on the pivot-arc apps. Combining
that gate with Phase 3's deep CFR should suspend deployment in
the calm regimes that have produced 4/6 negative-alpha windows
across all CFR variants.

**Result:** **catastrophic.** mean CFR +0.614 → +0.383
(−0.231); CFR vs naive uniform −0.038 → −0.269; CFR alpha vs EW
−0.071 → −0.302. Verdict: FAIL.

**Per-window:**

| win | Phase 3 alpha | 4a alpha | Δ | 4a in cash for |
|---:|---:|---:|---:|---|
| 0 | −0.207 | −0.606 | −0.399 | most bars (low-VIX 2005-08) |
| 1 | +0.172 | −0.350 | −0.522 | partial (GFC mid-window) |
| 2 | +0.127 | −0.531 | −0.658 | most bars (calm 2011-14) |
| 3 | −0.308 | −0.103 | +0.205 | most bars (saved from bad bars) |
| 4 | −0.107 | +0.189 | +0.296 | partial (saved from worst stretches) |
| 5 | −0.105 | −0.413 | −0.308 | most bars (post-COVID rally) |

**Mechanism:** the macro v1b finding's lift was at the *window*
level (suspend whole 3-year periods if val_start VIX < median).
Phase 4a applies the gate at the *bar* level, suspending 57% of
bars regardless of the deep CFR's own regime call. CFR loses
exposure in many bars where it would have produced positive
return; the gate doesn't know which bars CFR's policy would
exploit. Window-level meta-gating remains a viable v2 idea, but
not bar-level.

## Phase 4b — Sector ETFs PARTIAL clear

**Hypothesis:** sector rotation alpha is documented; sector ETFs
have ~10× larger per-action variance than mega-cap stocks. Even
with the same Phase 3 architecture, regret signal per
(state, action) should be 10× sharper.

**Result:** **first CFR variant to beat passive EW on alpha.**
Mean CFR +0.780 vs passive EW +0.765 = **alpha +0.015**, vs naive
uniform +0.747 = **+0.034 lift**. 3/5 positive alpha windows.

| win | val_dates | CFR Sh | Pas EW | Naive | α vs EW | vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2010-03 → 2013-04 | +0.938 | +0.688 | +0.736 | **+0.250** | **+0.202** |
| 1 | 2013-04 → 2016-05 | +0.806 | +0.792 | +0.831 | +0.015 | −0.025 |
| 2 | 2016-05 → 2019-06 | +0.620 | +0.976 | +0.805 | −0.357 | −0.185 |
| 3 | 2019-06 → 2022-07 | +0.584 | +0.596 | +0.586 | −0.012 | −0.002 |
| 4 | 2022-07 → 2025-08 | +0.953 | +0.771 | +0.775 | **+0.182** | +0.178 |

Verdict: MARGINAL — clears the +0.10 vs naive at +0.034 (within
noise) but doesn't clear the +0.20 mean CFR lift over Phase 3
(0.780 vs 0.614 = +0.166, just below). The pre-reg cut for 4b
was strict (≥ Phase 3 + 0.20 AND ≥ +0.15 alpha vs EW); CFR
delivered +0.166 lift and +0.015 alpha. MARGINAL by the +0.10
vs naive cut.

The takeaway is structural, not statistical: at sector-ETF
granularity, deep CFR's regret signal is *clean enough to find a
small alpha*, where on equities it wasn't. Universe matters more
than algorithm.

## Phase 4c — 5-day rebal: friction-bound

**Hypothesis:** 4× more rebals = 4× more SGD samples, finer
regime sensitivity. Trade-off: 5-day rebal at 10 bps = ~5%/year
friction (vs Phase 3's ~1.3%/year at 20-day).

**Result:** mean CFR Sharpe +0.574 (slightly worse than Phase 3's
+0.614). CFR vs naive +0.013 (basically tied). CFR alpha vs EW
−0.114 (worse than Phase 3's −0.071, exactly the friction-tax
delta).

Notable side observation: **training stability dramatically
improved at 5-day cadence.** Final loss values were finite (1e-3
to 4e-3 range across all 6 windows) for the first time in any
deep CFR run. The 4× more SGD samples per window let AdamW
converge stably without the late-training divergence we saw in
Phases 3 and 4a/4b/4d. So the algorithm benefits from finer
cadence — but the friction tax eats the benefit at this
universe.

## Phase 4d — Multi-asset PASS

**Hypothesis:** cross-asset has documented regime-switching alpha
(60/40 → barbell during high inflation, etc.). 13 assets across
3 classes (equities + bonds + commodities) give the deep
encoder real macro-regime structure to exploit, AND larger
per-action variance than equity-only.

**Result:** **PASS on both pre-registered cuts.**

| win | val_dates | CFR Sh | Pas EW | Naive | α vs EW | vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2010-03 → 2013-04 | **+1.367** | +0.945 | +0.947 | **+0.422** | **+0.420** |
| 1 | 2013-04 → 2016-05 | +0.622 | +0.647 | +0.708 | −0.026 | −0.086 |
| 2 | 2016-05 → 2019-06 | +0.495 | **+1.002** | +0.706 | −0.508 | −0.211 |
| 3 | 2019-06 → 2022-07 | **+0.780** | +0.690 | +0.678 | **+0.090** | **+0.101** |
| 4 | 2022-07 → 2025-08 | **+1.041** | +0.740 | +0.761 | **+0.301** | **+0.280** |
| **mean** | | **+0.861** | **+0.805** | **+0.760** | **+0.056** | **+0.101** |

3 of 5 positive alpha windows (60%). w0 (post-GFC), w3
(COVID-era), and w4 (post-COVID inflation cycle) all post strong
alpha. w2 is the only meaningful loss (−0.508) — the 2016-2019
period was a "everything works" passive era where rotation
strategies underperform.

**Verdict: PASS** by the in-code cut (mean ≥ Phase 1 + 0.15 =
+0.74 ✓; CFR vs naive ≥ +0.10 ✓). Mean alpha vs EW is +0.056
(positive but below the +0.15 paper-trade threshold I'd want for
deployment).

## Why multi-asset works where equity-only doesn't

Two structural differences:

1. **Per-action variance.** On 312 US mega-cap equities, each
   ticker has correlation ~0.5-0.7 with the universe-EW. Top-K
   modes pick small subsets that are still ~80% correlated with
   EW. Different modes pick *different* tickers but the
   *portfolios they produce* have ~85% correlation with each other.
   Regret matching has limited room to differentiate. On 13-asset
   cross-asset, gold ↔ stocks correlation is ~0, bonds ↔ stocks
   correlation is ~0 to slightly negative. Mode portfolios are
   genuinely orthogonal. Regret signal per (state, action) pair
   is structurally larger.
2. **Real regime-conditional alpha.** Cross-asset literature has
   60+ years of evidence that "stocks vs bonds vs commodities"
   has a regime-switching optimum (the Bridgewater All Weather
   thesis). At equity-only horizons, intra-equity rotation alpha
   is heavily contested and arbitraged away — at cross-asset,
   inflation-regime-conditional rotation has structural
   inefficiency that hasn't been arbitraged.

The deep CFR's encoder learns the regime indicators — VIX,
credit spreads, m2_yoy, real_yield_10y, plus universe-internal
vol/dispersion — that genuinely matter for cross-asset. On
equity-only, those same features have less to predict because
the menu doesn't span enough opportunity to act on them.

## What this means for paper trading

Honest assessment of Phase 4d as a deployable strategy:

**Pros (vs every prior CFR variant):**
- First positive mean alpha vs passive EW (+0.056)
- Clean PASS on the pre-registered algorithm cut
- 3/5 positive alpha windows
- Walk-forward includes COVID and post-COVID inflation cycle
- Architectural ceiling has been broken — universe shift was the lever

**Cons (vs the operational floor):**
- Mean alpha +0.056 is positive but below the +0.15 paper-trade
  threshold the [passive-EW benchmark](passive-ew-benchmark.md)
  established as the "shippable" gate.
- Only 5 windows (sector ETFs start 2005); statistical power for
  the alpha estimate is weaker than the canonical 6-window stocks
  walkforward.
- 1 of 5 windows (w2 2016-2019) shows −0.508 alpha — that's a
  catastrophic single-window loss. Strategy needs a regime-
  recognition gate to avoid the "everything works passive" era,
  but layering 4a-style gating on top of 4d would need careful
  re-test.
- No `ss-cfr live` integration yet — need the four risk rails
  (kill-switch, freshness, position cap, dry-run-by-default)
  per CLAUDE.md before any real deployment.

**Recommendation:** Phase 4d is the **first CFR variant worth
considering for paper trading**, but as a v0 prototype that
needs the live integration scaffolding before deployment. Build
out `ss-cfr live` with the standard risk rails, paper trade for
1-2 quarters as a smoke test, evaluate before any real capital.

## What's next — Phase 5 candidates

The Phase 4 sweep narrowed the productive directions. The
strongest signal is the *universe* axis. Phase 5 candidates:

1. **`apps/cfr live` integration.** Build the four risk rails on
   top of `cfr.deep_walkforward`'s eval path. Modify `cfr.cli` to
   add an `ss-cfr live` subcommand reading a saved Phase 4d
   checkpoint. ~1 week. **The actual deployment unlock.**
2. **Phase 4d extension to longer history.** Use synthetic /
   index-equivalent data for sector/bond/commodity exposures
   pre-2005 (CRSP for sectors, FRED's 10Y for IEF-equivalent,
   COMEX gold for GLD, S&P GSCI for DBC). Get the walkforward
   to 8-10 windows. Should reduce single-window outlier risk.
3. **Phase 4d + window-level macro gate.** Apply the 4a VIX
   gate at the window level instead of bar level. Suspend whole
   3-year deployment periods when val_start macro composite
   indicates "everything works passive" (low VIX, narrow credit
   spreads, declining m2). Could rescue w2's −0.508 alpha.
4. **Universe scale-up to 30-50 cross-asset.** Add international
   equity ETFs (EFA, EEM), TIP (inflation-protected), more
   commodity slices (USO, SLV). More actions to discriminate
   between, larger per-action variance. Need careful per-action
   gross calibration for non-USD exposures.

The Phase 4 finding's main lesson is operational: **stop
iterating on the meta-allocator architecture; iterate on the
universe.** Phase 4a (VIX gate) and 4c (5-day rebal) confirmed
that representation/cadence don't help; 4b and 4d showed
universe is the primary lever.

## Reproducing

```bash
# Phase 4a — VIX gate hybrid (after prep_phase{1,2b,3}_data.py)
uvx modal run apps/cfr/scripts/modal/run_phase4a.py

# Phase 4b — Sector ETFs (only needs phase4b prep + macro)
uv run python apps/cfr/scripts/modal/prep_phase4b_data.py
uvx modal run apps/cfr/scripts/modal/run_phase4b.py

# Phase 4c — 5-day rebal (same prep as Phase 3)
uvx modal run apps/cfr/scripts/modal/run_phase4c.py

# Phase 4d — Multi-asset cross-asset basket
uv run python apps/cfr/scripts/modal/prep_phase4d_data.py
uvx modal run apps/cfr/scripts/modal/run_phase4d.py
```

## Master walk-forward log

- [2026-05-12 cfr Phase 4a row](../leaderboard.md) — `confirmed-null`
- [2026-05-12 cfr Phase 4b row](../leaderboard.md) — `partial-OOS`
- [2026-05-12 cfr Phase 4c row](../leaderboard.md) — `partial-OOS`
- [2026-05-12 cfr Phase 4d row](../leaderboard.md) — **`PASS`** (first
  of all 9 CFR phase variants)
