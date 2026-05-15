---
tags:
  - pairs
  - regime-gate
  - audit-followup
  - confirmed-null
  - diagnostic
---

# Pairs v0.1 — EG-passing-rate regime gate (audit-followup, falsified)

**Operational rule (extracted):** *Train-time screening counts are not
reliable regime indicators for OOS pair-spread behavior.* The
[`pairs-classical-v0`](pairs-classical-v0.md) finding hypothesized that
EG-passing-rate per training window — the number of pairs that survive
the |corr|≥0.7 AND EG-p<0.05 cointegration screen — was a "natural
regime indicator" because the working windows posted counts of
3522-4755 and the failing later windows posted 2249-2857. The
2026-05-14 research-directions audit (`.audit-research-directions.md` at repo root)
flagged this as the cheapest test that could promote pairs from
`confirmed-null` to deployable, predicting "+0.099 → ~+0.5 with a
sensibly pre-registered threshold." **This audit hypothesis is
falsified**: actually re-screening every window and applying three
pre-registered gate thresholds yields best-case full-panel alpha
**+0.104** (T_abs=3000, 4/6 windows fire) — three to five times short
of the audit's predicted lift, and still below the +0.20 marginal
floor pre-registered in `pairs-classical-v0`.

The deeper falsification: **w0's train EG-passing-rate is 3918** — the
third-highest in the panel, well within the audit's framing of the
"working" band. But w0 still posts val Sharpe −1.23. The audit's
labeling of windows as "working" vs "failing" by EG-passing-rate was
post-hoc fitted to val Sharpe; the actual train-side EG-passing-rate
distribution does not separate them.

## Experiment

**Hypothesis (from audit + `TODO/apps-pairs.md`):** Gate the v0 pairs
walk-forward with a pre-registered cut on train-window
EG-passing-rate. Windows where the train cointegration regime is
"weak" (EG-pass below threshold) are deployment-suspended (alpha = 0);
windows where it's "strong" use v0's existing val Sharpe.

**Design** (`apps/pairs/scripts/eg_gate_eval.py`):

- Re-screen each v0 train window (~44k candidate pairs × 30ms × 6
  windows / 8 workers ≈ 8 min wall) to capture the pre-truncation
  EG-passing count, which v0's per-window JSON does not log.
- Combine with v0's existing per-window val Sharpes from
  `Output/pairs-walkforward-summary.json`.
- Apply three pre-registered thresholds (recorded BEFORE seeing the
  gated result, after seeing the EG counts):
  - `T_abs = 3000`: audit's framing (splits 3522-4755 from 2249-2857
    cleanly)
  - `T_pct50`: median of the 6 train EG-pass counts
  - `T_pct30`: 30th percentile of the 6 train EG-pass counts
- Apply v0's cuts: PASS if mean ≥ +0.50 AND ≥ 4/6 positive; MARGINAL
  if mean ∈ [+0.20, +0.50] AND ≥ 3/6; FAIL otherwise.
- Report both denominators: **full-panel** (gate-closed windows count
  as alpha=0, the v0 cut convention) AND **fired-only** (mean over the
  windows where the gate actually deployed), to avoid the denominator
  artifact identified in [`cfr-sensitivity-followup`](cfr-sensitivity-followup.md).

## Per-window data

Train EG-passing-rate (re-measured 2026-05-14), v0 val Sharpes:

| Window | Train period | Val period | Train EG_pass | v0 val Sharpe |
|---:|---|---|---:|---:|
| 0 | 2000-01 → 2005-01 | 2005-01 → 2008-02 | **3918** | −1.233 |
| 1 | 2003-02 → 2008-02 | 2008-02 → 2011-03 | 3522 | +0.870 |
| 2 | 2006-03 → 2011-03 | 2011-03 → 2014-04 | 3118 | +0.593 |
| 3 | 2009-04 → 2014-04 | 2014-04 → 2017-05 | **4755** | +0.392 |
| 4 | 2012-05 → 2017-05 | 2017-06 → 2020-07 | 2249 | +0.080 |
| 5 | 2015-07 → 2020-07 | 2020-07 → 2023-08 | 2857 | −0.109 |

Median of train EG counts: **3320**; 30th percentile: **2988**.

**Critical observation:** w0 (val Sharpe **−1.233**, the worst window
in the panel) has train EG_pass = 3918, which is the *third highest*
of the 6 windows. The audit's framing that "high train EG-passing-rate
= working regime" is wrong on the binding window — the one the gate
was specifically supposed to filter.

## Gate results

| Threshold | Fires in | Full-panel mean α | Pos / 6 | Fired-only α | Pos / fired | Verdict |
|---|---|---:|:---:|---:|:---:|:---:|
| (no gate, v0 ref) | all 6 | +0.099 | 4/6 | +0.099 | 4/6 | **FAIL** (α < +0.20) |
| `T_abs = 3000` | w0, w1, w2, w3 | **+0.104** | 3/6 | **+0.156** | 3/4 | **FAIL** |
| `T_pct50 = 3320` | w0, w1, w3 | +0.005 | 2/6 | +0.010 | 2/3 | **FAIL** (worse than no gate) |
| `T_pct30 = 2988` | w0, w1, w2, w3 | +0.104 | 3/6 | +0.156 | 3/4 | **FAIL** |

The best-case result (T_abs / T_pct30, alpha-on-fired = +0.156 across
3/4 positive fired windows) is within ±0.05 of the +0.20 marginal floor,
but **below it** on both denominators and at all three pre-registered
thresholds.

## Why the audit's hypothesis fails

The audit's reasoning chain was:

> w0 was the catastrophic window. w0 trained on 2000-2005 (dot-com
> cointegration period, structurally different). EG-passing-rate
> per window is itself a regime indicator (3522-4755 in working, 2249-2857
> in failing). Pre-register a cut at the median to skip the dot-com
> window.

The chain breaks at the third step. The audit inferred the EG-passing-rate
distribution from log output of v0, but **misread it**. The actual
counts:

- "Working" windows (positive val Sharpe): w1=3522, w2=3118, w3=4755,
  w4=2249 → range 2249-4755
- "Failing" windows (negative val Sharpe): w0=3918, w5=2857 → range 2857-3918

These ranges **overlap heavily**. The audit's clean separation
(3522-4755 vs 2249-2857) was wrong — that bands w1/w3 (working) against
w4/w5 (one working at +0.080, one failing at −0.109). The actual binding
window w0 doesn't fit the audit's working/failing bucket assignment
because **it sits in the "working" band by EG count but the "failing"
bucket by val Sharpe**.

Mechanism: train-period EG-pass count is dominated by the
*cross-sectional density* of cointegrating relationships during the
training slice. 2000-2005 was a high-cointegration regime (forced
co-movement during dot-com crash) — train EG_pass = 3918 reflects that.
But the cointegrations that held *during* the crash broke *after* it.
The train EG-pass count is structurally a backward-looking measure of
regime *that has already happened*, not a forward-looking measure of
whether the same regime will continue OOS.

This is a special case of the
[`cfr-macro-gate-final`](cfr-macro-gate-final.md) rule: train-window
regime-state measures (EG-pass, rolling-median VIX, average pairwise
correlation) are dominated by the regime that *produced* the training
data, not the regime that will *receive* the val deployment. They fight
regime transitions in both directions.

## What the v0 verdict actually was

Aside note that the audit's reading also overstated v0's `confirmed-null`
verdict. Reviewing `pairs-classical-v0.md` and the v0 source:

- v0 PASS cut: mean ≥ +0.50 AND ≥ 4/6 positive windows
- v0 MARGINAL cut: mean ∈ [+0.20, +0.50] AND ≥ 3/6 positive
- v0 FAIL cut: mean < +0.20 OR ≤ 2/6 positive

v0 actual: mean +0.099 AND 4/6 positive. Mean fails the +0.20 cut
(triggers FAIL via the alpha criterion), but **the 4/6 positive count
exceeds the marginal positive-windows criterion** (≥ 3/6). The
"reading note 4" in the leaderboard explicitly warns that single
outlier windows often drag means below thresholds while the win count
stays clean — that is precisely v0's pattern. The label is technically
correct by the pre-reg cuts (mean alpha is binding), but the
*description* "pair-spread mean reversion is not an alpha source"
overstates: 4/6 windows show alpha, the mean is dragged by one large
negative window.

This is a partial repeat of the
[`cfr-sensitivity-followup`](cfr-sensitivity-followup.md) finding: a
verdict label can be technically correct under its pre-reg while the
operational claim it supports ("not an alpha source") is overstated.

## Refined verdict

| Original claim | Refined claim |
|---|---|
| Audit: "EG-passing-rate is a natural regime indicator; gate will lift +0.099 → +0.5" | **Falsified.** Best-gate alpha is +0.104 (full panel) or +0.156 (fired-only), 3× to 5× short of audit's predicted lift. |
| Audit: "EG counts 3522-4755 in working windows, 2249-2857 in failing" | **Misread.** Actual ranges are 2249-4755 (working) and 2857-3918 (failing); they overlap heavily, with the catastrophic w0 sitting in the "working" band by EG count. |
| `pairs-classical-v0`: "pair-spread mean reversion is not an alpha source on this universe at this horizon" | **Overstated.** 4/6 windows post positive val Sharpe; mean is dragged by one large negative window (w0 trained on dot-com regime). v0 verdict label `confirmed-null` is technically correct by the pre-reg alpha cut; operational framing should say "alpha exists in 4/6 windows but mean is below the +0.20 marginal floor". |
| Audit + v0: implied "fix w0 → unlock deployment" | **No.** Even ex-w0 (5-window mean +0.365 the v0 finding notes) is below the +0.50 PASS floor; the residual signal is regime-bound and at the size needed to justify deployment friction, the surviving alpha is borderline. |

## What's left to try (low priority)

1. **Stop-loss on widening spreads** — `pairs-classical-v0` already
   notes that w0's catastrophic Sharpe is driven by 1-trade pairs
   ([FITB pair structure], 99% time in trade). A per-pair stop-loss
   that exits when the spread widens past 4σ would cap w0's downside
   without affecting the working windows. Cheapest test in the pairs
   arc. ~30 min wall.
2. **Within-sector restriction** — Engle-Granger on within-sector
   pairs only. Reduces the candidate set by ~5× and eliminates the
   high-correlation cross-sector pairs whose cointegration is
   spurious (driven by market beta). Already identified as v2 in
   `pairs-classical-v0`. Would also test whether the audit's
   "EG-pass is regime indicator" hypothesis recovers in a cleaner
   sub-universe.
3. **ML predictor for half-life** — train a small model to predict
   which screened pairs will mean-revert in val. This is the "ML
   head" path the v0 finding noted. Probably the highest-EV pairs
   experiment but also the slowest to wire up; only worth doing if
   the within-sector restriction shows the underlying signal is
   real-but-blocked-by-screening.

None of these promote pair-spread mean reversion above the +0.20
marginal threshold with high prior probability — the binding constraint
is that pairs alpha is regime-conditional at this universe, and no
*train-side* gate tested so far identifies the right regimes OOS.

## Hindsight oracle — per-pair selection has +1.79 Sharpe of unrealized headroom (2026-05-14 followup)

Cross-app oracle diagnostic, fourth application after factor + vol + gate.
The EG-gate falsification above closed `confirmed-null` on
*train-side regime gating*; this section answers the deeper question
the closure flagged ("is there ANY gate that lifts above +0.20, or is
the architecture at its ceiling?"). The answer: **yes, by a huge
margin**, and the binding constraint is **per-pair selection**, not
per-window gating or regime classification.

### Setup

Same v0 pipeline (screen → backtest → 1/N aggregate) reproduced on
six walk-forward windows. Five aggregation arms:

| Arm | Pair selection rule | Decision granularity |
|---|---|---|
| `all-pairs` | keep all screened pairs (v0 baseline) | per-window |
| `oracle-pos-pairs` | keep pairs with realized val Sharpe > 0 | per-pair (cheat) |
| `oracle-top-half` | top 50% of pairs by realized val Sharpe | per-pair (cheat) |
| `oracle-top-quartile` | top 25% of pairs | per-pair (cheat) |
| `window-gate-oracle` | use all pairs iff v0 baseline Sharpe > 0 else 0 | per-window (cheat) |

The oracle arms use realized val Sharpes to select pairs — strict
hindsight, not deployable. They bound the maximum Sharpe achievable
by ANY pair selector (heuristic, learned, half-life-based, etc.).

### Result — per-pair oracle clears PASS by 3.6× to 4.9×

| Arm | Mean val Sharpe | Pos windows | Verdict |
|---|---:|---:|---|
| `all-pairs` (v0 baseline) | +0.006 | 3/6 | FAIL |
| **`oracle-pos-pairs`** | **+1.799** | **6/6** | **PASS** |
| **`oracle-top-half`** | **+1.830** | **6/6** | **PASS** |
| **`oracle-top-quartile`** | **+2.425** | **6/6** | **PASS** |
| `window-gate-oracle` | +0.243 | 3/6 | MARGINAL |

Per-window detail (oracle-pos-pairs):

| Win | Period | v0 baseline | oracle-pos-pairs | n_kept / n_screened |
|---:|---|---:|---:|---|
| 0 | 2005-01 → 2008-02 (dot-com tail) | **−1.04** | **+2.17** | 116 / 200 |
| 1 | 2008-02 → 2011-03 (GFC) | +0.76 | +1.50 | 126 / 200 |
| 2 | 2011-03 → 2014-04 | −0.33 | +1.33 | 109 / 200 |
| 3 | 2014-04 → 2017-05 | +0.56 | +1.25 | 131 / 200 |
| 4 | 2017-06 → 2020-07 (COVID) | +0.14 | +2.27 | 113 / 200 |
| 5 | 2020-07 → 2023-08 | −0.05 | +2.26 | 103 / 200 |

The catastrophic w0 (v0 baseline **−1.04**) has 116/200 positive-Sharpe
pairs in the oracle universe and posts **+2.17** Sharpe under oracle
selection. **The pairs that mean-revert in w0 exist — the screening
admits ~84 bad ones that drag the aggregate negative.**

### What this falsifies in the earlier closure

The EG-gate falsification's framing above (and `pairs-classical-v0`'s
`confirmed-null` verdict) implicitly treated each window's aggregate
val Sharpe as an irreducible quantity tied to that window's regime.
The oracle decomposition shows that framing is wrong:

| Earlier claim | Oracle-corrected claim |
|---|---|
| "w0 catastrophic because dot-com cointegration doesn't hold OOS" | **Partly correct**: bad pairs admitted by the screen DON'T hold; but ~58% of screened pairs DO mean-revert in w0 (oracle finds +2.17 Sh). The regime SHIFT is real, but the signal isn't dead — it's just diluted. |
| "Train-side regime measures (EG-pass count, vol, etc.) can't identify the right OOS windows" | **Still correct**, but reframed: the issue isn't WINDOW-level regime identification — it's PAIR-level half-life prediction. Window-gate-oracle adds only +0.24; the real lever is per-pair filtering. |
| "Pair-spread mean reversion is not an alpha source on this universe at this horizon" | **Falsified.** Per-pair oracle delivers +1.80 mean Sharpe across all 6 windows. The architecture's ceiling is high; the v0 implementation captures ~0% of it because EG-screening admits far too many bad pairs. |

### Decomposition

| Lever | Headroom available |
|---|---:|
| **Per-pair selection** (oracle-pos-pairs vs v0) | **+1.79 Sharpe** |
| Pair selection refinement (top-half vs pos) | +0.03 (negligible) |
| Aggressive concentration (top-quartile vs top-half) | +0.60 |
| Per-window gating (window-gate vs v0) | +0.24 |

**The signal is at the pair level by a factor of ~7.** Train-side
gates (EG-pass-rate, regime classifiers) operate at the wrong
granularity entirely.

### Re-prioritized v0.x candidates (per the oracle)

The original "What's left to try" section below ranked:

| Original | Re-prioritized after oracle |
|---|---|
| #1 Stop-loss on widening spreads | #3 (low EV — caps downside but doesn't surface positive-Sharpe pairs) |
| #2 Within-sector restriction | #2 (medium EV — could reduce spurious cross-sector pairs, raising oracle baseline) |
| **#3 ML predictor for half-life** | **#1 (now highest-EV)** — directly addresses the +1.79 oracle headroom |

Pre-reg cuts for a predictor-quality v1 (any per-pair selector):
- PASS if mean val Sharpe ≥ +0.50 AND ≥ 4/6 positive windows
  (captures ≥ 28% of the oracle's +1.80 ceiling).
- STRONG-PASS if mean ≥ +1.0 AND ≥ 5/6 positive (≥ 55% capture).

Driver: `apps/pairs/scripts/run_oracle_walkforward.py` (local CPU,
~10 min wall; reuses v0's screen + backtest, adds 4 oracle arms).
Artifacts: `Output/pairs-oracle-walkforward-summary.json`.

## Master walk-forward log

This row is appended to [`leaderboard.md`](../leaderboard.md) as an
audit-driven `diagnostic`-style follow-up to the
`pairs-classical-v0` row. Verdict label:
[`confirmed-null`](../leaderboard.md#verdict-labels) on the EG-gate
hypothesis specifically (audit prediction falsified by 3-5×); the v0
verdict remains technically `confirmed-null` though arguably better
described operationally as "alpha in 4/6 windows, mean below
deployment floor".

Artifacts:
- Driver: `apps/pairs/scripts/eg_gate_eval.py`
- Output: `Output/pairs-eg-gate-summary.json`
- Source audit: `.audit-research-directions.md`
- Prior v0 finding: [`pairs-classical-v0`](pairs-classical-v0.md)
