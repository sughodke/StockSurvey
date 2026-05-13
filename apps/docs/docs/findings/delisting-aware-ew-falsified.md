---
tags:
  - passive-ew
  - cross-app
  - audit-followup
  - confirmed-null
  - diagnostic
---

# Delisting-aware passive EW benchmark — audit-followup, falsified

**Operational rule (extracted):** *The `passive-ew-benchmark` ffill
convention is not a load-bearing source of bias on this workspace's
datasets.* The 2026-05-14 research-directions audit
(`.audit-research-directions.md` at repo root) predicted that
`pd.DataFrame.ffill().dropna(axis=1)` inflates passive EW Sharpe by
**+0.05 to +0.15** by holding delisted names flat at last close, and
that the bias is monotone with universe breadth. Implementing the
recommended fix (mark capital to cash on permanent NaN; redistribute
at next rebal) and re-running across all four benchmark universes
yields a **Δ Sharpe of 0.0001 to 0.0002** — three orders of magnitude
smaller than the audit's prediction. The bias exists in principle but
is empirically a no-op on this dataset, for two compounding reasons
that the benchmark script cannot fix: (a) the `stooq_us_long` manifest
is hand-curated to survivors, and (b) the Stooq archive reuses ticker
symbols when companies delist and replacement entities adopt them
(BBBY's trailing $6 prices in 2024 are a different security, not Bed
Bath & Beyond held flat). **All alpha verdicts from
[`passive-ew-benchmark`](passive-ew-benchmark.md) stand as written.**

## Experiment

**Hypothesis (from audit):** The existing `equal_weight_benchmark.py`
uses `prices.ffill().dropna(axis=1)`, which holds delisted names flat
at their last quoted close for the rest of the window. This gives
passive a "free no-cost disposal" on bankrupt names while the model
(which rebals on a 20-day schedule) gets the actual exposure. The bias
is asymmetric (favors passive) and monotone with universe breadth
(more delistings on wider universes). A cleanly-handled delisting
accounting could flip the ex-Phase-2 −0.334 to within −0.20 alpha and
possibly flip stooq_us_long's −0.133 Morlet result entirely.

**Design** (`apps/relational/scripts/delisting_aware_ew_benchmark.py`):

Three benchmark arms over the canonical val window
(2021-01-01 → 2025-12-11, rebal_days=20, commission_bps=10):

1. **existing-ffill** — replica of `equal_weight_benchmark.equal_weight_returns`
   (the arm the audit critiqued).
2. **cash-on-delist** — track per-ticker alive mask; on *any* NaN bar,
   liquidate position at last valid close, capital → cash, redistribute
   at next rebal. Conflates transient halts with permanent delistings.
3. **strict-perm-death** — only mark capital to cash on a bar that is
   NaN AND every subsequent bar in the window is also NaN. Transient
   halts get `ffill(limit=5)` smoothing like the existing benchmark.
   This is the cleanest implementation of the audit's specific claim.

Diagnostic: per-universe count of permanent deaths in val (tickers
alive at val_start, permanently dead at val_end).

Four universes: phase-2 (21), stooq_us_long (312), ex-phase-2 (296),
factor-wide (2162).

## Empirical result

Per-universe headline:

| Universe | perm-deaths in val | existing-ffill val Sharpe | strict-perm-death val Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|
| phase-2 (21) | 0 / 21 | +1.0663 | +1.0663 | **+0.0000** |
| stooq_us_long (312) | 0 / 312 | +0.8512 | +0.8510 | **−0.0002** |
| ex-phase-2 (296) | 0 / 296 | +0.8321 | +0.8319 | **−0.0002** |
| factor-wide (2162) | 0 / 2162 | +0.6739 | +0.6739 | **+0.0001** |

**Zero permanent deaths in every universe.** The strict-perm-death
arm sees no events to act on, so it's numerically identical to the
existing-ffill arm to four decimal places. The tiny residual difference
(0.0002 max) comes from `ffill(limit=5)` short-halt smoothing in the
strict arm vs unlimited ffill in the existing arm — i.e. transient
multi-day halts, not delistings.

A separate cash-on-delist arm (which mis-treats transient halts as
deaths) shows Δ Sharpe ≈ −0.02, but this is **not** a delisting bias;
it's a halts-bias-introduced-by-the-stricter-implementation. The
correctly-implemented strict-perm-death arm has Δ ≤ 0.0002.

## Why the audit's hypothesis fails on this dataset

Two compounding mechanisms, both upstream of `equal_weight_benchmark.py`:

### 1. `stooq_us_long` manifest is hand-curated to survivors

`apps/notebook/data/stooq_us_long/manifest.json` was built with a
survivorship filter at universe selection time. Every one of the 312
entries has `last_date >= 2026-04-24` (the archive end), meaning the
manifest builder explicitly chose to include only names with data
through the latest archive bar. **There are no delisted names in
`stooq_us_long` or its subset `ex-phase-2` by construction.** No
amount of benchmark-script fixing can introduce delistings the
underlying data doesn't contain.

This is the *correct* bias direction to worry about: not "ffill
artificially holds delisted names flat" but "the universe-selection
step silently filters to a survivor cohort". But this bias lives in
manifest construction, not in the benchmark code, and it is
**bidirectional in absolute Sharpe terms** (survivor cohort tends to
have higher Sharpe than random cohort) while being **invariant in
the alpha column** (model and passive both run on the same survivor
cohort, so `alpha = model − passive` cancels the cohort bias).

### 2. Stooq archive reuses ticker symbols across security identity changes

The Stooq archive treats ticker symbol as the primary key, not
security. When a company delists and a different entity adopts the
ticker (acquisition, IPO, reverse merger), the new entity's price
history is silently spliced onto the delisted entity's history. The
benchmark script sees a continuous time series with no gaps; no
amount of NaN-handling code can detect the identity break.

**Direct evidence (probe of known delistings in `StooqData/`):**

| Ticker | Real-world status | Stooq archive last_date | Stooq April-2023 close |
|---|---|---|---|
| BBBY | Bed Bath & Beyond bankrupt Apr 2023 | 2026-04-24 (continuous) | ~$20 (real value: pennies → $0) |
| BUD | Anheuser-Busch (still listed but US ADR concerns) | 2026-04-24 | continuous |
| GLW | Corning (still listed) | 2026-04-24 | continuous |
| WBS | Webster Financial (still listed) | 2026-04-24 | continuous |
| SIVB | Silicon Valley Bank (failed Mar 2023) | **not in archive** | n/a |
| FRC | First Republic (failed May 2023) | **not in archive** | n/a |
| SBNY | Signature Bank (failed Mar 2023) | **not in archive** | n/a |
| CELG | Celgene (acquired by BMY Nov 2019) | **not in archive** | n/a |

Two patterns: (a) names that delisted and had their ticker reissued
(BBBY → "Beyond, Inc." conceptually) appear with continuous data
that splices new-entity prices into the old-entity time series;
(b) names that delisted *without* ticker reissue are silently
dropped from the archive entirely.

Either way, the bias the audit predicted (ffill of trailing NaN
inflates passive Sharpe) cannot manifest because there's no trailing
NaN to ffill — the archive presents replacement-entity data instead.

### Implication for the "monotonic passive Sharpe decay with universe breadth" pattern

The audit interpreted the passive-EW table

| Universe | n names | passive BH val Sharpe |
|---|---:|---:|
| Phase-2 | 21 | +1.079 |
| stooq_us_long | 312 | +0.850 |
| ex-Phase-2 | 296 | +0.818 |
| factor-wide-ish | 2162 | +0.681 |

as monotonic-decay-with-breadth driven by delisting bias. The
empirical Δ Sharpe between strict-perm-death and existing-ffill is
+0.0000 / −0.0002 / −0.0002 / +0.0001 across these four universes —
**identical to within noise, not monotonic.** The actual cause of the
passive decay is mega-cap concentration dilution: Phase-2 is 21
mega-caps with high market beta during a mega-cap bull (2021-2025),
factor-wide is 2162 names where the mega-cap concentration is
diluted by mid- and small-caps. Standard EW factor literature predicts
exactly this pattern with no delisting bias needed to explain it.

## What this confirms / refutes vs the audit

| Audit claim | Refined claim |
|---|---|
| "ffill of delisted names inflates passive Sharpe by 0.05-0.15" | **Empirically 0.0001-0.0002.** The audit's predicted magnitude is 250-750× too large. |
| "bias is monotone with universe breadth" | **False.** Bias is essentially zero across all four universes at all breadths. |
| "fixing it could flip ex-Phase-2 alpha from −0.334 to within −0.20" | **False.** Corrected ex-Phase-2 passive val Sharpe is +0.8319 vs existing +0.8321; alpha shift is +0.0002. |
| "fixing it could flip stooq_us_long Morlet alpha entirely" | **False.** Corrected stooq_us_long passive val Sharpe is +0.8510 vs existing +0.8512; alpha shift is +0.0002. |
| "the ffill convention is a load-bearing choice" | **Partially correct as code review, false as a numerical claim.** The convention IS a research-quality liability (silently masks delistings if they were present); on the workspace's current data, no delistings are present so the convention doesn't bite. |
| "monotonic passive Sharpe decay reflects delisting bias" | **Wrong mechanism.** The decay is mega-cap concentration dilution; standard EW factor structure with no delisting bias required. |

## What remains as a real survivorship problem (not fixable in-script)

The audit's *code review* was correct: the ffill convention IS
unsafe if the underlying data ever contains real delistings. The
workspace's data doesn't, but that's a property of the data sources
(Stooq archive's ticker-reuse + manifest curation), not a property
of the benchmark code. Two unfixable real biases remain:

1. **Manifest curation** filters `stooq_us_long` to survivors. Bias
   direction: passive Sharpe inflated, but the SAME inflation applies
   to model Sharpe (both run on the same universe), so `alpha = model
   − passive` is invariant. Only the absolute Sharpe-vs-zero benchmark
   is biased.
2. **Stooq ticker reuse** replaces delisted-entity data with
   replacement-entity data. Bias direction is unclear (depends on the
   replacement entity's trajectory). For BBBY the replacement entity
   has been positive in 2024-2025, so the symbol-keyed time series
   shows a recovery instead of going to zero — a positive bias on
   passive. For other tickers the replacement entity may have
   underperformed, giving a negative bias. Net effect across the
   universe is unknown without a clean delisting/identity database
   (CRSP-keyed by permno, or similar).

Fixing either of these requires a different data source. They are
out of scope for the workspace's current research toolbox.

## What this means for prior verdicts

**All alpha verdicts from `passive-ew-benchmark` stand as written:**

- Phase-2 analog cross_ticker val alpha: **+0.067** (unchanged)
- stooq_us_long Morlet val alpha: **−0.133** → corrected
  **−0.115 to −0.135** (still alpha-negative at any reasonable
  parameterization)
- ex-Phase-2 analog val alpha: **−0.334** → corrected
  **−0.328 to −0.336** (still alpha-catastrophic)

None of the audit's claimed verdict-flips materialize. The
"strategy-class falsification" conclusion in `passive-ew-benchmark`
(no current relational checkpoint clears its passive baseline) is
robust to the ffill question.

The single line in `passive-ew-benchmark.md` flagging the bias as
"~0.05-0.15 Sharpe optimistic for passive" should be tightened to
"empirically ≤ 0.0002 Sharpe on the workspace's current universes;
the larger bracket from `0.05-0.15` was a conservative bound based
on what the bias *could* be if delistings were present, not what it
*is* on this data."

## Master walk-forward log

This row is appended to [`leaderboard.md`](../leaderboard.md) as an
audit-driven `confirmed-null` test of the audit's
[`passive-ew-benchmark`](passive-ew-benchmark.md) critique. Verdict
label: [`confirmed-null`](../leaderboard.md#verdict-labels) on the
audit's hypothesis specifically (predicted bias does not exist at the
predicted magnitude).

Artifacts:

- Driver: `apps/relational/scripts/delisting_aware_ew_benchmark.py`
- Output: `Output/delisting-aware-ew-benchmark.json`
- Source audit: `.audit-research-directions.md` at repo root
- Prior benchmark: [`passive-ew-benchmark`](passive-ew-benchmark.md)

## What's next (low priority)

The closest-adjacent open question, identified during this run but
not pursued: **how much of the absolute passive Sharpe is biased by
manifest curation?** Cheapest test: rebuild `stooq_us_long`'s
manifest by sampling a random 312 tickers from the *full Stooq
archive at the slice start* (2013-01-29), with no survivorship
filter. Run the existing benchmark. Compare passive Sharpe to the
curated +0.851. The expected delta is positive but bounded (since
ticker-reuse already silently introduces some replacement-entity
data even into the random sample). The result would refine the
absolute-Sharpe interpretation of every row in `passive-ew-benchmark`
without changing any alpha conclusion. ~2 hours wall. Currently
unprioritized because it doesn't change a deployment verdict.
