# Cross-arc comparability: the Deflated Sharpe Ratio

**Operational rule.** A raw annualized Sharpe is *not* a fair cross-arc
ranking key, and neither is the leaderboard's "mean alpha" /
"mean val Sharpe" — those are **means of per-window Sharpe ratios**,
which can carry the opposite sign from the Sharpe of the actual
deployable return stream. Rank arcs on the **deflated-Sharpe t-stat**
(`ss_portfolio.standardize_oos`), computed on each arc's OOS *net
return stream*, with the multiple-testing deflation set by the number of
configurations the arc tried. Only rows that **form a return stream**
can carry this number; meta-evaluations that compose scalar Sharpes and
non-portfolio diagnostics are `DSR N/A` by construction.

## Why the existing Sharpe column isn't apples-to-apples

The leaderboard's Sharpe column mixes conventions across arcs — daily vs
block Sharpe, gross vs net, absolute vs alpha, long-only vs long-short,
21 mega-caps vs 2073 equities vs a 13-ETF basket vs an options panel.
All are annualized (shared `sqrt(252)` / `sqrt(252/rebal_days)`
convention), but annualization is the *only* thing they share. Worse,
many rows report a **mean of per-window Sharpe ratios**, which is a
statistic of statistics — not the Sharpe of the strategy you would
actually deploy.

The Deflated Sharpe Ratio (Bailey & López de Prado 2014) normalizes all
of this to one unit-free number by operating on the return stream and
correcting for three things a naive Sharpe ignores:

- **higher moments** — fat tails / skew inflate a naive Sharpe (PSR
  term);
- **sample length** — a 5-window arc is noisier than a 6-window one
  (the `sqrt(N-1)` term);
- **selection bias** — the more configurations an arc tried, the higher
  the Sharpe it should be *expected* to produce by chance (the
  expected-maximum-Sharpe deflation, `n_trials`).

The reported `deflated_tstat` is the z behind `DSR = Phi(z)`; it is the
cross-arc ranking key.

## Harness

- `ss_portfolio.standardize_oos(returns, *, periods_per_year, n_trials,
  trial_sharpes=None, sharpe_std=None, benchmark=None) -> MetricBlock` —
  the single source of truth. Self-contained PSR/DSR math (normal
  CDF/PPF implemented in-module; `ss_portfolio` stays numpy-only). 13
  unit tests in `packages/portfolio/tests/test_deflated.py`.
- `apps/docs/scripts/compute_dsr.py` — reads each arc's
  `Output/<arc>-returns.npz`, runs the harness, writes the ranked
  `Output/dsr-leaderboard.json`.
- Each deployable arc's eval driver gained a `--dump-returns` flag that
  concatenates the per-window OOS **net** return stream and writes the
  npz.

**Standalone vs overlay framing.** For rows claiming an absolute Sharpe
(factor, relational, vol) DSR is computed on the strategy's own stream.
For *overlay* rows claiming alpha over a benchmark (gate, any
timing/exposure overlay) DSR is computed on the **excess** stream — the
claimed edge — so "is the claimed edge real" stays comparable across
both kinds.

## Results — all six stream arcs

`compute_dsr.py` ranks every deployable arc, re-run to dump its OOS net
return stream:

| rank | arc | mode | n_trials | stream ann. Sharpe | skew | kurt | E[max SR] | DSR | deflated t |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | dca-canonical (live) | standalone | 4 | +0.692 | −0.52 | 11.0 | 0.017 | **0.973** | **+1.93** |
| 2 | relational analog cross_ticker | standalone | 16 | +1.146 | −0.05 | 9.1 | 0.028 | 0.937 | +1.53 |
| 3 | vol v3 regime-gated | standalone | 12 | +1.153 | +1.21 | 5.2 | 0.117 | 0.907 | +1.32 |
| 4 | pairs v0 | standalone | 4 | +0.203 | +0.61 | 20.8 | 0.017 | 0.397 | −0.26 |
| 5 | factor indicator-baseline LO | standalone | 50 | +0.192 | −1.92 | 12.1 | 0.160 | 0.062 | −1.54 |
| 6 | lie shape-kNN 1mo-reversal L/S | standalone | 9 | −0.018 | −1.37 | 5.6 | 0.110 | 0.194 | −0.86 |
| 7 | gate v0 (overlay) | overlay | 6 | −0.100 | +0.40 | 51.2 | 0.021 | 0.034 | −1.83 |
| 8 | factor long-short | standalone | 50 | −0.163 | −1.47 | 10.1 | 0.160 | 0.001 | −3.25 |
| 9 | lie shape-kNN reversal L/S (wide, 297) | standalone | 9 | −0.703 | −0.03 | 2.7 | 0.110 | 0.001 | −3.00 |

**Calibration update (2026-05-22, post-audit recompute).** A code audit
(UF-2) flagged that the first publication of this ladder used the
`standardize_oos` default fallback `sharpe_std = 1/sqrt(n_obs)` — the
null s.e. of a *single* Sharpe estimator, not the cross-trial
dispersion the DSR formula requires. That fallback was *bidirectionally*
wrong: it **over**-deflated short-block arcs (vol n=30: 1/√30 ≈ 0.183 >>
empirical) and **under**-deflated long-daily arcs (DCA n=5232: 1/√5232
≈ 0.014 << empirical). The corrected calibration uses the
**empirically-measured cross-trial annualized Sharpe std of 0.25**
(n=39 factor walk-forward arms in this workspace, observed std 0.245);
`n_trials=1` arcs are immune by construction. Net result: **no arc
clears the conventional t=+2 bar**. DCA stays #1 at t=+1.93 (was +2.07).
Vol v3 moves #5 → #3 (+0.13 → +1.32, lifted). Relational stays #2 but
+0.74 → +1.53 (lifted). The qualitative claim updates from "DCA is the
only t-confident edge" to "**DCA is closest to the bar at +1.93; no arc
clears it; vol v3 and relational are the next two at +1.32–1.53**".

**The shape-kNN reversal (row 6) is the cleanest IC→portfolio
cautionary tale.** Its cross-sectional IC reproduces at t=+3.75 (1178
daily-overlapping test dates) — a publication-grade signal — yet as a
deployable dollar-neutral long/short it has annualized Sharpe −0.02 and
deflated t −1.55. The t-stat borrows power from overlapping daily
observations; the harvestable book has only 57 non-overlapping monthly
realizations, and at **21-name breadth** the fundamental law
(IR ≈ IC·√breadth·TC) leaves too thin a top-minus-bottom spread to clear
costs + borrow. A high IC t-stat is necessary, not sufficient — breadth
and the transfer coefficient gate whether it becomes alpha.

**Breadth expansion (factor-narrow, 297 names) falsified the breadth
hypothesis** (2026-05-22): widening from 21 to 297 names did not rescue
the signal — the cross-sectional IC *reversed* to t=−0.81 and the L/S
deflated t fell to −3.45 (worse than at 21 names). The +3.75 IC was
**Phase-2 mega-cap-specific** (same universe-dependence as
[relational](relational-universe-shift.md)): on a heterogeneous 297-name
pool the shape-kNN neighbors no longer carry the 1-month-reversal
premium. Net: the shape-kNN reversal is small, universe-specific, and
non-harvestable as a long/short — the spread-trade line is closed. (The
parallelized 8-worker kNN did 1.35M train × 27.9k queries in 88s.)

**The 5d-horizon indicator signal as a deployable book (2026-05-22)**
was the other spread-trade candidate — higher breadth (factor-narrow,
297) and a dollar-neutral constructor (transfer coefficient ≈ 1). Two
ways it fails the DSR, both instructive:

- **Market-neutral long/short: firmly negative.** Pooled-stream Sharpe
  −0.45, deflated t −3.64 (focused) / −4.52 (conservative). The IC
  predicts top-decile *out*performance, not bottom-decile
  *under*performance, so the short leg adds cost + left-tail risk
  (skew −2.4, kurt 26) without return, and 5d doubles turnover so
  commission dominates. Same verdict as the 20d factor long-short
  (−3.07): the long/short *construction* kills an otherwise-real signal.
- **Long-only: a fat-tail mirage.** Raw Sharpe +0.357 looks deployable,
  but skew +9.6 / kurt 218 flag it — a single 5d block returns +92.8%
  (one small-name moonshot); ex-that-block the Sharpe falls to +0.272.
  The PSR moment correction discounts it to deflated t +0.10 (focused —
  a coin flip) / −0.91 (conservative). Not skill on either framing.

The 5d IC is real (the long-only mean val IC reproduces at +0.0114
skip-1), but **no construction of it clears a confident skill bar** —
exactly the failure the DSR (vs a raw or mean-of-windows Sharpe) is
built to surface.

**Regime-scaled DCA (2026-05-22)** tested the inverse direction —
monetizing the one signal that *is* predictable (aggregate regime) on
the one strategy that *does* ship (passive DCA), by scaling the basket's
exposure with a vol-target overlay (Moreira-Muir) and the gate's
predicted-drawdown signal. It failed too: both overlays add alpha of
only +0.02 (inside the gate's own ±0.10 noise band), vol-target reverses
sign across spans (+0.019 full → −0.063 val), and all arms' deflated-t
cluster at +1.8–2.1, statistically indistinguishable from passive. So
the regime signal is real but **not monetizable after costs** — passive
DCA is the final answer, and the whole DSR workstream converges on the
same place: the only deflated-t-confident edge here is being passively
long a diversified basket.

**12-1 cross-sectional momentum (2026-05-22)** closed the last open
question. The cross-sectional + search literature (Sullivan-Timmermann-White;
Harvey-Liu) says technical-rule *search* is data-snooping but the durable
cross-sectional *survivors* are value and momentum — and the repo had only
tested technical indicators at short (reversal) horizons, never 12-1
momentum at its proper horizon. Run pre-registered (not an Optuna sweep, to
avoid manufacturing a winner): skip-1 12-month formation, dollar-neutral
L/S, factor-narrow, 302 monthly blocks, net of costs + borrow. Result: ann
Sharpe +0.081; PSR deflated-t +0.40 (DSR 0.66, below significance even with
no selection penalty); ladder framing −0.88; 3/6 chunks positive. The
academic ~0.5-1.0 gross momentum premium is eaten by costs and the 2008-09
momentum crash (skew −1.09, kurt 8.15). This is the load-bearing null: even
the field's most durable cross-sectional anomaly is marginal-at-best on this
liquid US universe at realistic friction, so the "cross-section is bound"
conclusion is not an artifact of searching the wrong space badly. The
ladder's verdict stands — the only deflated-t-confident edge is passive DCA.

**Low-volatility / BAB (2026-05-22)** completed the durable-survivor sweep:
the structural anti-edge — long low-vol / short high-vol, the honest form
of "short what naive money overpays for" (Baker-Bradley-Wurgler;
Frazzini-Pedersen). Pre-registered, dollar-neutral, factor-narrow, 302
monthly blocks, net of costs + borrow. Result: ann Sharpe +0.128; PSR
deflated-t +0.64 (DSR 0.74) — **marginally the best cross-sectional arm on
the board** (vs momentum +0.40) but still sub-significant; family-deflation
(n=6) goes negative; last two 6-year chunks negative (post-2018 low-vol
crowding decay). So **both** of the field's durable cross-sectional survivors
— momentum and low-vol — are marginal-at-best here net of costs. (Caveat: the
dollar-neutral book carries a negative net-beta tilt; true beta-neutral BAB is
untested, but a +0.13 Sharpe with recent decay is unlikely to clear
significance even beta-neutralized.) The cross-sectional search is closed:
no single-factor arm clears the bar, and the only deflated-t-confident edge
remains being passively long a diversified basket.

**No arc clears the conventional t=+2 bar after calibration.** DCA
is closest at deflated t +1.93 (was +2.07 under the wrong fallback) —
a modest +0.69 Sharpe over 5232 daily bars and only 4 trials, just shy
of significance. Relational analog (+1.53) and vol v3 regime-gated
(+1.32) are the next two; both were *lifted* by the recalibration (the
old `1/sqrt(n_obs)` was over-deflating their short / dispersed streams)
but still sit in the "interesting, not confirmed" tier. The headline
Sharpes mislead the other way for the rest, in three distinct patterns
the DSR corrects:

- **mean-of-Sharpes ≠ stream Sharpe.** gate v0's "alpha +0.067" is a
  mean of per-window Sharpe *differences*; the actual excess stream
  (gated − EW, 4680 days) is −0.10 Sharpe, deflated t −1.83. factor's
  "+0.440 mean val Sharpe" is a mean of windows; the pooled stream
  (one −0.985 window included) is +0.192.
- **small sample × many trials, properly calibrated.** vol v3's +1.15
  Sharpe → deflated t +1.32 under the empirical 0.25 calibration (vs
  +0.13 under the old over-deflating 1/√30 fallback). 30 rebals × 12
  trials × proper cross-trial dispersion = "interesting, not
  conclusive".
- **selection bias on a heavily-searched arc.** factor's
  `confirmed-OOS` indicator baseline has a *negative* deflated t once
  its ~50-config search is priced in — the DSR does not overturn the
  mean-IC result, it prices the deployability.

pairs (−0.26) and the gate (−1.83) sharpen their existing
`confirmed-null` / `partial-OOS` verdicts. The fat tails matter: the
gate's excess-kurtosis-51 stream (it concentrates into crises) makes a
naive Sharpe especially misleading — the PSR moment term handles it.

## Scope: which rows can carry a DSR

Re-running surfaced that the 96 leaderboard rows fall into three classes
by **whether the eval ever forms a return stream**:

1. **Stream-bearing strategy arcs** (gate ✅, pairs ✅, factor,
   relational, vol, DCA / cfr-phase4d) — get a true returns-based DSR.
2. **Meta-evaluations** that compose *scalar* per-window Sharpes (cfr
   macro-gate, the oracle arms, sizing/overlay diagnostics, regime
   Optuna best-params) — `DSR N/A`; even re-running can't produce a
   stream because the eval is arithmetic on Sharpes.
3. **Non-portfolio diagnostics** (replay R², macro-regime Pearson,
   compression error) — `DSR N/A` (Sharpe undefined; already tagged).

A deflated Sharpe is *defined* only on a return stream, so the rankable
ladder is the stream-bearing strategy arcs; the rest stay as
falsification history.

## Reproduction notes

- **relational** reproduced its leaderboard row exactly (val ann.
  Sharpe +1.146).
- **factor** per-window val Sharpes [−0.99, +0.85, +0.60, +0.23, +0.42,
  +0.40] average to ≈+0.42, matching the +0.440 row; pooled stream is
  +0.192 (the −0.99 window weighs more in the pooled stream than in the
  mean-of-windows).
- **DCA** PassiveEW(rebal_days=80) on the Phase-4d close panel gives
  ann. Sharpe +0.692 vs the +0.673 row (80d cadence / period).
- **vol v3** — the re-run headline (60d-gate fired-α Sharpe −0.66)
  differs from the original +2.01 row (data-snapshot / composition
  drift); the DSR uses the 126d deployment-recipe full-panel stream.
  The reproduction gap itself argues for the DSR's skepticism.

## Per-arc drivers (reproduce)

- gate: `apps/gate/scripts/run_walkforward.py --threshold-quantile 0.95 --dump-returns`
- pairs: `apps/pairs/scripts/run_walkforward.py --dump-returns`
- factor: `uvx modal run apps/factor/scripts/modal/train_indicator.py::walkforward`
- relational: `uvx modal run apps/relational/scripts/modal/relational_dwt_phase2.py`
- vol: `apps/vol/scripts/run_walkforward_v3_regime_gated.py --dump-returns`
- dca: `apps/dca/scripts/dca_returns_dump.py`
- rank: `apps/docs/scripts/compute_dsr.py`
- **Trial-count reconstruction** — ✅ pinned per arc in
  `compute_dsr.py` SPECS (conservative; round up when ambiguous, since
  under-counting trials weakens the deflation):

  | arc | rows | `n_trials` | basis |
  |---|---:|---:|---|
  | gate | 4 | 6 | v0 threshold sweep {q=.85,.90,.95} × {binary, sigmoid} |
  | pairs | 4 | 4 | pre-reg config + screening-param variants |
  | factor | 38 | 50 | horizons × representations × losses × heads × universes |
  | relational | 14 | 16 | 8-arm scorer × ±DWT × {cross_ticker, per_ticker} |
  | vol | 10 | 12 | v0→v3.1 × sizing × OI filters × regime gates |
  | cfr / dca | 12 | 12 / 4 | CFR phase ladder; DCA = Phase 4a–d basket |
- **Leaderboard backfill** — once all stream arcs are computed, add the
  `deflated t-stat` in one append-only, provenance-tagged pass (mirrors
  the 2026-05-18 Sharpe backfill contract: additive, no
  verdicts/numbers altered), make it the primary sort key, and tag the
  meta/diagnostic rows `DSR N/A — <reason>`.

## Master walk-forward log

The DSR augments — does not replace — existing rows; see the
[leaderboard](../leaderboard.md) and its
[verdict labels](../leaderboard.md#verdict-labels).
