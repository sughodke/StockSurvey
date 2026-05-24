# Meta-allocator regime forecasting (pre-reg)

**Status:** done (eval landed 2026-05-23). Results appended at the end of this page; pre-reg text above is locked.

## Question

Does any causally-deployable meta-allocator across our 6 strategy arcs
(dca, gate, pairs, relational, dca_winner_4etf, vol_v3) beat the
trivial triple-benchmark (persistence / 1-over-N / inverse-arc-vol)
on a locked walk-forward, after Ledoit-Wolf studentized Sharpe-diff
CIs and Deflated-Sharpe deflation across the candidate count?

The literature brief (`.research-regime-forecasting-literature.md`)
points at four candidate techniques as the highest-EV plays:
Markov-with-Laplace-smoothing transition matrix; K=2
Kritzman-Page-Turkington turbulence overlay; López de Prado-style
meta-labeling per arc; Rapach-Strauss-Zhou-style combination forecast.
We add a fifth (BOCPD-CUSUM persistence modulator) because the brief
flags change-point detection as the "near-real-time" detector even
though it cannot estimate the new regime's mechanics.

## Steel-manned mechanism — why this should work

**Persistence works because** rolling-Sharpe winners are auto-correlated:
arcs ride macro tail-winds and the auto-correlation horizon (L=252)
is roughly an arc's own regime-length scale. The persistence baseline
is *not* a strawman — it captures the strongest published positive
result in this space (trend-following / TS-momentum-style allocation,
Moskowitz-Ooi-Pedersen 2012).

**Markov C1 should beat persistence because** persistence picks "the
arc that just won." Markov picks the arc that *historically* wins
*after* the current winner. If transitions are non-trivial (i.e. the
matrix is not the identity), Markov captures the structural switching
that persistence misses. Laplace-α=1 smoothing is mandatory at n=45
regimes / 30 free parameters per the brief's sample-size analysis;
without it the matrix has zero rows.

**Turbulence overlay C2 should help because** Kritzman-Page-Turkington
(FAJ 2012) found the OOS gain in HMM regime-switching concentrates in
the K=2 turbulent state and consists almost entirely of "switch to
cash in the bear state." This is the most replicated positive HMM
result. Used as an overlay on B2/B3 (not a stand-alone allocator) it
matches the published recipe exactly.

**Meta-labeling C3 should beat persistence because** persistence is a
hard 1-of-6 classifier; meta-labeling is 6 independent binary
classifiers with continuous probability outputs. The brief explicitly
flags meta-labeling as "the most direct analog to our 6-arc
allocation problem" — six independent
"will-this-arc-earn-its-Sharpe-in-the-next-H-days" classifiers
trained on `ss_macro` + arc-own features, soft-voted by predicted
probability.

**Change-point C4 should help because** persistence has *long lag*
when the current winner stops winning — it keeps holding the loser
until the rolling-Sharpe window grinds the comparison around (~ L/2
trading days of being wrong). A CUSUM detector on the arc's own
rolling Sharpe stream can flag the regime change in ~10-30 days
instead of ~120, capping the worst losses.

**Combination C5 (Rapach-Strauss-Zhou ensemble) should beat any
individual modeling candidate because** the brief flags it as "the
most empirically robust positive result in the regime-conditioning
literature" — simple-mean ensemble of imperfect-but-uncorrelated
forecasts cancels idiosyncratic forecast error and survives where
each individual member fails OOS.

What `confirmed-OOS` looks like operationally: the meta-allocator
ships as a thin layer above `cfr_phase4d_multiasset` that rebalances
weekly (or per the cadence the candidate uses) across the 6 arcs.
The deployed Sharpe is +0.3 to +0.5 above persistence on the 2024-25
OOS window, with a Ledoit-Wolf CI excluding 0 and deflated-t > +3.0
after deflating for N_trials = 8 (5 modeling candidates × 3
hyperparameter L choices, capped at 8 effective).

## Locked falsification design

### Universe

All 6 arcs from `apps/docs/scripts/count_regimes_since_2005.py::build_master`:
`dca`, `gate`, `pairs`, `relational`, `dca_winner_4etf`, `vol_v3`.
vol_v3 only has returns 2024-04-12 → 2025-12-11 (419 trading days);
all candidates exclude it from the selection set on dates < 2024-04-12.

Universe tag: **meta-alloc-arcs-6** (5 arcs pre-2024-04-12, 6 arcs
after).

### Windowing

Single walk-forward with three contiguous OOS folds. Each candidate
fits on the **expanding** training window up to fold-start, predicts
the next fold's allocation per rebal date, then advances.

- **Train history start:** 2005-01-03 (DCA inception). All causal
  features must use only data with a timestamp ≤ rebal date.
- **OOS fold 1:** 2015-01-01 → 2018-12-31 (4y, all 5 arcs available
  except vol_v3).
- **OOS fold 2:** 2019-01-01 → 2022-12-31 (4y, all 5 arcs available
  except vol_v3).
- **OOS fold 3:** 2023-01-01 → 2025-12-11 (3y, vol_v3 enters
  2024-04-12).
- **Rebal cadence:** 20 trading days (monthly). Switching cost
  10 bps per arc rotation paid on the *change in arc weights* per
  rebal (sum(|Δw|)/2 × 10 bps).
- **Trailing Sharpe lookback L:** canonical L=252. Sensitivity sweep
  L ∈ {126, 252, 504} reported.

Windowing tag: **meta-alloc-3fold-2015-25** (3 contiguous OOS folds
spanning 2015-01 → 2025-12).

### Candidates (locked)

- **B1 persistence(L=252):** allocate 100% to the arc with the
  highest trailing-L annualized Sharpe at each rebal date. Switching
  cost paid on rotation.
- **B2 1/N equal-weight:** equal across available arcs (1/5 pre
  2024-04, 1/6 after).
- **B3 inverse-arc-vol:** `w_i ∝ 1 / std(arc_i daily returns trailing L)`,
  normalized to sum to 1 over available arcs.
- **C1 Markov + Laplace α=1:** at each rebal date, build the
  transition matrix on regime-winners observed up to that date
  (Laplace-smoothed). Allocate weight ∝ row of matrix for current
  regime-winner (i.e. predicted distribution over next winner).
- **C2 K=2 turbulence overlay on B2:** fit Gaussian-HMM on
  `[VIX_level, slope_10y_3m, credit_baa]` standardized rolling-z;
  use `P(turbulent_state)` as a sizing scalar — final weights =
  B2 × (1 − P_turbulent). This is *not* arc-selection; it is a
  risk-off gate.
- **C3 Meta-labeling per arc:** 6 independent logistic regressions.
  Labels: `y_i,t = 1` if arc i's forward-H-day Sharpe exceeds the
  arc's full-sample Sharpe (H=63, locked). Features: 6-stack macro
  (`fed_funds, slope_10y_3m, credit_baa, m2_yoy, real_yield_10y,
  vix`) + arc's own trailing-63d and trailing-252d Sharpe + arc's
  trailing-63d realized vol = 9 features. Allocate weight ∝
  `predicted_proba_i`, normalized.
- **C4 CUSUM change-point on persistence:** like B1 but with a CUSUM
  detector on each arc's trailing-21d Sharpe stream. If a CP fires
  for the *currently allocated* arc, force re-evaluation against the
  full L=252 trailing-Sharpe board. CUSUM threshold h=4.0 (locked).
- **C5 combination ensemble:** simple mean of C1, C2-applied-to-B2,
  C3 weight vectors. (C4 is excluded — it's a hard 1-of-6 selector
  whose vector is degenerate and incompatible with mean-of-weights.)

### Deflation N

Total candidates = 5 modeling (C1-C5) + 3 hyperparameter L choices
(126/252/504) reported on the persistence baseline only = 8 effective
trials. Use n_trials=8 for deflated-Sharpe.

### Falsification bars (per candidate × benchmark)

For each candidate C_k vs each benchmark B_j on the pooled OOS
return stream (concatenation of folds 1+2+3):

- `confirmed-OOS`: Ledoit-Wolf 95% bootstrap CI on ΔSharpe_ann(C_k −
  B_j) **excludes 0 on the positive side** AND DSR-deflated-t > +3.0
  AND mean Δann_SR ≥ +0.3.
- `partial-OOS`: CI includes 0 but mean ΔSR_ann ≥ +0.15 AND DSR-t > +1.5.
- `confirmed-null`: CI includes 0 AND mean ΔSR_ann < +0.05.
- Otherwise: `partial-OOS` (neutral default for mixed evidence).

A candidate's **overall verdict** is the **worst** of its three
benchmark verdicts. To be `confirmed-OOS` overall, it must beat
persistence AND 1/N AND inverse-vol.

### Causality discipline

- Macro features: forward-fill releases onto trading days; the value
  at rebal date T is the latest macro reading published ≤ T (the
  `ss_macro.load_macro_panel` default).
- All trailing-window statistics use data strictly < the rebal date
  (we use `t − 1` close as the latest data point for a rebal at `t`).
- Regime-transition matrix for C1 uses only regime-end dates ≤ rebal
  date.
- Logistic regression in C3 is re-fit at each rebal date on the
  expanding history (no data-leakage from future labels).

### Compute placement

All-local. Total wall time estimate < 30 min for the full grid.

## Implementation

- `apps/docs/scripts/meta_allocator_run.py` — single end-to-end
  driver. Reuses `count_regimes_since_2005.build_master` for arc
  returns, builds macro panel via `ss_macro.load_macro_panel`,
  computes all candidate weight streams, evaluates with
  `ss_portfolio.sharpe_difference_ci` + `standardize_oos`.

## Results — appended after eval

**Verdict.** `confirmed-OOS` for **B3 inverse-arc-vol** vs B2 1/N
(ΔSR_ann **+0.367 [+0.028, +0.682]**, DSR-t **+4.17**). Every
modeling candidate (C1–C5) is `reversed-OOS` against B3 inverse-vol;
none clears the persistence (B1 L=252) or 1/N (B2) bars. Full
per-candidate per-benchmark numbers in
[`findings/meta-allocator-regime-forecasting.md`](../findings/meta-allocator-regime-forecasting.md).

**Operational rule extracted.** Do not forecast which arc will win
next. Weight inversely by each arc's trailing-252d realized vol,
renormalized over arcs available at the rebal date. This is the
Bridgewater All-Weather framing, reproduced on our specific 6-arc
panel.

**Surprises.**
1. Persistence L=252 ties with 1/N (ΔSR_ann +0.047, CI [−0.567, +0.689]) —
   the literature's "trailing-Sharpe winner is the bar" turned out to
   be statistically indistinguishable from naive equal-weight.
2. C5 combination ensemble (Rapach-Strauss-Zhou) did *worse* than its
   components individually — averaging three sub-baseline forecasts
   does not rescue them.
3. C4 CUSUM fired only 8 times across 2740 days at h=4.0, confirming
   the brief's "regime detection is intrinsically lagged" lower
   bound — the detector reacts after the loss is already locked in.

**Highest-EV follow-up.** Re-run B3 vs B2 with vol_v3 excluded from
the panel. If the +0.367 ΔSR survives without vol_v3, ship inverse-vol;
if it does not, the result is a 2024-25-window artifact.

Artifacts: `Output/meta-allocator-results.json` +
`Output/meta-allocator-daily-streams.npz`. Driver:
`apps/docs/scripts/meta_allocator_run.py` (single end-to-end run,
wall time ~5 min on local Intel 8-core).
