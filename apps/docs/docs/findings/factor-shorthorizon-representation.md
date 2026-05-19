# Factor short-horizon × fixed `(C, L)` representation

**Operational rule.** On the cross-sectional return-prediction problem,
the binding free parameter is the **prediction horizon, not the feature
representation**. The long-standing `apps/factor` "+0.012 val-IC
ceiling" was a *horizon artifact* of the hardcoded `rebal_days=20`: the
**existing deterministic indicator grid** at `rebal_days=5` posts mean
val IC **+0.0212 with all 6 walk-forward windows positive** (≈5× the
20-day cell), and at `rebal_days=10` **+0.0118, 6/6**. Two non-CWT fixed
`(channel × compressed-length)` encoders (truncated causal rFFT;
MiniRocket-PPV) were tested head-to-head and **lose to the indicator
grid at every horizon** — representation is `confirmed-null` as a lever,
exactly as the [CWT-recursive-compression
arc](cwt-recursive-compression.md) and the [indicator
baseline](factor-indicator-baseline.md) found, now extended to *non-CWT*
fixed `(C,L)` feature classes. **Sweep the prediction horizon before
hunting representations.** Caveat: short-horizon cross-sectional
reversal is a well-studied/arbitraged anomaly and the 5d result's
cost-sensitivity is not yet stress-tested — see *Caveats* below before
treating this as deployable.

## Why this experiment

Three priors set it up (full chain in the
[pre-registration](../TODO/factor-shorthorizon-representation.md)):

- Representation is not the 20-day lever — pure CWT ties the indicator
  grid ([`factor-indicator-baseline`](factor-indicator-baseline.md),
  `confirmed-null`), and every recurrent CWT compression `k` ties the
  same +0.0120 baseline
  ([`cwt-recursive-compression`](cwt-recursive-compression.md),
  `confirmed-null`, arc-closing).
- The horizon axis was the highest-EV untested branch and had only ever
  been pushed *longer* (63-day quarterly → `reversed-OOS`,
  `2026-05-04`). The short side (5/10-day, where the literature places
  cross-sectional reversal) had never been run on `apps/factor`.
- A non-CWT fixed descriptor beat CWT once at H=21 (memory
  `lie_test4_shape_vs_cwt`: shape t=+3.58 vs cwt t=−0.98).

So: cross a horizon where the return signal may differ × encoders that
are *not* the CWT.

## Eval setup

- **Universe:** `factor-narrow` — 297/297 `stooq_us_long` tickers built
  (`min_history_bars=6500`, 2000-01-01→2026-04-01). Same operating
  condition as the +0.0120 baseline row.
- **Arms (3 fixed encoders, drop-in via the backbone contract):**
  `IndicatorGridConfig` (74-ch, *control, run at every horizon*);
  spectral truncated causal rFFT, `(C=4 signals, L=16 low-freq bins)`;
  MiniRocket-style, `(C=84 canonical kernels, L=4 dilations)`, PPV
  pooled, bias=0 (no quantile fit → walk-forward-safe). Each:
  per-bar `(C,L)` block → `identity_backbone(K=C,F=L)` + pool z-norm →
  `train_scorer_walkforward`. Linear head, n_steps=200, lr=1e-2,
  wd=1e-3.
- **Horizons:** `rebal_days ∈ {20, 10, 5}` with **year-comparable block
  scaling** (20d 63/39/39 → 10d 126/78/78 → 5d 252/156/156) so the
  train/val calendar spans and the 6-window count are held fixed across
  horizons; the 20d cell reproduces the baseline windowing.
- **Decision metric: mean val IC** (commission-free). Sharpe recorded
  per arm but compared **only within a horizon** — at 5d commission
  drag is ~4× heavier than 20d, so cross-horizon Sharpe is mechanically
  confounded (the [`2026-05-04` quarterly `reversed-OOS`
  row](../leaderboard.md) and this leaderboard's closing
  commission-geometry note).

## Results

Mean val IC | mean val Sharpe | positive-val-IC window fraction:

| encoder | rebal=20 | rebal=10 | rebal=5 |
|---|---|---|---|
| **indicator** (control) | +0.0043 / +0.253 / 4-of-6 | +0.0118 / +0.620 / **6-of-6** | **+0.0212 / +0.728 / 6-of-6** |
| spectral (rFFT) | −0.0035 / +0.020 / 4-of-6 | +0.0010 / +0.435 / 4-of-6 | −0.0008 / +0.280 / 2-of-6 |
| MiniRocket | +0.0071 / +0.565 / 5-of-6 | +0.0039 / +0.472 / 5-of-6 | +0.0012 / +0.336 / 4-of-6 |

ΔIC vs the within-sweep horizon-matched indicator control:

| | r20 | r10 | r5 |
|---|---|---|---|
| spectral | −0.0078 | −0.0108 | −0.0220 |
| MiniRocket | +0.0028 | −0.0079 | −0.0200 |

Per-window val IC, indicator control:

- r20: `[0.002, −0.010, 0.020, 0.001, −0.009, 0.022]` (4/6)
- r10: `[0.025, 0.004, 0.007, 0.000, 0.010, 0.025]` (6/6, w3 razor-thin)
- r5: `[0.043, 0.029, 0.017, 0.010, 0.013, 0.016]` (6/6, min +0.0095)

## Pre-registered decision rule → verdict

- **Outcome 1 (representation works)** — fails: no encoder clears
  ΔIC ≥ +0.005 at h=5 or h=10; both go *negative* there. MiniRocket@20
  is the only positive ΔIC cell (+0.0028) — below the +0.005 bar and at
  the wrong horizon.
- **Outcome 2 (horizon-only effect)** — **holds**: the indicator grid
  self-improves massively at short horizon (IC +0.0043 → +0.0118 →
  +0.0212, window-consistency 4/6 → 6/6 → 6/6) and *no encoder
  separates from it*. The lever is **cadence, not representation**.

Verdicts: representation hypothesis **`confirmed-null`** (both non-CWT
`(C,L)` encoders × 3 horizons); horizon finding **`confirmed-OOS`**
(indicator @ r5, robust 6/6; r10 weaker, one window thin). Per the
pre-registered outcome-2 action the **gated learned conv-AE arm is not
built** — chasing a heavier representation when the result says
representation is not the lever is the precise "stop testing variations
of the same lever" anti-pattern.

> *Pre-reg note (transparency):* the TODO's staging paragraph loosely
> said "build the learned arm on outcome (1) or (2)", which contradicts
> the numbered decision rule (outcome 2 → *pivot cadence*, only outcome
> 1 → build the learned arm). The numbered rule is the binding
> pre-registration and governs; the staging wording was imprecise.

## Mechanism & caveats

- **Why short horizon works:** at a 5-day holding period the
  cross-sectional return is dominated by short-term reversal, a
  structurally different (and stronger, in raw IC) signal than the
  20-day momentum-ish regime the grid was always evaluated at. The grid
  already contains the features that express it (short-window RSI/CCI,
  5–20d vol); nothing new had to be learned — only the *horizon* the IC
  was measured against had to change.
- **Why the encoders lose:** consistent with the arc-wide finding that
  on cross-sectional returns the *feature class / representation* is not
  the binding constraint. Spectral truncation discards the
  level/short-window detail the reversal signal lives in; MiniRocket's
  PPV summary is too coarse at 5d. Both underperform the hand-crafted
  grid exactly where the grid is strongest.
- **Anchor drift (recorded, not suppressed):** the 20d indicator
  control came in at +0.0043 vs the 2026-04-30 leaderboard anchor
  +0.0120. 297/297 tickers built, so not a universe bug; the most
  plausible cause is the documented `IndicatorGridConfig` CCI-grid
  default change (`(n=80,w=63)`→`(n=40,w=21)`, post-dating that row).
  Internal validity is unaffected: the pre-registered rule is relative
  to the *within-sweep* horizon-matched cell and all 9 cells share one
  commit/harness/universe.
- **Deployability caveats (→ adjacent tests):** short-horizon
  cross-sectional reversal is a well-studied, heavily-arbitraged
  anomaly; the 5d net Sharpe (+0.728) survives 10bps commission at 4×
  the 20d turnover (so the commission headwind makes it *conservative*,
  not inflated) but cost-sensitivity beyond 10bps is untested, and the
  per-window IC is front-loaded (monotone calendar decay). Treat the
  signal as real-and-OOS within this sweep, not yet as a deployable
  recipe.

## Next experiment (pre-registered outcome-2 action)

Cadence is the lever: pivot `apps/factor`'s default `rebal_days` to the
short horizon with the *existing* grid, and run the `confirmed-OOS`
adjacent test — (i) commission/turnover stress at 5d (is the IC
microstructure?), (ii) a second universe, (iii) characterise the
front-loaded per-window decay. Feed the horizon result into the
relational [`rebal-days-sweep`](../TODO/rebal-days-sweep.md) thread (its
analog-kNN analogue of the same question).

## Master walk-forward log

Nine rows, `2026-05-18`, app `factor`, in
[`leaderboard.md`](../leaderboard.md): indicator r5/r10
[`confirmed-OOS`](../leaderboard.md#verdict-labels), indicator r20
[`diagnostic`](../leaderboard.md#verdict-labels) (control + anchor
drift), all spectral + MiniRocket cells
[`confirmed-null`](../leaderboard.md#verdict-labels). Pre-registration:
[`TODO/factor-shorthorizon-representation`](../TODO/factor-shorthorizon-representation.md).
Related: [`cwt-recursive-compression`](cwt-recursive-compression.md),
[`factor-indicator-baseline`](factor-indicator-baseline.md). Driver
`apps/factor/scripts/modal/train_shorthorizon_repr.py` (`23b4b2c`);
encoders `factor/cl_encoders.py` (`202cb3d`). Artifacts
`Output/sh-{encoder}-r{20,10,5}-windows.npz`,
`Output/shorthorizon-summary.json`, `Output/shorthorizon-comparison.png`.
