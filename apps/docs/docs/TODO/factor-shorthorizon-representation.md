# Factor short-horizon × fixed `(C, L)` representation

> **RESOLVED 2026-05-18 — outcome 2 (horizon-only effect).**
> Representation hypothesis **`confirmed-null`** (spectral rFFT +
> MiniRocket-PPV both lose to the indicator control at all of
> `rebal_days ∈ {20,10,5}`); horizon finding **`confirmed-OOS`** — the
> existing indicator grid at `rebal_days=5` posts mean val IC **+0.0212,
> 6/6 windows** (≈5× the 20d cell): the long-standing factor "+0.012
> ceiling" was a `rebal_days=20` artifact. Per the numbered decision
> rule's outcome-2 action, the **gated learned conv-AE arm was NOT
> built** (the staging paragraph's looser "(1) or (2)" wording is
> superseded by the numbered rule, which governs). **Spawned
> follow-up:** pivot `apps/factor`'s default cadence to the short
> horizon with the existing grid + run the `confirmed-OOS` adjacent
> test (5d commission/turnover stress → is it microstructure?; second
> universe; characterise the front-loaded per-window decay); feed the
> horizon result into the relational
> [`rebal-days-sweep`](rebal-days-sweep.md) thread. Evidence: 9
> leaderboard rows `2026-05-18`; finding
> [`factor-shorthorizon-representation`](../findings/factor-shorthorizon-representation.md).
> Pre-registration retained below as written.

**Verdict → next-experiment chain (as pre-registered):**

- If a non-CWT fixed `(channel × compressed-length)` encoder beats the
  **horizon-matched** indicator baseline at `rebal_days ∈ {5, 10}` →
  `confirmed-OOS`/`partial-OOS`: confirm scope (build the gated learned
  conv-AE arm; stratify windows for a regime gate if one window carries
  it).
- If the **indicator baseline itself** improves at short horizon but no
  encoder separates from it → the lever is *cadence, not
  representation*: pivot factor to the winning short horizon with the
  existing grid; drop the encoder hunt.
- If every encoder ties the horizon-matched baseline at every horizon →
  `confirmed-null`, encoder- **and** horizon-invariant: the `+0.0120`
  ceiling is intrinsic to the cross-sectional return-prediction problem;
  stop hunting representations/horizons and pivot to an orthogonal
  prediction problem (the [`confirmed-null`
  playbook](../leaderboard.md#verdict-labels)).

## Why this, why now

Three priors box this in:

1. **Representation is not the lever at 20-day returns.** The pure CWT
   bundle ties the hand-crafted indicator grid at noise on
   cross-sectional return prediction
   ([`factor-indicator-baseline`](../findings/factor-indicator-baseline.md),
   `confirmed-null`, +0.005 vs +0.0120 IC), and *compressing* the CWT —
   recursively via a return-coupled GRU at every `k ∈ {2,4,8,13}` —
   never clears the same +0.0120 baseline
   ([`cwt-recursive-compression`](../findings/cwt-recursive-compression.md),
   `confirmed-null`, closes the CWT-as-predictor question arc-wide). The
   binding constraint at `rebal_days=20` is the *task*, not the encoder.
2. **The horizon axis is the highest-EV untested branch**, per the
   research-directions audit — and it has been tested in exactly one
   direction so far: **longer** (63-day quarterly), which went
   `reversed-OOS` (leaderboard `2026-05-04`: q-return −0.0019 vs
   20d-return +0.0120). The short side (5/10-day, where the literature
   places cross-sectional reversal) has never been run on `apps/factor`.
3. **A non-CWT fixed descriptor already beat CWT once** at the
   cross-sectional H=21 task — per-ticker *shape features* beat a 168D
   CWT head-to-head (memory `lie_test4_shape_vs_cwt`: shape t=+3.58 vs
   cwt t=−0.98). That is the only positive prior for a
   `(C, L)`-style encoder and it motivates testing fixed deterministic
   `(C, L)` descriptors *off* the CWT, at a horizon where the return
   signal is structurally different.

This experiment crosses prior (2) with prior (3): a horizon where the
signal may differ × encoders that are *not* the CWT.

## Hypothesis (falsifiable)

At `rebal_days ∈ {5, 10}` the cross-sectional forward-return signal is
qualitatively different from 20-day (short-term reversal regime). A
fixed `(channel × compressed-length)` representation that is **not**
causal-CWT-of-price will, at one of those horizons, beat the
`IndicatorGridConfig` baseline *evaluated at the same horizon* by
≥ **+0.005 mean val IC** with ≥ **4/6 val windows** of consistent sign.
Null: every encoder lands within **±0.005 mean val IC** of the
horizon-matched baseline at every horizon.

## Arms (4)

The `(C, L)` contract is the existing backbone contract:
`identity_backbone(K=L, F=C)` + pool z-norm → head sees a flattened
`L·C` row, exactly as `make_indicator_backbone` does with `K=1`. Each
encoder is a drop-in `build_*_features → identity_backbone → 
train_scorer_walkforward`, mirroring `factor/indicator_features.py`. No
walk-forward harness change — `rebal_days` is already a free parameter.

| # | Arm | Shape `(C, L)` | Role |
|---|---|---|---|
| 0 | **`IndicatorGridConfig` baseline** | `(74, 1)` (existing) | **Confound control** — run at *every* horizon. Isolates "short horizon moves the signal" from "the encoder moves it". Non-negotiable. |
| 1 | **Truncated causal DCT/FFT** | `(n_signals, K)` low-freq bins | Native `(C, L)`, genuinely compressed, numpy. Cleanest CWT contrast: time-frequency → pure-frequency-truncated. |
| 2 | **MiniRocket → `(C, L)`** | `(n_kernels, pooled)` | Quasi-random dilated conv + PPV pooling, reshaped/grouped to `(C, L)`. Strong general TS rep, not arbitraged-anomaly space. |
| 3 | **Learned conv-AE bottleneck** | `(C_lat, L_lat)` | replay-style conv-AE on a *non-CWT* input signal. **Gated** — built only on outcome (1) or (2) below. |

## Universe & windowing

- **Universe:** `factor-narrow` — 297 tickers, `stooq_us_long` with
  `min_history_bars=6500`, common axis ~2000 → 2026 (the canonical
  factor walk-forward universe; matches the baseline row's operating
  conditions exactly).
- **Windowing — year-comparable block scaling.** `rebal_days` is *both*
  rebalance cadence *and* forward-return horizon in this harness, and
  windows are counted in *blocks* (= `rebal_days` bars). To hold the
  train/val *calendar* spans (and window count ≈ 6) fixed across
  horizons, scale block counts inversely with `rebal_days` — the
  convention the [`6-window factor (q)`](../leaderboard.md#operating-conditions)
  row established for the quarterly case:

  | `rebal_days` | blocks/yr | train | val | step | calendar |
  |---|---|---|---|---|---|
  | 20 (anchor) | 12.6 | 63 | 39 | 39 | ~5y / ~3y, ~6 win |
  | 10 | 25.2 | 126 | 78 | 78 | ~5y / ~3y, ~6 win |
  | 5 | 50.4 | 252 | 156 | 156 | ~5y / ~3y, ~6 win |

  The 20-day anchor reproduces the existing baseline row, binding the
  whole sweep to the leaderboard.

## Metrics & the commission-confound guardrail

The `2026-05-04` quarterly `reversed-OOS` row carries the load-bearing
methodological warning: *"quarterly Sharpe rises mechanically from less
commission, not skill."* The inverse holds here — at `rebal_days=5`
commission drag is ~4× heavier per year than at 20d, so **Sharpe is
mechanically depressed at short horizons regardless of signal.** Rules:

- **Primary decision metric: mean val IC** (commission-free,
  scale-invariant). All pre-registered bars are in IC.
- **Sharpe is reported for every arm** (block-Sharpe, per the
  always-record-Sharpe rule) but compared **only within a horizon**
  (encoder arm vs indicator baseline at the *same* `rebal_days`).
  Cross-horizon Sharpe comparisons (5d vs 20d) are commission-confounded
  and are **not** a decision input.
- Secondary: positive-val-IC window fraction and per-window sign
  consistency (distinguishes regime-break from genuine null).

## Pre-registered decision rule

Let `ΔIC(enc, h) = mean_val_IC(enc @ h) − mean_val_IC(baseline @ h)`.

1. **Representation works** (`confirmed-OOS` / `partial-OOS`): some
   encoder has `ΔIC ≥ +0.005` **and** ≥ 4/6 windows of consistent sign
   **and** within-horizon val Sharpe ≥ horizon-matched baseline's, at
   `h=5` or `h=10`. → Build the gated learned arm (#3); if exactly one
   window carries it, stratify for a regime gate.
2. **Horizon-only effect**: `|mean_val_IC(baseline @ h)|` beats the
   baseline's own 20-day +0.0120 by ≥ +0.005 with consistent sign at
   `h=5` or `h=10`, but no encoder clears `ΔIC ≥ +0.005` there. → Lever
   is cadence: pivot factor to the winning short horizon with the
   existing grid; feed the result into the relational
   [`rebal-days-sweep`](rebal-days-sweep.md) thread.
3. **Confirmed-null**: every encoder within `±0.005` of the
   horizon-matched baseline at every horizon **and** the baseline fails
   the outcome-2 self-improvement bar. → The +0.0120 ceiling is encoder-
   and horizon-invariant for cross-sectional returns; do not build arm
   #3; pivot to an orthogonal prediction problem.

## Staging (cost discipline)

The three fixed encoders (#0–#2) are numpy + a tiny head — cheap. The
learned conv-AE (#3) is the only new-trainer / heavy piece. So:

- **Phase 1:** arms #0–#2 × `rebal_days ∈ {5, 10, 20}` (9 cells) on
  Modal. This alone resolves the pre-registered rule above.
- **Phase 2 (gated):** build arm #3 **only** on outcome (1) or (2).
  Outcome (3) means the axis is dead across 3 encoders × 3 horizons —
  building a heavier encoder to chase an encoder-invariant ceiling is
  the exact "stop testing variations of the same lever" anti-pattern.

Smoke `--max-tickers 30 --n-steps 50` locally first; commit the green
scaffold before any Modal spend.

**Cost:** Phase 1 ≈ feature-build-bound, ~1–1.5h T4 wall, < $1
(9 cells, parallelized `mp.Pool` feature build, linear+mlp heads).
Phase 2 is a separate replay-style run, scoped if reached.

## Not the same as `rebal-days-sweep.md`

[`rebal-days-sweep`](rebal-days-sweep.md) is a *relational /
analog-kNN / Phase-2 / DWT-L1* workstream — different app, universe, and
signal. It is the relational analogue of the same horizon question and
its outcome-2 result here feeds that thread, but the two are orthogonal
(not superseded).

## Pointers

- Baseline row + operating conditions: [`leaderboard`](../leaderboard.md#operating-conditions)
  (`2026-04-30` factor deterministic-indicator, `factor-narrow`,
  `6-window factor`, +0.0120 ; +0.440, `confirmed-OOS`).
- Prior horizon pivot (longer): leaderboard `2026-05-04`,
  `reversed-OOS`.
- [`factor-indicator-baseline`](../findings/factor-indicator-baseline.md),
  [`cwt-recursive-compression`](../findings/cwt-recursive-compression.md)
  — why representation is not the 20-day lever.
