# E2E portfolio v4 — learned regime gate via multi-lookback features + symmetric vol action space

**Status: `pending` — pre-registration locked before the eval runs.**
Direct extension of e2e-portfolio v3 (one-universe two-head allocator,
expected `confirmed-null` per current run). v3 demonstrates the
paradigm (direct-Sharpe loss + raw inputs) and the limit (no
mechanism to survive a vol spike without a hand-coded gate). **v4
locks the first end-to-end attempt with all six preconditions for
learned regime gating met simultaneously.**

---

## Why this pre-reg exists

The repo has six prior empirical results that map the failure surface
for learned regime gating:

1. [`macro-regime-diagnostic`](../findings/macro-regime-diagnostic.md)
   v1a — macro features as direct predictor inputs were
   `confirmed-null` worse than baseline. COVID window posted alpha
   **−0.782** vs +1.045 ungated; raw macro levels are non-stationary
   across train/val.
2. [`macro-regime-diagnostic`](../findings/macro-regime-diagnostic.md)
   v1b — binary VIX-above-rolling-median gate: `partial-OOS` at
   z-score level (+0.215) but raw lift ≈ 0; threshold-cliff loses
   magnitude information.
3. [`cfr-macro-gate-final`](../findings/cfr-macro-gate-final.md) —
   1-year-median VIX gate stays elevated 12 months after stress,
   gating off CFR Phase 4d during exactly its strongest alpha
   regimes (w0 +0.422, w4 +0.263 in early recoveries).
4. [`vol-surface-v3-regime-gated`](../findings/vol-surface-v3-regime-gated.md) —
   hand-coded 126d-VIX-rolling-median gate is the only successful
   regime gate to date (`partial-OOS`, fired-α Sharpe +2.01, 37% fire
   rate). Memory window is goldilocks for vol but wrong for other
   apps (CFR uses 252d). NOT learned, hand-engineered.
5. [`cfr-sensitivity-followup`](../findings/cfr-sensitivity-followup.md) —
   tested 60d/90d/126d alternatives for CFR; all worse than the 252d
   baseline. No single hand-coded VIX lookback dominates across apps.
6. [`meta-allocator-regime-forecasting`](../findings/meta-allocator-regime-forecasting.md) —
   5 learned forecasters (Markov, turbulence, meta-labeling, CUSUM,
   combo) all `reversed-OOS` vs deterministic B3 inverse-vol.
   Forecasting-at-prediction-layer is the wrong objective.

These findings converge on **six preconditions** that no prior
experiment satisfied simultaneously. v4 locks all six.

---

## Mechanism — the steel-man

Why this could clear the bar:

1. **Direct-Sharpe loss + scale-invariant features removes the
   distribution-shift kill.** The macro-regime-diagnostic v1a
   collapse was caused by raw levels (Fed funds 5% ≠ 0%) on a
   forecasting loss. Percentile features (VIX_pct_60d / 126d /
   252d, IV_pct_60d / 126d / 252d) are scale-invariant — a 95th
   percentile reading means the same thing across regimes. The
   direct-Sharpe loss never asks the model to predict a number;
   only to size weights.
2. **Continuous output head eliminates the threshold cliff.**
   v1b's binary VIX-above-median gate killed vol w4 (+0.134
   alpha, just below median by chance). v3 / v4's continuous
   `vol_position ∈ [0, 2.5]` outputs a graduated exposure scaler.
3. **Multi-lookback inputs let the model learn its own memory.**
   The repo's empirical truth is that **126d works for vol, 252d
   works for CFR equities, no single lookback dominates**. v4
   gives the model VIX_pct at three lookbacks (60d / 126d / 252d)
   PLUS IV_pct at three lookbacks (60d / 126d / 252d) per name,
   so it can learn a regime-dependent + asset-class-dependent
   memory window from data — exactly the diagnostic the
   cfr-sensitivity-followup said was the binding constraint.
4. **Symmetric vol action space lets the model HEDGE, not just
   fade.** v3's vol head is short-vol only; its only escape from
   a vol spike is `vol_position → 0`, which loses money on the
   way down. v4 adds a `long_vol_position ∈ [0, 2.5]` head over a
   synthetic long-VIX return stream (derived from VIX futures or
   VIXY daily returns). This is the direct analog of
   Zhang-Zohren-Roberts's COVID survival mechanism — their model
   put weight on VIXY when stress hit. With both heads in the
   action space, the model can rotate from short-vol carry (calm
   regimes) to long-vol hedge (crash regimes) end-to-end.
5. **Daily-cadence IV closes the feature-resolution gap.** v3's
   weekly DoltHub IV forward-filled to daily under-resolves
   2-week regime shocks (COVID's most acute window was 4 trading
   days). v4 uses Theta Data Pro ($160/mo, 12-year history) or
   ORATS Delayed Data ($99/mo, 2007 history) for daily-cadence
   per-name IV, restoring the temporal resolution needed for the
   gate to respond on the right timescale.
6. **Aux IV-prediction head as a regularizer for the regime
   gate.** Train an auxiliary head that predicts next-week IV
   percentile from current macro state. This forces the encoder
   to internalize macro→IV regime structure as a representation,
   even when the direct-Sharpe loss is noisy. Loss-weight 0.1 per
   the [factor-multitask-aux-weight-sweep](../findings/factor-multitask-aux-weight-sweep.md)
   default.

If percentile-only inputs + continuous outputs + multi-lookback +
symmetric vol head + daily IV + IV-aux head together produce a
model that beats the deterministic 2-leg recipe **including on
fold-2 (COVID + 2022 Fed cycle)**, end-to-end learned gating is
solvable; we just had the wrong six preconditions. If they don't,
the conclusion is data-bound, not architecture-bound, and the next
lever is novel data per
[[research_strategy_arbitraged_space]].

---

## Architecture (locked — extends v3)

```
INPUTS — every feature is scale-invariant (percentile, log-ret,
or standardized within a 5y window):

PER-NAME (T=60 days, F_per_name = 17):
  v3's 6 price/return features (preserved):
    log_ret_1d, rv_20d, rv_60d, RSI14, normalized_price, mom_5d
  v3's 5 IV features (preserved):
    iv_current, hv_current, iv_vs_hv_gap, iv_pct_252d, iv_change_60d
  NEW: 6 multi-lookback IV percentile features:
    iv_pct_60d, iv_pct_126d              (additional IV regime windows)
    ivrp_pct_60d, ivrp_pct_126d, ivrp_pct_252d  (IV-vs-HV percentile at three lookbacks)
    iv_avail_flag                          (preserved)

MACRO SIDE CHANNEL (T=60, F_macro = 12):
  v3's 4 (VIX, T10Y3M, BAA10Y, VIX_pct_252d) — preserved
  NEW: 8 multi-lookback macro percentile features:
    VIX_pct_60d, VIX_pct_126d              (multi-window VIX regime)
    T10Y3M_pct_60d, T10Y3M_pct_252d        (yield curve regime windows)
    BAA10Y_pct_60d, BAA10Y_pct_252d        (credit regime windows)
    VVIX_pct_60d, VVIX_pct_252d            (vol-of-vol; the regime-of-the-regime)

ENCODER (same shape as v3):
  Per-name 1D conv (32 ch, k=5) → per-name embedding (B, K, 32)
  Macro MLP encoder (12 → 32 → 32) → macro context (B, 32)
  Concat → per-name head input (B, K, 64)
  Shared MLP body (64 → 32) → (B, K, 32)

THREE OUTPUT HEADS over the same universe (NEW vs v3: two vol heads):

  EQUITY HEAD (same as v3):
    Per-name equity logit → softmax over (K + cash) → equity_weights

  SHORT-VOL HEAD (same as v3):
    Per-name short_vol logit → top-K_active mask → per-name short_vol_weights
    Plus scalar short_vol_scale = 5.0 * sigmoid(z_short)

  LONG-VOL HEAD (NEW):
    Scalar long_vol_position = 5.0 * sigmoid(z_long)
    Multiplies a synthetic long-VIX daily return stream:
      long_vix_daily_return = vixy_close[t] / vixy_close[t-1] - 1.0
    Computed from VIXY daily prices (free via yfinance / Stooq).
    The long-vol leg is single-instrument, not per-name; it captures
    the systematic vol-spike hedge ZZR's VIXY allocation provided.

AUX IV-PREDICTION HEAD (NEW, regularizer):
  From the shared body, predict next-week IV percentile per name.
  MSE loss × 0.1 weight, added to direct-Sharpe loss.
  Provides supervised signal that anchors the encoder against IV
  regime structure even when Sharpe loss is noisy.

COMBINED RETURN:
  equity_part   = equity_weights[:K] @ asset_ret_next_1d
                + equity_weights[K] * 0  (cash)
  short_vol_part = short_vol_scale * (short_vol_weights @
                                       per_name_short_vol_daily)
  long_vol_part  = long_vol_position * long_vix_daily_return
  r_total = equity_part + short_vol_part + long_vol_part

LOSS:
  loss = -Sharpe(r_total) + 0.1 * aux_iv_prediction_mse
```

---

## Data sources (locked)

| feed | what | cadence | history | cost |
|---|---|---|---|---|
| Stooq archive (existing) | 13-ETF Phase 4d + VIXY daily closes | daily | 1996+ | free |
| FRED via `ss_macro` (existing) | VIX, T10Y3M, BAA10Y | daily | 1990+ | free |
| FRED `VVIXCLS` | VVIX (vol of vol) | daily | 2006+ | free |
| **NEW**: ORATS Delayed Data API | per-name ATM IV 30d/60d/90d, ~5000 symbols | daily | 2007+ | **$99/mo** |
| OR: Theta Data Pro | full chain → ATM IV computable | daily | 2014+ | $160/mo |
| Stooq close panel + daily realized HV | self-computed HV | daily | full | free |

**Recommended**: ORATS Delayed $99/mo. Pre-computed 30d ATM IV is
the exact feature shape v4 wants without us inverting Black-Scholes;
2007 history covers the GFC walk-forward fold and the rich pre-2019
period that DoltHub doesn't have (closes the fold-1+2 IV-blind
problem v3 had).

If $99/mo isn't acceptable in this experiment phase, fall back to
DoltHub weekly forward-filled (status-quo v3 substrate) — but note
explicitly that precondition 5 is then unmet and v4 is a 5-of-6
test, not 6-of-6.

---

## Walk-forward + verdict bar (locked)

Same 3-fold walk-forward as v3:
- fold-1 2015-2018 (n≈1006 daily)
- fold-2 2019-2022 (n≈1008 daily) — **COVID + 2022 Fed cycle — the
  load-bearing test of whether learned gating handles regime shocks**
- fold-3 2023-2025-12 (n≈718 daily, unseen 2024+ window)

n_steps=5000 per fold, AdamW lr=1e-3 wd=1e-4 batch=128, Modal T4
CUDA tinygrad. Same Modal Volumes (`ss-e2e-iv-data`,
`ss-e2e-artifacts`).

### Baselines (all carry from v3)

- EW (1/13 on Phase 4d ETFs).
- DCA (`PassiveEW(rebal_days=80, commission_bps=10)`).
- Deterministic 2-leg (`r_dca + 2.0 × r_vol_v3_daily`) — the
  load-bearing reference.
- Learned 2-leg (`0.0506 × r_dca + 2.2388 × r_vol_v3_daily`).
- v3 itself — measure v4 lift vs v3 to attribute the architectural
  changes.

### Pre-locked verdict bar

| condition | verdict |
|---|---|
| (1) pooled ΔSR_ann ≥ +0.10 vs deterministic 2-leg AND (2) Ledoit-Wolf 95% CI excludes 0 AND (3) **fold-2 daily Sharpe ≥ −0.10** (the load-bearing COVID-survival check) AND (4) `long_vol_position` mean during 2020-Q1 ≥ 0.5 (mechanism check) | **confirmed-OOS** — end-to-end learned gating works; deploy v4 as the canonical learned model. |
| ΔSR ≥ +0.05 vs deterministic AND CI excludes 0 AND fold-2 daily Sharpe ≥ −0.30 | **partial-OOS** — learned gating partial; iterate on lookback/feature set in v5. |
| ΔSR vs DCA ≥ +0.10 AND CI excludes 0 but doesn't beat deterministic 2-leg | **partial-OOS-vs-DCA** — recovers DCA-tier alpha but doesn't reach the hand-coded recipe's ceiling; record and continue. |
| ΔSR < +0.05 vs deterministic OR fold-2 daily Sharpe < −0.30 OR `long_vol_position` mean during 2020-Q1 < 0.2 (mechanism failure) | **confirmed-null** — learned end-to-end gating doesn't work even with all 6 preconditions met; conclusion is data-bound, next lever is novel data. |

**Sample-size honesty.** Fold-2 has n≈1008 daily observations,
adequate for CI half-width tightness. If the bootstrap CI on
fold-2 ΔSR alone is wider than ±0.50, the COVID-survival
verdict gets one auto-downgrade tier.

**Mechanism check is binding.** Even if the headline Sharpe
clears the bar, if `long_vol_position` mean during 2020-Q1
(Feb-Apr 2020) is below 0.2, the verdict downgrades — that
would indicate the model got the Sharpe by accident, not
because it learned the regime gate.

---

## What v4 must NOT do

- DO NOT use vol_v3 alpha (`Output/vol-v3-dolthub-oos-c200-returns.npz`)
  as a feature. Only as baseline.
- DO NOT hand-code regime gating logic, VIX lookback choice, vol
  scale, or per-name selection. **Every architectural decision the
  hand-coded `vol_v3` recipe makes (126d gate, top-200 OI, vega=2)
  must be learnable from the input features.** That's the
  load-bearing test.
- DO NOT skip Modal — tinygrad >2k steps must be Modal T4 per
  CLAUDE.md.
- DO NOT promote v4 to live if any of the 4 verdict-bar conditions
  fail. The deterministic 2-leg is the production fallback.

---

## Pragmatic substitution license

- If `long_vol_position`'s `long_vix_daily_return` substrate has
  thin coverage for fold-1 (pre-2019), zero-fill with availability
  flag exactly as v3 handled missing-IV folds. Document.
- If Theta/ORATS not yet contracted, fall back to DoltHub weekly
  + a note that this is a 5-of-6 test. Verdict still binding.
- If the aux IV-prediction loss destabilizes training, reduce
  weight from 0.1 → 0.05 → 0 in that order. Document the value used.
- If K=200 + 3 heads exhausts T4 VRAM, shrink K to 150. Document.
- If short-vol-and-long-vol jointly produce zero-sum positions
  (model long XYZ vol AND short XYZ vol simultaneously), record
  as a finding — that's a degenerate strategy and tells us the
  model isn't using the new degree of freedom.

---

## Acceptance criteria

1. Driver script `apps/e2e_portfolio/scripts/modal/train_v4_walkforward.py`
   executes the 3-fold walk-forward on Modal T4 and writes:
   - `Output/e2e-portfolio-v4-fold{1,2,3}-daily.npz`
   - `Output/e2e-portfolio-v4-pooled-daily.npz`
   - `Output/e2e-portfolio-v4-fold{1,2,3}.npz` (checkpoints)
   - `Output/e2e-portfolio-v4-results.json` — per-fold + pooled
     Sharpe + LW ΔSR CI vs each baseline + **per-fold
     short_vol_scale mean / long_vol_position mean / mechanism
     check on 2020-Q1**
2. Verdict label lands in `apps/docs/docs/leaderboard.md` per the
   locked bar above.
3. Finding writes to
   `apps/docs/docs/findings/e2e-portfolio-v4.md` regardless of
   verdict (the null is informative — closes path 2).
4. Update `apps/docs/docs/findings/index.md` listing and
   `apps/docs/mkdocs.yml` Findings nav.
5. If `confirmed-OOS`: update
   [`learner_layer_over_complexity`](file:///Users/sidghodke/.claude/projects/-Users-sidghodke-Code-StockSurvey/memory/learner_layer_over_complexity.md)
   memory and the `apps/ensemble` README to swap canonical recipe
   from `(DCA + 2x vol_v3)` to v4 deployment recipe.
6. Persist artifacts to `ss-e2e-artifacts` Volume.

---

## Pointers

- Parent finding (v3): `apps/docs/docs/findings/e2e-portfolio-v3.md`
  (verdict expected `confirmed-null`, mechanism diagnosis: lacks
  long-vol action space).
- Sister findings supplying the failure-surface map for path 2:
  - [`macro-regime-diagnostic`](../findings/macro-regime-diagnostic.md)
  - [`cfr-macro-gate-final`](../findings/cfr-macro-gate-final.md)
  - [`vol-surface-v3-regime-gated`](../findings/vol-surface-v3-regime-gated.md)
  - [`cfr-sensitivity-followup`](../findings/cfr-sensitivity-followup.md)
  - [`meta-allocator-regime-forecasting`](../findings/meta-allocator-regime-forecasting.md)
- Literature reference: Zhang-Zohren-Roberts 2020 [arXiv 2005.13665](https://arxiv.org/abs/2005.13665)
  — survived 2020 via VIXY in the action menu. v4's long-vol head
  is the architectural analog.
- Memory: [[learner_layer_over_complexity]] — the "learner layer
  + action space dominate" principle that motivated v4's
  symmetric-vol expansion.
