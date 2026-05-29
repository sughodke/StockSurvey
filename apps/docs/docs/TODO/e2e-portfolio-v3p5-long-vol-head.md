# E2E portfolio v3.5 — add long-vol output head (the ZZR action-space fix)

**Status: `pending` — pre-registration locked before the eval runs.**
Minimal, isolated extension of e2e-portfolio v3 that adds a **single
architectural change**: a `long_vol_position ∈ [0, 5]` output head
that multiplies a synthetic long-VIX daily return derived from VIXY
price action. Everything else (universe, encoder, training schedule,
short-vol head, macro side channel, Volumes, walk-forward folds)
stays identical to v3. **Purpose: test whether action-space expansion
alone fixes v3's fold-2 (COVID) collapse, isolated from the larger
v4 design.**

---

## Why this pre-reg exists (separately from v4)

v3 (one-universe two-head allocator, expected `confirmed-null`) fails
catastrophically on fold-2 (held-out daily Sharpe −0.64 across 2019-
2022, dominated by COVID in March 2020). The vol head is **short-vol
only** — its only escape from a vol spike is `vol_position → 0`,
which still loses money on the way down.

**Zhang-Zohren-Roberts 2020** ([arXiv 2005.13665](https://arxiv.org/abs/2005.13665))
survived 2020 in their published deep-learning portfolio paper
**because their action space included VIXY** (a long-vol ETF). When
VTI crashed, VIXY exploded; the model put weight there. Survival came
from the action space, not from any sophisticated regime model.

Two architectural fixes for v3's COVID problem exist:

| Path | What it changes | Cost / complexity |
|---|---|---|
| **Path 1 (this TODO)**: add long-vol head | Single output head over VIXY daily returns | Small — ~50 LoC, no new data, no architecture rewrite |
| **Path 2 (v4 TODO)**: full 6-precondition learned regime gate | Multi-lookback features + aux IV head + daily IV feed + symmetric vol heads + more | Large — new $99/mo data feed, multiple architecture changes, conflates 5 effects |

This TODO formalizes Path 1 as its own falsifiable test, **isolated
from v4's other changes**, so the answer to "does adding long-vol
alone fix COVID?" is unambiguous. Good experimental hygiene: don't
bundle 6 changes when 1 might suffice.

---

## Mechanism — the steel-man

Why this could clear the bar with **only** the action-space change:

1. **ZZR's published mechanism transfers directly.** Their published
   result survived 2020 specifically because VIXY's price action let
   the model dynamically hedge equity drawdown via a long-vol leg.
   Our v3 architecture is a direct descendant (per-asset encoder +
   direct-Sharpe loss); adding the same action it had should give us
   the same survival mechanism.
2. **The data is already free.** VIXY daily closes are in the Stooq
   archive from 2011-01-19 onward (ETF inception). No vendor
   subscription required. Coverage includes all of fold-2 (the COVID
   window) and fold-3. fold-1 (2015-2018) has full coverage too.
3. **The architecture is already in place.** v3 has a `vol_scale`
   scalar head ∈ [0, 5] for short-vol. Replicating that exact head
   pattern for `long_vol_position` is a near-zero-risk code change.
4. **Direct-Sharpe loss + appropriate action space is the
   demonstrated working recipe.** Per
   [`learned-ensemble-beats-deterministic`](../findings/learned-ensemble-beats-deterministic.md),
   when a learner's action space includes the deterministic answer
   AND the objective is portfolio Sharpe directly, learners reliably
   extract additional alpha. Adding long-vol gives the action space
   the **VIXY** the deterministic ZZR recipe used to survive COVID.
5. **The COVID test is unambiguous.** fold-2's COVID Q1 2020 either
   has a learned long-vol response or it doesn't. The mechanism
   check on `long_vol_position` mean during 2020-Q1 is binary; no
   handwaving room.

If Path 1 alone clears the COVID survival bar, Path 2's heavier
changes (multi-lookback, IV-aux head, daily-cadence vendor feed) are
unnecessary for survival and can be pursued separately on their own
merits. If Path 1 does NOT clear the COVID survival bar, we have
isolated evidence that action-space alone is insufficient and Path 2's
multi-precondition design is the right next step.

---

## Architecture (locked — extends v3 minimally)

Everything in v3 is preserved verbatim. Only the additions below.

```
NEW INPUT (1 new daily series, no per-name features needed):
  vixy_daily_return = vixy_close[t] / vixy_close[t-1] - 1.0
  Source: Stooq archive `daily/us/nyse stocks/.../VIXY.US.txt`,
    coverage 2011-01-19 → present. Forward-fill 2 missing days max;
    zero-fill pre-2011 (fold-1 partial coverage; document).

NEW OUTPUT HEAD (single scalar, added to v3 model_v3.py):
  long_vol_position = 5.0 * sigmoid(z_long)
  z_long is a single logit emitted from the same pooled state v3
  uses for vol_scale (mean over K names of shared body + macro).

COMBINED RETURN (modified):
  equity_part   = equity_weights[:K] @ asset_ret_next_1d
                + equity_weights[K] * 0  (cash)
  short_vol_part = vol_scale * (vol_weights @ per_name_short_vol_daily)
  long_vol_part  = long_vol_position * vixy_daily_return  ← NEW
  r_total = equity_part + short_vol_part + long_vol_part

LOSS:
  loss = -Sharpe(r_total)  -- identical to v3
```

**That's the entire change.** No new features per asset, no aux
head, no multi-lookback inputs, no new vendor feed, no encoder
modification. Just one logit → one sigmoid → one scalar → one
multiplier on one daily return series.

---

## Data sources (locked — all free)

| feed | purpose | source | history |
|---|---|---|---|
| Stooq archive `VIXY` | long-vol substrate (NEW) | local archive `daily/us/nyse stocks/.../VIXY.US.txt` | 2011-01-19+ |
| DoltHub IV parquet (existing) | per-name IV features | `ss_iv.load_dolthub_iv_parquet` | 2019-02+ |
| Stooq close panel (existing) | per-name price features | local archive | full |
| FRED via `ss_macro` (existing) | macro side channel | `ss_macro.loaders` | full |

**Total incremental data cost: $0.** No vendor subscription. This is
the minimum-cost falsification of v4's most expensive precondition
(action-space symmetry).

VIXY substrate caveats:
- VIXY itself has a steep negative roll cost in calm regimes (~−30%
  annualized in long bull markets). The model should learn to keep
  `long_vol_position ≈ 0` in calm regimes via the direct-Sharpe loss.
- VIXY is a leveraged long-VIX-futures product, not pure VIX. Tracking
  error to spot VIX is real but is what a real-money implementation
  would actually trade.
- fold-1 (2015-2018) has full VIXY coverage from 2011; fold-2 (2019-
  2022, includes COVID) has full coverage; fold-3 (2023-2025) has full
  coverage. **All three folds cleanly testable**, unlike DoltHub IV
  which gives fold-1 zero coverage.

---

## Walk-forward + verdict bar (locked)

Same 3-fold walk-forward as v3:
- fold-1 2015-2018 (n≈1006 daily)
- fold-2 2019-2022 (n≈1008 daily) — **COVID + 2022 Fed cycle**;
  the load-bearing test
- fold-3 2023-2025-12 (n≈718 daily; unseen 2024+ window)

n_steps=5000 per fold, AdamW lr=1e-3 wd=1e-4 batch=128, Modal T4 CUDA
tinygrad. Same Modal Volumes (`ss-e2e-iv-data`, `ss-e2e-artifacts`).

### Baselines (carry from v3)

- EW (1/13 on Phase 4d ETFs).
- DCA.
- Deterministic 2-leg (`r_dca + 2.0 × r_vol_v3_daily`) — load-bearing
  reference; survived COVID via hand-coded 126d-VIX-rolling-median
  fired-regime gate that masked vol_v3 during the COVID regime.
- Learned 2-leg (`0.0506 × r_dca + 2.2388 × r_vol_v3_daily`).
- **v3 itself** — direct comparison of pure architectural addition.

### Pre-locked verdict bar

| condition | verdict | what it tells us |
|---|---|---|
| (1) pooled ΔSR_ann ≥ +0.10 vs **v3** AND (2) **fold-2 daily Sharpe ≥ −0.10** (COVID survival) AND (3) **`long_vol_position` mean during 2020-Q1 (Feb–Apr 2020) ≥ 0.5** (mechanism check) | **confirmed-OOS** | Path 1 alone fixes the COVID problem. Path 2's heavier changes are not required for survival. |
| pooled ΔSR ≥ +0.05 vs v3 AND fold-2 daily Sharpe ≥ −0.30 AND 2020-Q1 long-vol mean ≥ 0.3 | **partial-OOS** | Action-space change helps but isn't sufficient alone. Path 2's multi-precondition design becomes the next test. |
| Pooled ΔSR ≥ +0.10 vs deterministic 2-leg AND CI excludes 0 AND fold-2 ≥ −0.10 (full ship bar) | **confirmed-OOS-ship** | Path 1 closes the gap to the deterministic recipe end-to-end. Ship as v3.5. |
| pooled ΔSR < +0.05 vs v3 OR fold-2 < −0.30 OR 2020-Q1 long-vol mean < 0.2 | **confirmed-null** | Action-space alone is insufficient. Path 2's multi-precondition design is the right next step (not optional). |

**Mechanism check is binding.** Even if headline Sharpe clears the
bar, if `long_vol_position` mean during 2020-Q1 is below 0.2, the
verdict downgrades — that would indicate v3.5 got the Sharpe lift
elsewhere (lucky equity rotation, short-vol timing) rather than from
the long-vol hedge. The whole point of this experiment is to test
**whether the model uses the new degree of freedom**, not whether it
incidentally happens to do better.

---

## What v3.5 must NOT do

- DO NOT add multi-lookback percentile features. That's v4.
- DO NOT add an aux IV-prediction head. That's v4.
- DO NOT swap the DoltHub weekly IV substrate. That's v4.
- DO NOT modify the encoder, the short-vol head, the macro encoder,
  the universe selection (K=200), the loss function, or any
  hyperparameter. **One change, isolated.**
- DO NOT use vol_v3 alpha as a feature. Only as baseline.

If v3.5's verdict is `confirmed-null` (long-vol head alone doesn't
help), the result is clean evidence that v4's more comprehensive
design is needed and motivates the $99/mo ORATS spend. If v3.5's
verdict is `confirmed-OOS-ship`, v4 becomes optional.

---

## Pragmatic substitution license (per CLAUDE.md "document and keep going")

- If VIXY isn't in the Stooq archive under the expected path, fall
  back to **UVXY** (1.5x leveraged) or build a synthetic long-VIX
  return from VIX index daily change via `ss_macro.loaders.load_fred_series('VIXCLS')`.
  Document the substrate used.
- If `long_vol_position`'s gradient is dominated by the noisy VIXY
  return signal and destabilizes training, add a small entropy
  regularizer on `z_long` (target → small mean during stable
  regimes). Document.
- If 5000 steps × 3 folds runs out of Modal T4 time (unlikely given
  the change is small), drop to 3000 steps and document.

---

## Acceptance criteria

1. Driver script `apps/e2e_portfolio/scripts/modal/train_v3p5_walkforward.py`
   executes the 3-fold walk-forward on Modal T4 and writes:
   - `Output/e2e-portfolio-v3p5-fold{1,2,3}-daily.npz`
   - `Output/e2e-portfolio-v3p5-pooled-daily.npz`
   - `Output/e2e-portfolio-v3p5-fold{1,2,3}.npz` (checkpoints)
   - `Output/e2e-portfolio-v3p5-results.json` with per-fold + pooled
     Sharpe + LW ΔSR CI vs each baseline (including v3) + **per-fold
     long_vol_position mean / std / 2020-Q1 mean** for the mechanism
     check.
2. Verdict label lands in `apps/docs/docs/leaderboard.md` per the
   locked bar above.
3. Finding writes to
   `apps/docs/docs/findings/e2e-portfolio-v3p5.md` regardless of
   verdict (null is informative — gates the v4 spend).
4. Update `apps/docs/docs/findings/index.md` and
   `apps/docs/mkdocs.yml` Findings nav.
5. If `confirmed-OOS-ship`: update `apps/ensemble/README.md` and the
   canonical deployment recipe references to point at v3.5.
6. Persist artifacts to `ss-e2e-artifacts` Volume.

---

## Pointers

- Parent finding (v3): `apps/docs/docs/findings/e2e-portfolio-v3.md`
  (when it lands; expected `confirmed-null` with fold-2 daily Sharpe
  ≈ −0.64).
- Sister TODO (path 2): [`e2e-portfolio-v4-learned-regime-gate`](e2e-portfolio-v4-learned-regime-gate.md)
  — the 6-precondition design that subsumes v3.5's change. v3.5
  isolates the action-space change for an independent read.
- Mechanism reference: [`learned-ensemble-beats-deterministic`](../findings/learned-ensemble-beats-deterministic.md)
  — established that learners reliably extract additional alpha when
  the action space includes the deterministic answer AND objective is
  direct portfolio Sharpe.
- Literature: Zhang-Zohren-Roberts 2020 [arXiv 2005.13665](https://arxiv.org/abs/2005.13665) —
  COVID survival came from the action menu, not from regime modeling.
- Memory: [[learner_layer_over_complexity]] — action space dominates
  learner complexity.
